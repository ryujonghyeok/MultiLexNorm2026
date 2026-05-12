#!/usr/bin/env python3
"""Train a QLoRA adapter for MultiLexNorm.

This script is intended for Colab GPU. It trains a PEFT LoRA adapter on
MultiLexNorm raw -> norm examples and saves only the adapter folder.
"""

from __future__ import annotations

import argparse
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Gemma LoRA adapter for MultiLexNorm.")
    parser.add_argument("--model-id", default="google/gemma-3-1b-it")
    parser.add_argument("--dataset-id", default="weerayut/multilexnorm2026-dev-pub")
    parser.add_argument("--output-dir", default="adapters/gemma-multilexnorm-lora")
    parser.add_argument("--hf-token", default=None, help="Optional Hugging Face token for gated models.")
    parser.add_argument("--include-validation-in-train", action="store_true")

    parser.add_argument("--max-train-examples", type=int, default=1000)
    parser.add_argument("--max-eval-examples", type=int, default=200)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--max-new-tokens", type=int, default=96)

    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help='Comma-separated module names, or "all-linear" for every linear layer.',
    )

    parser.add_argument("--no-4bit", action="store_true", help="Disable QLoRA 4-bit loading.")
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--preview-examples", type=int, default=3)
    return parser.parse_args()


def as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def user_prompt(row: dict[str, Any]) -> str:
    raw = json.dumps(as_list(row["raw"]), ensure_ascii=False)
    return (
        "Normalize this tokenized sentence for the MultiLexNorm task.\n"
        "Return only a JSON list of normalized tokens.\n"
        "The output list must have exactly the same length as the raw token list.\n\n"
        f"Language: {row['lang']}\n"
        f"Raw tokens: {raw}"
    )


def answer_text(row: dict[str, Any]) -> str:
    return json.dumps(as_list(row["norm"]), ensure_ascii=False)


def render_prompt(tokenizer: Any, row: dict[str, Any]) -> str:
    message = [{"role": "user", "content": user_prompt(row)}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    return f"### User\n{user_prompt(row)}\n\n### Assistant\n"


def render_training_text(tokenizer: Any, row: dict[str, Any]) -> tuple[str, str]:
    prompt = render_prompt(tokenizer, row)
    target = answer_text(row)
    if getattr(tokenizer, "chat_template", None):
        full = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": user_prompt(row)},
                {"role": "assistant", "content": target},
            ],
            tokenize=False,
        )
    else:
        eos = tokenizer.eos_token or ""
        full = f"{prompt}{target}{eos}"
    return prompt, full


def tokenize_example(row: dict[str, Any], tokenizer: Any, max_length: int) -> dict[str, list[int]]:
    prompt, full = render_training_text(tokenizer, row)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_enc = tokenizer(full, truncation=True, max_length=max_length, add_special_tokens=False)

    labels = list(full_enc["input_ids"])
    prompt_len = min(len(prompt_ids), len(labels))
    labels[:prompt_len] = [-100] * prompt_len

    return {
        "input_ids": full_enc["input_ids"],
        "attention_mask": full_enc["attention_mask"],
        "labels": labels,
    }


@dataclass
class CausalLMCollator:
    pad_token_id: int

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        max_len = max(len(feature["input_ids"]) for feature in features)

        def pad(values: list[int], pad_value: int) -> list[int]:
            return values + [pad_value] * (max_len - len(values))

        return {
            "input_ids": torch.tensor(
                [pad(feature["input_ids"], self.pad_token_id) for feature in features],
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                [pad(feature["attention_mask"], 0) for feature in features],
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                [pad(feature["labels"], -100) for feature in features],
                dtype=torch.long,
            ),
        }


def subset(dataset: Any, max_examples: int | None, seed: int) -> Any:
    if max_examples is None or max_examples <= 0 or len(dataset) <= max_examples:
        return dataset
    return dataset.shuffle(seed=seed).select(range(max_examples))


def load_multilexnorm(dataset_id: str) -> Any:
    from datasets import load_dataset, load_from_disk

    path = Path(dataset_id)
    if path.exists():
        return load_from_disk(str(path))
    return load_dataset(dataset_id)


def make_training_args(args: argparse.Namespace, has_eval: bool) -> Any:
    import torch
    from transformers import TrainingArguments

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    kwargs: dict[str, Any] = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": args.grad_accum_steps,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": 2,
        "bf16": use_bf16,
        "fp16": use_fp16,
        "optim": "paged_adamw_8bit" if not args.no_4bit else "adamw_torch",
        "report_to": "none",
        "remove_unused_columns": False,
        "gradient_checkpointing": True,
        "seed": args.seed,
    }

    signature = inspect.signature(TrainingArguments.__init__).parameters
    eval_value = "steps" if has_eval else "no"
    if "eval_strategy" in signature:
        kwargs["eval_strategy"] = eval_value
    else:
        kwargs["evaluation_strategy"] = eval_value
    if has_eval:
        kwargs["eval_steps"] = args.eval_steps
    if "gradient_checkpointing_kwargs" in signature:
        kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}

    return TrainingArguments(**kwargs)


def parse_json_list(text: str) -> list[Any] | None:
    start = text.find("[")
    while start != -1:
        for end in range(len(text) - 1, start, -1):
            if text[end] != "]":
                continue
            try:
                value = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
            return value if isinstance(value, list) else None
        start = text.find("[", start + 1)
    return None


def preview_generation(model: Any, tokenizer: Any, dataset: Any, max_examples: int, max_new_tokens: int) -> None:
    if max_examples <= 0 or len(dataset) == 0:
        return

    import torch

    print("\nPreview generations:")
    model.eval()
    for idx in range(min(max_examples, len(dataset))):
        row = dataset[idx]
        prompt = render_prompt(tokenizer, row)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        parsed = parse_json_list(generated_text)

        print(f"\nExample {idx + 1}")
        print("lang:", row["lang"])
        print("raw: ", as_list(row["raw"]))
        print("gold:", as_list(row["norm"]))
        print("text:", generated_text)
        print("json:", parsed)


def main() -> None:
    args = parse_args()

    import torch
    from datasets import concatenate_datasets
    from huggingface_hub import login
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer

    if args.hf_token:
        login(token=args.hf_token)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_multilexnorm(args.dataset_id)
    train_raw = data["train"]
    eval_raw = data["validation"]
    if args.include_validation_in_train:
        train_raw = concatenate_datasets([data["train"], data["validation"]])
        eval_raw = None

    train_raw = subset(train_raw, args.max_train_examples, args.seed)
    if eval_raw is not None:
        eval_raw = subset(eval_raw, args.max_eval_examples, args.seed)

    print("Training rows:", len(train_raw))
    print("Evaluation rows:", 0 if eval_raw is None else len(eval_raw))

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    quantization_config = None
    if not args.no_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map="auto",
        torch_dtype=compute_dtype,
        quantization_config=quantization_config,
    )
    model.config.use_cache = False
    if not args.no_4bit:
        model = prepare_model_for_kbit_training(model)

    target_modules: str | list[str]
    if args.lora_target_modules == "all-linear":
        target_modules = "all-linear"
    else:
        target_modules = [item.strip() for item in args.lora_target_modules.split(",") if item.strip()]

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    tokenized_train = train_raw.map(
        lambda row: tokenize_example(row, tokenizer, args.max_length),
        remove_columns=train_raw.column_names,
        desc="Tokenizing train",
    )
    tokenized_eval = None
    if eval_raw is not None and len(eval_raw) > 0:
        tokenized_eval = eval_raw.map(
            lambda row: tokenize_example(row, tokenizer, args.max_length),
            remove_columns=eval_raw.column_names,
            desc="Tokenizing validation",
        )

    training_args = make_training_args(args, tokenized_eval is not None)
    collator = CausalLMCollator(pad_token_id=tokenizer.pad_token_id)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=collator,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    with (output_dir / "train_lora_args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    print(f"\nSaved LoRA adapter to: {args.output_dir}")
    print("Important adapter files:")
    print(f"- {output_dir / 'adapter_config.json'}")
    print(f"- {output_dir / 'adapter_model.safetensors'}")

    if eval_raw is not None:
        preview_generation(model, tokenizer, eval_raw, args.preview_examples, args.max_new_tokens)


if __name__ == "__main__":
    main()

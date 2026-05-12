#!/usr/bin/env python3
"""Generate a Codabench submission using a trained LoRA adapter."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import re
import zipfile
from pathlib import Path
from typing import Any

from train_lora import (
    as_list,
    kst_timestamp,
    load_multilexnorm,
    model_name_for_path,
    parse_json_list,
    render_prompt,
    timestamped_name,
)
from utils import counting, mfr


def check_torchao_version() -> None:
    try:
        version = importlib.metadata.version("torchao")
    except importlib.metadata.PackageNotFoundError:
        return

    try:
        from packaging.version import Version
    except ImportError:
        return

    if Version(version) < Version("0.16.0"):
        raise RuntimeError(
            f"Found torchao=={version}, but PEFT requires torchao>=0.16.0 in this environment. "
            "Run: pip install -U 'torchao>=0.16.0' or rerun pip install -U -r requirements.txt"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Codabench predictions with Gemma + LoRA adapter.")
    parser.add_argument("--adapter-dir", required=True, help="Path to the trained LoRA adapter folder.")
    parser.add_argument("--model-id", default=None, help="Base model ID. Defaults to adapter_config.json value.")
    parser.add_argument("--dataset-id", default="weerayut/multilexnorm2026-dev-pub")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--output-root",
        default="outputs",
        help="Parent directory for timestamped prediction output folders.",
    )
    parser.add_argument(
        "--run-prefix",
        default=None,
        help="Prefix for the timestamped output directory and ZIP file. Defaults to the model name.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Explicit output directory. Prefer --output-root and --run-prefix for KST timestamped runs.",
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN"),
        help="Optional Hugging Face token for gated models. Defaults to HF_TOKEN.",
    )
    parser.add_argument("--quantization", choices=["4bit", "none"], default="4bit")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--max-examples", type=int, default=0, help="0 means all examples.")
    parser.add_argument("--fallback", choices=["raw", "mfr"], default="mfr")
    parser.add_argument("--preview-examples", type=int, default=5)
    parser.add_argument(
        "--disable-stop-after-json",
        action="store_true",
        help="Keep generating until max_new_tokens instead of stopping after a valid JSON token list.",
    )
    return parser.parse_args()


def resolve_output_dir(
    output_dir: str | None,
    output_root: str,
    run_prefix: str | None,
    model_id: str,
) -> tuple[Path, str]:
    timestamp = kst_timestamp()
    if output_dir:
        path = Path(output_dir)
        if not re.search(r"\d{6}_\d{4}", path.name):
            raise ValueError(
                "Explicit --output-dir must include a YYMMDD_HHMM timestamp in the final folder name. "
                "Prefer --output-root plus --run-prefix so the script creates a KST timestamp automatically."
            )
        return path, timestamp
    prefix = run_prefix if run_prefix is not None else model_name_for_path(model_id)
    return Path(output_root) / timestamped_name(prefix, timestamp), timestamp


def model_id_from_adapter(adapter_dir: Path) -> str:
    config_path = adapter_dir / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing adapter config: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_id = config.get("base_model_name_or_path")
    if not model_id:
        raise ValueError(f"adapter_config.json does not contain base_model_name_or_path: {config_path}")
    return model_id


def make_mfr_counts(data: Any) -> dict[str, dict[str, dict[str, int]]]:
    from datasets import concatenate_datasets

    train = concatenate_datasets([data["train"], data["validation"]])
    train_df = train.to_pandas()
    counts_by_lang = {}
    for lang, lang_df in train_df.groupby("lang"):
        counts_by_lang[lang] = counting(lang_df.to_dict(orient="records"))
    return counts_by_lang


def fallback_prediction(raw: list[str], lang: str, fallback: str, counts_by_lang: dict[str, Any] | None) -> list[str]:
    if fallback == "mfr" and counts_by_lang is not None:
        return mfr(raw, counts_by_lang.get(lang, {}))
    return list(raw)


def normalize_prediction(
    generated_text: str,
    raw: list[str],
    lang: str,
    fallback: str,
    counts_by_lang: dict[str, Any] | None,
) -> tuple[list[str], str]:
    parsed = parse_json_list(generated_text)
    if isinstance(parsed, list) and len(parsed) == len(raw) and all(isinstance(token, str) for token in parsed):
        return parsed, "model"
    return fallback_prediction(raw, lang, fallback, counts_by_lang), "fallback"


def load_model_and_tokenizer(args: argparse.Namespace, model_id: str) -> tuple[Any, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=args.hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    quantization_config = None
    if args.quantization == "4bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        dtype=compute_dtype,
        quantization_config=quantization_config,
        token=args.hf_token,
    )
    check_torchao_version()
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.eval()
    return model, tokenizer


def generation_config_for_inference(model: Any, tokenizer: Any) -> dict[str, Any]:
    generation_config = model.generation_config
    generation_config.do_sample = False
    generation_config.top_p = None
    generation_config.top_k = None
    return {
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "generation_config": generation_config,
    }


def make_stop_after_json(tokenizer: Any, prompt_len: int, expected_len: int) -> Any:
    from transformers import StoppingCriteria

    class StopAfterValidJsonList(StoppingCriteria):
        def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
            generated_ids = input_ids[0][prompt_len:]
            text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            parsed = parse_json_list(text)
            return (
                isinstance(parsed, list)
                and len(parsed) == expected_len
                and all(isinstance(token, str) for token in parsed)
            )

    return StopAfterValidJsonList()


def generate_one(
    model: Any,
    tokenizer: Any,
    row: dict[str, Any],
    max_new_tokens: int,
    stop_after_json: bool,
) -> str:
    import torch
    from transformers import StoppingCriteriaList

    prompt = render_prompt(tokenizer, row)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[-1]
    stopping_criteria = None
    if stop_after_json:
        stopping_criteria = StoppingCriteriaList([make_stop_after_json(tokenizer, prompt_len, len(as_list(row["raw"])))])

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            stopping_criteria=stopping_criteria,
            **generation_config_for_inference(model, tokenizer),
        )
    generated_ids = output_ids[0][prompt_len:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def write_submission(records: list[dict[str, Any]], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.json"
    zip_path = output_dir.with_suffix(".zip")

    with predictions_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(predictions_path, arcname="predictions.json")

    return predictions_path, zip_path


def main() -> None:
    args = parse_args()
    adapter_dir = Path(args.adapter_dir)
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

    model_id = args.model_id or model_id_from_adapter(adapter_dir)
    output_dir, run_timestamp_kst = resolve_output_dir(args.output_dir, args.output_root, args.run_prefix, model_id)

    print("Adapter directory:", adapter_dir)
    print("Base model:", model_id)
    print("Dataset:", args.dataset_id)
    print("Split:", args.split)
    print("Run timestamp (KST):", run_timestamp_kst)
    print("Output directory:", output_dir)

    data = load_multilexnorm(args.dataset_id)
    target = data[args.split]
    if args.max_examples and args.max_examples > 0:
        target = target.select(range(min(args.max_examples, len(target))))
    print("Prediction rows:", len(target))

    counts_by_lang = make_mfr_counts(data) if args.fallback == "mfr" else None
    model, tokenizer = load_model_and_tokenizer(args, model_id)

    records = []
    source_counts = {"model": 0, "fallback": 0}
    for idx, row in enumerate(target):
        raw = as_list(row["raw"])
        norm = as_list(row["norm"]) if "norm" in row else [""] * len(raw)
        if len(norm) != len(raw):
            norm = [""] * len(raw)

        generated_text = generate_one(
            model,
            tokenizer,
            row,
            args.max_new_tokens,
            stop_after_json=not args.disable_stop_after_json,
        )
        pred, source = normalize_prediction(generated_text, raw, row["lang"], args.fallback, counts_by_lang)
        source_counts[source] += 1

        records.append({"raw": raw, "norm": norm, "lang": row["lang"], "pred": pred})

        if idx < args.preview_examples:
            print(f"\nExample {idx + 1}")
            print("lang:", row["lang"])
            print("raw: ", raw)
            print("model_text:", generated_text)
            print("submitted_pred:", pred)
            print("source:", source)

    for idx, record in enumerate(records):
        if set(record) != {"raw", "norm", "lang", "pred"}:
            raise ValueError(f"Bad keys at row {idx}: {record.keys()}")
        if len(record["raw"]) != len(record["norm"]) or len(record["raw"]) != len(record["pred"]):
            raise ValueError(f"Length mismatch at row {idx}")

    predictions_path, zip_path = write_submission(records, output_dir)
    print("\nPrediction source counts:", source_counts)
    print(f"Wrote {predictions_path}")
    print(f"Wrote {zip_path}")


if __name__ == "__main__":
    main()

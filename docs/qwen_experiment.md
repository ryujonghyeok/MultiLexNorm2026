# Qwen Experiment Branch

This branch is for testing `Qwen/Qwen3.5-0.8B` without changing the main Gemma path.

`Qwen/Qwen3.5-0.8B` is a newer image-text model, so this branch adds an experimental
`--model-family qwen3_5` loader that uses `AutoModelForImageTextToText` when needed.

## Colab Setup

```python
%cd /content/MultiLexNorm2026
!git fetch origin
!git checkout qwen-experiment
!git pull origin qwen-experiment
!pip uninstall -y torchao
!pip install -q -r requirements.txt
```

If the Qwen3.5 architecture is still not recognized, update Transformers in Colab:

```python
!pip install -q -U git+https://github.com/huggingface/transformers.git accelerate peft
```

Then restart the runtime and rerun setup.

## Smoke Training

Use this first to check that the architecture, LoRA injection, and JSON output all work.

```python
!python train_lora.py \
  --model-id Qwen/Qwen3.5-0.8B \
  --model-family qwen3_5 \
  --dataset-id /content/drive/MyDrive/MultiLexNorm2026/data/multilexnorm2026-dev-pub \
  --output-root /content/drive/MyDrive/MultiLexNorm2026/adapters \
  --quantization 4bit \
  --max-train-examples 1000 \
  --max-eval-examples 200 \
  --max-length 512 \
  --epochs 1 \
  --batch-size 8 \
  --grad-accum-steps 2 \
  --eval-batch-size 16 \
  --learning-rate 1e-4 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --eval-steps 100 \
  --save-steps 500 \
  --logging-steps 20 \
  --dataloader-num-workers 2 \
  --preview-examples 5
```

## Full Training Candidate

Run this only after the smoke run works.

```python
!python train_lora.py \
  --model-id Qwen/Qwen3.5-0.8B \
  --model-family qwen3_5 \
  --dataset-id /content/drive/MyDrive/MultiLexNorm2026/data/multilexnorm2026-dev-pub \
  --output-root /content/drive/MyDrive/MultiLexNorm2026/adapters \
  --quantization 4bit \
  --max-train-examples 0 \
  --max-eval-examples 1000 \
  --max-length 512 \
  --epochs 3 \
  --batch-size 16 \
  --grad-accum-steps 1 \
  --eval-batch-size 32 \
  --learning-rate 1e-4 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --eval-steps 500 \
  --save-steps 1000 \
  --logging-steps 20 \
  --dataloader-num-workers 2 \
  --preview-examples 5
```

If this OOMs, keep the effective batch size but lower per-step memory:

```text
--batch-size 8
--grad-accum-steps 2
--eval-batch-size 16
```

## Validation Score

After training, run validation prediction. This now prints overall and per-language scores.

```python
!python predict_lora.py \
  --adapter-dir /content/drive/MyDrive/MultiLexNorm2026/adapters/YOUR_QWEN_ADAPTER_FOLDER \
  --model-family qwen3_5 \
  --dataset-id /content/drive/MyDrive/MultiLexNorm2026/data/multilexnorm2026-dev-pub \
  --split validation \
  --output-root /content/drive/MyDrive/MultiLexNorm2026/outputs \
  --quantization 4bit \
  --prediction-strategy mfr-known \
  --preview-examples 5
```

Compare this validation ERR against the Gemma adapter and the MFR-only baseline before submitting to Codabench.

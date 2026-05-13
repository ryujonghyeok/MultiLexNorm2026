#!/usr/bin/env python3
"""Score MultiLexNorm predictions overall and by language.

Use this on validation predictions, where gold ``norm`` labels are public.
Hidden Codabench test predictions cannot be scored locally because their
``norm`` labels are blank.
"""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_LOCAL_DATASET = Path("data/multilexnorm2026-dev-pub")


@dataclass
class Metrics:
    sentences: int
    tokens: int
    changed: int
    correct: int
    lai: float
    accuracy: float
    err: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute MultiLexNorm LAI, accuracy, and ERR overall and by language."
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to predictions.json, a submission .zip, or an output directory containing predictions.json.",
    )
    parser.add_argument(
        "--dataset-id",
        default=None,
        help=(
            "Optional dataset path/Hub ID used as trusted gold labels by row order. "
            "Use this for validation runs if you want to ignore any norm values stored in predictions.json. "
            f"If omitted, {DEFAULT_LOCAL_DATASET} is used automatically when it exists and row counts match."
        ),
    )
    parser.add_argument("--split", default="validation", help="Dataset split to use with --dataset-id.")
    parser.add_argument(
        "--sort-by",
        choices=["lang", "err", "accuracy", "lai", "sentences", "tokens", "changed"],
        default="lang",
    )
    parser.add_argument("--descending", action="store_true", help="Reverse the table sort order.")
    parser.add_argument("--ignore-caps", action="store_true", help="Lowercase raw/gold/pred before scoring.")
    parser.add_argument("--csv-output", default=None, help="Optional path to write the per-language table as CSV.")
    return parser.parse_args()


def as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def load_predictions(path_text: str) -> list[dict[str, Any]]:
    path = Path(path_text)
    if path.is_dir():
        path = path / "predictions.json"

    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            candidates = [name for name in zf.namelist() if name.endswith("predictions.json")]
            if not candidates:
                raise FileNotFoundError(f"No predictions.json found inside {path}")
            with zf.open(candidates[0]) as f:
                records = json.loads(f.read().decode("utf-8"))
    else:
        with path.open(encoding="utf-8") as f:
            records = json.load(f)

    if not isinstance(records, list):
        raise ValueError("Predictions file must contain a JSON list of records.")
    return records


def load_dataset_split(dataset_id: str, split: str) -> list[dict[str, Any]]:
    from datasets import load_dataset, load_from_disk

    path = Path(dataset_id)
    data = load_from_disk(str(path)) if path.exists() else load_dataset(dataset_id)
    if split not in data:
        available = ", ".join(data.keys())
        raise KeyError(f"Split {split!r} not found. Available splits: {available}")
    return [dict(row) for row in data[split]]


def normalize_records(
    predictions: list[dict[str, Any]],
    dataset_rows: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    if dataset_rows is not None and len(dataset_rows) != len(predictions):
        raise ValueError(
            f"Dataset rows ({len(dataset_rows)}) and prediction rows ({len(predictions)}) have different lengths."
        )

    normalized = []
    for idx, pred_record in enumerate(predictions):
        gold_record = dataset_rows[idx] if dataset_rows is not None else pred_record
        try:
            raw = [str(token) for token in as_list(gold_record["raw"])]
            norm = [str(token) for token in as_list(gold_record["norm"])]
            pred = [str(token) for token in as_list(pred_record["pred"])]
            lang = str(gold_record["lang"])
        except KeyError as error:
            raise KeyError(f"Missing required key {error} at row {idx}") from error

        if len(raw) != len(norm):
            raise ValueError(f"raw/norm length mismatch at row {idx}")
        if len(raw) != len(pred):
            raise ValueError(f"raw/pred length mismatch at row {idx}")
        normalized.append({"raw": raw, "norm": norm, "pred": pred, "lang": lang})
    return normalized


def has_public_gold(records: list[dict[str, Any]]) -> bool:
    return any(any(token != "" for token in record["norm"]) for record in records)


def compute_metrics(records: list[dict[str, Any]], ignore_caps: bool = False) -> Metrics:
    correct = 0
    changed = 0
    tokens = 0

    for record in records:
        for raw_token, gold_token, pred_token in zip(record["raw"], record["norm"], record["pred"]):
            if ignore_caps:
                raw_token = raw_token.lower()
                gold_token = gold_token.lower()
                pred_token = pred_token.lower()
            if raw_token != gold_token:
                changed += 1
            if gold_token == pred_token:
                correct += 1
            tokens += 1

    if tokens == 0:
        return Metrics(len(records), 0, changed, correct, 0.0, 0.0, None)

    accuracy = correct / tokens
    lai = (tokens - changed) / tokens
    err = None if changed == 0 else (accuracy - lai) / (1 - lai)
    return Metrics(len(records), tokens, changed, correct, lai, accuracy, err)


def group_by_language(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["lang"]].append(record)
    return dict(groups)


def metric_sort_value(row: dict[str, Any], sort_by: str) -> Any:
    value = row[sort_by]
    if value is None:
        return float("-inf")
    return value


def format_percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:6.2f}"


def print_overall(metrics: Metrics) -> None:
    print("Overall score:")
    print(f"Baseline acc.(LAI): {metrics.lai * 100:.2f}")
    print(f"Accuracy:           {metrics.accuracy * 100:.2f}")
    print(f"ERR:                {'n/a' if metrics.err is None else f'{metrics.err * 100:.2f}'}")


def print_table(rows: list[dict[str, Any]]) -> None:
    print("\nBy-language score:")
    print(
        f"{'lang':<8} {'sent':>6} {'tokens':>8} {'changed':>8} "
        f"{'LAI':>8} {'Accuracy':>9} {'ERR':>8}"
    )
    print("-" * 65)
    for row in rows:
        print(
            f"{row['lang']:<8} {row['sentences']:>6} {row['tokens']:>8} {row['changed']:>8} "
            f"{format_percent(row['lai']):>8} {format_percent(row['accuracy']):>9} {format_percent(row['err']):>8}"
        )


def write_csv(rows: list[dict[str, Any]], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["lang", "sentences", "tokens", "changed", "correct", "lai", "accuracy", "err"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {path}")


def main() -> None:
    args = parse_args()
    predictions = load_predictions(args.predictions)
    dataset_rows = None
    dataset_source = args.dataset_id
    if dataset_source is None and DEFAULT_LOCAL_DATASET.exists():
        dataset_source = str(DEFAULT_LOCAL_DATASET)

    if dataset_source is not None:
        try:
            candidate_rows = load_dataset_split(dataset_source, args.split)
        except ModuleNotFoundError as error:
            if args.dataset_id is None and error.name == "datasets":
                candidate_rows = None
            else:
                raise
        if candidate_rows is not None and (args.dataset_id is not None or len(candidate_rows) == len(predictions)):
            dataset_rows = candidate_rows
            print(f"Using gold labels from {dataset_source} split={args.split}")

    records = normalize_records(predictions, dataset_rows)

    if not has_public_gold(records):
        raise ValueError(
            "No public gold norm labels found. Score validation predictions, or pass "
            "--dataset-id with --split validation. Hidden Codabench test predictions cannot be scored locally."
        )

    overall = compute_metrics(records, ignore_caps=args.ignore_caps)
    print_overall(overall)

    rows = []
    for lang, lang_records in group_by_language(records).items():
        metrics = compute_metrics(lang_records, ignore_caps=args.ignore_caps)
        rows.append(
            {
                "lang": lang,
                "sentences": metrics.sentences,
                "tokens": metrics.tokens,
                "changed": metrics.changed,
                "correct": metrics.correct,
                "lai": metrics.lai,
                "accuracy": metrics.accuracy,
                "err": metrics.err,
            }
        )

    rows.sort(key=lambda row: metric_sort_value(row, args.sort_by), reverse=args.descending)
    print_table(rows)

    if args.csv_output:
        write_csv(rows, args.csv_output)


if __name__ == "__main__":
    main()

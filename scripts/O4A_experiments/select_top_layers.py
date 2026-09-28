#!/usr/bin/env python3
"""
Select the top-k layers per (model, dataset, layer_type) from the layer study results
and save them as the layer selection consumed by o4a.experiments.

Input:  results/layers_study/<model>/<dataset>/layer_study_<model>_<dataset>_<layer_type>.csv
Output: src/o4a/selected_layers.json (default)

Only in-domain rows (eval_dataset == train_dataset) and per-seed rows (no mean/std
aggregates) are used; the metric is averaged across seeds for each layer.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
METRICS = ["accuracy", "precision", "recall", "f1", "auroc"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select top-k layers from layer study CSVs.")
    parser.add_argument(
        "--layers-study-dir",
        type=str,
        default="results/layers_study",
        help="Root directory of the layer study results (<model>/<dataset>/*.csv).",
    )
    parser.add_argument("--metric", type=str, default="f1", choices=METRICS, help="Metric used for ranking.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of layers to select.")
    parser.add_argument(
        "--output",
        type=str,
        default="src/o4a/selected_layers.json",
        help="Output JSON path (loaded by o4a.experiments).",
    )
    return parser.parse_args()


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def rank_layers(csv_path: Path, metric: str) -> tuple[str, str, str, list[tuple[int, float]]]:
    """Return (model, dataset, layer_type, [(layer, mean_metric), ...]) sorted best-first."""
    scores: dict[int, list[float]] = defaultdict(list)
    model = dataset = layer_type = None

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["eval_dataset"] != row["train_dataset"]:
                continue
            if not row["split_seed"].lstrip("-").isdigit():
                continue
            model, dataset, layer_type = row["model"], row["train_dataset"], row["layer_type"]
            scores[int(row["layer"])].append(float(row[metric]))

    if not scores:
        raise ValueError(f"No in-domain per-seed rows found in {csv_path}")

    means = [(layer, sum(v) / len(v)) for layer, v in scores.items()]
    # Best metric first; ties broken by lower layer index
    means.sort(key=lambda x: (-x[1], x[0]))
    return model, dataset, layer_type, means


def main() -> None:
    args = parse_args()
    study_dir = _resolve(args.layers_study_dir)
    output_path = _resolve(args.output)

    csv_paths = sorted(study_dir.glob("*/*/layer_study_*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No layer study CSVs found under {study_dir}")

    layers: dict = {}
    scores: dict = {}
    for csv_path in csv_paths:
        model, dataset, layer_type, ranked = rank_layers(csv_path, args.metric)
        top = ranked[: args.top_k]
        layers.setdefault(model, {}).setdefault(dataset, {})[layer_type] = sorted(l for l, _ in top)
        scores.setdefault(model, {}).setdefault(dataset, {})[layer_type] = {
            str(l): round(s, 6) for l, s in top
        }
        top_str = ", ".join(f"L{l}={s:.4f}" for l, s in top)
        print(f"{model:<24} {dataset:<25} {layer_type:<7} {top_str}")

    payload = {
        "metadata": {
            "metric": args.metric,
            "top_k": args.top_k,
            "selection": "in_domain mean over split seeds",
            "source_dir": str(study_dir.relative_to(PROJECT_ROOT) if study_dir.is_relative_to(PROJECT_ROOT) else study_dir),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "layers": layers,
        "scores": scores,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"\nSaved layer selection ({len(csv_paths)} studies) to {output_path}")


if __name__ == "__main__":
    main()

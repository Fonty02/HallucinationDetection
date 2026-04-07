#!/usr/bin/env python3
"""Generate cross-domain accuracy bar plots from cross_domain.csv.

This script recreates the two figures used in the thesis:
  - fig_cross_domain_gemma-2-9b-it_to_Llama-3.1-8B-Instruct
  - fig_cross_domain_Llama-3.1-8B-Instruct_to_gemma-2-9b-it

Values are read from:
  results/cross_domain_one_for_all/cross_domain.csv

By default it uses seed=42, matching the table values in Section 4.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


LAYER_ORDER = ("attn", "hidden", "mlp")

# Order used in the original figure:
# F->L, C->L, L->F, C->F, L->C, F->C
DOMAIN_ORDER: List[Tuple[str, str, str]] = [
    ("belief_bank_facts", "belief_bank_constraints", "F.->L."),
    ("halu_eval", "belief_bank_constraints", "C.->L."),
    ("belief_bank_constraints", "belief_bank_facts", "L.->F."),
    ("halu_eval", "belief_bank_facts", "C.->F."),
    ("belief_bank_constraints", "halu_eval", "L.->C."),
    ("belief_bank_facts", "halu_eval", "F.->C."),
]

MODEL_PAIRS = (
    ("gemma-2-9b-it", "Llama-3.1-8B-Instruct"),
    ("Llama-3.1-8B-Instruct", "gemma-2-9b-it"),
)

TRAINER_COLORS = {"attn": "#1f77b4", "hidden": "#2ca02c", "mlp": "#d62728"}
TESTER_COLORS = {"attn": "#aec7e8", "hidden": "#98df8a", "mlp": "#ff9896"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("results/cross_domain_one_for_all/cross_domain.csv"),
        help="Path to cross_domain.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Documentazione/images/grafici"),
        help="Where figures are written.",
    )
    parser.add_argument(
        "--seed",
        default="42",
        help="Seed to filter rows (default: 42).",
    )
    parser.add_argument(
        "--formats",
        default="pdf,png",
        help="Comma-separated output formats, e.g. 'pdf,png' or 'pdf'.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for raster outputs (png).",
    )
    parser.add_argument(
        "--ymin",
        type=float,
        default=None,
        help="Y-axis lower bound. If omitted, it is computed from data.",
    )
    parser.add_argument(
        "--ymax",
        type=float,
        default=1.01,
        help="Y-axis upper bound.",
    )
    return parser.parse_args()


def read_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def pick_rows_for_pair(
    rows: Iterable[Dict[str, str]],
    teacher_model: str,
    student_model: str,
    seed: str,
) -> Dict[Tuple[str, str], Dict[str, Tuple[float, float]]]:
    """Return mapping:
    (dataset_train, dataset_activation) -> layer -> (trainer_acc, tester_acc)
    """
    data: Dict[Tuple[str, str], Dict[str, Tuple[float, float]]] = {}
    for row in rows:
        if row["teacher_model"] != teacher_model:
            continue
        if row["student_model"] != student_model:
            continue
        if row["seed"] != seed:
            continue
        layer = row["layer_type"]
        if layer not in LAYER_ORDER:
            continue

        key = (row["dataset_train"], row["dataset_activation"])
        trainer_acc = float(row["teacher_on_eval_accuracy"])
        tester_acc = float(row["student_adapter_on_eval_accuracy"])
        if key not in data:
            data[key] = {}
        data[key][layer] = (trainer_acc, tester_acc)
    return data


def validate_data(
    pair_data: Dict[Tuple[str, str], Dict[str, Tuple[float, float]]],
    teacher_model: str,
    student_model: str,
    seed: str,
) -> None:
    missing: List[str] = []
    for train_ds, test_ds, label in DOMAIN_ORDER:
        key = (train_ds, test_ds)
        if key not in pair_data:
            missing.append(f"{label} (no rows)")
            continue
        for layer in LAYER_ORDER:
            if layer not in pair_data[key]:
                missing.append(f"{label} ({layer} missing)")
    if missing:
        details = "; ".join(missing)
        raise ValueError(
            "Missing data for pair "
            f"{teacher_model} -> {student_model} at seed={seed}: {details}"
        )


def plot_pair(
    pair_data: Dict[Tuple[str, str], Dict[str, Tuple[float, float]]],
    teacher_model: str,
    student_model: str,
    output_dir: Path,
    formats: List[str],
    dpi: int,
    ymin: float | None,
    ymax: float,
) -> List[Path]:
    x = np.arange(len(DOMAIN_ORDER), dtype=float)
    width = 0.12

    fig, ax = plt.subplots(figsize=(12, 4.5))

    for i, layer in enumerate(LAYER_ORDER):
        offset_train = (i - 1) * width * 2 - (width / 2)
        offset_test = (i - 1) * width * 2 + (width / 2)

        trainer_vals: List[float] = []
        tester_vals: List[float] = []
        for train_ds, test_ds, _ in DOMAIN_ORDER:
            tr, te = pair_data[(train_ds, test_ds)][layer]
            trainer_vals.append(tr)
            tester_vals.append(te)

        ax.bar(
            x + offset_train,
            trainer_vals,
            width,
            color=TRAINER_COLORS[layer],
            label=f"{layer} (Tr)",
        )
        ax.bar(
            x + offset_test,
            tester_vals,
            width,
            color=TESTER_COLORS[layer],
            edgecolor=TRAINER_COLORS[layer],
            linewidth=1.0,
            label=f"{layer} (Te)",
        )

    # Compute y-limits from data when ymin is not explicitly provided.
    all_values: List[float] = []
    for train_ds, test_ds, _ in DOMAIN_ORDER:
        for layer in LAYER_ORDER:
            tr, te = pair_data[(train_ds, test_ds)][layer]
            all_values.extend((tr, te))

    if ymin is None:
        min_val = min(all_values)
        ymin_eff = max(0.0, np.floor((min_val - 0.03) * 20.0) / 20.0)
    else:
        ymin_eff = ymin

    ax.set_ylabel("Accuracy", fontsize=12, fontweight="bold")
    ax.set_ylim(ymin_eff, ymax)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [label for _, _, label in DOMAIN_ORDER],
        rotation=0,
        ha="center",
        fontsize=11,
        fontweight="bold",
    )
    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")

    legend_handles = []
    for layer in LAYER_ORDER:
        legend_handles.append(Patch(facecolor=TRAINER_COLORS[layer], label=f"{layer} (Tr)"))
        legend_handles.append(
            Patch(
                facecolor=TESTER_COLORS[layer],
                edgecolor=TRAINER_COLORS[layer],
                linewidth=1.0,
                label=f"{layer} (Te)",
            )
        )

    ax.legend(
        handles=legend_handles,
        title="Layer (Model)",
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=True,
        prop={"size": 10, "weight": "bold"},
        title_fontproperties={"weight": "bold", "size": 11},
    )

    fig.tight_layout()

    stem = f"fig_cross_domain_{teacher_model}_to_{student_model}"
    generated: List[Path] = []
    for fmt in formats:
        out_path = output_dir / f"{stem}.{fmt}"
        save_kwargs = {"bbox_inches": "tight"}
        if fmt.lower() in {"png", "jpg", "jpeg", "tiff", "webp"}:
            save_kwargs["dpi"] = dpi
        fig.savefig(out_path, **save_kwargs)
        generated.append(out_path)

    plt.close(fig)
    return generated


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path: Path = args.csv
    rows = read_rows(csv_path)
    formats = [fmt.strip().lower() for fmt in args.formats.split(",") if fmt.strip()]
    if not formats:
        raise ValueError("No output format selected.")

    all_generated: List[Path] = []
    for teacher_model, student_model in MODEL_PAIRS:
        pair_data = pick_rows_for_pair(rows, teacher_model, student_model, seed=args.seed)
        validate_data(pair_data, teacher_model, student_model, seed=args.seed)
        generated = plot_pair(
            pair_data=pair_data,
            teacher_model=teacher_model,
            student_model=student_model,
            output_dir=output_dir,
            formats=formats,
            dpi=args.dpi,
            ymin=args.ymin,
            ymax=args.ymax,
        )
        all_generated.extend(generated)

    print("Generated files:")
    for path in all_generated:
        print(f"- {path}")


if __name__ == "__main__":
    main()

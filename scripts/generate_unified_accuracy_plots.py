#!/usr/bin/env python3
"""Generate unified accuracy bar plots directly from 4_results.tex.

This script recreates the two figures referenced in Section 4:
  - accuracy_bars_unified_gemma_to_llama
  - accuracy_bars_unified_llama_to_gemma

Data is parsed from the first three hallucination-detection tables in:
  Documentazione/sections/4_results.tex

Compared to older plot versions, this script:
  - uses RR (RidgeRegressor) in place of FL (FullLinear)
  - includes all approaches present in the parsed tables
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


LAYER_ORDER: Tuple[str, ...] = ("attn", "hidden", "mlp")

DATASET_ORDER: List[Tuple[str, str]] = [
    ("BeliefBankFacts", "Factual"),
    ("BeliefBankConstraints", "Logical"),
    ("HaluEval", "Contextual"),
]

MODEL_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("gemma-2-9b-it", "Llama-3.1-8B-Instruct"),
    ("Llama-3.1-8B-Instruct", "gemma-2-9b-it"),
)

OUTPUT_STEM_BY_PAIR: Dict[Tuple[str, str], str] = {
    ("gemma-2-9b-it", "Llama-3.1-8B-Instruct"): "accuracy_bars_unified_gemma_to_llama",
    ("Llama-3.1-8B-Instruct", "gemma-2-9b-it"): "accuracy_bars_unified_llama_to_gemma",
}

TRAINER_COLORS: Dict[str, str] = {"attn": "#1f77b4", "hidden": "#2ca02c", "mlp": "#d62728"}
TESTER_COLORS: Dict[str, str] = {"attn": "#aec7e8", "hidden": "#98df8a", "mlp": "#ff9896"}

APPROACH_CANONICAL: Dict[str, str] = {
    # Backward compatibility with older table naming.
    "FullLinear": "RidgeRegressor",
}

APPROACH_SHORT_LABELS: Dict[str, str] = {
    "RidgeRegressor": "RR",
    "Procrustes": "Proc",
    "CKA": "CKA",
    "CCA": "CCA",
    "AdapterMLP": "AdMLP",
    "FullNonLinear": "FNL",
    "ReducedNonLinear": "RNL",
    "One-For-All": "O4A",
}


@dataclass(frozen=True)
class DetectionRow:
    dataset: str
    trainer: str
    tester: str
    approach: str
    layer: str
    acc_tr: float
    acc_te: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tex",
        type=Path,
        default=Path("Documentazione/sections/4_results.tex"),
        help="Path to the LaTeX file containing result tables.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Documentazione/images/grafici"),
        help="Directory where figures are written.",
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
        help="DPI for raster formats.",
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


def canonicalize_approach(name: str) -> str:
    name = name.strip()
    return APPROACH_CANONICAL.get(name, name)


def approach_label(name: str) -> str:
    return APPROACH_SHORT_LABELS.get(name, name)


def parse_detection_rows(tex_path: Path) -> List[DetectionRow]:
    subsection_re = re.compile(
        r"\\subsection\{.* on (BeliefBankFacts|BeliefBankConstraints|HaluEval)\}"
    )
    row_re = re.compile(
        r"^\s*(?:\\rowcolor\{[^}]+\})?\s*"
        r"([^&]+?)\s*&\s*"  # trainer
        r"([^&]+?)\s*&\s*"  # tester
        r"([^&]+?)\s*&\s*"  # approach
        r"([^&]+?)\s*&\s*"  # type/layer
        r"([0-9]*\.?[0-9]+)\s*&\s*"  # AccTr
        r"([0-9]*\.?[0-9]+)\s*&\s*"  # AccTe
        r"([0-9]*\.?[0-9]+)\s*&\s*"  # AurocTr
        r"([0-9]*\.?[0-9]+)\s*"  # AurocTe
        r"\\\\\s*$"
    )

    text = tex_path.read_text(encoding="utf-8")
    current_dataset: str | None = None
    rows: List[DetectionRow] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()

        subsection_match = subsection_re.search(line)
        if subsection_match:
            current_dataset = subsection_match.group(1)
            continue

        if current_dataset is None:
            continue

        row_match = row_re.match(line)
        if not row_match:
            continue

        trainer, tester, approach, layer, acc_tr, acc_te, _, _ = row_match.groups()
        approach = canonicalize_approach(approach)
        layer = layer.strip().lower()
        if layer not in LAYER_ORDER:
            continue

        rows.append(
            DetectionRow(
                dataset=current_dataset,
                trainer=trainer.strip(),
                tester=tester.strip(),
                approach=approach,
                layer=layer,
                acc_tr=float(acc_tr),
                acc_te=float(acc_te),
            )
        )

    return rows


def collect_approach_order(rows: Iterable[DetectionRow]) -> List[str]:
    ordered: List[str] = []
    seen = set()
    for row in rows:
        if row.approach in seen:
            continue
        seen.add(row.approach)
        ordered.append(row.approach)
    return ordered


def build_pair_data(
    rows: Iterable[DetectionRow],
    trainer: str,
    tester: str,
) -> Dict[str, Dict[str, Dict[str, Tuple[float, float]]]]:
    """Return:
    dataset -> approach -> layer -> (acc_tr, acc_te)
    """
    data: Dict[str, Dict[str, Dict[str, Tuple[float, float]]]] = {}
    for row in rows:
        if row.trainer != trainer or row.tester != tester:
            continue
        data.setdefault(row.dataset, {}).setdefault(row.approach, {})[row.layer] = (
            row.acc_tr,
            row.acc_te,
        )
    return data


def validate_pair_data(
    pair_data: Dict[str, Dict[str, Dict[str, Tuple[float, float]]]],
    approach_order: Iterable[str],
    trainer: str,
    tester: str,
) -> None:
    missing: List[str] = []
    for dataset_name, dataset_label in DATASET_ORDER:
        if dataset_name not in pair_data:
            missing.append(f"{dataset_label}: no rows")
            continue
        for approach in approach_order:
            if approach not in pair_data[dataset_name]:
                missing.append(f"{dataset_label}: {approach} missing")
                continue
            for layer in LAYER_ORDER:
                if layer not in pair_data[dataset_name][approach]:
                    missing.append(f"{dataset_label}: {approach} {layer} missing")

    if missing:
        raise ValueError(
            "Missing values for pair "
            f"{trainer} -> {tester}: {'; '.join(missing)}"
        )


def compute_ymin(values: List[float], ymin_arg: float | None) -> float:
    if ymin_arg is not None:
        return ymin_arg
    min_val = min(values)
    return max(0.0, np.floor((min_val - 0.03) * 20.0) / 20.0)


def plot_pair(
    pair_data: Dict[str, Dict[str, Dict[str, Tuple[float, float]]]],
    approach_order: List[str],
    trainer: str,
    tester: str,
    output_dir: Path,
    formats: List[str],
    dpi: int,
    ymin: float | None,
    ymax: float,
) -> List[Path]:
    x = np.arange(len(approach_order), dtype=float)
    width = 0.12

    fig, axes = plt.subplots(1, len(DATASET_ORDER), figsize=(18, 5), sharey=True)
    if len(DATASET_ORDER) == 1:
        axes = [axes]

    all_values: List[float] = []
    for dataset_name, _ in DATASET_ORDER:
        for approach in approach_order:
            for layer in LAYER_ORDER:
                tr, te = pair_data[dataset_name][approach][layer]
                all_values.extend((tr, te))

    ymin_eff = compute_ymin(all_values, ymin_arg=ymin)

    for axis, (dataset_name, dataset_label) in zip(axes, DATASET_ORDER):
        for i, layer in enumerate(LAYER_ORDER):
            offset_train = (i - 1) * width * 2 - (width / 2)
            offset_test = (i - 1) * width * 2 + (width / 2)

            trainer_vals: List[float] = []
            tester_vals: List[float] = []
            for approach in approach_order:
                tr, te = pair_data[dataset_name][approach][layer]
                trainer_vals.append(tr)
                tester_vals.append(te)

            axis.bar(
                x + offset_train,
                trainer_vals,
                width,
                color=TRAINER_COLORS[layer],
            )
            axis.bar(
                x + offset_test,
                tester_vals,
                width,
                color=TESTER_COLORS[layer],
                edgecolor=TRAINER_COLORS[layer],
                linewidth=0.9,
            )

        axis.set_title(dataset_label, fontsize=11, fontweight="bold")
        axis.set_xticks(x)
        axis.set_xticklabels(
            [approach_label(name) for name in approach_order],
            rotation=20,
            ha="right",
            fontsize=10,
            fontweight="bold",
        )
        axis.set_ylim(ymin_eff, ymax)
        axis.grid(axis="y", linestyle="--", alpha=0.25, linewidth=0.8)
        axis.tick_params(axis="y", labelsize=10)
        for tick in axis.get_yticklabels():
            tick.set_fontweight("bold")

    axes[0].set_ylabel("Accuracy", fontsize=11, fontweight="bold")

    legend_handles: List[Patch] = []
    for layer in LAYER_ORDER:
        legend_handles.append(Patch(facecolor=TRAINER_COLORS[layer], label=f"{layer} (Tr)"))
        legend_handles.append(
            Patch(
                facecolor=TESTER_COLORS[layer],
                edgecolor=TRAINER_COLORS[layer],
                linewidth=0.9,
                label=f"{layer} (Te)",
            )
        )

    fig.legend(
        handles=legend_handles,
        title="Layer (Model)",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=6,
        frameon=True,
        prop={"size": 9, "weight": "bold"},
        title_fontproperties={"weight": "bold", "size": 10},
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    stem = OUTPUT_STEM_BY_PAIR.get(
        (trainer, tester),
        f"accuracy_bars_unified_{trainer}_to_{tester}".replace(" ", "_"),
    )
    generated: List[Path] = []
    for fmt in formats:
        output_path = output_dir / f"{stem}.{fmt}"
        save_kwargs = {"bbox_inches": "tight"}
        if fmt in {"png", "jpg", "jpeg", "tiff", "webp"}:
            save_kwargs["dpi"] = dpi
        fig.savefig(output_path, **save_kwargs)
        generated.append(output_path)

    plt.close(fig)
    return generated


def main() -> None:
    args = parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    formats = [fmt.strip().lower() for fmt in args.formats.split(",") if fmt.strip()]
    if not formats:
        raise ValueError("No output format selected.")

    rows = parse_detection_rows(args.tex)
    if not rows:
        raise ValueError(
            f"No detection rows found in {args.tex}. "
            "Check subsection/table format."
        )

    all_generated: List[Path] = []

    for trainer, tester in MODEL_PAIRS:
        pair_rows = [row for row in rows if row.trainer == trainer and row.tester == tester]
        if not pair_rows:
            raise ValueError(f"No rows found for pair {trainer} -> {tester}")

        approach_order = collect_approach_order(pair_rows)
        pair_data = build_pair_data(pair_rows, trainer=trainer, tester=tester)
        validate_pair_data(
            pair_data=pair_data,
            approach_order=approach_order,
            trainer=trainer,
            tester=tester,
        )
        generated = plot_pair(
            pair_data=pair_data,
            approach_order=approach_order,
            trainer=trainer,
            tester=tester,
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

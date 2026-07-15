"""Generate the cross-domain One-For-All figures used in the thesis.

Unlike ``plot_cross_domain_means_complete.py`` (which averages across seeds),
this script reproduces exactly the data reported in the thesis cross-domain
table (Tab. cross-domain-results), i.e. **seed 42 only**, for the
Gemma <-> Llama pair, and writes the two PDFs with the file names that
``4_results.tex`` already references:

    fig_cross_domain_gemma-2-9b-it_to_Llama-3.1-8B-Instruct.pdf
    fig_cross_domain_Llama-3.1-8B-Instruct_to_gemma-2-9b-it.pdf

Style (colors, bar layout, typography) is kept identical to
``plot_cross_domain_means_complete.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

# -----------------------------
# Constants / ordering (identical to plot_cross_domain_means_complete.py)
# -----------------------------
LAYER_ORDER = ["attn", "hidden", "mlp"]
SPLIT_ORDER = ["trainer", "tester"]
DATASET_CODE = {
    "belief_bank_facts": "F.",
    "belief_bank_constraints": "L.",
    "halu_eval": "C.",
}
SCENARIO_ORDER = ["F.->L.", "C.->L.", "L.->F.", "C.->F.", "L.->C.", "F.->C."]

COLORS_TRAINER = {"attn": "#1f77b4", "hidden": "#2ca02c", "mlp": "#d62728"}
COLORS_TESTER = {"attn": "#aec7e8", "hidden": "#98df8a", "mlp": "#ff9896"}

BAR_WIDTH = 0.12
OFFSETS = {
    ("attn", "trainer"): -2.5 * BAR_WIDTH,
    ("attn", "tester"): -1.5 * BAR_WIDTH,
    ("hidden", "trainer"): -0.5 * BAR_WIDTH,
    ("hidden", "tester"): 0.5 * BAR_WIDTH,
    ("mlp", "trainer"): 1.5 * BAR_WIDTH,
    ("mlp", "tester"): 2.5 * BAR_WIDTH,
}

LABEL_STYLE = {"fontsize": 24, "fontweight": "bold"}
LEGEND_PROP = {"size": 20, "weight": "bold"}
LEGEND_TITLE_FONTSIZE = 24
TICK_FONTSIZE = 24

# Thesis cross-domain table only reports the Gemma <-> Llama pair.
THESIS_PAIRS = [
    ("gemma-2-9b-it", "Llama-3.1-8B-Instruct"),
    ("Llama-3.1-8B-Instruct", "gemma-2-9b-it"),
]

SEED = 42


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Plot seed-42 cross-domain metrics for the thesis figures."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=repo_root / "results" / "cross_domain_one_for_all" / "cross_domain.csv",
        help="Path to cross_domain.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=repo_root / "Documentazione" / "images" / "grafici",
        help="Output directory (defaults to the thesis images folder)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="accuracy",
        choices=["accuracy", "precision", "recall", "f1", "auroc"],
        help="Metric to plot",
    )
    return parser.parse_args()


def build_long_dataframe(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    teacher_col = f"teacher_on_eval_{metric}"
    student_col = f"student_adapter_on_eval_{metric}"
    id_columns = [
        "teacher_model", "student_model",
        "dataset_train", "dataset_activation",
        "layer_type", "seed",
    ]
    missing = [c for c in [teacher_col, student_col, *id_columns] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing cross-domain columns: {missing}")

    teacher_df = df[id_columns + [teacher_col]].rename(columns={teacher_col: "score"})
    teacher_df["split"] = "trainer"
    student_df = df[id_columns + [student_col]].rename(columns={student_col: "score"})
    student_df["split"] = "tester"

    long_df = pd.concat([teacher_df, student_df], ignore_index=True)
    long_df["score"] = pd.to_numeric(long_df["score"], errors="coerce")
    long_df = long_df.dropna(subset=["score"])

    long_df = long_df[long_df["layer_type"].isin(LAYER_ORDER)]
    long_df["train_code"] = long_df["dataset_train"].map(DATASET_CODE)
    long_df["activation_code"] = long_df["dataset_activation"].map(DATASET_CODE)
    long_df = long_df.dropna(subset=["train_code", "activation_code"])
    long_df = long_df[long_df["train_code"] != long_df["activation_code"]]

    long_df["scenario"] = long_df["train_code"] + "->" + long_df["activation_code"]
    long_df = long_df[long_df["scenario"].isin(SCENARIO_ORDER)]
    return long_df


def plot_pair(pair_df: pd.DataFrame, metric: str, out_path: Path) -> None:
    scenario_order = [s for s in SCENARIO_ORDER if s in pair_df["scenario"].unique()]
    if not scenario_order:
        print(f"  (skip {out_path.name}: no data)")
        return

    x = np.arange(len(scenario_order))
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    ax.set_facecolor("#fdfdfd")

    for layer in LAYER_ORDER:
        for split, color, edge, lw in (
            ("trainer", COLORS_TRAINER[layer], "black", 0.0),
            ("tester", COLORS_TESTER[layer], COLORS_TRAINER[layer], 1.0),
        ):
            values = [
                pair_df.loc[
                    (pair_df["scenario"] == sc)
                    & (pair_df["layer_type"] == layer)
                    & (pair_df["split"] == split),
                    "score",
                ].mean()
                if not pair_df.loc[
                    (pair_df["scenario"] == sc)
                    & (pair_df["layer_type"] == layer)
                    & (pair_df["split"] == split)
                ].empty
                else 0
                for sc in scenario_order
            ]
            ax.bar(
                x + OFFSETS[(layer, split)],
                values,
                width=BAR_WIDTH,
                color=color,
                edgecolor=edge,
                linewidth=lw,
            )

    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel(metric.capitalize(), **LABEL_STYLE)
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_order, rotation=0, ha="center")
    for lbl in ax.get_xticklabels():
        lbl.set_fontsize(TICK_FONTSIZE)
        lbl.set_fontweight("bold")
    for lbl in ax.get_yticklabels():
        lbl.set_fontsize(TICK_FONTSIZE)
        lbl.set_fontweight("bold")

    legend_handles = []
    for layer in LAYER_ORDER:
        legend_handles.append(Patch(facecolor=COLORS_TRAINER[layer], label=f"{layer} (Tr)"))
        legend_handles.append(
            Patch(
                facecolor=COLORS_TESTER[layer],
                edgecolor=COLORS_TRAINER[layer],
                linewidth=1,
                label=f"{layer} (Te)",
            )
        )
    lg = fig.legend(
        handles=legend_handles,
        title="Layer (Model)",
        loc="center left",
        bbox_to_anchor=(0.9, 0.5),
        ncol=1,
        frameon=True,
        prop=LEGEND_PROP,
    )
    lg.get_title().set_fontweight("bold")
    lg.get_title().set_fontsize(LEGEND_TITLE_FONTSIZE)

    fig.tight_layout(rect=[0, 0, 0.86, 1])
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)
    df = df[df["seed"] == SEED]
    if df.empty:
        raise SystemExit(f"No rows with seed=={SEED} in {args.csv}")

    long_df = build_long_dataframe(df, args.metric)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for teacher, student in THESIS_PAIRS:
        pair_df = long_df[
            (long_df["teacher_model"] == teacher)
            & (long_df["student_model"] == student)
        ]
        out_path = args.out_dir / f"fig_cross_domain_{teacher}_to_{student}.pdf"
        plot_pair(pair_df, args.metric, out_path)


if __name__ == "__main__":
    main()

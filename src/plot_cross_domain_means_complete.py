from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

# -----------------------------
# Constants / ordering
# -----------------------------
LAYER_ORDER = ["attn", "hidden", "mlp"]
SPLIT_ORDER = ["trainer", "tester"]
SPLIT_LABEL = {"trainer": "Tr", "tester": "Te"}
DATASET_CODE = {
    "belief_bank_facts": "F.",
    "belief_bank_constraints": "L.",
    "halu_eval": "C.",
}
SCENARIO_ORDER = ["F.->L.", "C.->L.", "L.->F.", "C.->F.", "L.->C.", "F.->C."]

# -----------------------------
# Colors  ←  identici al primo script
# -----------------------------
COLORS_TRAINER = {
    "attn":   "#1f77b4",
    "hidden": "#2ca02c",
    "mlp":    "#d62728",
}
COLORS_TESTER = {
    "attn":   "#aec7e8",
    "hidden": "#98df8a",
    "mlp":    "#ff9896",
}

# -----------------------------
# Bar-layout  ←  identico al primo script
# -----------------------------
BAR_WIDTH = 0.12
OFFSETS = {
    ("attn",   "trainer"): -2.5 * BAR_WIDTH,
    ("attn",   "tester"):  -1.5 * BAR_WIDTH,
    ("hidden", "trainer"): -0.5 * BAR_WIDTH,
    ("hidden", "tester"):   0.5 * BAR_WIDTH,
    ("mlp",    "trainer"):  1.5 * BAR_WIDTH,
    ("mlp",    "tester"):   2.5 * BAR_WIDTH,
}

# -----------------------------
# Typography  ←  identica al primo script
# -----------------------------
LABEL_STYLE   = {"fontsize": 24, "fontweight": "bold"}
LEGEND_PROP   = {"size": 20, "weight": "bold"}
TITLE_FONTSIZE = 26
LEGEND_TITLE_FONTSIZE = 24
TICK_FONTSIZE  = 24


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Plot mean cross-domain metrics across seeds."
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
        default=repo_root / "src" / "plots" / "cross_domain_means",
        help="Output directory for generated plots",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="accuracy",
        choices=["accuracy", "precision", "recall", "f1", "auroc"],
        help="Metric to plot",
    )
    return parser.parse_args()


def safe_name(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", text)


# -----------------------------
# Data helpers
# -----------------------------
def build_long_dataframe(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    teacher_col = f"teacher_on_eval_{metric}"
    student_col = f"student_adapter_on_eval_{metric}"
    required = [teacher_col, student_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing cross-domain columns: {missing}")

    required_ids = [
        "teacher_model", "student_model",
        "dataset_train", "dataset_activation",
        "layer_type", "seed",
    ]
    missing_ids = [col for col in required_ids if col not in df.columns]
    if missing_ids:
        raise ValueError(f"Missing cross-domain id columns: {missing_ids}")

    id_columns = [
        "teacher_model", "student_model",
        "dataset_train", "dataset_activation",
        "layer_type", "seed",
    ]

    teacher_df = df[id_columns + [teacher_col]].copy()
    teacher_df = teacher_df.rename(columns={teacher_col: "score"})
    teacher_df["split"] = "trainer"

    student_df = df[id_columns + [student_col]].copy()
    student_df = student_df.rename(columns={student_col: "score"})
    student_df["split"] = "tester"

    long_df = pd.concat([teacher_df, student_df], ignore_index=True)
    long_df["score"] = pd.to_numeric(long_df["score"], errors="coerce")
    long_df["seed"]  = pd.to_numeric(long_df["seed"],  errors="coerce")
    long_df = long_df.dropna(
        subset=["teacher_model", "student_model",
                "dataset_train", "dataset_activation",
                "layer_type", "split", "score", "seed"]
    )

    long_df = long_df[long_df["layer_type"].isin(LAYER_ORDER)]
    long_df = long_df[long_df["split"].isin(SPLIT_ORDER)]

    long_df["train_code"]      = long_df["dataset_train"].map(DATASET_CODE)
    long_df["activation_code"] = long_df["dataset_activation"].map(DATASET_CODE)
    long_df = long_df.dropna(subset=["train_code", "activation_code"])
    long_df = long_df[long_df["train_code"] != long_df["activation_code"]]

    long_df["scenario"]   = long_df["train_code"] + "->" + long_df["activation_code"]
    long_df              = long_df[long_df["scenario"].isin(SCENARIO_ORDER)]
    long_df["pair_label"] = long_df["teacher_model"] + " -> " + long_df["student_model"]

    return long_df


# -----------------------------
# Plotting  ←  stile identico al primo script
# -----------------------------
def plot_pair_mean(
    pair_df: pd.DataFrame,
    pair_label: str,
    metric: str,
    output_dir: Path,
) -> None:
    scenario_order = [s for s in SCENARIO_ORDER if s in pair_df["scenario"].unique()]
    if not scenario_order:
        return

    x = np.arange(len(scenario_order))

    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    ax.set_facecolor("#fdfdfd")

    for layer in LAYER_ORDER:
        # — Trainer bars (solid fill, no visible border) —
        values_trainer = [
            pair_df.loc[
                (pair_df["scenario"] == sc) &
                (pair_df["layer_type"] == layer) &
                (pair_df["split"] == "trainer"),
                "score",
            ].mean() if not pair_df.loc[
                (pair_df["scenario"] == sc) &
                (pair_df["layer_type"] == layer) &
                (pair_df["split"] == "trainer")
            ].empty else 0
            for sc in scenario_order
        ]
        ax.bar(
            x + OFFSETS[(layer, "trainer")],
            values_trainer,
            width=BAR_WIDTH,
            color=COLORS_TRAINER[layer],
            edgecolor="black",
            linewidth=0.0,
        )

        # — Tester bars (lighter fill, colored border) —
        values_tester = [
            pair_df.loc[
                (pair_df["scenario"] == sc) &
                (pair_df["layer_type"] == layer) &
                (pair_df["split"] == "tester"),
                "score",
            ].mean() if not pair_df.loc[
                (pair_df["scenario"] == sc) &
                (pair_df["layer_type"] == layer) &
                (pair_df["split"] == "tester")
            ].empty else 0
            for sc in scenario_order
        ]
        ax.bar(
            x + OFFSETS[(layer, "tester")],
            values_tester,
            width=BAR_WIDTH,
            color=COLORS_TESTER[layer],
            edgecolor=COLORS_TRAINER[layer],
            linewidth=1,
        )

    # — Axis styling —
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel(metric.capitalize(), **LABEL_STYLE)
    ax.set_title(pair_label, fontsize=TITLE_FONTSIZE, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(scenario_order, rotation=0, ha="center")

    for lbl in ax.get_xticklabels():
        lbl.set_fontsize(TICK_FONTSIZE)
        lbl.set_fontweight("bold")
    for lbl in ax.get_yticklabels():
        lbl.set_fontsize(TICK_FONTSIZE)
        lbl.set_fontweight("bold")

    # — Legend: vertical, on the right —
    legend_handles = []
    for layer in LAYER_ORDER:
        legend_handles.append(
            Patch(facecolor=COLORS_TRAINER[layer], label=f"{layer} (Tr)")
        )
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

    output_path = output_dir / f"{safe_name(pair_label)}_{metric}_mean.pdf"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.csv)
    long_df = build_long_dataframe(df, args.metric)

    mean_df = (
        long_df.groupby(
            ["teacher_model", "student_model", "pair_label",
             "scenario", "layer_type", "split"],
            as_index=False,
        )["score"]
        .mean()
        .sort_values(["teacher_model", "student_model", "scenario", "split", "layer_type"])
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    pairs = (
        mean_df[["teacher_model", "student_model", "pair_label"]]
        .drop_duplicates()
        .sort_values(["teacher_model", "student_model"])
    )

    for row in pairs.itertuples(index=False):
        subset = mean_df[mean_df["pair_label"] == row.pair_label]
        if subset.empty:
            continue
        plot_pair_mean(
            pair_df=subset,
            pair_label=row.pair_label,
            metric=args.metric,
            output_dir=args.out_dir,
        )

    print(f"\nGenerated {len(pairs)} cross-domain mean plots in: {args.out_dir}")


if __name__ == "__main__":
    main()
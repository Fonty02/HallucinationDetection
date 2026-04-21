from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

LAYER_ORDER = ["attn", "hidden", "mlp"]
SPLIT_ORDER = ["trainer", "tester"]
SPLIT_LABEL = {"trainer": "Tr", "tester": "Te"}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Plot cross-domain boxplots across seeds."
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
        default=repo_root / "src" / "plots" / "cross_domain_boxplots",
        help="Output directory for generated plots",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="accuracy",
        choices=["accuracy", "precision", "recall", "f1", "auroc"],
        help="Metric to plot",
    )
    parser.add_argument(
        "--show-fliers",
        action="store_true",
        help="Show outliers in boxplots",
    )
    return parser.parse_args()


def safe_name(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", text)


def set_dynamic_ylim(ax: plt.Axes, scores: pd.Series, pad_ratio: float = 0.08, min_span: float = 0.08) -> None:
    values = pd.to_numeric(scores, errors="coerce").dropna()
    if values.empty:
        return

    vmin = float(values.min())
    vmax = float(values.max())
    span = vmax - vmin
    pad = max(span * pad_ratio, min_span / 2)

    lower = max(0.0, vmin - pad)
    upper = min(1.0, vmax + pad)

    if upper - lower < min_span:
        center = (lower + upper) / 2
        lower = max(0.0, center - min_span / 2)
        upper = min(1.0, center + min_span / 2)
        if upper - lower < min_span:
            if lower == 0.0:
                upper = min(1.0, lower + min_span)
            else:
                lower = max(0.0, upper - min_span)

    ax.set_ylim(lower, upper)


def build_long_dataframe(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    teacher_col = f"teacher_on_eval_{metric}"
    student_col = f"student_adapter_on_eval_{metric}"
    required = [teacher_col, student_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing cross-domain columns: {missing}")

    id_columns = ["source_experiment", "encoder_experiment", "dataset_activation", "layer_type", "seed"]

    teacher_df = df[id_columns + [teacher_col]].copy()
    teacher_df = teacher_df.rename(columns={teacher_col: "score"})
    teacher_df["split"] = "trainer"

    student_df = df[id_columns + [student_col]].copy()
    student_df = student_df.rename(columns={student_col: "score"})
    student_df["split"] = "tester"

    long_df = pd.concat([teacher_df, student_df], ignore_index=True)
    long_df["score"] = pd.to_numeric(long_df["score"], errors="coerce")
    long_df["seed"] = pd.to_numeric(long_df["seed"], errors="coerce")
    long_df = long_df.dropna(subset=["source_experiment", "layer_type", "split", "score", "seed"])

    long_df = long_df[long_df["layer_type"].isin(LAYER_ORDER)]
    long_df = long_df[long_df["split"].isin(SPLIT_ORDER)]

    long_df["layer_split"] = (
        long_df["layer_type"]
        + " ("
        + long_df["split"].map(SPLIT_LABEL).fillna(long_df["split"])
        + ")"
    )
    return long_df


def plot_source_experiment_boxplot(
    source_df: pd.DataFrame,
    source_experiment: str,
    metric: str,
    output_dir: Path,
    show_fliers: bool,
) -> None:
    order = [f"{layer} ({SPLIT_LABEL[split]})" for layer in LAYER_ORDER for split in SPLIT_ORDER]
    title_suffix = source_df["dataset_activation"].iloc[0]

    plt.figure(figsize=(8.5, 5.5), dpi=150)
    ax = sns.boxplot(
        data=source_df,
        x="layer_split",
        y="score",
        order=order,
        hue="layer_split",
        hue_order=order,
        dodge=False,
        showfliers=show_fliers,
    )

    set_dynamic_ylim(ax, source_df["score"])
    plt.xlabel("Layer (Model)")
    plt.ylabel(metric.capitalize())
    plt.title(f"{source_experiment} [{title_suffix}] - {metric} distribution across seeds")
    plt.xticks(rotation=20, ha="right")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        plt.legend(title="Layer (Model)", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    output_path = output_dir / f"{safe_name(source_experiment)}_{metric}_boxplot.pdf"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()
    sns.set_theme(style="whitegrid")

    df = pd.read_csv(args.csv)
    long_df = build_long_dataframe(df, args.metric)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    experiments = sorted(long_df["source_experiment"].unique().tolist())

    for source_experiment in experiments:
        subset = long_df[long_df["source_experiment"] == source_experiment]
        if subset.empty:
            continue
        plot_source_experiment_boxplot(
            source_df=subset,
            source_experiment=source_experiment,
            metric=args.metric,
            output_dir=args.out_dir,
            show_fliers=args.show_fliers,
        )

    print(f"Generated {len(experiments)} cross-domain boxplots in: {args.out_dir}")


if __name__ == "__main__":
    main()

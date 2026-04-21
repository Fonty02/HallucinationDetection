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
METHOD_ORDER = [
    "ridge_regressor",
    "procrustes",
    "cka",
    "cca",
    "hybrid",
    "full_nonlinear",
    "reduced_nonlinear",
    "one_for_all",
]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Plot classic-experiment boxplots across seeds."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=repo_root / "results" / "experiments" / "experiments.csv",
        help="Path to experiments.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=repo_root / "src" / "plots" / "classic_boxplots",
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


def get_method_order(values: pd.Series) -> list[str]:
    seen = set(values.unique().tolist())
    ordered = [m for m in METHOD_ORDER if m in seen]
    ordered.extend(sorted(seen - set(ordered)))
    return ordered


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
    pattern = re.compile(
        rf"^(?P<method>.+)_(?P<split>trainer|tester)_{re.escape(metric)}$"
    )

    metric_columns: list[str] = []
    meta_map: dict[str, tuple[str, str]] = {}
    for col in df.columns:
        match = pattern.match(col)
        if match is None:
            continue
        metric_columns.append(col)
        meta_map[col] = (match.group("method"), match.group("split"))

    if not metric_columns:
        raise ValueError(f"No columns found for metric '{metric}'.")

    id_columns = ["experiment", "seed", "layer_type"]
    long_df = df[id_columns + metric_columns].melt(
        id_vars=id_columns,
        value_vars=metric_columns,
        var_name="metric_col",
        value_name="score",
    )

    long_df["method"] = long_df["metric_col"].map(lambda c: meta_map[c][0])
    long_df["split"] = long_df["metric_col"].map(lambda c: meta_map[c][1])
    long_df = long_df.drop(columns=["metric_col"])

    long_df["score"] = pd.to_numeric(long_df["score"], errors="coerce")
    long_df["seed"] = pd.to_numeric(long_df["seed"], errors="coerce")
    long_df = long_df.dropna(
        subset=["experiment", "seed", "layer_type", "method", "split", "score"]
    )
    long_df = long_df[long_df["layer_type"].isin(LAYER_ORDER)]
    long_df = long_df[long_df["split"].isin(SPLIT_ORDER)]

    long_df["layer_split"] = (
        long_df["layer_type"]
        + " ("
        + long_df["split"].map(SPLIT_LABEL).fillna(long_df["split"])
        + ")"
    )
    return long_df


def plot_experiment_boxplot(
    experiment_df: pd.DataFrame,
    experiment_name: str,
    metric: str,
    output_dir: Path,
    show_fliers: bool,
) -> None:
    method_order = get_method_order(experiment_df["method"])
    hue_order = [f"{layer} ({SPLIT_LABEL[split]})" for layer in LAYER_ORDER for split in SPLIT_ORDER]

    width = max(12.0, len(method_order) * 1.8)
    plt.figure(figsize=(width, 6), dpi=150)
    ax = sns.boxplot(
        data=experiment_df,
        x="method",
        y="score",
        hue="layer_split",
        order=method_order,
        hue_order=hue_order,
        showfliers=show_fliers,
    )

    set_dynamic_ylim(ax, experiment_df["score"])
    plt.xlabel("Method")
    plt.ylabel(metric.capitalize())
    plt.title(f"{experiment_name} - {metric} distribution across seeds")
    plt.xticks(rotation=20, ha="right")
    plt.legend(title="Layer (Model)", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    output_path = output_dir / f"{safe_name(experiment_name)}_{metric}_boxplot.pdf"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()
    sns.set_theme(style="whitegrid")

    df = pd.read_csv(args.csv)
    long_df = build_long_dataframe(df, args.metric)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    experiments = sorted(long_df["experiment"].unique().tolist())

    for experiment_name in experiments:
        subset = long_df[long_df["experiment"] == experiment_name]
        if subset.empty:
            continue
        plot_experiment_boxplot(
            experiment_df=subset,
            experiment_name=experiment_name,
            metric=args.metric,
            output_dir=args.out_dir,
            show_fliers=args.show_fliers,
        )

    print(f"Generated {len(experiments)} boxplots in: {args.out_dir}")


if __name__ == "__main__":
    main()

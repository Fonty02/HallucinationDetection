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
        description="Plot mean classic-experiment metrics across seeds."
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
        default=repo_root / "src" / "plots" / "classic_means",
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


def get_method_order(values: pd.Series) -> list[str]:
    seen = set(values.unique().tolist())
    ordered = [m for m in METHOD_ORDER if m in seen]
    ordered.extend(sorted(seen - set(ordered)))
    return ordered


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
    return long_df


def add_layer_split_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["layer_split"] = (
        df["layer_type"] + " (" + df["split"].map(SPLIT_LABEL).fillna(df["split"]) + ")"
    )
    return df


def plot_experiment_mean(
    experiment_df: pd.DataFrame,
    experiment_name: str,
    metric: str,
    output_dir: Path,
) -> None:
    method_order = get_method_order(experiment_df["method"])
    hue_order = [f"{layer} ({SPLIT_LABEL[split]})" for layer in LAYER_ORDER for split in SPLIT_ORDER]

    width = max(12.0, len(method_order) * 1.6)
    plt.figure(figsize=(width, 6), dpi=150)
    sns.barplot(
        data=experiment_df,
        x="method",
        y="score",
        hue="layer_split",
        order=method_order,
        hue_order=hue_order,
        errorbar=None,
    )

    plt.ylim(0.0, 1.0)
    plt.xlabel("Method")
    plt.ylabel(metric.capitalize())
    plt.title(f"{experiment_name} - Mean {metric} across seeds")
    plt.xticks(rotation=20, ha="right")
    plt.legend(title="Layer (Model)", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    output_path = output_dir / f"{safe_name(experiment_name)}_{metric}_mean.pdf"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()
    sns.set_theme(style="whitegrid")

    df = pd.read_csv(args.csv)
    long_df = build_long_dataframe(df, args.metric)

    mean_df = (
        long_df.groupby(["experiment", "method", "layer_type", "split"], as_index=False)["score"]
        .mean()
        .sort_values(["experiment", "method", "layer_type", "split"])
    )
    mean_df = add_layer_split_label(mean_df)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    experiments = sorted(mean_df["experiment"].unique().tolist())

    for experiment_name in experiments:
        subset = mean_df[mean_df["experiment"] == experiment_name]
        if subset.empty:
            continue
        plot_experiment_mean(
            experiment_df=subset,
            experiment_name=experiment_name,
            metric=args.metric,
            output_dir=args.out_dir,
        )

    print(f"Generated {len(experiments)} mean plots in: {args.out_dir}")


if __name__ == "__main__":
    main()

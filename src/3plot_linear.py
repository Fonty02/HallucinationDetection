from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

LAYER_ORDER = ["attn", "hidden", "mlp"]
SPLIT_ORDER = ["trainer", "tester"]
SPLIT_LABEL = {"trainer": "Tr", "tester": "Te"}
METHOD_ORDER = [
    "Procrustes",
    "Cca",
    "Cka",
    "RidgeRegressor",
    "OneForAll",
]

COLORS_TRAINER = {
    "attn": '#1f77b4',
    "hidden": '#2ca02c',
    "mlp": '#d62728'
}
COLORS_TESTER = {
    "attn": '#aec7e8',
    "hidden": '#98df8a',
    "mlp": '#ff9896'
}

def dataset_to_camel_case(name: str) -> str:
    parts = name.split('_')
    return ''.join(p.capitalize() for p in parts)

def method_to_display_name(method: str) -> str:
    lower = method.lower()
    if lower == "cca": return "CCA"
    if lower == "cka": return "CKA"
    return method

def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Plot mean classic-experiment metrics across seeds, one figure per model pair."
    )
    parser.add_argument("--csv", type=Path, default=repo_root / "results" / "experiments" / "experiments.csv")
    parser.add_argument("--out-dir", type=Path, default=repo_root / "src" / "plots" / "linear_classic_means_multiplot")
    parser.add_argument("--metric", type=str, default="accuracy",
                        choices=["accuracy", "precision", "recall", "f1", "auroc"])
    return parser.parse_args()

def safe_name(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", text)

def build_long_dataframe(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    pattern = re.compile(rf"^(?P<method>.+)_(?P<split>trainer|tester)_{re.escape(metric)}$")
    metric_columns = []
    meta_map = {}
    for col in df.columns:
        match = pattern.match(col)
        if match:
            metric_columns.append(col)
            meta_map[col] = (match.group("method"), match.group("split"))
    if not metric_columns:
        raise ValueError(f"No columns found for metric '{metric}'.")

    id_columns = ["experiment", "seed", "layer_type", "dataset"]
    long_df = df[id_columns + metric_columns].melt(
        id_vars=id_columns,
        value_vars=metric_columns,
        var_name="metric_col",
        value_name="score",
    )
    # Normalise method names to title-case without spaces (matches METHOD_ORDER)
    long_df["method"] = long_df["metric_col"].map(
        lambda c: meta_map[c][0].replace('_', ' ').title().replace(' ', '')
    )
    long_df["split"] = long_df["metric_col"].map(lambda c: meta_map[c][1])
    long_df = long_df.drop(columns=["metric_col"])
    long_df["score"] = pd.to_numeric(long_df["score"], errors="coerce")
    long_df["seed"] = pd.to_numeric(long_df["seed"], errors="coerce")
    long_df = long_df.dropna(subset=["experiment", "seed", "layer_type", "method", "split", "score"])
    long_df = long_df[long_df["method"].isin(METHOD_ORDER)]
    long_df = long_df[long_df["layer_type"].isin(LAYER_ORDER)]
    long_df = long_df[long_df["split"].isin(SPLIT_ORDER)]
    return long_df

def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)
    long_df = build_long_dataframe(df, args.metric)

    # FIX: step 1 — media sui seed mantenendo il nome completo dell'experiment,
    # esattamente come fa plot_classic_means_linear.py.
    # In questo modo ridge_regressor e one_for_all hanno gli stessi valori
    # del single-plot corrispondente.
    mean_df = (
        long_df
        .groupby(["experiment", "dataset", "method", "layer_type", "split"], as_index=False)["score"]
        .mean()
        .sort_values(["experiment", "dataset", "method", "layer_type", "split"])
    )

    # FIX: step 2 — model_pair estratto DOPO la media, usato solo per
    # raggruppare i subplot nella stessa figura, non per aggregare dati.
    mean_df["model_pair"] = mean_df["experiment"].apply(
        lambda x: x.split('_')[0] if isinstance(x, str) else x
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_pairs = sorted(mean_df["model_pair"].unique())

    bar_width = 0.12
    offsets = {
        ("attn",   "trainer"): -2.5 * bar_width,
        ("attn",   "tester"):  -1.5 * bar_width,
        ("hidden", "trainer"): -0.5 * bar_width,
        ("hidden", "tester"):   0.5 * bar_width,
        ("mlp",    "trainer"):  1.5 * bar_width,
        ("mlp",    "tester"):   2.5 * bar_width,
    }

    for pair in model_pairs:
        pair_data = mean_df[mean_df["model_pair"] == pair]
        datasets = sorted(pair_data["dataset"].unique())

        if not datasets:
            continue

        print(f"Generating plot for: {pair} | Datasets: {datasets}")

        n_ds = len(datasets)
        fig, axes = plt.subplots(1, n_ds, figsize=(5.5 * n_ds, 5.5), sharey=True)
        if n_ds == 1:
            axes = [axes]

        for idx, ds in enumerate(datasets):
            ax = axes[idx]
            ax.set_facecolor("#fdfdfd")
            subset = pair_data[pair_data["dataset"] == ds]

            data = {}
            for method in METHOD_ORDER:
                data[method] = {}
                for layer in LAYER_ORDER:
                    for split in SPLIT_ORDER:
                        vals = subset[
                            (subset["method"] == method) &
                            (subset["layer_type"] == layer) &
                            (subset["split"] == split)
                        ]["score"].values
                        data[method][(layer, split)] = float(vals[0]) if len(vals) > 0 else 0.0

            x = np.arange(len(METHOD_ORDER))

            for layer in LAYER_ORDER:
                tr_vals = [data[m].get((layer, "trainer"), 0) for m in METHOD_ORDER]
                ax.bar(x + offsets[(layer, "trainer")], tr_vals, width=bar_width,
                       color=COLORS_TRAINER[layer], edgecolor='black', linewidth=0.0)
                te_vals = [data[m].get((layer, "tester"), 0) for m in METHOD_ORDER]
                ax.bar(x + offsets[(layer, "tester")], te_vals, width=bar_width,
                       color=COLORS_TESTER[layer], edgecolor=COLORS_TRAINER[layer], linewidth=1.0)

            ax.set_ylim(0.0, 1.0)
            ax.set_xticks(x)

            xtick_labels = [method_to_display_name(m) for m in METHOD_ORDER]
            ax.set_xticklabels(xtick_labels, rotation=15, ha='right', fontsize=10, fontweight='bold')
            for label in ax.get_yticklabels():
                label.set_fontweight('bold')
            ax.set_title(dataset_to_camel_case(ds), fontsize=14, fontweight='bold')

        axes[0].set_ylabel(args.metric.capitalize(), fontsize=12, fontweight='bold')

        legend_handles = []
        for layer in LAYER_ORDER:
            legend_handles.append(Patch(facecolor=COLORS_TRAINER[layer], label=f"{layer} (Tr)"))
            legend_handles.append(Patch(facecolor=COLORS_TESTER[layer],
                                        edgecolor=COLORS_TRAINER[layer],
                                        linewidth=1, label=f"{layer} (Te)"))

        fig.legend(handles=legend_handles, title="Layer (Model)",
                   loc="lower center", bbox_to_anchor=(0.5, 0.82),
                   ncol=6, frameon=True,
                   prop={'size': 10, 'weight': 'bold'},
                   title_fontproperties={'weight': 'bold', 'size': 11})

        fig.tight_layout(rect=[0, 0, 1, 0.82])

        out_path = args.out_dir / f"{safe_name(pair)}_{args.metric}_mean_subplots.pdf"
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()
        print(f"Salvato: {out_path}")

if __name__ == "__main__":
    main()
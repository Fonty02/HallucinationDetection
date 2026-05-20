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

# Colori vividi esattamente come nel codice di riferimento
COLORS_TRAINER = {
    "attn": '#1f77b4',   # blue
    "hidden": '#2ca02c', # green
    "mlp": '#d62728'     # red
}
COLORS_TESTER = {
    "attn": '#aec7e8',   # light blue
    "hidden": '#98df8a', # light green
    "mlp": '#ff9896'     # light red
}

def dataset_to_camel_case(name: str) -> str:
    parts = name.split('_')
    return ''.join(p.capitalize() for p in parts)

def method_to_display_name(method: str) -> str:
    lower = method.lower()
    if lower == "cca":
        return "CCA"
    if lower == "cka":
        return "CKA"
    # Procrustes, RidgeRegressor, OneForAll restano così
    return method

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
        default=repo_root / "src" / "plots" / "classic_means_revised",
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

def build_long_dataframe(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    pattern = re.compile(
        rf"^(?P<method>.+)_(?P<split>trainer|tester)_{re.escape(metric)}$"
    )
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
    long_df["method"] = long_df["metric_col"].map(lambda c: meta_map[c][0].replace('_', ' ').title().replace(' ', ''))
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

    # Media su tutti i seed
    mean_df = (
        long_df.groupby(["experiment", "dataset", "method", "layer_type", "split"], as_index=False)["score"]
        .mean()
        .sort_values(["experiment", "dataset", "method", "layer_type", "split"])
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    experiments = sorted(mean_df["experiment"].unique())

    for exp_name in experiments:
        subset = mean_df[mean_df["experiment"] == exp_name]
        if subset.empty:
            continue
        dataset_name = subset["dataset"].iloc[0]

        # Prepara i dati per il plotting stile "crossValScript.py"
        approaches = METHOD_ORDER  # ordine fisso
        layers = LAYER_ORDER
        # Costruiamo una matrice dei valori
        data = {}
        for approach in approaches:
            data[approach] = {}
            for layer in layers:
                for split in SPLIT_ORDER:
                    val = subset[(subset["method"] == approach) &
                                 (subset["layer_type"] == layer) &
                                 (subset["split"] == split)]["score"].values
                    data[approach][(layer, split)] = val[0] if len(val) > 0 else 0.0

        # Parametri grafici come nel codice originale
        x = np.arange(len(approaches))
        bar_width = 0.12
        offsets = {
            ("attn", "trainer"): -2.5 * bar_width,
            ("attn", "tester"): -1.5 * bar_width,
            ("hidden", "trainer"): -0.5 * bar_width,
            ("hidden", "tester"): 0.5 * bar_width,
            ("mlp", "trainer"): 1.5 * bar_width,
            ("mlp", "tester"): 2.5 * bar_width,
        }

        fig, ax = plt.subplots(figsize=(11, 4))
        ax.set_facecolor("#fdfdfd")

        for layer in layers:
            # Trainer bars (nessun bordo)
            values_trainer = [data[approach].get((layer, "trainer"), 0) for approach in approaches]
            ax.bar(
                x + offsets[(layer, "trainer")],
                values_trainer,
                width=bar_width,
                color=COLORS_TRAINER[layer],
                edgecolor='black',
                linewidth=0.0,   # nessun bordo visibile
                label=f"{layer} (Tr)" if layer == layers[0] else ""  # evita duplicati in legenda
            )
            # Tester bars (con bordo colorato)
            values_tester = [data[approach].get((layer, "tester"), 0) for approach in approaches]
            ax.bar(
                x + offsets[(layer, "tester")],
                values_tester,
                width=bar_width,
                color=COLORS_TESTER[layer],
                edgecolor=COLORS_TRAINER[layer],
                linewidth=1.0,
                label=f"{layer} (Te)" if layer == layers[0] else ""
            )

        ax.set_ylim(0.5, 1.0)   # come nel codice originale, se vuoi 0-1 cambia
        ax.set_ylabel(args.metric.capitalize(), fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        # Nomi dei metodi con CCA/CKA maiuscoli
        xtick_labels = [method_to_display_name(m) for m in approaches]
        ax.set_xticklabels(xtick_labels, rotation=0, ha='center', fontsize=12, fontweight='bold')
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')

        # Legenda come nel codice originale
        legend_handles = []
        for layer in layers:
            legend_handles.append(Patch(facecolor=COLORS_TRAINER[layer], label=f"{layer} (Tr)"))
            legend_handles.append(Patch(facecolor=COLORS_TESTER[layer],
                                        edgecolor=COLORS_TRAINER[layer],
                                        linewidth=1,
                                        label=f"{layer} (Te)"))
        ax.legend(
            handles=legend_handles,
            title="Layer (Model)",
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=True,
            prop={'size': 10, 'weight': 'bold'},
            title_fontproperties={'weight': 'bold', 'size': 14}
        )

        # Titolo con dataset in CamelCase
        ax.set_title(dataset_to_camel_case(dataset_name), fontsize=20, fontweight='bold')

        fig.tight_layout()
        out_path = args.out_dir / f"{safe_name(exp_name)}_{args.metric}_mean.pdf"
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
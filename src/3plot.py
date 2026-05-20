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
    "ridge_regressor",
    "hybrid",
    "full_nonlinear",
    "reduced_nonlinear",
    "one_for_all",
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
    if method == "full_nonlinear": return "Full_NonLinear"
    if method == "reduced_nonlinear": return "Reduced_NonLinear"
    if method == "one_for_all": return "OneForAll"
    return method.title()

def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=repo_root / "results" / "experiments" / "experiments.csv")
    parser.add_argument("--out-dir", type=Path, default=repo_root / "src" / "plots" / "3classic_means")
    parser.add_argument("--metric", type=str, default="accuracy",
                        choices=["accuracy", "precision", "recall", "f1", "auroc"])
    return parser.parse_args()

def build_long_dataframe(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    pattern = re.compile(rf"^(?P<method>.+)_(?P<split>trainer|tester)_{re.escape(metric)}$")
    metric_cols = []
    meta = {}
    for col in df.columns:
        m = pattern.match(col)
        if m:
            metric_cols.append(col)
            meta[col] = (m.group("method"), m.group("split"))
    if not metric_cols:
        raise ValueError(f"No columns found for metric '{metric}'.")
    id_cols = ["experiment", "seed", "layer_type", "dataset"]
    long_df = df[id_cols + metric_cols].melt(
        id_vars=id_cols, value_vars=metric_cols, var_name="metric_col", value_name="score"
    )
    long_df["method"] = long_df["metric_col"].map(lambda c: meta[c][0])
    long_df["split"] = long_df["metric_col"].map(lambda c: meta[c][1])
    long_df = long_df.drop(columns=["metric_col"])
    long_df["score"] = pd.to_numeric(long_df["score"], errors="coerce")
    long_df = long_df.dropna(subset=["experiment", "seed", "layer_type", "method", "split", "score"])
    long_df = long_df[long_df["method"].isin(METHOD_ORDER)]
    long_df = long_df[long_df["layer_type"].isin(LAYER_ORDER)]
    long_df = long_df[long_df["split"].isin(SPLIT_ORDER)]
    return long_df

def main():
    args = parse_args()
    df = pd.read_csv(args.csv)
    long_df = build_long_dataframe(df, args.metric)

    # --- CORREZIONE CHIAVE ---
    # Se 'experiment' contiene stringhe come 'QwenToLlama_BBF', estraiamo solo la parte del modello ('QwenToLlama')
    # in modo da poter raggruppare i diversi dataset sotto la stessa combinazione di modelli.
    long_df["model_pair"] = long_df["experiment"].apply(lambda x: x.split('_')[0] if isinstance(x, str) else x)

    mean_df = (
        long_df.groupby(["model_pair", "dataset", "method", "layer_type", "split"], as_index=False)["score"]
        .mean()
        .sort_values(["model_pair", "dataset", "method", "layer_type", "split"])
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    # Adesso iteriamo sulle coppie di modelli pulite (GemmaToLlama, QwenToLlama, FalconToGemma...)
    model_pairs = sorted(mean_df["model_pair"].unique())

    bar_width = 0.12
    offsets = {
        ("attn", "trainer"): -2.5 * bar_width,
        ("attn", "tester"): -1.5 * bar_width,
        ("hidden", "trainer"): -0.5 * bar_width,
        ("hidden", "tester"): 0.5 * bar_width,
        ("mlp", "trainer"): 1.5 * bar_width,
        ("mlp", "tester"): 2.5 * bar_width,
    }

    for pair in model_pairs:
        pair_data = mean_df[mean_df["model_pair"] == pair]
        datasets = sorted(pair_data["dataset"].unique())
        
        if not datasets:
            continue

        print(f"Generating plot for: {pair} | Datasets: {datasets}")

        # Griglia dinamica (se trova i 3 dataset, crea correttamente le 3 colonne affiancate)
        fig, axes = plt.subplots(1, len(datasets), figsize=(5.5 * len(datasets), 5.5), sharey=True)
        if len(datasets) == 1:
            axes = [axes]

        for idx, ds in enumerate(datasets):
            ax = axes[idx]
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
                        data[method][(layer, split)] = vals[0] if len(vals) > 0 else 0.0

            x = np.arange(len(METHOD_ORDER))

            for layer in LAYER_ORDER:
                tr_vals = [data[m].get((layer, "trainer"), 0) for m in METHOD_ORDER]
                ax.bar(x + offsets[(layer, "trainer")], tr_vals, width=bar_width,
                       color=COLORS_TRAINER[layer], edgecolor='black', linewidth=0.0)
                te_vals = [data[m].get((layer, "tester"), 0) for m in METHOD_ORDER]
                ax.bar(x + offsets[(layer, "tester")], te_vals, width=bar_width,
                       color=COLORS_TESTER[layer], edgecolor=COLORS_TRAINER[layer], linewidth=1.0)

            ax.set_ylim(0.3, 1.0)
            ax.set_xticks(x)
            
            # Formattazione asse X
            display_labels = []
            for m in METHOD_ORDER:
                name = method_to_display_name(m).replace("_", "")
                if name == "RidgeRegressor": name = "Ridge"
                if name == "FullNonLinear": name = "Full"
                if name == "ReducedNonLinear": name = "Reduced"
                display_labels.append(name)

            ax.set_xticklabels(display_labels, rotation=15, ha='right', fontsize=10, fontweight='bold')
            
            for label in ax.get_yticklabels():
                label.set_fontweight('bold')
                
            ax.set_title(dataset_to_camel_case(ds), fontsize=14, fontweight='bold')

        axes[0].set_ylabel(args.metric.capitalize(), fontsize=12, fontweight='bold')

        # Struttura Legenda Superiore Orizzontale
        legend_handles = []
        for layer in LAYER_ORDER:
            legend_handles.append(Patch(facecolor=COLORS_TRAINER[layer], label=f"{layer} (Tr)"))
            legend_handles.append(Patch(facecolor=COLORS_TESTER[layer],
                                        edgecolor=COLORS_TRAINER[layer],
                                        linewidth=1, label=f"{layer} (Te)"))
        
        # Posizionata stabilmente sopra i subplot
        fig.legend(handles=legend_handles, title="Layer (Model)",
                   loc="lower center", bbox_to_anchor=(0.5, 0.82),
                   ncol=6, frameon=True, prop={'size': 10, 'weight': 'bold'},
                   title_fontproperties={'weight': 'bold', 'size': 11})

        # Titolo principale con il nome del modello pulito
        #fig.suptitle(pair, fontsize=16, fontweight='bold', y=0.96)
        
        # Spazio controllato per evitare che la legenda vada sopra i titoli dei subplot
        fig.tight_layout(rect=[0, 0, 1, 0.82])

        out_path = args.out_dir / f"{re.sub(r'[^a-zA-Z0-9._-]', '_', pair)}_{args.metric}_mean_subplots.pdf"
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()
        print(f"Salvato con successo: {out_path}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
error_analysis.py

Generalised error analysis for One4All experiments.

Loads all prediction JSON files under results/one4all_experiment_details
and produces the full suite of diagnostic plots.

Plot files land in src/plots/error_analysis/*.pdf (same as the notebook).

Usage
-----
    python error_analysis.py                  # all experiments, skip semantic
    python error_analysis.py --semantic       # include heavy UMAP/t-SNE plots
    python error_analysis.py --exp GemmaToLlama_HE --semantic
    python error_analysis.py --reduce tsne    # use t-SNE instead of UMAP
"""

import argparse
import json
import re
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix

warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams["figure.dpi"] = 110

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR     = PROJECT_ROOT / "results" / "one4all_experiment_details"
PLOTS_DIR    = PROJECT_ROOT / "src" / "plots" / "error_analysis"

# ── Aesthetics ─────────────────────────────────────────────────────────────────
ET_ORDER  = ["TP", "TN", "FP", "FN"]
PALETTE   = {"TP": "#4CAF50", "TN": "#2196F3", "FP": "#FF5722", "FN": "#FF9800"}
ET_COLORS = {"TP": "#27ae60", "TN": "#2980b9", "FP": "#e74c3c", "FN": "#e67e22"}
ET_LABELS = {"TP": "True Positive", "TN": "True Negative",
             "FP": "False Positive", "FN": "False Negative"}
LAYER_LABELS = {"attn": "Attention", "hidden": "Hidden", "mlp": "MLP"}


# ── Data loading ───────────────────────────────────────────────────────────────

def load_predictions(base_dir: Path) -> pd.DataFrame:
    """Load all prediction JSON files into a single flat DataFrame."""
    if not base_dir.exists():
        raise FileNotFoundError(f"Base dir not found: {base_dir}")
    rows = []
    for fname in base_dir.rglob("*_test_predictions.json"):
        with open(fname, encoding="utf-8") as f:
            payload = json.load(f)
        model_name = payload.get("model_name", "unknown")
        layer_type = payload.get("layer_type")
        if not layer_type:
            layer_dir = fname.parent.name
            layer_type = layer_dir.replace("layer_", "")
        rel_parts = fname.relative_to(base_dir).parts
        experiment = rel_parts[0] if rel_parts else "unknown"
        seed = rel_parts[1] if len(rel_parts) > 1 else ""
        dataset = payload.get("dataset", "unknown")
        for rec in payload["records"]:
            instance = rec["Istanza Dataset"]
            question = instance["question"]
            if "[Further Knowledge]" in question:
                q_text, ctx = question.split("[Further Knowledge]", 1)
            else:
                q_text, ctx = question, ""
            rows.append({
                "experiment":     experiment,
                "seed":           seed,
                "dataset":        dataset,
                "model":          model_name,
                "layer":          layer_type,
                "instance_id":    instance["instance_id"],
                "question":       q_text.strip(),
                "context":        ctx.strip(),
                "llm_answer":     rec["Risposta data dall LLM"].strip(),
                "true_answer":    rec["Risposta vera"].strip(),
                "pred":           int(rec["Predizione One4All"]),
                "label":          int(rec["GroundTruth per One4All"]),
                "label_balanced": int(rec["GroundTruth split bilanciato"]),
                "prob":           float(rec["Probabilita One4All"]),
            })
    return pd.DataFrame(rows)


# ── Feature engineering ────────────────────────────────────────────────────────

def _word_count(text: str) -> int:
    return len(text.split())

def _sentence_count(text: str) -> int:
    return max(1, len(re.split(r"[.!?]+", text)))

def _has_numbers(text: str) -> bool:
    return bool(re.search(r"\d", text))

def _cap_ratio(text: str) -> float:
    words = text.split()
    if not words:
        return 0.0
    return sum(1 for w in words if w and w[0].isupper()) / len(words)

def _avg_word_len(text: str) -> float:
    words = re.findall(r"\b\w+\b", text)
    return sum(len(w) for w in words) / len(words) if words else 0.0

def _question_type(q: str) -> str:
    q_lower = q.lower().strip()
    for kw in ["who", "what", "when", "where", "which", "how", "why",
               "is ", "are ", "was ", "were ", "did ", "does ", "do ",
               "can ", "has ", "have "]:
        if q_lower.startswith(kw):
            return kw.strip()
    return "other"

def _overlap_ratio(llm: str, truth: str) -> float:
    truth_words = set(re.findall(r"\b\w+\b", truth.lower()))
    llm_words   = set(re.findall(r"\b\w+\b", llm.lower()))
    if not truth_words:
        return 1.0
    return len(truth_words & llm_words) / len(truth_words)

def _error_type(row) -> str:
    p, l = row["pred"], row["label"]
    if   p == 1 and l == 1: return "TP"
    elif p == 0 and l == 0: return "TN"
    elif p == 1 and l == 0: return "FP"
    else:                   return "FN"

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["q_word_count"]    = df["question"].apply(_word_count)
    df["ctx_word_count"]  = df["context"].apply(_word_count)
    df["llm_word_count"]  = df["llm_answer"].apply(_word_count)
    df["tru_word_count"]  = df["true_answer"].apply(_word_count)
    df["llm_sent_count"]  = df["llm_answer"].apply(_sentence_count)
    df["llm_has_numbers"] = df["llm_answer"].apply(_has_numbers)
    df["llm_cap_ratio"]   = df["llm_answer"].apply(_cap_ratio)
    df["llm_avg_wlen"]    = df["llm_answer"].apply(_avg_word_len)
    df["q_type"]          = df["question"].apply(_question_type)
    df["overlap_ratio"]   = df.apply(lambda r: _overlap_ratio(r["llm_answer"], r["true_answer"]), axis=1)
    df["error_type"]      = df.apply(_error_type, axis=1)
    df["correct"]         = (df["pred"] == df["label"]).astype(int)
    return df


# ── Plot helpers ───────────────────────────────────────────────────────────────

def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]", "_", text)

def _savefig(fig, plots_dir: Path, name: str) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    path = plots_dir / f"{name}.pdf"
    fig.savefig(path, bbox_inches="tight")
    print(f"  saved → {path}")
    plt.close(fig)


# ── Plot functions ─────────────────────────────────────────────────────────────

def plot_error_type_breakdown(df: pd.DataFrame, models: list, layers: list,
                               plots_dir: Path) -> None:
    et_counts = (
        df.groupby(["model", "layer", "error_type"])
        .size()
        .reset_index(name="count")
    )
    fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 5), sharey=False)
    if len(models) == 1:
        axes = [axes]
    for ax, model in zip(axes, models):
        sub = et_counts[et_counts["model"] == model].pivot(
            index="layer", columns="error_type", values="count"
        ).fillna(0)
        sub.plot(kind="bar", ax=ax,
                 color=[PALETTE.get(c, "grey") for c in sub.columns],
                 stacked=False, rot=0, width=0.7)
        ax.set_title(model)
        ax.set_ylabel("Count")
        ax.legend(title="Error type")
    plt.tight_layout()
    _savefig(fig, plots_dir, "error_type_breakdown")


def plot_confidence_histograms(df: pd.DataFrame, models: list,
                                plots_dir: Path) -> None:
    for model in models:
        fig, ax = plt.subplots(figsize=(4.5, 3))
        sub = df[df["model"] == model]
        for et, color in PALETTE.items():
            vals = sub.loc[sub["error_type"] == et, "prob"]
            ax.hist(vals, bins=30, alpha=0.55,
                    label=f"{et} (n={len(vals)})", color=color, density=True)
        ax.axvline(0.5, color="black", linestyle="--", linewidth=1)
        ax.set_title(model, fontsize=10)
        ax.set_xlabel("P(hallucination)", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7)
        plt.tight_layout(pad=0.4)
        _savefig(fig, plots_dir, f"confidence_histogram__{_slug(model)}")


def plot_confidence_boxstrip(df: pd.DataFrame, models: list,
                              plots_dir: Path) -> None:
    fig, axes = plt.subplots(1, len(models), figsize=(4.5 * len(models), 3), sharey=True)
    if len(models) == 1:
        axes = [axes]
    for ax, model in zip(axes, models):
        sub = df[df["model"] == model]
        sns.boxplot(data=sub, x="error_type", y="prob", order=ET_ORDER,
                    palette=PALETTE, ax=ax, showfliers=False)
        sns.stripplot(data=sub.sample(min(600, len(sub)), random_state=42),
                      x="error_type", y="prob", order=ET_ORDER,
                      color="black", alpha=0.2, size=2, ax=ax, jitter=True)
        ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8)
        ax.set_title(model, fontsize=10)
        ax.set_xlabel("Error type", fontsize=9)
        ax.set_ylabel("P(hallucination)" if ax is axes[0] else "", fontsize=9)
        ax.tick_params(labelsize=8)
    plt.tight_layout(pad=0.4)
    _savefig(fig, plots_dir, "confidence_boxstrip")


def plot_calibration_curves(df: pd.DataFrame, models: list, layers: list,
                             plots_dir: Path) -> None:
    for model in models:
        fig, ax = plt.subplots(figsize=(4, 3))
        for layer in layers:
            sub = df[(df["model"] == model) & (df["layer"] == layer)]
            if len(sub) < 10:
                continue
            try:
                frac_pos, mean_pred = calibration_curve(sub["label"], sub["prob"], n_bins=10)
                ax.plot(mean_pred, frac_pos, marker="o",
                        label=LAYER_LABELS.get(layer, layer))
            except Exception:
                pass
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Perfect")
        ax.set_title(model, fontsize=10)
        ax.set_xlabel("Mean predicted probability", fontsize=9)
        ax.set_ylabel("Fraction of positives", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=8)
        plt.tight_layout(pad=0.4)
        _savefig(fig, plots_dir, f"calibration_curves__{_slug(model)}")


def plot_feature_distributions(df: pd.DataFrame, plots_dir: Path) -> None:
    features = [
        ("llm_word_count", "LLM answer word count"),
        ("overlap_ratio",  "Answer overlap ratio"),
        ("q_word_count",   "Question word count"),
        ("ctx_word_count", "Context word count"),
    ]
    for feat, label in features:
        fig, ax = plt.subplots(figsize=(5, 3))
        for et in ET_ORDER:
            vals = df.loc[df["error_type"] == et, feat].dropna()
            if len(vals) == 0:
                continue
            cap = np.percentile(vals, 99)
            vals = vals.clip(upper=cap)
            ax.hist(vals, bins=40, alpha=0.45, density=True,
                    label=et, color=ET_COLORS[et])
        ax.set_xlabel(label, fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=8)
        plt.tight_layout(pad=0.4)
        _savefig(fig, plots_dir, f"feature_dist__{feat}")


def plot_question_type_heatmap(df: pd.DataFrame, plots_dir: Path) -> None:
    qt_counts = (
        df.groupby(["q_type", "error_type"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=ET_ORDER, fill_value=0)
    )
    qt_pct = qt_counts.div(qt_counts.sum(axis=1), axis=0) * 100
    qt_pct = qt_pct.sort_values("FN", ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(6, max(3, len(qt_pct) * 0.35)))
    sns.heatmap(qt_pct, annot=True, fmt=".1f", cmap="YlOrRd",
                linewidths=0.4, ax=ax, cbar_kws={"label": "% of row"})
    ax.set_xlabel("Error type", fontsize=9)
    ax.set_ylabel("Question type", fontsize=9)
    ax.tick_params(labelsize=8)
    plt.tight_layout(pad=0.4)
    _savefig(fig, plots_dir, "question_type_heatmap")


def plot_confusion_matrices(df: pd.DataFrame, models: list, layers: list,
                             plots_dir: Path) -> None:
    n_cols = len(layers)
    for model in models:
        fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 3.5))
        if n_cols == 1:
            axes = [axes]
        for ax, layer in zip(axes, layers):
            sub = df[(df["model"] == model) & (df["layer"] == layer)]
            if len(sub) == 0:
                ax.axis("off")
                continue
            cm = confusion_matrix(sub["label"], sub["pred"])
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                        xticklabels=["Pred 0", "Pred 1"],
                        yticklabels=["True 0", "True 1"])
            ax.set_title(f"{LAYER_LABELS.get(layer, layer)}", fontsize=9)
            ax.tick_params(labelsize=8)
        fig.suptitle(model, fontsize=10, y=1.01)
        plt.tight_layout(pad=0.4)
        _savefig(fig, plots_dir, f"confusion_matrix__{_slug(model)}")


# ── Semantic (optional, heavy) ─────────────────────────────────────────────────

def plot_semantic_2d(df: pd.DataFrame, models: list,
                     reduce_method: str, layer_for_plot: str,
                     plots_dir: Path) -> None:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("  [skip] sentence-transformers not installed — skipping semantic plots")
        return

    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    unique_q = (
        df.drop_duplicates("instance_id")
          [["instance_id", "question", "context"]]
          .reset_index(drop=True)
    )
    unique_q["question_ctx"] = unique_q["question"] + " [SEP] " + unique_q["context"]

    print(f"  encoding {len(unique_q)} unique instances …")
    emb_q  = encoder.encode(unique_q["question"].tolist(),     batch_size=64, show_progress_bar=True)
    emb_qc = encoder.encode(unique_q["question_ctx"].tolist(), batch_size=64, show_progress_bar=True)

    def _reduce(embeddings):
        if reduce_method == "umap":
            import umap as umap_lib
            r = umap_lib.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        else:
            from sklearn.manifold import TSNE
            r = TSNE(n_components=2, random_state=42, perplexity=40, n_iter=1000, verbose=0)
        return r.fit_transform(embeddings)

    print(f"  reducing with {reduce_method.upper()} …")
    coords_q  = _reduce(emb_q)
    coords_qc = _reduce(emb_qc)
    unique_q[["x_q",  "y_q"]]  = coords_q
    unique_q[["x_qc", "y_qc"]] = coords_qc

    plot_df = (
        df[df["layer"] == layer_for_plot][["model", "instance_id", "error_type", "prob"]]
          .merge(unique_q[["instance_id", "x_q", "y_q", "x_qc", "y_qc"]],
                 on="instance_id", how="left")
          .dropna(subset=["x_q"])
    )
    MAX_PTS = 1000
    per_type_cap = MAX_PTS // len(ET_ORDER)  # equal budget per error type
    balanced = []
    for _, model_group in plot_df.groupby("model"):
        et_sizes = model_group.groupby("error_type").size()
        n_sample = min(per_type_cap, et_sizes.min())
        for _, et_group in model_group.groupby("error_type"):
            balanced.append(et_group.sample(n=n_sample, random_state=42))
    plot_df = pd.concat(balanced).reset_index(drop=True)

    for model in models:
        sub = plot_df[plot_df["model"] == model]
        m_slug = _slug(model)

        # ── small multiples ──
        fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
        all_x, all_y = sub["x_q"], sub["y_q"]
        pad = 0.5
        for col_i, et in enumerate(ET_ORDER):
            ax   = axes[col_i]
            mask = sub["error_type"] == et
            bg, fg = sub[~mask], sub[mask]
            ax.scatter(bg["x_q"], bg["y_q"],
                       color="#cecece", alpha=0.25, s=5, linewidths=0, zorder=1)
            if fg.shape[0] > 30:
                try:
                    sns.kdeplot(x=fg["x_q"], y=fg["y_q"], ax=ax,
                                fill=True,  color=ET_COLORS[et], alpha=0.22,
                                levels=6, thresh=0.05, zorder=2)
                    sns.kdeplot(x=fg["x_q"], y=fg["y_q"], ax=ax,
                                fill=False, color=ET_COLORS[et], alpha=0.9,
                                levels=6, thresh=0.05, linewidths=1.0, zorder=3)
                except Exception:
                    pass
            ax.scatter(fg["x_q"], fg["y_q"],
                       color=ET_COLORS[et], alpha=0.55, s=12, linewidths=0, zorder=4)
            if not fg.empty:
                cx, cy = fg["x_q"].mean(), fg["y_q"].mean()
                ax.plot(cx, cy, marker="*", markersize=12, zorder=5,
                        color="white", markeredgecolor=ET_COLORS[et], markeredgewidth=1.5)
            ax.set_title(ET_LABELS[et], fontsize=9, color=ET_COLORS[et],
                         fontweight="bold", pad=4)
            ax.set_xlabel(f"{reduce_method.upper()} 1", fontsize=8)
            ax.set_ylabel(f"{model}\n{reduce_method.upper()} 2"
                          if col_i == 0 else "", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.set_xlim(all_x.min() - pad, all_x.max() + pad)
            ax.set_ylim(all_y.min() - pad, all_y.max() + pad)
        plt.tight_layout(pad=0.4)
        _savefig(fig, plots_dir, f"semantic_small_multiples__{m_slug}")

        # ── all types together ──
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        ax.scatter(sub["x_q"], sub["y_q"],
                   color="#e0e0e0", alpha=0.3, s=8, linewidths=0, zorder=1)
        for et in ["TN", "TP", "FN", "FP"]:
            fg = sub[sub["error_type"] == et]
            if fg.shape[0] > 30:
                try:
                    sns.kdeplot(x=fg["x_q"], y=fg["y_q"], ax=ax,
                                fill=True,  color=ET_COLORS[et], alpha=0.18,
                                levels=5, thresh=0.1, zorder=2)
                    sns.kdeplot(x=fg["x_q"], y=fg["y_q"], ax=ax,
                                fill=False, color=ET_COLORS[et], alpha=0.75,
                                levels=5, thresh=0.1, linewidths=1.0, zorder=3)
                except Exception:
                    pass
            ax.scatter(fg["x_q"], fg["y_q"],
                       color=ET_COLORS[et], alpha=0.4, s=10, linewidths=0,
                       label=f"{et}  (n={fg.shape[0]})", zorder=4)
            if not fg.empty:
                cx, cy = fg["x_q"].mean(), fg["y_q"].mean()
                ax.annotate(et, xy=(cx, cy), fontsize=9, fontweight="bold",
                            color="white", ha="center", va="center", zorder=6,
                            bbox=dict(boxstyle="round,pad=0.3",
                                      facecolor=ET_COLORS[et],
                                      edgecolor="white", linewidth=1.4, alpha=0.92))
        ax.set_title(model, fontsize=10, fontweight="bold")
        ax.set_xlabel(f"{reduce_method.upper()} dim 1", fontsize=9)
        ax.set_ylabel(f"{reduce_method.upper()} dim 2", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.legend(markerscale=2, fontsize=8, loc="upper right",
                  framealpha=0.92, edgecolor="#aaa")
        plt.tight_layout(pad=0.4)
        _savefig(fig, plots_dir, f"semantic_all_types__{m_slug}")

        # ── confidence overlay ──
        fig, ax = plt.subplots(figsize=(5, 4))
        wrong = sub[sub["error_type"].isin(["FP", "FN"])]
        sc = ax.scatter(sub["x_q"], sub["y_q"],
                        c=sub["prob"], cmap="RdYlGn_r",
                        alpha=0.65, s=14, linewidths=0,
                        vmin=0, vmax=1, zorder=2)
        ax.scatter(wrong["x_q"], wrong["y_q"],
                   facecolors="none", edgecolors="black",
                   s=45, linewidths=0.8, alpha=0.65,
                   label=f"Wrong  (n={len(wrong)})", zorder=3)
        for et in ET_ORDER:
            fg = sub[sub["error_type"] == et]
            if not fg.empty:
                cx, cy = fg["x_q"].mean(), fg["y_q"].mean()
                ax.annotate(et, xy=(cx, cy), fontsize=8, fontweight="bold",
                            color="white", zorder=5, ha="center", va="center",
                            bbox=dict(boxstyle="round,pad=0.22",
                                      facecolor=ET_COLORS[et],
                                      edgecolor="white", linewidth=1.0, alpha=0.88))
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("P(hallucination)", fontsize=8)
        cbar.ax.tick_params(labelsize=7)
        ax.set_title(f"{model}  |  layer = {layer_for_plot}", fontsize=10)
        ax.set_xlabel(f"{reduce_method.upper()} dim 1", fontsize=9)
        ax.set_ylabel(f"{reduce_method.upper()} dim 2", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
        plt.tight_layout(pad=0.4)
        _savefig(fig, plots_dir, f"semantic_confidence__{m_slug}")


# ── Main ───────────────────────────────────────────────────────────────────────

def run_analysis(df: pd.DataFrame, args: argparse.Namespace) -> None:
    print(f"Plots dir  : {PLOTS_DIR}")
    print("Engineering features …")
    df = add_features(df)

    models = sorted(df["model"].unique())
    layers = sorted(df["layer"].unique())
    print(f"  models: {models}")
    print(f"  layers: {layers}")

    print("Plotting …")
    plot_error_type_breakdown(df, models, layers, PLOTS_DIR)
    plot_confidence_histograms(df, models, PLOTS_DIR)
    plot_confidence_boxstrip(df, models, PLOTS_DIR)
    plot_calibration_curves(df, models, layers, PLOTS_DIR)
    plot_feature_distributions(df, PLOTS_DIR)
    plot_question_type_heatmap(df, PLOTS_DIR)
    plot_confusion_matrices(df, models, layers, PLOTS_DIR)

    if args.semantic:
        layer_for_plot = args.layer if args.layer in layers else layers[0]
        print(f"Semantic analysis (layer={layer_for_plot}, reduce={args.reduce}) …")
        plot_semantic_2d(df, models, args.reduce, layer_for_plot, PLOTS_DIR)


def main():
    parser = argparse.ArgumentParser(description="One4All error analysis")
    parser.add_argument("--exp",      default=None,
                        help="Run only this experiment folder name (default: all)")
    parser.add_argument("--semantic", action="store_true",
                        help="Include UMAP/t-SNE semantic embedding plots (slow)")
    parser.add_argument("--reduce",   default="umap", choices=["umap", "tsne"],
                        help="Dimensionality reduction method for semantic plots")
    parser.add_argument("--layer",    default="attn",
                        help="Layer whose predictions colour the semantic plots")
    args = parser.parse_args()

    print(f"Base dir  : {BASE_DIR}")
    print("Loading predictions …")
    df = load_predictions(BASE_DIR)
    if df.empty:
        raise SystemExit("No prediction files found.")
    print(f"  {len(df):,} records, {df['model'].nunique()} models, {df['layer'].nunique()} layers")

    if args.exp:
        df = df[df["experiment"].str.contains(args.exp, na=False)]
        if df.empty:
            raise SystemExit(f"No experiment matching '{args.exp}' found under {BASE_DIR}")

    run_analysis(df, args)
    print("\nDone. All plots saved under:", PLOTS_DIR.resolve())


if __name__ == "__main__":
    main()

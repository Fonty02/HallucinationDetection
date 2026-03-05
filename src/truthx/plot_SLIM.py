import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATASET_LABELS = {
    "belief_bank_facts": "BBF",
    "belief_bank_constraints": "BBC",
    "halu_eval": "HE",
}
DATASET_ORDER = ["belief_bank_facts", "belief_bank_constraints", "halu_eval"]
ABBR_TO_DATASET = {abbr: name for name, abbr in DATASET_LABELS.items()}

MODEL_LABELS = {
    "google/gemma-2-9b-it": "Gemma",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama",
}

CATEGORY_ORDER = [
    "baseline",
    "slim_standard",
    "cross_dataset_from_BBF",
    "cross_dataset_from_BBC",
    "cross_dataset_from_HE",
    "cross_model_legacy",
    "cross_model_v2",
    "cross_model_mlp",
    "cross_model_mlp2",
    "cross_model_procrustes",
]

CATEGORY_LABELS = {
    "baseline": "baseline",
    "slim_standard": "SLiM",
    "cross_dataset_from_BBF": "cross_dataset_from_BBF",
    "cross_dataset_from_BBC": "cross_dataset_from_BBC",
    "cross_dataset_from_HE": "cross_dataset_from_HE",
    "cross_model_legacy": "cross_model_legacy",
    "cross_model_v2": "cross_model_v2",
    "cross_model_mlp": "cross_model_mlp",
    "cross_model_mlp2": "cross_model_mlp2",
    "cross_model_procrustes": "cross_model_procrustes",
}

BASE_COLORS = {
    "baseline": "#FF6B6B",
    "slim_standard": "#4169E1",
    "cross_dataset_from_BBF": "#90BE6D",
    "cross_dataset_from_BBC": "#43AA8B",
    "cross_dataset_from_HE": "#4D908E",
    "cross_model_legacy": "#FFB347",
    "cross_model_v2": "#FFB6C1",
    "cross_model_mlp": "#FFD166",
    "cross_model_mlp2": "#F4A261",
    "cross_model_procrustes": "#9B5DE5",
}


def find_csv_path() -> Path:
    candidates = [
        Path("SLiMExperiments.csv"),
        Path(__file__).resolve().parents[2] / "SLiMExperiments.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("SLiMExperiments.csv non trovato.")


def source_dataset_from_row(row) -> str | None:
    dataset_train = str(row.get("dataset_train", "") or "").strip()
    if dataset_train in DATASET_LABELS:
        return dataset_train

    slim_path = str(row.get("slim_checkpoint", "") or "")
    match = re.search(r"SteeringVectors/SLiM/[^/]+/([^/]+)/", slim_path)
    if match:
        candidate = match.group(1)
        if candidate in DATASET_LABELS:
            return candidate

    exp_id = str(row.get("experiment_id", "") or "").upper()
    match = re.search(r"_(BBF|BBC|HE)_TO_", exp_id)
    if match:
        return ABBR_TO_DATASET.get(match.group(1))
    return None


def infer_cross_model_variant(row) -> str:
    exp_id = str(row.get("experiment_id", "") or "").upper()
    exp_type = str(row.get("type", "") or "").strip().lower()

    if "PROCRUSTES" in exp_id:
        return "cross_model_procrustes"
    if "MLP2" in exp_id:
        return "cross_model_mlp2"
    if "_MLP" in exp_id:
        return "cross_model_mlp"
    if exp_type == "cross_model2" or exp_id.endswith("_2"):
        return "cross_model_v2"
    return "cross_model_legacy"


def normalize_experiment(row) -> tuple[str, str]:
    exp_type = str(row.get("type", "") or "").strip().lower()

    if exp_type == "baseline":
        return "baseline", CATEGORY_LABELS["baseline"]

    if exp_type == "slim":
        return "slim_standard", CATEGORY_LABELS["slim_standard"]

    if exp_type in {"crossdataset", "cross-dataset"}:
        source_dataset = source_dataset_from_row(row)
        if source_dataset is None:
            key = "cross_dataset_from_unknown"
            return key, key
        key = f"cross_dataset_from_{DATASET_LABELS[source_dataset]}"
        return key, CATEGORY_LABELS.get(key, key)

    if exp_type in {"crossmodel", "cross_model", "cross_model1", "cross_model2"}:
        key = infer_cross_model_variant(row)
        return key, CATEGORY_LABELS.get(key, key)

    fallback_key = f"other_{exp_type or 'unknown'}"
    fallback_key = re.sub(r"[^a-z0-9_]+", "_", fallback_key)
    return fallback_key, fallback_key


def prepare_data_for_model(df: pd.DataFrame, model_name: str):
    model_df = df[df["model"] == model_name].copy()
    plot_data = {dataset: {} for dataset in DATASET_ORDER}

    for _, row in model_df.iterrows():
        dataset = row.get("dataset_eval")
        if dataset not in plot_data:
            continue

        key, label = normalize_experiment(row)
        entry = {
            "rate": float(row["hallucination_rate"]),
            "id": str(row["experiment_id"]),
            "label": label,
        }
        current = plot_data[dataset].get(key)
        # Se ci sono duplicati nella stessa categoria, mantiene il migliore.
        if current is None or entry["rate"] < current["rate"]:
            plot_data[dataset][key] = entry

    return plot_data


def ordered_categories(plot_data: dict) -> list[str]:
    found = {key for ds_data in plot_data.values() for key in ds_data.keys()}
    ordered = [key for key in CATEGORY_ORDER if key in found]
    remaining = sorted(found - set(CATEGORY_ORDER))
    return ordered + remaining


def build_color_map(categories: list[str]) -> dict[str, str]:
    color_map = {}
    for category in categories:
        if category in BASE_COLORS:
            color_map[category] = BASE_COLORS[category]

    if len(color_map) == len(categories):
        return color_map

    cmap = plt.get_cmap("tab20")
    idx = 0
    for category in categories:
        if category not in color_map:
            color_map[category] = cmap(idx % cmap.N)
            idx += 1
    return color_map


def plot_model_results(plot_data: dict, model_label: str):
    categories = ordered_categories(plot_data)
    if not categories:
        raise ValueError(f"Nessun esperimento trovato per {model_label}.")

    fig_width = max(14, 8 + len(categories) * 0.9)
    fig, ax = plt.subplots(figsize=(fig_width, 8))

    x_pos = np.arange(len(DATASET_ORDER))
    width = min(0.82 / max(len(categories), 1), 0.16)
    center_offset = (len(categories) - 1) / 2
    color_map = build_color_map(categories)

    max_rate = 0.0
    for ds in DATASET_ORDER:
        for exp in plot_data[ds].values():
            max_rate = max(max_rate, exp["rate"])
    if max_rate == 0:
        max_rate = 1.0

    for i, category in enumerate(categories):
        positions = []
        values = []
        for j, dataset in enumerate(DATASET_ORDER):
            exp = plot_data[dataset].get(category)
            if exp is None:
                continue
            positions.append(x_pos[j] + (i - center_offset) * width)
            values.append(exp["rate"])

        if not values:
            continue

        bars = ax.bar(
            positions,
            values,
            width,
            label=CATEGORY_LABELS.get(category, category),
            color=color_map[category],
            edgecolor="black",
            linewidth=0.5,
        )

        for bar, rate in zip(bars, values):
            x = bar.get_x() + bar.get_width() / 2
            y = bar.get_height()
            ax.text(
                x,
                y + max_rate * 0.01,
                f"{rate * 100:.2f}%",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax.set_ylabel("Hallucination Rate", fontsize=12)
    ax.set_title(f"{model_label} - SLiM", fontsize=20, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        [DATASET_LABELS[ds] for ds in DATASET_ORDER],
        fontsize=15,
        fontweight="bold",
    )
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max_rate * 1.20)

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=10,
        framealpha=0.9,
        title="Legenda",
        title_fontsize=10,
    )

    plt.tight_layout(rect=[0, 0, 0.82, 1])
    return fig


def main():
    csv_path = find_csv_path()
    df = pd.read_csv(csv_path)

    for model_name, model_label in MODEL_LABELS.items():
        model_plot_data = prepare_data_for_model(df, model_name)
        fig = plot_model_results(model_plot_data, model_label)
        output_name = f"{model_label.lower()}_slim_results.pdf"
        fig.savefig(output_name, dpi=300, bbox_inches="tight")
        print(f"Grafico {model_label} salvato come '{output_name}'")

    plt.show()


if __name__ == "__main__":
    main()

# %%
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# -----------------------------
# Configuration (change these to plot other datasets)
# -----------------------------
config = {
    "dataset": "LLama_Gemma_HE",
    "trainer_model": "gemma-2-9b-it",
    "tester_model": "Llama-3.1-8B-Instruct",
    "include_reverse": True,
}

dataset = config["dataset"]
trainer_model_name = config["trainer_model"]
tester_model_name = config["tester_model"]
base_dir = Path.cwd()

approaches = [
    "FullLinear",
    "AdapterMLP",
    "FullNonLinear",
    "ReducedNonLinear",
    "One-For-All",
]
layers = ["attn", "hidden", "mlp"]

path_templates = {
    "FullLinear": "linearApproach/{dataset}/results_metrics/linear_probe_results.json",
    "AdapterMLP": "HybridApproach/{dataset}/results_metrics/hybrid_adapter_logreg_results.json",
    "FullNonLinear": "nonLinearApproach/approach1FullDIM/{dataset}/results_metrics/approach1_mlp_prober_results.json",
    "ReducedNonLinear": "nonLinearApproach/approach2Projected/{dataset}/results_metrics/approach2_autoencoder_results.json",
    "One-For-All": "nonLinearApproach/approach3OneForAll/{dataset}/results_metrics/approach3_frozen_head_results.json",
}

paths = {
    approach: base_dir / template.format(dataset=dataset)
    for approach, template in path_templates.items()
}

scenario_pairs = [(trainer_model_name, tester_model_name)]
if config.get("include_reverse") and trainer_model_name != tester_model_name:
    scenario_pairs.append((tester_model_name, trainer_model_name))

scenario_data = {pair: {} for pair in scenario_pairs}

# -----------------------------
# Helpers
# -----------------------------
def normalize_metrics(result: dict) -> dict:
    if "metrics" in result:
        return result["metrics"]

    metrics = {}
    if "teacher" in result:
        metrics["teacher"] = result["teacher"]
    if "student_on_teacher" in result:
        metrics["student_on_teacher"] = result["student_on_teacher"]
    if "student_adapter" in result:
        metrics["student_adapter"] = result["student_adapter"]
    return metrics

for approach, path in paths.items():
    if not path.exists():
        print(f"Warning: file not found for {approach}: {path}")
        continue

    with open(path, "r", encoding="utf-8") as f:
        content = json.load(f)

    if approach == "One-For-All":
        for entry in content:
            teacher = entry.get("teacher_model", "")
            student = entry.get("student_model", "")
            pair_key = (teacher, student)
            if pair_key not in scenario_data:
                continue

            layer = entry.get("layer_type")
            if layer not in layers:
                continue

            metrics = entry.get("metrics", {})
            acc_t = metrics.get("teacher", {}).get("accuracy", 0)
            acc_s = metrics.get("student_adapter", {}).get("accuracy", 0)

            scenario_data[pair_key].setdefault(approach, {})[layer] = {
                "Trainer": acc_t,
                "Tester": acc_s,
            }
        continue

    for scenario_group in content:
        results = scenario_group.get("results", [])
        if not results:
            continue

        first_result = results[0]
        teacher = first_result.get("teacher_model") or first_result.get("teacher_name")
        student = first_result.get("student_model") or first_result.get("student_name")
        pair_key = (teacher, student)
        if pair_key not in scenario_data:
            continue

        for result in results:
            layer = result.get("layer_type") or result.get("type")
            if layer not in layers:
                continue

            metrics = normalize_metrics(result)
            trainer_metrics = metrics.get("teacher", {})
            tester_metrics = (
                metrics.get("student_on_teacher")
                or metrics.get("student_adapter")
                or {}
            )
            acc_t = trainer_metrics.get("accuracy", 0)
            acc_s = tester_metrics.get("accuracy", 0)

            scenario_data[pair_key].setdefault(approach, {})[layer] = {
                "Trainer": acc_t,
                "Tester": acc_s,
            }

# -----------------------------
# Plot parameters - UPDATED TO MATCH crossValScript.py
# -----------------------------
x = np.arange(len(approaches))
bar_width = 0.12

# Updated colors to match crossValScript.py
colors_trainer = {
    "attn": '#1f77b4',    # Blue
    "hidden": '#2ca02c',  # Green
    "mlp": '#d62728'      # Red
}
colors_tester = {
    "attn": '#aec7e8',    # Light blue
    "hidden": '#98df8a',  # Light green
    "mlp": '#ff9896'      # Light red
}

offsets = {
    ("attn", "Trainer"): -2.5 * bar_width,
    ("attn", "Tester"): -1.5 * bar_width,
    ("hidden", "Trainer"): -0.5 * bar_width,
    ("hidden", "Tester"): 0.5 * bar_width,
    ("mlp", "Trainer"): 1.5 * bar_width,
    ("mlp", "Tester"): 2.5 * bar_width,
}

title_style = {"fontsize": 10, "fontweight": "bold"}
label_style = {"fontsize": 10, "fontweight": "bold"}
tick_style = {"fontsize": 10, "fontweight": "bold"}
legend_prop = {"size": 12, "weight": "bold"}

scenarios_to_plot = [pair for pair in scenario_pairs if scenario_data[pair]]
if not scenarios_to_plot:
    raise RuntimeError("Nessun scenario valido trovato per i modelli trainer/tester configurati.")


# %%
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# -----------------------------
# Configuration (change these to plot other datasets)
# -----------------------------
config = {
    "dataset": "LLama_Gemma_HE",
    "trainer_model": "gemma-2-9b-it",
    "tester_model": "Llama-3.1-8B-Instruct",
    "include_reverse": True,
}

dataset = config["dataset"]
trainer_model_name = config["trainer_model"]
tester_model_name = config["tester_model"]
base_dir = Path.cwd()

approaches = [
    "FullLinear",
    "AdapterMLP",
    "FullNonLinear",
    "ReducedNonLinear",
    "One-For-All",
]
layers = ["attn", "hidden", "mlp"]

path_templates = {
    "FullLinear": "linearApproach/{dataset}/results_metrics/linear_probe_results.json",
    "AdapterMLP": "HybridApproach/{dataset}/results_metrics/hybrid_adapter_logreg_results.json",
    "FullNonLinear": "nonLinearApproach/approach1FullDIM/{dataset}/results_metrics/approach1_mlp_prober_results.json",
    "ReducedNonLinear": "nonLinearApproach/approach2Projected/{dataset}/results_metrics/approach2_autoencoder_results.json",
    "One-For-All": "nonLinearApproach/approach3OneForAll/{dataset}/results_metrics/approach3_frozen_head_results.json",
}

paths = {
    approach: base_dir / template.format(dataset=dataset)
    for approach, template in path_templates.items()
}

scenario_pairs = [(trainer_model_name, tester_model_name)]
if config.get("include_reverse") and trainer_model_name != tester_model_name:
    scenario_pairs.append((tester_model_name, trainer_model_name))

scenario_data = {pair: {} for pair in scenario_pairs}

# -----------------------------
# Helpers
# -----------------------------
def normalize_metrics(result: dict) -> dict:
    if "metrics" in result:
        return result["metrics"]

    metrics = {}
    if "teacher" in result:
        metrics["teacher"] = result["teacher"]
    if "student_on_teacher" in result:
        metrics["student_on_teacher"] = result["student_on_teacher"]
    if "student_adapter" in result:
        metrics["student_adapter"] = result["student_adapter"]
    return metrics

for approach, path in paths.items():
    if not path.exists():
        print(f"Warning: file not found for {approach}: {path}")
        continue

    with open(path, "r", encoding="utf-8") as f:
        content = json.load(f)

    if approach == "One-For-All":
        for entry in content:
            teacher = entry.get("teacher_model", "")
            student = entry.get("student_model", "")
            pair_key = (teacher, student)
            if pair_key not in scenario_data:
                continue

            layer = entry.get("layer_type")
            if layer not in layers:
                continue

            metrics = entry.get("metrics", {})
            acc_t = metrics.get("teacher", {}).get("accuracy", 0)
            acc_s = metrics.get("student_adapter", {}).get("accuracy", 0)

            scenario_data[pair_key].setdefault(approach, {})[layer] = {
                "Trainer": acc_t,
                "Tester": acc_s,
            }
        continue

    for scenario_group in content:
        results = scenario_group.get("results", [])
        if not results:
            continue

        first_result = results[0]
        teacher = first_result.get("teacher_model") or first_result.get("teacher_name")
        student = first_result.get("student_model") or first_result.get("student_name")
        pair_key = (teacher, student)
        if pair_key not in scenario_data:
            continue

        for result in results:
            layer = result.get("layer_type") or result.get("type")
            if layer not in layers:
                continue

            metrics = normalize_metrics(result)
            trainer_metrics = metrics.get("teacher", {})
            tester_metrics = (
                metrics.get("student_on_teacher")
                or metrics.get("student_adapter")
                or {}
            )
            acc_t = trainer_metrics.get("accuracy", 0)
            acc_s = tester_metrics.get("accuracy", 0)

            scenario_data[pair_key].setdefault(approach, {})[layer] = {
                "Trainer": acc_t,
                "Tester": acc_s,
            }

# -----------------------------
# Plot parameters - UPDATED TO MATCH crossValScript.py
# -----------------------------
x = np.arange(len(approaches))
bar_width = 0.12

# Updated colors to match crossValScript.py
colors_trainer = {
    "attn": '#1f77b4',    # Blue
    "hidden": '#2ca02c',  # Green
    "mlp": '#d62728'      # Red
}
colors_tester = {
    "attn": '#aec7e8',    # Light blue
    "hidden": '#98df8a',  # Light green
    "mlp": '#ff9896'      # Light red
}

offsets = {
    ("attn", "Trainer"): -2.5 * bar_width,
    ("attn", "Tester"): -1.5 * bar_width,
    ("hidden", "Trainer"): -0.5 * bar_width,
    ("hidden", "Tester"): 0.5 * bar_width,
    ("mlp", "Trainer"): 1.5 * bar_width,
    ("mlp", "Tester"): 2.5 * bar_width,
}

title_style = {"fontsize": 10, "fontweight": "bold"}
label_style = {"fontsize": 10, "fontweight": "bold"}
tick_style = {"fontsize": 10, "fontweight": "bold"}
legend_prop = {"size": 12, "weight": "bold"}

scenarios_to_plot = [pair for pair in scenario_pairs if scenario_data[pair]]
if not scenarios_to_plot:
    raise RuntimeError("Nessun scenario valido trovato per i modelli trainer/tester configurati.")

# Salva figure separate per ogni scenario
for scenario_key in scenarios_to_plot:
    teacher_name, student_name = scenario_key
    setup_data = scenario_data[scenario_key]

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.set_facecolor("#fdfdfd")

    for layer in layers:
        # Plot Trainer bars (solid, no edge)
        values_trainer = [
            setup_data.get(approach, {}).get(layer, {}).get("Trainer", 0)
            for approach in approaches
        ]
        ax.bar(
            x + offsets[(layer, "Trainer")],
            values_trainer,
            width=bar_width,
            color=colors_trainer[layer],
            edgecolor='black',
            linewidth=0.0,
        )
        
        # Plot Tester bars (lighter with colored edge matching trainer)
        values_tester = [
            setup_data.get(approach, {}).get(layer, {}).get("Tester", 0)
            for approach in approaches
        ]
        ax.bar(
            x + offsets[(layer, "Tester")],
            values_tester,
            width=bar_width,
            color=colors_tester[layer],
            edgecolor=colors_trainer[layer],
            linewidth=1,
        )

    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("Accuracy", **label_style)
    ax.set_xticks(x)
    ax.set_xticklabels(approaches, rotation=0, ha="center")
    
    for label in ax.get_xticklabels():
        label.set_fontsize(tick_style["fontsize"])
        label.set_fontweight(tick_style["fontweight"])
    
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")

    # Updated legend to match crossValScript.py style
    legend_handles = []
    for layer in layers:
        legend_handles.append(
            Patch(
                facecolor=colors_trainer[layer],
                label=f"{layer} (Trainer)",
            )
        )
        legend_handles.append(
            Patch(
                facecolor=colors_tester[layer],
                edgecolor=colors_trainer[layer],
                linewidth=1,
                label=f"{layer} (Tester)",
            )
        )

    ax.legend(
        handles=legend_handles,
        title="Layer (Model)",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        prop=legend_prop,
        title_fontproperties={'weight': 'bold', 'size': 14}
    )

    fig.tight_layout()
    filename = f"accuracy_bars_approach_{dataset}_{teacher_name.replace('/', '_')}_{student_name.replace('/', '_')}.pdf"
    plt.savefig(filename, bbox_inches='tight')
    #plt.show()
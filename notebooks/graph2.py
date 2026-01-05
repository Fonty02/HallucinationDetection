import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# -----------------------------
# Configuration
# -----------------------------
datasets = ["LLama_Gemma_BBF", "LLama_Gemma_BBC", "LLama_Gemma_HE"]
dataset_titles = {
    "LLama_Gemma_BBF": "Factual",
    "LLama_Gemma_BBC": "Logical", 
    "LLama_Gemma_HE": "Contextual"
}

# Define both scenarios explicitly
scenario_llama_gemma = ("Llama-3.1-8B-Instruct", "gemma-2-9b-it")
scenario_gemma_llama = ("gemma-2-9b-it", "Llama-3.1-8B-Instruct")

base_dir = Path.cwd()

approaches = ["FL", "AdMLP", "FNL", "RNL", "O4A"]
layers = ["attn", "hidden", "mlp"]

path_templates = {
    "FL": "linearApproach/{dataset}/results_metrics/linear_probe_results.json",
    "AdMLP": "HybridApproach/{dataset}/results_metrics/hybrid_adapter_logreg_results.json",
    "FNL": "nonLinearApproach/approach1FullDIM/{dataset}/results_metrics/approach1_mlp_prober_results.json",
    "RNL": "nonLinearApproach/approach2Projected/{dataset}/results_metrics/approach2_autoencoder_results.json",
    "O4A": "nonLinearApproach/approach3OneForAll/{dataset}/results_metrics/approach3_frozen_head_results.json",
}

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


def load_dataset_data(dataset, target_scenario):
    """Load data for a specific dataset and scenario"""
    paths = {
        approach: base_dir / template.format(dataset=dataset)
        for approach, template in path_templates.items()
    }
    
    scenario_data = {}
    
    for approach, path in paths.items():
        if not path.exists():
            print(f"Warning: file not found for {approach}: {path}")
            continue

        with open(path, "r", encoding="utf-8") as f:
            content = json.load(f)

        if approach == "O4A":
            for entry in content:
                teacher = entry.get("teacher_model", "")
                student = entry.get("student_model", "")
                pair_key = (teacher, student)
                if pair_key != target_scenario:
                    continue

                layer = entry.get("layer_type")
                if layer not in layers:
                    continue

                metrics = entry.get("metrics", {})
                acc_t = metrics.get("teacher", {}).get("accuracy", 0)
                acc_s = metrics.get("student_adapter", {}).get("accuracy", 0)

                scenario_data.setdefault(approach, {})[layer] = {
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
            if pair_key != target_scenario:
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

                scenario_data.setdefault(approach, {})[layer] = {
                    "Trainer": acc_t,
                    "Tester": acc_s,
                }
    
    return scenario_data


def create_unified_plot(datasets, scenario, filename_suffix):
    """Create a unified plot for the given scenario"""
    # Plot parameters
    x = np.arange(len(approaches))
    bar_width = 0.12

    colors_trainer = {
        "attn": '#1f77b4',
        "hidden": '#2ca02c',
        "mlp": '#d62728'
    }
    colors_tester = {
        "attn": '#aec7e8',
        "hidden": '#98df8a',
        "mlp": '#ff9896'
    }

    offsets = {
        ("attn", "Trainer"): -2.5 * bar_width,
        ("attn", "Tester"): -1.5 * bar_width,
        ("hidden", "Trainer"): -0.5 * bar_width,
        ("hidden", "Tester"): 0.5 * bar_width,
        ("mlp", "Trainer"): 1.5 * bar_width,
        ("mlp", "Tester"): 2.5 * bar_width,
    }

    label_style = {"fontsize": 24, "fontweight": "bold"}
    legend_prop = {"size": 28, "weight": "bold"}

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(33, 6))

    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        ax.set_facecolor("#fdfdfd")
        
        # Load data for this dataset and scenario
        setup_data = load_dataset_data(dataset, scenario)
        
        if not setup_data:
            print(f"Warning: No valid data found for {dataset} with scenario {scenario}")
            continue
        
        # Plot bars for each layer
        for layer in layers:
            # Plot Trainer bars
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
            
            # Plot Tester bars
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
        
        # Styling
        ax.set_ylim(0.5, 1.0)
        ax.set_title(dataset_titles[dataset], fontsize=26, fontweight='bold')
        
        # Only set ylabel for first subplot
        if idx == 0:
            ax.set_ylabel("Accuracy", **label_style)
        
        ax.set_xticks(x)
        ax.set_xticklabels(approaches, rotation=0, ha="center")
        
        for label in ax.get_xticklabels():
            label.set_fontsize(24)
            label.set_fontweight("bold")
        
        for label in ax.get_yticklabels():
            label.set_fontweight("bold")
            label.set_fontsize(24)

    # Add legend only to the last subplot
    legend_handles = []
    for layer in layers:
        legend_handles.append(
            Patch(
                facecolor=colors_trainer[layer],
                label=f"{layer} (Tr)",
            )
        )
        legend_handles.append(
            Patch(
                facecolor=colors_tester[layer],
                edgecolor=colors_trainer[layer],
                linewidth=1,
                label=f"{layer} (Te)",
            )
        )

    # Place a flat (horizontal) legend above the three subplots
    lg = fig.legend(
        handles=legend_handles,
        title="Layer (Model)",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.25),
        ncol=6,
        frameon=True,
        prop=legend_prop,
    )
    # Rendiamo il titolo della legenda in grassetto e impostiamo la dimensione
    lg.get_title().set_fontweight('bold')
    lg.get_title().set_fontsize(36)

    # Make room for the legend at the top
    fig.tight_layout(rect=[0, 0, 1, 0.88])

    # Save figure
    filename = f"accuracy_bars_unified_{filename_suffix}.pdf"
    plt.savefig(filename, bbox_inches='tight')
    print(f"Saved unified plot: {filename}")
    plt.close()


# -----------------------------
# Generate both plots
# -----------------------------

# Plot 1: Llama → Gemma (Llama as Trainer, Gemma as Tester)
print("Generating plot: Llama (Trainer) → Gemma (Tester)")
create_unified_plot(datasets, scenario_llama_gemma, "llama_to_gemma")

# Plot 2: Gemma → Llama (Gemma as Trainer, Llama as Tester)
print("Generating plot: Gemma (Trainer) → Llama (Tester)")
create_unified_plot(datasets, scenario_gemma_llama, "gemma_to_llama")

print("\nBoth plots generated successfully!")
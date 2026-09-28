"""EXPERIMENTS definition, built from the models/datasets below and the selected layers JSON.

The layer selection is produced by scripts/O4A_experiments/select_top_layers.py.
Set O4A_SELECTED_LAYERS to load a different selection file.

This module has no torch / env-var requirements, so scripts can import it directly.
"""

import itertools
import json
import os
from pathlib import Path

# Model name -> short name used in experiment keys (order defines the pair order)
MODELS = {
    "Qwen2.5-7B": "Qwen",
    "Falcon3-7B-Base": "Falcon",
    "gemma-2-9b-it": "Gemma",
    "Llama-3.1-8B-Instruct": "Llama",
}

# Dataset name -> short name used in experiment keys
DATASETS = {
    "belief_bank_constraints": "BBC",
    "belief_bank_facts": "BBF",
    "halu_eval": "HE",
}

DEFAULT_SELECTED_LAYERS_PATH = Path(__file__).resolve().with_name("selected_layers.json")


def experiment_name(trainer: str, tester: str, dataset: str) -> str:
    """E.g. ("Qwen2.5-7B", "Falcon3-7B-Base", "belief_bank_constraints") -> "QwenToFalcon_BBC"."""
    return f"{MODELS[trainer]}To{MODELS[tester]}_{DATASETS[dataset]}"


def load_selected_layers(path: str | Path | None = None) -> dict:
    """Return {model: {dataset: {layer_type: [layers]}}} from the selection JSON."""
    path = Path(path or os.environ.get("O4A_SELECTED_LAYERS") or DEFAULT_SELECTED_LAYERS_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"Selected layers file not found: {path}. "
            "Run scripts/O4A_experiments/select_top_layers.py first."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["layers"]


def build_experiments(selected_layers: dict) -> dict:
    """
    Build the EXPERIMENTS dict for every dataset and every ordered (trainer, tester) pair.

    Each entry defines:
      - dataset:        name of the dataset folder in activation_cache
      - trainer:        model name used as "teacher" (trains the prober)
      - tester:         model name used as "student" (cross-model eval)
      - trainer_layers: dict with attn/mlp/hidden layer indices for trainer
      - tester_layers:  dict with attn/mlp/hidden layer indices for tester

    Pairs without a layer selection for both models on the dataset are skipped.
    """
    experiments = {}
    for dataset in DATASETS:
        for a, b in itertools.combinations(MODELS, 2):
            for trainer, tester in ((a, b), (b, a)):
                trainer_layers = selected_layers.get(trainer, {}).get(dataset)
                tester_layers = selected_layers.get(tester, {}).get(dataset)
                if trainer_layers is None or tester_layers is None:
                    continue
                experiments[experiment_name(trainer, tester, dataset)] = {
                    "dataset": dataset,
                    "trainer": trainer,
                    "tester": tester,
                    "trainer_layers": trainer_layers,
                    "tester_layers": tester_layers,
                }
    return experiments


EXPERIMENTS = build_experiments(load_selected_layers())

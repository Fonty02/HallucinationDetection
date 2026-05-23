# HallucinationDetection

Tools for hallucination detection research based on LLM activations.

This repository currently focuses on:
- activation extraction from LLM layers (`hidden`, `mlp`, `attn`)
- One-For-All (O4A) cross-model experiments
- aggregation and plotting utilities for experiment outputs

## Setup (uv + pyproject only)

This project now uses `pyproject.toml` as the single source of dependencies.

1. Create the environment with Python 3.12.

```bash
uv venv --python 3.12
```

2. Install dependencies from `pyproject.toml`.

```bash
uv sync
```

3. Optional: activate the virtual environment.

```bash
# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

## Current Repository Structure

```text
HallucinationDetection/
|-- pyproject.toml
|-- uv.lock
|-- README.md
|-- src/
|   |-- data/
|   |-- evaluation/
|   |-- model/
|   |-- o4a/
|   |-- visualization/
|   |-- 3plot_ablation.py
|   |-- 3plot_linear.py
|   |-- plot_classic_means_ablation.py
|   |-- plot_classic_means_complete.py
|   |-- plot_classic_means_linear.py
|   `-- plot_cross_domain_means_complete.py
|-- scripts/
|   |-- save_activations_all.py
|   |-- plot_pca_multidataset.py
|   |-- O4A/
|   |-- O4A_experiments/
|   `-- PCA/
|-- results/            # generated outputs
|-- logs/               # runtime logs
`-- notebooks/          # optional notebooks
```

Notes:
- `activation_cache/` is expected at runtime for activation-based experiments.
- In this branch, `data/` at project root is not tracked in Git.

## Quick Start

Run single activation extraction:

```bash
uv run python -m src.model.predict \
  --model_name "meta-llama/Llama-3.1-8B-Instruct" \
  --data_name "halu_eval" \
  --quantization
```

Run preconfigured activation batches:

```bash
uv run python scripts/save_activations_all.py --list
uv run python scripts/save_activations_all.py --experiment qwen_bb_facts
```

## O4A Runners

The scripts under `src/o4a/` require these environment variables:
- `O4A_SEED`
- `O4A_DEVICE` (for example `cpu` or `cuda:0`)

Example:

```bash
# Linux/macOS
O4A_SEED=42 O4A_DEVICE=cpu uv run python src/o4a/run_experiments.py \
  --experiments GemmaToLlama_HE \
  --output results/experiments/run.csv
```

Other entry points:
- `src/o4a/run_layers_study.py`
- `src/o4a/run_cross_domain_one_for_all.py`
- `src/o4a/run_one4all_evaluation.py`

## Plotting

PCA multi-dataset plot:

```bash
uv run python scripts/plot_pca_multidataset.py \
  --model-name Qwen2.5-7B \
  --dataset belief_bank_facts \
  --layer-type attn
```

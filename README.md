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
|   |-- data/                          # dataset loaders (HaluEval, BeliefBank) and utilities
|   |-- evaluation/                    # evaluation utilities
|   |-- model/                         # LLM wrappers, activation hooks, prompts, autorater
|   |-- o4a/                           # One-For-All cross-model experiments
|   |   |-- methods/                   # ridge, procrustes, cka, cca, hybrid, nonlinear, one_for_all
|   |   |-- config.py                  # EXPERIMENTS dict + per-method hyperparameters
|   |   |-- data.py
|   |   |-- models.py
|   |   |-- run_experiments.py
|   |   |-- run_layers_study.py
|   |   |-- run_cross_domain_one_for_all.py
|   |   `-- run_one4all_evaluation.py
|   |-- visualization/
|   |-- 3plot_ablation.py
|   |-- 3plot_linear.py
|   |-- plot_classic_means_ablation.py
|   |-- plot_classic_means_complete.py
|   |-- plot_classic_means_linear.py
|   `-- plot_cross_domain_means_complete.py
|-- scripts/
|   |-- save_activations_all.py        # preconfigured activation extraction batches
|   |-- plot_pca_multidataset.py       # PCA scatter plots across datasets
|   |-- O4A/                           # HTCondor submit-file generators for activation jobs
|   |-- O4A_experiments/               # HTCondor submit + aggregation for O4A runners
|   `-- PCA/                           # HTCondor submit-file generator for PCA plots
|-- activation_cache/                  # generated activations (created at runtime)
|-- saved_models/                      # trained probers / adapters (created at runtime)
|-- results/                           # generated CSV / JSON outputs
|-- logs/                              # runtime logs
`-- notebooks/                         # optional notebooks
```

Notes:
- `activation_cache/` is expected at runtime for activation-based experiments.
- `saved_models/` is created by O4A runners when persisting trained weights.

## Supported Models and Datasets

Models referenced in the experiment configs:
- `meta-llama/Llama-3.1-8B-Instruct`
- `Qwen/Qwen2.5-7B`
- `tiiuae/Falcon3-7B-Base`
- `google/gemma-2-9b-it`

Datasets supported by `src/data/`:
- `halu_eval` (HaluEval)
- `belief_bank` with `belief_bank_data_type` in {`facts`, `constraints`}

## Quick Start

Run a single activation-extraction job:

```bash
uv run python -m src.model.predict \
  --model_name "meta-llama/Llama-3.1-8B-Instruct" \
  --data_name "halu_eval" \
  --quantization
```

Run a preconfigured batch from `scripts/save_activations_all.py`:

```bash
# list available configurations
uv run python scripts/save_activations_all.py --list

# run one experiment
uv run python scripts/save_activations_all.py --experiment qwen_bb_facts

# run all experiments sequentially
uv run python scripts/save_activations_all.py --experiment all

# override device or cap samples
uv run python scripts/save_activations_all.py \
  --experiment llama_halu --device cuda:0 --max-samples 1000
```

Preconfigured experiment IDs: `qwen_bb_facts`, `qwen_bb_constraints`, `qwen_halu`,
`falcon_bb_facts`, `falcon_bb_constraints`, `falcon_halu`,
`gemma_bb_facts`, `gemma_bb_constraints`, `gemma_halu`,
`llama_bb_facts`, `llama_bb_constraints`, `llama_halu`.

## O4A Runners

The scripts under `src/o4a/` require these environment variables:
- `O4A_SEED` — integer seed for the run
- `O4A_DEVICE` — torch device string, e.g. `cpu`, `cuda:0`

These are validated at import time (`src/o4a/config.py`). There are no defaults: every job must set them explicitly (HTC-friendly).

Constants used by the runners:
- layer types: `attn`, `mlp`, `hidden`
- methods: `ridge_regressor`, `procrustes`, `cka`, `cca`, `hybrid`, `full_nonlinear`, `reduced_nonlinear`, `one_for_all`
- metrics: `accuracy`, `precision`, `recall`, `f1`, `auroc`

### Main experiment loop

Run all configured methods over selected experiments and write a CSV:

```bash
# Linux/macOS
O4A_SEED=42 O4A_DEVICE=cpu uv run python src/o4a/run_experiments.py \
  --experiments GemmaToLlama_HE \
  --output results/experiments/run.csv
```

```powershell
# Windows PowerShell
$env:O4A_SEED=42; $env:O4A_DEVICE="cpu"
uv run python src/o4a/run_experiments.py `
  --experiments GemmaToLlama_HE `
  --output results/experiments/run.csv
```

Experiments are named `<Trainer>To<Tester>_<Dataset>`, where dataset suffixes are
`BBC` (belief_bank_constraints), `BBF` (belief_bank_facts), `HE` (halu_eval).
See `src/o4a/config.py::EXPERIMENTS` for the full list of trainer/tester pairs
and the layer indices used per model.

### Other entry points

- `src/o4a/run_layers_study.py` — layer-wise study with 10 deterministic balanced
  train/test splits and logistic-regression probes per layer; outputs JSON.
- `src/o4a/run_cross_domain_one_for_all.py` — train OneForAll on one dataset
  domain and evaluate on another.
- `src/o4a/run_one4all_evaluation.py` — train only OneForAll for a single O4A
  experiment, save weights, and export per-instance predictions as JSON
  (one file per LLM).

## Plotting

PCA multi-dataset scatter (single layer):

```bash
uv run python scripts/plot_pca_multidataset.py \
  --model-name Qwen2.5-7B \
  --dataset belief_bank_facts \
  --layer-type attn
```

PCA across multiple `(dataset, layer)` pairs in one figure:

```bash
uv run python scripts/plot_pca_multidataset.py \
  --model-name gemma-2-9b-it \
  --dataset-layers belief_bank_constraints:23,belief_bank_facts:21,halu_eval:21 \
  --layer-type attn \
  --max-points-per-class 1500 \
  --balance
```

Aggregate plots over `results/`:
- `src/plot_classic_means_complete.py`, `src/plot_classic_means_linear.py`,
  `src/plot_classic_means_ablation.py` — per-method mean curves
- `src/plot_cross_domain_means_complete.py` — cross-domain mean curves
- `src/3plot_linear.py`, `src/3plot_ablation.py` — composite 3-panel plots

## HTCondor Submit Generators

Helpers under `scripts/` generate Condor submit files for the most common jobs:
- `scripts/O4A/generate_condor_save_activations.py` — activation extraction
- `scripts/O4A_experiments/generate_condor_run_experiments.py` — main O4A loop
- `scripts/O4A_experiments/generate_condor_run_layers_study.py` — layer study
- `scripts/O4A_experiments/generate_condor_cross_domain_one_for_all.py` — cross-domain O4A
- `scripts/PCA/generate_condor_plot_pca.py` — PCA plots
- `scripts/O4A_experiments/aggregate_results.py` — aggregate per-job CSV outputs
- `scripts/O4A_experiments/export_sampling_stats.py` — export sampling statistics

## Outputs

- `activation_cache/<model>/<dataset>/<layer_type>/...` — extracted activations
- `saved_models/...` — trained probers / adapters (O4A runners)
- `results/...` — CSV and JSON metric files
- `logs/...` — runtime logs

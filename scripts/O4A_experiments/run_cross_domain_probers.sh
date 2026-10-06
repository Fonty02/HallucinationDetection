#!/bin/bash
# HTCondor/SLURM wrapper for single-LLM cross-domain prober jobs.
# Probers are trained on TRAIN_DATASET and evaluated on ACTIVATION_DATASET (same model, no alignment).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

# Activate virtualenv if available.
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Arguments MUST be provided by the generator (12 required + optional layer types)
if [ $# -lt 12 ]; then
    echo "ERROR: Insufficient arguments. At least 12 arguments are required:"
    echo "  1. MODEL_NAME (e.g., gemma-2-9b-it)"
    echo "  2. TRAIN_DATASET (e.g., belief_bank_facts)"
    echo "  3. ACTIVATION_DATASET (e.g., halu_eval)"
    echo "  4. SEED (e.g., 42)"
    echo "  5. DEVICE (e.g., cuda:0)"
    echo "  6. NUM_WORKERS (e.g., 8)"
    echo "  7. PIN_MEMORY (e.g., true)"
    echo "  8. PREFETCH_FACTOR (e.g., 4)"
    echo "  9. CUDNN_BENCHMARK (e.g., true)"
    echo " 10. USE_AMP (e.g., true)"
    echo " 11. COMPILE_MODEL (e.g., false)"
    echo " 12. PROBERS (all, or comma-separated subset, e.g. ridge_regressor,full_nonlinear,one_for_all)"
    echo " 13+. Optional LAYER_TYPES (e.g., hidden or attn mlp hidden)"
    exit 1
fi

MODEL_NAME="$1"
TRAIN_DATASET="$2"
ACTIVATION_DATASET="$3"
SEED="$4"
DEVICE="$5"
NUM_WORKERS="$6"
PIN_MEMORY="$7"
PREFETCH_FACTOR="$8"
CUDNN_BENCHMARK="$9"
USE_AMP="${10}"
COMPILE_MODEL="${11}"
PROBERS="${12}"
LAYER_TYPES_ARGS=()
if [ "$#" -gt 12 ]; then
    LAYER_TYPES_ARGS=("${@:13}")
fi

# ==================================================================
# PERFORMANCE OPTIMIZATIONS
# ==================================================================

export CUDA_LAUNCH_BLOCKING=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NVIDIA_TF32_OVERRIDE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_SHOW_CPP_STACKTRACES=0
export TORCH_LOGS="-all"
export OMP_NUM_THREADS="$NUM_WORKERS"
export MKL_NUM_THREADS="$NUM_WORKERS"
export NCCL_DEBUG=WARN

ulimit -n 65536 2>/dev/null || true

# ==================================================================
# BUILD AND RUN COMMAND
# ==================================================================

RUN_ID="${MODEL_NAME}__${TRAIN_DATASET}__activation__${ACTIVATION_DATASET}"
OUTPUT_DIR="results/cross_domain_probers/${RUN_ID}/seed_${SEED}"
OUTPUT_CSV="${OUTPUT_DIR}/results.csv"
mkdir -p "$OUTPUT_DIR"

CMD=(
    python -u -W ignore
    src/o4a/run_cross_domain_probers.py
    --model "$MODEL_NAME"
    --train-dataset "$TRAIN_DATASET"
    --activation-dataset "$ACTIVATION_DATASET"
    --output "$OUTPUT_CSV"
)
if [ "$PROBERS" != "all" ]; then
    IFS=',' read -r -a PROBERS_ARGS <<< "$PROBERS"
    CMD+=(--probers "${PROBERS_ARGS[@]}")
fi
if [ ${#LAYER_TYPES_ARGS[@]} -gt 0 ]; then
    CMD+=(--layer-types "${LAYER_TYPES_ARGS[@]}")
fi

# Mandatory environment variables for o4a.config
export O4A_SEED="$SEED"
export O4A_DEVICE="$DEVICE"

# Performance environment variables for o4a.data
export O4A_NUM_WORKERS="$NUM_WORKERS"
export O4A_PIN_MEMORY="$PIN_MEMORY"
export O4A_PREFETCH_FACTOR="$PREFETCH_FACTOR"
export O4A_CUDNN_BENCHMARK="$CUDNN_BENCHMARK"
export O4A_USE_AMP="$USE_AMP"
export O4A_COMPILE_MODEL="$COMPILE_MODEL"

echo "========================================"
echo "Starting single-LLM cross-domain prober evaluation"
echo "Model: $MODEL_NAME"
echo "Train dataset: $TRAIN_DATASET"
echo "Activation dataset: $ACTIVATION_DATASET"
echo "Seed: $SEED"
echo "Device: $DEVICE"
echo "Probers: $PROBERS"
if [ ${#LAYER_TYPES_ARGS[@]} -gt 0 ]; then
    echo "Layer types: ${LAYER_TYPES_ARGS[*]}"
else
    echo "Layer types: all (default)"
fi
echo "Output: $OUTPUT_CSV"
echo "========================================"

time "${CMD[@]}"

echo "========================================"
echo "Completed single-LLM cross-domain prober evaluation"
echo "Run id: $RUN_ID (seed=$SEED)"
echo "========================================"

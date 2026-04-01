#!/bin/bash
# HTCondor wrapper for OneForAll retraining + cross-dataset evaluation jobs.
# Training is done once on ENCODER_EXPERIMENT domain; evaluation is on ACTIVATION_DATASET.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

# Activate virtualenv if available.
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Arguments MUST be provided by HTC (10 required + optional layer types)
if [ $# -lt 10 ]; then
    echo "ERROR: Insufficient arguments. HTC must provide at least 10 arguments:"
    echo "  1. ENCODER_EXPERIMENT (e.g., LlamaToGemma_BBF)"
    echo "  2. ACTIVATION_DATASET (e.g., halu_eval)"
    echo "  3. SEED (e.g., 42)"
    echo "  4. DEVICE (e.g., cuda:0)"
    echo "  5. NUM_WORKERS (e.g., 8)"
    echo "  6. PIN_MEMORY (e.g., true)"
    echo "  7. PREFETCH_FACTOR (e.g., 4)"
    echo "  8. CUDNN_BENCHMARK (e.g., true)"
    echo "  9. USE_AMP (e.g., true)"
    echo " 10. COMPILE_MODEL (e.g., false)"
    echo " 11+. Optional LAYER_TYPES (e.g., hidden or attn mlp hidden)"
    exit 1
fi

ENCODER_EXPERIMENT="$1"
ACTIVATION_DATASET="$2"
SEED="$3"
DEVICE="$4"
NUM_WORKERS="$5"
PIN_MEMORY="$6"
PREFETCH_FACTOR="$7"
CUDNN_BENCHMARK="$8"
USE_AMP="$9"
COMPILE_MODEL="${10}"
LAYER_TYPES_ARGS=()
if [ "$#" -gt 10 ]; then
    LAYER_TYPES_ARGS=("${@:11}")
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

RUN_ID="${ENCODER_EXPERIMENT}__activation__${ACTIVATION_DATASET}"
OUTPUT_DIR="results/cross_domain_one_for_all/${RUN_ID}/seed_${SEED}"
OUTPUT_JSON="${OUTPUT_DIR}/results.json"
mkdir -p "$OUTPUT_DIR"

CMD=(
    python -u -W ignore
    src/o4a/run_cross_domain_one_for_all.py
    --encoder-experiment "$ENCODER_EXPERIMENT"
    --activation-dataset "$ACTIVATION_DATASET"
    --output "$OUTPUT_JSON"
)
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
echo "Starting OneForAll retraining + cross-dataset evaluation"
echo "Encoder experiment: $ENCODER_EXPERIMENT"
echo "Activation dataset: $ACTIVATION_DATASET"
echo "Seed: $SEED"
echo "Device: $DEVICE"
echo "Workers: $NUM_WORKERS"
echo "AMP: $USE_AMP"
if [ ${#LAYER_TYPES_ARGS[@]} -gt 0 ]; then
    echo "Layer types: ${LAYER_TYPES_ARGS[*]}"
else
    echo "Layer types: all (default)"
fi
echo "Output: $OUTPUT_JSON"
echo "========================================"

time "${CMD[@]}"

echo "========================================"
echo "Completed OneForAll retraining + cross-dataset evaluation"
echo "Run id: $RUN_ID (seed=$SEED)"
echo "========================================"

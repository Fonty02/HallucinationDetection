#!/bin/bash
# HTCondor wrapper for OneForAll-only training on a selected O4A experiment.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Required arguments from HTC:
# 1) EXPERIMENT_NAME
# 2) SEED
# 3) DEVICE
# 4) NUM_WORKERS
# 5) PIN_MEMORY
# 6) PREFETCH_FACTOR
# 7) CUDNN_BENCHMARK
# 8) USE_AMP
# 9) COMPILE_MODEL
# 10) LAYER_TYPE
if [ $# -lt 10 ]; then
    echo "ERROR: Expected 10 arguments:"
    echo "  1. EXPERIMENT_NAME (e.g., GemmaToLlama_HE)"
    echo "  2. SEED (e.g., 42)"
    echo "  3. DEVICE (e.g., cuda:0)"
    echo "  4. NUM_WORKERS (e.g., 8)"
    echo "  5. PIN_MEMORY (e.g., true)"
    echo "  6. PREFETCH_FACTOR (e.g., 4)"
    echo "  7. CUDNN_BENCHMARK (e.g., true)"
    echo "  8. USE_AMP (e.g., true)"
    echo "  9. COMPILE_MODEL (e.g., false)"
    echo " 10. LAYER_TYPE (attn|mlp|hidden)"
    exit 1
fi

EXPERIMENT_NAME="$1"
SEED="$2"
DEVICE="$3"
NUM_WORKERS="$4"
PIN_MEMORY="$5"
PREFETCH_FACTOR="$6"
CUDNN_BENCHMARK="$7"
USE_AMP="$8"
COMPILE_MODEL="$9"
LAYER_TYPE="${10}"

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

OUTPUT_DIR="results/one4all_experiment_details"

export O4A_SEED="$SEED"
export O4A_DEVICE="$DEVICE"
export O4A_NUM_WORKERS="$NUM_WORKERS"
export O4A_PIN_MEMORY="$PIN_MEMORY"
export O4A_PREFETCH_FACTOR="$PREFETCH_FACTOR"
export O4A_CUDNN_BENCHMARK="$CUDNN_BENCHMARK"
export O4A_USE_AMP="$USE_AMP"
export O4A_COMPILE_MODEL="$COMPILE_MODEL"

CMD=(
    python -u -W ignore
    src/o4a/run_one4all_evaluation.py
    --experiment "$EXPERIMENT_NAME"
    --layer-type "$LAYER_TYPE"
    --output-dir "$OUTPUT_DIR"
)

echo "========================================"
echo "Starting OneForAll experiment"
echo "Experiment: $EXPERIMENT_NAME"
echo "Seed: $SEED"
echo "Device: $DEVICE"
echo "Workers: $NUM_WORKERS"
echo "AMP: $USE_AMP"
echo "Layer type: $LAYER_TYPE"
echo "Output root: $OUTPUT_DIR"
echo "========================================"

time "${CMD[@]}"

echo "========================================"
echo "Completed OneForAll experiment"
echo "Experiment: $EXPERIMENT_NAME | Seed: $SEED | Layer: $LAYER_TYPE"
echo "========================================"

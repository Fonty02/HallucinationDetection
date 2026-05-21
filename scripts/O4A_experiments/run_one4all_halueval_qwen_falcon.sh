#!/bin/bash
# HTCondor wrapper for OneForAll-only training on HaluEval (Qwen -> Falcon).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Required arguments from HTC:
# 1) SEED
# 2) DEVICE
# 3) NUM_WORKERS
# 4) PIN_MEMORY
# 5) PREFETCH_FACTOR
# 6) CUDNN_BENCHMARK
# 7) USE_AMP
# 8) COMPILE_MODEL
# 9) LAYER_TYPE
if [ $# -lt 9 ]; then
    echo "ERROR: Expected 9 arguments:"
    echo "  1. SEED (e.g., 42)"
    echo "  2. DEVICE (e.g., cuda:0)"
    echo "  3. NUM_WORKERS (e.g., 8)"
    echo "  4. PIN_MEMORY (e.g., true)"
    echo "  5. PREFETCH_FACTOR (e.g., 4)"
    echo "  6. CUDNN_BENCHMARK (e.g., true)"
    echo "  7. USE_AMP (e.g., true)"
    echo "  8. COMPILE_MODEL (e.g., false)"
    echo "  9. LAYER_TYPE (attn|mlp|hidden)"
    exit 1
fi

SEED="$1"
DEVICE="$2"
NUM_WORKERS="$3"
PIN_MEMORY="$4"
PREFETCH_FACTOR="$5"
CUDNN_BENCHMARK="$6"
USE_AMP="$7"
COMPILE_MODEL="$8"
LAYER_TYPE="$9"

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

OUTPUT_DIR="results/one4all_halueval_qwen_falcon"

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
    src/o4a/run_one_for_all_halueval_qwen_falcon.py
    --layer-type "$LAYER_TYPE"
    --output-dir "$OUTPUT_DIR"
)

echo "========================================"
echo "Starting OneForAll HaluEval (Qwen -> Falcon)"
echo "Seed: $SEED"
echo "Device: $DEVICE"
echo "Workers: $NUM_WORKERS"
echo "AMP: $USE_AMP"
echo "Layer type: $LAYER_TYPE"
echo "Output root: $OUTPUT_DIR"
echo "========================================"

time "${CMD[@]}"

echo "========================================"
echo "Completed OneForAll HaluEval (Qwen -> Falcon)"
echo "Seed: $SEED | Layer: $LAYER_TYPE"
echo "========================================"

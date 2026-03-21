#!/bin/bash
# HTCondor wrapper script for run_experiments.py
# Optimized for high RAM, VRAM, and CPU utilization
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

# Activate virtualenv if available
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Parse arguments
EXPERIMENT_NAME="$1"
SEED="$2"
DEVICE="${3:-cuda:0}"
NUM_WORKERS="${4:-8}"
PIN_MEMORY="${5:-true}"
PREFETCH_FACTOR="${6:-4}"
CUDNN_BENCHMARK="${7:-true}"
USE_AMP="${8:-true}"
COMPILE_MODEL="${9:-false}"

# ==================================================================
# PERFORMANCE OPTIMIZATIONS
# ==================================================================

# CUDA optimizations
export CUDA_LAUNCH_BLOCKING=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Enable TF32 for Ampere+ GPUs (faster matrix ops with slight precision loss)
export NVIDIA_TF32_OVERRIDE=1

# Memory allocator tuning for large allocations
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Disable debug/profiling overhead
export TORCH_SHOW_CPP_STACKTRACES=0
export TORCH_LOGS="-all"

# OpenMP threading for CPU-bound operations
export OMP_NUM_THREADS="$NUM_WORKERS"
export MKL_NUM_THREADS="$NUM_WORKERS"

# NCCL tuning (if multi-GPU, though we use single GPU)
export NCCL_DEBUG=WARN

# Increase file descriptor limit for data loading
ulimit -n 65536 2>/dev/null || true

# ==================================================================
# BUILD AND RUN COMMAND
# ==================================================================

# Create output directory for this specific run
OUTPUT_DIR="results/experiments/${EXPERIMENT_NAME}/seed_${SEED}"
mkdir -p "$OUTPUT_DIR"

CMD=(
    python -u -W ignore
    src/o4a/run_experiments.py
    --experiments "$EXPERIMENT_NAME"
    --output "${OUTPUT_DIR}/results.csv"
)

# Add seed handling via environment (run_experiments.py should pick this up)
export O4A_SEED="$SEED"
export O4A_DEVICE="$DEVICE"

# Performance environment variables for the Python script
export O4A_NUM_WORKERS="$NUM_WORKERS"
export O4A_PIN_MEMORY="$PIN_MEMORY"
export O4A_PREFETCH_FACTOR="$PREFETCH_FACTOR"
export O4A_CUDNN_BENCHMARK="$CUDNN_BENCHMARK"
export O4A_USE_AMP="$USE_AMP"
export O4A_COMPILE_MODEL="$COMPILE_MODEL"

echo "========================================"
echo "Starting experiment: $EXPERIMENT_NAME"
echo "Seed: $SEED"
echo "Device: $DEVICE"
echo "Workers: $NUM_WORKERS"
echo "AMP: $USE_AMP"
echo "Output: $OUTPUT_DIR"
echo "========================================"

# Run with timing
time "${CMD[@]}"

echo "========================================"
echo "Completed: $EXPERIMENT_NAME (seed=$SEED)"
echo "========================================"

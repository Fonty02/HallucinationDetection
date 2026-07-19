#!/bin/bash
# HTCondor wrapper for layer-wise study jobs.
# One job runs one (model, dataset, layer_type) combination across all split seeds.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

# Activate virtualenv if available.
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

if [ $# -lt 9 ]; then
    echo "ERROR: Insufficient arguments. HTC must provide at least 9 arguments:"
    echo "  1. MODEL_NAME (e.g., gemma-2-9b-it)"
    echo "  2. DATASET_NAME (e.g., belief_bank_facts)"
    echo "  3. LAYER_TYPE (attn|mlp|hidden)"
    echo "  4. SPLIT_SEEDS_CSV (e.g., 2,4,24,42,64,67,104,123,420,511)"
    echo "  5. TEST_SIZE (e.g., 0.3)"
    echo "  6. DEVICE (e.g., cpu or cuda:0)"
    echo "  7. MAX_ITER (e.g., 10000)"
    echo "  8. LOGREG_N_JOBS (e.g., -1)"
    echo "  9. OUTPUT_BASE_DIR (e.g., results/layers_study)"
    exit 1
fi

MODEL_NAME="$1"
DATASET_NAME="$2"
LAYER_TYPE="$3"
SPLIT_SEEDS_CSV="$4"
TEST_SIZE="$5"
DEVICE="$6"
MAX_ITER="$7"
LOGREG_N_JOBS="$8"
OUTPUT_BASE_DIR="$9"

SAFE_MODEL="${MODEL_NAME//\//__}"
SAFE_DATASET="${DATASET_NAME//\//__}"
SAFE_LAYER="${LAYER_TYPE//\//__}"

OUTPUT_DIR="${OUTPUT_BASE_DIR}/${SAFE_MODEL}/${SAFE_DATASET}"
mkdir -p "$OUTPUT_DIR"
OUTPUT_CSV="${OUTPUT_DIR}/layer_study_${SAFE_MODEL}_${SAFE_DATASET}_${SAFE_LAYER}.csv"

CMD=(
    python -u -W ignore
    src/o4a/run_layers_study.py
    --model "$MODEL_NAME"
    --dataset "$DATASET_NAME"
    --layer-type "$LAYER_TYPE"
    --split-seeds "$SPLIT_SEEDS_CSV"
    --test-size "$TEST_SIZE"
    --device "$DEVICE"
    --max-iter "$MAX_ITER"
    --logreg-n-jobs "$LOGREG_N_JOBS"
    --output "$OUTPUT_CSV"
)

echo "========================================"
echo "Starting layer study job"
echo "Model: $MODEL_NAME"
echo "Dataset: $DATASET_NAME"
echo "Layer type: $LAYER_TYPE"
echo "Split seeds: $SPLIT_SEEDS_CSV"
echo "Test size: $TEST_SIZE"
echo "Device: $DEVICE"
echo "Max iter: $MAX_ITER"
echo "LogReg n_jobs: $LOGREG_N_JOBS"
echo "Output: $OUTPUT_CSV"
echo "========================================"

time "${CMD[@]}"

echo "========================================"
echo "Completed layer study job"
echo "Model: $MODEL_NAME | Dataset: $DATASET_NAME | Layer type: $LAYER_TYPE"
echo "========================================"

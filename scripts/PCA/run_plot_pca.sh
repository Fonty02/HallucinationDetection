#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

# Optional virtualenv activation.
if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
fi

if [ $# -lt 9 ]; then
  echo "ERROR: Expected at least 9 arguments:"
  echo "  1. MODEL_NAME"
  echo "  2. DATASET_LAYERS (dataset:layer,dataset:layer,...)"
  echo "  3. LAYER_TYPE (attn|mlp|hidden)"
  echo "  4. MAX_POINTS_PER_CLASS (integer or None)"
  echo "  5. BALANCE (true|false)"
  echo "  6. STANDARDIZE (true|false)"
  echo "  7. OUTPUT_DIR"
  echo "  8. FILENAME"
  echo "  9. SEED"
  echo " 10. CACHE_DIR (optional, default activation_cache)"
  exit 1
fi

MODEL_NAME="$1"
DATASET_LAYERS="$2"
LAYER_TYPE="$3"
MAX_POINTS_PER_CLASS="$4"
BALANCE="$5"
STANDARDIZE="$6"
OUTPUT_DIR="$7"
FILENAME="$8"
SEED="$9"
CACHE_DIR="${10:-activation_cache}"

CMD=(
  python -u -W ignore
  scripts/plot_pca_multidataset.py
  --cache-dir "$CACHE_DIR"
  --model-name "$MODEL_NAME"
  --dataset-layers "$DATASET_LAYERS"
  --layer-type "$LAYER_TYPE"
  --output-dir "$OUTPUT_DIR"
  --filename "$FILENAME"
  --seed "$SEED"
)

if [ "$MAX_POINTS_PER_CLASS" != "None" ]; then
  CMD+=(--max-points-per-class "$MAX_POINTS_PER_CLASS")
fi

if [ "$BALANCE" = "true" ]; then
  CMD+=(--balance)
fi

if [ "$STANDARDIZE" = "true" ]; then
  CMD+=(--standardize)
fi

echo "========================================"
echo "MODEL_NAME: $MODEL_NAME"
echo "DATASET_LAYERS: $DATASET_LAYERS"
echo "LAYER_TYPE: $LAYER_TYPE"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "FILENAME: $FILENAME"
echo "SEED: $SEED"
echo "========================================"

time "${CMD[@]}"

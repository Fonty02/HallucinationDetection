#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

# Optional virtualenv activation when available.
if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
fi

if [ $# -lt 1 ]; then
  echo "Usage: $0 OUTPUT_JSON [CACHE_DIR] [CONFIG_PATH] [STATS_LAYER_TYPE]"
  echo "Example: $0 results/o4a_experiments/sampling_hallucination_stats.json activation_cache src/o4a/config.py attn"
  exit 1
fi

OUTPUT_JSON="$1"
CACHE_DIR="${2:-activation_cache}"
CONFIG_PATH="${3:-src/o4a/config.py}"
STATS_LAYER_TYPE="${4:-attn}"

CMD=(
  python -u -W ignore
  scripts/O4A_experiments/export_sampling_stats.py
  --output "$OUTPUT_JSON"
  --cache-dir "$CACHE_DIR"
  --config-path "$CONFIG_PATH"
  --stats-layer-type "$STATS_LAYER_TYPE"
)

echo "========================================"
echo "Export sampling stats"
echo "OUTPUT_JSON: $OUTPUT_JSON"
echo "CACHE_DIR: $CACHE_DIR"
echo "CONFIG_PATH: $CONFIG_PATH"
echo "STATS_LAYER_TYPE: $STATS_LAYER_TYPE"
echo "========================================"

time "${CMD[@]}"

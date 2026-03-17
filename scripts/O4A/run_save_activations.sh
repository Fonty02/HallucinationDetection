#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

# Optional virtualenv activation when available.
if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
fi

EXPERIMENT_ID="$1"
MAX_SAMPLES="$2"
DEVICE="$3"

CMD=(python -W ignore scripts/save_activations_all.py --experiment "$EXPERIMENT_ID")

if [ "$MAX_SAMPLES" != "None" ]; then
  CMD+=(--max-samples "$MAX_SAMPLES")
fi

if [ "$DEVICE" != "None" ]; then
  CMD+=(--device "$DEVICE")
fi

"${CMD[@]}"

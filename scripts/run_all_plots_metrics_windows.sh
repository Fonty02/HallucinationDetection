#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

METRICS=(accuracy precision recall f1 auroc)

if command -v uv >/dev/null 2>&1; then
  RUNNER=(uv run python)
elif command -v python >/dev/null 2>&1; then
  RUNNER=(python)
else
  echo "Errore: non trovo ne 'uv' ne 'python' nel PATH." >&2
  exit 1
fi

echo "Repository root: ${REPO_ROOT}"
echo "Runner: ${RUNNER[*]}"
echo "Metriche: ${METRICS[*]}"

for metric in "${METRICS[@]}"; do
  echo ""
  echo "=== Generazione grafici per metrica: ${metric} ==="

  "${RUNNER[@]}" src/plot_classic_means.py --metric "${metric}"
  "${RUNNER[@]}" src/plot_classic_boxplots.py --metric "${metric}"
  "${RUNNER[@]}" src/plot_cross_domain_means.py --metric "${metric}"
  "${RUNNER[@]}" src/plot_cross_domain_boxplots.py --metric "${metric}"
done

echo ""
echo "Completato: generati grafici per tutte le metriche in src/plots/."

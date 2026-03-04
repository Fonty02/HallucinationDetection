#!/usr/bin/env bash

set -euo pipefail

# Esegue in sequenza più esperimenti SLiM con parametri condivisi.
# Uso:
#   bash scripts/run_slim_experiments.sh
#   bash scripts/run_slim_experiments.sh -- 200 --batch_size 8

COMMON_ARGS=(
  --epochs 100
)

# Definisci qui gli esperimenti da lanciare in sequenza.
# Ogni elemento dell'array è una singola stringa contenente tutti i
# parametri per un esperimento. In questo modo ${#EXPERIMENTS[@]}
# restituisce il numero di esperimenti reali invece che il numero di
# token (che era 60 prima).

EXPERIMENTS=(
 "--num_pairs 2000  --batch_size 2 --accumulation_steps 1 --model_name google/gemma-2-9b-it --dataset halu_eval --target_layer 21 --device cuda:2"
)

echo "=================================================="
echo "SLiM Sequential Experiments"
echo "=============s====================================="
echo "Esperimenti totali: ${#EXPERIMENTS[@]}"
echo

# iteriamo direttamente sugli elementi dell'array, così non c'è
# bisogno di preoccuparsi degli indici.
count=${#EXPERIMENTS[@]}
for ((i=0; i<count; i++)); do
  exp_args="${EXPERIMENTS[i]}"
  run_id=$((i + 1))

  echo "[${run_id}/${count}] Avvio: ${exp_args}"
  uv run src/SLiM/train_hallucination.py "${COMMON_ARGS[@]}" ${exp_args} "$@"
  echo "[${run_id}/${count}] Completato"
done

echo "Tutti gli esperimenti sono terminati con successo."

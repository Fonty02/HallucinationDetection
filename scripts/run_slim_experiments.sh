#!/usr/bin/env bash

set -euo pipefail

# Esegue in sequenza più esperimenti SLiM con parametri condivisi.
# Uso:
#   bash scripts/run_slim_experiments.sh
#   bash scripts/run_slim_experiments.sh --epochs 200 --batch_size 8

COMMON_ARGS=(
  --epochs 1000
  --device cuda:2
)

# Definisci qui gli esperimenti da lanciare in sequenza.


EXPERIMENTS=(
  #"--model_name google/gemma-2-9b-it --dataset belief_bank_facts --num_pairs 6500 --batch_size 16 --accumulation_steps 16"
  "--model_name google/gemma-2-9b-it --dataset belief_bank_constraints --num_pairs 6500 --batch_size 16 --accumulation_steps 16"
  #"--model_name google/gemma-2-9b-it --dataset halu_eval --num_pairs 2500 --batch_size 4 --accumulation_steps 16"
  #"--model_name meta-llama/Llama-3.1-8B-Instruct --dataset belief_bank_facts --num_pairs 6500 --batch_size 16 --accumulation_steps 16"
  #"--model_name meta-llama/Llama-3.1-8B-Instruct --dataset belief_bank_constraints --num_pairs 6500 --batch_size 16 --accumulation_steps 16"
  #"--model_name meta-llama/Llama-3.1-8B-Instruct --dataset halu_eval --num_pairs 2500 --batch_size 4 --accumulation_steps 16"


)

echo "=================================================="
echo "SLiM Sequential Experiments"
echo "=================================================="
echo "Esperimenti totali: ${#EXPERIMENTS[@]}"
echo

for index in "${!EXPERIMENTS[@]}"; do
  exp_args="${EXPERIMENTS[$index]}"
  run_id=$((index + 1))

  echo "[${run_id}/${#EXPERIMENTS[@]}] Avvio: ${exp_args}"
  uv run src/SLiM/train_hallucination.py "${COMMON_ARGS[@]}" ${exp_args} "$@"
  echo "[${run_id}/${#EXPERIMENTS[@]}] Completato"
  echo
  uv run src/SLiM/inference_hallucination.py --num_samples 500 --device cuda:2
done

echo "Tutti gli esperimenti sono terminati con successo."

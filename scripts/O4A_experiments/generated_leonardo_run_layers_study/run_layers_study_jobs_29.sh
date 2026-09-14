#!/bin/bash

#SBATCH -A IscrC_ISAAC
#SBATCH -p boost_usr_prod
#SBATCH --qos normal
#SBATCH --time=24:00:00
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=0
#SBATCH --mem=123000
#SBATCH --job-name=o4a_layers_study_29
#SBATCH --output=logs/run_layers_study_leonardo/run_layers_study_29_%j.out
#SBATCH --error=logs/run_layers_study_leonardo/run_layers_study_29_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=l.laraspata3@phd.uniba.it


source .venv/bin/activate

srun -u bash scripts/O4A_experiments/run_layers_study.sh Llama-3.1-8B-Instruct belief_bank_constraints mlp 2,4,24,42,64,67,104,123,420,511 0.3 cpu 10000 -1 results/layers_study

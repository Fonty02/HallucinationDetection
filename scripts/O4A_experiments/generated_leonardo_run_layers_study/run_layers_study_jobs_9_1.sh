#!/bin/bash

#SBATCH -A IscrC_IMCAI
#SBATCH -p boost_usr_prod
#SBATCH --qos normal
#SBATCH --time=24:00:00
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem=123000
#SBATCH --job-name=o4a_layers_study_9_1
#SBATCH --output=logs/run_layers_study_leonardo/run_layers_study_9_1_%j.out
#SBATCH --error=logs/run_layers_study_leonardo/run_layers_study_9_1_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=l.laraspata3@phd.uniba.it


source .venv/bin/activate

srun -u bash scripts/O4A_experiments/run_layers_study.sh Falcon3-7B-Base halu_eval hidden 2,4,24,42,64,67,104,123,420,511 0.3 cuda:0 10000 -1 results/layers_study

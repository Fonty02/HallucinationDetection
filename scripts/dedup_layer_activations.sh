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
#SBATCH --job-name=dedup_layer_activations
#SBATCH --out=out_dedup_layer_activations.log
#SBATCH --err=err_dedup_layer_activations.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=l.laraspata3@phd.uniba.it


source .venv/bin/activate

srun -u python scripts/dedup_layer_activations.py --backup --model "Falcon3-7B-Base" --dataset "belief_bank_facts"

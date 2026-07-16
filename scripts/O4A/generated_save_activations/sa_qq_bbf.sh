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
#SBATCH --job-name=sa_qq_bbf
#SBATCH --out=out_sa_qq_bbf.log
#SBATCH --err=err_sa_qq_bbf.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=l.laraspata3@phd.uniba.it


source .venv/bin/activate

srun -u python -W ignore scripts/save_activations_all.py --experiment "qwen_bb_facts"
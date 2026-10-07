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
#SBATCH --job-name=43_o4a_cd_probers
#SBATCH --output=logs/run_cross_domain_probers_leonardo/run_cross_domain_probers_43_%j.out
#SBATCH --error=logs/run_cross_domain_probers_leonardo/run_cross_domain_probers_43_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=l.laraspata3@phd.uniba.it


source .venv/bin/activate

srun -u bash scripts/O4A_experiments/run_cross_domain_probers.sh gemma-2-9b-it none halu_eval belief_bank_constraints 420 cuda:0 8 true 4 true true false all
srun -u bash scripts/O4A_experiments/run_cross_domain_probers.sh gemma-2-9b-it none halu_eval belief_bank_constraints 511 cuda:0 8 true 4 true true false all
srun -u bash scripts/O4A_experiments/run_cross_domain_probers.sh gemma-2-9b-it none halu_eval belief_bank_facts 2 cuda:0 8 true 4 true true false all
srun -u bash scripts/O4A_experiments/run_cross_domain_probers.sh gemma-2-9b-it none halu_eval belief_bank_facts 4 cuda:0 8 true 4 true true false all

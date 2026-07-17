#!/bin/bash

#SBATCH -A IscrC_IMCAI
#SBATCH -p boost_usr_prod
#SBATCH --qos normal
#SBATCH --time=24:00:00
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem=32000
#SBATCH --job-name=o4a_run_exp_4
#SBATCH --output=logs/run_experiments_leonardo/run_experiments_4_%j.out
#SBATCH --error=logs/run_experiments_leonardo/run_experiments_4_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=l.laraspata3@phd.uniba.it


source .venv/bin/activate

srun -u bash scripts/O4A_experiments/run_experiments.sh QwenToFalcon_HE 64 cuda:0 8 true 4 true true false attn
srun -u bash scripts/O4A_experiments/run_experiments.sh QwenToFalcon_HE 64 cuda:0 8 true 4 true true false mlp
srun -u bash scripts/O4A_experiments/run_experiments.sh QwenToFalcon_HE 64 cuda:0 8 true 4 true true false hidden
srun -u bash scripts/O4A_experiments/run_experiments.sh QwenToFalcon_HE 67 cuda:0 8 true 4 true true false attn
srun -u bash scripts/O4A_experiments/run_experiments.sh QwenToFalcon_HE 67 cuda:0 8 true 4 true true false mlp
srun -u bash scripts/O4A_experiments/run_experiments.sh QwenToFalcon_HE 67 cuda:0 8 true 4 true true false hidden
srun -u bash scripts/O4A_experiments/run_experiments.sh QwenToFalcon_HE 104 cuda:0 8 true 4 true true false attn
srun -u bash scripts/O4A_experiments/run_experiments.sh QwenToFalcon_HE 104 cuda:0 8 true 4 true true false mlp
srun -u bash scripts/O4A_experiments/run_experiments.sh QwenToFalcon_HE 104 cuda:0 8 true 4 true true false hidden
srun -u bash scripts/O4A_experiments/run_experiments.sh QwenToFalcon_HE 123 cuda:0 8 true 4 true true false attn
srun -u bash scripts/O4A_experiments/run_experiments.sh QwenToFalcon_HE 123 cuda:0 8 true 4 true true false mlp
srun -u bash scripts/O4A_experiments/run_experiments.sh QwenToFalcon_HE 123 cuda:0 8 true 4 true true false hidden
srun -u bash scripts/O4A_experiments/run_experiments.sh QwenToFalcon_HE 420 cuda:0 8 true 4 true true false attn
srun -u bash scripts/O4A_experiments/run_experiments.sh QwenToFalcon_HE 420 cuda:0 8 true 4 true true false mlp
srun -u bash scripts/O4A_experiments/run_experiments.sh QwenToFalcon_HE 420 cuda:0 8 true 4 true true false hidden
srun -u bash scripts/O4A_experiments/run_experiments.sh QwenToFalcon_HE 511 cuda:0 8 true 4 true true false attn
srun -u bash scripts/O4A_experiments/run_experiments.sh QwenToFalcon_HE 511 cuda:0 8 true 4 true true false mlp
srun -u bash scripts/O4A_experiments/run_experiments.sh QwenToFalcon_HE 511 cuda:0 8 true 4 true true false hidden
srun -u bash scripts/O4A_experiments/run_experiments.sh QwenToGemma_BBC 2 cuda:0 8 true 4 true true false attn
srun -u bash scripts/O4A_experiments/run_experiments.sh QwenToGemma_BBC 2 cuda:0 8 true 4 true true false mlp
srun -u bash scripts/O4A_experiments/run_experiments.sh QwenToGemma_BBC 2 cuda:0 8 true 4 true true false hidden
srun -u bash scripts/O4A_experiments/run_experiments.sh QwenToGemma_BBC 4 cuda:0 8 true 4 true true false attn
srun -u bash scripts/O4A_experiments/run_experiments.sh QwenToGemma_BBC 4 cuda:0 8 true 4 true true false mlp
srun -u bash scripts/O4A_experiments/run_experiments.sh QwenToGemma_BBC 4 cuda:0 8 true 4 true true false hidden

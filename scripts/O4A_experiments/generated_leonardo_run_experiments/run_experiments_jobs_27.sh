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
#SBATCH --job-name=o4a_run_exp_27
#SBATCH --output=logs/run_experiments_leonardo/run_experiments_27_%j.out
#SBATCH --error=logs/run_experiments_leonardo/run_experiments_27_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=l.laraspata3@phd.uniba.it


source .venv/bin/activate

srun -u bash scripts/O4A_experiments/run_experiments.sh GemmaToQwen_HE 420 cuda:0 8 true 4 true true false attn
srun -u bash scripts/O4A_experiments/run_experiments.sh GemmaToQwen_HE 420 cuda:0 8 true 4 true true false mlp
srun -u bash scripts/O4A_experiments/run_experiments.sh GemmaToQwen_HE 420 cuda:0 8 true 4 true true false hidden
srun -u bash scripts/O4A_experiments/run_experiments.sh GemmaToQwen_HE 511 cuda:0 8 true 4 true true false attn
srun -u bash scripts/O4A_experiments/run_experiments.sh GemmaToQwen_HE 511 cuda:0 8 true 4 true true false mlp
srun -u bash scripts/O4A_experiments/run_experiments.sh GemmaToQwen_HE 511 cuda:0 8 true 4 true true false hidden
srun -u bash scripts/O4A_experiments/run_experiments.sh GemmaToFalcon_BBC 2 cuda:0 8 true 4 true true false attn
srun -u bash scripts/O4A_experiments/run_experiments.sh GemmaToFalcon_BBC 2 cuda:0 8 true 4 true true false mlp
srun -u bash scripts/O4A_experiments/run_experiments.sh GemmaToFalcon_BBC 2 cuda:0 8 true 4 true true false hidden
srun -u bash scripts/O4A_experiments/run_experiments.sh GemmaToFalcon_BBC 4 cuda:0 8 true 4 true true false attn
srun -u bash scripts/O4A_experiments/run_experiments.sh GemmaToFalcon_BBC 4 cuda:0 8 true 4 true true false mlp
srun -u bash scripts/O4A_experiments/run_experiments.sh GemmaToFalcon_BBC 4 cuda:0 8 true 4 true true false hidden
srun -u bash scripts/O4A_experiments/run_experiments.sh GemmaToFalcon_BBC 24 cuda:0 8 true 4 true true false attn
srun -u bash scripts/O4A_experiments/run_experiments.sh GemmaToFalcon_BBC 24 cuda:0 8 true 4 true true false mlp
srun -u bash scripts/O4A_experiments/run_experiments.sh GemmaToFalcon_BBC 24 cuda:0 8 true 4 true true false hidden
srun -u bash scripts/O4A_experiments/run_experiments.sh GemmaToFalcon_BBC 42 cuda:0 8 true 4 true true false attn
srun -u bash scripts/O4A_experiments/run_experiments.sh GemmaToFalcon_BBC 42 cuda:0 8 true 4 true true false mlp
srun -u bash scripts/O4A_experiments/run_experiments.sh GemmaToFalcon_BBC 42 cuda:0 8 true 4 true true false hidden
srun -u bash scripts/O4A_experiments/run_experiments.sh GemmaToFalcon_BBC 64 cuda:0 8 true 4 true true false attn
srun -u bash scripts/O4A_experiments/run_experiments.sh GemmaToFalcon_BBC 64 cuda:0 8 true 4 true true false mlp
srun -u bash scripts/O4A_experiments/run_experiments.sh GemmaToFalcon_BBC 64 cuda:0 8 true 4 true true false hidden
srun -u bash scripts/O4A_experiments/run_experiments.sh GemmaToFalcon_BBC 67 cuda:0 8 true 4 true true false attn
srun -u bash scripts/O4A_experiments/run_experiments.sh GemmaToFalcon_BBC 67 cuda:0 8 true 4 true true false mlp
srun -u bash scripts/O4A_experiments/run_experiments.sh GemmaToFalcon_BBC 67 cuda:0 8 true 4 true true false hidden

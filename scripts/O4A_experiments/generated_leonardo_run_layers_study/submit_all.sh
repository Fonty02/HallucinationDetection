#!/bin/bash
# Submit all generated SLURM layer study files

SCRIPT_DIR="scripts/O4A_experiments/generated_leonardo_run_layers_study"

sbatch "$SCRIPT_DIR/run_layers_study_jobs_2.sh"
sbatch "$SCRIPT_DIR/run_layers_study_jobs_3.sh"
sbatch "$SCRIPT_DIR/run_layers_study_jobs_4.sh"
sbatch "$SCRIPT_DIR/run_layers_study_jobs_5.sh"
sbatch "$SCRIPT_DIR/run_layers_study_jobs_6.sh"
sbatch "$SCRIPT_DIR/run_layers_study_jobs_7.sh"
sbatch "$SCRIPT_DIR/run_layers_study_jobs_8.sh"
sbatch "$SCRIPT_DIR/run_layers_study_jobs_9.sh"
sbatch "$SCRIPT_DIR/run_layers_study_jobs_10.sh"
sbatch "$SCRIPT_DIR/run_layers_study_jobs_11.sh"
sbatch "$SCRIPT_DIR/run_layers_study_jobs_12.sh"
sbatch "$SCRIPT_DIR/run_layers_study_jobs_14.sh"
sbatch "$SCRIPT_DIR/run_layers_study_jobs_17.sh"
sbatch "$SCRIPT_DIR/run_layers_study_jobs_18.sh"

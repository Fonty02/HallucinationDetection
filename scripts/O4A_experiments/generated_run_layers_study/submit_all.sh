#!/bin/bash
# Submit all generated HTCondor layer study files

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

condor_submit "$SCRIPT_DIR/run_layers_study_jobs_1.htc"

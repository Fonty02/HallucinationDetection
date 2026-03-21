#!/bin/bash
# Submit all generated HTCondor job files

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

condor_submit "$SCRIPT_DIR/run_experiments_jobs_1.htc"

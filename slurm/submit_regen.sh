#!/bin/bash
# Rerun the pipeline after fixing the summarization prompts, as 4 jobs:
#
#     regen_summ ──┬── b3 (notebook)
#                  └── b4 ──── b5
#
# Run on the LOGIN node from the repository root:
#     bash slurm/submit_regen.sh
#
# B7, B8 and B9 would be a 5th job, so submit them once these have finished:
#     sbatch slurm/b7_b8_b9.sh

set -eo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

submit() { sbatch --parsable --kill-on-invalid-dep=yes "$@" | cut -d';' -f1; }

regen=$(submit slurm/regen_summarization.sh)
b3=$(submit --dependency=afterok:$regen slurm/b3.sh)
b4=$(submit --dependency=afterok:$regen slurm/b4.sh)
b5=$(submit --dependency=afterok:$b4 slurm/b5.sh)

echo "regen_summ $regen  b3 $b3  b4 $b4  b5 $b5"
echo "When they finish: sbatch slurm/b7_b8_b9.sh"

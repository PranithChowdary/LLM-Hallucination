#!/bin/bash
#SBATCH --job-name=labels_b6
#SBATCH --account=cminds_anandi
#SBATCH --partition=cn4_mangala
#SBATCH --qos=mangala
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8   ## max allowd is 8 for cn4_mangala partition
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Provisional labels (GPU), then B6 naive log-prob baseline (CPU, minutes).
# Combined into one job because the QoS allows only 4 submitted jobs per user.
# Outputs: data/processed/generation_labels.parquet, data/b6/
# A B6 failure is reported but does not fail the job, so B3/B4 still start.

source "${SLURM_SUBMIT_DIR:-.}/slurm/env.sh"

echo "== labels started at $(date)"
srun python scripts/label_generations.py "${LIMIT_ARGS[@]}"
echo "== labels finished at $(date)"

echo "== B6 started at $(date)"
if srun python scripts/b6_logprob_baseline.py; then
    echo "== B6 finished at $(date)"
else
    echo "[WARNING] B6 failed (see .err log); resubmit later with: sbatch slurm/b6.sbatch"
fi

job_done

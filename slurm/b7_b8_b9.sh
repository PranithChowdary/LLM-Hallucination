#!/bin/bash
#SBATCH --job-name=b7_b8_b9
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

# Feature correlation, single-pass classifier and sampling budget sweep.
# Needs: A3/B1/B2 features, labels, data/b6/ and data/b4/.
# Outputs: data/b7/, data/b8/, data/b9/

source "${SLURM_SUBMIT_DIR:-.}/slurm/env.sh"

echo "== B7 started at $(date)"
srun python src/eval/b7_feature_correlation.py
echo "== B8 started at $(date)"
srun python src/eval/b8_single_pass_classifier.py
echo "== B9 started at $(date)"
srun python scripts/b9_sampling_budget.py

job_done

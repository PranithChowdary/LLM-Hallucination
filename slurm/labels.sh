#!/bin/bash
#SBATCH --job-name=labels
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

# Provisional hallucination labels for the A2 main answers (NLI-based).
# Output: data/processed/generation_labels.parquet  (needed by B3, B4, B5, B6)

source "${SLURM_SUBMIT_DIR:-.}/slurm/env.sh"

srun python scripts/label_generations.py "${LIMIT_ARGS[@]}"

job_done

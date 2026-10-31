#!/bin/bash
#SBATCH --job-name=b4_semantic_entropy
#SBATCH --account=cminds_anandi
#SBATCH --partition=cn4_mangala
#SBATCH --qos=mangala
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8   ## max allowd is 8 for cn4_mangala partition
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# B4 Semantic Entropy -> data/b4/
# Resumable: if it hits the time limit, resubmit and it skips finished prompts.

source "${SLURM_SUBMIT_DIR:-.}/slurm/env.sh"

srun python scripts/b4_semantic_entropy.py "${LIMIT_ARGS[@]}"

job_done

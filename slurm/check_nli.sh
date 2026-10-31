#!/bin/bash
#SBATCH --job-name=check_nli
#SBATCH --account=cminds_anandi
#SBATCH --partition=cn4_mangala
#SBATCH --qos=mangala
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8   ## max allowd is 8 for cn4_mangala partition
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Sanity check of both NLI models (tokenization round-trip + 5 known pairs).
# Read the result with: cat logs/check_nli_<jobid>.out

source "${SLURM_SUBMIT_DIR:-.}/slurm/env.sh"

srun python scripts/check_nli.py

job_done

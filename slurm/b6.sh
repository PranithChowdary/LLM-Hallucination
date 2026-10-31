#!/bin/bash
#SBATCH --job-name=b6_logprob
#SBATCH --account=cminds_anandi
#SBATCH --partition=cn4_mangala
#SBATCH --qos=mangala
## CPU-only. If the partition rejects jobs without a GPU, add: #SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8   ## max allowd is 8 for cn4_mangala partition
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# B6 naive log-probability threshold baseline -> data/b6/   (CPU, minutes)
# Needs: data/processed/b1_answer_features.parquet, labels.

source "${SLURM_SUBMIT_DIR:-.}/slurm/env.sh"

srun python scripts/b6_logprob_baseline.py

job_done

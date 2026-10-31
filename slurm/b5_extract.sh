#!/bin/bash
#SBATCH --job-name=b5_extract
#SBATCH --account=cminds_anandi
#SBATCH --partition=cn4_mangala
#SBATCH --qos=mangala
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8   ## max allowd is 8 for cn4_mangala partition
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# B5 step 1: hidden states at TBG and SLT for every layer -> data/b5/states/
# Resumable. Independent of the other jobs.

source "${SLURM_SUBMIT_DIR:-.}/slurm/env.sh"

srun python scripts/b5_se_probes.py extract "${LIMIT_ARGS[@]}"

job_done

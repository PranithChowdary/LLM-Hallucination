#!/bin/bash
#SBATCH --job-name=b5_train
#SBATCH --account=cminds_anandi
#SBATCH --partition=cn4_mangala
#SBATCH --qos=mangala
## CPU-only. If the partition rejects jobs without a GPU, add: #SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8   ## max allowd is 8 for cn4_mangala partition
#SBATCH --mem=90G   ## partition max is 94.4 GB per GPU
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# B5 step 2: train SEPs + accuracy probes -> data/b5/
# Needs: data/b4/scores.parquet, data/b5/states/, labels; uses data/b6 if present.

source "${SLURM_SUBMIT_DIR:-.}/slurm/env.sh"

srun python scripts/b5_se_probes.py train

job_done

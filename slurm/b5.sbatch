#!/bin/bash
#SBATCH --job-name=b5_probes
#SBATCH --account=cminds_anandi
#SBATCH --partition=cn4_mangala
#SBATCH --qos=mangala
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8   ## max allowd is 8 for cn4_mangala partition
#SBATCH --mem=90G   ## partition max is 94.4 GB per GPU
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# B5 Semantic Entropy Probes: hidden-state extraction (GPU, resumable),
# then probe training (CPU). Combined into one job for the 4-job QoS limit.
# Needs: data/b4/scores.parquet and labels; uses data/b6 if present.
# Outputs: data/b5/

source "${SLURM_SUBMIT_DIR:-.}/slurm/env.sh"

echo "== B5 extract started at $(date)"
srun python scripts/b5_se_probes.py extract "${LIMIT_ARGS[@]}"
echo "== B5 train started at $(date)"
srun python scripts/b5_se_probes.py train

job_done

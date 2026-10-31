#!/bin/bash
#SBATCH --job-name=download_models
#SBATCH --account=cminds_anandi
#SBATCH --partition=cn3_anandi
#SBATCH --qos=anandi
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8   ## max allowd is 8 for cn4_mangala partition
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source ~/miniconda3/bin/activate

cd /users/staff/pranith/LLM_Hallucinations/
mkdir -p logs

echo "Job Started at: $(date)"
echo ""
srun python scripts/download_model.py
echo ""
srun python scripts/download_signal_models.py
echo ""
echo "Job Ended at: $(date)"

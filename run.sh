#!/bin/bash
#SBATCH --job-name=a3
#SBATCH --account=cminds_anandi
#SBATCH --partition=cn4_mangala
#SBATCH --qos=mangala
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8   ## max allowd is 8 for cn4_mangala partition
#SBATCH --mem=188G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source ~/miniconda3/bin/activate

cd /users/staff/pranith/LLM_Hallucinations/
mkdir -p logs

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "Job Started at: $(date)"
echo ""
srun python src/extract_features/extract_token_features.py
echo "Feature extraction completed at: $(date)"
echo ""
srun python scripts/finalize_a3.py
echo ""
echo "Job Ended at: $(date)"

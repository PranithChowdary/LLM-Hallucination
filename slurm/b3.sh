#!/bin/bash
#SBATCH --job-name=b3_selfcheck
#SBATCH --account=cminds_anandi
#SBATCH --partition=cn4_mangala
#SBATCH --qos=mangala
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8   ## max allowd is 8 for cn4_mangala partition
#SBATCH --mem=90G   ## partition max is 94.4 GB per GPU
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# B3 SelfCheckGPT. Executes the notebook in place: every table and figure
# ends up in the output cells of notebooks/b3_selfcheckgpt.ipynb.
# Needs: data/processed/generation_labels.parquet (labels job).

source "${SLURM_SUBMIT_DIR:-.}/slurm/env.sh"

export B3_LIMIT="${LIMIT:-}"   # read by the notebook's config cell

srun jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 \
    --ExecutePreprocessor.kernel_name=python3 \
    notebooks/b3_selfcheckgpt.ipynb

job_done

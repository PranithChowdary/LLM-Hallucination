#!/bin/bash
#SBATCH --job-name=regen_summ
#SBATCH --account=cminds_anandi
#SBATCH --partition=cn4_mangala
#SBATCH --qos=mangala
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8   ## max allowd is 8 for cn4_mangala partition
#SBATCH --mem=90G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Regenerate summarization answers with the document in the prompt, then
# redo everything that reads the answers but not B3-B5: A3, B1, B2, labels, B6.
# One job because the QoS allows only 4 submitted jobs per user.
# With LIMIT=<n> it only generates n test prompts and changes no data file.

source "${SLURM_SUBMIT_DIR:-.}/slurm/env.sh"

echo "== regeneration started at $(date)"
srun python src/generation/regenerate_summarization.py "${LIMIT_ARGS[@]}"

if [ -n "${LIMIT:-}" ]; then
    job_done
    exit 0
fi

# These folders resume by uid and would keep the old summarization results.
mkdir -p data/pre_regen
for dir in data/a3/token_features data/a3/hidden_states data/b4/shards data/b5/states data/b9/shards; do
    if [ -d "$dir" ]; then
        mv "$dir" "data/pre_regen/$(echo "$dir" | tr / _)_${SLURM_JOB_ID}"
    fi
done

echo "== A3 started at $(date)"
srun python src/extract_features/extract_token_features.py
srun python scripts/finalize_a3.py

echo "== B1 and B2 started at $(date)"
srun python scripts/b1_features.py
srun python src/extract_features/extract_linguistic_features.py

echo "== labels started at $(date)"
srun python scripts/label_generations.py

echo "== B6 started at $(date)"
srun python scripts/b6_logprob_baseline.py

job_done

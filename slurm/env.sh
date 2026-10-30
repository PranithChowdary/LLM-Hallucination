# Sourced by every slurm/*.sbatch job. Not meant to be run directly.
#
# Jobs must be submitted from the repository root (slurm/submit_all.sh
# does this for you), because SLURM_SUBMIT_DIR is used as the project dir.

set -eo pipefail

PROJECT="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [ ! -f "$PROJECT/src/signals/common.py" ]; then
    echo "[ERROR] Submit from the repository root (got: $PROJECT)" >&2
    exit 1
fi
cd "$PROJECT"

# Same environment as run.sh. Set CONDA_ENV to use a named env instead of base.
source ~/miniconda3/bin/activate ${CONDA_ENV:-}

# Compute nodes are offline; every model must already be in models/.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"

# LIMIT=<n> (set by `submit_all.sh --test`) runs each step on n prompts only.
LIMIT_ARGS=()
if [ -n "${LIMIT:-}" ]; then
    LIMIT_ARGS=(--limit "$LIMIT")
fi

echo "=================================================================="
echo "Job:        ${SLURM_JOB_NAME:-local} (${SLURM_JOB_ID:-no id})"
echo "Node:       $(hostname)"
echo "Started:    $(date)"
echo "Project:    $PROJECT"
echo "Git commit: $(git rev-parse --short HEAD 2>/dev/null || echo 'n/a')"
echo "Python:     $(which python)"
echo "Limit:      ${LIMIT:-none (full run)}"
if command -v nvidia-smi >/dev/null 2>&1 && [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi
echo "=================================================================="

job_done() {
    echo "=================================================================="
    echo "Finished:   $(date)"
    echo "=================================================================="
}

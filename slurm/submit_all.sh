#!/bin/bash
# Submit the whole B3-B6 pipeline as a SLURM dependency chain.
# Run on the LOGIN node from anywhere inside the repo:
#
#     bash slurm/submit_all.sh                 # full run
#     bash slurm/submit_all.sh --test          # every step on 50 prompts (catch errors fast)
#     bash slurm/submit_all.sh --skip-checks   # skip slurm/preflight.sh
#
# The mangala QoS allows at most 4 submitted (running + pending) jobs per
# user, so the pipeline is packed into exactly 4 jobs:
#
#     labels_b6 ──┬── b3 (notebook)
#                 └── b4 ──── b5 (extract + train)
#
# afterok = a job only starts if its parent succeeded; if a parent fails,
# its children are cancelled (--kill-on-invalid-dep). B4 and B5-extract are
# resumable: resubmit the same file and they continue.

set -eo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

MAX_JOBS="${MAX_JOBS:-4}"   # QOSMaxSubmitJobPerUserLimit on cn4_mangala
N_NEW=4

TEST=0
CHECKS=1
for arg in "$@"; do
    case "$arg" in
        --test) TEST=1 ;;
        --skip-checks) CHECKS=0 ;;
        *) echo "usage: bash slurm/submit_all.sh [--test] [--skip-checks]"; exit 1 ;;
    esac
done

queued=$(squeue -h -u "$USER" | wc -l)
if [ $((queued + N_NEW)) -gt "$MAX_JOBS" ]; then
    echo "[ERROR] You already have $queued job(s) queued/running; this needs $N_NEW more"
    echo "        and the QoS allows $MAX_JOBS in total. Wait for them or cancel them:"
    squeue -u "$USER"
    echo "        scancel -u $USER     # cancels ALL your jobs"
    exit 1
fi

if [ $CHECKS -eq 1 ]; then
    bash slurm/preflight.sh
    echo
fi

EXPORT="ALL"
if [ $TEST -eq 1 ]; then
    EXPORT="ALL,LIMIT=50"
    echo ">>> TEST MODE: every step runs on 50 prompts."
    echo ">>> Outputs go to the normal folders and are overwritten / resumed by the full run."
    echo
fi

# submit <var> <sbatch args...>: stores the job id in $<var>. If sbatch refuses
# a job, every job this run already submitted is cancelled, so nothing is
# left half-queued.
SUBMITTED=()
submit() {
    local var="$1"; shift
    local id
    if ! id=$(sbatch --parsable --kill-on-invalid-dep=yes --export="$EXPORT" "$@"); then
        echo "[ERROR] sbatch refused: $*" >&2
        if [ ${#SUBMITTED[@]} -gt 0 ]; then
            echo "[ERROR] cancelling the jobs this run already submitted: ${SUBMITTED[*]}" >&2
            scancel "${SUBMITTED[@]}"
        fi
        exit 1
    fi
    id="${id%%;*}"
    SUBMITTED+=("$id")
    printf -v "$var" "%s" "$id"
}

submit labels slurm/labels_b6.sbatch
submit b3 --dependency=afterok:$labels slurm/b3.sbatch
submit b4 --dependency=afterok:$labels slurm/b4.sbatch
submit b5 --dependency=afterok:$b4 slurm/b5.sbatch

record="logs/submitted_$(date +%Y%m%d_%H%M%S).txt"
{
    echo "mode: $([ $TEST -eq 1 ] && echo test || echo full)   git: $(git rev-parse --short HEAD 2>/dev/null || echo n/a)"
    printf "%-10s %s\n" labels_b6 "$labels" b3 "$b3" b4 "$b4" b5 "$b5"
} | tee "$record"

cat <<EOF

Submitted. Job ids saved to $record
  watch queue : squeue -u \$USER
  follow logs : tail -f logs/labels_b6_${labels}.out
  job history : sacct -j ${labels},${b3},${b4},${b5} --format=JobID,JobName%20,State,Elapsed,MaxRSS
  cancel all  : scancel ${labels} ${b3} ${b4} ${b5}
EOF

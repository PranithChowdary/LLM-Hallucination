#!/bin/bash
#SBATCH --job-name=demo_circuits
#SBATCH --account=cminds_anandi
#SBATCH --partition=cn4_mangala
#SBATCH --qos=mangala
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8   ## max allowd is 8 for cn4_mangala partition
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Pre-compute the Circuit Tracer demo page: 10 HaluEval examples with
# detector scores, SelfCheck-NLI per sentence and attribution graphs.
# Needs: data/b4, data/b5, data/b6, labels, extra15 samples, models/.
# Output: demo/circuit_examples.json
#
# Hand-pick examples instead of the automatic selection:
#   sbatch --export=ALL,DEMO_UIDS="halueval-qa-00012 halueval-qa-00440" slurm/demo_circuits.sbatch

source "${SLURM_SUBMIT_DIR:-.}/slurm/env.sh"

if [ -n "${DEMO_UIDS:-}" ]; then
    srun python demo/build_circuit_examples.py --uids $DEMO_UIDS
else
    srun python demo/build_circuit_examples.py
fi

job_done

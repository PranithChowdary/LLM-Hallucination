# Running B3–B6 on the cluster

Server: `pranith@kalpana.minds.iitb.ac.in`, project dir `/users/staff/pranith/LLM_Hallucinations`.

`slurm/submit_all.sh` submits **4 jobs**, because the `mangala` QoS allows at
most 4 submitted (running + pending) jobs per user:

| Job | File | GPU | Writes to |
|---|---|---|---|
| labels + B6 | `slurm/labels_b6.sbatch` | yes | `data/processed/generation_labels.parquet`, `data/b6/` |
| B3 | `slurm/b3.sbatch` | yes | output cells of `notebooks/b3_selfcheckgpt.ipynb` |
| B4 | `slurm/b4.sbatch` | yes | `data/b4/` |
| B5 (extract + train) | `slurm/b5.sbatch` | yes | `data/b5/` |

```
labels_b6 ──┬── b3 (notebook)
            └── b4 ──── b5
```

Single-step files for resubmitting just one piece: `labels.sbatch`, `b6.sbatch`,
`b5_extract.sbatch`, `b5_train.sbatch`.

---

## 1. Copy the code to the server (on your Mac, from the repo root)

```bash
rsync -avR requirements.txt slurm/ notebooks/b3_selfcheckgpt.ipynb scripts/b4_semantic_entropy.py scripts/b5_se_probes.py scripts/b6_logprob_baseline.py scripts/label_generations.py scripts/download_signal_models.py src/eval/ src/signals/ tests/test_signals.py tests/test_eval_metrics.py pranith@kalpana.minds.iitb.ac.in:/users/staff/pranith/LLM_Hallucinations/
```

## 2. One-time setup (on the server's login node)

```bash
ssh pranith@kalpana.minds.iitb.ac.in
```
```bash
cd /users/staff/pranith/LLM_Hallucinations && source ~/miniconda3/bin/activate
```
```bash
pip install -r requirements.txt
```
```bash
python -m ipykernel install --user --name python3
```
Download the NLI / BERTScore models. The login node has internet; compute nodes don't:
```bash
python scripts/download_signal_models.py
```
If the A3 token features are still only zipped, unzip them. The B3 grey-box baselines need them:
```bash
unzip -n data/a3/token_features.zip -d data/a3/
```
Optional: run the unit tests (pure Python, no models, takes seconds):
```bash
python -m pytest tests/test_signals.py tests/test_eval_metrics.py -q
```

## 3. Check everything is in place

```bash
bash slurm/preflight.sh
```
Fix every `[FAIL]` line. The most likely one is a missing
`data/generation_qwen_calibration_extra15.parquet`: B3, B4 and B5 need
those 15 samples per prompt from `src/generation/generate_extra.py`.

## 4. Test run first (50 prompts, all steps, ~30–60 min)

```bash
bash slurm/submit_all.sh --test
```
Watch it:
```bash
squeue -u $USER
```
```bash
tail -f logs/labels_b6_*.out
```
When every job has finished, check none failed:
```bash
sacct -u $USER -S today --format=JobID,JobName%22,State,Elapsed,MaxRSS
```
Errors are in `logs/<job-name>_<job-id>.err`.

## 5. Full run

```bash
bash slurm/submit_all.sh
```
The test outputs are overwritten. B4 and B5-extract simply continue from
the shards the test run already wrote.

## 6. Resubmitting one step

Submit from the repo root, since the jobs use it as their working directory:
```bash
cd /users/staff/pranith/LLM_Hallucinations && sbatch slurm/b4.sbatch
```
- **B4 / B5-extract hit the 24 h limit:** resubmit the same file. They resume from their shards.
- **A step failed:** jobs that depended on it are cancelled automatically. Fix the problem, resubmit that step, then resubmit its dependents. To chain them by hand:
```bash
sbatch --dependency=afterok:<JOBID> slurm/b5.sbatch
```
- **Test mode for a single step:**
```bash
sbatch --export=ALL,LIMIT=50 slurm/b4.sbatch
```
- **Cancel everything:**
```bash
scancel -u $USER
```

## 7. Copy results back (on your Mac, from the repo root)

```bash
rsync -av --exclude 'states/' --exclude 'shards/' pranith@kalpana.minds.iitb.ac.in:/users/staff/pranith/LLM_Hallucinations/data/b4 pranith@kalpana.minds.iitb.ac.in:/users/staff/pranith/LLM_Hallucinations/data/b5 pranith@kalpana.minds.iitb.ac.in:/users/staff/pranith/LLM_Hallucinations/data/b6 data/
```
```bash
rsync -av pranith@kalpana.minds.iitb.ac.in:/users/staff/pranith/LLM_Hallucinations/notebooks/b3_selfcheckgpt.ipynb pranith@kalpana.minds.iitb.ac.in:/users/staff/pranith/LLM_Hallucinations/data/processed/generation_labels.parquet ./tmp_results/
```
(`states/` holds the ~2 GB of B5 hidden states and `shards/` the B4 resume
files; neither is needed locally.) Then move `tmp_results/b3_selfcheckgpt.ipynb`
to `notebooks/` and `tmp_results/generation_labels.parquet` to `data/processed/`.

---

### Notes

- **Job limit:** `submit_all.sh` refuses to submit if you already have jobs queued, since 4 more would exceed the QoS limit. Override with `MAX_JOBS=<n>` if the limit changes.
- **Resources** (edit the `#SBATCH` lines if the queue complains): every job asks for 1 GPU, 8 CPUs and 64–128 GB, for 8–24 h. The single-step `b5_train` and `b6` files ask for no GPU; if the partition refuses that, add `#SBATCH --gres=gpu:1`.
- **Conda env:** jobs use `~/miniconda3` base, like `run.sh`. To use a named env:
```bash
CONDA_ENV=myenv bash slurm/submit_all.sh
```
- **Job ids** of every submission are saved in `logs/submitted_<timestamp>.txt`.

#!/usr/bin/env python
"""B9: incremental Semantic Entropy sampling-budget sweep.

Reuses B4's saved 15-answer NLI argmax matrices and sample likelihoods.
Adds the A2 main answer as answer 0, computes only its 30 directed NLI
comparisons against the 15 saved samples, and teacher-forced-scores only
the main answer. B4 files are read-only and never overwritten.

Answer order: [main_answer, sample_1, ..., sample_15].
Budgets: N = 1, 2, 4, 8, 16.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

from src.signals.common import NLIModel, load_labels, load_responses
from src.signals.semantic_entropy import (
    SequenceScorer,
    cluster_ids,
    discrete_semantic_entropy,
    naive_entropy,
    semantic_entropy,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
B4_SCORES = PROJECT_ROOT / "data" / "b4" / "scores.parquet"
OUTPUT_DIR = PROJECT_ROOT / "data" / "b9"
BUDGETS = [1, 2, 4, 8, 16]
NLI_LOCAL = PROJECT_ROOT / "models" / "deberta-v2-xlarge-mnli"
LLM_LOCAL = PROJECT_ROOT / "models" / "Qwen2.5-0.5B-Instruct"
NLI_HUB = "microsoft/deberta-v2-xlarge-mnli"
LLM_HUB = "Qwen/Qwen2.5-0.5B-Instruct"



def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda", help="Use cuda for the original GPU run.")
    p.add_argument("--nli-model", type=str, default=str(NLI_LOCAL),
                   help="Path to the original DeBERTa NLI checkpoint.")
    p.add_argument("--llm", type=str, default=str(LLM_LOCAL),
                   help="Path to the original Qwen checkpoint.")
    p.add_argument("--nli-batch-size", type=int, default=64)
    p.add_argument("--checkpoint-size", type=int, default=200)
    p.add_argument("--limit", type=int, default=None,
                   help="Optional small pilot; rerun without it to finish.")
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return p.parse_args()


def reconstruct_matrix(old_flat):
    """Restore B4's 15x15 predicted labels as one-hot 3-class probabilities."""
    labels = np.asarray(old_flat, dtype=np.int64)
    if labels.size != 15 * 15 or np.any((labels < 0) | (labels > 2)):
        raise ValueError(f"Expected 225 NLI labels in [0,1,2], got {labels.size}.")
    return np.eye(3, dtype=np.float32)[labels.reshape(15, 15)]


def make_metrics(scores: pd.DataFrame, labels: pd.DataFrame):
    merged = scores.merge(labels, on="uid", how="inner")
    merged = merged.dropna(subset=["hallucinated"])
    merged["hallucinated"] = merged["hallucinated"].astype(int)

    score_cols = [
        c for c in scores.columns
        if re.match(r"^(semantic_entropy|discrete_semantic_entropy|naive_entropy|n_clusters)(?:_n\d+)?$", c)
        or re.match(r"^(semantic_entropy|discrete_semantic_entropy|n_clusters)_nonstrict(?:_n\d+)?$", c)
    ]
    rows = []
    for subset in ["all"] + sorted(merged["subset"].dropna().unique().tolist()):
        part = merged if subset == "all" else merged[merged["subset"] == subset]
        y = part["hallucinated"].to_numpy()
        for col in score_cols:
            x = part[col].to_numpy(dtype=float)
            valid = np.isfinite(x)
            yy, xx = y[valid], x[valid]
            if len(yy) == 0 or len(np.unique(yy)) < 2:
                auc = ap = np.nan
            else:
                auc = float(roc_auc_score(yy, xx))
                ap = float(average_precision_score(yy, xx))
            rows.append({
                "subset": subset,
                "score": col,
                "n_labeled": int(len(yy)),
                "positive_fraction": float(yy.mean()) if len(yy) else np.nan,
                "auroc": auc,
                "auprc": ap,
            })
    metrics = pd.DataFrame(rows)
    parsed = metrics["score"].str.extract(
        r"^(?P<method>.*?)(?:_n(?P<n>\d+))?$"
    )
    metrics["method"] = parsed["method"]
    metrics["n_samples"] = parsed["n"].astype(float).fillna(16)
    return metrics


def main():
    args = parse_args()
    started = time.time()
    if args.device.startswith("cuda"):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. Run this script in the same GPU environment "
                "used for B4; do not switch models or run the full job on CPU."
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = args.output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    if not B4_SCORES.exists():
        raise FileNotFoundError(f"Cannot find existing B4 scores: {B4_SCORES}")

    print("[1/5] Loading the saved B4 scores and original response files...")
    b4 = pd.read_parquet(B4_SCORES)
    b4["uid"] = b4["uid"].astype(str)
    responses = load_responses(n_samples=15)
    responses["uid"] = responses["uid"].astype(str)

    if b4["uid"].duplicated().any() or responses["uid"].duplicated().any():
        raise ValueError("Duplicate UIDs found; refusing to combine rows.")
    if set(b4["uid"]) != set(responses["uid"]):
        raise ValueError(
            f"UID mismatch: B4={len(b4)}, responses={len(responses)}, "
            f"intersection={len(set(b4.uid) & set(responses.uid))}."
        )

    b4_by_uid = b4.set_index("uid")
    responses = responses.reset_index(drop=True)
    b4_aligned = b4_by_uid.loc[responses["uid"]].reset_index()
    if not (b4_aligned["n_answers"].astype(int) == 15).all():
        raise ValueError("B4 does not contain exactly 15 answers for every prompt.")

    done = set()
    for path in shard_dir.glob("part_*.parquet"):
        done.update(pd.read_parquet(path, columns=["uid"])["uid"].astype(str))
    todo = responses[~responses["uid"].isin(done)].reset_index(drop=True)
    if args.limit is not None:
        todo = todo.head(args.limit).copy()
    print(f"Prompts total: {len(responses):,}; already completed: {len(done):,}; remaining this run: {len(todo):,}")

    if len(todo):
        nli_path = args.nli_model
        llm_path = args.llm

        print(f"[2/5] Loading original NLI checkpoint: {nli_path}")
        nli = NLIModel(nli_path, device=args.device, batch_size=args.nli_batch_size)

        print(f"[3/5] Loading original LLM checkpoint: {llm_path}")
        scorer = SequenceScorer(llm_path, device=args.device)

        pending = []
        for start in tqdm(range(0, len(todo), 8), desc="B9 incremental scoring"):
            batch = todo.iloc[start:start + 8]
            batch_b4 = b4_by_uid.loc[batch["uid"]]

            # Only 30 new directed comparisons per prompt: main<->each sample.
            premises, hypotheses, pair_indices = [], [], []
            matrices = []
            for local_i, (row, old_row) in enumerate(zip(batch.itertuples(index=False), batch_b4.itertuples())):
                samples = list(row.samples)
                if len(samples) != 15:
                    raise ValueError(f"UID {row.uid} has {len(samples)} samples, expected 15.")
                matrix = np.zeros((16, 16, 3), dtype=np.float32)
                matrix[1:, 1:, :] = reconstruct_matrix(old_row.nli_argmax)
                matrix[np.arange(16), np.arange(16), 2] = 1.0
                matrices.append(matrix)

                texts = [f"{row.prompt} {row.main_answer}"] + [
                    f"{row.prompt} {answer}" for answer in samples
                ]
                for j in range(1, 16):
                    premises.extend([texts[0], texts[j]])
                    hypotheses.extend([texts[j], texts[0]])
                    pair_indices.extend([(local_i, 0, j), (local_i, j, 0)])

            new_probs = nli.probs(premises, hypotheses)
            if len(new_probs) != len(pair_indices):
                raise RuntimeError("NLI returned an unexpected number of predictions.")
            for (local_i, i, j), probs in zip(pair_indices, new_probs):
                matrices[local_i][i, j] = probs

            for row, matrix in zip(batch.itertuples(index=False), matrices):
                # Score only the A2 main answer. The other 15 likelihoods are reused.
                main_sum, main_tokens = scorer.score(row.prompt, [row.main_answer])
                old = b4_by_uid.loc[row.uid]
                ll_sum = np.concatenate([
                    np.asarray([main_sum[0]], dtype=float),
                    np.asarray(old.loglik_sum, dtype=float),
                ])
                ll_tokens = np.concatenate([
                    np.asarray([main_tokens[0]], dtype=int),
                    np.asarray(old.loglik_tokens, dtype=int),
                ])
                if len(ll_sum) != 16 or len(ll_tokens) != 16 or np.any(ll_tokens <= 0):
                    raise ValueError(f"Invalid 16-answer likelihood arrays for UID {row.uid}.")
                ll_norm = ll_sum / ll_tokens

                result = {
                    "uid": row.uid,
                    "dataset": row.dataset,
                    "subset": row.subset,
                    "n_answers": 16,
                    "nli_argmax": matrix.argmax(axis=-1).astype(np.int8).ravel().tolist(),
                    "loglik_sum": ll_sum.tolist(),
                    "loglik_tokens": ll_tokens.tolist(),
                }

                for strict in (True, False):
                    suffix = "" if strict else "_nonstrict"
                    ids = cluster_ids(matrix, strict=strict)
                    result[f"cluster_ids{suffix}"] = ids
                    for n in BUDGETS:
                        tag = "" if n == 16 else f"_n{n}"
                        ids_n = ids[:n]
                        result[f"discrete_semantic_entropy{suffix}{tag}"] = discrete_semantic_entropy(ids_n)
                        result[f"n_clusters{suffix}{tag}"] = int(len(set(ids_n)))
                        if strict:
                            result[f"naive_entropy{tag}"] = naive_entropy(ll_norm[:n])
                        result[f"semantic_entropy{suffix}{tag}"] = semantic_entropy(ids_n, ll_norm[:n])
                pending.append(result)

            if len(pending) >= args.checkpoint_size:
                existing = sorted(shard_dir.glob("part_*.parquet"))
                next_num = max([int(p.stem.split("_")[-1]) for p in existing] or [0]) + 1
                out_path = shard_dir / f"part_{next_num:06d}.parquet"
                tmp_path = out_path.with_suffix(".tmp")
                pd.DataFrame(pending).to_parquet(tmp_path, index=False)
                tmp_path.replace(out_path)
                pending = []

        if pending:
            existing = sorted(shard_dir.glob("part_*.parquet"))
            next_num = max([int(p.stem.split("_")[-1]) for p in existing] or [0]) + 1
            out_path = shard_dir / f"part_{next_num:06d}.parquet"
            tmp_path = out_path.with_suffix(".tmp")
            pd.DataFrame(pending).to_parquet(tmp_path, index=False)
            tmp_path.replace(out_path)

    print("[4/5] Combining B9 checkpoints...")
    shard_paths = sorted(shard_dir.glob("part_*.parquet"))
    if not shard_paths:
        raise RuntimeError("No B9 output shards found.")
    scores = pd.concat([pd.read_parquet(p) for p in shard_paths], ignore_index=True)
    scores = scores.drop_duplicates("uid", keep="last")
    scores = scores.sort_values("uid").reset_index(drop=True)
    scores.to_parquet(args.output_dir / "scores.parquet", index=False)

    labels = load_labels()
    if labels is not None:
        metrics = make_metrics(scores, labels)
        metrics.to_csv(args.output_dir / "metrics.csv", index=False)
        sweep = metrics[metrics["subset"] == "all"].copy()
        sweep.to_csv(args.output_dir / "budget_sweep.csv", index=False)
        print("\nB9 budget sweep (all labelled calibration prompts):")
        print(sweep[["method", "n_samples", "n_labeled", "positive_fraction", "auroc", "auprc"]].to_string(index=False))

    config = {
        "experiment": "B9-incremental-semantic-entropy-budget-sweep",
        "source_b4_scores": str(B4_SCORES),
        "n_prompts": int(len(scores)),
        "budgets": BUDGETS,
        "answer_order": ["A2 main answer", "15 saved extra samples in response_index order"],
        "nli_checkpoint": args.nli_model,
        "llm_checkpoint": args.llm,
        "nli_reused_pairs_per_prompt": 225,
        "nli_new_directed_pairs_per_prompt": 30,
        "loglik_reused_answers_per_prompt": 15,
        "loglik_new_answers_per_prompt": 1,
        "condition_on_question": True,
        "label_caveat": "Labels are provisional; metrics are exploratory calibration results, not held-out test performance.",
        "elapsed_seconds_this_run": round(time.time() - started, 2),
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"\n[SUCCESS] B9 outputs saved under: {args.output_dir}")
    print(f"Rows: {len(scores):,}; elapsed this run: {(time.time() - started) / 3600:.2f} hours")


if __name__ == "__main__":
    main()

"""
B4: Reproduce Semantic Entropy (NLI clustering, then entropy over clusters).

Run from the repository root:
    python scripts/b4_semantic_entropy.py
    python scripts/b4_semantic_entropy.py --limit 50          # smoke test
    python scripts/b4_semantic_entropy.py --no-loglik         # discrete SE only

Inputs:
    data/generation_qwen_calibration.parquet           main answer (A2)
    data/generation_qwen_calibration_extra15.parquet   15 samples / prompt
    data/processed/generation_labels.parquet           optional, for AUROC

Outputs (data/b4/):
    shards/part_*.parquet   resumable per-prompt results, including the
                            M x M NLI argmax matrix (enough to re-cluster
                            strict / non-strict without re-running NLI)
    scores.parquet          one row per uid
    metrics.csv             AUROC (bootstrap CI) / AUPRC / AURAC (if labels)
    budget_sweep.csv        the same metrics for N = 1, 2, 4, 8, M samples
    figures/*.png           paper-style figures (see make_figures)
    config.json

Score columns (higher = more likely hallucinated):
    semantic_entropy            Rao estimator, length-normalized log-liks
    discrete_semantic_entropy   cluster-count version (Farquhar et al. 2024)
    naive_entropy               no clustering (baseline)
    n_clusters
  plus the same for the first N samples, N in BUDGETS (feeds B9), and
  *_nonstrict variants of the clustering.

Method details and departures from the paper: src/signals/semantic_entropy.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval import plots  # noqa: E402
from src.eval.metrics import evaluate_scores  # noqa: E402
from src.signals.common import (  # noqa: E402
    DATA_DIR,
    LLM_DIR,
    MODELS_DIR,
    NLIModel,
    ShardWriter,
    git_commit_sha,
    load_labels,
    load_responses,
)
from src.signals.semantic_entropy import (  # noqa: E402
    SequenceScorer,
    cluster_ids,
    discrete_semantic_entropy,
    naive_entropy,
    nli_pair_matrices,
    semantic_entropy,
)


OUTPUT_DIR = DATA_DIR / "b4"

# Farquhar et al. (2024) use DeBERTa-v2-xlarge-MNLI for clustering.
# Kuhn et al. (2023) used deberta-large-mnli (faster, ~3x smaller).
DEFAULT_NLI = MODELS_DIR / "deberta-v2-xlarge-mnli"
DEFAULT_NLI_HUB = "microsoft/deberta-v2-xlarge-mnli"

# Budgets for the nested B9 sweep. The full M is always added.
BUDGETS = [1, 2, 4, 8]

PROMPTS_PER_SHARD = 200
PROMPTS_PER_BATCH = 8


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--n-samples", type=int, default=15)
    parser.add_argument(
        "--include-main",
        action="store_true",
        help="Cluster the A2 main answer together with the samples (M = 16).",
    )
    parser.add_argument("--nli-model", default=None)
    parser.add_argument("--llm", default=str(LLM_DIR))
    parser.add_argument(
        "--no-loglik",
        action="store_true",
        help="Skip LLM re-scoring; only discrete SE is computed.",
    )
    parser.add_argument(
        "--no-question",
        action="store_true",
        help="Do not prepend the question before NLI (paper default: prepend).",
    )
    parser.add_argument("--subsets", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--nli-batch-size", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def resolve_nli(path: str | None) -> str:
    if path:
        return path
    return str(DEFAULT_NLI) if DEFAULT_NLI.exists() else DEFAULT_NLI_HUB


def entropies(matrix, ll_norm, budgets, strict) -> dict:
    """All entropy scores for the first n answers, n in budgets."""
    suffix = "" if strict else "_nonstrict"
    ids_full = cluster_ids(matrix, strict=strict)
    out = {f"cluster_ids{suffix}": ids_full}

    for n in budgets:
        ids = ids_full[:n]  # Algorithm 1 is prefix-consistent
        tag = "" if n == budgets[-1] else f"_n{n}"
        out[f"discrete_semantic_entropy{suffix}{tag}"] = discrete_semantic_entropy(ids)
        out[f"n_clusters{suffix}{tag}"] = int(len(set(ids)))
        if ll_norm is not None:
            out[f"semantic_entropy{suffix}{tag}"] = semantic_entropy(ids, ll_norm[:n])
            if strict:
                out[f"naive_entropy{tag}"] = naive_entropy(ll_norm[:n])
    return out


# -------------------------------------------------------------------
# Figures (Farquhar et al. 2024 Fig. 2-style; Kuhn et al. 2023 Fig. 3)
# -------------------------------------------------------------------

DISPLAY = {
    "semantic_entropy": "Semantic entropy",
    "discrete_semantic_entropy": "Discrete semantic entropy",
    "naive_entropy": "Naive entropy",
    "n_clusters": "Number of clusters",
    "semantic_entropy_nonstrict": "Semantic entropy (non-strict)",
    "discrete_semantic_entropy_nonstrict": "Discrete SE (non-strict)",
}


def budget_table(metrics):
    """Rows of metrics.csv re-keyed by (method, n_samples) for the B9 sweep."""
    rows = metrics[metrics["subset"] == "all"].copy()
    parsed = rows["score"].str.extract(r"^(?P<method>.*?)(?:_n(?P<n>\d+))?$")
    rows["method"] = parsed["method"]
    rows["n_samples"] = parsed["n"].astype(float)
    return rows.drop(columns="subset")


def make_figures(scores, labels, metrics, sweep, out_dir):
    plt = plots.setup(headless=True)
    main = [m for m in ["semantic_entropy", "discrete_semantic_entropy", "naive_entropy", "n_clusters"] if m in scores]
    colors = plots.ColorMap([DISPLAY[m] for m in main])

    merged = scores.merge(labels, on="uid").dropna(subset=["hallucinated"])
    y = merged["hallucinated"].to_numpy().astype(int)
    curves = {DISPLAY[m]: merged[m].to_numpy(dtype=float) for m in main}

    # 1. AUROC / AURAC bars, all subsets and per subset
    overall = metrics[metrics["score"].isin(main)].copy()
    overall["score"] = overall["score"].map(DISPLAY)
    subsets = ["all"] + sorted(s for s in overall["subset"].unique() if s != "all")
    for metric, chance in [("auroc", 0.5), ("aurac", None)]:
        fig, axes = plt.subplots(1, len(subsets), figsize=(3.3 * len(subsets), 2.9), sharex=True, squeeze=False)
        for ax, s in zip(axes[0], subsets):
            plots.metric_bars(ax, overall[overall["subset"] == s], metric, colors,
                              ci=("auroc_ci_low", "auroc_ci_high") if metric == "auroc" else None,
                              chance=chance)
            ax.set_title(s)
            if ax is not axes[0][0]:
                ax.set_yticklabels([])
        fig.suptitle(f"B4 {metric.upper()} by subset", fontweight="bold")
        fig.tight_layout()
        plots.save(fig, out_dir / f"{metric}_by_subset.png")
        plt.close(fig)

    # 2. ROC + PR, and rejection-accuracy
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    plots.roc_pr_panels(axes, curves, y, colors, " — semantic entropy")
    fig.tight_layout()
    plots.save(fig, out_dir / "roc_pr.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    plots.rejection_panel(ax, curves, y, colors)
    fig.tight_layout()
    plots.save(fig, out_dir / "rejection_accuracy.png")
    plt.close(fig)

    # 3. AUROC vs number of samples (nested prefixes; feeds B9)
    n_full = int(scores["n_answers"].max())
    sweep = sweep.copy()
    sweep["n_samples"] = sweep["n_samples"].fillna(n_full)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, metric in zip(axes, ["auroc", "aurac"]):
        for m in [m for m in main if m != "n_clusters"]:
            d = sweep[sweep["method"] == m].sort_values("n_samples")
            ax.plot(d["n_samples"], d[metric], marker="o", color=colors[DISPLAY[m]], label=DISPLAY[m])
        ax.set(xscale="log", xlabel="Number of sampled answers N", ylabel=metric.upper(),
               title=f"{metric.upper()} vs N")
        ticks = sorted(sweep["n_samples"].unique())
        ax.set_xticks(ticks, [str(int(t)) for t in ticks])
        ax.minorticks_off()
    axes[0].legend(loc="best")
    fig.tight_layout()
    plots.save(fig, out_dir / "budget_sweep.png")
    plt.close(fig)

    # 4. Cluster-count distribution per subset (diagnoses long-answer collapse)
    subsets_only = sorted(scores["subset"].unique())
    fig, axes = plt.subplots(1, len(subsets_only), figsize=(3.3 * len(subsets_only), 2.9), sharey=True, squeeze=False)
    bins = np.arange(0.5, n_full + 1.5)
    for ax, s in zip(axes[0], subsets_only):
        ax.hist(scores.loc[scores["subset"] == s, "n_clusters"], bins=bins,
                color=plots.PALETTE[0], edgecolor=plots.SURFACE, linewidth=2)
        ax.set(title=s, xlabel="Semantic clusters")
    axes[0][0].set_ylabel("Prompts")
    fig.suptitle(f"Clusters among {n_full} samples (M = all distinct)", fontweight="bold")
    fig.tight_layout()
    plots.save(fig, out_dir / "n_clusters_hist.png")
    plt.close(fig)

    # 5. SE distribution by label
    fig, axes = plt.subplots(1, len(main), figsize=(3.3 * len(main), 2.8), squeeze=False)
    for ax, m in zip(axes[0], main):
        x = merged[m].to_numpy(dtype=float)
        b = np.linspace(np.nanmin(x), np.nanmax(x), 30)
        ax.hist(x[y == 0], bins=b, density=True, histtype="step", lw=2, color=plots.PALETTE[0], label="faithful")
        ax.hist(x[y == 1], bins=b, density=True, histtype="step", lw=2, color=plots.PALETTE[1], label="hallucinated")
        ax.set(title=DISPLAY[m], yticks=[])
    axes[0][0].legend()
    fig.tight_layout()
    plots.save(fig, out_dir / "score_by_label.png")
    plt.close(fig)


def main():
    args = parse_args()
    started = time.time()

    responses = load_responses(args.n_samples, args.subsets, args.limit)
    print(f"[INFO] Prompts with samples: {len(responses):,}")

    writer = ShardWriter(args.output_dir / "shards")
    done = writer.done_uids()
    todo = responses[~responses["uid"].isin(done)].reset_index(drop=True)
    print(f"[INFO] Already done: {len(done):,}  remaining: {len(todo):,}")

    nli = scorer = None
    if len(todo):
        nli = NLIModel(resolve_nli(args.nli_model), args.device, batch_size=args.nli_batch_size)
        if not args.no_loglik:
            scorer = SequenceScorer(args.llm, args.device)

    pending = []
    for start in tqdm(range(0, len(todo), PROMPTS_PER_BATCH), desc="B4 semantic entropy"):
        batch = todo.iloc[start : start + PROMPTS_PER_BATCH]

        answer_sets = []
        for row in batch.itertuples(index=False):
            answers = list(row.samples)
            if args.include_main:
                answers = [row.main_answer] + answers
            answer_sets.append(answers)

        matrices = nli_pair_matrices(
            nli,
            [(row.prompt, answers) for row, answers in zip(batch.itertuples(index=False), answer_sets)],
            condition_on_question=not args.no_question,
        )

        for row, answers, matrix in zip(batch.itertuples(index=False), answer_sets, matrices):
            m = len(answers)
            budgets = [b for b in BUDGETS if b < m] + [m]

            ll_sum = ll_len = ll_norm = None
            if scorer is not None:
                ll_sum, ll_len = scorer.score(row.prompt, answers)
                ll_norm = ll_sum / np.maximum(ll_len, 1)

            result = {
                "uid": row.uid,
                "dataset": row.dataset,
                "subset": row.subset,
                "n_answers": m,
                "nli_argmax": matrix.argmax(axis=-1).astype(np.int8).ravel().tolist(),
            }
            result.update(entropies(matrix, ll_norm, budgets, strict=True))
            result.update(entropies(matrix, ll_norm, budgets, strict=False))
            if ll_norm is not None:
                result["loglik_sum"] = ll_sum.tolist()
                result["loglik_tokens"] = ll_len.tolist()
            pending.append(result)

        if len(pending) >= PROMPTS_PER_SHARD:
            writer.write(pending)
            pending = []
    writer.write(pending)

    scores = writer.read_all()
    scores.to_parquet(args.output_dir / "scores.parquet", index=False)
    print(f"[SUCCESS] Scores for {len(scores):,} prompts -> {args.output_dir / 'scores.parquet'}")

    summary = scores.groupby("subset")[["n_clusters", "discrete_semantic_entropy"]].mean()
    print("\nMean clusters / discrete SE per subset:")
    print(summary.to_string())

    score_columns = [
        c
        for c in scores.columns
        if c.startswith(("semantic_entropy", "discrete_semantic_entropy", "naive_entropy", "n_clusters"))
    ]
    labels = load_labels()
    if labels is not None:
        metrics = evaluate_scores(scores, labels, score_columns)
        metrics.to_csv(args.output_dir / "metrics.csv", index=False)
        print(metrics[metrics["subset"] == "all"].to_string(index=False))
        sweep = budget_table(metrics)
        sweep.to_csv(args.output_dir / "budget_sweep.csv", index=False)
        make_figures(scores, labels, metrics, sweep, args.output_dir / "figures")

    config = {
        "experiment": "B4-semantic-entropy",
        "n_samples": args.n_samples,
        "include_main": args.include_main,
        "nli_model": resolve_nli(args.nli_model),
        "llm": None if args.no_loglik else args.llm,
        "condition_on_question": not args.no_question,
        "budgets": BUDGETS,
        "subsets": args.subsets,
        "limit": args.limit,
        "n_examples": int(len(scores)),
        "git_commit_sha": git_commit_sha(),
        "elapsed_seconds": round(time.time() - started, 1),
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()

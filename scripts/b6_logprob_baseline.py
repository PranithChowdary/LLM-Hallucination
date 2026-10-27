"""
B6: Naive log-probability threshold baseline.

The weakest baseline the paper needs: flag an answer as hallucinated
when the model's own average token log-probability is low, i.e.

    score(x) = - (1/T) sum_t log p(y_t | y_<t, x)        (flag if score >= tau)

No extra generations, no extra models: it reuses the B1 per-answer
features (themselves aggregated from A3's teacher-forced token scores).

Variants reported alongside the headline score:
    logprob_neg_mean   - mean token log-prob           (headline)
    logprob_neg_min    - min token log-prob            (worst single token)
    logprob_neg_sum    - total sequence log-prob       (length-confounded)
    perplexity         - exp(neg_mean), same ranking as neg_mean

The threshold tau is chosen on training folds only (grouped 5-fold CV on
norm_key, the same folds as B5) and evaluated on the held-out fold:
    youden      - max TPR - FPR
    recall@0.90 - largest tau with >= 90% recall on the training folds
    recall@0.95 - same at 95% (what a recall guarantee would demand)

Run from the repository root (CPU, seconds):
    python scripts/b6_logprob_baseline.py

Inputs:
    data/processed/b1_answer_features.parquet
    data/processed/generation_labels.parquet
    data/calibration.parquet                      (norm_key for grouped folds)

Outputs (data/b6/):
    scores.parquet          one row per uid, all score variants
    metrics.csv             AUROC (bootstrap CI) / AUPRC / AURAC, overall + per subset
    threshold_results.csv   held-out recall / precision / FPR / flag rate per fold
    threshold_summary.csv   mean +- std over folds per rule
    figures/*.png
    config.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval import plots  # noqa: E402
from src.eval.metrics import (  # noqa: E402
    evaluate_scores,
    recall_threshold,
    threshold_metrics,
    youden_threshold,
)
from src.signals.common import (  # noqa: E402
    DATA_DIR,
    PROCESSED_DIR,
    git_commit_sha,
    load_labels,
    load_prompts,
)
from src.signals.probes import grouped_folds  # noqa: E402


OUTPUT_DIR = DATA_DIR / "b6"
B1_FEATURES = PROCESSED_DIR / "b1_answer_features.parquet"

SEED = 42
N_FOLDS = 5
TARGET_RECALLS = [0.90, 0.95]

SCORES = {
    "logprob_neg_mean": "-mean log p (headline)",
    "logprob_neg_min": "-min log p",
    "logprob_neg_sum": "-sum log p",
}
HEADLINE = "logprob_neg_mean"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--b1-features", type=Path, default=B1_FEATURES)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def build_scores(b1_path: Path) -> pd.DataFrame:
    b1 = pd.read_parquet(b1_path)
    b1["uid"] = b1["uid"].astype(str)

    prompts = load_prompts()
    frame = b1.merge(prompts[["subset", "norm_key"]], left_on="uid", right_index=True)

    frame["logprob_neg_mean"] = -frame["mean_token_log_probability"]
    frame["logprob_neg_min"] = -frame["min_token_log_probability"]
    frame["logprob_neg_sum"] = -frame["mean_token_log_probability"] * frame["output_token_length"]
    frame["perplexity"] = np.exp(frame["logprob_neg_mean"])

    columns = ["uid", "subset", "norm_key", *SCORES, "perplexity", "output_token_length"]
    return frame[columns].sort_values("uid", kind="stable").reset_index(drop=True)


def threshold_experiment(frame: pd.DataFrame, score: str) -> pd.DataFrame:
    """Choose tau on training folds, evaluate on the held-out fold."""
    labeled = frame.dropna(subset=["hallucinated"]).reset_index(drop=True)
    folds = grouped_folds(labeled, N_FOLDS, SEED)
    s = labeled[score].to_numpy(dtype=float)
    y = labeled["hallucinated"].to_numpy().astype(int)

    rules = {"youden": lambda ss, yy: youden_threshold(ss, yy)}
    for r in TARGET_RECALLS:
        rules[f"recall@{r:.2f}"] = lambda ss, yy, r=r: recall_threshold(ss, yy, r)

    rows = []
    for fold in range(N_FOLDS):
        train, test = folds != fold, folds == fold
        if len(np.unique(y[train])) < 2 or not test.any():
            print(f"[WARNING] {score}: fold {fold} has one class in training; skipped")
            continue
        for rule, choose in rules.items():
            tau = choose(s[train], y[train])
            result = threshold_metrics(s[test], y[test], tau)
            rows.append({"score": score, "rule": rule, "fold": fold, **result})
    return pd.DataFrame(rows)


def make_figures(frame: pd.DataFrame, metrics: pd.DataFrame, thresholds: pd.DataFrame, out_dir: Path):
    plt = plots.setup(headless=True)
    names = list(SCORES.values())
    colors = plots.ColorMap(names)

    labeled = frame.dropna(subset=["hallucinated"])
    y = labeled["hallucinated"].to_numpy().astype(int)
    curves = {SCORES[c]: labeled[c].to_numpy(dtype=float) for c in SCORES}

    # 1. ROC + PR
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    plots.roc_pr_panels(axes, curves, y, colors, " — naive log-prob")
    fig.tight_layout()
    plots.save(fig, out_dir / "roc_pr.png")
    plt.close(fig)

    # 2. AUROC per subset
    table = metrics.copy()
    table["score"] = table["score"].map(SCORES)
    subsets = ["all"] + sorted(s for s in table["subset"].unique() if s != "all")
    fig, axes = plt.subplots(1, len(subsets), figsize=(3.3 * len(subsets), 2.6), sharex=True, squeeze=False)
    for ax, s in zip(axes[0], subsets):
        plots.metric_bars(ax, table[table["subset"] == s], "auroc", colors)
        ax.set_title(s)
        if ax is not axes[0][0]:
            ax.set_yticklabels([])
    fig.suptitle("B6 AUROC by subset", fontweight="bold")
    fig.tight_layout()
    plots.save(fig, out_dir / "auroc_by_subset.png")
    plt.close(fig)

    # 3. Headline score distribution by label, with the chosen thresholds
    x = labeled[HEADLINE].to_numpy(dtype=float)
    bins = np.linspace(np.nanpercentile(x, 0.5), np.nanpercentile(x, 99.5), 50)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.hist(x[y == 0], bins=bins, density=True, histtype="step", lw=2, color=plots.PALETTE[0], label="faithful")
    ax.hist(x[y == 1], bins=bins, density=True, histtype="step", lw=2, color=plots.PALETTE[1], label="hallucinated")
    head = thresholds[thresholds["score"] == HEADLINE].groupby("rule")["threshold"].mean()
    for (rule, tau), ls in zip(head.items(), ["--", ":", "-."]):
        ax.axvline(tau, color=plots.TEXT_MUTED, lw=1.2, ls=ls, label=f"tau ({rule}) = {tau:.2f}")
    ax.set(xlabel="-mean token log-probability", yticks=[], title="Score by label and learned thresholds")
    ax.legend()
    fig.tight_layout()
    plots.save(fig, out_dir / "score_by_label.png")
    plt.close(fig)

    # 4. What a threshold costs: recall and flag (escalation) rate vs tau
    taus = np.quantile(x, np.linspace(0, 1, 200))
    recall = [(x[y == 1] >= t).mean() for t in taus]
    flag = [(x >= t).mean() for t in taus]
    accepted_acc = [((x < t) & (y == 0)).sum() / max((x < t).sum(), 1) for t in taus]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    c = plots.ColorMap(["recall (hallucinations caught)", "flag rate (escalated)", "accuracy of accepted"])
    ax.plot(taus, recall, color=c["recall (hallucinations caught)"], label="recall (hallucinations caught)")
    ax.plot(taus, flag, color=c["flag rate (escalated)"], label="flag rate (escalated)")
    ax.plot(taus, accepted_acc, color=c["accuracy of accepted"], label="accuracy of accepted")
    for r in TARGET_RECALLS:
        ax.axhline(r, color=plots.NEUTRAL, lw=1, ls=":")
    ax.set(xlabel="threshold tau on -mean log p", ylabel="fraction", ylim=(0, 1.02),
           title="Threshold trade-off (all labeled answers)")
    ax.legend(loc="center right")
    fig.tight_layout()
    plots.save(fig, out_dir / "threshold_tradeoff.png")
    plt.close(fig)

    # 5. Rejection-accuracy
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    plots.rejection_panel(ax, curves, y, colors)
    fig.tight_layout()
    plots.save(fig, out_dir / "rejection_accuracy.png")
    plt.close(fig)


def main():
    args = parse_args()
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame = build_scores(args.b1_features)
    frame.drop(columns="norm_key").to_parquet(args.output_dir / "scores.parquet", index=False)
    print(f"[INFO] Scores for {len(frame):,} answers -> {args.output_dir / 'scores.parquet'}")

    labels = load_labels()
    config = {
        "experiment": "B6-naive-logprob-threshold",
        "b1_features": str(args.b1_features),
        "headline_score": HEADLINE,
        "n_folds": N_FOLDS,
        "seed": SEED,
        "target_recalls": TARGET_RECALLS,
        "n_examples": int(len(frame)),
        "git_commit_sha": git_commit_sha(),
    }

    if labels is None:
        print("[WARNING] No labels; only scores were written.")
    else:
        frame = frame.merge(labels, on="uid", how="left")
        config["n_labeled"] = int(frame["hallucinated"].notna().sum())

        metrics = evaluate_scores(frame.drop(columns="hallucinated"), labels, list(SCORES))
        metrics.to_csv(args.output_dir / "metrics.csv", index=False)
        print(metrics[metrics["subset"] == "all"].to_string(index=False))

        thresholds = pd.concat([threshold_experiment(frame, s) for s in SCORES], ignore_index=True)
        thresholds.to_csv(args.output_dir / "threshold_results.csv", index=False)

        value_cols = ["threshold", "recall", "precision", "f1", "fpr", "flag_rate", "accepted_accuracy"]
        summary = thresholds.groupby(["score", "rule"])[value_cols].agg(["mean", "std"])
        summary.columns = [f"{a}_{b}" for a, b in summary.columns]
        summary = summary.reset_index()
        summary.to_csv(args.output_dir / "threshold_summary.csv", index=False)
        print("\nHeld-out threshold results (mean over folds):")
        print(summary[summary["score"] == HEADLINE][
            ["rule", "threshold_mean", "recall_mean", "precision_mean", "fpr_mean", "flag_rate_mean"]
        ].to_string(index=False))

        make_figures(frame, metrics, thresholds, args.output_dir / "figures")

    config["elapsed_seconds"] = round(time.time() - started, 1)
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"[SUCCESS] B6 outputs in {args.output_dir}")


if __name__ == "__main__":
    main()

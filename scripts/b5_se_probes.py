"""
B5: Reproduce Semantic Entropy Probes (SEPs).

Two stages, run from the repository root:

    # 1. GPU: hidden states at TBG and SLT for every layer (A2 main answers)
    python scripts/b5_se_probes.py extract

    # 2. CPU is fine: train SEPs (+ accuracy-probe baseline) on B4 output
    python scripts/b5_se_probes.py train
    python scripts/b5_se_probes.py train --states a3   # final layer only, from A3

Inputs:
    data/b4/scores.parquet                            SE targets (run B4 first)
    data/b6/scores.parquet                            optional, B6 baseline in the comparison
    data/calibration.parquet                          norm_key for grouped folds
    data/processed/generation_labels.parquet          optional, for AUROC and
                                                      the accuracy probe

Outputs (data/b5/):
    states/part_*.npz        uid, tbg (n, L+1, H), slt (n, L+1, H) float16
    per_layer_metrics.csv    AUROC per (position, layer, probe, target)
    scores.parquet           out-of-fold probe scores, one row per uid
    comparison_metrics.csv   SEP vs accuracy probe vs SE vs naive baselines
    figures/*.png            per-layer AUROC (Kossen et al. Fig. 3-style),
                             method comparison, ROC/PR, rejection-accuracy
    config.json

Layer selection uses AUROC against binarized SE only, never against the
hallucination labels, so the reported hallucination AUROC of the
selected layer is not tuned on those labels.

Method details: src/signals/probes.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.report import auroc  # noqa: E402
from src.eval import plots  # noqa: E402
from src.eval.metrics import evaluate_scores  # noqa: E402
from src.signals.common import (  # noqa: E402
    DATA_DIR,
    LLM_DIR,
    MAIN_ANSWERS_PATH,
    PROJECT_ROOT,
    git_commit_sha,
    load_labels,
    load_prompts,
    take_limit,
)
from src.signals.probes import (  # noqa: E402
    ProbeStateExtractor,
    best_split_threshold,
    cross_fit_probe,
    grouped_folds,
    load_a3_last_token_states,
)


OUTPUT_DIR = DATA_DIR / "b5"
STATES_DIR = OUTPUT_DIR / "states"
B4_SCORES = DATA_DIR / "b4" / "scores.parquet"
B6_SCORES = DATA_DIR / "b6" / "scores.parquet"
A3_HIDDEN_DIR = PROJECT_ROOT / "data" / "a3" / "hidden_states"

SEED = 42
N_FOLDS = 5
EXAMPLES_PER_SHARD = 1000


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = parser.add_subparsers(dest="stage", required=True)

    ext = sub.add_parser("extract", help="Extract TBG/SLT hidden states (GPU).")
    ext.add_argument("--llm", default=str(LLM_DIR))
    ext.add_argument("--limit", type=int, default=None)
    ext.add_argument("--device", default=None)
    ext.add_argument("--batch-size", type=int, default=16)

    tr = sub.add_parser("train", help="Train SEPs and accuracy probes.")
    tr.add_argument("--states", choices=["full", "a3"], default="full")
    tr.add_argument(
        "--target",
        default="semantic_entropy",
        help="B4 column to probe (e.g. discrete_semantic_entropy).",
    )
    tr.add_argument("--layers", type=int, nargs="*", default=None)
    tr.add_argument("--c", type=float, default=1.0, help="Inverse L2 strength.")
    tr.add_argument("--b4-scores", type=Path, default=B4_SCORES)
    return parser.parse_args()


# -------------------------------------------------------------------
# Stage 1: extraction
# -------------------------------------------------------------------

def done_state_uids() -> set[str]:
    done = set()
    for path in sorted(STATES_DIR.glob("part_*.npz")):
        done.update(np.load(path, allow_pickle=False)["uid"].tolist())
    return done


def run_extract(args):
    STATES_DIR.mkdir(parents=True, exist_ok=True)
    answers = pd.read_parquet(MAIN_ANSWERS_PATH)
    answers["uid"] = answers["uid"].astype(str)
    answers = answers.sort_values("uid", kind="stable").reset_index(drop=True)
    answers = take_limit(answers, args.limit)

    done = done_state_uids()
    todo = answers[~answers["uid"].isin(done)].reset_index(drop=True)
    print(f"[INFO] Already extracted: {len(done):,}  remaining: {len(todo):,}")
    if not len(todo):
        return

    extractor = ProbeStateExtractor(args.llm, args.device, args.batch_size)
    shard_number = len(list(STATES_DIR.glob("part_*.npz")))

    for start in tqdm(range(0, len(todo), EXAMPLES_PER_SHARD), desc="B5 extract"):
        chunk = todo.iloc[start : start + EXAMPLES_PER_SHARD]
        tbg, slt = extractor.extract(
            chunk["prompt"].tolist(), chunk["generated_answer"].tolist()
        )
        shard_number += 1
        path = STATES_DIR / f"part_{shard_number:06d}.npz"
        tmp = path.with_name(f"tmp_{path.name}")  # not matched by part_*.npz
        np.savez(tmp, uid=chunk["uid"].to_numpy(dtype=str), tbg=tbg, slt=slt)
        tmp.replace(path)

    print(f"[SUCCESS] States saved to {STATES_DIR}  (layers incl. embeddings: {tbg.shape[1]})")


# -------------------------------------------------------------------
# Stage 2: training
# -------------------------------------------------------------------

def load_states(kind: str, uids: list[str]) -> dict[str, np.ndarray]:
    if kind == "a3":
        return {"slt": load_a3_last_token_states(uids, A3_HIDDEN_DIR)}

    parts = [np.load(p, allow_pickle=False) for p in sorted(STATES_DIR.glob("part_*.npz"))]
    if not parts:
        raise FileNotFoundError(f"No states in {STATES_DIR}; run the extract stage first.")
    all_uids = np.concatenate([p["uid"] for p in parts])
    position = {u: i for i, u in enumerate(all_uids)}
    missing = [u for u in uids if u not in position]
    if missing:
        raise ValueError(f"{len(missing):,} uids have no states, e.g. {missing[:3]}")
    order = np.array([position[u] for u in uids])
    return {
        name: np.concatenate([p[name] for p in parts])[order]
        for name in ("tbg", "slt")
    }


def run_train(args):
    started = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    se = pd.read_parquet(args.b4_scores, columns=["uid", "subset", args.target])
    se["uid"] = se["uid"].astype(str)
    prompts = load_prompts()
    frame = se.merge(prompts[["norm_key"]], left_on="uid", right_index=True)
    frame = frame.sort_values("uid", kind="stable").reset_index(drop=True)

    labels = load_labels()
    if labels is not None:
        frame = frame.merge(labels, on="uid", how="left")
    else:
        frame["hallucinated"] = np.nan

    print(f"[INFO] Examples with SE targets: {len(frame):,}")
    states = load_states(args.states, frame["uid"].tolist())

    folds = grouped_folds(frame, N_FOLDS, SEED)
    entropy = frame[args.target].to_numpy(dtype=float)
    halluc = frame["hallucinated"].to_numpy(dtype=float)
    has_labels = np.isfinite(halluc)

    # Global gamma is used ONLY to report "how well does the probe
    # recover high-SE"; the probes themselves use per-fold gammas.
    gamma_all = best_split_threshold(entropy)
    se_binary = (entropy >= gamma_all).astype(int)
    print(f"[INFO] gamma* = {gamma_all:.4f}  ->  {se_binary.mean():.1%} high-SE")

    out_scores = frame[["uid", "subset"]].copy()
    rows = []

    for position, X_all in states.items():
        n_layers = X_all.shape[1]
        layers = args.layers if args.layers else list(range(n_layers))
        for layer in tqdm(layers, desc=f"probes @ {position}"):
            X = X_all[:, layer, :]
            tag = f"{position}_l{layer:02d}"

            sep, gammas = cross_fit_probe(
                X, entropy, folds, binarize_entropy=True, c=args.c, seed=SEED
            )
            out_scores[f"sep_{tag}"] = sep
            row = {
                "position": position,
                "layer": layer,
                "probe": "sep",
                "auroc_vs_high_se": auroc(sep, se_binary),
                "auroc_vs_hallucination": (
                    auroc(sep[has_labels], halluc[has_labels].astype(int))
                    if has_labels.any() else np.nan
                ),
                "gamma_mean": float(np.mean(gammas)),
            }
            rows.append(row)

            if has_labels.sum() > 100:
                acc, _ = cross_fit_probe(X, halluc, folds, c=args.c, seed=SEED)
                out_scores[f"accprobe_{tag}"] = acc
                rows.append({
                    "position": position,
                    "layer": layer,
                    "probe": "accuracy",
                    "auroc_vs_high_se": auroc(acc, se_binary),
                    "auroc_vs_hallucination": auroc(
                        acc[has_labels], halluc[has_labels].astype(int)
                    ),
                    "gamma_mean": np.nan,
                })

    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUTPUT_DIR / "per_layer_metrics.csv", index=False)

    # Pick the SEP layer per position by high-SE AUROC (label-free).
    for position in states:
        sep_rows = metrics[(metrics.position == position) & (metrics.probe == "sep")]
        best = sep_rows.sort_values("auroc_vs_high_se", ascending=False).iloc[0]
        layer = int(best["layer"])
        out_scores[f"sep_{position}"] = out_scores[f"sep_{position}_l{layer:02d}"]
        print(
            f"[RESULT] SEP {position}: best layer {layer}  "
            f"AUROC(high SE) = {best['auroc_vs_high_se']:.3f}  "
            f"AUROC(hallucination) = {best['auroc_vs_hallucination']:.3f}"
        )

    # Accuracy-probe layer: chosen on out-of-fold AUROC against the labels
    # (mildly optimistic; this is the baseline, so the bias favours it).
    for position in states:
        acc_rows = metrics[(metrics.position == position) & (metrics.probe == "accuracy")]
        if len(acc_rows):
            best = acc_rows.sort_values("auroc_vs_hallucination", ascending=False).iloc[0]
            layer = int(best["layer"])
            out_scores[f"accprobe_{position}"] = out_scores[f"accprobe_{position}_l{layer:02d}"]

    out_scores.to_parquet(OUTPUT_DIR / "scores.parquet", index=False)

    if labels is not None:
        comparison = compare_methods(out_scores, labels, args.b4_scores)
        comparison.to_csv(OUTPUT_DIR / "comparison_metrics.csv", index=False)
        print(comparison[comparison["subset"] == "all"].to_string(index=False))
    make_figures(metrics, out_scores, labels, args.b4_scores, OUTPUT_DIR / "figures")

    # Reference point: SE itself (the expensive thing SEPs approximate).
    if has_labels.any():
        se_auroc = auroc(entropy[has_labels], halluc[has_labels].astype(int))
        print(f"[RESULT] {args.target} itself: AUROC(hallucination) = {se_auroc:.3f}")

    config = {
        "experiment": "B5-semantic-entropy-probes",
        "states": args.states,
        "target": args.target,
        "gamma_all": gamma_all,
        "frac_high_se": float(se_binary.mean()),
        "n_folds": N_FOLDS,
        "seed": SEED,
        "c": args.c,
        "n_examples": int(len(frame)),
        "n_labeled": int(has_labels.sum()),
        "git_commit_sha": git_commit_sha(),
        "elapsed_seconds": round(time.time() - started, 1),
    }
    (OUTPUT_DIR / "config.json").write_text(json.dumps(config, indent=2))


# -------------------------------------------------------------------
# Comparison and figures
# -------------------------------------------------------------------

COMPARE = {
    "sep_tbg": "SEP (TBG)",
    "sep_slt": "SEP (SLT)",
    "accprobe_tbg": "Accuracy probe (TBG)",
    "accprobe_slt": "Accuracy probe (SLT)",
    "semantic_entropy": "Semantic entropy (B4)",
    "naive_entropy": "Naive entropy (B4)",
    "logprob_neg_mean": "Naive log-prob (B6)",
}


def comparison_frame(out_scores, b4_path):
    frame = out_scores.copy()
    b4 = pd.read_parquet(b4_path)
    keep = [c for c in ("semantic_entropy", "naive_entropy") if c in b4]
    frame = frame.merge(b4[["uid", *keep]], on="uid", how="left")
    if B6_SCORES.exists():
        b6 = pd.read_parquet(B6_SCORES, columns=["uid", "logprob_neg_mean"])
        frame = frame.merge(b6, on="uid", how="left")
    present = [c for c in COMPARE if c in frame]
    return frame.rename(columns=COMPARE), [COMPARE[c] for c in present]


def compare_methods(out_scores, labels, b4_path):
    frame, names = comparison_frame(out_scores, b4_path)
    return evaluate_scores(frame, labels, names)


def make_figures(metrics, out_scores, labels, b4_path, out_dir):
    plt = plots.setup(headless=True)

    # 1. Per-layer AUROC (Kossen et al. Fig. 3-style), one panel per target
    series = [(p, probe) for probe in ("sep", "accuracy") for p in ("tbg", "slt")
              if len(metrics[(metrics.position == p) & (metrics.probe == probe)])]
    names = [f"{'SEP' if probe == 'sep' else 'Accuracy probe'} ({p.upper()})" for p, probe in series]
    colors = plots.ColorMap(names)
    targets = [("auroc_vs_high_se", "Predicting high semantic entropy")]
    if metrics["auroc_vs_hallucination"].notna().any():
        targets.append(("auroc_vs_hallucination", "Predicting hallucination"))
    fig, axes = plt.subplots(1, len(targets), figsize=(5.8 * len(targets), 3.9), sharey=True, squeeze=False)
    for ax, (column, title) in zip(axes[0], targets):
        for (p, probe), name in zip(series, names):
            d = metrics[(metrics.position == p) & (metrics.probe == probe)].sort_values("layer")
            ax.plot(d["layer"], d[column], marker="o", color=colors[name], label=name,
                    ls="-" if probe == "sep" else "--")
        ax.axhline(0.5, color=plots.NEUTRAL, lw=1, ls=":")
        ax.set(xlabel="Layer (0 = embeddings)", title=title)
    axes[0][0].set_ylabel("AUROC (out-of-fold)")
    axes[0][0].legend(loc="lower right")
    fig.tight_layout()
    plots.save(fig, out_dir / "per_layer_auroc.png")
    plt.close(fig)

    if labels is None:
        return

    # 2. Method comparison: does the cheap probe recover what SE detects?
    frame, names = comparison_frame(out_scores, b4_path)
    table = evaluate_scores(frame, labels, names)
    colors = plots.ColorMap(names)
    subsets = ["all"] + sorted(s for s in table["subset"].unique() if s != "all")
    fig, axes = plt.subplots(1, len(subsets), figsize=(3.4 * len(subsets), 0.42 * len(names) + 1.2),
                             sharex=True, squeeze=False)
    for ax, s in zip(axes[0], subsets):
        plots.metric_bars(ax, table[table["subset"] == s], "auroc", colors)
        ax.set_title(s)
        if ax is not axes[0][0]:
            ax.set_yticklabels([])
    fig.suptitle("AUROC for hallucination: probes vs sampling-based SE", fontweight="bold")
    fig.tight_layout()
    plots.save(fig, out_dir / "comparison_auroc.png")
    plt.close(fig)

    # 3. ROC/PR and rejection-accuracy for the same methods
    merged = frame.merge(labels, on="uid").dropna(subset=["hallucinated"])
    y = merged["hallucinated"].to_numpy().astype(int)
    curves = {n: merged[n].to_numpy(dtype=float) for n in names}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    plots.roc_pr_panels(axes, curves, y, colors, " — B5 comparison")
    fig.tight_layout()
    plots.save(fig, out_dir / "roc_pr.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    plots.rejection_panel(ax, curves, y, colors)
    fig.tight_layout()
    plots.save(fig, out_dir / "rejection_accuracy.png")
    plt.close(fig)


def main():
    args = parse_args()
    if args.stage == "extract":
        run_extract(args)
    else:
        run_train(args)


if __name__ == "__main__":
    main()

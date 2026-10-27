"""
Detection metrics shared by B3-B6.

Convention everywhere: label 1 = hallucinated (positive class), and a
HIGHER score means MORE likely hallucinated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.report import auroc


def auprc(scores: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    labels = np.asarray(labels).astype(int)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(average_precision_score(labels, scores))


def rejection_accuracy_curve(
    scores: np.ndarray, labels: np.ndarray, n_points: int = 101
) -> tuple[np.ndarray, np.ndarray]:
    """
    Farquhar et al. (2024) rejection-accuracy curve: reject the answers
    with the highest scores first; report accuracy (fraction NOT
    hallucinated) of the answers that remain.

    Returns (retained_fraction, accuracy), retained from 1.0 down to 1/n.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(int)
    order = np.argsort(scores, kind="stable")  # most confident first
    correct = 1 - labels[order]
    cum_acc = np.cumsum(correct) / np.arange(1, len(correct) + 1)

    retained = np.linspace(1.0, 1.0 / len(scores), n_points)
    keep = np.maximum(1, np.round(retained * len(scores)).astype(int))
    return retained, cum_acc[keep - 1]


def aurac(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the rejection-accuracy curve (higher is better)."""
    retained, acc = rejection_accuracy_curve(scores, labels)
    trapezoid = getattr(np, "trapezoid", None) or np.trapz  # numpy < 2.0
    return float(trapezoid(acc[::-1], retained[::-1]))


def evaluate_scores(
    scores: pd.DataFrame,
    labels: pd.DataFrame,
    score_columns: list[str],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    AUROC (bootstrap 95% CI), AUPRC and AURAC per score column, overall
    and per subset. `scores` and `labels` are joined on uid; `scores`
    needs a `subset` column.
    """
    merged = scores.merge(labels[["uid", "hallucinated"]], on="uid", how="inner")
    merged = merged.dropna(subset=["hallucinated"])
    rng = np.random.default_rng(seed)

    rows = []
    groups = [("all", merged)] + list(merged.groupby("subset"))

    for group_name, frame in groups:
        y = frame["hallucinated"].to_numpy().astype(int)
        if len(np.unique(y)) < 2:
            continue

        for column in score_columns:
            if column not in frame:
                continue
            s = frame[column].to_numpy(dtype=float)
            keep = np.isfinite(s)
            s_k, y_k = s[keep], y[keep]
            if len(np.unique(y_k)) < 2:
                continue

            boot = []
            for _ in range(n_bootstrap):
                idx = rng.integers(0, len(s_k), len(s_k))
                if len(np.unique(y_k[idx])) == 2:
                    boot.append(auroc(s_k[idx], y_k[idx]))

            rows.append({
                "subset": group_name,
                "score": column,
                "n": int(len(s_k)),
                "frac_hallucinated": float(y_k.mean()),
                "auroc": auroc(s_k, y_k),
                "auroc_ci_low": float(np.percentile(boot, 2.5)),
                "auroc_ci_high": float(np.percentile(boot, 97.5)),
                "auprc": auprc(s_k, y_k),
                "aurac": aurac(s_k, y_k),
            })

    return pd.DataFrame(rows)


# -------------------------------------------------------------------
# Threshold selection (B6 and later the cascade)
# -------------------------------------------------------------------

def youden_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """Threshold maximizing TPR - FPR (flag as hallucinated if score >= t)."""
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(labels, scores)
    best = np.argmax(tpr - fpr)
    return float(thresholds[best])


def recall_threshold(scores: np.ndarray, labels: np.ndarray, target_recall: float) -> float:
    """
    Largest threshold whose recall on (scores, labels) is >= target.
    Flagging = score >= t, so a lower t flags more and raises recall.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(int)
    positives = np.sort(scores[labels == 1])[::-1]
    k = int(np.ceil(target_recall * len(positives)))
    k = min(max(k, 1), len(positives))
    return float(positives[k - 1])


def threshold_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(int)
    flagged = scores >= threshold
    tp = int((flagged & (labels == 1)).sum())
    fp = int((flagged & (labels == 0)).sum())
    fn = int((~flagged & (labels == 1)).sum())
    tn = int((~flagged & (labels == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    return {
        "threshold": float(threshold),
        "recall": recall,
        "precision": precision,
        "f1": (2 * precision * recall / (precision + recall))
        if precision and recall and np.isfinite(precision) else float("nan"),
        "fpr": fp / (fp + tn) if fp + tn else float("nan"),
        "accuracy": (tp + tn) / len(labels),
        "flag_rate": float(flagged.mean()),
        # accuracy of the answers NOT flagged (what the user actually sees)
        "accepted_accuracy": tn / (tn + fn) if tn + fn else float("nan"),
    }

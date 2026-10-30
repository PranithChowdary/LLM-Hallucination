import numpy as np
import pandas as pd
import pytest

from src.eval.metrics import (
    aurac,
    evaluate_scores,
    recall_threshold,
    rejection_accuracy_curve,
    threshold_metrics,
    youden_threshold,
)
from src.signals.selfcheck import greybox_scores, selfcheck_nli_batch


# ---------------------------------------------------------------------------
# Rejection-accuracy
# ---------------------------------------------------------------------------

def test_rejection_curve_perfect_detector_reaches_full_accuracy():
    labels = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    retained, acc = rejection_accuracy_curve(scores, labels, n_points=6)
    assert acc[0] == pytest.approx(0.6)  # nothing rejected
    assert acc[-1] == pytest.approx(1.0)  # keep only the most confident


def test_aurac_perfect_beats_reversed():
    labels = np.array([0] * 50 + [1] * 50)
    good = np.arange(100, dtype=float)
    assert aurac(good, labels) > aurac(-good, labels)


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

def test_recall_threshold_hits_target():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 500)
    scores = labels + rng.normal(0, 0.8, 500)
    for target in (0.5, 0.9, 0.95):
        tau = recall_threshold(scores, labels, target)
        recall = (scores[labels == 1] >= tau).mean()
        assert recall >= target
        # and it is the largest such threshold
        assert (scores[labels == 1] > tau).mean() < target + 1e-9 or recall == 1.0


def test_youden_separates_clean_data():
    labels = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    tau = youden_threshold(scores, labels)
    m = threshold_metrics(scores, labels, tau)
    assert m["recall"] == 1.0 and m["fpr"] == 0.0


def test_threshold_metrics_counts():
    labels = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.1, 0.8, 0.2])
    m = threshold_metrics(scores, labels, 0.5)
    assert m["recall"] == 0.5
    assert m["precision"] == 0.5
    assert m["flag_rate"] == 0.5
    assert m["accepted_accuracy"] == 0.5


def test_evaluate_scores_overall_and_subsets():
    scores = pd.DataFrame({
        "uid": [f"u{i}" for i in range(8)],
        "subset": ["qa"] * 4 + ["dialogue"] * 4,
        "s": [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6],
    })
    labels = pd.DataFrame({"uid": scores["uid"], "hallucinated": [0, 1, 0, 1, 0, 1, 0, 1]})
    out = evaluate_scores(scores, labels, ["s"], n_bootstrap=20)
    assert set(out["subset"]) == {"all", "qa", "dialogue"}
    assert (out["auroc"] == 1.0).all()


# ---------------------------------------------------------------------------
# SelfCheck pieces that do not need a real model
# ---------------------------------------------------------------------------

class FakeNLI:
    """Contradiction iff the hypothesis contains 'wrong'."""

    def probs(self, premises, hypotheses):
        out = np.zeros((len(hypotheses), 3), dtype=np.float32)
        for k, h in enumerate(hypotheses):
            out[k] = [0.9, 0.0, 0.1] if "wrong" in h else [0.1, 0.0, 0.9]
        return out


def test_selfcheck_nli_batch_returns_sample_by_sentence_matrices():
    items = [
        (["fine one.", "wrong one."], ["s1", "s2", "s3"]),
        (["fine."], ["s1"]),
    ]
    out = selfcheck_nli_batch(FakeNLI(), items)
    assert out[0].shape == (3, 2)
    assert out[1].shape == (1, 1)
    assert out[0][:, 0] == pytest.approx([0.1] * 3)
    assert out[0][:, 1] == pytest.approx([0.9] * 3)


def test_greybox_scores_align_tokens_to_sentences():
    answer = "Paris is big. It is old."
    sentences = ["Paris is big.", "It is old."]
    tokens = ["Paris", " is", " big", ".", " It", " is", " old", "."]
    logp = np.array([-1, -1, -1, -1, -3, -3, -3, -3], dtype=float)
    ent = np.array([0.5] * 4 + [1.5] * 4)
    out = greybox_scores(answer, sentences, tokens, logp, ent)
    assert out["greybox_aligned"]
    assert out["greybox_avg_neg_logprob"] == pytest.approx(2.0)
    assert out["greybox_max_entropy"] == pytest.approx(1.0)

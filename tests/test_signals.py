import numpy as np
import pandas as pd
import pytest

from src.signals.common import split_sentences
from src.signals.probes import best_split_threshold, cross_fit_probe, grouped_folds
from src.signals.selfcheck import contradiction_score, selfcheck_unigram
from src.signals.semantic_entropy import (
    CONTRADICTION,
    ENTAILMENT,
    NEUTRAL,
    are_equivalent,
    cluster_ids,
    discrete_semantic_entropy,
    naive_entropy,
    semantic_entropy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def matrix_from_labels(labels: list[list[int]]) -> np.ndarray:
    """One-hot (M, M, 3) NLI matrix from an argmax label grid."""
    labels = np.asarray(labels)
    m = np.zeros(labels.shape + (3,), dtype=np.float32)
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            m[i, j, labels[i, j]] = 1.0
    return m


E, N, C = ENTAILMENT, NEUTRAL, CONTRADICTION


# ---------------------------------------------------------------------------
# B4: clustering and entropies
# ---------------------------------------------------------------------------

def test_cluster_ids_groups_bidirectional_entailment():
    # 0 <=> 1, 2 alone, 3 <=> 0
    m = matrix_from_labels([
        [E, E, C, E],
        [E, E, C, E],
        [C, C, E, N],
        [E, E, N, E],
    ])
    assert cluster_ids(m) == [0, 0, 1, 0]


def test_cluster_ids_one_way_entailment_is_not_equivalence():
    m = matrix_from_labels([
        [E, E],
        [N, E],
    ])
    assert cluster_ids(m, strict=True) == [0, 1]
    # non-strict: (entail, neutral) has no contradiction and is not (N, N)
    assert cluster_ids(m, strict=False) == [0, 0]


def test_non_strict_rejects_contradiction_and_double_neutral():
    m = matrix_from_labels([[E, N], [N, E]])
    assert not are_equivalent(m, 0, 1, strict=False)
    m = matrix_from_labels([[E, E], [C, E]])
    assert not are_equivalent(m, 0, 1, strict=False)


def test_cluster_ids_are_prefix_consistent():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 3, size=(12, 12))
    np.fill_diagonal(labels, E)
    m = matrix_from_labels(labels.tolist())
    full = cluster_ids(m)
    for k in range(1, 13):
        assert cluster_ids(m[:k, :k]) == full[:k]


def test_discrete_semantic_entropy():
    assert discrete_semantic_entropy([0, 0, 0, 0]) == pytest.approx(0.0)
    assert discrete_semantic_entropy([0, 1, 2, 3]) == pytest.approx(np.log(4))
    assert discrete_semantic_entropy([0, 0, 1, 1]) == pytest.approx(np.log(2))


def test_semantic_entropy_equal_likelihoods_matches_discrete():
    ids = [0, 0, 1, 2]
    ll = np.full(4, -1.3)
    assert semantic_entropy(ids, ll) == pytest.approx(discrete_semantic_entropy(ids))


def test_semantic_entropy_weights_clusters_by_probability():
    # cluster 0 has much higher likelihood -> entropy close to 0
    ids = [0, 1]
    assert semantic_entropy(ids, np.array([0.0, -20.0])) < 1e-6
    assert semantic_entropy([0, 0], np.array([-1.0, -5.0])) == pytest.approx(0.0)


def test_naive_entropy():
    assert naive_entropy(np.array([-1.0, -3.0])) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# B3: SelfCheckGPT pieces
# ---------------------------------------------------------------------------

def test_contradiction_score_ignores_neutral():
    probs = np.array([[0.2, 0.6, 0.2], [0.9, 0.0, 0.1]])
    assert contradiction_score(probs) == pytest.approx([0.5, 0.9])


def test_unigram_consistent_tokens_score_lower():
    main = ["Paris is the capital of France."]
    agree = ["Paris is the capital of France."] * 5
    disagree = ["Lyon is a big city in the south."] * 5
    low = selfcheck_unigram(main, agree)
    high = selfcheck_unigram(main, disagree)
    assert low["avg_neg_logprob"] < high["avg_neg_logprob"]
    assert low["avg_max_neg_logprob"] < high["avg_max_neg_logprob"]


def test_split_sentences():
    text = "Paris is in France. It has 2 million people!\nIt is old."
    assert len(split_sentences(text)) == 3
    assert split_sentences("") == []


# ---------------------------------------------------------------------------
# B5: SEP pieces
# ---------------------------------------------------------------------------

def test_best_split_threshold_separates_two_modes():
    h = np.array([0.0, 0.1, 0.05, 2.0, 2.1, 1.9])
    gamma = best_split_threshold(h)
    assert 0.1 < gamma < 1.9


def test_best_split_threshold_constant_raises():
    with pytest.raises(ValueError):
        best_split_threshold(np.ones(5))


def test_grouped_folds_keep_groups_together():
    frame = pd.DataFrame({
        "subset": ["qa"] * 20 + ["dialogue"] * 20,
        "norm_key": [f"k{i // 2}" for i in range(40)],
    })
    folds = grouped_folds(frame, n_splits=5, seed=0)
    assert (folds >= 0).all()
    assert frame.assign(f=folds).groupby("norm_key")["f"].nunique().max() == 1


def test_cross_fit_probe_learns_linear_signal():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 8))
    entropy = np.where(X[:, 0] > 0, 2.0, 0.1) + rng.normal(0, 0.05, 400)
    folds = np.arange(400) % 5
    scores, gammas = cross_fit_probe(X, entropy, folds, binarize_entropy=True)
    assert len(gammas) == 5
    assert np.isfinite(scores).all()
    acc = ((scores > 0.5) == (X[:, 0] > 0)).mean()
    assert acc > 0.9

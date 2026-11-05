"""synthetic tests for src/calibration/conformal.py.

Scores here have a known, controllable relationship to the labels, so the
recall guarantee can be checked without touching any real data. These must
pass before C1 is run on real scores.

If your other tests import differently (see tests/test_loaders.py), adjust
the import below to match.
"""

import numpy as np
import pytest

from src.calibration.conformal import calibrate_recall_threshold


def _expected_k(n, alpha):
    return int(np.floor(alpha * (n + 1) + 1e-9))


def _mean_recall(draw, n, alpha, trials=4000, m=2000, seed=0):
    """Average recall over many calibration sets, using fresh positives."""
    rng = np.random.default_rng(seed)
    recalls = np.empty(trials)
    for t in range(trials):
        cal = draw(rng, n)
        tau = calibrate_recall_threshold(cal, np.ones(n, dtype=int), alpha)
        recalls[t] = np.mean(draw(rng, m) >= tau)
    return recalls.mean()


#  exact, hand-checkable cases

def test_worked_example_n19_alpha005():
    # n = 19, alpha = 0.05 -> k = 1 -> tau = smallest positive score
    scores = np.arange(19, dtype=float)[::-1]  # unsorted on purpose
    labels = np.ones(19, dtype=int)
    assert calibrate_recall_threshold(scores, labels, 0.05) == 0.0


def test_k_zero_flags_everything():
    # n = 10, alpha = 0.05 -> floor(0.55) = 0 -> too little data
    scores = np.linspace(0, 1, 10)
    labels = np.ones(10, dtype=int)
    tau = calibrate_recall_threshold(scores, labels, 0.05)
    assert tau == -np.inf
    assert np.all(scores >= tau)


def test_float_rounding_case():
    # alpha = 1 - 0.9 is 0.09999999999999998; with n = 29 a bare floor()
    # gives k = 2 instead of 3.
    alpha = 1 - 0.9
    positives = np.arange(1, 30, dtype=float)  # 1..29, n = 29
    labels = np.ones(29, dtype=int)
    assert calibrate_recall_threshold(positives, labels, alpha) == 3.0


def test_negatives_do_not_change_tau():
    rng = np.random.default_rng(1)
    pos = rng.normal(2, 1, 200)
    neg = rng.normal(-2, 1, 500)
    tau_pos_only = calibrate_recall_threshold(pos, np.ones(200, int), 0.1)
    scores = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(200, int), np.zeros(500, int)])
    shuffle = rng.permutation(len(scores))
    tau_mixed = calibrate_recall_threshold(scores[shuffle], labels[shuffle], 0.1)
    assert tau_pos_only == tau_mixed


def test_higher_score_means_riskier_orientation():
    # Well-separated data: positives high, negatives low. tau must sit among
    # the positives, so almost no negatives are flagged.
    rng = np.random.default_rng(2)
    pos = rng.normal(10, 1, 1000)
    neg = rng.normal(0, 1, 1000)
    scores = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(1000, int), np.zeros(1000, int)])
    tau = calibrate_recall_threshold(scores, labels, 0.05)
    assert tau > neg.max() - 1e-9 or np.mean(neg >= tau) < 0.001
    assert np.mean(pos >= tau) >= 0.94


# the guarantee itself 

@pytest.mark.parametrize("n", [19, 50, 100, 500])
def test_mean_recall_matches_one_minus_k_over_n_plus_1(n):
    alpha = 0.05
    k = _expected_k(n, alpha)
    expected = 1 - k / (n + 1)
    got = _mean_recall(lambda r, size: r.normal(0, 1, size), n, alpha)
    assert abs(got - expected) < 0.004, (n, k, expected, got)
    assert got >= 1 - alpha - 0.004  # never meaningfully below target


@pytest.mark.parametrize("alpha", [0.05, 0.10])
def test_distribution_free(alpha):
    # Same expected recall for a light-tailed, a heavy-tailed and a skewed score.
    n = 100
    expected = 1 - _expected_k(n, alpha) / (n + 1)
    draws = {
        "normal": lambda r, s: r.normal(0, 1, s),
        "cauchy": lambda r, s: r.standard_cauchy(s),
        "exponential": lambda r, s: r.exponential(1.0, s),
    }
    for name, draw in draws.items():
        got = _mean_recall(draw, n, alpha, trials=3000)
        assert abs(got - expected) < 0.005, (name, expected, got)


def test_ties_are_conservative():
    # Discrete scores produce many ties; recall must still reach the target.
    n, alpha = 100, 0.10
    got = _mean_recall(lambda r, s: r.integers(0, 5, s).astype(float), n, alpha,
                       trials=3000)
    assert got >= 1 - alpha - 0.004


def test_recall_converges_to_target_as_n_grows():
    alpha = 0.05
    gaps = []
    for n in (19, 99, 499):
        expected = 1 - _expected_k(n, alpha) / (n + 1)
        gaps.append(abs(expected - (1 - alpha)))
    assert gaps[0] >= gaps[1] >= gaps[2]
    assert gaps[-1] < 0.002


#input validation 

def test_rejects_bad_alpha():
    s, y = np.array([0.1, 0.2, 0.3]), np.array([1, 1, 0])
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            calibrate_recall_threshold(s, y, bad)


def test_rejects_no_positives():
    with pytest.raises(ValueError):
        calibrate_recall_threshold(np.array([0.1, 0.2]), np.array([0, 0]), 0.1)


def test_rejects_nan_label():
    with pytest.raises(ValueError):
        calibrate_recall_threshold(np.array([0.1, 0.2, 0.3]),
                                   np.array([1.0, np.nan, 1.0]), 0.1)


def test_rejects_nan_score():
    with pytest.raises(ValueError):
        calibrate_recall_threshold(np.array([0.1, np.nan, 0.3]),
                                   np.array([1, 1, 1]), 0.1)


def test_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        calibrate_recall_threshold(np.array([0.1, 0.2]), np.array([1, 1, 1]), 0.1)
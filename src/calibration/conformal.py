"""C1 - split-conformal recall threshold.

A generic calibration routine. It knows nothing about any particular detector
(SelfCheck, Semantic Entropy, B8 classifier, ...) and nothing about which
recall target is wanted: scores, labels and alpha are all inputs.

Convention
----------
Higher score = more likely hallucination. An answer is flagged as a
hallucination when ``score >= tau``.

Guarantee
---------
Calibrate on the hallucinated (label == 1) calibration examples only. Let
n be their number and k = floor(alpha * (n + 1)). Take tau = the k-th
smallest of their scores (tau = -inf if k == 0, i.e. flag everything).

A new hallucinated example that is exchangeable with the calibration
positives is missed (score < tau) with probability at most k / (n + 1)
<= alpha, so

    E[recall] >= 1 - k / (n + 1) >= 1 - alpha

The expectation is over the random draw of the calibration set (marginal
guarantee). Any single calibration set can land below the target; that is
what C6 (repeated splits) measures. Ties only make the miss probability
smaller, so the guarantee still holds.
"""

from __future__ import annotations

import numpy as np

# alpha is usually computed as 1 - target_recall, which is not exact in
# floating point: 1 - 0.9 == 0.09999999999999998, so 0.09999999999999998 * 30
# == 2.9999999999999996 and a bare floor() would give 2 instead of 3. The
# tolerance below only removes that rounding noise.
_FLOOR_TOL = 1e-9


def calibrate_recall_threshold(scores, labels, alpha) -> float:
    """Return the threshold tau for a target recall of ``1 - alpha``.

    Parameters
    ----------
    scores : array-like of float, shape (m,)
        Detector scores on the calibration set; higher = riskier.
    labels : array-like of {0, 1}, shape (m,)
        1 = hallucination, 0 = not. Rows with missing labels (e.g. refusals)
        must be removed before calling.
    alpha : float in (0, 1)
        Allowed miss rate, alpha = 1 - target_recall.

    Returns
    -------
    float
        tau. Flag an answer as a hallucination when ``score >= tau``.
        ``-inf`` means the calibration set is too small for this alpha and
        everything must be flagged.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)

    if scores.ndim != 1 or labels.shape != scores.shape:
        raise ValueError(
            f"scores and labels must be 1-D arrays of the same length; "
            f"got shapes {scores.shape} and {labels.shape}"
        )
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")
    if np.isnan(scores).any():
        raise ValueError("scores contain NaN; drop or fix those rows first")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError(
            "labels must be 0 or 1 (no NaN); drop unlabeled rows such as "
            "refusals before calibrating"
        )

    positives = np.sort(scores[labels == 1])
    n = len(positives)
    if n == 0:
        raise ValueError("no hallucinated (label == 1) examples to calibrate on")

    k = int(np.floor(alpha * (n + 1) + _FLOOR_TOL))
    if k == 0:
        return -np.inf
    return float(positives[k - 1])
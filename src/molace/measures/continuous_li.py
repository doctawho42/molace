"""The two estimators a continuous label informativeness would need, and why it cannot be built.

Label informativeness is LI = I(y_xi, y_eta) / H(y_xi), with (xi, eta) a random edge under a random
orientation. For a categorical label both parts are entropies of a discrete variable and the ratio
is a clean fraction of the information that a neighbour's label carries. Replacing the label with a
continuous one breaks the construction in exactly one place, and it is worth being able to show
which place rather than asserting it:

  * The NUMERATOR survives. Mutual information is defined for continuous variables and is invariant
    under any smooth invertible reparametrisation of either argument, so I(cY, cY') = I(Y, Y').

  * The DENOMINATOR does not. Differential entropy is not an entropy: it can be negative, and it
    shifts under rescaling, h(cY) = h(Y) + log|c|. So the ratio depends on the units the label is
    measured in, and for every label there is a scale at which h = 0 and the ratio diverges.

Both estimators here are the standard nearest-neighbour ones. Kozachenko-Leonenko for the entropy,
and Kraskov-Stoegbauer-Grassberger (estimator 1) for the mutual information.
"""
from __future__ import annotations

import numpy as np
from scipy.special import digamma
from sklearn.neighbors import NearestNeighbors

DEFAULT_K = 3
TIE_POLICIES = ("raise", "nudge")


def differential_entropy(y: np.ndarray, k: int = DEFAULT_K, tie_policy: str = "raise",
                         nudge: float = 1e-6) -> float:
    """Kozachenko-Leonenko estimate of h(Y) in nats, for a one-dimensional sample.

    TIES ARE NOT A DETAIL HERE. A duplicated value gives a zero k-th neighbour distance and
    log(0) = -inf, so something has to give. The two honest options are both offered and neither is
    the default silently:

      * "raise" (default) refuses the sample. On an assay label rounded to a reporting grid, that
        is usually the right answer.
      * "nudge" replaces zero distances by `nudge` times the smallest positive one. THE RESULT IS
        THEN A FUNCTION OF `nudge`, roughly log-linear in it: on this project's own labels, moving
        it from 1e-2 to 1e-15 moved h from -0.22 to -7.67 and from -0.43 to -20.67. Any LEVEL
        quoted from a nudged estimate is a property of the constant. What survives is the
        DIFFERENCE between two scales of the same sample, because the nudge scales with the data:
        h(cY) - h(Y) = log|c| holds exactly whatever `nudge` is.
    """
    if tie_policy not in TIE_POLICIES:
        raise KeyError(f"unknown tie_policy {tie_policy!r}; expected one of {TIE_POLICIES}")
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    n = len(y)
    if n <= k:
        raise ValueError(f"need more than k={k} samples, got {n}")
    nn = NearestNeighbors(n_neighbors=k + 1).fit(y)
    d, _ = nn.kneighbors(y)
    eps = d[:, k]
    n_tied = int((eps <= 0).sum())
    if n_tied:
        if tie_policy == "raise":
            raise ValueError(
                f"{n_tied} of {n} samples have a zero {k}-th neighbour distance, so the estimate "
                f"would be -inf. Pass tie_policy='nudge' to proceed, and then do not quote the "
                f"level: it depends on the nudge constant."
            )
        pos = eps[eps > 0]
        if pos.size == 0:
            raise ValueError("every sample is a duplicate; differential entropy is undefined here")
        eps = np.maximum(eps, pos.min() * nudge)
    # d = 1, so the volume of the unit ball is 2
    return float(digamma(n) - digamma(k) + np.log(2.0) + np.mean(np.log(eps)))


def mutual_information(a: np.ndarray, b: np.ndarray, k: int = DEFAULT_K,
                       standardise: bool = False) -> float:
    """Kraskov-Stoegbauer-Grassberger estimate of I(A;B) in nats, both one-dimensional.

    `standardise` divides each variable by its own spread first. Read what that does carefully,
    because an earlier version of this module had it on by default and then presented the result as
    evidence:

    KSG uses the max-norm in the joint space, so scaling ONE variable changes the joint geometry and
    the estimate moves, while the true mutual information does not. Standardising removes that, and
    thereby makes the estimator affine-invariant BY CONSTRUCTION. An invariance observed with
    `standardise=True` is therefore imposed by these two lines, not measured -- a stub returning
    zero would pass the same check. It is off by default so that the estimator's own behaviour is
    what gets reported.

    This estimator has no tie handling. On a sample with heavy duplication it stops behaving like a
    mutual information at all: plain duplication inflates it by orders of magnitude (0.048 to 5.5 at
    a multiplicity of five) and rounding to a grid drives it negative, which no true mutual
    information is. Callers that might see ties should check the sign; `is_degenerate` says so.
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} against {b.shape}")
    n = len(a)
    if n <= k:
        raise ValueError(f"need more than k={k} samples, got {n}")
    if standardise:
        a = (a - a.mean()) / (a.std() or 1.0)
        b = (b - b.mean()) / (b.std() or 1.0)
    joint = np.column_stack([a, b])

    nn = NearestNeighbors(n_neighbors=k + 1, metric="chebyshev").fit(joint)
    d, _ = nn.kneighbors(joint)
    eps = d[:, k]

    def count_within(x: np.ndarray) -> np.ndarray:
        order = np.argsort(x)
        xs = x[order]
        lo = np.searchsorted(xs, x - eps, side="left")
        hi = np.searchsorted(xs, x + eps, side="right")
        return (hi - lo - 1).astype(float)   # exclude the point itself

    nx, ny = count_within(a), count_within(b)
    return float(digamma(k) + digamma(n) - np.mean(digamma(nx + 1.0) + digamma(ny + 1.0)))


def is_degenerate(value: float) -> bool:
    """True when a mutual-information estimate cannot be a mutual information.

    I(A;B) >= 0 always. A negative estimate means the estimator has broken down -- on this data,
    because the sample is heavily tied -- and the value must not be reported as a measurement or
    divided by anything.
    """
    return not np.isfinite(value) or value < 0.0

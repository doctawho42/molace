"""Monotone label transforms, for the holdout sensitivity analysis.

The pre-registered holdout mixed endpoints already expressed as logarithms with endpoints in
raw concentration units, and it standardised nothing. Two separate things went wrong, and only
one of them was named in the report:

1. The gap was reported in the label's own units, so a task measured in hours and a task
   measured in log units are not comparable. Dividing by the label's standard deviation fixes
   that, and fixing only that makes the correlation stronger rather than weaker.

2. The statistic is a Pearson correlation across edge endpoints, and on a heavy-tailed label it
   reads low: on `vdss_lombardo`, whose label spans 0.01 to 700 L/kg, the raw-label value is 0.082
   while the same graph with the label's plain ranks gives 0.426, and with the van der Waerden
   scores this module produces, 0.391. The low assortativities in the holdout were a property of
   the label's shape, not of the graph. (Note this is what was measured here, not a general law:
   outliers can inflate a Pearson correlation as readily as deflate it.)

Both transforms here are monotone, so neither changes which molecules are more active than
which; they change the scale on which "similar" is measured.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def log10_if_positive(y: np.ndarray) -> tuple[np.ndarray, bool]:
    """Take a base-10 logarithm when the label's domain allows it, and say whether it did.

    The admissibility test is the label's own domain, not its shape: a label that reaches zero
    or goes negative is already a logarithm (or a signed quantity), and taking another one is
    undefined. On the nine TDC regression endpoints this rule separates the four that are
    already logarithms from the five in raw units without a threshold to choose.
    """
    y = np.asarray(y, dtype=float)
    if y.min() <= 0.0:
        return y, False
    return np.log10(y), True


def rank_to_normal(y: np.ndarray) -> np.ndarray:
    """Map a label to standard-normal quantiles through its ranks.

    This is the strongest removal of units available: it applies the same rule to every task,
    needs no admissibility branch, and leaves every task with unit variance, so a gap expressed
    in these units is dimensionless by construction. Ties share a value, which is why the rank
    is averaged rather than ordinal.
    """
    y = np.asarray(y, dtype=float)
    if y.std() == 0.0:
        raise ValueError("cannot rank-normalise a constant label")
    r = stats.rankdata(y)
    return np.asarray(stats.norm.ppf(r / (len(r) + 1.0)), dtype=float)

"""The label transforms the holdout sensitivity analysis runs on.

These exist because the holdout as pre-registered mixed log and raw endpoints, and the
statistic -- a Pearson correlation -- is deflated by a heavy tail. Both transforms are
monotone, so neither changes the ordering of the label; they change only its scale.
"""
from __future__ import annotations

import numpy as np
import pytest

from molace.analysis.transforms import log10_if_positive, rank_to_normal


def test_log10_applied_when_every_label_is_strictly_positive():
    y = np.array([1.0, 10.0, 100.0])
    out, applied = log10_if_positive(y)
    assert applied is True
    assert np.allclose(out, [0.0, 1.0, 2.0])


def test_log10_refused_when_a_label_is_zero_or_negative():
    for y in (np.array([0.0, 1.0, 2.0]), np.array([-1.0, 1.0, 2.0])):
        out, applied = log10_if_positive(y)
        assert applied is False
        assert np.allclose(out, y), "a refused transform must return the label untouched"


def test_rank_to_normal_is_standardised_and_order_preserving():
    rng = np.random.default_rng(0)
    y = rng.lognormal(size=500)
    out = rank_to_normal(y)
    assert abs(out.mean()) < 0.05
    assert 0.9 < out.std() < 1.1
    assert (np.argsort(y) == np.argsort(out)).all(), "the transform must be monotone"


def test_rank_to_normal_gives_tied_labels_the_same_value():
    out = rank_to_normal(np.array([5.0, 1.0, 5.0, 2.0]))
    assert out[0] == pytest.approx(out[2])


def test_rank_to_normal_refuses_a_constant_label():
    with pytest.raises(ValueError, match="constant"):
        rank_to_normal(np.array([3.0, 3.0, 3.0]))

"""A candidate continuous label informativeness: the categorical one on quantile bins.

The pre-registration forbade binning a continuous label because the VALUE moves -- measured, 7.5x
for adjusted homophily and 100x for LI across bin counts. Quantile binning does not fix that, and
is not claimed to. What it fixes is the two things that make a binning arbitrary:

  * WHERE the cuts go is not chosen -- equal frequency by rank, so no analyst picks a boundary;
  * the result is invariant under every strictly monotone relabelling, which is exactly the units
    problem that killed this project's holdout (the same data in hours and in seconds).

One declared parameter is left, the number of bins. Whether that is survivable is an empirical
question about ORDERINGS, not values, and these tests pin the properties the question rests on.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from molace.measures.quantile_li import quantile_bins, quantile_label_informativeness


def _g(n=120, seed=3):
    return nx.random_geometric_graph(n, 0.25, seed=seed)


def test_bins_are_equal_frequency_within_one():
    y = np.random.default_rng(0).normal(size=100)
    counts = np.bincount(quantile_bins(y, 4))[1:]
    assert counts.max() - counts.min() <= 1


def test_the_binning_is_invariant_under_every_strictly_monotone_relabelling():
    rng = np.random.default_rng(1)
    y = rng.lognormal(sigma=1.5, size=200)
    base = quantile_bins(y, 8)
    for f in (np.log, np.sqrt, lambda v: 3.0 * v + 7.0, lambda v: v**3, lambda v: 60.0 * v):
        assert (quantile_bins(f(y), 8) == base).all(), f"{f} changed the bins"


def test_the_measure_inherits_that_invariance():
    """The units failure this project hit: hours against seconds must give the same number."""
    rng = np.random.default_rng(2)
    g = _g()
    y = rng.lognormal(sigma=1.5, size=g.number_of_nodes())
    hours = quantile_label_informativeness(g, y, 8).value
    seconds = quantile_label_informativeness(g, 3600.0 * y, 8).value
    assert hours == pytest.approx(seconds, abs=1e-12)


def test_never_produces_more_classes_than_asked_for():
    y = np.random.default_rng(3).normal(size=50)
    for b in (2, 3, 5, 16):
        assert len(np.unique(quantile_bins(y, b))) <= b


def test_a_label_with_fewer_distinct_values_than_bins_collapses_rather_than_inventing_classes():
    y = np.array([1.0] * 10 + [2.0] * 10)
    assert len(np.unique(quantile_bins(y, 8))) <= 2


def test_the_value_lies_in_the_unit_interval():
    rng = np.random.default_rng(4)
    g = _g()
    y = rng.normal(size=g.number_of_nodes())
    v = quantile_label_informativeness(g, y, 4).value
    assert 0.0 <= v <= 1.0 + 1e-12


def test_a_smooth_label_scores_above_a_shuffled_one():
    g = nx.path_graph(200)
    y = np.arange(200, dtype=float)
    smooth = quantile_label_informativeness(g, y, 4).value
    shuffled = quantile_label_informativeness(g, np.random.default_rng(5).permutation(y), 4).value
    assert smooth > shuffled + 0.1


def test_the_value_really_does_move_with_the_bin_count():
    """The instability the pre-registration objected to. It is not fixed here, and must not be hidden."""
    rng = np.random.default_rng(6)
    g = _g(300, seed=9)
    y = rng.normal(size=g.number_of_nodes())
    vals = [quantile_label_informativeness(g, y, b).value for b in (2, 4, 8, 16)]
    assert max(vals) > 2 * min(vals), f"expected the value to move; got {vals}"


def test_a_constant_label_is_refused():
    with pytest.raises(ValueError):
        quantile_label_informativeness(_g(), np.ones(_g().number_of_nodes()), 4)

"""A candidate continuous label informativeness: the categorical measure on quantile bins.

`molace.measures.continuous_li` shows why the obvious construction fails. LI = I/H needs a
normaliser, and substituting differential entropy gives a quantity that shifts by log|c| when the
label is rescaled -- the same dataset in hours and in seconds reads +0.056 and -0.281, with a unit
in between where it is undefined.

Binning sidesteps the normaliser entirely, since a binned label is categorical and H is a genuine
entropy. The pre-registration forbade it because the VALUE moves with the bin count, measured at
100x for LI. That objection stands and is not answered here. What quantile binning answers is the
other two objections, and they are the ones that make a binning arbitrary rather than merely
parameterised:

  * The cut POSITIONS are not chosen. Equal frequency by rank leaves no boundary for an analyst to
    place, and therefore nothing to tune after seeing a result.
  * The result is invariant under every strictly monotone relabelling, because it depends on the
    label only through its ranks. That is exactly the failure that invalidated this project's
    holdout, fixed by construction rather than by a transform chosen afterwards.

What is left is one declared integer. Whether a measure with one declared integer is usable for
comparing datasets is an empirical question about whether the ORDERING of datasets is stable in it,
not about whether the value is -- and the value is known not to be. See
`scripts/quantile_li_study.py`.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
from scipy import stats

from molace.measures.assortativity import MeasureResult
from molace.measures.informativeness import li_from_classes


def quantile_bins(y: np.ndarray, n_bins: int) -> np.ndarray:
    """Equal-frequency bins by rank, labelled 1..n_bins. Depends on y only through its ranks.

    Ties take an average rank, so a tied group that straddles a boundary can split across two bins.
    That is the only place the label's values enter at all, and it cannot be steered by rescaling.
    """
    if n_bins < 2:
        raise ValueError(f"n_bins must be at least 2, got {n_bins}")
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        raise ValueError("cannot bin an empty label")
    r = stats.rankdata(y, method="average")
    b = np.ceil(r / len(r) * n_bins).astype(int)
    return np.clip(b, 1, n_bins)


def quantile_label_informativeness(g: nx.Graph, y: np.ndarray, n_bins: int) -> MeasureResult:
    """LI of the label discretised into n_bins equal-frequency bins.

    This goes through `li_from_classes`, not through the categorical guard, and that is deliberate:
    the guard refuses a binned continuous label because binning is usually an undeclared choice,
    while here it is the declared subject of the measurement. Bypassing a guard quietly would be the
    bad version of this; the module says so in its own docstring and in this one.
    """
    bins = quantile_bins(y, n_bins)
    if len(np.unique(bins)) < 2:
        raise ValueError(
            f"the label collapses into a single bin at n_bins={n_bins}: it has "
            f"{len(np.unique(y))} distinct values"
        )
    if len(y) != g.number_of_nodes():
        raise ValueError(f"length mismatch: {len(y)} labels for {g.number_of_nodes()} nodes")
    if g.number_of_edges() == 0:
        raise ValueError("label informativeness is undefined on a graph with no edges")
    return li_from_classes(g, bins)

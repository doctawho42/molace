"""Label informativeness, eq. (3) of arXiv 2209.06177: LI = I(y_xi, y_eta) / H(y_xi).

Over a uniformly random edge with a random orientation, so the marginal is degree-weighted,
p(k) = D_k / 2|E|. Categorical only, and there is no published continuous analogue: the string
"label informativeness" does not appear in GraphLand at all. Writing one would be an unbudgeted
methods contribution with nothing to validate against, so this module refuses instead.
"""
from __future__ import annotations

import networkx as nx
import numpy as np

from molace.measures.assortativity import MeasureResult
from molace.measures.homophily import _check, _coverage


def label_informativeness(g: nx.Graph, labels: np.ndarray) -> MeasureResult:
    """The categorical measure. The guard in `_check` refuses a binned continuous label."""
    return li_from_classes(g, _check(g, labels))


def li_from_classes(g: nx.Graph, a: np.ndarray) -> MeasureResult:
    """LI of an ALREADY VALIDATED class vector, with no guard of its own.

    Split out so that `measures.quantile_li` can declare its binning instead of sneaking past the
    guard. The guard exists to stop a continuous label being binned by accident or convenience; a
    caller whose entire subject is the binning states that in its own module and comes through here.
    Nothing else should.
    """
    nodes = sorted(g.nodes())
    pos = {v: i for i, v in enumerate(nodes)}
    classes = np.unique(a)
    idx = {c: i for i, c in enumerate(classes)}
    k = len(classes)
    joint = np.zeros((k, k), dtype=float)
    for u, v in g.edges():
        i, j = idx[a[pos[u]]], idx[a[pos[v]]]
        joint[i, j] += 1.0
        joint[j, i] += 1.0
    joint /= joint.sum()
    p = joint.sum(axis=1)
    h = -float(np.sum(p[p > 0] * np.log(p[p > 0])))
    if h == 0.0:
        raise ValueError("label informativeness is undefined: the label entropy is zero")
    mi = 0.0
    for i in range(k):
        for j in range(k):
            if joint[i, j] > 0.0:
                mi += joint[i, j] * np.log(joint[i, j] / (p[i] * p[j]))
    cov, n_used = _coverage(g)
    return MeasureResult(float(mi / h), cov, n_used)

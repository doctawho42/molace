"""Edge and adjusted homophily, categorical labels only.

Definitions from Platonov, Kuznedelev, Babenko, Prokhorenkova, "Characterizing Graph Datasets
for Node Classification: Homophily-Heterophily Dichotomy and Beyond", arXiv 2209.06177, eq. (2):

    h_adj = (h_edge - sum_k (D_k / 2|E|)^2) / (1 - sum_k (D_k / 2|E|)^2)

where h_edge is the fraction of edges joining same-class endpoints and D_k is the summed degree
of class-k nodes.

The guard below is the point of this module. Binning a continuous target was measured to move
adjusted homophily 7.5x and label informativeness 100x on one real target at one fixed graph,
which would make the headline number a function of a free parameter the project claims to have
fixed. So binning is refused in code rather than discouraged in prose.
"""
from __future__ import annotations

import networkx as nx
import numpy as np

from molace.measures.assortativity import MeasureResult

MAX_CLASSES = 8

_REFUSAL = (
    "adjusted homophily and label informativeness are defined for categorical labels only "
    "(Platonov et al., arXiv 2209.06177: a class label y_v in {1..C}, class degree sums D_k, "
    "and a discrete mutual information). Binning a continuous target is forbidden by "
    "prereg/increment1.yaml: on CHEMBL2835_Ki, one graph and one bin count, swapping quantile "
    "for equal-width bins moved adjusted homophily 7.5x and label informativeness 100x. "
    "For a continuous target use measures.assortativity.target_assortativity."
)


def require_categorical(labels: np.ndarray) -> np.ndarray:
    """Accept a genuine categorical label; refuse anything that is a binning in disguise."""
    a = np.asarray(labels)
    if a.dtype.kind == "f" and not np.all(a == np.floor(a)):
        raise TypeError(_REFUSAL)
    a = a.astype(np.int64)
    if len(np.unique(a)) > MAX_CLASSES:
        raise TypeError(_REFUSAL + f" Got {len(np.unique(a))} distinct values.")
    return a


def _check(g: nx.Graph, labels: np.ndarray) -> np.ndarray:
    a = require_categorical(labels)
    if len(a) != g.number_of_nodes():
        raise ValueError(f"length mismatch: {len(a)} labels for {g.number_of_nodes()} nodes")
    if g.number_of_edges() == 0:
        raise ValueError("homophily is undefined on a graph with no edges")
    if len(np.unique(a)) < 2:
        raise ValueError(
            "the label takes only one class on this target, so homophily and label "
            "informativeness are 0/0; report the target as degenerate instead"
        )
    return a


def _coverage(g: nx.Graph) -> tuple[float, int]:
    used = sum(1 for _, d in g.degree() if d > 0)
    return used / g.number_of_nodes(), used


def edge_homophily(g: nx.Graph, labels: np.ndarray) -> MeasureResult:
    a = _check(g, labels)
    nodes = sorted(g.nodes())
    pos = {v: i for i, v in enumerate(nodes)}
    same = sum(1 for u, v in g.edges() if a[pos[u]] == a[pos[v]])
    cov, n_used = _coverage(g)
    return MeasureResult(same / g.number_of_edges(), cov, n_used)


def adjusted_homophily(g: nx.Graph, labels: np.ndarray) -> MeasureResult:
    a = _check(g, labels)
    nodes = sorted(g.nodes())
    pos = {v: i for i, v in enumerate(nodes)}
    two_m = 2.0 * g.number_of_edges()
    deg = dict(g.degree())
    h_edge = edge_homophily(g, labels).value
    baseline = 0.0
    for c in np.unique(a):
        d_k = sum(deg[v] for v in nodes if a[pos[v]] == c)
        baseline += (d_k / two_m) ** 2
    if baseline >= 1.0:
        raise ValueError("adjusted homophily is undefined: the degree baseline reached 1")
    cov, n_used = _coverage(g)
    return MeasureResult((h_edge - baseline) / (1.0 - baseline), cov, n_used)

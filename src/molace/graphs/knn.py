"""Degree-controlled similarity graph.

Primary construction, because at a fixed Tanimoto threshold the measured edge density across
the 30 MoleculeACE targets spans 98x and mean degree 82x, and on identical labels the
threshold-to-kNN switch moves adjusted homophily 12x. A headline correlation computed across
graphs of 98x differing density would be confounded with density.

Union symmetrisation gives every node degree AT LEAST k, not exactly k: a molecule appearing
in many neighbour lists accumulates extra edges. Mean degree is therefore still a covariate,
and the sweep records it.
"""
from __future__ import annotations

import networkx as nx
import numpy as np


def knn_graph(T: np.ndarray, k: int) -> nx.Graph:
    n = T.shape[0]
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if n < 2:
        raise ValueError(f"a kNN graph needs at least 2 molecules, got {n}")
    kk = min(k, n - 1)
    s = T.astype(np.float64).copy()
    np.fill_diagonal(s, -np.inf)
    # lexsort on (-similarity, index) makes the neighbour choice deterministic under ties.
    order = np.lexsort((np.tile(np.arange(n), (n, 1)), -s), axis=1)[:, :kk]
    g = nx.Graph()
    g.add_nodes_from(range(n))
    for i in range(n):
        for j in order[i]:
            j = int(j)
            if i != j:
                g.add_edge(i, j, weight=float(T[i, j]))
    return g

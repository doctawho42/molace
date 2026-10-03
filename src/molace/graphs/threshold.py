"""Tanimoto-threshold similarity graph: robustness check only, never primary.

Kept because the Chemical Space Network literature this project works inside is built on
threshold graphs, and because the density spread it produces is itself a reportable
measurement.
"""
from __future__ import annotations

import networkx as nx
import numpy as np


def threshold_graph(T: np.ndarray, tau: float) -> nx.Graph:
    n = T.shape[0]
    g = nx.Graph()
    g.add_nodes_from(range(n))
    iu = np.triu_indices(n, k=1)
    keep = T[iu] >= tau
    for u, v, w in zip(iu[0][keep], iu[1][keep], T[iu][keep]):
        g.add_edge(int(u), int(v), weight=float(w))
    return g

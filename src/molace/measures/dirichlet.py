"""Normalised Dirichlet energy of the label.

Presented as an import, not as a measure of the Prokhorenkova line: the string "dirichlet" does
not appear in any of their five relevant papers. It is defended on its own merits, namely that it
is defined for a continuous label, and its normalisation is declared as its one free parameter.

    E = sum_{(u,v) in E} (y_u - y_v)^2 / (|E| * var(y))

The denominator makes it comparable across targets with different label scales and edge counts.
"""
from __future__ import annotations

import networkx as nx
import numpy as np

from molace.measures.assortativity import MeasureResult


def dirichlet_energy(g: nx.Graph, y: np.ndarray) -> MeasureResult:
    y = np.asarray(y, dtype=float)
    if len(y) != g.number_of_nodes():
        raise ValueError(f"length mismatch: {len(y)} labels for {g.number_of_nodes()} nodes")
    if g.number_of_edges() == 0:
        raise ValueError("Dirichlet energy is undefined on a graph with no edges")
    var = float(np.var(y))
    if var == 0.0:
        raise ValueError("Dirichlet energy is undefined when the label is constant")
    nodes = sorted(g.nodes())
    pos = {v: i for i, v in enumerate(nodes)}
    total = sum((y[pos[u]] - y[pos[v]]) ** 2 for u, v in g.edges())
    used = sum(1 for _, d in g.degree() if d > 0)
    return MeasureResult(
        value=float(total / (g.number_of_edges() * var)),
        coverage=used / g.number_of_nodes(),
        n_used=used,
    )

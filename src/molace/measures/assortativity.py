"""Newman (2003) scalar target assortativity: the project's primary statistic.

This is the group's own choice for a continuous target. GraphLand: "To measure the similarity
of labels of connected nodes for regression datasets, we use target assortativity - the Pearson
correlation coefficient of target values between pairs of connected nodes", and GraphPFN
repeats it. It carries no binning parameter, and the Platonov et al. paper notes that adjusted
homophily "is known in graph analysis literature as assortativity coefficient", so this is the
continuous reduction of the categorical measure rather than a substitute for it.
"""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np


@dataclass(frozen=True)
class MeasureResult:
    """One graph statistic plus how much of the data it actually describes.

    coverage is the fraction of nodes that participate in at least one edge. It is carried on
    every measure because on sparse graphs a measure can describe a minority of the molecules
    and still look like a dataset-level number.
    """

    value: float
    coverage: float
    n_used: int


def target_assortativity(g: nx.Graph, y: np.ndarray) -> MeasureResult:
    y = np.asarray(y, dtype=float)
    if len(y) != g.number_of_nodes():
        raise ValueError(
            f"length mismatch: {len(y)} labels for {g.number_of_nodes()} nodes"
        )
    if g.number_of_edges() == 0:
        raise ValueError("target assortativity is undefined on a graph with no edges")
    nodes = sorted(g.nodes())
    pos = {v: i for i, v in enumerate(nodes)}
    ends = np.array([(y[pos[u]], y[pos[v]]) for u, v in g.edges()], dtype=float)
    a = np.concatenate([ends[:, 0], ends[:, 1]])
    b = np.concatenate([ends[:, 1], ends[:, 0]])
    if a.std() == 0.0:
        raise ValueError(
            "target assortativity is undefined when the label is constant over edge endpoints"
        )
    used = sum(1 for _, d in g.degree() if d > 0)
    return MeasureResult(
        value=float(np.corrcoef(a, b)[0, 1]),
        coverage=used / g.number_of_nodes(),
        n_used=used,
    )

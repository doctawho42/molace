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
from scipy import stats


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


def rank_assortativity(g: nx.Graph, y: np.ndarray) -> MeasureResult:
    """Newman's coefficient computed on the label's NODE ranks.

    Newman's coefficient is a Pearson correlation, so it measures LINEAR agreement between the two
    endpoints of an edge and is pulled around by a skewed label. Measured on this project's own
    holdout: on vdss_lombardo, whose label runs 0.01 to 700 L/kg with skewness 27, the Pearson
    version reads 0.082 and this one 0.426 on the identical graph. Any claim compared ACROSS
    datasets needs the version that does not depend on the label's shape.

    Ranks are averaged over ties, so the value depends only on the ordering the label induces.

    Be precise about which rank statistic this is, because there are two and they are not equal.
    This one ranks the |V| NODE labels once and then takes Pearson across edge endpoints. The
    Spearman of the edge-endpoint SAMPLE ranks that sample instead, which is degree-weighted, so a
    high-degree molecule's label gets a different rank. Measured on this project's graphs the two
    agree to 0.0002-0.005 (CHEMBL1862_Ki: 0.6188 against 0.6146), and no conclusion here turns on
    the difference -- but the node-rank version is what this function computes, and
    prereg/increment2_separation.yaml describes the other one. That deviation is recorded in
    results/results_increment2_separation.md.
    """
    y = np.asarray(y, dtype=float)
    if len(y) != g.number_of_nodes():
        raise ValueError(
            f"length mismatch: {len(y)} labels for {g.number_of_nodes()} nodes"
        )
    if np.all(y == y[0]):
        raise ValueError("rank assortativity is undefined for a constant label")
    return target_assortativity(g, np.asarray(stats.rankdata(y), dtype=float))

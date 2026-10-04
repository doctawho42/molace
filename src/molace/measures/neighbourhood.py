"""Second-order label agreement, and the closed form it puts under the kNN floor.

Target assortativity answers "do the two ends of an edge agree". This module answers the question
one step out: "do two nodes that share a neighbour agree". The second quantity is not decoration.
Expanding the floor's squared error gives

    mean_v (y_v - mean_{u in N(v)} y_u)^2 / sigma^2  =  1 + 1/k + (k-1)/k * rho_nn - 2 * r

with r target assortativity and rho_nn the shared-neighbour correlation. On a k-regular graph that
is an identity, exact to machine precision: when every degree is k, an edge endpoint drawn uniformly
has the label's own mean and variance, and so does a shared-neighbour endpoint, because every node
is counted k(k-1) times. tests/test_neighbourhood.py pins that exactness on regular graphs rather
than on this project's data, so a broken derivation fails the suite and not a results document.

On an irregular graph the degrees vary and it becomes an approximation. Measured on this project's
40 ChEMBL targets at k = 10 it predicts the floor's relative error to 0.0256 mean absolute error
over a range of 0.108 to 0.819; see results/results_increment4.md.

Why it is worth having in the library at all: both statistics are read off the graph and the label
with nothing trained, so the right-hand side is computable before any model exists. For the floor,
"the homophily measure predicts attainable accuracy" is therefore a theorem rather than a finding,
which is exactly what makes that arm a positive control.
"""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

from molace.measures.assortativity import target_assortativity

MAX_PAIRS = 4_000_000
SEED = 0


@dataclass(frozen=True)
class NeighbourAgreement:
    """One correlation plus how much of the graph it was taken over.

    n_pairs counts PAIR OBSERVATIONS in the symmetrised multiset, not nodes, which is why this is a
    type of its own instead of the MeasureResult the node-level measures return. A pair is counted
    once per common neighbour, because that is what the derivation's sum over v does.
    """

    value: float
    coverage: float
    n_pairs: int


def shared_neighbour_correlation(
    g: nx.Graph, y: np.ndarray, max_pairs: int = MAX_PAIRS, seed: int = SEED
) -> NeighbourAgreement:
    """Pearson correlation of the label over pairs of nodes that share a neighbour.

    Above max_pairs the multiset is subsampled with a seeded generator, so the number is
    reproducible but not exact; the cap is reported through n_pairs.
    """
    y = np.asarray(y, dtype=float)
    if len(y) != g.number_of_nodes():
        raise ValueError(f"length mismatch: {len(y)} labels for {g.number_of_nodes()} nodes")
    if np.all(y == y[0]):
        raise ValueError("shared-neighbour correlation is undefined for a constant label")
    total = sum(d * (d - 1) // 2 for _, d in g.degree())
    if total == 0:
        raise ValueError(
            "shared-neighbour correlation is undefined: no node shares a neighbour with another"
        )
    nodes = sorted(g.nodes())
    pos = {v: i for i, v in enumerate(nodes)}
    keep = 1.0 if total <= max_pairs else max_pairs / total
    rng = np.random.default_rng(seed)
    # Enumerated per node with triu_indices rather than a Python double loop: at k = 40 a target of
    # 8000 molecules carries about 10 million pairs, and the loop version made that minutes per graph.
    left: list[np.ndarray] = []
    right: list[np.ndarray] = []
    for v in nodes:
        nb = np.fromiter((pos[u] for u in g.neighbors(v)), dtype=np.int64)
        if len(nb) < 2:
            continue
        i, j = np.triu_indices(len(nb), k=1)
        li, ri = nb[i], nb[j]
        if keep < 1.0:
            m = rng.random(len(li)) <= keep
            li, ri = li[m], ri[m]
        if len(li):
            left.append(li)
            right.append(ri)
    if not left:
        raise ValueError(
            "shared-neighbour correlation is undefined: the subsample kept no pair at all"
        )
    a, b = np.concatenate(left), np.concatenate(right)
    first = np.concatenate([y[a], y[b]])
    second = np.concatenate([y[b], y[a]])
    if first.std() == 0.0:
        raise ValueError(
            "shared-neighbour correlation is undefined: the label is constant over these pairs"
        )
    seen = np.unique(np.concatenate([a, b]))
    return NeighbourAgreement(
        value=float(np.corrcoef(first, second)[0, 1]),
        coverage=len(seen) / g.number_of_nodes(),
        n_pairs=2 * len(a),
    )


def predicted_floor_error(
    g: nx.Graph, y: np.ndarray, k: int, max_pairs: int = MAX_PAIRS, seed: int = SEED
) -> float:
    """The identity's right-hand side: the floor's relative error, with nothing trained.

    k is the floor's own averaging width, which on a union-symmetrised kNN graph is NOT the mean
    degree. Passing the mean degree instead is a different and worse prediction, measured in
    results/results_increment4.md, and the choice is the caller's to make explicitly.
    """
    if k < 2:
        raise ValueError(f"k must be at least 2 for the identity to have its second term, got {k}")
    r = target_assortativity(g, y).value
    rho = shared_neighbour_correlation(g, y, max_pairs=max_pairs, seed=seed).value
    return 1.0 + 1.0 / k + (k - 1.0) / k * rho - 2.0 * r

"""Unbiased homophily: the measure the group's current benchmarks actually report.

Mironov, Prokhorenkova, "Revisiting Graph Homophily Measures", arXiv 2412.09663, LoG 2024,
PMLR v269. GraphLand states verbatim that this paper "constructed the first known homophily
measure that satisfies all these properties - unbiased homophily. Thus, in our work, we use
unbiased homophily", and GraphPFN does the same. Adjusted homophily is the group's 2023
vocabulary and its own source paper concedes it fails minimal agreement.

Equation (3) of 2412.09663, transcribed from the paper, not from memory:

    h_unb(C) := h_unb^0(C) = sum_{i<j} ( sqrt(c_ii * c_jj) - c_ij )
                             -------------------------------------
                             sum_{i<j} ( sqrt(c_ii * c_jj) + c_ij )

where C is the normalised class adjacency matrix: c_ii is the fraction of edges inside class i
and c_ij (i != j) the fraction of edges between classes i and j. The recommended version is
alpha = 0, which is the form above; a parameterised h_unb^alpha exists for alpha > 0 and the
paper advises the simpler form for practice. GraphLand uses this recommended form, so there is
no free parameter in our hands.

Normalisation, which the formula depends on and the paper's prose leaves implicit: C is built by
counting both orientations of every edge and dividing by their total, 2|E|. That gives
c_ii = E_ii / |E| and c_ij = E_ij / (2|E|), the convention under which the paper's own stated
values come out right -- a balanced random labelling gives 0, perfect separation gives +1, and a
fully heterophilous labelling gives -1. It is the same joint matrix label informativeness uses.
"""
from __future__ import annotations

import networkx as nx
import numpy as np

from molace.measures.assortativity import MeasureResult
from molace.measures.homophily import _check, _coverage


def unbiased_homophily(g: nx.Graph, labels: np.ndarray) -> MeasureResult:
    a = _check(g, labels)
    nodes = sorted(g.nodes())
    pos = {v: i for i, v in enumerate(nodes)}
    classes = np.unique(a)
    idx = {c: i for i, c in enumerate(classes)}
    k = len(classes)

    c = np.zeros((k, k), dtype=float)
    for u, v in g.edges():
        i, j = idx[a[pos[u]]], idx[a[pos[v]]]
        c[i, j] += 1.0
        c[j, i] += 1.0
    c /= c.sum()

    num = 0.0
    den = 0.0
    for i in range(k):
        for j in range(i + 1, k):
            geo = float(np.sqrt(c[i, i] * c[j, j]))
            num += geo - c[i, j]
            den += geo + c[i, j]
    if den == 0.0:
        raise ValueError(
            "unbiased homophily is undefined here: every off-diagonal pair contributes zero, "
            "which happens when one class has neither intra-class nor inter-class edges"
        )
    cov, n_used = _coverage(g)
    return MeasureResult(value=float(num / den), coverage=cov, n_used=n_used)

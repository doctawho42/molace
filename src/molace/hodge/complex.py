"""Boundary operators of the clique 2-complex, and the dimension census.

Edges are (u, v) with u < v, oriented u -> v: B1[u, e] = -1, B1[v, e] = +1.
Triangles are (i, j, k) with i < j < k and contribute +1 to (i,j), +1 to (j,k), -1 to (i,k).
With those signs B1 @ B2 = 0 identically, which is asserted as a test, and that identity is why
the gradient and curl subspaces are orthogonal.

The census reports what is exact as exact and what is estimated as estimated. dim_gradient and
dim_cycle_space are cheap and exact. Splitting the cycle space into curl and harmonic needs
rank(B2); that is computed exactly only on small graphs and estimated above a stated edge count,
because reporting an estimate as exact is how a measurement becomes a false finding.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import scipy.sparse as sp


@dataclass
class Complex:
    B1: sp.csr_matrix
    B2: sp.csr_matrix
    nodes: list
    edges: list
    triangles: list
    triangles_total: int
    triangles_sampled: bool = False


def _triangles(g: nx.Graph) -> list[tuple[int, int, int]]:
    out = []
    adj = {v: set(g.neighbors(v)) for v in g}
    for v in sorted(g.nodes()):
        higher = sorted(u for u in adj[v] if u > v)
        for a, b in itertools.combinations(higher, 2):
            if b in adj[a]:
                out.append((v, a, b))
    return out


def build(g: nx.Graph, triangle_budget: int | None = None,
          require_triangles: bool = True, seed: int = 0) -> Complex:
    nodes = sorted(g.nodes())
    nidx = {v: i for i, v in enumerate(nodes)}
    edges = sorted(tuple(sorted(e)) for e in g.edges())
    eidx = {e: i for i, e in enumerate(edges)}

    tris = _triangles(g)
    total = len(tris)
    sampled = False
    if require_triangles and total == 0:
        raise ValueError(
            "this graph has no triangles, so the curl subspace is empty by construction and a "
            "measured curl of zero would be an artefact of the graph rather than a finding"
        )
    if triangle_budget is not None and total > triangle_budget:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(total, size=triangle_budget, replace=False))
        tris = [tris[i] for i in keep]
        sampled = True

    rows, cols, vals = [], [], []
    for (u, v), j in eidx.items():
        rows += [nidx[u], nidx[v]]
        cols += [j, j]
        vals += [-1.0, 1.0]
    B1 = sp.csr_matrix((vals, (rows, cols)), shape=(len(nodes), len(edges)))

    rows, cols, vals = [], [], []
    for t, (i, j, k) in enumerate(tris):
        for e, s in (((i, j), 1.0), ((j, k), 1.0), ((i, k), -1.0)):
            rows.append(eidx[e])
            cols.append(t)
            vals.append(s)
    B2 = sp.csr_matrix((vals, (rows, cols)), shape=(len(edges), len(tris)))

    return Complex(B1=B1, B2=B2, nodes=nodes, edges=edges, triangles=tris,
                   triangles_total=total, triangles_sampled=sampled)


def _rank_exact(B2: sp.csr_matrix) -> int:
    if B2.shape[1] == 0:
        return 0
    return int(np.linalg.matrix_rank(B2.toarray(), tol=1e-8))


def _rank_estimated(B2: sp.csr_matrix, tol: float = 1e-8, seed: int = 0) -> int:
    """Randomised range-finder estimate of rank(B2), capped by the cycle space dimension."""
    if B2.shape[1] == 0:
        return 0
    rng = np.random.default_rng(seed)
    probe = min(B2.shape[1], 2048)
    omega = rng.normal(size=(B2.shape[1], probe))
    y = np.asarray(B2 @ omega)
    s = np.linalg.svd(y, compute_uv=False)
    return int((s > tol * max(1.0, s[0])).sum())


def census(g: nx.Graph, triangle_budget: int | None = None,
           exact_rank_max_edges: int = 5000, require_triangles: bool = True,
           seed: int = 0) -> dict:
    c = build(g, triangle_budget=triangle_budget, require_triangles=require_triangles, seed=seed)
    n, m = len(c.nodes), len(c.edges)
    comps = nx.number_connected_components(g)
    dim_grad = n - comps
    dim_cycle = m - n + comps
    if m <= exact_rank_max_edges:
        dim_curl, method, tol = _rank_exact(c.B2), "exact", None
    else:
        tol = 1e-8
        dim_curl, method = _rank_estimated(c.B2, tol=tol, seed=seed), "estimated"
    dim_curl = min(dim_curl, dim_cycle)
    out = {
        "n_nodes": n, "n_edges": m, "n_components": comps,
        "n_triangles": len(c.triangles), "n_triangles_total": c.triangles_total,
        "triangles_sampled": c.triangles_sampled,
        "dim_gradient": dim_grad, "dim_cycle_space": dim_cycle,
        "dim_curl": dim_curl, "dim_harmonic": dim_cycle - dim_curl,
        "rank_method": method,
    }
    if tol is not None:
        out["rank_tolerance"] = tol
    return out

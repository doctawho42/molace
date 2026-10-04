"""Boundary operators of the clique 2-complex, and the dimension census.

Edges are (u, v) with u < v, oriented u -> v: B1[u, e] = -1, B1[v, e] = +1.
Triangles are (i, j, k) with i < j < k and contribute +1 to (i,j), +1 to (j,k), -1 to (i,k).
With those signs B1 @ B2 = 0 identically, which is asserted as a test, and that identity is why
the gradient and curl subspaces are orthogonal.

The census reports what it can prove and refuses to guess the rest. dim_gradient and
dim_cycle_space are cheap and exact. Splitting the cycle space into curl and harmonic needs rank(B2),
which is computed exactly where a dense SVD is affordable and reported as unavailable otherwise.

The first version offered a randomised range-finder estimate above that threshold. It was wrong in a
dangerous direction: a range finder cannot return a rank above its probe width, so on CHEMBL1862_Ki
it returned dim_curl = 2048 exactly, the probe width, and inflated the harmonic part to 2866 against a
true value of 491 -- a factor of 5.8. The estimate pointed the same way as the finding the project was
looking for, which is the worst property an estimator can have, so it is gone.

The census here still refuses to guess, and that refusal is now mostly moot: `molace.hodge.nullity`
computes the same split exactly and with a certificate, by taking the nullity of the edge Laplacian
instead of the rank of B2 -- the small number instead of the large one. Prefer it. What remains here
is the cheap, exact part (gradient and cycle-space dimensions) plus a dense rank where it is
affordable, which is what validated the nullity route in the first place.
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


def check_boundary_identity(B1: sp.csr_matrix, B2: sp.csr_matrix, tol: float = 1e-12) -> None:
    """Assert B1 @ B2 == 0, the identity every projection downstream rests on.

    This is where wrong orientation signs must be caught, because nothing downstream catches them.
    The pointwise-floor control cannot: a pure gradient flow leaves a zero residual, so the curl
    solve returns zero whatever B2 holds. Nor does breaking the triangle signs disturb the
    decomposition's arithmetic -- measured, with all three triangle edges made positive so that
    max|B1 @ B2| = 2.0, the gradient and curl components stayed orthogonal to 2e-15 and the squared
    norms still summed to the flow's energy within 1.3e-15. So the identity is checked on every
    complex that gets built, not left to one test on one graph.
    """
    if B2.shape[1] == 0:
        return
    worst = float(abs(B1 @ B2).max())
    if worst > tol:
        raise ValueError(
            f"boundary composition is not zero: max|B1 @ B2| = {worst:g}. The orientation signs are "
            "wrong, and no statistic computed from this complex would reveal it."
        )


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

    check_boundary_identity(B1, B2)
    return Complex(B1=B1, B2=B2, nodes=nodes, edges=edges, triangles=tris,
                   triangles_total=total, triangles_sampled=sampled)


def _rank_exact(B2: sp.csr_matrix) -> int:
    if B2.shape[1] == 0:
        return 0
    return int(np.linalg.matrix_rank(B2.toarray(), tol=1e-8))


#: An exact dense rank needs an |E| x |T| float64 array. Above this many bytes we do not attempt it.
MAX_DENSE_BYTES = 400_000_000


def _rank_affordable(B2: sp.csr_matrix) -> bool:
    return B2.shape[0] * B2.shape[1] * 8 <= MAX_DENSE_BYTES


def census(g: nx.Graph, triangle_budget: int | None = None,
           exact_rank_max_edges: int = 5000, require_triangles: bool = True,
           seed: int = 0) -> dict:
    c = build(g, triangle_budget=triangle_budget, require_triangles=require_triangles, seed=seed)
    n, m = len(c.nodes), len(c.edges)
    comps = nx.number_connected_components(g)
    dim_grad = n - comps
    dim_cycle = m - n + comps
    if m <= exact_rank_max_edges and _rank_affordable(c.B2):
        dim_curl = min(_rank_exact(c.B2), dim_cycle)
        dim_harm = dim_cycle - dim_curl
        method = "exact"
    else:
        # No estimate is offered. A randomised range finder bounds the estimated rank by its probe
        # width, which on a real target put dim_curl at exactly 2048 and inflated the harmonic part
        # from 124 to 2866 -- an artefact pointing the same way as the finding we were looking for.
        # An estimator whose ceiling sits below the quantity is worse than none, so the split is
        # reported as unavailable and the energy fractions, which need no rank, carry the claim.
        dim_curl = None
        dim_harm = None
        method = "unavailable"
    return {
        "n_nodes": n, "n_edges": m, "n_components": comps,
        "n_triangles": len(c.triangles), "n_triangles_total": c.triangles_total,
        "triangles_sampled": c.triangles_sampled,
        "dim_gradient": dim_grad, "dim_cycle_space": dim_cycle,
        "dim_curl": dim_curl, "dim_harmonic": dim_harm,
        "rank_method": method,
    }

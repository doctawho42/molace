"""The harmonic dimension, computed as a certified nullity rather than guessed.

This exists because the first version of the census estimated rank(B2) with a randomised
range finder, which cannot return a rank above its probe width and so silently returned the
probe width. The replacement computes the SMALL number -- the nullity of the edge Laplacian --
and carries a certificate that it found all of it.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from molace.hodge import complex as cx
from molace.hodge.nullity import harmonic_dimension


def _dim(g: nx.Graph):
    # require_triangles=False because several of these graphs are deliberately triangle-free:
    # the constructor refuses those by default, and rightly so, since a measured curl of zero on
    # a triangle-free graph says nothing about the model. Here the whole point is the hole.
    return harmonic_dimension(cx.build(g, triangle_budget=10000, require_triangles=False))


def test_square_has_one_hole():
    # C4 has one independent cycle and no triangle to fill it, so the whole cycle space is harmonic.
    r = _dim(nx.cycle_graph(4))
    assert r.dim == 1
    assert r.certified is True


def test_a_filled_triangle_has_no_hole():
    r = _dim(nx.complete_graph(3))
    assert r.dim == 0


def test_two_triangles_sharing_an_edge_have_no_hole():
    g = nx.Graph([(0, 1), (1, 2), (0, 2), (1, 3), (2, 3)])
    assert _dim(g).dim == 0


def test_complete_graph_on_four_nodes_has_no_hole():
    assert _dim(nx.complete_graph(4)).dim == 0


def test_a_triangle_with_a_square_glued_on_keeps_the_square_hole():
    # triangle 0-1-2 filled; square 2-3-4-5-2 unfilled -> exactly one hole
    g = nx.Graph([(0, 1), (1, 2), (0, 2), (2, 3), (3, 4), (4, 5), (5, 2)])
    assert _dim(g).dim == 1


def test_disconnected_components_each_contribute_their_holes():
    g = nx.disjoint_union(nx.cycle_graph(4), nx.cycle_graph(5))
    assert _dim(g).dim == 2


def test_rank_b2_matches_a_dense_rank_rather_than_an_identity():
    """The previous version of this test asserted dim + grad + rank_b2 == |E|.

    That is how rank_b2 is COMPUTED, so the assertion reduced to |E| == |E| and would have passed
    against any dim at all. This compares rank(B2) against an independent dense computation.
    """
    for g in (nx.random_geometric_graph(60, 0.35, seed=3),
              nx.random_geometric_graph(90, 0.30, seed=11),
              nx.complete_graph(8)):
        c = cx.build(g, triangle_budget=10000, require_triangles=False)
        want = int(np.linalg.matrix_rank(c.B2.toarray())) if c.B2.shape[1] else 0
        assert harmonic_dimension(c).rank_b2 == want


def test_the_dense_and_lanczos_paths_agree_on_the_same_complex(monkeypatch):
    """Two code paths, one answer. The threshold is lowered so both run on the same complex."""
    from molace.hodge import nullity as nl

    for g in (nx.random_geometric_graph(60, 0.30, seed=5), nx.cycle_graph(40)):
        c = cx.build(g, triangle_budget=10000, require_triangles=False)
        dense = harmonic_dimension(c)
        assert dense.method == "dense", f"expected the dense path at |E|={c.B1.shape[1]}"
        monkeypatch.setattr(nl, "DENSE_MAX_EDGES", 1)
        sparse = harmonic_dimension(c, k_start=8)
        monkeypatch.undo()
        assert sparse.method == "lanczos"
        assert sparse.certified is True
        assert dense.dim == sparse.dim, f"{dense.dim} against {sparse.dim}"


def test_a_singular_laplacian_does_not_crash_the_solver():
    """L1 is singular exactly when the harmonic space is non-empty -- the case this module is for.

    eigsh(sigma=0) factorises L1 itself and dies with "Factor is exactly singular" on a grid or a
    ladder, while real kNN complexes survive on pivoting luck. Both of these used to raise.
    """
    for g in (nx.convert_node_labels_to_integers(nx.grid_2d_graph(20, 20)), nx.ladder_graph(400)):
        c = cx.build(g, triangle_budget=10000, require_triangles=False)
        r = harmonic_dimension(c, k_start=64)
        assert r.certified is True
        assert r.dim == g.number_of_edges() - g.number_of_nodes() + nx.number_connected_components(g)


@pytest.mark.parametrize("relabel", [
    lambda g: nx.relabel_nodes(g, lambda v: v + 1),
    lambda g: nx.relabel_nodes(g, lambda v: f"m{v}"),
    lambda g: nx.relabel_nodes(g, lambda v: (v, 0)),
])
def test_graphs_whose_nodes_are_not_numbered_from_zero(relabel):
    """build() accepts these, so harmonic_dimension must too; it used to index a matrix with them."""
    g = relabel(nx.cycle_graph(6))
    assert _dim(g).dim == 1


def test_an_uncertified_result_reports_no_count_at_all(monkeypatch):
    """The predecessor returned a number saturated at its probe width. This must not.

    complete_bipartite_graph(20,30) has 600 edges and a 551-dimensional harmonic space, so a solve
    capped below that cannot bracket the kernel. The old code returned the eigenvalue count.
    """
    from molace.hodge import nullity as nl
    from molace.hodge.nullity import UNCERTIFIED

    c = cx.build(nx.complete_bipartite_graph(12, 16), triangle_budget=10000,
                 require_triangles=False)
    monkeypatch.setattr(nl, "DENSE_MAX_EDGES", 1)             # the dense path always certifies
    r = harmonic_dimension(c, k_start=16, positive_tol=1e9)   # a threshold nothing can satisfy
    assert r.certified is False
    assert r.dim == UNCERTIFIED and r.rank_b2 == UNCERTIFIED
    assert "saturated" in r.reason


def test_the_iteration_reaches_the_last_available_k():
    """`while k < m` with doubling stopped around (m-1)/2, so a large kernel could never certify."""
    c = cx.build(nx.complete_bipartite_graph(20, 30), triangle_budget=10000,
                 require_triangles=False)
    r = harmonic_dimension(c, k_start=64)
    assert r.certified is True
    assert r.dim == 600 - 50 + 1


def test_the_same_complex_gives_the_same_answer_every_time():
    """ARPACK starts from a random vector unless seeded, and this module's output is a certificate.

    Before the starting vector was fixed, the same test file passed and failed on alternate runs:
    whether a target certified depended on the draw. A census that disagrees with itself between
    runs is not a census.
    """
    c = cx.build(nx.random_geometric_graph(140, 0.22, seed=4), triangle_budget=10000,
                 require_triangles=False)
    results = [harmonic_dimension(c, k_start=16) for _ in range(4)]
    dims = {r.dim for r in results}
    certs = {r.certified for r in results}
    gaps = {round(r.spectral_gap, 10) for r in results}
    assert len(dims) == 1, f"the dimension varied between runs: {dims}"
    assert len(certs) == 1, f"certification varied between runs: {certs}"
    assert len(gaps) == 1, f"the reported spectral gap varied between runs: {gaps}"

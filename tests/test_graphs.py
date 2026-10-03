import networkx as nx
import numpy as np
import pytest
from molace.graphs import diagnostics as dg
from molace.graphs import fingerprints as fpmod
from molace.graphs import knn, threshold


def _toy_T(n=12, seed=0):
    rng = np.random.default_rng(seed)
    fp = (rng.random((n, 2048)) < 0.02).astype(np.uint8)
    return fpmod.tanimoto_matrix(fp)


def test_knn_gives_every_node_at_least_k_neighbours():
    g = knn.knn_graph(_toy_T(), k=3)
    assert min(dict(g.degree()).values()) >= 3


def test_knn_is_union_symmetrised_so_degree_is_not_exactly_k():
    # Union symmetrisation means degree >= k, not == k. The spec depends on knowing this.
    g = knn.knn_graph(_toy_T(n=40), k=3)
    assert max(dict(g.degree()).values()) > 3


def test_knn_has_no_self_loops_and_keeps_all_nodes():
    g = knn.knn_graph(_toy_T(), k=3)
    assert nx.number_of_selfloops(g) == 0
    assert g.number_of_nodes() == 12


def test_knn_neighbour_choice_is_deterministic_under_ties():
    T = np.full((6, 6), 0.5, dtype=np.float32)
    np.fill_diagonal(T, 1.0)
    a = sorted(knn.knn_graph(T, k=2).edges())
    b = sorted(knn.knn_graph(T, k=2).edges())
    assert a == b


def test_threshold_graph_keeps_only_edges_at_or_above_tau():
    T = _toy_T()
    g = threshold.threshold_graph(T, tau=0.3)
    for u, v in g.edges():
        assert T[u, v] >= 0.3


def test_diagnostics_reports_components_and_triangles_on_a_known_graph():
    g = nx.Graph()
    g.add_edges_from([(0, 1), (1, 2), (0, 2)])   # one triangle
    g.add_edges_from([(3, 4)])                   # a second component
    g.add_node(5)                                # an isolated node
    d = dg.graph_diagnostics(g)
    assert d["n_nodes"] == 6 and d["n_edges"] == 4
    assert d["n_components"] == 3
    assert d["n_isolated"] == 1
    assert d["n_triangles"] == 1
    assert d["dim_gradient"] == 6 - 3          # |V| - c
    assert d["dim_cycle_space"] == 4 - 6 + 3   # |E| - |V| + c


def test_diagnostics_on_an_edgeless_graph_does_not_divide_by_zero():
    d = dg.graph_diagnostics(nx.empty_graph(4))
    assert d["n_edges"] == 0
    assert d["density"] == 0.0 and d["mean_degree"] == 0.0
    assert d["n_triangles"] == 0

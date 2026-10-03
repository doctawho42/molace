import networkx as nx
import numpy as np
import pytest
from molace.measures.assortativity import MeasureResult, target_assortativity


def test_matches_networkx_on_a_random_graph():
    rng = np.random.default_rng(0)
    g = nx.gnp_random_graph(60, 0.15, seed=1)
    y = rng.normal(size=60)
    nx.set_node_attributes(g, {i: float(y[i]) for i in g.nodes}, "y")
    expected = nx.numeric_assortativity_coefficient(g, "y")
    assert target_assortativity(g, y).value == pytest.approx(expected, abs=1e-8)


def test_a_perfectly_smooth_label_on_a_path_is_near_one():
    g = nx.path_graph(50)
    y = np.arange(50, dtype=float)
    assert target_assortativity(g, y).value > 0.9


def test_an_alternating_label_on_a_path_is_negative():
    g = nx.path_graph(50)
    y = np.array([i % 2 for i in range(50)], dtype=float)
    assert target_assortativity(g, y).value < -0.9


def test_coverage_excludes_isolated_nodes():
    g = nx.path_graph(8)
    g.add_nodes_from([8, 9])        # two isolated molecules
    y = np.arange(10, dtype=float)
    r = target_assortativity(g, y)
    assert r.n_used == 8
    assert r.coverage == pytest.approx(0.8)


def test_edgeless_graph_raises_rather_than_returning_nan():
    with pytest.raises(ValueError, match="no edges"):
        target_assortativity(nx.empty_graph(5), np.arange(5, dtype=float))


def test_constant_label_raises_rather_than_returning_nan():
    with pytest.raises(ValueError, match="constant"):
        target_assortativity(nx.path_graph(6), np.ones(6))


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="length"):
        target_assortativity(nx.path_graph(6), np.ones(5))

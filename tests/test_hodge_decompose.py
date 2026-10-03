import networkx as nx
import numpy as np
import pytest
from molace.hodge import complex as cx
from molace.hodge import energy
from molace.hodge.decompose import decompose


def _c(n=25, p=0.35, seed=0):
    return cx.build(nx.gnp_random_graph(n, p, seed=seed))


def test_a_gradient_flow_decomposes_to_pure_gradient():
    c = _c()
    rng = np.random.default_rng(0)
    phi = rng.normal(size=len(c.nodes))
    f = np.asarray(c.B1.T @ phi).ravel()
    d = decompose(c, f)
    fr = energy.fractions(d)
    assert fr["gradient"] == pytest.approx(1.0, abs=1e-8)
    assert fr["curl"] < 1e-8 and fr["harmonic"] < 1e-8


def test_a_curl_flow_decomposes_to_pure_curl():
    c = _c()
    rng = np.random.default_rng(1)
    psi = rng.normal(size=len(c.triangles))
    f = np.asarray(c.B2 @ psi).ravel()
    fr = energy.fractions(decompose(c, f))
    assert fr["curl"] == pytest.approx(1.0, abs=1e-8)
    assert fr["gradient"] < 1e-8


def test_the_three_components_are_mutually_orthogonal():
    c = _c()
    rng = np.random.default_rng(2)
    f = rng.normal(size=len(c.edges))
    d = decompose(c, f)
    assert abs(float(d.gradient @ d.curl)) < 1e-8
    assert abs(float(d.gradient @ d.harmonic)) < 1e-8
    assert abs(float(d.curl @ d.harmonic)) < 1e-8


def test_the_components_reconstruct_the_flow():
    c = _c()
    rng = np.random.default_rng(3)
    f = rng.normal(size=len(c.edges))
    d = decompose(c, f)
    assert np.allclose(d.gradient + d.curl + d.harmonic, f, atol=1e-8)


def test_fractions_sum_to_one():
    c = _c()
    rng = np.random.default_rng(4)
    fr = energy.fractions(decompose(c, rng.normal(size=len(c.edges))))
    assert sum(fr.values()) == pytest.approx(1.0, abs=1e-8)


def test_a_difference_of_node_labels_is_curl_free_under_true_and_shuffled_labels():
    """The original exit-experiment null, shown to be void.

    dy_ij = y_j - y_i is the gradient of a node labelling, so it is curl-free identically, and a
    permutation of y is still a node labelling.
    """
    c = _c()
    rng = np.random.default_rng(5)
    y = rng.normal(size=len(c.nodes))
    for labels in (y, rng.permutation(y)):
        f = np.array([labels[c.nodes.index(v)] - labels[c.nodes.index(u)] for u, v in c.edges])
        fr = energy.fractions(decompose(c, f))
        assert fr["curl"] < 1e-10
        assert fr["harmonic"] < 1e-10


def test_harmonic_is_nonzero_on_a_graph_with_an_empty_cycle():
    g = nx.cycle_graph(5)
    c = cx.build(g, require_triangles=False)
    f = np.ones(len(c.edges))            # circulating once around the hole
    fr = energy.fractions(decompose(c, f))
    assert fr["harmonic"] > 0.9


def test_per_edge_curl_is_zero_for_a_gradient_flow():
    c = _c()
    rng = np.random.default_rng(6)
    f = np.asarray(c.B1.T @ rng.normal(size=len(c.nodes))).ravel()
    assert float(np.abs(energy.per_edge_curl(c, decompose(c, f))).max()) < 1e-8


def test_flow_length_mismatch_raises():
    c = _c()
    with pytest.raises(ValueError, match="edges"):
        decompose(c, np.ones(3))

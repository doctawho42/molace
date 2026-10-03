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
    """A cycle with no triangles has a purely harmonic circulation.

    The vector has to be built with the traversal signs. Edges are stored as sorted pairs oriented
    low to high, so walking the cycle traverses one of them against its orientation and an all-ones
    vector is not divergence-free: measured on a 4-cycle it is 75 percent gradient.
    """
    c = cx.build(nx.cycle_graph(5), require_triangles=False)
    eidx = {e: i for i, e in enumerate(c.edges)}
    f = np.zeros(len(c.edges))
    walk = [(i, (i + 1) % 5) for i in range(5)]
    for a, b in walk:
        key = (a, b) if (a, b) in eidx else (b, a)
        f[eidx[key]] = 1.0 if key == (a, b) else -1.0
    assert np.allclose(np.asarray(c.B1 @ f).ravel(), 0.0)   # divergence-free, as a circulation is
    fr = energy.fractions(decompose(c, f))
    assert fr["harmonic"] == pytest.approx(1.0, abs=1e-9)
    assert fr["curl"] < 1e-12                               # no triangles, so no curl subspace


def test_per_edge_curl_is_zero_for_a_gradient_flow():
    c = _c()
    rng = np.random.default_rng(6)
    f = np.asarray(c.B1.T @ rng.normal(size=len(c.nodes))).ravel()
    assert float(np.abs(energy.per_edge_curl(c, decompose(c, f))).max()) < 1e-8


def test_flow_length_mismatch_raises():
    c = _c()
    with pytest.raises(ValueError, match="edges"):
        decompose(c, np.ones(3))


def test_a_complex_whose_boundary_composition_is_nonzero_cannot_be_built():
    """The real defence against wrong orientation signs, and the reason it belongs in build().

    The pointwise-floor control cannot detect wrong signs: a pure gradient flow leaves a zero
    residual, and lsmr(B2, 0) is zero whatever B2 contains. Nor does breaking the triangle signs
    destroy orthogonality -- measured, with all three triangle edges made positive so that
    max|B1 @ B2| = 2.0, gradient . curl stayed at -2.1e-15 and the squared norms still summed to
    ||f||^2 within 1.3e-15. So no downstream statistic catches it.

    What catches it is the identity itself, which is why build() now asserts it on every complex it
    returns rather than leaving it to one test on one graph.
    """
    import scipy.sparse as sp

    c = cx.build(nx.complete_graph(8))
    eidx = {e: i for i, e in enumerate(c.edges)}
    rows, cols, vals = [], [], []
    for t, (i, j, k) in enumerate(c.triangles):
        for e in ((i, j), (j, k), (i, k)):
            rows.append(eidx[e]); cols.append(t); vals.append(1.0)
    bad = sp.csr_matrix((vals, (rows, cols)), shape=c.B2.shape)

    with pytest.raises(ValueError, match="boundary"):
        cx.check_boundary_identity(c.B1, bad)
    cx.check_boundary_identity(c.B1, c.B2)          # the sound one passes


def test_fractions_are_shares_of_the_flows_own_energy():
    """Dividing by the sum of the component norms would make this sum to 1 by construction.

    Dividing by ||f||^2 makes it the Pythagorean identity instead, so the assertion has content: it
    fails if the three components stop being mutually orthogonal or stop reconstructing the flow.
    """
    c = cx.build(nx.gnp_random_graph(25, 0.35, seed=0))
    rng = np.random.default_rng(7)
    f = rng.normal(size=len(c.edges))
    d = decompose(c, f)
    fr = energy.fractions(d)
    total = float(f @ f)
    assert fr["gradient"] == pytest.approx(float(d.gradient @ d.gradient) / total, abs=1e-12)
    assert sum(fr.values()) == pytest.approx(1.0, abs=1e-8)
    assert fr["residual_check"] == pytest.approx(0.0, abs=1e-8)

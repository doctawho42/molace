"""The identity is checked against graphs whose answer is known by hand, not against this project's data.

The central test is the regular-graph one. Expanding the floor's squared error gives

    mean_v (y_v - mean_{u in N(v)} y_u)^2 / sigma^2  =  1 + 1/k + (k-1)/k * rho_nn - 2 r

and on a k-REGULAR graph this is an identity, exact to machine precision, because both multisets the
two correlations are taken over then have the same mean and variance as the label itself: an edge
endpoint is uniform when every degree is k, and a shared-neighbour endpoint is uniform because every
node is counted k(k-1) times. If that test ever fails, the derivation is wrong, not the data.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from molace.measures.neighbourhood import predicted_floor_error, shared_neighbour_correlation


def exact_floor_error(g: nx.Graph, y: np.ndarray) -> float:
    """The quantity the identity claims to predict, computed by brute force."""
    nodes = sorted(g.nodes())
    pos = {v: i for i, v in enumerate(nodes)}
    errs = []
    for v in nodes:
        nb = [pos[u] for u in g.neighbors(v)]
        errs.append((y[pos[v]] - y[nb].mean()) ** 2)
    return float(np.mean(errs) / np.var(y))


@pytest.mark.parametrize("k,n,seed", [(4, 20, 0), (6, 30, 1), (3, 16, 2), (8, 40, 3)])
def test_the_identity_is_exact_on_a_regular_graph(k, n, seed):
    g = nx.random_regular_graph(k, n, seed=seed)
    y = np.random.default_rng(seed).normal(5.0, 2.0, n)
    assert predicted_floor_error(g, y, k) == pytest.approx(exact_floor_error(g, y), abs=1e-12)


def test_exactness_does_not_depend_on_the_label_being_centred():
    g = nx.random_regular_graph(4, 24, seed=7)
    y = np.random.default_rng(7).normal(0.0, 1.0, 24)
    a = predicted_floor_error(g, y, 4)
    b = predicted_floor_error(g, y + 1000.0, 4)
    c = predicted_floor_error(g, 3.0 * y, 4)
    assert a == pytest.approx(b, abs=1e-12), "a shift must not move it"
    assert a == pytest.approx(c, abs=1e-12), "a scale must not move it either"


def test_disjoint_cliques_with_a_constant_label_inside_each_give_exactly_zero():
    # k-regular with perfect agreement everywhere: r = rho_nn = 1, so the identity collapses to
    # 1 + 1/k + (k-1)/k - 2 = 0, and the floor is in fact exact.
    k = 4
    g = nx.disjoint_union_all([nx.complete_graph(k + 1) for _ in range(5)])
    y = np.concatenate([[float(i)] * (k + 1) for i in range(5)])
    assert shared_neighbour_correlation(g, y).value == pytest.approx(1.0, abs=1e-12)
    assert predicted_floor_error(g, y, k) == pytest.approx(0.0, abs=1e-12)
    assert exact_floor_error(g, y) == pytest.approx(0.0, abs=1e-12)


def test_a_permuted_label_puts_shared_neighbour_agreement_near_zero():
    g = nx.random_regular_graph(6, 200, seed=11)
    rng = np.random.default_rng(11)
    y = rng.normal(size=200)
    vals = [shared_neighbour_correlation(g, rng.permutation(y)).value for _ in range(60)]
    assert abs(float(np.mean(vals))) < 0.05


def test_a_pair_is_counted_once_per_common_neighbour():
    # In C4 the two opposite pairs each share BOTH of the other two nodes, so each contributes
    # twice. Labels +1,-1,+1,-1 make every shared-neighbour pair a same-label pair, so agreement
    # is perfect; a version that de-duplicated pairs would get the same number here, so the count
    # is pinned by the multiset size instead.
    g = nx.cycle_graph(4)
    y = np.array([1.0, -1.0, 1.0, -1.0])
    r = shared_neighbour_correlation(g, y)
    assert r.value == pytest.approx(1.0, abs=1e-12)
    assert r.n_pairs == 8, "4 nodes x 1 pair in each 2-neighbourhood, both orientations"


def test_a_graph_with_no_two_hop_pair_is_refused():
    g = nx.Graph()
    g.add_edges_from([(0, 1), (2, 3)])
    with pytest.raises(ValueError, match="shares a neighbour"):
        shared_neighbour_correlation(g, np.array([1.0, 2.0, 3.0, 4.0]))


def test_a_constant_label_is_refused():
    g = nx.random_regular_graph(4, 20, seed=5)
    with pytest.raises(ValueError, match="constant"):
        shared_neighbour_correlation(g, np.ones(20))


def test_length_mismatch_is_refused():
    g = nx.random_regular_graph(4, 20, seed=5)
    with pytest.raises(ValueError, match="length mismatch"):
        shared_neighbour_correlation(g, np.ones(19))


def test_subsampling_is_deterministic_and_close_to_the_full_value():
    g = nx.random_regular_graph(10, 300, seed=3)
    y = np.random.default_rng(3).normal(size=300)
    full = shared_neighbour_correlation(g, y)
    a = shared_neighbour_correlation(g, y, max_pairs=4000)
    b = shared_neighbour_correlation(g, y, max_pairs=4000)
    assert a.value == b.value, "same cap, same seed, same answer"
    assert a.n_pairs < full.n_pairs
    assert a.value == pytest.approx(full.value, abs=0.06)

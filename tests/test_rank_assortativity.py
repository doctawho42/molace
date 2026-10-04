"""Rank assortativity: Spearman of the label across edge endpoints.

Increment 1 learned the hard way that Newman's Pearson assortativity is not comparable across
datasets whose labels have different shapes: on a label spanning 0.01 to 700 with skewness 27 it
read 0.082, and on the same graph with the label's ranks 0.426. The rank version is invariant to
any strictly monotone relabelling, which is exactly the comparability the cross-dataset claim needs.

The construction is not a coinage: Litvak and van der Hofstad proposed Spearman's rho for
degree-degree correlations for the same reason, that the Pearson coefficient misbehaves on
heavy-tailed quantities.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from molace.measures.assortativity import rank_assortativity, target_assortativity


def _g():
    return nx.random_geometric_graph(80, 0.3, seed=7)


def test_invariant_under_a_monotone_relabelling_where_pearson_is_not():
    g = _g()
    rng = np.random.default_rng(0)
    y = rng.lognormal(sigma=2.0, size=g.number_of_nodes())
    y_exp = np.exp(y)  # strictly increasing, so every ranking is unchanged

    assert rank_assortativity(g, y).value == pytest.approx(rank_assortativity(g, y_exp).value, abs=1e-12)
    assert target_assortativity(g, y).value != pytest.approx(target_assortativity(g, y_exp).value, abs=1e-3), (
        "the Pearson version must move under this relabelling, or the test proves nothing"
    )


def test_is_the_node_rank_version_and_not_the_endpoint_spearman():
    """Pins WHICH rank statistic this is. There are two and they are close but not equal.

    This function ranks the |V| node labels once, then takes Pearson across edge endpoints. The
    Spearman of the edge-endpoint SAMPLE ranks that degree-weighted sample instead. The previous
    version of this test asserted rank == Pearson for y = arange(n), which only says Pearson ignores
    a shift of +1 per node and would have passed against either definition.
    """
    from scipy import stats

    g = _g()
    rng = np.random.default_rng(12)
    y = rng.lognormal(sigma=1.3, size=g.number_of_nodes())

    # the definition this function implements, computed here without calling it
    assert rank_assortativity(g, y).value == pytest.approx(
        target_assortativity(g, stats.rankdata(y)).value, abs=1e-12
    )

    # the other one, and the measured gap between them
    nodes = sorted(g.nodes())
    pos = {v: i for i, v in enumerate(nodes)}
    ends = np.array([(y[pos[u]], y[pos[v]]) for u, v in g.edges()])
    a = np.concatenate([ends[:, 0], ends[:, 1]])
    b = np.concatenate([ends[:, 1], ends[:, 0]])
    endpoint = float(stats.spearmanr(a, b).statistic)
    gap = abs(rank_assortativity(g, y).value - endpoint)
    assert 0.0 < gap < 0.05, f"expected close but distinct; got a gap of {gap:.5f}"


def test_ties_share_a_rank_rather_than_breaking_arbitrarily():
    g = nx.path_graph(6)
    a = rank_assortativity(g, np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])).value
    b = rank_assortativity(g, np.array([5.0, 5.0, 9.0, 9.0, 11.0, 11.0])).value
    assert a == pytest.approx(b, abs=1e-12)


def test_a_perfectly_smooth_label_on_a_path_is_near_one():
    g = nx.path_graph(50)
    assert rank_assortativity(g, np.arange(50, dtype=float)).value > 0.9


def test_coverage_counts_the_nodes_that_actually_have_an_edge():
    """A real count, not a range check: 0 < coverage <= 1 and n_used <= n hold by construction."""
    g = nx.Graph()
    g.add_nodes_from(range(10))
    g.add_edges_from([(0, 1), (1, 2), (2, 3)])      # 4 of 10 nodes touch an edge
    r = rank_assortativity(g, np.arange(10, dtype=float))
    assert r.n_used == 4
    assert r.coverage == pytest.approx(0.4)


def test_a_constant_label_is_refused():
    with pytest.raises(ValueError):
        rank_assortativity(nx.path_graph(5), np.ones(5))

import networkx as nx
import numpy as np
import pytest
from molace.hodge import complex as cx


def test_the_fundamental_identity_b1_b2_is_zero():
    g = nx.complete_graph(7)
    c = cx.build(g)
    assert abs((c.B1 @ c.B2)).max() == 0.0


def test_identity_holds_on_a_random_graph_too():
    g = nx.gnp_random_graph(40, 0.25, seed=0)
    c = cx.build(g)
    assert abs((c.B1 @ c.B2)).max() == 0.0


def test_b1_has_one_minus_one_and_one_plus_one_per_edge():
    # require_triangles=False: the incidence structure of B1 is a property of the edges alone, and a
    # path has no triangles, which build() refuses by default -- a refusal another test here pins.
    c = cx.build(nx.path_graph(5), require_triangles=False)
    col_sums = np.asarray(c.B1.sum(axis=0)).ravel()
    assert np.allclose(col_sums, 0.0)
    assert abs(c.B1).sum() == 2 * c.B1.shape[1]


def test_triangle_count_matches_networkx():
    g = nx.gnp_random_graph(30, 0.3, seed=1)
    c = cx.build(g)
    assert len(c.triangles) == sum(nx.triangles(g).values()) // 3


def test_a_graph_with_no_triangles_raises_because_curl_is_not_defined_there():
    with pytest.raises(ValueError, match="no triangles"):
        cx.build(nx.path_graph(10))


def test_triangle_budget_subsamples_and_records_it():
    g = nx.complete_graph(20)          # 1140 triangles
    c = cx.build(g, triangle_budget=100)
    assert len(c.triangles) == 100
    assert c.triangles_sampled is True
    assert c.triangles_total == 1140
    assert abs((c.B1 @ c.B2)).max() == 0.0     # the identity survives subsampling


def test_triangle_sampling_is_deterministic():
    g = nx.complete_graph(20)
    assert cx.build(g, triangle_budget=50).triangles == cx.build(g, triangle_budget=50).triangles


def test_census_reports_exact_combinatorics_on_a_known_graph():
    g = nx.Graph()
    g.add_edges_from([(0, 1), (1, 2), (0, 2)])
    g.add_edges_from([(3, 4), (4, 5), (3, 5)])
    d = cx.census(g)
    assert d["n_nodes"] == 6 and d["n_edges"] == 6 and d["n_components"] == 2
    assert d["n_triangles"] == 2
    assert d["dim_gradient"] == 4                 # 6 - 2
    assert d["dim_cycle_space"] == 2              # 6 - 6 + 2
    assert d["dim_curl"] == 2                     # two independent triangles
    assert d["dim_harmonic"] == 0
    assert d["rank_method"] == "exact"


def test_census_finds_a_harmonic_component_on_a_hollow_square():
    # A 4-cycle has a cycle space of dimension 1 and no triangles, so the cycle is harmonic.
    g = nx.cycle_graph(4)
    d = cx.census(g, require_triangles=False)
    assert d["n_triangles"] == 0
    assert d["dim_cycle_space"] == 1
    assert d["dim_curl"] == 0
    assert d["dim_harmonic"] == 1


def test_census_refuses_to_guess_the_rank_above_the_exact_threshold():
    """No estimate is offered above the threshold, and the exact quantities still are.

    A randomised range finder cannot return a rank above its probe width, and on a real target that
    ceiling inflated the harmonic part from 124 to 2866 -- in the direction of the finding the project
    is looking for. So the split is unavailable rather than estimated.
    """
    g = nx.gnp_random_graph(90, 0.4, seed=2)
    d = cx.census(g, exact_rank_max_edges=10)
    assert d["rank_method"] == "unavailable"
    assert d["dim_curl"] is None and d["dim_harmonic"] is None
    assert d["dim_gradient"] == 90 - 1
    assert d["dim_cycle_space"] == d["n_edges"] - 90 + 1
    assert d["n_triangles"] > 0

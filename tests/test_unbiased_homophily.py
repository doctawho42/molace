import networkx as nx
import numpy as np
import pytest
from molace.measures.homophily import adjusted_homophily
from molace.measures.unbiased import unbiased_homophily


def test_maximal_agreement_is_one_for_perfect_separation():
    g = nx.disjoint_union(nx.complete_graph(6), nx.complete_graph(6))
    labels = np.array([0] * 6 + [1] * 6)
    assert unbiased_homophily(g, labels).value == pytest.approx(1.0, abs=1e-9)


def test_constant_baseline_under_label_permutation():
    rng = np.random.default_rng(0)
    g = nx.gnp_random_graph(200, 0.05, seed=3)
    labels = np.array([0] * 100 + [1] * 100)
    vals = []
    for _ in range(200):
        vals.append(unbiased_homophily(g, rng.permutation(labels)).value)
    assert abs(float(np.mean(vals))) < 0.05


def test_minimal_agreement_the_property_adjusted_homophily_fails():
    """Three classes, unbalanced degree mass, zero intra-class edges.

    A two-class bipartite graph is the wrong discriminator: there the class degree sums are forced
    equal, so adjusted homophily also reaches exactly -1 (measured). The measures part company when
    a maximally heterophilous labelling has unbalanced classes: K(4,4,2) with the parts as classes
    has no intra-class edge at all, so a measure satisfying minimal agreement must return -1, while
    adjusted homophily returns about -0.524 because its baseline depends on the class distribution.
    """
    g = nx.complete_multipartite_graph(4, 4, 2)
    labels = np.array([0] * 4 + [1] * 4 + [2] * 2)
    assert sum(1 for u, v in g.edges() if labels[u] == labels[v]) == 0
    assert unbiased_homophily(g, labels).value == pytest.approx(-1.0, abs=1e-9)
    assert adjusted_homophily(g, labels).value == pytest.approx(-0.52381, abs=1e-4)


def test_comparable_across_different_class_counts():
    # The paper's purpose: a measure usable "across datasets with different label
    # distributions". Two balanced labellings with 2 and 4 classes on the same structure must
    # both sit near zero under permutation.
    rng = np.random.default_rng(1)
    g = nx.gnp_random_graph(200, 0.05, seed=5)
    for n_classes in (2, 4):
        labels = np.tile(np.arange(n_classes), 200 // n_classes)
        vals = [unbiased_homophily(g, rng.permutation(labels)).value for _ in range(150)]
        assert abs(float(np.mean(vals))) < 0.06, n_classes


def test_a_continuous_label_is_refused_here_too():
    g = nx.path_graph(30)
    with pytest.raises(TypeError, match="categorical"):
        unbiased_homophily(g, np.linspace(4.0, 10.0, 30))


def test_coverage_is_reported():
    g = nx.path_graph(8)
    g.add_nodes_from([8, 9])
    labels = np.array([0, 1] * 5)
    r = unbiased_homophily(g, labels)
    assert r.n_used == 8 and r.coverage == pytest.approx(0.8)

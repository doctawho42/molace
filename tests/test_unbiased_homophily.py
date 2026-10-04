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
    """Three classes, zero intra-class edges.

    K(4,4,2) with the parts as classes has no intra-class edge at all, so a measure satisfying
    minimal agreement must return -1. Unbiased homophily does; adjusted homophily returns about
    -0.524, because its baseline depends on the class distribution.

    What drives the divergence is the NUMBER OF CLASSES, not their balance -- see
    test_minimal_agreement_is_driven_by_class_count_not_balance below, which exists because an
    earlier version of this docstring said "unbalanced classes" and that is measurably wrong.
    """
    g = nx.complete_multipartite_graph(4, 4, 2)
    labels = np.array([0] * 4 + [1] * 4 + [2] * 2)
    assert sum(1 for u, v in g.edges() if labels[u] == labels[v]) == 0
    assert unbiased_homophily(g, labels).value == pytest.approx(-1.0, abs=1e-9)
    assert adjusted_homophily(g, labels).value == pytest.approx(-0.52381, abs=1e-4)


def test_minimal_agreement_is_driven_by_class_count_not_balance():
    """Pins WHY adjusted homophily fails minimal agreement, because the prose kept getting it wrong.

    This test passes against unchanged code: what it guards is a factual claim, not a behaviour.
    It is here so that the explanation cannot drift again without a red suite.

    Measured, on maximally heterophilous labellings (no intra-class edge anywhere):
      * two classes, unbalanced -- both measures reach exactly -1, so imbalance alone does not
        make them part company;
      * three classes, perfectly balanced -- they already part company, -0.5 against -1;
      * with balanced parts the adjusted value is exactly -1/(C-1), drifting toward 0 as classes
        are added, which is the actual failure;
      * imbalance moves adjusted slightly TOWARD the minimum, not away from it (-0.524 at K(4,4,2)
        against -0.5 at K(4,4,4)), so the old "unbalanced" wording had the sign of the effect
        backwards as well.
    """
    def both(*parts):
        g = nx.complete_multipartite_graph(*parts)
        labels = np.concatenate([[i] * n for i, n in enumerate(parts)])
        assert sum(1 for u, v in g.edges() if labels[u] == labels[v]) == 0
        return adjusted_homophily(g, labels).value, unbiased_homophily(g, labels).value

    a2, u2 = both(6, 2)
    assert a2 == pytest.approx(-1.0, abs=1e-9), "two classes: imbalance alone does not separate them"
    assert u2 == pytest.approx(-1.0, abs=1e-9)

    a3, u3 = both(4, 4, 4)
    assert u3 == pytest.approx(-1.0, abs=1e-9)
    assert a3 == pytest.approx(-0.5, abs=1e-9), "balanced three classes already separate them"

    for c in (3, 4, 5):
        a, u = both(*([3] * c))
        assert u == pytest.approx(-1.0, abs=1e-9)
        assert a == pytest.approx(-1.0 / (c - 1), abs=1e-9)

    a_unbal, _ = both(4, 4, 2)
    assert a_unbal < a3, "imbalance moves adjusted toward the minimum, not away from it"


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

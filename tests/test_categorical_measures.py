import networkx as nx
import numpy as np
import pytest
from molace.measures.homophily import (adjusted_homophily, edge_homophily,
                                       require_categorical)
from molace.measures.informativeness import label_informativeness


def test_edge_homophily_on_a_hand_counted_graph():
    g = nx.Graph([(0, 1), (1, 2), (2, 3)])
    labels = np.array([0, 0, 1, 1])
    # edges: (0,1) same, (1,2) different, (2,3) same -> 2/3
    assert edge_homophily(g, labels).value == pytest.approx(2 / 3)


def test_adjusted_homophily_matches_the_formula_by_hand():
    g = nx.Graph([(0, 1), (1, 2), (2, 3)])
    labels = np.array([0, 0, 1, 1])
    # degrees 1,2,2,1; 2|E| = 6; D_0 = 3, D_1 = 3; sum pbar^2 = 0.25 + 0.25 = 0.5
    # h_adj = (2/3 - 0.5) / (1 - 0.5) = 1/3
    assert adjusted_homophily(g, labels).value == pytest.approx(1 / 3)


def test_adjusted_homophily_is_one_for_perfect_separation():
    g = nx.disjoint_union(nx.complete_graph(5), nx.complete_graph(5))
    labels = np.array([0] * 5 + [1] * 5)
    assert adjusted_homophily(g, labels).value == pytest.approx(1.0)


def test_label_informativeness_is_zero_when_the_label_is_independent_of_structure():
    g = nx.complete_graph(40)
    labels = np.array([0, 1] * 20)
    assert label_informativeness(g, labels).value < 0.05


def test_label_informativeness_is_one_for_perfect_separation():
    g = nx.disjoint_union(nx.complete_graph(6), nx.complete_graph(6))
    labels = np.array([0] * 6 + [1] * 6)
    assert label_informativeness(g, labels).value == pytest.approx(1.0, abs=1e-9)


def test_label_informativeness_matches_a_hand_computed_two_class_case():
    g = nx.Graph([(0, 1), (1, 2), (2, 3)])
    labels = np.array([0, 0, 1, 1])
    # 2|E| = 6. joint counts (both directions): (0,0) 2, (1,1) 2, (0,1) 1, (1,0) 1
    # p = (0.5, 0.5); H = ln 2
    # I = 2*(1/3)ln((1/3)/0.25) + 2*(1/6)ln((1/6)/0.25)
    p = np.array([0.5, 0.5])
    joint = np.array([[2 / 6, 1 / 6], [1 / 6, 2 / 6]])
    h = -(p * np.log(p)).sum()
    i = sum(joint[a, b] * np.log(joint[a, b] / (p[a] * p[b])) for a in range(2) for b in range(2))
    assert label_informativeness(g, labels).value == pytest.approx(i / h)


def test_a_continuous_label_is_refused_by_both_measures():
    g = nx.path_graph(30)
    y = np.linspace(4.0, 10.0, 30)          # a real pIC50-like vector
    for fn in (adjusted_homophily, label_informativeness):
        with pytest.raises(TypeError) as e:
            fn(g, y)
        assert "categorical" in str(e.value)
        assert "assortativity" in str(e.value)


def test_many_integer_classes_is_also_refused_because_that_is_binning():
    g = nx.path_graph(40)
    labels = np.arange(40) % 12          # 12 "classes" is a binning, not a natural label
    with pytest.raises(TypeError, match="categorical"):
        adjusted_homophily(g, labels)


def test_a_single_class_raises_rather_than_returning_nan():
    g = nx.path_graph(8)
    labels = np.zeros(8, dtype=int)
    with pytest.raises(ValueError, match="one class"):
        adjusted_homophily(g, labels)
    with pytest.raises(ValueError, match="one class"):
        label_informativeness(g, labels)


def test_require_categorical_accepts_the_real_cliff_flag():
    labels = require_categorical(np.array([0, 1, 1, 0, 1]))
    assert labels.dtype.kind in "iu"

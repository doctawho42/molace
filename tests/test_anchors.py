import numpy as np
import pytest
from molace.models import anchors, knn_floor


def test_selects_m_highest_similarity_training_indices():
    sim = np.array([[0.1, 0.9, 0.5, 0.7]])     # one test molecule, four training molecules
    idx = anchors.select(sim, m=2)
    assert idx.shape == (1, 2)
    assert set(idx[0]) == {1, 3}


def test_ties_break_on_ascending_training_index_and_are_stable():
    sim = np.full((1, 5), 0.5)
    a = anchors.select(sim, m=3)
    b = anchors.select(sim, m=3)
    assert a.tolist() == [[0, 1, 2]]
    assert a.tolist() == b.tolist()


def test_m_larger_than_the_training_set_is_clipped():
    sim = np.full((2, 3), 0.4)
    assert anchors.select(sim, m=10).shape == (2, 3)


def test_default_m_is_ten_and_is_not_the_graph_k():
    """m and the graph's k share the value 10 today but must not share a source.

    The second assertion is the one with teeth: if anyone later wires m to the graph degree, the
    anchors module would have to reach for the pre-registration or the graphs package, and this
    fails. The first only pins today's value.
    """
    import inspect

    from molace.analysis.prereg import load_prereg

    p = load_prereg()
    assert anchors.M == 10 == p["arms"]["knn_floor"]["m"]
    src = inspect.getsource(anchors)
    assert "load_prereg" not in src
    assert "molace.graphs" not in src


def test_floor_is_the_uniform_mean_of_the_anchor_labels():
    y_train = np.array([1.0, 2.0, 3.0, 4.0])
    idx = np.array([[0, 1], [2, 3]])
    assert knn_floor.predict(y_train, idx).tolist() == [1.5, 3.5]


def test_floor_is_deterministic_and_needs_no_seed():
    y_train = np.linspace(0, 1, 50)
    sim = np.random.default_rng(0).random((20, 50))
    idx = anchors.select(sim)
    a = knn_floor.predict(y_train, idx)
    b = knn_floor.predict(y_train, anchors.select(sim))
    assert np.array_equal(a, b)


def test_empty_training_set_raises():
    with pytest.raises(ValueError, match="training"):
        anchors.select(np.zeros((3, 0)))

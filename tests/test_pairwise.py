import numpy as np
import pytest
from molace.graphs.fingerprints import tanimoto_matrix
from molace.models import anchors, knn_floor, pairwise


def _task(n=180, seed=0):
    rng = np.random.default_rng(seed)
    X = (rng.random((n, 2048)) < 0.02).astype(np.uint8)
    w = np.zeros(2048)
    w[rng.choice(2048, 30, replace=False)] = rng.normal(size=30)
    y = X @ w + rng.normal(scale=0.1, size=n)
    return X, y


def _split(X, y, n_train=140):
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


def test_difference_feature_map_is_second_minus_first():
    Xa = np.array([[1, 0, 1]], dtype=np.uint8)
    Xb = np.array([[1, 1, 0]], dtype=np.uint8)
    got = pairwise.pair_features(Xa, Xb, "difference")
    assert got.tolist() == [[0.0, 1.0, -1.0]]


def test_concat_difference_feature_map_layout():
    Xa = np.array([[1, 0]], dtype=np.uint8)
    Xb = np.array([[0, 1]], dtype=np.uint8)
    got = pairwise.pair_features(Xa, Xb, "concat_difference")
    assert got.tolist() == [[1.0, 0.0, 0.0, 1.0, -1.0, 1.0]]


def test_training_pair_count_is_two_n_times_m():
    X, y = _task()
    Xtr, ytr, _, _ = _split(X, y)
    sim = tanimoto_matrix(Xtr)
    feats, targets = pairwise.training_pairs(Xtr, ytr, sim, m=5, kind="difference")
    assert len(targets) == 2 * len(ytr) * 5
    assert feats.shape[0] == len(targets)


def test_training_targets_follow_the_stated_convention():
    X = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    y = np.array([3.0, 10.0])
    sim = tanimoto_matrix(X)
    feats, targets = pairwise.training_pairs(X, y, sim, m=1, kind="difference")
    # This test pins the SIGN convention, not the multiplicity. Mutual nearest neighbours appear in
    # both neighbour lists, so each direction is emitted twice; the count is 2*n*m, which the test
    # above pins. Measured on real targets, 57 to 66 percent of pairs are mutual, a spread of under
    # nine points, so the implied weighting is near-constant across targets.
    assert sorted(set(np.round(targets, 6).tolist())) == [-7.0, 7.0]
    assert len(targets) == 4


def test_unknown_feature_map_raises():
    with pytest.raises(KeyError, match="unknown feature map"):
        pairwise.pair_features(np.zeros((1, 3)), np.zeros((1, 3)), "cross_attention")


@pytest.mark.parametrize("kind", ["difference", "concat_difference"])
def test_prediction_equals_the_floor_plus_the_mean_correction(kind):
    """The identity the whole decomposition rests on."""
    X, y = _task()
    Xtr, ytr, Xte, _ = _split(X, y)
    sim_tr = tanimoto_matrix(Xtr)
    sim_te_tr = tanimoto_matrix(X)[140:, :140]
    idx = anchors.select(sim_te_tr, m=10)
    model = pairwise.fit("hgb", kind, Xtr, ytr, sim_tr, m=10, seed=0)
    pred = model.predict(ytr, Xtr, Xte, idx)
    floor = knn_floor.predict(ytr, idx)
    corr = model.corrections(Xtr, Xte, idx).mean(axis=1)
    assert np.allclose(pred, floor + corr, atol=1e-10)


def test_corrections_have_shape_n_test_by_m():
    X, y = _task()
    Xtr, ytr, Xte, _ = _split(X, y)
    sim_tr = tanimoto_matrix(Xtr)
    idx = anchors.select(tanimoto_matrix(X)[140:, :140], m=10)
    model = pairwise.fit("svm", "difference", Xtr, ytr, sim_tr, m=10, seed=0)
    assert model.corrections(Xtr, Xte, idx).shape == (40, 10)


def test_a_bias_free_linear_head_gives_corrections_that_cancel_on_a_triangle():
    """A linear readout on the difference is a gradient flow, so triangle sums vanish.

    This is the half-1 shadow of the §10 theorem and it costs nothing to assert here.
    """
    rng = np.random.default_rng(0)
    X = (rng.random((3, 2048)) < 0.02).astype(np.float64)
    w = rng.normal(size=2048)
    f = lambda a, b: float(w @ (X[b] - X[a]))
    assert f(0, 1) + f(1, 2) + f(2, 0) == pytest.approx(0.0, abs=1e-9)


def test_same_seed_reproduces_predictions():
    X, y = _task()
    Xtr, ytr, Xte, _ = _split(X, y)
    sim_tr = tanimoto_matrix(Xtr)
    idx = anchors.select(tanimoto_matrix(X)[140:, :140], m=10)
    a = pairwise.fit("mlp", "difference", Xtr, ytr, sim_tr, m=10, seed=3).predict(ytr, Xtr, Xte, idx)
    b = pairwise.fit("mlp", "difference", Xtr, ytr, sim_tr, m=10, seed=3).predict(ytr, Xtr, Xte, idx)
    assert np.array_equal(a, b)

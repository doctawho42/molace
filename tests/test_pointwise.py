import numpy as np
import pytest
from molace.models import pointwise


def _easy_task(n=160, seed=0):
    """A learnable task: y depends on a few bits that are common enough to be learnable.

    The plan's original fixture drew every bit at 2 percent density, so an informative bit appeared
    in two or three of the 120 training rows and the most frequent bit of all appeared eight times.
    Measured, all three learners failed to beat the mean on it -- a tree cannot split on a feature
    that is set three times, and the kernel and the network had nothing dense to work with either.
    A 40-column block at 50 percent density carries the signal here, with the rest of the vector
    left sparse so the input still looks like a fingerprint.
    """
    rng = np.random.default_rng(seed)
    X = (rng.random((n, 2048)) < 0.02).astype(np.uint8)
    X[:, :40] = (rng.random((n, 40)) < 0.5).astype(np.uint8)
    w = np.zeros(2048)
    w[:6] = np.array([1.5, -1.2, 0.9, 1.1, -0.8, 0.7])
    y = X @ w + rng.normal(scale=0.15, size=n)
    return X, y


def test_the_three_learners_are_named_in_order_with_svm_first():
    assert pointwise.LEARNERS == ("svm", "hgb", "mlp")


def test_each_learner_beats_predicting_the_mean():
    X, y = _easy_task()
    Xtr, ytr, Xte, yte = X[:120], y[:120], X[120:], y[120:]
    baseline = float(np.sqrt(np.mean((yte - ytr.mean()) ** 2)))
    for name in pointwise.LEARNERS:
        pred = pointwise.fit_predict(name, Xtr, ytr, Xte, seed=0)
        rmse = float(np.sqrt(np.mean((yte - pred) ** 2)))
        assert rmse < baseline, name


def test_predictions_have_the_right_shape():
    X, y = _easy_task()
    pred = pointwise.fit_predict("svm", X[:120], y[:120], X[120:], seed=0)
    assert pred.shape == (40,)


def test_same_seed_reproduces_exactly():
    X, y = _easy_task()
    a = pointwise.fit_predict("mlp", X[:120], y[:120], X[120:], seed=7)
    b = pointwise.fit_predict("mlp", X[:120], y[:120], X[120:], seed=7)
    assert np.array_equal(a, b)


def test_cv_selection_returns_a_named_learner_and_all_scores():
    X, y = _easy_task()
    winner, scores = pointwise.select_by_cv(X[:120], y[:120], seed=0)
    assert winner in pointwise.LEARNERS
    assert set(scores) == set(pointwise.LEARNERS)
    assert scores[winner] == min(scores.values())


def test_cv_selection_never_touches_the_test_split():
    # Guard by signature: select_by_cv takes no test arguments at all.
    import inspect
    params = set(inspect.signature(pointwise.select_by_cv).parameters)
    assert not {"X_test", "y_test"} & params


def test_unknown_learner_raises():
    with pytest.raises(KeyError, match="unknown learner"):
        pointwise.make_learner("randomforest", seed=0)


def test_lightgbm_is_not_a_learner_here():
    """It was, in the plan. It needs a system OpenMP runtime this project will not depend on."""
    assert "lightgbm" not in pointwise.LEARNERS
    with pytest.raises(KeyError, match="unknown learner"):
        pointwise.make_learner("lightgbm", seed=0)

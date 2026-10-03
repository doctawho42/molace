"""The pointwise arm: one molecule in, one property out, no test-time label access.

SVM leads because it is the real bar. From MoleculeACE's own shipped results matrix, ECFP-SVM takes
21 of 30 targets on RMSE with mean rank 1.60, against ECFP-GBM at 2 of 30 and rank 3.00, so naming
gradient boosting as the baseline inflates every reported gap. The neural arm is here because SQRL's
own table shows tree models gain nothing from pairing, and a tree-only comparison would make the gap
degenerate.

The boosted-tree arm is scikit-learn's histogram gradient boosting rather than LightGBM, which the
plan named. LightGBM's wheel needs a system OpenMP runtime (`libomp.dylib`) that is absent here;
installing it is a change outside this repository and would make a repository meant to be publicly
reproducible depend on a system package manager. Histogram gradient boosting is the same algorithm
family, ships inside scikit-learn, and needs no external runtime. `min_samples_leaf` is lowered from
the default 20 because a fingerprint bit is set in a small minority of molecules and the default
forbids the splits that carry the signal.

Hyperparameters are sized for a 2048-dimensional input, which is what the plan's values were not.
Measured on one real target (489 training molecules): the plan's boosting settings cost 70.7s for a
single fit and its network 12.7s, against 19.5s and 0.7s after resizing. Boosting stays the most
expensive arm in both directions because scikit-learn bins all 2048 features at every fit regardless
of `max_features`, and that cost is irreducible here.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

LEARNERS: tuple[str, ...] = ("svm", "hgb", "mlp")


def make_learner(name: str, seed: int):
    if name == "svm":
        return SVR(kernel="rbf", C=10.0, gamma="scale", epsilon=0.1)
    if name == "hgb":
        return HistGradientBoostingRegressor(
            max_iter=100, learning_rate=0.1, max_leaf_nodes=31, max_features=0.05,
            min_samples_leaf=10, random_state=seed,
        )
    if name == "mlp":
        return MLPRegressor(
            hidden_layer_sizes=(64,), max_iter=120, early_stopping=True,
            n_iter_no_change=8, random_state=seed,
        )
    raise KeyError(f"unknown learner {name!r}; expected one of {LEARNERS}")


def fit_predict(name: str, X_train, y_train, X_test, seed: int) -> np.ndarray:
    model = make_learner(name, seed)
    model.fit(np.asarray(X_train, dtype=np.float32), np.asarray(y_train, dtype=float))
    return np.asarray(model.predict(np.asarray(X_test, dtype=np.float32)), dtype=float)


def select_by_cv(X_train, y_train, seed: int, n_folds: int = 5):
    """Pick the learner by mean CV RMSE on the training split only.

    Takes no test arguments by design: selection must not see the evaluation split.
    """
    X = np.asarray(X_train, dtype=np.float32)
    y = np.asarray(y_train, dtype=float)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores: dict[str, float] = {}
    for name in LEARNERS:
        errs = []
        for tr, va in kf.split(X):
            pred = fit_predict(name, X[tr], y[tr], X[va], seed)
            errs.append(float(np.sqrt(np.mean((y[va] - pred) ** 2))))
        scores[name] = float(np.mean(errs))
    return min(scores, key=scores.__getitem__), scores

"""The pairwise arm.

Sign convention, obeyed everywhere: f(x_a, x_b) estimates y_b - y_a, the second argument minus
the first. Training pairs (a, b) carry target y_b - y_a with features from x_b - x_a, and
inference is y_new = mean_a [ y_a + f(x_a, x_new) ] with the anchor first. This is written down
because SQRL's published equations are inconsistent: its eq. 3 under eq. 2's convention estimates
2 y_i - y_new. We do not reproduce SQRL.

Pair budget: each training molecule with its m nearest other training molecules, both directions,
2 * n_train * m ordered pairs. These are not all distinct: a mutual nearest-neighbour relation sits
in both molecules' lists, so such a pair is emitted twice in each direction and carries double
weight. Measured on six real targets, 57 to 66 percent of pairs are mutual, a spread of under nine
points, so the implied weighting is near-constant across targets and is not a density confound. The
realised distinct-pair count is recorded per target by the sweep rather than left implied. All ordered pairs would be 8.5 million on the largest target, and training
on all pairs would also train on a distribution the model never meets at test time, where anchors
are nearest neighbours.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import LinearSVR

from molace.models.anchors import M

FEATURE_MAPS: tuple[str, ...] = ("difference", "concat_difference")

#: Families whose pointwise and pairwise learners are the same model class, so their gap is a
#: matched comparison. The kernel family is deliberately absent: see make_pair_learner.
MATCHED_FAMILIES: tuple[str, ...] = ("hgb", "mlp")


def make_pair_learner(name: str, seed: int):
    """A learner sized for the pair training set, which is 20 to 60 times the molecule count.

    The kernel slot is a LINEAR support vector regressor, not the RBF one the pointwise arm uses.
    RBF is quadratic in samples and the largest target has 58,480 pairs; measured, the pointwise RBF
    fit costs 9.9s on 2,924 molecules while a linear pairwise fit costs 42.9s on their 58,480 pairs
    and an RBF one does not finish. Two consequences, both stated rather than hidden: the kernel
    family's pointwise-against-pairwise gap is NOT a matched comparison and is therefore excluded
    from the primary, and a bias-free linear readout on a difference of a fixed encoder is exactly
    the gradient-flow case of the design's second half, so this arm doubles as its empirical check.
    """
    if name == "svm":
        return LinearSVR(C=1.0, max_iter=5000, random_state=seed)
    if name == "hgb":
        return HistGradientBoostingRegressor(
            max_iter=100, learning_rate=0.1, max_features=0.05,
            min_samples_leaf=10, random_state=seed,
        )
    if name == "mlp":
        return MLPRegressor(
            hidden_layer_sizes=(64,), max_iter=120, early_stopping=True,
            n_iter_no_change=8, random_state=seed,
        )
    raise KeyError(f"unknown learner {name!r}")


def pair_features(Xa: np.ndarray, Xb: np.ndarray, kind: str) -> np.ndarray:
    a = np.asarray(Xa, dtype=np.float32)
    b = np.asarray(Xb, dtype=np.float32)
    if kind == "difference":
        return b - a
    if kind == "concat_difference":
        return np.hstack([a, b, b - a])
    raise KeyError(f"unknown feature map {kind!r}; expected one of {FEATURE_MAPS}")


def training_pairs(X_train, y_train, sim_train_train: np.ndarray, m: int, kind: str):
    """Both directions of (molecule, its m nearest other training molecules)."""
    X = np.asarray(X_train)
    y = np.asarray(y_train, dtype=float)
    n = len(y)
    s = np.asarray(sim_train_train, dtype=np.float64).copy()
    np.fill_diagonal(s, -np.inf)
    mm = min(m, n - 1)
    tie = np.tile(np.arange(n), (n, 1))
    nbr = np.lexsort((tie, -s), axis=1)[:, :mm]
    a_idx = np.repeat(np.arange(n), mm)
    b_idx = nbr.reshape(-1)
    a_all = np.concatenate([a_idx, b_idx])
    b_all = np.concatenate([b_idx, a_idx])
    feats = pair_features(X[a_all], X[b_all], kind)
    targets = y[b_all] - y[a_all]
    return feats, targets


@dataclass
class PairwiseModel:
    learner_name: str
    kind: str
    model: object

    def corrections(self, X_train, X_test, anchor_idx: np.ndarray) -> np.ndarray:
        """f(anchor, query) for every (query, anchor) pair, shape (n_test, m)."""
        Xtr = np.asarray(X_train)
        Xte = np.asarray(X_test)
        idx = np.asarray(anchor_idx, dtype=np.int64)
        n_test, mm = idx.shape
        a = Xtr[idx.reshape(-1)]                                 # anchors first
        b = np.repeat(Xte, mm, axis=0)                           # query second
        preds = self.model.predict(pair_features(a, b, self.kind))
        return np.asarray(preds, dtype=float).reshape(n_test, mm)

    def predict(self, y_train, X_train, X_test, anchor_idx: np.ndarray) -> np.ndarray:
        y = np.asarray(y_train, dtype=float)
        idx = np.asarray(anchor_idx, dtype=np.int64)
        return (y[idx] + self.corrections(X_train, X_test, idx)).mean(axis=1)


def fit(learner: str, kind: str, X_train, y_train, sim_train_train, m: int = M,
        seed: int = 0) -> PairwiseModel:
    feats, targets = training_pairs(X_train, y_train, sim_train_train, m, kind)
    model = make_pair_learner(learner, seed)
    model.fit(feats, targets)
    return PairwiseModel(learner_name=learner, kind=kind, model=model)

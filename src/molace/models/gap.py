"""The dependent variable, defined once.

gap        = RMSE_pointwise - RMSE_pairwise      (positive means pairwise wins)
access     = RMSE_pointwise - RMSE_knn_floor     (what test-time label access buys)
correction = RMSE_knn_floor - RMSE_pairwise      (what the learned pairwise function buys)

The telescoping access + correction = gap is trivially true for any three numbers and carries no
content by itself. What gives `correction` meaning is the prediction identity enforced in
models.pairwise: with uniform weights over the same anchors the pairwise prediction is the floor plus
the mean learned correction, so the error change is attributable to that function and nothing else.

RMSE is nonlinear, so `correction` is NOT a share of `gap` and must never be reported as a
percentage of it.

Two departures from the design document, both forced by measured cost and both symmetric.

First, there is no cross-validated selection inside either arm. The design had each arm pick its best
of three learners by 5-fold CV on the training split. Measured, a single fit costs between 0.3s and
75s depending on learner and arm, so 5-fold CV over three learners and two feature maps in both arms
is roughly 36 fits per target and seed: a 20-hour sweep. One fit per learner per arm instead, which
brings 30 targets by 3 seeds to a few hours.

Second, selection is replaced by averaging over MATCHED families rather than by picking a winner.
Letting one arm choose its best while the other is fixed would bias the gap toward the chooser. The
primary therefore averages the families whose pointwise and pairwise learners are the same model
class, which is how SQRL reports its own results: per base model, standard against paired. The kernel
family is excluded from that average because its pairwise slot is a linear support vector regressor
while its pointwise slot is an RBF one -- RBF is quadratic in samples and does not finish on 58,480
pairs -- so its gap is reported but is not a matched comparison. The feature map is `difference`
only; `concat_difference` triples the feature width and is deferred.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from molace.graphs.fingerprints import ecfp4, tanimoto_matrix
from molace.models import anchors, knn_floor, pairwise, pointwise
from molace.models.pairwise import MATCHED_FAMILIES

PAIRWISE_KIND = "difference"


def rmse(y_true, y_pred) -> float:
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    if a.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((a - b) ** 2)))


@dataclass(frozen=True)
class ArmResult:
    rmse_all: float
    rmse_cliff: float
    selected: str


@dataclass(frozen=True)
class GapResult:
    pointwise: ArmResult
    knn_floor: ArmResult
    pairwise: ArmResult
    gap: float
    access: float
    correction: float
    per_learner_gap: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if abs((self.access + self.correction) - self.gap) > 1e-9:
            raise ValueError(
                "decomposition is inconsistent: access + correction must equal gap, got "
                f"{self.access} + {self.correction} != {self.gap}"
            )


def evaluate_target(df: pd.DataFrame, seed: int) -> GapResult:
    tr = df["split"].to_numpy() == "train"
    te = ~tr
    fp = ecfp4(df["smiles"].tolist())
    y = df["y"].to_numpy(dtype=float)
    cliff = df["cliff_mol"].to_numpy(dtype=int)[te].astype(bool)

    X_tr, y_tr, X_te, y_te = fp[tr], y[tr], fp[te], y[te]
    sim_all = tanimoto_matrix(fp)
    sim_tr = sim_all[np.ix_(tr, tr)]
    idx = anchors.select(sim_all[np.ix_(te, tr)])

    pw_rmse: dict[str, float] = {}
    pr_rmse: dict[str, float] = {}
    pw_cliff: dict[str, float] = {}
    pr_cliff: dict[str, float] = {}
    per_learner: dict[str, float] = {}

    for name in pointwise.LEARNERS:
        p = pointwise.fit_predict(name, X_tr, y_tr, X_te, seed)
        model = pairwise.fit(name, PAIRWISE_KIND, X_tr, y_tr, sim_tr, anchors.M, seed)
        q = model.predict(y_tr, X_tr, X_te, idx)
        pw_rmse[name] = rmse(y_te, p)
        pr_rmse[name] = rmse(y_te, q)
        pw_cliff[name] = rmse(y_te[cliff], p[cliff]) if cliff.any() else float("nan")
        pr_cliff[name] = rmse(y_te[cliff], q[cliff]) if cliff.any() else float("nan")
        per_learner[name] = pw_rmse[name] - pr_rmse[name]

    label = "mean(" + ",".join(MATCHED_FAMILIES) + ")"
    pw = ArmResult(
        rmse_all=float(np.mean([pw_rmse[f] for f in MATCHED_FAMILIES])),
        rmse_cliff=float(np.mean([pw_cliff[f] for f in MATCHED_FAMILIES])),
        selected=label,
    )
    pr = ArmResult(
        rmse_all=float(np.mean([pr_rmse[f] for f in MATCHED_FAMILIES])),
        rmse_cliff=float(np.mean([pr_cliff[f] for f in MATCHED_FAMILIES])),
        selected=label,
    )
    floor_pred = knn_floor.predict(y_tr, idx)
    fl = ArmResult(
        rmse_all=rmse(y_te, floor_pred),
        rmse_cliff=rmse(y_te[cliff], floor_pred[cliff]) if cliff.any() else float("nan"),
        selected="knn",
    )

    return GapResult(
        pointwise=pw, knn_floor=fl, pairwise=pr,
        gap=pw.rmse_all - pr.rmse_all,
        access=pw.rmse_all - fl.rmse_all,
        correction=fl.rmse_all - pr.rmse_all,
        per_learner_gap=per_learner,
    )

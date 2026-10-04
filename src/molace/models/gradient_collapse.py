"""What the pairwise arm reduces to when its edge flow is a gradient.

Two facts the project already has. First, algebra: with uniform weights over the same anchors, the
pairwise read-out equals the kNN floor plus the mean learned correction,

    yhat_pair = mean_a (y_a + f(x_a, x_new)) = floor + mean_a f(x_a, x_new).

Second, measurement: the trained pairwise flow is 96.6 % gradient, curl 2.6 %, harmonic 0.8 %, and
shuffling the labels leaves curl where it was, so the circulation carries no label information.

Put together they say something sharper than either. A gradient flow is f(x_a, x_b) = phi_b - phi_a
for a node potential phi, so the mean learned correction is not free: it is phi at the new point minus
phi averaged over the anchors. Hence

    yhat_pair = phi_new + (mean_a y_a - mean_a phi_a),

a POINTWISE predictor phi plus the floor residual of phi. That is the precise version of "the model
collapses into a pointwise one plus anchor averaging", and it explains without any further measurement
why the pairwise arm tracks the floor as closely as it does: the floor residual is literally one of
its two terms.

One caveat the numbers make necessary. What survives the read-out is the MEAN of the non-gradient part
over the anchors, not its energy. A flow can be far from gradient and still predict exactly as a
gradient one does, if its non-gradient part averages to zero over the anchors; `non_gradient_residual`
is the quantity that actually moves the prediction. So "96.6 % gradient" bounds the energy, not the
prediction gap, and the two should not be conflated.
"""
from __future__ import annotations

import numpy as np


def _check(phi_anchors: np.ndarray, y_anchors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    phi = np.asarray(phi_anchors, dtype=float)
    y = np.asarray(y_anchors, dtype=float)
    if phi.shape != y.shape:
        raise ValueError(
            f"the potential and the labels must have the same length, got {phi.shape} and {y.shape}"
        )
    if phi.size == 0:
        raise ValueError("a pairwise read-out needs at least one anchor")
    return phi, y


def gradient_pairwise_prediction(
    phi_new: float, phi_anchors: np.ndarray, y_anchors: np.ndarray
) -> float:
    """The pairwise read-out when the flow is exactly the gradient of phi.

    Equal to `phi_new + (mean y over anchors - mean phi over anchors)`: the potential at the new
    point, corrected by how much the anchors' true labels exceed the potential's own values there.
    """
    phi, y = _check(phi_anchors, y_anchors)
    return float(phi_new) + float(y.mean() - phi.mean())


def non_gradient_residual(
    phi_new: float, phi_anchors: np.ndarray, y_anchors: np.ndarray, flow: np.ndarray
) -> float:
    """How far a real flow's prediction sits from the gradient collapse, in label units.

    This is the mean over anchors of whatever the flow carries beyond `phi_new - phi_a`, which is the
    only part of the non-gradient component that reaches the prediction at all.
    """
    phi, y = _check(phi_anchors, y_anchors)
    f = np.asarray(flow, dtype=float)
    if f.shape != phi.shape:
        raise ValueError(
            f"the flow and the potential must have the same length, got {f.shape} and {phi.shape}"
        )
    return float(np.mean(y + f)) - gradient_pairwise_prediction(phi_new, phi, y)

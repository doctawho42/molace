"""What a pairwise model IS when its edge flow is a gradient.

The project already measures that the trained pairwise flow is 96.6 % gradient and already states the
algebraic decomposition prediction = floor + mean learned correction. This pins the sharper statement
those two facts imply together: when the flow is exactly the gradient of a node potential phi, the
mean learned correction is not an arbitrary quantity, it is exactly phi at the new point minus phi
averaged over the anchors. So the pairwise arm is a POINTWISE predictor phi plus the floor residual of
phi, and nothing else.

That is testable without any trained model, because it is a statement about the estimator.
"""
from __future__ import annotations

import numpy as np
import pytest

from molace.models.gradient_collapse import (
    gradient_pairwise_prediction,
    non_gradient_residual,
)


def aggregate(y_anchors: np.ndarray, flow: np.ndarray) -> float:
    """The project's own pairwise read-out: mean over anchors of (anchor label + predicted delta)."""
    return float(np.mean(y_anchors + flow))


def test_a_gradient_flow_collapses_the_pairwise_readout_exactly():
    rng = np.random.default_rng(0)
    phi = rng.normal(size=12)              # the node potential, anchors 0..11
    phi_new = float(rng.normal())
    y_anchors = rng.normal(size=12)
    flow = phi_new - phi                   # f(x_a, x_new) = phi_new - phi_a
    assert aggregate(y_anchors, flow) == pytest.approx(
        gradient_pairwise_prediction(phi_new, phi, y_anchors), abs=1e-13
    )


@pytest.mark.parametrize("m", [1, 2, 5, 50])
def test_the_collapse_holds_for_any_number_of_anchors(m):
    rng = np.random.default_rng(m)
    phi, y = rng.normal(size=m), rng.normal(size=m)
    phi_new = float(rng.normal())
    assert aggregate(y, phi_new - phi) == pytest.approx(
        gradient_pairwise_prediction(phi_new, phi, y), abs=1e-13
    )


def test_the_prediction_is_the_potential_plus_the_floor_residual_of_the_potential():
    """The decomposition the derivation claims, read off term by term."""
    rng = np.random.default_rng(3)
    phi, y = rng.normal(size=8), rng.normal(size=8)
    phi_new = 1.75
    got = gradient_pairwise_prediction(phi_new, phi, y)
    assert got == pytest.approx(phi_new + (y.mean() - phi.mean()), abs=1e-13)


def test_a_constant_potential_makes_the_pairwise_arm_the_floor_shifted():
    # phi constant means a zero flow, so the read-out must be exactly the floor.
    y = np.array([1.0, 2.0, 3.0, 4.0])
    phi = np.full(4, 7.0)
    assert gradient_pairwise_prediction(7.0, phi, y) == pytest.approx(y.mean(), abs=1e-13)


def test_a_non_gradient_flow_breaks_it_by_exactly_its_mean_non_gradient_part():
    rng = np.random.default_rng(7)
    phi, y = rng.normal(size=10), rng.normal(size=10)
    phi_new = float(rng.normal())
    extra = rng.normal(size=10)            # whatever the flow carries beyond the gradient
    flow = (phi_new - phi) + extra
    gap = aggregate(y, flow) - gradient_pairwise_prediction(phi_new, phi, y)
    assert gap == pytest.approx(float(np.mean(extra)), abs=1e-13)
    assert non_gradient_residual(phi_new, phi, y, flow) == pytest.approx(gap, abs=1e-13)


def test_a_non_gradient_flow_with_zero_mean_leaves_the_prediction_untouched():
    # Why "96.6 % gradient" is not the whole story: what survives the read-out is the MEAN of the
    # non-gradient part over the anchors, which can vanish even when the part itself does not.
    rng = np.random.default_rng(11)
    phi, y = rng.normal(size=6), rng.normal(size=6)
    extra = np.array([1.0, -1.0, 2.0, -2.0, 0.5, -0.5])
    assert extra.mean() == pytest.approx(0.0, abs=1e-13)
    assert non_gradient_residual(0.4, phi, y, (0.4 - phi) + extra) == pytest.approx(0.0, abs=1e-13)


def test_shape_mismatch_is_refused():
    with pytest.raises(ValueError, match="same length"):
        gradient_pairwise_prediction(0.0, np.ones(4), np.ones(3))


def test_no_anchors_is_refused():
    with pytest.raises(ValueError, match="at least one anchor"):
        gradient_pairwise_prediction(0.0, np.array([]), np.array([]))

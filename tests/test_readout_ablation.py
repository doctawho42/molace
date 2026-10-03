import networkx as nx
import numpy as np
import pytest
from molace.hodge import complex as cx
from molace.hodge import energy


def _setup(n=60, seed=0):
    rng = np.random.default_rng(seed)
    fp = (rng.random((n, 2048)) < 0.02).astype(np.uint8)
    w = np.zeros(2048); w[rng.choice(2048, 30, replace=False)] = rng.normal(size=30)
    y = fp @ w + rng.normal(scale=0.1, size=n)
    g = nx.gnp_random_graph(n, 0.3, seed=seed)
    return fp, y, cx.build(g)


def test_a_bias_free_linear_readout_is_a_gradient_flow_to_machine_precision():
    """The §10 theorem. A linear readout on a difference of a frozen encoder is phi_i - phi_j."""
    fp, y, c = _setup()
    out = energy.readout_ablation(fp, y, c, seed=0)
    assert out["linear"]["curl"] < 1e-12
    assert out["linear"]["harmonic"] < 1e-12
    assert out["linear"]["gradient"] == pytest.approx(1.0, abs=1e-10)


def test_an_affine_readout_puts_exactly_three_c_on_every_triangle():
    """h(z) = w.z + c gives a triangle sum of 3c regardless of the data."""
    fp, y, c = _setup()
    out = energy.readout_ablation(fp, y, c, seed=0, affine_bias=0.7)
    assert out["affine_triangle_sum"] == pytest.approx(3 * 0.7, abs=1e-9)


def test_a_nonlinear_readout_has_curl_bounded_away_from_zero():
    """Magnitudes are probe-specific; only the sign of the effect is asserted."""
    fp, y, c = _setup()
    out = energy.readout_ablation(fp, y, c, seed=0)
    assert out["mlp"]["curl"] > 1e-6
    assert out["mlp"]["curl"] > out["linear"]["curl"] * 1000


def test_the_encoder_is_frozen_so_the_linear_problem_is_convex_and_reproducible():
    fp, y, c = _setup()
    a = energy.readout_ablation(fp, y, c, seed=0)["linear"]
    b = energy.readout_ablation(fp, y, c, seed=1)["linear"]
    assert a["curl"] == pytest.approx(b["curl"], abs=1e-12)
    assert a["gradient"] == pytest.approx(b["gradient"], abs=1e-10)

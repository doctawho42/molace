import numpy as np
import pytest
from molace.measures.rogi import ROGI_FLAVOUR, roughness


def _two_cluster_fp(n=120, seed=0):
    """Two well-separated fingerprint clusters."""
    rng = np.random.default_rng(seed)
    base_a = (rng.random(2048) < 0.02)
    base_b = (rng.random(2048) < 0.02)
    rows = []
    for i in range(n):
        base = base_a if i < n // 2 else base_b
        row = base.copy()
        flip = rng.choice(2048, size=5, replace=False)
        row[flip] = ~row[flip]
        rows.append(row)
    return np.array(rows, dtype=np.uint8)


def test_flavour_is_recorded():
    assert ROGI_FLAVOUR in {"rogi", "rogi-xd"}


def test_a_smooth_landscape_is_less_rough_than_a_shuffled_one():
    fp = _two_cluster_fp()
    y_smooth = np.concatenate([np.full(60, 5.0), np.full(60, 9.0)])
    rng = np.random.default_rng(1)
    y_shuffled = rng.permutation(y_smooth)
    assert roughness(fp, y_smooth).value < roughness(fp, y_shuffled).value


def test_value_is_finite_and_in_the_unit_interval():
    fp = _two_cluster_fp()
    y = np.concatenate([np.full(60, 5.0), np.full(60, 9.0)])
    v = roughness(fp, y).value
    assert np.isfinite(v) and 0.0 <= v <= 1.0


def test_deterministic_across_calls():
    fp = _two_cluster_fp()
    y = np.linspace(4.0, 10.0, 120)
    assert roughness(fp, y).value == pytest.approx(roughness(fp, y).value)


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="length"):
        roughness(_two_cluster_fp(), np.ones(10))


def test_constant_label_raises_rather_than_returning_nan():
    fp = _two_cluster_fp()
    with pytest.raises(ValueError, match="constant"):
        roughness(fp, np.ones(120))

import numpy as np
import pandas as pd
import pytest
from molace.hodge import controls


#: Branch notation, because a halogen in the middle of a chain is divalent and will not parse.
UNITS = ["C", "CC", "C(C)", "C(O)", "C(N)", "C(F)", "C(Cl)", "C(S)"]


def _frame(n=140, seed=0):
    """A synthetic target with enough structural variety to give a non-constant prediction.

    The plan's fixture built alcohols whose only variable was chain length, so the fingerprints were
    nearly identical, every model predicted a near-constant on the test split, and the pointwise floor
    flow came out identically zero -- which `fractions` correctly refuses to score. Measured here: 138
    distinct fingerprints out of 140 and a test-prediction standard deviation of 1.23 to 1.60 across
    the three learners.
    """
    rng = np.random.default_rng(seed)
    smiles, counts = [], []
    for _ in range(n):
        k = int(rng.integers(3, 9))
        picks = rng.integers(0, len(UNITS), size=k)
        smiles.append("C" + "".join(UNITS[i] for i in picks))
        counts.append(np.bincount(picks, minlength=len(UNITS)))
    counts = np.array(counts, dtype=float)
    w = np.array([1.4, -1.1, 0.8, 1.2, -0.7, 0.6, -0.9, 1.0])
    y = 6.0 + counts @ w + rng.normal(scale=0.2, size=n)
    return pd.DataFrame({
        "smiles": smiles, "y": y,
        "cliff_mol": (rng.random(n) < 0.2).astype(int),
        "split": ["train"] * int(0.8 * n) + ["test"] * (n - int(0.8 * n)),
    })


def test_the_pointwise_floor_is_curl_free_to_machine_precision():
    """Differencing a pointwise model on the edges must give an exactly gradient flow.

    This is a validity check on the whole pipeline: if the floor shows curl, the complex or the
    projection is wrong, not the science.
    """
    out = controls.energy_controls(_frame(), seed=0, k=10, triangle_budget=20000)
    assert out["pointwise_floor"]["curl"] < 1e-10
    assert out["pointwise_floor"]["harmonic"] < 1e-10
    assert out["pointwise_floor"]["gradient"] == pytest.approx(1.0, abs=1e-9)


def test_all_three_controls_report_fractions_that_sum_to_one():
    out = controls.energy_controls(_frame(), seed=0, k=10, triangle_budget=20000)
    for arm in ("trained", "shuffled", "pointwise_floor"):
        assert sum(out[arm].values()) == pytest.approx(1.0, abs=1e-8), arm


def test_fractions_are_scale_invariant_so_no_norm_matching_is_needed():
    """The reason the spec's norm-matching control is not implemented."""
    from molace.hodge import complex as cx
    from molace.hodge.decompose import decompose
    from molace.hodge.energy import fractions
    import networkx as nx
    c = cx.build(nx.gnp_random_graph(30, 0.3, seed=0))
    rng = np.random.default_rng(0)
    f = rng.normal(size=len(c.edges))
    a = fractions(decompose(c, f))
    b = fractions(decompose(c, 11.7 * f))
    for key in a:
        assert a[key] == pytest.approx(b[key], abs=1e-9), key


def test_the_shuffled_arm_is_reported_and_differs_from_the_floor():
    out = controls.energy_controls(_frame(), seed=0, k=10, triangle_budget=20000)
    assert out["shuffled"]["curl"] > out["pointwise_floor"]["curl"]


def test_curl_versus_dispersion_returns_both_and_their_difference():
    out = controls.curl_versus_dispersion(_frame(), seed=0, k=10, triangle_budget=20000)
    for key in ("rho_curl", "rho_dispersion", "rho_difference", "n_edges"):
        assert key in out
    assert out["rho_difference"] == pytest.approx(out["rho_curl"] - out["rho_dispersion"], abs=1e-9)


def test_a_complex_without_triangles_raises_instead_of_reporting_zero_curl():
    df = _frame(n=40)
    with pytest.raises(ValueError, match="no triangles"):
        controls.energy_controls(df, seed=0, k=1, triangle_budget=10)

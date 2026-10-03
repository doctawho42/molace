import numpy as np
import pandas as pd
import pytest
from molace.models import gap


def test_rmse_matches_the_definition():
    y = np.array([1.0, 2.0, 3.0])
    p = np.array([1.0, 2.0, 5.0])
    assert gap.rmse(y, p) == pytest.approx(np.sqrt(4 / 3))


def test_the_three_error_numbers_telescope_to_the_gap_exactly():
    r = gap.GapResult(
        pointwise=gap.ArmResult(1.00, 1.20, "svm"),
        knn_floor=gap.ArmResult(0.90, 1.10, "knn"),
        pairwise=gap.ArmResult(0.70, 0.95, "hgb"),
        gap=0.30, access=0.10, correction=0.20, per_learner_gap={},
    )
    assert r.access + r.correction == pytest.approx(r.gap, abs=1e-12)


def test_gap_result_rejects_an_inconsistent_decomposition():
    with pytest.raises(ValueError, match="decomposition"):
        gap.GapResult(
            pointwise=gap.ArmResult(1.0, 1.2, "svm"),
            knn_floor=gap.ArmResult(0.9, 1.1, "knn"),
            pairwise=gap.ArmResult(0.7, 0.95, "hgb"),
            gap=0.30, access=0.10, correction=0.99, per_learner_gap={},
        )


def _toy_frame(n=200, seed=0):
    """A synthetic target shaped like a MoleculeACE frame."""
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=(n, 12))
    smiles = ["C" * (1 + int(b.sum())) + "O" for b in bits]
    y = 5.0 + bits[:, :4] @ np.array([1.0, -0.8, 0.6, 1.3]) + rng.normal(scale=0.2, size=n)
    return pd.DataFrame({
        "smiles": smiles,
        "y": y,
        "cliff_mol": (rng.random(n) < 0.25).astype(int),
        "split": ["train"] * int(0.8 * n) + ["test"] * (n - int(0.8 * n)),
    })


def test_evaluate_target_returns_a_consistent_decomposition():
    r = gap.evaluate_target(_toy_frame(), seed=0)
    assert r.access + r.correction == pytest.approx(r.gap, abs=1e-9)
    assert r.knn_floor.selected == "knn"


def test_both_arms_aggregate_the_same_matched_families_so_neither_picks_its_best():
    """A best-of-arm selection on one side only would bias the gap toward that side.

    The primary therefore averages the families whose pointwise and pairwise learners are the same
    model class. The kernel family is excluded because its pairwise slot is linear while its
    pointwise slot is RBF, which is not a matched comparison.
    """
    r = gap.evaluate_target(_toy_frame(), seed=0)
    assert r.pointwise.selected == r.pairwise.selected == "mean(hgb,mlp)"


def test_the_kernel_family_is_reported_but_not_in_the_primary():
    r = gap.evaluate_target(_toy_frame(), seed=0)
    assert "svm" in r.per_learner_gap            # reported
    assert "svm" not in gap.MATCHED_FAMILIES     # but not averaged into the primary


def test_evaluate_target_reports_cliff_rmse_separately():
    r = gap.evaluate_target(_toy_frame(), seed=0)
    for arm in (r.pointwise, r.knn_floor, r.pairwise):
        assert np.isfinite(arm.rmse_all) and np.isfinite(arm.rmse_cliff)


def test_a_target_with_no_cliff_compounds_in_test_gives_nan_cliff_not_a_crash():
    df = _toy_frame()
    df.loc[df["split"] == "test", "cliff_mol"] = 0
    r = gap.evaluate_target(df, seed=0)
    assert np.isnan(r.pointwise.rmse_cliff)
    assert np.isfinite(r.pointwise.rmse_all)


def test_per_learner_gaps_cover_all_three_learners():
    r = gap.evaluate_target(_toy_frame(), seed=0)
    assert set(r.per_learner_gap) == {"svm", "hgb", "mlp"}


def test_no_cross_validation_runs_inside_evaluate_target():
    """Measured cost, not taste: with 5-fold CV in both arms the sweep is a 20-hour job.

    One fit per learner per arm instead, which is what makes 30 targets by 3 seeds affordable.
    """
    import inspect

    src = inspect.getsource(gap.evaluate_target)
    assert "select_by_cv" not in src
    assert "KFold" not in src

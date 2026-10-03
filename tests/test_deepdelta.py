import numpy as np
import pytest
from molace.data import deepdelta as dd


def test_ten_benchmark_datasets():
    assert len(dd.DD_DATASETS) == 10
    assert "Caco2" in dd.DD_DATASETS and "FUBrain" in dd.DD_DATASETS


def test_prediction_files_are_two_rows_true_then_predicted():
    true, pred = dd.load_predictions("Caco2", "RandomForest", 0)
    assert true.shape == pred.shape
    assert true.shape[0] == 82810


def test_the_self_pair_has_a_true_delta_of_zero():
    true, _ = dd.load_predictions("Caco2", "DeepDelta5", 0)
    assert true[0] == pytest.approx(0.0)


def test_the_families_share_the_pair_multiset_but_not_its_order():
    """The precise comparability statement, measured on all ten datasets.

    The three families evaluate the same set of pairs but enumerate them in different orders. So
    order-invariant aggregate metrics are comparable across families, which is what the gap needs,
    and a pair-by-pair comparison across families is not available.
    """
    for ds in dd.DD_DATASETS:
        rows = [dd.load_predictions(ds, fam, 0)[0] for fam in dd.FAMILIES]
        ref = rows[0]
        assert all(r.shape == ref.shape for r in rows), ds
        assert all(np.allclose(np.sort(r), np.sort(ref)) for r in rows), ds


def test_the_order_really_does_differ_so_no_one_relies_on_it():
    a, _ = dd.load_predictions("Caco2", "RandomForest", 0)
    b, _ = dd.load_predictions("Caco2", "DeepDelta5", 0)
    assert not np.allclose(a, b)


def test_published_gaps_has_one_row_per_dataset_and_finite_numbers():
    g = dd.published_gaps()
    assert len(g) == 10
    for col in ("rmse_RandomForest", "rmse_ChemProp50", "rmse_DeepDelta5",
                "gap_vs_rf", "gap_vs_chemprop"):
        assert np.isfinite(g[col]).all(), col


def test_gap_sign_convention_matches_the_project(dataset="Caco2"):
    """Positive means the pairwise model wins, as everywhere else in this project."""
    g = dd.published_gaps().set_index("dataset")
    row = g.loc[dataset]
    assert row["gap_vs_rf"] == pytest.approx(row["rmse_RandomForest"] - row["rmse_DeepDelta5"])


def test_unknown_family_raises():
    with pytest.raises(KeyError, match="family"):
        dd.load_predictions("Caco2", "XGBoost", 0)

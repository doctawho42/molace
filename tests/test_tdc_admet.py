import numpy as np
import pytest
from molace.data import tdc_admet as t


def test_nine_regression_tasks_matching_the_prereg():
    from molace.analysis.prereg import load_prereg
    assert len(t.TDC_REGRESSION) == 9
    assert list(t.TDC_REGRESSION) == load_prereg()["task_sets"]["holdout"]["datasets"]


def test_a_task_loads_into_the_project_frame_shape():
    df = t.load_tdc("caco2_wang")
    assert list(df.columns) == ["smiles", "y", "cliff_mol", "split"]
    assert set(df["split"]) == {"train", "test"}
    assert (df["cliff_mol"] == 0).all()        # TDC carries no cliff annotation
    assert len(df) > 200


def test_labels_are_finite_and_non_constant():
    df = t.load_tdc("caco2_wang")
    assert np.isfinite(df["y"]).all()
    assert df["y"].std() > 0


def test_unknown_task_raises():
    with pytest.raises(KeyError, match="holdout"):
        t.load_tdc("tox21")


def test_the_holdout_refuses_to_run_before_the_spine_is_recorded(tmp_path, monkeypatch):
    monkeypatch.setattr(t, "SPINE_REPORT", tmp_path / "absent.txt")
    with pytest.raises(RuntimeError, match="holdout"):
        t.assert_spine_recorded()

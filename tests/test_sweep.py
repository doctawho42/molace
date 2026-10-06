import json

import numpy as np
import pytest
from molace.analysis import sweep


def test_record_schema_has_every_field_the_analysis_reads(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep, "CACHE", tmp_path)
    rec = sweep.measure_target("CHEMBL2835_Ki", seed=0)
    for key in ("dataset", "receptor_class", "seed", "prereg", "n_molecules",
                "assortativity", "assortativity_coverage",
                "unbiased_homophily", "label_informativeness", "adjusted_homophily",
                "rogi", "rogi_flavour", "dirichlet",
                "mean_degree", "n_components", "largest_component_fraction", "n_triangles",
                "rmse_pointwise", "rmse_knn_floor", "rmse_pairwise",
                "rmse_cliff_pointwise", "rmse_cliff_knn_floor", "rmse_cliff_pairwise",
                "gap", "access", "correction", "selected_pointwise", "selected_pairwise"):
        assert key in rec, key


def test_record_is_cached_and_reused(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep, "CACHE", tmp_path)
    first = sweep.measure_target("CHEMBL2835_Ki", seed=0)
    path = tmp_path / "CHEMBL2835_Ki__seed0.json"
    assert path.is_file()
    # corrupt the cache with a sentinel; a reuse must return the sentinel, not recompute
    rec = json.loads(path.read_text(encoding="utf-8"))
    rec["gap"] = -999.0
    path.write_text(json.dumps(rec), encoding="utf-8")
    assert sweep.measure_target("CHEMBL2835_Ki", seed=0)["gap"] == -999.0
    assert sweep.measure_target("CHEMBL2835_Ki", seed=0, force=True)["gap"] != -999.0
    assert first["dataset"] == "CHEMBL2835_Ki"


def test_decomposition_holds_in_the_record(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep, "CACHE", tmp_path)
    rec = sweep.measure_target("CHEMBL2835_Ki", seed=0)
    assert rec["access"] + rec["correction"] == pytest.approx(rec["gap"], abs=1e-9)


def test_every_record_carries_the_prereg_hash(tmp_path, monkeypatch):
    from molace.analysis.prereg import prereg_fingerprint
    monkeypatch.setattr(sweep, "CACHE", tmp_path)
    assert sweep.measure_target("CHEMBL2835_Ki", seed=0)["prereg"] == prereg_fingerprint()


def test_sweep_averages_over_seeds_and_returns_one_row_per_target(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep, "CACHE", tmp_path)
    df = sweep.run_sweep(["CHEMBL2835_Ki", "CHEMBL4203_Ki"], seeds=(0, 1))
    assert len(df) == 2
    assert set(df["dataset"]) == {"CHEMBL2835_Ki", "CHEMBL4203_Ki"}
    # Operand order matters: run_sweep drops the seed column, so touching df["seed"] first raises
    # KeyError before the short circuit can save it.
    assert "seed" not in df.columns or df["seed"].isna().all()


def test_a_failing_target_does_not_lose_the_others(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep, "CACHE", tmp_path)
    df = sweep.run_sweep(["CHEMBL2835_Ki", "CHEMBL_nope"], seeds=(0,))
    assert len(df) == 1
    assert df.iloc[0]["dataset"] == "CHEMBL2835_Ki"


def test_a_cache_written_under_a_different_plan_is_not_served_silently(tmp_path, monkeypatch, caplog):
    """Serving it silently once put 30 of 90 records into the spine under a superseded plan.

    The guard decides on the parameters the measurement actually reads, not on the blob hash: a
    prose-only edit to the plan must not force a 70-minute recompute, and a changed k must not pass.
    """
    import json
    import logging

    monkeypatch.setattr(sweep, "CACHE", tmp_path)
    name, seed = "CHEMBL2835_Ki", 0
    path = tmp_path / f"{name}__seed{seed}.json"
    k = sweep.load_prereg()["graphs"]["primary"]["k"]

    # same parameters, superseded stamp: served, with a warning that says so
    path.write_text(json.dumps({"dataset": name, "seed": seed, "prereg": "0" * 40,
                                "plan_k": k, "assortativity": 0.123}), encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        rec = sweep.measure_target(name, seed)
    assert rec["assortativity"] == 0.123, "a prose-only plan edit must not discard the measurement"
    assert "rather than" in caplog.text

    # written before parameters were recorded: served, but loudly, because it cannot be checked
    path.write_text(json.dumps({"dataset": name, "seed": seed, "prereg": "0" * 40,
                                "assortativity": 0.456}), encoding="utf-8")
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        rec = sweep.measure_target(name, seed)
    assert rec["assortativity"] == 0.456
    assert "cannot be checked" in caplog.text

    # a parameter the measurement reads has changed: refused rather than mixed into one table
    path.write_text(json.dumps({"dataset": name, "seed": seed, "prereg": "0" * 40,
                                "plan_k": k + 1, "assortativity": 0.123}),
                    encoding="utf-8")
    with pytest.raises(ValueError, match="would mix two plans"):
        sweep.measure_target(name, seed)

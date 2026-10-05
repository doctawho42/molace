"""Checkpoint files must survive a resume without changing or losing a measurement.

Three scripts write a partial CSV per target and skip finished work on restart. Before 2026-10-05 the
write was a non-atomic full-file rewrite, the read accepted whatever that left behind, and
`pd.read_csv` without `float_precision` did not round-trip, so a resumed run produced numbers an
uninterrupted one would not have.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from molace.analysis.checkpoint import load_checkpoint, save_checkpoint

REQUIRED = ["dataset", "skill"]


def test_a_row_torn_by_an_interrupted_write_is_dropped_rather_than_resumed(tmp_path, capsys):
    p = tmp_path / "partial.csv"
    p.write_text("dataset,skill,extra\nA,0.5,1\nB,0.7,2\nC,")        # killed mid-write
    rows = load_checkpoint(p, REQUIRED)
    assert [r["dataset"] for r in rows] == ["A", "B"]
    assert "dropping 1 incomplete row" in capsys.readouterr().out


def test_a_checkpoint_from_an_older_schema_is_refused_not_resumed_onto(tmp_path):
    p = tmp_path / "partial.csv"
    p.write_text("dataset,other\nA,1\n")
    with pytest.raises(ValueError, match="older schema"):
        load_checkpoint(p, REQUIRED)


def test_a_missing_checkpoint_is_simply_an_empty_start(tmp_path):
    assert load_checkpoint(tmp_path / "absent.csv", REQUIRED) == []


def test_every_float_survives_the_round_trip_exactly(tmp_path):
    """The defect this pins: pandas' fast converter returns a nearby double, not the same one."""
    rng = np.random.default_rng(0)
    vals = list(rng.normal(size=400) * 10.0 ** rng.integers(-8, 8, size=400))
    p = tmp_path / "partial.csv"
    save_checkpoint(p, [{"dataset": f"T{i}", "skill": v} for i, v in enumerate(vals)])
    back = [r["skill"] for r in load_checkpoint(p, REQUIRED)]
    assert back == vals, "a resumed value must be the value that was written"


def test_save_is_atomic_and_leaves_no_temporary_behind(tmp_path):
    p = tmp_path / "partial.csv"
    save_checkpoint(p, [{"dataset": "A", "skill": 0.5}])
    save_checkpoint(p, [{"dataset": "A", "skill": 0.5}, {"dataset": "B", "skill": 0.7}])
    assert sorted(q.name for q in tmp_path.iterdir()) == ["partial.csv"]
    assert len(load_checkpoint(p, REQUIRED)) == 2


def test_a_row_complete_but_refused_by_a_design_gate_is_kept(tmp_path):
    """`required` must name only columns a gated-out row still carries, or gates look like tears."""
    p = tmp_path / "partial.csv"
    save_checkpoint(p, [{"dataset": "A", "skill": 0.5, "gated_measurement": None}])
    rows = load_checkpoint(p, REQUIRED)
    assert len(rows) == 1 and pd.isna(rows[0]["gated_measurement"])

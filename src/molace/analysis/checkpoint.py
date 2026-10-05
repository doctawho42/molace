"""Checkpoint files that a resumed run cannot corrupt or silently perturb.

Three scripts write a partial CSV per target and skip finished work on restart. They shared three
defects. The write was a full-file rewrite and not atomic, so a kill mid-write leaves a torn trailing
row. The read accepted that row, because any row whose key parsed counted as finished, and pandas
NaN-pads the rest; the row was then copied verbatim into the committed table and never recomputed. And
`pd.read_csv` without `float_precision` uses an inexact converter, so a value written at full
precision does not parse back to the same double, and resumed rows carried numbers an uninterrupted
run would not have produced.

One implementation, so the next script with a checkpoint inherits the fixes rather than the defects.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


def load_checkpoint(path: Path, required: list[str]) -> list[dict]:
    """Rows from a checkpoint, dropping any a torn write left incomplete.

    `required` names the columns every complete row must carry. Keep it to columns a legitimately
    gated-out row still has: a row that is complete but whose measurement was refused by a design
    gate must not be mistaken for a torn one.
    """
    if not path.is_file():
        return []
    t = pd.read_csv(path, float_precision="round_trip")
    missing = [c for c in required if c not in t.columns]
    if missing:
        raise ValueError(
            f"{path} was written by an older schema and is missing {missing}; "
            "delete it to recompute rather than resuming onto it"
        )
    good = t.dropna(subset=required)
    if len(good) < len(t):
        print(f"  {path}: dropping {len(t) - len(good)} incomplete row(s) left by a torn write")
    return good.to_dict("records")


def save_checkpoint(path: Path, rows: list[dict]) -> None:
    """Write to a temporary file and rename, so a kill can never leave a torn row behind."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    os.replace(tmp, path)                    # atomic within a filesystem

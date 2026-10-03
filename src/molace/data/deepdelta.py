"""The only published per-task pointwise-against-pairwise gap, recovered from shipped files.

Fralish, Chen, Skaluba, Reker, J. Cheminform. 15:101 (2023). Results/<Family>/<Dataset>_<Family>_
<repeat>.csv is a 2-row wide CSV: row 0 the true delta, row 1 the predicted delta, one column per
pair, 5 repeats of 10-fold cross-validation per dataset with the folds pooled per file.

All three families are scored on the same delta task -- for the pointwise families the predicted delta
is the difference of their pointwise predictions -- but "the same" needs to be precise, and a test
pins the precise version rather than the hoped-for one. Measured on all ten datasets: the three
families' true-delta rows are the same MULTISET but not in the same ORDER. So order-invariant
aggregate metrics (RMSE, MAE) are comparable across families, which is what the gap needs, while a
pair-by-pair comparison across families is not available and must not be attempted.

Upstream has one filename irregularity: in the ChemProp50 directory the HalfLife files are named
`HalfLife_DeepDelta_50_Epoch_Trad_<n>.csv`, their internal label for the traditional single-molecule
mode, while the other nine follow `<Dataset>_ChemProp50_<n>.csv`. Verified to be the matching arm:
same pair count and the same true-delta multiset as the other two families. The loader falls back to a
glob within the family directory rather than hard-coding the exception.
"""
from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3] / "data" / "raw" / "DeepDelta"
RESULTS = ROOT / "Results"
FAMILIES = ("RandomForest", "ChemProp50", "DeepDelta5")
PAIRWISE_FAMILY = "DeepDelta5"
REPEATS = (0, 1, 2, 3, 4)

DD_DATASETS: tuple[str, ...] = (
    "Caco2", "FUBrain", "FreeSolv", "HalfLife", "HemoTox",
    "HepClear", "MicroClear", "RenClear", "Sol", "VDss",
)


def load_predictions(dataset: str, family: str, repeat: int):
    if family not in FAMILIES:
        raise KeyError(f"unknown family {family!r}; expected one of {FAMILIES}")
    path = RESULTS / family / f"{dataset}_{family}_{repeat}.csv"
    if not path.is_file():
        # Upstream names ChemProp50's HalfLife files after their internal label. One glob inside the
        # family directory absorbs that rather than a hard-coded special case.
        alts = sorted((RESULTS / family).glob(f"{dataset}_*_{repeat}.csv"))
        if len(alts) != 1:
            raise FileNotFoundError(
                f"{path} is missing and {len(alts)} alternatives matched "
                f"{dataset}_*_{repeat}.csv in {RESULTS / family}; run scripts/fetch_data.sh"
            )
        path = alts[0]
    arr = pd.read_csv(path).to_numpy(dtype=float)
    if arr.shape[0] != 2:
        raise ValueError(f"{path} has {arr.shape[0]} rows, expected 2 (true, predicted)")
    return arr[0], arr[1]


def _rmse_over_repeats(dataset: str, family: str) -> float:
    errs = []
    for r in REPEATS:
        true, pred = load_predictions(dataset, family, r)
        errs.append(float(np.sqrt(np.mean((true - pred) ** 2))))
    return float(np.mean(errs))


@functools.lru_cache(maxsize=1)
def published_gaps() -> pd.DataFrame:
    rows = []
    for ds in DD_DATASETS:
        rec = {"dataset": ds}
        for fam in FAMILIES:
            rec[f"rmse_{fam}"] = _rmse_over_repeats(ds, fam)
        rec["gap_vs_rf"] = rec["rmse_RandomForest"] - rec[f"rmse_{PAIRWISE_FAMILY}"]
        rec["gap_vs_chemprop"] = rec["rmse_ChemProp50"] - rec[f"rmse_{PAIRWISE_FAMILY}"]
        rows.append(rec)
    return pd.DataFrame(rows)

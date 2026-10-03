"""MoleculeACE loader.

Three traps this module exists to close, all measured on the shipped data:
  * the canonical label is `y [pEC50/pKi]`; the column named `y` is standardised and negative,
  * `cliff_mol` must come from the per-target CSV, because metadata/datasets.csv still
    describes the pre-Correction data (Correction: PubMed 36995229) and disagrees with the
    CSVs on cliff counts for 28 of 30 targets,
  * the repo must be cloned, not pip-installed, or Data/results/ is missing.
"""
from __future__ import annotations

import functools
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3] / "data" / "raw" / "MoleculeACE" / "MoleculeACE" / "Data"
BENCH = ROOT / "benchmark_data"
METADATA = BENCH / "metadata" / "datasets.csv"
RESULTS = ROOT / "results" / "MoleculeACE_results.csv"

LABEL_COLUMN = "y [pEC50/pKi]"


def _require_data() -> None:
    if not BENCH.is_dir():
        raise FileNotFoundError(
            f"{BENCH} is missing. Run scripts/fetch_data.sh; MoleculeACE must be cloned, "
            "because Data/results/ is not in the pip package."
        )


@functools.lru_cache(maxsize=1)
def _datasets() -> tuple[str, ...]:
    _require_data()
    return tuple(sorted(p.stem for p in BENCH.glob("*.csv")))


DATASETS = list(_datasets())


@functools.lru_cache(maxsize=64)
def load_target(name: str) -> pd.DataFrame:
    """One target as `smiles, y, cliff_mol, split`, in file order."""
    if name not in _datasets():
        raise KeyError(f"{name!r} is not one of the 30 MoleculeACE targets")
    raw = pd.read_csv(BENCH / f"{name}.csv")
    return pd.DataFrame(
        {
            "smiles": raw["smiles"].astype(str),
            "y": raw[LABEL_COLUMN].astype(float),
            "cliff_mol": raw["cliff_mol"].astype(int),
            "split": raw["split"].astype(str),
        }
    )


@functools.lru_cache(maxsize=1)
def _classes() -> dict[str, str]:
    _require_data()
    md = pd.read_csv(METADATA)
    # Only the Receptor Class column is trusted here; the cliff counts in this file are stale.
    return dict(zip(md["Dataset"].astype(str), md["Receptor Class"].astype(str)))


RECEPTOR_CLASSES = _classes()


def receptor_class(name: str) -> str:
    return _classes()[name]

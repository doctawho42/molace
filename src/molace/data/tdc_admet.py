"""The pre-registered out-of-domain holdout: 9 TDC ADMET regression tasks.

Nine, not twenty-two: the other thirteen tasks in TDC's admet_benchmark are classification. The
names come from the pre-registration and a test asserts they match, so the holdout cannot quietly
grow or shrink.

These frames carry no cliff annotation, so the categorical arm does not apply here and only the
continuous statistic is computed. The split is TDC's own scaffold split, not MoleculeACE's.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

SPINE_REPORT = Path(__file__).resolve().parents[3] / "results" / "report_spine.txt"

TDC_REGRESSION: tuple[str, ...] = (
    "caco2_wang", "lipophilicity_astrazeneca", "solubility_aqsoldb", "ppbr_az",
    "vdss_lombardo", "half_life_obach", "clearance_hepatocyte_az",
    "clearance_microsome_az", "ld50_zhu",
)


def assert_spine_recorded() -> None:
    """The holdout may only be opened after the spine result exists."""
    if not SPINE_REPORT.is_file():
        raise RuntimeError(
            "the holdout may not be opened before the spine result is recorded at "
            f"{SPINE_REPORT}; opening it first turns the holdout into a second training set"
        )


def load_tdc(name: str) -> pd.DataFrame:
    if name not in TDC_REGRESSION:
        raise KeyError(f"{name!r} is not one of the 9 pre-registered holdout tasks")
    from tdc.single_pred import ADME, Tox

    loader = Tox if name == "ld50_zhu" else ADME
    data = loader(name=name)
    split = data.get_split(method="scaffold")
    frames = []
    for part, tag in (("train", "train"), ("valid", "train"), ("test", "test")):
        d = split[part]
        frames.append(pd.DataFrame({
            "smiles": d["Drug"].astype(str),
            "y": d["Y"].astype(float),
            "cliff_mol": 0,
            "split": tag,
        }))
    return pd.concat(frames, ignore_index=True)

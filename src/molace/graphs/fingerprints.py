"""The project's single molecular representation.

The design fixes exactly one free parameter, the representation, because the cliff set and
every graph statistic move with it. That discipline only holds if there is one call site, so
this module is the only place a fingerprint is computed. A second one is a bug.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

RADIUS = 2
N_BITS = 2048

_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)


def ecfp4(smiles: Sequence[str]) -> np.ndarray:
    """Binary ECFP4 (Morgan radius 2, 2048 bits) as an (n, 2048) uint8 array.

    Raises on an unparseable SMILES rather than dropping it: a dropped row would shift every
    label index downstream and silently corrupt every per-target number.
    """
    out = np.zeros((len(smiles), N_BITS), dtype=np.uint8)
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            raise ValueError(f"RDKit could not parse SMILES at index {i}: {s!r}")
        out[i] = _GEN.GetFingerprintAsNumPy(mol).astype(np.uint8)
    return out


def tanimoto_matrix(fp: np.ndarray) -> np.ndarray:
    """Dense pairwise Tanimoto, (n, n) float32, 1.0 on the diagonal."""
    x = fp.astype(np.float32)
    inter = x @ x.T
    counts = x.sum(axis=1)
    union = counts[:, None] + counts[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(union > 0.0, inter / union, 0.0).astype(np.float32)
    np.fill_diagonal(t, 1.0)
    return t

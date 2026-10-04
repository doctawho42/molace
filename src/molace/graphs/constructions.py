"""Fingerprints, similarity metrics and graph rules, as the independent variable.

Increment 1 fixed exactly one of each so that nothing could be chosen after seeing a result, and
`molace.graphs.fingerprints` stays the single call site for that frozen choice. This module exists
for the opposite purpose: the spread of a homophily measure ACROSS reasonable constructions is the
quantity being measured, so the constructions have to vary.

Both things are true at once and the distinction matters. Tuning is choosing a construction because
of what it does to your answer. Measuring construction sensitivity is reporting what every
reasonable choice does, before picking any.
"""
from __future__ import annotations

from typing import Sequence

import networkx as nx
import numpy as np

FINGERPRINTS = ("ecfp4", "ecfp6", "maccs", "rdkit", "atompair")
METRICS = ("tanimoto", "dice", "cosine")


def _mols(smiles: Sequence[str]):
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    out = []
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        if m is None:
            raise ValueError(f"RDKit could not parse the SMILES at index {i}: {s!r}")
        out.append(m)
    return out


def fingerprint(smiles: Sequence[str], kind: str = "ecfp4") -> np.ndarray:
    """A binary fingerprint matrix, one row per molecule. Raises on an unparseable SMILES."""
    if kind not in FINGERPRINTS:
        raise KeyError(f"unknown fingerprint {kind!r}; expected one of {FINGERPRINTS}")
    from rdkit import DataStructs
    from rdkit.Chem import MACCSkeys, rdFingerprintGenerator
    from rdkit.Chem import rdMolDescriptors  # noqa: F401  (kept for RDKit import ordering)

    mols = _mols(smiles)
    if kind == "maccs":
        bits = [MACCSkeys.GenMACCSKeys(m) for m in mols]
    else:
        gen = {
            "ecfp4": lambda: rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048),
            "ecfp6": lambda: rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048),
            "rdkit": lambda: rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=2048),
            "atompair": lambda: rdFingerprintGenerator.GetAtomPairGenerator(fpSize=2048),
        }[kind]()
        bits = [gen.GetFingerprint(m) for m in mols]
    out = np.zeros((len(bits), bits[0].GetNumBits()), dtype=np.uint8)
    for i, b in enumerate(bits):
        arr = np.zeros((b.GetNumBits(),), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(b, arr)
        out[i] = arr
    return out


def similarity(fp: np.ndarray, metric: str = "tanimoto") -> np.ndarray:
    """Dense pairwise similarity in [0, 1], symmetric with a unit diagonal."""
    if metric not in METRICS:
        raise KeyError(f"unknown metric {metric!r}; expected one of {METRICS}")
    x = np.asarray(fp, dtype=np.float32)
    inter = x @ x.T
    counts = x.sum(axis=1)
    a = counts[:, None]
    b = counts[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        if metric == "tanimoto":
            s = inter / (a + b - inter)
        elif metric == "dice":
            s = 2.0 * inter / (a + b)
        else:
            s = inter / np.sqrt(a * b)
    s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(s, 1.0)
    return np.clip((s + s.T) / 2.0, 0.0, 1.0)


def threshold_graph(sim: np.ndarray, threshold: float) -> nx.Graph:
    """Every pair at or above the threshold gets an edge. Isolated nodes are kept as nodes."""
    n = sim.shape[0]
    g = nx.Graph()
    g.add_nodes_from(range(n))
    iu = np.triu_indices(n, k=1)
    hit = sim[iu] >= threshold
    g.add_edges_from(zip(iu[0][hit].tolist(), iu[1][hit].tolist()))
    return g

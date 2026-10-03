"""Three-way Hodge decomposition of an edge flow.

f = gradient + curl + harmonic, mutually orthogonal because B1 @ B2 = 0.

The two-way "gradient plus circulation" language of the source proposal is wrong, not merely
incomplete: curl-free implies gradient only when the first homology vanishes, and on these graphs
it does not. Harmonic flows are both curl-free and divergence-free yet globally inconsistent, so a
HodgeRank potential recovered on such a graph is locally consistent and globally meaningless.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse.linalg as spla

from molace.hodge.complex import Complex


@dataclass(frozen=True)
class Decomposition:
    gradient: np.ndarray
    curl: np.ndarray
    harmonic: np.ndarray
    potential: np.ndarray


def decompose(c: Complex, f: np.ndarray, atol: float = 1e-12) -> Decomposition:
    f = np.asarray(f, dtype=float).ravel()
    if len(f) != len(c.edges):
        raise ValueError(f"flow has {len(f)} entries for {len(c.edges)} edges")
    phi = spla.lsmr(c.B1.T, f, atol=atol, btol=atol)[0]
    grad = np.asarray(c.B1.T @ phi).ravel()
    rest = f - grad
    if c.B2.shape[1] == 0:
        curl = np.zeros_like(f)
    else:
        psi = spla.lsmr(c.B2, rest, atol=atol, btol=atol)[0]
        curl = np.asarray(c.B2 @ psi).ravel()
    return Decomposition(gradient=grad, curl=curl, harmonic=rest - curl, potential=phi)

"""The harmonic dimension as a certified nullity, replacing the estimator that could not fail loudly.

The split of the cycle space into curl and harmonic needs rank(B2). Computing it directly means
asking for a LARGE number -- on these graphs three to twenty thousand -- and every cheap way of
doing that saturates silently. The randomised range finder the first census used returns
rank(A @ Omega) <= min(rank A, p), so with a probe of width p = 2048 against a true rank of 4,423 it
returned exactly 2048 and nothing in its own spectrum gave it away.

Asking instead for the SMALL number is both cheaper and self-certifying:

    L1 = B1^T B1 + B2 B2^T          (the edge, or 1-, Hodge Laplacian)
    ker L1 = ker B1 cap ker B2^T    = the harmonic space
    dim harmonic = nullity(L1),     rank B2 = |E| - rank B1 - nullity(L1)

and rank B1 = |V| - c is exact and free. A shift-invert Lanczos solve returns the k eigenvalues
NEAREST zero, so if the largest of them is clearly positive then every zero has been found. That
positive eigenvalue is the certificate the range finder never had, and k is doubled until it
appears. When no certificate is obtained the result says so rather than reporting a number.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import ArpackError, ArpackNoConvergence, eigsh

from .complex import Complex

DENSE_MAX_EDGES = 400
ZERO_TOL = 1e-8
POSITIVE_TOL = 1e-6
K_START = 256
# Seeded from the cycle space rather than a constant, because doubling from 256 discards every
# solve that was too small. 0.20 covers every share measured on these graphs (the largest is
# CHEMBL4203_Ki at 0.168, which a 0.16 seed missed by 36 columns and paid for with a wasted solve).
SEED_SHARE = 0.20
# eigsh(sigma=0) factorises L1 itself, which is singular EXACTLY when the harmonic space is
# non-empty -- that is, exactly in the case this module exists for. Whether it survives is down to
# pivoting luck: real kNN complexes happen to get away with it, a 20x20 grid does not. Shifting
# into the negatives factorises L1 + |sigma| I instead, which is positive definite because L1 is
# PSD, and still returns the eigenvalues nearest zero.
SIGMA = -1e-6
# ARPACK starts from a random vector unless told otherwise, so two runs of the same census could
# disagree about whether a target certified. For a module whose output is a certificate that is not
# acceptable: the starting vector is seeded here, and the result is reproducible run to run.
V0_SEED = 0


UNCERTIFIED = -1


@dataclass(frozen=True)
class HarmonicDim:
    """dim and rank_b2 are UNCERTIFIED (-1) unless certified is True.

    The predecessor of this module returned a number saturated at its probe width and said nothing,
    which is the failure the whole increment is about. An uncertified result here therefore carries
    no count at all: there is no value to misread.
    """

    dim: int
    rank_b2: int
    certified: bool
    method: str
    spectral_gap: float
    k_used: int
    reason: str = ""


def _n_components(c: Complex) -> int:
    """Components of the 1-skeleton.

    c.edges holds ORIGINAL node labels, which need not be 0..n-1 and need not be integers at all,
    so they are mapped through c.nodes before they index a matrix. Indexing with the raw labels
    works only for graphs that happen to be labelled 0..n-1, which every test here used to be.
    """
    n = len(c.nodes)
    if not c.edges:
        return n
    pos = {v: i for i, v in enumerate(c.nodes)}
    rows = [pos[u] for u, _ in c.edges]
    cols = [pos[v] for _, v in c.edges]
    a = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    return int(connected_components(a, directed=False, return_labels=False))


def harmonic_dimension(
    c: Complex,
    zero_tol: float = ZERO_TOL,
    positive_tol: float = POSITIVE_TOL,
    k_start: int | None = None,
) -> HarmonicDim:
    """k_start defaults to a share of the cycle space rather than a constant.

    Doubling from a fixed 256 restarts the whole solve each time it is too small, and on the
    larger graphs that is most of the cost: a target whose harmonic part is 10% of a cycle space
    of 18,000 needs k above 1,800, reached only after four discarded solves. Seeding k at
    SEED_SHARE of the cycle space -- comfortably above every share measured so far -- usually
    gets it in one.
    """
    m = c.B1.shape[1]
    rank_b1 = len(c.nodes) - _n_components(c)
    if k_start is None:
        k_start = max(K_START, int(SEED_SHARE * (m - rank_b1)))
    L1 = (c.B1.T @ c.B1 + c.B2 @ c.B2.T).tocsc().astype(float)

    if m <= DENSE_MAX_EDGES:
        vals = np.sort(np.linalg.eigvalsh(L1.toarray()))
        dim = int((vals <= zero_tol).sum())
        nz = vals[vals > zero_tol]
        # a dense symmetric eigendecomposition returns the WHOLE spectrum, so there is nothing left
        # outside it to saturate against; that, not a gap test, is what certifies this branch
        return HarmonicDim(dim, m - rank_b1 - dim, True, "dense",
                           float(nz[0]) if nz.size else float("nan"), m, "full spectrum")

    k, tried = k_start, []
    while True:
        kk = int(min(k, m - 1))
        if kk in tried:
            break
        tried.append(kk)
        try:
            v0 = np.random.default_rng(V0_SEED).standard_normal(L1.shape[0])
            vals = np.sort(eigsh(L1, k=kk, sigma=SIGMA, which="LM",
                                 return_eigenvectors=False, tol=0, v0=v0))
        except (ArpackNoConvergence, ArpackError, RuntimeError) as exc:
            # ARPACK not converging, or a factorisation that fails, is a third way of not knowing.
            # It gets the same treatment as a kernel that was never bracketed: no count.
            return HarmonicDim(
                UNCERTIFIED, UNCERTIFIED, False, "lanczos", float("nan"), kk,
                f"the eigensolver failed at k={kk} ({type(exc).__name__}), so no count is "
                f"reported rather than a saturated one",
            )
        nz = vals[vals > zero_tol]
        if nz.size and nz[0] > positive_tol:
            dim = int((vals <= zero_tol).sum())
            return HarmonicDim(dim, m - rank_b1 - dim, True, "lanczos",
                               float(nz[0]), kk, "positive eigenvalue bracketed the kernel")
        if kk >= m - 1:
            break
        k *= 2
    return HarmonicDim(
        UNCERTIFIED, UNCERTIFIED, False, "lanczos", float("nan"), tried[-1],
        f"no eigenvalue above {positive_tol:g} appeared in the {tried[-1]} nearest zero, so the "
        f"kernel was never bracketed; no count is reported rather than a saturated one",
    )

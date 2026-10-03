"""ROGI, the roughness baseline the primary statistic has to beat.

Why this is load-bearing rather than decorative: target assortativity on a kNN similarity graph
measures the smoothness of a property over chemical space, which is what the roughness index
measures under a different normalisation. Hidden, that invites the reading that this project
rediscovered roughness in graph vocabulary. Stated in front, the contribution becomes the
translation between the two vocabularies plus the question of which one predicts the choice of
model class, so the headline claim is reported as incremental over this baseline.

Flavour. The spec names ROGI-XD (arXiv 2305.08238), whose selling point is comparability across
dataset sizes. Its only implementation (github.com/coleygroup/rogi-xd, last pushed 2023-07-27)
requires a separate conda environment on Python 3.9 with torch 1.13 and CUDA 11.6, git submodules
and pretrained foundation-model encoders, so it cannot be installed into this project's Python 3.11
environment. Plain ROGI ships instead, openly: ROGI_FLAVOUR records it, and the README and the
pre-registration say so. ROGI-XD's axis is the roughness of *learned* representations, while this
project fixes one representation by design, so plain ROGI on that fixed fingerprint is the
apples-to-apples comparison anyway.

Input path. `RoughnessIndex(..., metric="precomputed")` takes a square distance matrix, which lets
this wrapper hand it OUR Tanimoto matrix built from OUR single fingerprint rather than letting the
library recompute its own. That keeps the project's one-fingerprint-one-place discipline intact.
"""
from __future__ import annotations

import numpy as np
from rogi import RoughnessIndex

from molace.graphs.fingerprints import tanimoto_matrix
from molace.measures.assortativity import MeasureResult

ROGI_FLAVOUR = "rogi"


def roughness(fp: np.ndarray, y: np.ndarray) -> MeasureResult:
    """ROGI of the property landscape over Tanimoto distance on the project's fingerprint.

    Coverage is 1.0 and n_used is every molecule: roughness is computed over all of them, not only
    over those that happen to have graph neighbours.
    """
    x = np.asarray(fp)
    v = np.asarray(y, dtype=float)
    if len(v) != x.shape[0]:
        raise ValueError(f"length mismatch: {len(v)} labels for {x.shape[0]} fingerprints")
    if float(np.std(v)) == 0.0:
        raise ValueError("ROGI is undefined when the label is constant")

    d = (1.0 - tanimoto_matrix(x)).astype(np.float64)
    np.fill_diagonal(d, 0.0)
    value = RoughnessIndex(
        Y=v, X=d, metric="precomputed", norm_Y=True, verbose=False
    ).compute_index()
    return MeasureResult(value=float(value), coverage=1.0, n_used=int(len(v)))

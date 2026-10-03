"""The one place anchors are chosen.

The kNN floor arm and the pairwise arm must draw the identical set with uniform weights.
With that, the pairwise prediction is exactly the floor plus the mean learned correction:

    y_pair = (1/m) sum_i [ y_i + f(x_i, x_new) ] = y_knn + (1/m) sum_i f(x_i, x_new)

which is what licenses attributing the error change to the learned function and nothing else.
Two call sites would let the sets drift apart while the numbers stayed plausible.

m is a separate constant from the similarity graph's k. They happen to share the value 10.
Do not tie them.
"""
from __future__ import annotations

import numpy as np

M = 10


def select(sim_test_train: np.ndarray, m: int = M) -> np.ndarray:
    """Indices of the m most similar training molecules for each test molecule.

    Ties break on ascending training index, so the set is identical across runs. That
    determinism is a requirement, not a nicety: without it the decomposition identity holds in
    one run and not the next.
    """
    sim = np.asarray(sim_test_train, dtype=np.float64)
    if sim.ndim != 2:
        raise ValueError(f"expected a 2-D (n_test, n_train) matrix, got shape {sim.shape}")
    n_test, n_train = sim.shape
    if n_train == 0:
        raise ValueError("cannot choose anchors from an empty training set")
    mm = min(m, n_train)
    tie = np.tile(np.arange(n_train), (n_test, 1))
    return np.lexsort((tie, -sim), axis=1)[:, :mm].astype(np.int64)

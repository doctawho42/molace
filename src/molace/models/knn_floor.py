"""The label-access floor.

The pairwise arm reads m measured labels at test time; the pointwise arm reads none. Without
this arm a pointwise-against-pairwise gap partly measures access to data rather than
architecture. Uniform weights over the anchors chosen in models.anchors, no learned function,
no hyperparameter, no seed.
"""
from __future__ import annotations

import numpy as np


def predict(y_train: np.ndarray, anchor_idx: np.ndarray) -> np.ndarray:
    y = np.asarray(y_train, dtype=float)
    idx = np.asarray(anchor_idx, dtype=np.int64)
    if idx.ndim != 2:
        raise ValueError(f"expected (n_test, m) anchor indices, got shape {idx.shape}")
    return y[idx].mean(axis=1)

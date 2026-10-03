"""The three controls, and the criterion half 2 must pass.

Controls:
  trained          - the trained pairwise flow on the test-split comparison graph
  shuffled         - the same architecture trained on permuted labels
  pointwise_floor  - a trained pointwise model differenced on the same edges, which is exactly a
                     gradient flow and therefore a validity check on the complex and projection

The spec asks the shuffled arm to be compared at matched total flow norm. That requirement came
from the original design's raw-norm comparison, where shuffling inflates the mean-square edge
target 2.1x to 11.7x. Reporting the scale-invariant fraction removes that confound by
construction, so no rescaling happens here. If a raw norm is ever reported, norm matching must come
back with it.

Criterion: per-edge curl must beat plain anchor dispersion at predicting an edge's held-out error.
Anchor dispersion is published three times and is the thing to beat.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from molace.graphs import knn
from molace.graphs.fingerprints import ecfp4, tanimoto_matrix
from molace.hodge import complex as cx
from molace.hodge.decompose import decompose
from molace.hodge.energy import fractions, per_edge_curl
from molace.models import anchors, pairwise, pointwise


def flow_on(model: pairwise.PairwiseModel, c: cx.Complex, X: np.ndarray) -> np.ndarray:
    """Evaluate a pairwise model on every edge, oriented u -> v (second minus first)."""
    pos = {v: i for i, v in enumerate(c.nodes)}
    a = np.array([pos[u] for u, v in c.edges])
    b = np.array([pos[v] for u, v in c.edges])
    feats = pairwise.pair_features(X[a], X[b], model.kind)
    return np.asarray(model.model.predict(feats), dtype=float)


def _setup(df: pd.DataFrame, seed: int, k: int, triangle_budget: int):
    tr = df["split"].to_numpy() == "train"
    te = ~tr
    fp = ecfp4(df["smiles"].tolist())
    y = df["y"].to_numpy(dtype=float)
    X_tr, y_tr, X_te, y_te = fp[tr], y[tr], fp[te], y[te]
    sim_tr = tanimoto_matrix(X_tr)
    g_te = knn.knn_graph(tanimoto_matrix(X_te), k=k)
    c = cx.build(g_te, triangle_budget=triangle_budget, seed=seed)
    return X_tr, y_tr, X_te, y_te, sim_tr, c


def energy_controls(df: pd.DataFrame, seed: int, k: int, triangle_budget: int) -> dict:
    X_tr, y_tr, X_te, y_te, sim_tr, c = _setup(df, seed, k, triangle_budget)
    kind = "difference"

    trained = pairwise.fit("hgb", kind, X_tr, y_tr, sim_tr, anchors.M, seed)
    rng = np.random.default_rng(seed)
    shuffled = pairwise.fit("hgb", kind, X_tr, rng.permutation(y_tr), sim_tr, anchors.M, seed)

    pw_pred = pointwise.fit_predict("hgb", X_tr, y_tr, X_te, seed)
    pos = {v: i for i, v in enumerate(c.nodes)}
    floor = np.array([pw_pred[pos[v]] - pw_pred[pos[u]] for u, v in c.edges], dtype=float)

    return {
        "trained": fractions(decompose(c, flow_on(trained, c, X_te))),
        "shuffled": fractions(decompose(c, flow_on(shuffled, c, X_te))),
        "pointwise_floor": fractions(decompose(c, floor)),
        "n_edges": len(c.edges),
        "n_triangles": len(c.triangles),
    }


def curl_versus_dispersion(df: pd.DataFrame, seed: int, k: int, triangle_budget: int) -> dict:
    """Does per-edge curl predict an edge's held-out error better than anchor dispersion?"""
    X_tr, y_tr, X_te, y_te, sim_tr, c = _setup(df, seed, k, triangle_budget)
    kind = "difference"
    model = pairwise.fit("hgb", kind, X_tr, y_tr, sim_tr, anchors.M, seed)

    f = flow_on(model, c, X_te)
    d = decompose(c, f)
    curl = per_edge_curl(c, d)

    pos = {v: i for i, v in enumerate(c.nodes)}
    truth = np.array([y_te[pos[v]] - y_te[pos[u]] for u, v in c.edges], dtype=float)
    err = np.abs(f - truth)

    # Anchor dispersion, the published baseline: per molecule, the spread of y_a + f(a, molecule)
    # over its anchors; for an edge, the mean of its two endpoints' dispersions.
    idx = anchors.select(tanimoto_matrix(np.vstack([X_te, X_tr]))[: len(X_te), len(X_te):])
    per_mol = (y_tr[idx] + model.corrections(X_tr, X_te, idx)).std(axis=1)
    disp = np.array([0.5 * (per_mol[pos[u]] + per_mol[pos[v]]) for u, v in c.edges], dtype=float)

    rho_curl = float(stats.spearmanr(curl, err).statistic)
    rho_disp = float(stats.spearmanr(disp, err).statistic)
    return {
        "rho_curl": rho_curl,
        "rho_dispersion": rho_disp,
        "rho_difference": rho_curl - rho_disp,
        "n_edges": len(c.edges),
    }

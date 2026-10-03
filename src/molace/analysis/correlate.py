"""The frozen analysis, in the order the plan fixes.

The positive control comes first because kNN regression works precisely when the label is smooth
on the kNN graph, so assortativity must predict what label access buys. An interval covering zero
there indicts the pipeline, not the hypothesis, and nothing downstream is interpreted until it is
resolved.

Resampling is over receptor classes, not rows. The 30 targets are not independent: two are the
same protein measured two ways, 26.7 percent of molecules appear in more than one task, and JAK1
and JAK2 overlap 88.5 percent. n_eff is reported wherever n is.

One behaviour worth knowing before reading an interval: a resample in which the retained clusters
leave either variable constant yields an undefined Spearman, and such draws are dropped rather than
counted. That narrows the interval, and it narrows it most when the association rests on few clusters.
The number of draws actually retained is therefore worth checking alongside the interval when a
result looks surprisingly tight.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class BootResult:
    rho: float
    lo: float
    hi: float
    excludes_zero: bool
    n: int
    n_clusters: int


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    return float(stats.spearmanr(x[ok], y[ok]).statistic)


def cluster_bootstrap_spearman(x, y, clusters, n_resamples: int = 10000,
                               seed: int = 0) -> BootResult:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    cl = np.asarray(clusters)
    uniq = np.unique(cl)
    groups = [np.flatnonzero(cl == c) for c in uniq]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_resamples):
        pick = rng.integers(0, len(groups), size=len(groups))
        idx = np.concatenate([groups[p] for p in pick])
        v = _spearman(x[idx], y[idx])
        if np.isfinite(v):
            draws.append(v)
    d = np.asarray(draws)
    lo, hi = (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))) if d.size else (np.nan, np.nan)
    return BootResult(
        rho=_spearman(x, y), lo=lo, hi=hi,
        excludes_zero=bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0.0 or hi < 0.0)),
        n=int((np.isfinite(x) & np.isfinite(y)).sum()),
        n_clusters=len(uniq),
    )


def effective_n(clusters, rho_intra: float = 0.3) -> float:
    """Block-exchangeable effective sample size: n / (1 + (m_bar - 1) * rho)."""
    cl = pd.Series(list(clusters))
    sizes = cl.value_counts().to_numpy(dtype=float)
    n = float(sizes.sum())
    m_bar = float((sizes ** 2).sum() / n)         # size-weighted mean cluster size
    return n / (1.0 + (m_bar - 1.0) * rho_intra)


def incremental_contribution(frame: pd.DataFrame, target: str, extra_cols,
                             predictor: str = "assortativity",
                             n_resamples: int = 10000, seed: int = 0) -> BootResult:
    """Spearman of the predictor against the target's residual after the extra columns.

    Rank-based throughout: ranks in, OLS residual, Spearman out. This is the quantity the plan's
    decision rule needs — does the predictor say anything the roughness baseline, the mean degree
    and the task size do not already say.
    """
    cols = [target, predictor, *extra_cols]
    f = frame.dropna(subset=cols)
    r = f[cols].rank()
    design = np.column_stack([np.ones(len(f)), *[r[c].to_numpy() for c in extra_cols]])
    beta, *_ = np.linalg.lstsq(design, r[target].to_numpy(), rcond=None)
    resid = r[target].to_numpy() - design @ beta
    return cluster_bootstrap_spearman(
        r[predictor].to_numpy(), resid, f["receptor_class"].to_numpy(),
        n_resamples=n_resamples, seed=seed,
    )


def report(frame: pd.DataFrame, n_resamples: int = 10000, seed: int = 0) -> dict:
    cl = frame["receptor_class"].to_numpy()
    a = frame["assortativity"].to_numpy(dtype=float)
    out = {
        "positive_control": cluster_bootstrap_spearman(
            a, frame["access"].to_numpy(dtype=float), cl, n_resamples, seed),
        "primary": cluster_bootstrap_spearman(
            a, frame["gap"].to_numpy(dtype=float), cl, n_resamples, seed),
        "incremental_over_rogi": incremental_contribution(
            frame, "gap", ["rogi", "mean_degree", "n_molecules"],
            n_resamples=n_resamples, seed=seed),
        "decisive_secondary": cluster_bootstrap_spearman(
            a, frame["correction"].to_numpy(dtype=float), cl, n_resamples, seed),
        "correction_mean": float(np.nanmean(frame["correction"].to_numpy(dtype=float))),
        "meta": {"n": int(len(frame)), "n_eff": round(effective_n(cl), 1),
                 "rho_intra_assumed": 0.3, "n_clusters": int(len(np.unique(cl)))},
    }
    out["supported"] = bool(
        out["primary"].excludes_zero and out["incremental_over_rogi"].excludes_zero
    )
    return out

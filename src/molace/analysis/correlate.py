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
    n_draws: int


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    return float(stats.spearmanr(x[ok], y[ok]).statistic)


def cluster_groups(clusters) -> list[np.ndarray]:
    """Row indices grouped by cluster label, in sorted label order.

    Split out so the three cluster bootstraps in this project share one resampling scheme. Two of
    them once wrote their own, as `frame.cell.isin(set(rng.choice(cells, len(cells), replace=True)))`,
    which silently dropped the multiplicity a bootstrap depends on and turned each draw into a
    without-replacement subsample of about 1 - 1/e of the clusters.
    """
    cl = np.asarray(clusters)
    return [np.flatnonzero(cl == c) for c in np.unique(cl)]


def cluster_resample(groups: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """One bootstrap draw: clusters chosen WITH replacement, their rows concatenated.

    A cluster drawn twice contributes its rows twice. That multiplicity is the whole mechanism; a
    membership test over the drawn set is not a bootstrap.
    """
    pick = rng.integers(0, len(groups), size=len(groups))
    return np.concatenate([groups[p] for p in pick])


def cluster_bootstrap_spearman(x, y, clusters, n_resamples: int = 10000,
                               seed: int = 0) -> BootResult:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    cl = np.asarray(clusters)
    uniq = np.unique(cl)
    groups = cluster_groups(cl)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_resamples):
        idx = cluster_resample(groups, rng)
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
        n_draws=int(d.size),
    )


@dataclass(frozen=True)
class MeanResult:
    mean: float
    lo: float
    hi: float
    excludes_zero: bool
    n: int
    n_clusters: int
    n_draws: int


def cluster_bootstrap_mean(x, clusters, n_resamples: int = 10000, seed: int = 0) -> MeanResult:
    """A cluster bootstrap of a mean, for quantities summarised per target rather than correlated.

    The Hodge criterion needs this: its statistic is the mean per-target difference between two rank
    correlations, and bootstrapping the thirty values as if independent contradicts the same
    non-independence that makes n_eff about ten everywhere else.
    """
    v = np.asarray(x, dtype=float)
    cl = np.asarray(clusters)
    groups = cluster_groups(cl)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_resamples):
        idx = cluster_resample(groups, rng)
        m = float(np.nanmean(v[idx]))
        if np.isfinite(m):
            draws.append(m)
    d = np.asarray(draws)
    lo, hi = (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))) if d.size else (np.nan, np.nan)
    return MeanResult(
        mean=float(np.nanmean(v)), lo=lo, hi=hi,
        excludes_zero=bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0.0 or hi < 0.0)),
        n=int(np.isfinite(v).sum()), n_clusters=len(groups), n_draws=int(d.size),
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

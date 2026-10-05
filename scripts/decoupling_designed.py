"""Increment 7: move neighbourhood overlap on purpose, and test the difference of degradations.

Pre-registration: prereg/increment7_decoupling.yaml, blob 0018a8760d7881322b23dc6dfe6b8465917ab225.

Increment 5's experiment B could not test its claim because the frozen construction grid moved the
decoupling by 4.2 % of its variance against 77 % between targets. Here the construction is designed to
move exactly that, at fixed k:

  nearest  each node joined to its k most similar, the project's standard graph, high overlap
  spread   the same number of edges per node, chosen from the 4k nearest so that each new pick is the
           least similar to those already picked, so neighbourhoods overlap as little as possible

The design gate is evaluated FIRST and the outcome is not read if it fails. That rule exists because
increment 5 nearly banked a result its design had not earned.
"""
from __future__ import annotations

import statistics as st
import sys
import time
import warnings
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

from molace.graphs import knn
from molace.graphs.fingerprints import ecfp4, tanimoto_matrix
from molace.measures.assortativity import target_assortativity
from molace.measures.neighbourhood import shared_neighbour_correlation
from molace.models import anchors

PREREG = "0018a8760d7881322b23dc6dfe6b8465917ab225"
DATA = Path("data/raw/chembl_targets")
K, POOL, SEED, DRAWS = 10, 4, 0, 10000
MIN_CONSTRUCTION_SHARE, MIN_LAMBDA_GAP = 0.25, 0.05


def spread_pick(sim_row: np.ndarray, cand_sim, k: int, pool: int, forbid: int | None) -> np.ndarray:
    """From the pool*k most similar candidates, greedily take k that overlap least with each other."""
    # ties break on ascending index, as in graphs/knn.py and models/anchors.py. Tanimoto on 2048-bit
    # ECFP4 is a ratio of small integers, so exact ties at the pool boundary are common and an
    # unstable sort would make the pool, and so the whole spread graph, a property of the numpy build.
    order = np.lexsort((np.arange(sim_row.shape[0]), -sim_row))
    cand = [c for c in order[: pool * k + 1] if c != forbid][: pool * k]
    if len(cand) <= k:
        return np.asarray(cand, dtype=np.int64)
    cand = np.asarray(cand, dtype=np.int64)
    picked = [0]                                   # the nearest candidate is always taken first
    worst = cand_sim(cand, cand[0]).copy()         # max similarity of each candidate to the picked set
    for _ in range(k - 1):
        worst[picked] = np.inf
        nxt = int(np.argmin(worst))
        picked.append(nxt)
        worst = np.maximum(worst, cand_sim(cand, cand[nxt]))
    return cand[picked]


def spread_graph(sim: np.ndarray, k: int, pool: int) -> nx.Graph:
    n = sim.shape[0]
    g = nx.Graph()
    g.add_nodes_from(range(n))
    cs = lambda cand, j: sim[cand, j]
    for v in range(n):
        for u in spread_pick(sim[v], cs, k, pool, forbid=v):
            if u != v:
                g.add_edge(v, int(u))
    return g


def spread_anchors(sim_tt: np.ndarray, sim_trtr: np.ndarray, k: int, pool: int) -> np.ndarray:
    cs = lambda cand, j: sim_trtr[cand, j]
    return np.vstack([spread_pick(sim_tt[i], cs, k, pool, forbid=None)
                      for i in range(sim_tt.shape[0])])


def cell(g, y, idx, y_tr, y_te, base, name):
    rmse = float(np.sqrt(np.mean((y_te - y_tr[idx].mean(axis=1)) ** 2)))
    r = target_assortativity(g, y).value
    rho = shared_neighbour_correlation(g, y).value
    pred = 1.0 + 1.0 / K + (K - 1.0) / K * rho - 2.0 * r      # the identity, reusing both statistics
    meas = (rmse / base) ** 2
    return {
        "construction": name, "assortativity": r, "rho_shared_neighbour": rho,
        "decoupling": abs(r - rho), "lambda": rho / r if r > 0 else float("nan"),
        "clustering": nx.average_clustering(g),
        "mean_degree": float(np.mean([d for _, d in g.degree()])),
        "predicted_rel_mse": pred, "measured_rel_mse": meas, "error": pred - meas,
    }


def main() -> int:
    meta = pd.read_csv(DATA / "selected.csv")
    print(f"pre-registration blob: {PREREG}")
    print(f"{len(meta)} targets, two constructions at k = {K}, nothing trained")
    print()
    rows = []
    for _, m in meta.iterrows():
        t0 = time.time()
        d = pd.read_csv(DATA / f"{m.dataset}.csv")
        y = d.y.to_numpy(dtype=float)
        sim = tanimoto_matrix(ecfp4(d.smiles.tolist()))
        np.fill_diagonal(sim, -1.0)
        rng = np.random.default_rng(SEED)
        tr = rng.random(len(y)) < 0.8
        y_tr, y_te = y[tr], y[~tr]
        base = float(np.sqrt(np.mean((y_te - y_tr.mean()) ** 2)))
        sim_tt, sim_trtr = sim[np.ix_(~tr, tr)], sim[np.ix_(tr, tr)]
        np.fill_diagonal(sim, 1.0)

        gn = knn.knn_graph(sim, k=K)
        idxn = anchors.select(sim_tt, m=K)          # the project's own selector, ties and all
        np.fill_diagonal(sim, -1.0)
        gs = spread_graph(sim, K, POOL)
        idxs = spread_anchors(sim_tt, sim_trtr, K, POOL)
        np.fill_diagonal(sim, 1.0)

        for g, idx, name in ((gn, idxn, "nearest"), (gs, idxs, "spread")):
            rec = cell(g, y, idx, y_tr, y_te, base, name)
            rec.update({"dataset": m.dataset, "cluster": m.dataset, "n": len(y)})
            rows.append(rec)
        a, b = rows[-2], rows[-1]
        print(f"  {m.dataset:22s} n={len(y):5d}  nearest: C={a['clustering']:.3f} "
              f"lam={a['lambda']:.3f} d={a['decoupling']:.3f} | spread: C={b['clustering']:.3f} "
              f"lam={b['lambda']:.3f} d={b['decoupling']:.3f}  [{time.time()-t0:.0f}s]", flush=True)

    t = pd.DataFrame(rows)
    t.to_csv("results/decoupling_designed.csv", index=False)

    print()
    print("=" * 96)
    print("THE DESIGN GATE, evaluated before the outcome is read")
    print("=" * 96)
    gm = t.decoupling.mean()
    tot = float(((t.decoupling - gm) ** 2).sum())
    share_c = sum(len(g) * (g.decoupling.mean() - gm) ** 2
                  for _, g in t.groupby("construction")) / tot
    share_t = sum(len(g) * (g.decoupling.mean() - gm) ** 2
                  for _, g in t.groupby("dataset")) / tot
    lam = {n: g["lambda"].mean() for n, g in t.groupby("construction")}
    gap = abs(lam["nearest"] - lam["spread"])
    for n, g in t.groupby("construction"):
        print(f"  {n:8s} clustering {g.clustering.mean():.3f}  lambda {g['lambda'].mean():.3f}  "
              f"decoupling {g.decoupling.mean():.4f}  mean degree {g.mean_degree.mean():.1f}")
    print(f"  decoupling variance between constructions: {100*share_c:5.1f} %   "
          f"(increment 5's grid managed 4.2 %)")
    print(f"  decoupling variance between targets:       {100*share_t:5.1f} %")
    print(f"  mean per-hop decay gap:                    {gap:.4f}")
    passed = bool(share_c >= MIN_CONSTRUCTION_SHARE and gap >= MIN_LAMBDA_GAP)
    print(f"  GATE (needs >= {MIN_CONSTRUCTION_SHARE:.0%} and >= {MIN_LAMBDA_GAP}): {passed}")

    print()
    print("=" * 96)
    print("reported unconditionally: does the identity still describe the estimator?")
    print("=" * 96)
    for n, g in t.groupby("construction"):
        print(f"  {n:8s} mean |error| {g.error.abs().mean():.4f}  mean signed {g.error.mean():+.4f}  "
              f"Pearson {g.predicted_rel_mse.corr(g.measured_rel_mse):.3f}")

    if not passed:
        print()
        print("=" * 96)
        print("  THE DESIGN GATE FAILED. The outcome is NOT reported, in numbers or in words: a")
        print("  construction that cannot move the quantity it contrasts has nothing to say about it.")
        print("=" * 96)
        return 0

    print()
    print("=" * 96)
    print("THE OUTCOME: difference of the two degradations")
    print("=" * 96)
    piv = {n: g.set_index("dataset") for n, g in t.groupby("construction")}
    targets = np.array(sorted(set(piv["nearest"].index) & set(piv["spread"].index)))

    def power(sel, col):
        n_, s_ = piv["nearest"].loc[sel], piv["spread"].loc[sel]
        return (stats.spearmanr(s_[col], s_.measured_rel_mse).statistic
                - stats.spearmanr(n_[col], n_.measured_rel_mse).statistic)

    for frame in piv.values():
        frame["one_stat"] = 1 - frame.assortativity
    obs = power(targets, "one_stat") - power(targets, "predicted_rel_mse")
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(DRAWS):
        sel = rng.choice(targets, len(targets), replace=True)
        try:
            v = power(sel, "one_stat") - power(sel, "predicted_rel_mse")
            if np.isfinite(v):
                draws.append(v)
        except Exception:
            pass
    lo, hi = np.percentile(draws, [2.5, 97.5])
    print(f"  assortativity alone, spread minus nearest: {power(targets,'one_stat'):+.4f}")
    print(f"  full identity,       spread minus nearest: {power(targets,'predicted_rel_mse'):+.4f}")
    print(f"  D = difference:                            {obs:+.4f}  [{lo:+.4f}, {hi:+.4f}]  "
          f"{'EXCLUDES 0' if lo > 0 or hi < 0 else 'covers 0'}   ({len(draws)} draws)")
    print()
    if obs < 0 and (lo > 0 or hi < 0):
        print("  THE CLAIM IS SUPPORTED. Assortativity alone loses more ranking power than the full")
        print("  identity does when neighbourhoods are made to overlap less, so its near-sufficiency")
        print("  on this project's graphs is a property of those graphs and not of the statistic.")
    else:
        print(f"  NOT SUPPORTED. The interval has half-width {(hi-lo)/2:.4f} and covers zero, so the")
        print("  data do not separate the two. The project does not read that as no difference.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

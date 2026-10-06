"""Increment 11: MODI as the baseline a JCIM submission requires, and whether the identity reproduces it.

Pre-registration: prereg/increment11_modi.yaml, blob acdba0f77ff9c8fc9149dc9cd8ed39db94122ae1.

MODI (Golbraikh, Muratov, Tropsha, JCIM 2014) is edge homophily on a 1-NN graph, and its continuous
form MODI_q2 is leave-one-out q2 of similarity search. That is this project's kNN floor, evaluated by
LOO instead of on a split. The identity says MODI_q2 is a deterministic function of two graph
statistics, so the question is not which index wins but whether the closed form reproduces the
published one without running the leave-one-out sweep at all.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

from molace.analysis.correlate import incremental_contribution
from molace.data import deepdelta as dd
from molace.data import moleculeace as ma
from molace.graphs import knn
from molace.graphs.fingerprints import ecfp4, tanimoto_matrix
from molace.measures.assortativity import rank_assortativity, target_assortativity
from molace.measures.neighbourhood import shared_neighbour_correlation

PREREG = "acdba0f77ff9c8fc9149dc9cd8ed39db94122ae1"
CHEMBL = Path("data/raw/chembl_targets")
K, DRAWS, SEED, TOL = 10, 10000, 0, 0.05


def collections():
    meta = pd.read_csv(CHEMBL / "selected.csv")
    for _, m in meta.iterrows():
        d = pd.read_csv(CHEMBL / f"{m.dataset}.csv")
        yield "ChEMBL-40", m.dataset, d.smiles.tolist(), d.y.to_numpy(dtype=float), None
    for n in ma.DATASETS:
        d = ma.load_target(n)
        yield ("MoleculeACE-30", n, d["smiles"].tolist(), d["y"].to_numpy(dtype=float),
               d["cliff_mol"].to_numpy(dtype=int))
    for n in dd.DD_DATASETS:
        d = pd.read_csv(dd.ROOT / "Datasets" / "Benchmarks" / f"{n}.csv")
        yield "DeepDelta-10", n, d["SMILES"].tolist(), d["Y"].to_numpy(dtype=float), None


def modi_q2(sim: np.ndarray, y: np.ndarray, k: int) -> float:
    """Leave-one-out q2 of similarity search: Golbraikh's continuous modelability criterion."""
    s = sim.copy()
    np.fill_diagonal(s, -np.inf)                       # never your own neighbour
    # stable, so ties break on ascending index: argpartition gives no guarantee about which of
    # several equal similarities lands inside the k, and 27 % of rows on the largest target have the
    # kth and (k+1)th similarity exactly equal.
    idx = np.argsort(-s, axis=1, kind="stable")[:, :k]
    yhat = y[idx].mean(axis=1)
    return float(1.0 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2))


def modi_binary(sim: np.ndarray, cls: np.ndarray) -> float:
    """The 2014 index: class-weighted 1-NN agreement."""
    s = sim.copy()
    np.fill_diagonal(s, -np.inf)
    nn = np.argmax(s, axis=1)
    vals = []
    for c in np.unique(cls):
        m = cls == c
        if m.sum():
            vals.append(float((cls[nn][m] == c).mean()))
    return float(np.mean(vals))


def attained(coll: str) -> dict[str, float]:
    if coll == "ChEMBL-40":
        t = pd.read_csv("results/separation_chembl.csv")
        return {r.dataset: max(r.skill_pointwise, r.skill_knn_floor, r.skill_pairwise)
                for r in t.itertuples()}
    if coll == "DeepDelta-10":
        t = pd.read_csv("results/separation_deepdelta.csv")
        return {r.dataset: max(r.skill_RandomForest, r.skill_ChemProp50, r.skill_DeepDelta5)
                for r in t.itertuples()}
    t = pd.read_csv("results/spine.csv")
    out = {}
    for r in t.itertuples():
        d = ma.load_target(r.dataset)
        y = d["y"].to_numpy(dtype=float); tr = (d["split"] == "train").to_numpy()
        base = float(np.sqrt(np.mean((y[~tr] - y[tr].mean()) ** 2)))
        out[r.dataset] = max(1 - r.rmse_pointwise / base, 1 - r.rmse_knn_floor / base,
                             1 - r.rmse_pairwise / base)
    return out


def main() -> int:
    print(f"pre-registration blob: {PREREG}\n")
    best = {c: attained(c) for c in ("ChEMBL-40", "MoleculeACE-30", "DeepDelta-10")}
    rows = []
    for coll, name, smiles, y, cls in collections():
        t0 = time.time()
        sim = tanimoto_matrix(ecfp4(smiles))
        g = knn.knn_graph(sim, k=K)
        r = target_assortativity(g, y).value
        rho = shared_neighbour_correlation(g, y).value
        q2 = modi_q2(sim, y, K)
        rows.append({
            "collection": coll, "dataset": name, "cluster": name, "receptor_class": name, "n": len(y),
            "assortativity": r, "rank_assortativity": rank_assortativity(g, y).value,
            "rho_shared_neighbour": rho,
            "identity": 1 + 1 / K + (K - 1) / K * rho - 2 * r,
            "modi_q2": q2, "modi_rel_error": 1.0 - q2,
            "modi_binary": modi_binary(sim, cls) if cls is not None else np.nan,
            "best_skill": best[coll].get(name, np.nan),
        })
        # progress goes to stderr: results/report_modi_baseline.txt is a tracked artefact and an
        # elapsed time makes every reproduction of it show a spurious diff.
        print(f"  {coll:15s} {name:22s} n={len(y):5d}  MODI_q2={q2:+.3f}  "
              f"identity={rows[-1]['identity']:.3f}  1-q2={1-q2:.3f}  [{time.time()-t0:.0f}s]",
              file=sys.stderr, flush=True)
    t = pd.DataFrame(rows)
    t.to_csv("results/modi_baseline.csv", index=False)

    print()
    print("=" * 100)
    print("A. DOES THE IDENTITY REPRODUCE MODI_q2 WITHOUT ANY CROSS-VALIDATION")
    print("=" * 100)
    ok = True
    for coll, s in t.groupby("collection", sort=False):
        d = (s.identity - s.modi_rel_error).abs()
        held = d.mean() < TOL
        ok &= bool(held)
        print(f"  {coll:15s} n={len(s):3d}  mean |difference| {d.mean():.4f}  max {d.max():.4f}  "
              f"Pearson {s.identity.corr(s.modi_rel_error):.4f}  "
              f"{'holds' if held else 'DOES NOT HOLD'}")
    print(f"  frozen threshold {TOL} on every collection: {ok}")

    print()
    print("=" * 100)
    print("B AND C. DOES EITHER INDEX SAY ANYTHING OVER THE OTHER (expected: neither)")
    print("=" * 100)
    for coll, s in t.groupby("collection", sort=False):
        s = s.dropna(subset=["best_skill"]).copy()
        if len(s) < 8:
            print(f"  {coll}: too few targets carry an attained skill"); continue
        b = incremental_contribution(s, "best_skill", ["modi_q2"], predictor="rank_assortativity",
                                     n_resamples=DRAWS, seed=SEED)
        c = incremental_contribution(s, "best_skill", ["rank_assortativity"], predictor="modi_q2",
                                     n_resamples=DRAWS, seed=SEED)
        f = lambda x: "excludes 0" if x.excludes_zero else "covers 0"
        print(f"  {coll:15s} assortativity over MODI {b.rho:+.3f} [{b.lo:+.3f}, {b.hi:+.3f}] {f(b)}"
              f"   |   MODI over assortativity {c.rho:+.3f} [{c.lo:+.3f}, {c.hi:+.3f}] {f(c)}")

    print()
    print("=" * 100)
    print("D. WHAT THE SECOND MOMENT ADDS, WHICH NO PUBLISHED FORM OF MODI CARRIES")
    print("=" * 100)
    for coll, s in t.groupby("collection", sort=False):
        rng = np.random.default_rng(SEED)              # per collection, so a collection's interval
        s = s.dropna(subset=["best_skill"]).copy()     # does not depend on the ones before it
        if len(s) < 8:
            continue
        a = stats.spearmanr(1 - s.assortativity, s.best_skill).statistic
        f = stats.spearmanr(s.identity, s.best_skill).statistic
        draws = []
        for _ in range(DRAWS):
            pick = rng.choice(len(s), len(s), replace=True)
            ss = s.iloc[pick]
            v = (abs(stats.spearmanr(ss.identity, ss.best_skill).statistic)
                 - abs(stats.spearmanr(1 - ss.assortativity, ss.best_skill).statistic))
            if np.isfinite(v):
                draws.append(v)
        lo, hi = np.percentile(draws, [2.5, 97.5])
        print(f"  {coll:15s} assortativity alone {abs(a):.3f}   full identity {abs(f):.3f}   "
              f"difference {abs(f)-abs(a):+.4f} [{lo:+.4f}, {hi:+.4f}]  "
              f"{'excludes 0' if lo > 0 or hi < 0 else 'covers 0'}")

    bm = t.dropna(subset=["modi_binary"])
    if len(bm):
        print()
        print(f"  binary MODI 2014 on the cliff flag, {len(bm)} MoleculeACE targets: "
              f"mean {bm.modi_binary.mean():.3f}, range "
              f"[{bm.modi_binary.min():.3f}, {bm.modi_binary.max():.3f}]; "
              f"the paper's modelability threshold is 0.65, and "
              f"{int((bm.modi_binary > 0.65).sum())} of {len(bm)} clear it")
    return 0


if __name__ == "__main__":
    sys.exit(main())

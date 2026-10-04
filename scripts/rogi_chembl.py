"""The ROGI question on targets that share nothing with the discovery set.

Pre-registration: prereg/increment3_chembl.yaml, blob 7519ef31c4374637483bae167221f9b7567ece7f.

Every target here was selected by activity count from ChEMBL, excluding every ChEMBL id that
appears in MoleculeACE, so none of them is in the set where the claim was found. That is the one
thing increment2_rogi could not get: its pooled set was three quarters discovery set, and its fresh
slice was ten datasets with no power.

The frozen decision rule can express three outcomes, not two. If both incremental directions
exclude zero that is an ASYMMETRY and is reported as one -- the rule in increment2_separation could
not say that, and reported "unresolved" for a case it had no words for.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from molace.analysis.correlate import cluster_bootstrap_spearman, incremental_contribution
from molace.graphs import knn
from molace.graphs.fingerprints import ecfp4, tanimoto_matrix
from molace.measures.assortativity import rank_assortativity, target_assortativity
from molace.measures.rogi import roughness
from molace.models import pointwise

PREREG = "7519ef31c4374637483bae167221f9b7567ece7f"
DATA = Path("data/raw/chembl_targets")
DRAWS, K, SEED = 10000, 10, 0
COVARS = ["rogi", "mean_degree", "n_molecules"]


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def main() -> int:
    meta = pd.read_csv(DATA / "selected.csv")
    print(f"pre-registration blob: {PREREG}")
    print(f"{len(meta)} ChEMBL targets, none sharing an id with MoleculeACE")
    print()

    rows = []
    for _, m in meta.iterrows():
        t0 = time.time()
        d = pd.read_csv(DATA / f"{m.dataset}.csv")
        smiles, y = d.smiles.tolist(), d.y.to_numpy(dtype=float)
        try:
            fp = ecfp4(smiles)
        except ValueError as exc:
            print(f"  SKIP {m.dataset}: {exc}", flush=True)
            continue
        rng = np.random.default_rng(SEED)
        tr = rng.random(len(y)) < 0.8
        if tr.sum() < 50 or (~tr).sum() < 20:
            print(f"  SKIP {m.dataset}: split too small", flush=True)
            continue
        base = rmse(y[~tr], np.full((~tr).sum(), y[tr].mean()))
        # the pre-registered pointwise arm, averaged over its matched families
        preds = [pointwise.fit_predict(name, fp[tr], y[tr], fp[~tr], SEED)
                 for name in pointwise.LEARNERS]
        err = float(np.mean([rmse(y[~tr], p) for p in preds]))
        g = knn.knn_graph(tanimoto_matrix(fp), k=K)
        rows.append({
            "dataset": m.dataset, "cluster": m.dataset, "receptor_class": m.dataset,
            "n_molecules": len(y),
            "assortativity": target_assortativity(g, y).value,
            "rank_assortativity": rank_assortativity(g, y).value,
            "rogi": roughness(fp, y).value,
            "mean_degree": 2.0 * g.number_of_edges() / g.number_of_nodes(),
            "skill": 1.0 - err / base,
        })
        print(f"  {m.dataset:22s} n={len(y):5d} skill={rows[-1]['skill']:+.3f} "
              f"a={rows[-1]['rank_assortativity']:+.3f} rogi={rows[-1]['rogi']:.4f} "
              f"[{time.time()-t0:.0f}s]", flush=True)

    t = pd.DataFrame(rows)
    t.to_csv("results/rogi_chembl.csv", index=False)

    print()
    print("=" * 92)
    print(f"INDEPENDENT TARGETS, n = {len(t)}")
    print("=" * 92)
    out = {}
    for pred, other in (("rank_assortativity", "rogi"), ("rogi", "rank_assortativity")):
        cov = [other] + [c for c in COVARS if c not in (pred, other)]
        b = incremental_contribution(t, "skill", cov, predictor=pred, n_resamples=DRAWS, seed=0)
        flag = "EXCLUDES 0" if b.excludes_zero else "covers 0 "
        print(f"  {pred:20s} incremental  rho={b.rho:+.3f}  [{b.lo:+.3f}, {b.hi:+.3f}]  {flag}")
        out[pred] = b
    for col in ("rank_assortativity", "assortativity", "rogi"):
        b = cluster_bootstrap_spearman(t[col], t.skill, t.cluster, DRAWS, 0)
        flag = "EXCLUDES 0" if b.excludes_zero else "covers 0 "
        print(f"  {col:20s} alone        rho={b.rho:+.3f}  [{b.lo:+.3f}, {b.hi:+.3f}]  {flag}")

    a, r = out["rank_assortativity"], out["rogi"]
    print()
    print("=" * 92)
    print("VERDICT against the frozen rule")
    print("=" * 92)
    if a.excludes_zero != r.excludes_zero:
        who = "assortativity" if a.excludes_zero else "ROGI"
        print(f"  SETTLED: exactly one direction excludes zero -- {who} carries what the other")
        print(f"  does not, on targets independent of the discovery set.")
    elif a.excludes_zero and r.excludes_zero:
        print(f"  ASYMMETRY, not a settlement: both directions exclude zero, assortativity at")
        print(f"  {a.rho:+.3f} and ROGI at {r.rho:+.3f}. Each carries something the other does not,")
        print(f"  and the frozen rule says to report that rather than pick a winner.")
    else:
        print("  NOT ESTABLISHED: neither direction excludes zero. The project stops claiming the")
        print("  increment over the roughness baseline in either direction.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

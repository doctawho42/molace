"""Does target assortativity add anything over the roughness index, and why did two sets disagree?

Pre-registration: prereg/increment2_rogi.yaml, blob 929acfaf7c8be3b5d5baae4257b88f1696bbf4ae.

increment2_separation found assortativity's increment over ROGI covering zero on the confirmatory
set (DeepDelta, n = 10) and excluding zero on the discovery set (MoleculeACE, n = 30), reported the
disagreement, and claimed nothing. Two candidate reasons were frozen before this ran:

  H_power  it is n, and 40 datasets resolve it;
  H_task   it is the TASK -- the discovery set scores an ABSOLUTE regression, the confirmatory set a
           DELTA regression, and a statistic about label smoothness over neighbours may have more to
           say about one than the other.

H_task is tested from a single set of fits: the same models on the same split, scored twice.
"""
from __future__ import annotations

import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from molace.analysis.correlate import cluster_bootstrap_spearman, incremental_contribution
from molace.data import moleculeace as ma
from molace.data.tdc_admet import TDC_REGRESSION, load_tdc
from molace.graphs import knn
from molace.graphs.fingerprints import ecfp4, tanimoto_matrix
from molace.measures.assortativity import rank_assortativity, target_assortativity
from molace.measures.rogi import roughness
from molace.models.gap import evaluate_target

PREREG = "929acfaf7c8be3b5d5baae4257b88f1696bbf4ae"
DRAWS, K = 10000, 10
COVARS = ["rogi", "mean_degree", "n_molecules"]


def covariates(smiles, y):
    fp = ecfp4(smiles)
    g = knn.knn_graph(tanimoto_matrix(fp), k=K)
    return {
        "assortativity": target_assortativity(g, y).value,
        "rank_assortativity": rank_assortativity(g, y).value,
        "rogi": roughness(fp, y).value,
        "mean_degree": 2.0 * g.number_of_edges() / g.number_of_nodes(),
        "n_molecules": len(y),
    }


def both(label, frame, target, clusters):
    """Always both directions, so 'adds nothing' can never be reported for one alone."""
    out = {}
    for pred, other in (("rank_assortativity", "rogi"), ("rogi", "rank_assortativity")):
        cov = [c for c in COVARS if c != pred]
        if other not in cov:
            cov = [other] + [c for c in cov if c != other]
        b = incremental_contribution(frame, target, cov, predictor=pred,
                                     n_resamples=DRAWS, seed=0)
        flag = "EXCLUDES 0" if b.excludes_zero else "covers 0 "
        print(f"    {pred:20s} incremental  rho={b.rho:+.3f}  [{b.lo:+.3f}, {b.hi:+.3f}]  {flag}")
        out[pred] = b
    for col in ("rank_assortativity", "rogi"):
        b = cluster_bootstrap_spearman(frame[col], frame[target], clusters, DRAWS, 0)
        flag = "EXCLUDES 0" if b.excludes_zero else "covers 0 "
        print(f"    {col:20s} alone        rho={b.rho:+.3f}  [{b.lo:+.3f}, {b.hi:+.3f}]  {flag}")
    return out


def main() -> int:
    print(f"pre-registration blob: {PREREG}")
    print()

    rows = []
    s = pd.read_csv("results/spine.csv")
    print("MoleculeACE, 30 targets, reading the committed fits:")
    for n in ma.DATASETS:
        d = ma.load_target(n)
        tr = d["split"].to_numpy() == "train"
        y = d["y"].to_numpy(dtype=float)
        base = float(np.sqrt(np.mean((y[~tr] - y[tr].mean()) ** 2)))
        r = s[s.dataset == n].iloc[0]
        rec = covariates(d["smiles"].tolist(), y)
        rec.update(dataset=f"ACE:{n}", collection="MoleculeACE",
                   cluster=ma.receptor_class(n), baseline=base,
                   skill_absolute=1.0 - r.rmse_pointwise / base,
                   skill_delta=1.0 - r.rmse_pairwise / base)
        rows.append(rec)
    print(f"  {len(rows)} done")

    print("TDC, 10 regression tasks, fitting:")
    for n in TDC_REGRESSION:
        t0 = time.time()
        d = load_tdc(n)
        tr = d["split"].to_numpy() == "train"
        y = d["y"].to_numpy(dtype=float)
        base = float(np.sqrt(np.mean((y[~tr] - y[tr].mean()) ** 2)))
        res = evaluate_target(d, seed=0)
        rec = covariates(d["smiles"].tolist(), y)
        rec.update(dataset=f"TDC:{n}", collection="TDC", cluster=f"TDC:{n}", baseline=base,
                   skill_absolute=1.0 - res.pointwise.rmse_all / base,
                   skill_delta=1.0 - res.pairwise.rmse_all / base)
        rows.append(rec)
        print(f"  {n:28s} abs={rec['skill_absolute']:+.3f} delta={rec['skill_delta']:+.3f} "
              f"[{time.time()-t0:.0f}s]", flush=True)

    t = pd.DataFrame(rows)
    t["receptor_class"] = t.cluster
    t.to_csv("results/rogi_showdown.csv", index=False)

    print()
    print("=" * 94)
    print("H_power -- POOLED, n = 40. NOT confirmatory: it contains the discovery set.")
    print("=" * 94)
    pooled = both("pooled", t, "skill_absolute", t.cluster)

    print()
    print("=" * 94)
    print("FRESH SLICE -- the 10 TDC tasks alone, never used for this question")
    print("=" * 94)
    tdc = t[t.collection == "TDC"].copy()
    both("tdc", tdc, "skill_absolute", tdc.cluster)

    print()
    print("=" * 94)
    print("H_task -- same 10 fits, same split, scored twice: absolute against delta")
    print("=" * 94)
    print("  ABSOLUTE skill (the pointwise arm):")
    both("tdc-abs", tdc, "skill_absolute", tdc.cluster)
    print("  DELTA skill (the pairwise arm, the task the confirmatory set used):")
    both("tdc-delta", tdc, "skill_delta", tdc.cluster)

    print()
    print("=" * 94)
    print("VERDICT against the frozen rule")
    print("=" * 94)
    a, r = pooled["rank_assortativity"], pooled["rogi"]
    resolved = a.excludes_zero != r.excludes_zero
    print(f"  H_power resolved (exactly one direction excludes zero on the pooled set): {resolved}")
    if resolved:
        winner = "assortativity" if a.excludes_zero else "ROGI"
        print(f"    direction: {winner} carries what the other does not")
    print("  H_task: compare the two blocks above. The frozen rule asks whether the sign or the")
    print("  significance of the assortativity increment differs between absolute and delta skill")
    print("  computed from the SAME fits. If it does, the two sets disagreed about the task.")
    print()
    print("  If neither resolves, the previous document's 'not established' stands and nothing is")
    print("  claimed in either direction.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

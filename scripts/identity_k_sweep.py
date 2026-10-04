"""Experiment A of increment 5: does the identity hold as a FUNCTION of k?

Pre-registration: prereg/increment5_identity.yaml, blob c2dcd4ef750f579a89bdfeb5e043b381307abdfe.

Increment 4 checked the identity once, at k = 10, and got 0.0256 mean absolute error. One point is
not a functional form. If a term depending on k is missing from the derivation, sweeping k is what
exposes it, and the 1/k term is the one most exposed: it is the whole reason the floor cannot reach
the Bayes error.

Stated in the frozen plan before running: the identity should UNDER-predict the error, because the
measured floor averages the k nearest TRAINING molecules of a test molecule while both statistics are
computed over the whole graph, and that mismatch should GROW with k.
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

from molace.analysis.correlate import cluster_bootstrap_spearman
from molace.graphs import knn
from molace.graphs.fingerprints import ecfp4, tanimoto_matrix
from molace.measures.assortativity import target_assortativity
from molace.measures.neighbourhood import predicted_floor_error, shared_neighbour_correlation
from molace.models import anchors, knn_floor

PREREG = "c2dcd4ef750f579a89bdfeb5e043b381307abdfe"
DATA = Path("data/raw/chembl_targets")
KS = (3, 5, 10, 20, 40)
SEED, DRAWS = 0, 10000


def main() -> int:
    meta = pd.read_csv(DATA / "selected.csv")
    print(f"pre-registration blob: {PREREG}")
    print(f"k in {KS}, {len(meta)} ChEMBL targets, nothing trained at any k")
    print()

    rows = []
    for _, m in meta.iterrows():
        t0 = time.time()
        d = pd.read_csv(DATA / f"{m.dataset}.csv")
        y = d.y.to_numpy(dtype=float)
        sim = tanimoto_matrix(ecfp4(d.smiles.tolist()))
        rng = np.random.default_rng(SEED)
        tr = rng.random(len(y)) < 0.8
        base = float(np.sqrt(np.mean((y[~tr] - y[tr].mean()) ** 2)))
        sim_tt = sim[np.ix_(~tr, tr)]
        out = []
        for k in KS:
            g = knn.knn_graph(sim, k=k)
            pred = predicted_floor_error(g, y, k)
            idx = anchors.select(sim_tt, m=k)
            rmse = float(np.sqrt(np.mean((y[~tr] - knn_floor.predict(y[tr], idx)) ** 2)))
            meas = (rmse / base) ** 2
            rows.append({
                "dataset": m.dataset, "cluster": m.dataset, "k": k, "n": len(y),
                "assortativity": target_assortativity(g, y).value,
                "rho_shared_neighbour": shared_neighbour_correlation(g, y).value,
                "mean_degree": float(np.mean([dd for _, dd in g.degree()])),
                "predicted_rel_mse": pred, "measured_rel_mse": meas, "error": pred - meas,
                "skill_floor": 1.0 - rmse / base,
            })
            out.append(f"k={k:2d} err={pred-meas:+.4f}")
        print(f"  {m.dataset:22s} n={len(y):5d}  " + "  ".join(out) + f"  [{time.time()-t0:.0f}s]",
              flush=True)

    t = pd.DataFrame(rows)
    t.to_csv("results/identity_k_sweep.csv", index=False)

    print()
    print("=" * 100)
    print("the identity's accuracy at each k")
    print("=" * 100)
    print(f"  {'k':>3s} {'mean |err|':>11s} {'mean err':>10s} {'max |err|':>10s} "
          f"{'Pearson':>9s} {'Spearman':>9s} {'mean deg':>9s} {'rho_nn':>8s} {'r':>8s}")
    per_k = {}
    for k in KS:
        s = t[t.k == k]
        per_k[k] = s
        print(f"  {k:3d} {s.error.abs().mean():11.4f} {s.error.mean():10.4f} "
              f"{s.error.abs().max():10.4f} "
              f"{s.predicted_rel_mse.corr(s.measured_rel_mse):9.3f} "
              f"{s.predicted_rel_mse.corr(s.measured_rel_mse, method='spearman'):9.3f} "
              f"{s.mean_degree.mean():9.1f} {s.rho_shared_neighbour.mean():8.3f} "
              f"{s.assortativity.mean():8.3f}")

    print()
    print("=" * 100)
    print("VERDICT against the frozen rule")
    print("=" * 100)
    worst = max(per_k[k].error.abs().mean() for k in KS)
    form = worst < 0.05
    print(f"  functional form, mean |error| < 0.05 at every k:   {form}   "
          f"(worst {worst:.4f} at k = {max(KS, key=lambda k: per_k[k].error.abs().mean())})")

    all_neg = all(per_k[k].error.mean() < 0 for k in KS)

    # The frozen rule asks for Spearman(k, |MEAN signed error|): the modulus of the mean, one number
    # per k, five points. An earlier version of this script computed Spearman(k, |per-cell error|)
    # instead, which pools 200 cells and is dominated by differences between targets rather than by
    # k. That was an implementation deviation from the plan, not a change of plan; both are printed,
    # and the verdict rests on the pre-registered one.
    means = np.array([per_k[k].error.mean() for k in KS])
    obs = float(stats.spearmanr(KS, np.abs(means)).statistic)
    targets = t.dataset.unique()
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(DRAWS):
        pick = rng.choice(targets, size=len(targets), replace=True)
        s = pd.concat([t[t.dataset == q] for q in pick], ignore_index=True)
        mm = np.abs(np.array([s[s.k == k].error.mean() for k in KS]))
        draws.append(stats.spearmanr(KS, mm).statistic)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    grows = bool((lo > 0 or hi < 0) and obs > 0)
    secondary = cluster_bootstrap_spearman(t.k, t.error.abs(), t.cluster, DRAWS, SEED)
    print(f"  bias is negative at every k:                      {all_neg}")
    print(f"  |mean signed error| grows with k (PRE-REGISTERED): rho={obs:+.3f} "
          f"[{lo:+.3f}, {hi:+.3f}]   {grows}")
    print(f"    per-k values: " + "  ".join(f"k={k}: {abs(m):.4f}" for k, m in zip(KS, means)))
    print(f"  secondary, not the frozen statistic: Spearman(k, |per-cell error|) "
          f"rho={secondary.rho:+.3f} [{secondary.lo:+.3f}, {secondary.hi:+.3f}]")
    print()
    if form and all_neg and grows:
        print("  THE IDENTITY HOLDS ACROSS k, AND THE EXPLANATION OF ITS RESIDUAL HOLDS TOO.")
    elif form:
        print("  THE IDENTITY HOLDS ACROSS k. My explanation of its residual does NOT, and those")
        print("  are separate claims: the derivation stands, the story about the train-only")
        print("  neighbour mismatch is not supported as stated.")
    else:
        print("  THE IDENTITY DOES NOT DESCRIBE THIS ESTIMATOR AT EVERY k. A term depending on k is")
        print("  missing from the derivation, and the project says so with the k and the size.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

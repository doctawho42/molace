"""Is the kNN floor's error PREDICTED, not just correlated, by graph statistics alone?

Not pre-registered. This is theory validation after the fact, and it is labelled that way
everywhere it is reported.

Centring the label and writing sigma^2 = Var(y), the floor's prediction yhat_v = mean of the k
neighbours' labels gives, as plain arithmetic,

    E(y_v - yhat_v)^2 / sigma^2  =  1 + 1/k + (k-1)/k * rho_nn  -  2 * r

where r is the mean label correlation across an edge, which IS target assortativity, and rho_nn is
the mean label correlation between two nodes that share a neighbour. Both read off the graph with
nothing trained. So the identity is a POINT prediction for the floor's relative error, and either it
lands on the measured value or the derivation does not describe this estimator.

Two known reasons it may miss, stated before the numbers:
  * the measured floor averages the k nearest TRAINING molecules for a test molecule, while r and
    rho_nn are computed over the whole graph;
  * union symmetrisation gives degree >= k, so a node's own averaging width is k while its
    contribution to rho_nn comes from a larger neighbourhood.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from molace.graphs import knn
from molace.graphs.fingerprints import ecfp4, tanimoto_matrix
from molace.measures.assortativity import target_assortativity

DATA = Path("data/raw/chembl_targets")
K, SEED, MAX_PAIRS = 10, 0, 4_000_000


def shared_neighbour_correlation(g, y, rng):
    """Pearson correlation of the label over pairs of nodes that share a neighbour."""
    nodes = sorted(g.nodes())
    pos = {v: i for i, v in enumerate(nodes)}
    a, b = [], []
    total = sum(d * (d - 1) // 2 for _, d in g.degree())
    keep = 1.0 if total <= MAX_PAIRS else MAX_PAIRS / total
    for v in nodes:
        nb = [pos[u] for u in g.neighbors(v)]
        for i in range(len(nb)):
            for j in range(i + 1, len(nb)):
                if keep < 1.0 and rng.random() > keep:
                    continue
                a.append(nb[i]); b.append(nb[j])
    ia, ib = np.asarray(a), np.asarray(b)
    first = np.concatenate([y[ia], y[ib]])
    second = np.concatenate([y[ib], y[ia]])
    return float(np.corrcoef(first, second)[0, 1]), len(ia), total


def main() -> int:
    meta = pd.read_csv(DATA / "selected.csv")
    measured = pd.read_csv("results/separation_chembl.csv").set_index("dataset")
    rows = []
    for _, m in meta.iterrows():
        t0 = time.time()
        d = pd.read_csv(DATA / f"{m.dataset}.csv")
        y = d.y.to_numpy(dtype=float)
        g = knn.knn_graph(tanimoto_matrix(ecfp4(d.smiles.tolist())), k=K)
        nodes = sorted(g.nodes())
        yg = y[[i for i in range(len(y))]] if len(nodes) == len(y) else y
        r = target_assortativity(g, yg).value
        rho, used, total = shared_neighbour_correlation(g, yg, np.random.default_rng(SEED))
        deg = float(np.mean([dd for _, dd in g.degree()]))
        skill = float(measured.loc[m.dataset, "skill_knn_floor"])
        rows.append({
            "dataset": m.dataset, "n": len(y), "assortativity": r, "rho_shared_neighbour": rho,
            "mean_degree": deg, "pairs_used": used, "pairs_total": total,
            "predicted_rel_mse_k10": 1 + 1 / K + (K - 1) / K * rho - 2 * r,
            "predicted_rel_mse_kdeg": 1 + 1 / deg + (deg - 1) / deg * rho - 2 * r,
            "measured_rel_mse": (1.0 - skill) ** 2,
            "skill_knn_floor": skill,
        })
        print(f"  {m.dataset:22s} r={r:+.3f} rho_nn={rho:+.3f} deg={deg:5.1f} "
              f"pred={rows[-1]['predicted_rel_mse_k10']:+.3f} "
              f"meas={rows[-1]['measured_rel_mse']:+.3f} [{time.time()-t0:.0f}s]", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv("results/floor_identity.csv", index=False)

    print()
    print("=" * 92)
    print(f"the identity as a POINT prediction of the floor's relative error, n = {len(t)}")
    print("=" * 92)
    for col, lab in (("predicted_rel_mse_k10", "k = 10, the floor's own averaging width"),
                     ("predicted_rel_mse_kdeg", "k = the graph's mean degree")):
        err = t[col] - t.measured_rel_mse
        print(f"  {lab}")
        print(f"    mean signed error {err.mean():+.4f}   mean |error| {err.abs().mean():.4f}   "
              f"max |error| {err.abs().max():.4f}")
        print(f"    Spearman(predicted, measured) = "
              f"{t[col].corr(t.measured_rel_mse, method='spearman'):+.3f}   "
              f"Pearson = {t[col].corr(t.measured_rel_mse):+.3f}")
    print()
    print(f"  for reference, the statistic alone: Spearman(1 - r, measured) = "
          f"{(1 - t.assortativity).corr(t.measured_rel_mse, method='spearman'):+.3f}")
    print(f"  rho_shared_neighbour against assortativity: "
          f"{t.rho_shared_neighbour.corr(t.assortativity, method='spearman'):+.3f} "
          f"(range {t.rho_shared_neighbour.min():+.3f} to {t.rho_shared_neighbour.max():+.3f})")
    print()
    print("  implied crossover r* = 1/k + (k-1)/k * rho_nn, where the floor's excess over Bayes")
    print("  equals a fitted model's, compared with the descriptive split measured at 0.546:")
    star = 1 / K + (K - 1) / K * t.rho_shared_neighbour
    print(f"    median r* = {star.median():.3f}, range [{star.min():.3f}, {star.max():.3f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())

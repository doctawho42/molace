"""Is a quantile-binned label informativeness usable, given that its value is not stable?

The pre-registration forbade binning because the VALUE moves with the bin count. That is true and is
reproduced here. But a measure for comparing datasets does not need a stable value; it needs a
stable ORDERING. Nobody asks what adjusted homophily "is" in absolute terms either -- they ask which
dataset has more of it.

So three questions, in order:
  1. How much does the value move with the bin count?      (expected: a lot)
  2. Does the ORDERING of datasets move with it?           (the question that decides usability)
  3. Does it say anything rank assortativity does not, and does either predict attainable accuracy?

Frozen graph throughout: ECFP4, Tanimoto, kNN k = 10 -- increment 1's construction, so the bin count
is the only thing varying.
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from molace.analysis.correlate import cluster_bootstrap_spearman, incremental_contribution
from molace.data import deepdelta as dd
from molace.data import moleculeace as ma
from molace.graphs import knn
from molace.graphs.fingerprints import ecfp4, tanimoto_matrix
from molace.measures.assortativity import rank_assortativity
from molace.measures.quantile_li import quantile_bins, quantile_label_informativeness

BINS = (2, 3, 4, 6, 8, 12, 16, 24, 32)
K = 10


def datasets():
    for n in ma.DATASETS:
        d = ma.load_target(n)
        yield f"ACE:{n}", d["smiles"].tolist(), d["y"].to_numpy(dtype=float)
    for n in dd.DD_DATASETS:
        d = pd.read_csv(dd.ROOT / "Datasets" / "Benchmarks" / f"{n}.csv")
        yield f"DD:{n}", d["SMILES"].tolist(), d["Y"].to_numpy(dtype=float)


def main() -> int:
    rows = []
    for name, smiles, y in datasets():
        g = knn.knn_graph(tanimoto_matrix(ecfp4(smiles)), k=K)
        rec = {"dataset": name, "n": len(y), "distinct_labels": int(len(np.unique(y))),
               "rank_assortativity": rank_assortativity(g, y).value}
        for b in BINS:
            try:
                rec[f"li_q{b}"] = quantile_label_informativeness(g, y, b).value
                rec[f"achieved_{b}"] = int(len(np.unique(quantile_bins(y, b))))
            except ValueError:
                rec[f"li_q{b}"] = np.nan
                rec[f"achieved_{b}"] = np.nan
        rows.append(rec)
        print(f"  {name:28s} distinct={rec['distinct_labels']:5d}  "
              f"LI_q4={rec['li_q4']:.4f}  LI_q16={rec['li_q16']:.4f}", flush=True)
    t = pd.DataFrame(rows)
    t["receptor_class"] = t.dataset
    t.to_csv("results/quantile_li.csv", index=False)
    cols = [f"li_q{b}" for b in BINS]

    print()
    print("=" * 96)
    print("1. THE VALUE MOVES, AS THE PRE-REGISTRATION SAID")
    print("=" * 96)
    ratio = t[cols].max(axis=1) / t[cols].min(axis=1)
    print(f"  per-dataset max/min across bin counts: median {ratio.median():.1f}x, "
          f"max {ratio.max():.1f}x")
    print(f"  the measure's mean value at each bin count:")
    for b in BINS:
        print(f"    b={b:3d}  mean {t[f'li_q{b}'].mean():.4f}   "
              f"range {t[f'li_q{b}'].min():.4f} to {t[f'li_q{b}'].max():.4f}")

    print()
    print("=" * 96)
    print("2. DOES THE ORDERING MOVE?  Spearman of the dataset ranking, bin count against bin count")
    print("=" * 96)
    m = t[cols].corr(method="spearman")
    print("      " + "".join(f"{b:>8d}" for b in BINS))
    for i, b in enumerate(BINS):
        print(f"  {b:3d} " + "".join(f"{m.iloc[i, j]:8.3f}" for j in range(len(BINS))))
    off = m.to_numpy()[np.triu_indices(len(BINS), k=1)]
    print(f"  worst off-diagonal {off.min():.3f}, median {np.median(off):.3f}")
    adjacent = [m.iloc[i, i + 1] for i in range(len(BINS) - 1)]
    print(f"  adjacent bin counts: worst {min(adjacent):.3f}")

    print()
    print("=" * 96)
    print("3. IS IT SAYING ANYTHING NEW, AND IS IT USEFUL?")
    print("=" * 96)
    print("  against rank assortativity on the same graph:")
    for b in (2, 4, 8, 16, 32):
        rho = t[f"li_q{b}"].corr(t.rank_assortativity, method="spearman")
        print(f"    LI_q{b:<3d} vs rank assortativity   rho={rho:+.3f}")

    # attainable accuracy, where it is already measured
    ace = t[t.dataset.str.startswith("ACE:")].copy()
    ace["dataset_short"] = ace.dataset.str.replace("ACE:", "", regex=False)
    s = pd.read_csv("results/spine.csv")
    base = {}
    for n in ma.DATASETS:
        d = ma.load_target(n)
        tr = d["split"].to_numpy() == "train"
        yy = d["y"].to_numpy(dtype=float)
        base[n] = float(np.sqrt(np.mean((yy[~tr] - yy[tr].mean()) ** 2)))
    s["skill_pointwise"] = 1.0 - s.rmse_pointwise / s.dataset.map(base)
    j = ace.merge(s[["dataset", "skill_pointwise", "rogi", "mean_degree", "assortativity"]],
                  left_on="dataset_short", right_on="dataset", suffixes=("", "_s"))
    j["n_molecules"] = j["n"]
    j["receptor_class"] = [ma.receptor_class(n) for n in j.dataset_short]
    print()
    print("  against attainable accuracy (pointwise skill, 30 MoleculeACE targets):")
    for b in (2, 4, 8, 16):
        bres = cluster_bootstrap_spearman(j[f"li_q{b}"], j.skill_pointwise, j.receptor_class, 10000, 0)
        flag = "EXCLUDES 0" if bres.excludes_zero else "covers 0 "
        print(f"    LI_q{b:<3d} vs skill   rho={bres.rho:+.3f}  [{bres.lo:+.3f}, {bres.hi:+.3f}]  {flag}")
    print("  and incrementally over rank assortativity, ROGI, degree and size:")
    for b in (4, 8, 16):
        bres = incremental_contribution(j, "skill_pointwise",
                                        ["rank_assortativity", "rogi", "mean_degree", "n_molecules"],
                                        predictor=f"li_q{b}", n_resamples=10000, seed=0)
        flag = "EXCLUDES 0" if bres.excludes_zero else "covers 0 "
        print(f"    LI_q{b:<3d} incremental rho={bres.rho:+.3f}  [{bres.lo:+.3f}, {bres.hi:+.3f}]  {flag}")

    print()
    print("Read block 2 as the verdict. A measure whose value moves but whose ordering does not is")
    print("usable for comparing datasets with a declared bin count. A measure whose ordering moves")
    print("too is not a dataset property, and the pre-registration was right to forbid it outright.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""How much of a homophily measure is the data, and how much is the graph the analyst built.

Prokhorenkova's measures are defined for a GIVEN graph. In molecular machine learning there is no
given graph: the analyst picks a fingerprint, a similarity and a neighbourhood rule, and the graph
appears. The same is true of much of GraphLand, where edges are constructed rather than observed.
So a number reported as a dataset's homophily is partly a property of the dataset and partly a
property of the choice, and nobody reports the split.

This measures it. The design is one factor at a time around increment 1's frozen choice
(ECFP4, Tanimoto, kNN at k = 10), so every alternative is a defensible choice an analyst might have
made rather than a strawman. No model is fitted anywhere.

Two quantities come out, and the second is the one that matters:
  * the variance split -- between datasets against between constructions within a dataset;
  * whether the ORDERING of datasets survives a change of construction. A measure whose value moves
    but whose ordering does not is still usable for comparing datasets, which is what it is for.
"""
from __future__ import annotations

import sys
import time
import warnings

import networkx as nx
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from molace.data import deepdelta as dd
from molace.data import moleculeace as ma
from molace.graphs import knn
from molace.graphs.constructions import fingerprint, similarity
from molace.measures.assortativity import rank_assortativity, target_assortativity

FROZEN = ("ecfp4", "tanimoto", 10)

# one factor at a time around the frozen choice
GRID = (
    [("ecfp4", "tanimoto", k) for k in (5, 10, 20, 30)]
    + [(f, "tanimoto", 10) for f in ("ecfp6", "maccs", "rdkit", "atompair")]
    + [("ecfp4", m, 10) for m in ("dice", "cosine")]
)


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
        t0 = time.time()
        fps, sims = {}, {}
        for fp_kind, metric, _ in GRID:
            if fp_kind not in fps:
                fps[fp_kind] = fingerprint(smiles, fp_kind)
            key = (fp_kind, metric)
            if key not in sims:
                sims[key] = similarity(fps[fp_kind], metric)
        for fp_kind, metric, k in GRID:
            g = knn.knn_graph(sims[(fp_kind, metric)], k=k)
            rows.append({
                "dataset": name, "fingerprint": fp_kind, "metric": metric, "k": k,
                "construction": f"{fp_kind}/{metric}/k{k}",
                "frozen": (fp_kind, metric, k) == FROZEN,
                "n_edges": g.number_of_edges(),
                "mean_degree": 2.0 * g.number_of_edges() / g.number_of_nodes(),
                "assortativity": target_assortativity(g, y).value,
                "rank_assortativity": rank_assortativity(g, y).value,
                "avg_clustering": nx.average_clustering(g),
            })
        print(f"  {name:28s} {len(GRID)} constructions  [{time.time()-t0:.0f}s]", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv("results/construction_variance.csv", index=False)

    print()
    print("=" * 96)
    print(f"{len(t.dataset.unique())} datasets x {len(GRID)} constructions, no model fitted")
    print("=" * 96)
    for measure in ("assortativity", "rank_assortativity"):
        piv = t.pivot_table(index="dataset", columns="construction", values=measure)
        per_dataset_mean = piv.mean(axis=1)
        between = float(per_dataset_mean.var(ddof=1))
        within = float(piv.var(axis=1, ddof=1).mean())
        icc = between / (between + within)
        print()
        print(f"--- {measure} ---")
        print(f"  spread BETWEEN datasets   (sd of each dataset's mean): {np.sqrt(between):.4f}")
        print(f"  spread WITHIN a dataset   (mean sd across constructions): {np.sqrt(within):.4f}")
        print(f"  share of variance that is the dataset rather than the construction: {icc:.3f}")
        rng = (piv.max(axis=1) - piv.min(axis=1))
        print(f"  per-dataset range across constructions: median {rng.median():.4f}, "
              f"max {rng.max():.4f} ({rng.idxmax()})")
        print(f"  for comparison, the full range of dataset means: "
              f"{per_dataset_mean.max() - per_dataset_mean.min():.4f}")

        frozen_col = f"{FROZEN[0]}/{FROZEN[1]}/k{FROZEN[2]}"
        print(f"  DOES THE ORDERING SURVIVE? Spearman of the dataset ranking, {frozen_col} against:")
        orders = []
        for col in piv.columns:
            if col == frozen_col:
                continue
            rho = piv[frozen_col].corr(piv[col], method="spearman")
            orders.append(rho)
            print(f"    {col:28s} rho={rho:+.3f}")
        print(f"    worst {min(orders):+.3f}, median {float(np.median(orders)):+.3f}")
    print()
    print("Read the ordering block last and hardest. If a measure's value moves with the")
    print("construction but its ordering of datasets does not, it is still doing the job it exists")
    print("for. If the ordering moves too, a homophily number quoted without its construction is")
    print("not a dataset property at all.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

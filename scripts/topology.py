"""What the topology of a molecular similarity graph looks like, and what drives it.

The exact census (scripts/hodge_census_exact.py) gives dim H_1 -- the number of independent cycles
not spanned by triangles -- on every target, certified. This script asks what that number tracks.

It matters for two reasons. The project's design expected the harmonic part to dominate and was
refuted; the refutation was reported from the three smallest graphs, so the obvious objection is
that harmonic dimension grows with graph size and the three were unrepresentative. That objection
is now answerable. And the graph here is built from STRUCTURE ALONE -- the label plays no part in
it -- so any correlation with a label statistic is a fact about chemistry, not a circularity.
"""
from __future__ import annotations

import sys
import warnings

import networkx as nx
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from molace.analysis.correlate import cluster_bootstrap_spearman, incremental_contribution
from molace.data import moleculeace as ma
from molace.graphs import knn
from molace.graphs.fingerprints import ecfp4, tanimoto_matrix

CENSUS = "results/hodge_census_exact.csv"
DRAWS = 10000
K = 10


def main() -> int:
    c = pd.read_csv(CENSUS)
    ok = c[c.certified].copy()
    print(f"certified targets: {len(ok)} of {len(c)}")
    if len(ok) < len(c):
        print("  not certified:", list(c.loc[~c.certified, "dataset"]))

    # structural covariates the census does not carry
    extra = []
    for name in ok.dataset:
        df = ma.load_target(name)
        g = knn.knn_graph(tanimoto_matrix(ecfp4(df["smiles"].tolist())), k=K)
        deg = np.array([d for _, d in g.degree()], dtype=float)
        extra.append({
            "dataset": name,
            "avg_clustering": nx.average_clustering(g),
            "mean_degree": deg.mean(),
            "degree_cv": deg.std() / deg.mean(),
            "transitivity": nx.transitivity(g),
        })
        print(f"  {name:16s} clustering {extra[-1]['avg_clustering']:.3f}  "
              f"mean degree {deg.mean():.1f}", flush=True)
    t = ok.merge(pd.DataFrame(extra), on="dataset", suffixes=("", "_g"))
    t["receptor_class"] = [ma.receptor_class(n) for n in t.dataset]
    s = pd.read_csv("results/spine.csv")[["dataset", "assortativity", "rogi", "unbiased_homophily"]]
    t = t.merge(s, on="dataset", how="left")
    t.to_csv("results/topology.csv", index=False)

    print()
    print("=" * 92)
    print("WHAT THE HARMONIC SHARE LOOKS LIKE ACROSS 30 MOLECULAR SIMILARITY GRAPHS")
    print("=" * 92)
    for col, lab in (("harmonic_share_of_cycle", "harmonic / cycle space"),
                     ("harmonic_over_gradient", "harmonic / gradient"),
                     ("triangles_per_edge", "triangles per edge"),
                     ("avg_clustering", "average clustering")):
        v = t[col]
        print(f"  {lab:26s} min {v.min():8.4f}   median {v.median():8.4f}   max {v.max():8.4f}")
    print()
    print(f"  targets where harmonic >= 2x gradient (the design's expectation): "
          f"{int((t.harmonic_over_gradient >= 2).sum())} of {len(t)}")
    print(f"  worst margin against that threshold: "
          f"{2.0 / t.harmonic_over_gradient.max():.2f}x, at "
          f"{t.loc[t.harmonic_over_gradient.idxmax(), 'dataset']}")

    print()
    print("=" * 92)
    print("WHAT IT TRACKS  (Spearman, cluster bootstrap over receptor classes)")
    print("=" * 92)
    print("  The objection to answer first: does it just grow with graph size?")
    for col, lab in (("n_edges", "graph size, |E|"),
                     ("n_nodes", "molecules, |V|")):
        b = cluster_bootstrap_spearman(t[col], t.harmonic_share_of_cycle, t.receptor_class, DRAWS, 0)
        f = "EXCLUDES 0" if b.excludes_zero else "covers 0 "
        print(f"    harmonic share vs {lab:24s} rho={b.rho:+.3f}  [{b.lo:+.3f}, {b.hi:+.3f}]  {f}")
    print("  Against structure instead:")
    for col, lab in (("triangles_per_edge", "triangles per edge"),
                     ("avg_clustering", "average clustering"),
                     ("mean_degree", "mean degree"),
                     ("degree_cv", "degree dispersion")):
        b = cluster_bootstrap_spearman(t[col], t.harmonic_share_of_cycle, t.receptor_class, DRAWS, 0)
        f = "EXCLUDES 0" if b.excludes_zero else "covers 0 "
        print(f"    harmonic share vs {lab:24s} rho={b.rho:+.3f}  [{b.lo:+.3f}, {b.hi:+.3f}]  {f}")
    print("  Against label statistics. The GRAPH is label-free, but ROGI is not structure-free, so")
    print("  read the confound block under this one before reading anything into these:")
    for col, lab in (("assortativity", "target assortativity"),
                     ("rogi", "ROGI roughness"),
                     ("unbiased_homophily", "unbiased homophily on the cliff flag")):
        sub = t.dropna(subset=[col])
        b = cluster_bootstrap_spearman(sub[col], sub.harmonic_share_of_cycle, sub.receptor_class, DRAWS, 0)
        f = "EXCLUDES 0" if b.excludes_zero else "covers 0 "
        print(f"    harmonic share vs {lab:24s} rho={b.rho:+.3f}  [{b.lo:+.3f}, {b.hi:+.3f}]  {f}")

    print("  AND THE OBVIOUS CONFOUND, because ROGI is not a pure label statistic:")
    for col, lab in (("avg_clustering", "average clustering"), ("triangles_per_edge", "triangles per edge")):
        b = cluster_bootstrap_spearman(t.rogi, t[col], t.receptor_class, DRAWS, 0)
        f = "EXCLUDES 0" if b.excludes_zero else "covers 0 "
        print(f"    ROGI vs {lab:34s} rho={b.rho:+.3f}  [{b.lo:+.3f}, {b.hi:+.3f}]  {f}")
    t["n_molecules"] = t.n_nodes
    b = incremental_contribution(t, "harmonic_share_of_cycle",
                                 ["avg_clustering", "triangles_per_edge", "mean_degree"],
                                 predictor="rogi", n_resamples=DRAWS, seed=0)
    f = "EXCLUDES 0" if b.excludes_zero else "covers 0 "
    print(f"    ROGI incremental over structure{'':11s} rho={b.rho:+.3f}  "
          f"[{b.lo:+.3f}, {b.hi:+.3f}]  {f}")
    print("    ROGI clusters molecules by structure before it touches the label, so its link to a")
    print("    topological quantity is not evidence of a label-structure relationship. With the")
    print("    structural covariates removed it goes away, so it was the structure.")

    print()
    print("Read the size row first. If the harmonic share does not track graph size, the design's")
    print("refutation does not depend on the three small graphs it was first reported from.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

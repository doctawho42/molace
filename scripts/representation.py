"""Experiment B: is the link a property of the graph, or of the ECFP4 everything shares?

Pre-registration: prereg/increment4_separation_and_representation.yaml, blob
c8ebac86ffd1f0813144121c84927632fef1ab1e.

Every measure and every model in this project reads the same ECFP4 fingerprint. So "target
assortativity predicts attainable accuracy" might be a statement about ECFP4 rather than about the
graph, and the results documents say so in their limits section.

The test keeps the graph and the statistic on ECFP4 Tanimoto, untouched, and refits the MODEL on a
different representation. If a statistic computed on the ECFP4 graph still predicts the skill of a
model that never sees ECFP4, the link is about the label's structure rather than about one
fingerprint.

MACCS and atom-pair are reported too, but they are still substructure bit vectors and share most of
their information with ECFP4. The physicochemical descriptor block is the real test.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from molace.analysis.correlate import cluster_bootstrap_spearman
from molace.graphs import knn
from molace.graphs.constructions import descriptors, fingerprint
from molace.graphs.fingerprints import ecfp4, tanimoto_matrix
from molace.measures.assortativity import rank_assortativity
from molace.models import pointwise

PREREG = "c8ebac86ffd1f0813144121c84927632fef1ab1e"
DATA = Path("data/raw/chembl_targets")
DRAWS, K, SEED = 10000, 10, 0
REPS = ("ecfp4", "rdkit_descriptors", "maccs", "atompair")
PARTIAL = Path("results/representation_partial.csv")

# Same checkpoint as scripts/separation_chembl.py, for the same reason: the first run was killed by a
# two-hour limit at 29 of 40 targets with nothing written. Each target is now saved as it finishes.


def features(kind, smiles, fp_ecfp4):
    if kind == "ecfp4":
        return fp_ecfp4.astype(float)
    if kind == "rdkit_descriptors":
        return descriptors(smiles)
    return fingerprint(smiles, kind).astype(float)


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def main() -> int:
    meta = pd.read_csv(DATA / "selected.csv")
    print(f"pre-registration blob: {PREREG}")
    print("graph and statistic stay on ECFP4 Tanimoto throughout; only the MODEL's features change")
    print()

    rows = pd.read_csv(PARTIAL).to_dict("records") if PARTIAL.exists() else []
    done = {r["dataset"] for r in rows}
    if done:
        print(f"resuming from {PARTIAL}: {len(done)} targets already done")
        print()

    for _, m in meta.iterrows():
        if m.dataset in done:
            continue
        t0 = time.time()
        d = pd.read_csv(DATA / f"{m.dataset}.csv")
        smiles, y = d.smiles.tolist(), d.y.to_numpy(dtype=float)
        rng = np.random.default_rng(SEED)
        tr = rng.random(len(y)) < 0.8
        base = rmse(y[~tr], np.full((~tr).sum(), y[tr].mean()))
        try:
            fp = ecfp4(smiles)
            g = knn.knn_graph(tanimoto_matrix(fp), k=K)
            rec = {"dataset": m.dataset, "cluster": m.dataset, "receptor_class": m.dataset,
                   "n_molecules": len(y),
                   "rank_assortativity": rank_assortativity(g, y).value}
            for kind in REPS:
                X = features(kind, smiles, fp)
                preds = [pointwise.fit_predict(n, X[tr], y[tr], X[~tr], SEED)
                         for n in pointwise.LEARNERS]
                rec[f"skill_{kind}"] = 1.0 - float(np.mean([rmse(y[~tr], p) for p in preds])) / base
        except Exception as exc:
            print(f"  SKIP {m.dataset}: {type(exc).__name__}: {exc}", flush=True)
            continue
        rec["provenance"] = "computed"
        rows.append(rec)
        pd.DataFrame(rows).to_csv(PARTIAL, index=False)
        print(f"  {m.dataset:22s} " + " ".join(f"{k.split('_')[0][:5]}={rec[f'skill_{k}']:+.3f}"
                                               for k in REPS) + f"  [{time.time()-t0:.0f}s]",
              flush=True)

    t = pd.DataFrame(rows)
    t.to_csv("results/representation.csv", index=False)

    print()
    print("=" * 96)
    print(f"assortativity on the ECFP4 GRAPH against the skill of a model on each representation, "
          f"n = {len(t)}")
    print("=" * 96)
    res = {}
    for kind in REPS:
        b = cluster_bootstrap_spearman(t.rank_assortativity, t[f"skill_{kind}"], t.cluster, DRAWS, 0)
        flag = "EXCLUDES 0" if b.excludes_zero else "covers 0 "
        note = "  <- the real test" if kind == "rdkit_descriptors" else (
               "  (same representation as the graph)" if kind == "ecfp4" else "")
        print(f"  {kind:20s} rho={b.rho:+.3f}  [{b.lo:+.3f}, {b.hi:+.3f}]  {flag}{note}")
        res[kind] = b
    print()
    print("  how similar the representations actually are, by the skills they produce:")
    for kind in REPS[1:]:
        r = t[f"skill_{kind}"].corr(t.skill_ecfp4, method="spearman")
        print(f"    skill on {kind:20s} vs skill on ecfp4   rho={r:+.3f}")

    print()
    print("=" * 96)
    print("VERDICT against the frozen rule")
    print("=" * 96)
    d = res["rdkit_descriptors"]
    if d.excludes_zero and d.rho > 0:
        print("  SURVIVES A DIFFERENT REPRESENTATION. A statistic computed on the ECFP4 graph")
        print("  predicts the skill of a model that never sees ECFP4, so the link is not an")
        print("  artefact of sharing one fingerprint. It survives ONE different representation,")
        print("  which is one more than zero, and does not make the claim representation-free.")
    else:
        print("  BOUNDED TO ONE REPRESENTATION. The link does not reach a model built on")
        print("  physicochemical descriptors, so the claim is about prediction in the ECFP4")
        print("  representation and the results documents must say that in the headline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

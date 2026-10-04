"""What target assortativity predicts, and what it does not.

Pre-registration: prereg/increment2_separation.yaml, blob 419d0556420388d3bdfc21bde907941de2a526bc.

H1  assortativity predicts ATTAINABLE ACCURACY (skill over a no-graph baseline)
H2  assortativity does NOT predict WHICH family wins (the difference of two families' skills)

The discovery set is MoleculeACE, where the claim was noticed post hoc; it cannot confirm itself and
is printed first, labelled. The confirmatory set is DeepDelta's published per-fold predictions: three
families that are not mine, splits that are not mine, no refitting, and no family that averages
neighbours -- which is what keeps H1 from being a restatement of the kNN floor's definition.
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
from molace.measures.assortativity import rank_assortativity, target_assortativity
from molace.measures.rogi import roughness

PREREG = "419d0556420388d3bdfc21bde907941de2a526bc"
DRAWS = 10000
K = 10


def band(x, y, clusters, label):
    b = cluster_bootstrap_spearman(x, y, clusters, DRAWS, 0)
    flag = "EXCLUDES 0" if b.excludes_zero else "covers 0 "
    print(f"    {label:46s} rho={b.rho:+.3f}  95% CI [{b.lo:+.3f}, {b.hi:+.3f}]  {flag}")
    return b


def discovery_set() -> None:
    print("=" * 94)
    print("DISCOVERY SET -- MoleculeACE, 30 targets. The claim was noticed here, post hoc.")
    print("This set CANNOT confirm it. Printed so the origin is visible, not as evidence.")
    print("=" * 94)
    s = pd.read_csv("results/spine.csv")
    base = {}
    for name in ma.DATASETS:
        d = ma.load_target(name)
        tr = d["split"].to_numpy() == "train"
        y = d["y"].to_numpy(dtype=float)
        base[name] = float(np.sqrt(np.mean((y[~tr] - y[tr].mean()) ** 2)))
    s["rmse_meanbase"] = s.dataset.map(base)
    for arm in ("pointwise", "knn_floor", "pairwise"):
        s[f"skill_{arm}"] = 1.0 - s[f"rmse_{arm}"] / s.rmse_meanbase
    print("  H1, assortativity against each arm's skill:")
    for arm in ("pointwise", "pairwise"):
        band(s.assortativity, s[f"skill_{arm}"], s.receptor_class, f"skill of the {arm} arm")
    band(s.assortativity, s.skill_knn_floor, s.receptor_class,
         "skill of the kNN floor (POSITIVE CONTROL, near definitional)")
    print("  H1 INCREMENTAL over the roughness baseline, mean degree and task size:")
    for arm in ("pointwise", "pairwise", "knn_floor"):
        b = incremental_contribution(s, f"skill_{arm}", ["rogi", "mean_degree", "n_molecules"],
                                     n_resamples=DRAWS, seed=0)
        flag = "EXCLUDES 0" if b.excludes_zero else "covers 0 "
        print(f"    incremental, skill of {arm:14s}      rho={b.rho:+.3f}  "
              f"95% CI [{b.lo:+.3f}, {b.hi:+.3f}]  {flag}")
    print("  the reverse, does ROGI add anything over assortativity:")
    for arm in ("pointwise", "knn_floor"):
        b = incremental_contribution(s, f"skill_{arm}", ["assortativity", "mean_degree", "n_molecules"],
                                     predictor="rogi", n_resamples=DRAWS, seed=0)
        flag = "EXCLUDES 0" if b.excludes_zero else "covers 0 "
        print(f"    rogi over assortativity, skill of {arm:14s} rho={b.rho:+.3f}  "
              f"95% CI [{b.lo:+.3f}, {b.hi:+.3f}]  {flag}")
    print("  ROGI alone, for comparison:")
    for arm in ("pointwise", "knn_floor"):
        band(s.rogi, s[f"skill_{arm}"], s.receptor_class, f"rogi vs skill of {arm}")
    print("  H2, assortativity against differences of skill:")
    band(s.assortativity, s.skill_pointwise - s.skill_pairwise, s.receptor_class,
         "pointwise minus pairwise")
    band(s.assortativity, s.skill_knn_floor - s.skill_pointwise, s.receptor_class,
         "floor minus pointwise")
    print()


def confirmatory_set() -> pd.DataFrame:
    print("=" * 94)
    print("CONFIRMATORY SET -- DeepDelta, 10 benchmarks, 3 published families, no refitting.")
    print("=" * 94)
    rows = []
    for ds in dd.DD_DATASETS:
        df = pd.read_csv(dd.ROOT / "Datasets" / "Benchmarks" / f"{ds}.csv")
        fp = ecfp4(df["SMILES"].tolist())
        g = knn.knn_graph(tanimoto_matrix(fp), k=K)
        y = df["Y"].to_numpy(dtype=float)
        rec = {
            "dataset": ds, "n_molecules": len(df), "n": len(df),
            "assortativity_pearson": target_assortativity(g, y).value,
            "assortativity_rank": rank_assortativity(g, y).value,
            # the competing baseline the plan requires the claim to beat, plus the two nuisance
            # covariates increment 1 residualised against
            "rogi": roughness(fp, y).value,
            "mean_degree": 2.0 * g.number_of_edges() / g.number_of_nodes(),
        }
        # baseline: predict the mean true delta on the fold, averaged over repeats as the RMSEs are
        for fam in dd.FAMILIES:
            errs, bases = [], []
            for r in dd.REPEATS:
                true, pred = dd.load_predictions(ds, fam, r)
                errs.append(float(np.sqrt(np.mean((true - pred) ** 2))))
                bases.append(float(np.sqrt(np.mean((true - true.mean()) ** 2))))
            rec[f"rmse_{fam}"] = float(np.mean(errs))
            rec[f"base_{fam}"] = float(np.mean(bases))
            rec[f"skill_{fam}"] = 1.0 - float(np.mean(errs)) / float(np.mean(bases))
        rows.append(rec)
    t = pd.DataFrame(rows)
    t["cluster"] = t.dataset
    print(t[["dataset", "n", "assortativity_pearson", "assortativity_rank"]
            + [f"skill_{f}" for f in dd.FAMILIES]].to_string(index=False))
    print()

    results = {}
    for stat in ("assortativity_rank", "assortativity_pearson"):
        tag = "PRIMARY" if stat.endswith("rank") else "secondary"
        print(f"  H1 on {stat} ({tag}):")
        bs = [band(t[stat], t[f"skill_{f}"], t.cluster, f"skill of {f}") for f in dd.FAMILIES]
        results[stat] = bs
        print(f"  H2 on {stat} ({tag}) -- differences of family skills:")
        for i in range(len(dd.FAMILIES)):
            for j in range(i + 1, len(dd.FAMILIES)):
                a, b = dd.FAMILIES[i], dd.FAMILIES[j]
                results.setdefault(f"{stat}_diff", []).append(
                    band(t[stat], t[f"skill_{a}"] - t[f"skill_{b}"], t.cluster, f"{a} minus {b}"))
        print()

    print("  INCREMENT over the roughness baseline, mean degree and task size")
    print("  (rank-residualised; does the graph statistic say anything ROGI does not):")
    t["receptor_class"] = t.dataset
    inc = []
    for stat in ("assortativity_rank",):
        for fam in dd.FAMILIES:
            b = incremental_contribution(t, f"skill_{fam}", ["rogi", "mean_degree", "n_molecules"],
                                         predictor=stat, n_resamples=DRAWS, seed=0)
            flag = "EXCLUDES 0" if b.excludes_zero else "covers 0 "
            print(f"    skill of {fam:16s} incremental  rho={b.rho:+.3f}  "
                  f"95% CI [{b.lo:+.3f}, {b.hi:+.3f}]  {flag}")
            inc.append(b)
    print(f"    ROGI alone against skill, for comparison:")
    for fam in dd.FAMILIES:
        band(t.rogi, t[f"skill_{fam}"], t.cluster, f"rogi vs skill of {fam}")
    print()

    prim = results["assortativity_rank"]
    diffs = results["assortativity_rank_diff"]
    h1 = all(b.excludes_zero and b.rho > 0 for b in prim)
    h2 = not any(b.excludes_zero for b in diffs)
    print("  DECISION, on the primary (rank) statistic and the pre-registered rule:")
    print(f"    H1 supported (all three families positive, zero excluded): {h1}")
    print(f"    H2 'supported' by the frozen rule (no difference excludes zero): {h2}")
    print("    READ THAT ROW AS A NON-REJECTION, NOT AS EVIDENCE. The rule accepts a null from a")
    print("    failure to reject, and at n = 10 this test can barely reject anything: one of the")
    print("    three differences sits at -0.503 inside an interval nearly a unit wide. The effect is")
    print("    bounded only on the discovery set, where n = 30 gives +0.073 [-0.150, +0.457].")
    print(f"    H1 survives the ROGI baseline on all three families:      {all(b.excludes_zero and b.rho > 0 for b in inc)}")
    t.to_csv("results/separation_deepdelta.csv", index=False)
    return t


def main() -> int:
    print(f"pre-registration blob: {PREREG}")
    print("Both H1 and H2 are reported whatever they say; see the decision rule in the file.")
    print()
    discovery_set()
    confirmatory_set()
    print()
    print("READ THIS BEFORE THE NUMBERS ABOVE. The kNN floor's own correlation is a positive")
    print("control, not evidence: the floor IS a neighbour average and assortativity IS label")
    print("smoothness over the neighbour graph, so a link there is close to definitional. The claim")
    print("under test is that it reaches families that average no neighbours, which is what the")
    print("confirmatory set is made of.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Increment 6: does the two-statistic ceiling survive a set it was not found on?

Pre-registration: prereg/increment6_ceiling.yaml, blob be0e9415878b85f76994ea75fc0338488e284dcc.

Model the label correlation as decaying geometrically with hop distance, rho_d = (1 - nu) lambda^d.
Target assortativity is the one-hop value and the shared-neighbour correlation the two-hop value, so

    1 - nu = r^2 / rho_nn        and        skill <= 1 - sqrt(1 - r^2 / rho_nn)

with nothing trained and nothing fitted: one parameter, two equations. On the 40 ChEMBL targets this
is violated 0 times against 29 for the one-statistic version, but those 40 are where it was found.
The 30 MoleculeACE targets have never been read for this question, so they can confirm it.

The three arms are NOT refitted. Their RMSEs come from results/spine.csv, computed on MoleculeACE's
own split, and only the baseline is recomputed here to turn them into skills.
"""
from __future__ import annotations

import math
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from molace.analysis.correlate import cluster_bootstrap_spearman
from molace.data import moleculeace as ma
from molace.graphs import knn
from molace.graphs.fingerprints import ecfp4, tanimoto_matrix
from molace.measures.assortativity import target_assortativity
from molace.measures.neighbourhood import shared_neighbour_correlation

PREREG = "be0e9415878b85f76994ea75fc0338488e284dcc"
K, DRAWS, SEED = 10, 10000, 0
MAX_VIOLATIONS, MAX_SLACK = 1, 0.15
ARMS = ("pointwise", "knn_floor", "pairwise")


def main() -> int:
    spine = pd.read_csv("results/spine.csv").set_index("dataset")
    print(f"pre-registration blob: {PREREG}")
    print(f"{len(spine)} MoleculeACE targets, arms taken from results/spine.csv, not refitted")
    print()

    rows = []
    for name in ma.DATASETS:
        t0 = time.time()
        d = ma.load_target(name)
        y = d["y"].to_numpy(dtype=float)
        tr = (d["split"] == "train").to_numpy()
        base = float(np.sqrt(np.mean((y[~tr] - y[tr].mean()) ** 2)))
        g = knn.knn_graph(tanimoto_matrix(ecfp4(d["smiles"].tolist())), k=K)
        r = target_assortativity(g, y).value
        rho = shared_neighbour_correlation(g, y).value
        row = spine.loc[name]
        skills = {a: 1.0 - float(row[f"rmse_{a}"]) / base for a in ARMS}
        best = max(skills.values())
        signal = r * r / rho if rho > 0 else float("nan")
        admissible = 0.0 < signal <= 1.0
        new_c = 1.0 - math.sqrt(1.0 - signal) if admissible else float("nan")
        old_c = 1.0 - math.sqrt(1.0 - r) if r <= 1.0 else float("nan")
        rows.append({
            "dataset": name, "cluster": str(row["receptor_class"]),
            "receptor_class": str(row["receptor_class"]), "n_molecules": len(y),
            "assortativity": r, "rho_shared_neighbour": rho, "signal_estimate": signal,
            "admissible": admissible, "ceiling_two_stat": new_c, "ceiling_one_stat": old_c,
            "best_skill": best, "base_rmse": base,
            **{f"skill_{a}": skills[a] for a in ARMS},
        })
        flag = "" if (admissible and best <= new_c) else "  <-- VIOLATED" if admissible else "  <-- INADMISSIBLE"
        print(f"  {name:22s} r={r:+.3f} rho_nn={rho:+.3f} 1-nu={signal:.3f} "
              f"ceiling={new_c:.3f} best={best:+.3f}{flag}  [{time.time()-t0:.0f}s]", flush=True)

    t = pd.DataFrame(rows)
    t.to_csv("results/ceiling_holdout.csv", index=False)
    ok = t[t.admissible]

    print()
    print("=" * 96)
    print(f"the ceiling on the held-out set, n = {len(t)}")
    print("=" * 96)
    v_new = int((ok.best_skill > ok.ceiling_two_stat).sum())
    v_old = int((t.best_skill > t.ceiling_one_stat).sum())
    slack = (ok.ceiling_two_stat - ok.best_skill)
    print(f"  inadmissible targets (r^2/rho_nn outside (0,1]):   {int((~t.admissible).sum())}")
    print(f"  two-statistic ceiling violated on:                 {v_new} / {len(ok)}")
    print(f"  one-statistic ceiling violated on:                 {v_old} / {len(t)}")
    print(f"  mean slack, ceiling minus best attained skill:     {slack.mean():+.4f}  "
          f"[{slack.min():+.4f}, {slack.max():+.4f}]")
    print(f"  targets within 0.02 of their ceiling:              "
          f"{int((slack < 0.02).sum())} / {len(ok)}")
    b = cluster_bootstrap_spearman(ok.ceiling_two_stat, ok.best_skill, ok.cluster, DRAWS, SEED)
    print(f"  Spearman(ceiling, best attained skill):            {b.rho:+.3f} "
          f"[{b.lo:+.3f}, {b.hi:+.3f}]  {'EXCLUDES 0' if b.excludes_zero else 'covers 0'}")

    print()
    print("=" * 96)
    print("VERDICT against the frozen rule")
    print("=" * 96)
    valid = v_new <= MAX_VIOLATIONS
    tight = bool(slack.mean() < MAX_SLACK)
    informative = bool(b.excludes_zero and b.rho > 0)
    adds = v_old > v_new
    print(f"  valid, at most {MAX_VIOLATIONS} violation:                          {valid}")
    print(f"  not vacuous, mean slack below {MAX_SLACK}:                 {tight}")
    print(f"  informative, ranks the attained skill:                {informative}")
    print(f"  the second statistic adds something:                  {adds}")
    print()
    if valid and tight and informative and adds:
        print("  THE CEILING SURVIVES A SET IT WAS NOT FOUND ON. Two statistics read off the graph,")
        print("  nothing trained and nothing fitted, bound the attainable accuracy of every arm and")
        print("  rank it. It remains a one-parameter decay MODEL, not a theorem, and the project says")
        print("  so wherever it appears.")
    elif not valid:
        print("  IT DOES NOT HOLD OUT OF SAMPLE. The formula was a within-set fit, and the talk")
        print("  carries it as an idea with a measured failure rather than as a result.")
    elif not adds:
        print("  THE SECOND STATISTIC ADDS NOTHING HERE: the one-statistic ceiling is not violated")
        print("  more often on this set, so the comparison that motivated the formula does not")
        print("  reproduce, and that is reported before anything else.")
    else:
        print("  PARTIAL. It is a valid bound but fails one of the other frozen conditions, named")
        print("  above, and is reported with that condition attached.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

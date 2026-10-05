"""Experiment A: does the statistic predict WHICH model class wins, at n = 40?

Pre-registration: prereg/increment4_separation_and_representation.yaml, blob
c8ebac86ffd1f0813144121c84927632fef1ab1e.

The project's headline is a separation: target assortativity predicts the LEVEL of attainable
accuracy and not WHICH model class wins. The first half is confirmed on three collections. The
second half has never been established: on DeepDelta's ten benchmarks the test could not reject
anything, and increment2_separation downgraded it from "supported" to a non-rejection after review.

The 40 ChEMBL targets share no id with the discovery set, and they are four times the n. The frozen
rule here asks for more than "the interval covers zero": it asks for an interval narrow enough to
BOUND the effect, because a wide interval around zero is the absence of evidence, not evidence of
absence. That distinction is what the earlier rule got wrong.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from molace.analysis.checkpoint import load_checkpoint, save_checkpoint
from molace.analysis.correlate import cluster_bootstrap_spearman
from molace.graphs import knn
from molace.graphs.fingerprints import ecfp4, tanimoto_matrix
from molace.measures.assortativity import rank_assortativity, target_assortativity
from molace.models.gap import evaluate_target

PREREG = "c8ebac86ffd1f0813144121c84927632fef1ab1e"
DATA = Path("data/raw/chembl_targets")
DRAWS, K, SEED = 10000, 10, 0
MAX_HALF_WIDTH = 0.50          # frozen: a difference interval wider than this bounds nothing
PARTIAL = Path("results/separation_chembl_partial.csv")

# Why there is a checkpoint at all: the first run of this script wrote its table only at the end and
# was killed by a two-hour limit at 28 of 40 targets. Each target now lands in PARTIAL the moment it
# finishes, and a restart skips what is already there. Nothing in the frozen analysis changes; this is
# bookkeeping. The 28 recovered targets carry provenance "recovered_3dp" because their skills were
# read back from the printed log at three decimals while their statistics were joined at full
# precision; results_increment4 reports the rounding check that shows the verdict does not depend on
# those digits.


def main() -> int:
    meta = pd.read_csv(DATA / "selected.csv")
    print(f"pre-registration blob: {PREREG}")
    print(f"{len(meta)} ChEMBL targets, none sharing an id with the discovery set")
    print()

    rows = load_checkpoint(PARTIAL, ["dataset", "skill_pointwise", "skill_knn_floor",
                                     "skill_pairwise"])
    done = {r["dataset"] for r in rows}
    if done:
        print(f"resuming from {PARTIAL}: {len(done)} targets already done")
        print()

    for _, m in meta.iterrows():
        if m.dataset in done:
            continue
        t0 = time.time()
        d = pd.read_csv(DATA / f"{m.dataset}.csv")
        y = d.y.to_numpy(dtype=float)
        rng = np.random.default_rng(SEED)           # identical split to scripts/rogi_chembl.py
        tr = rng.random(len(y)) < 0.8
        d = d.assign(split=np.where(tr, "train", "test"), cliff_mol=0)
        try:
            fp = ecfp4(d.smiles.tolist())
            res = evaluate_target(d, seed=SEED)
        except Exception as exc:
            print(f"  SKIP {m.dataset}: {type(exc).__name__}: {exc}", flush=True)
            continue
        base = float(np.sqrt(np.mean((y[~tr] - y[tr].mean()) ** 2)))
        g = knn.knn_graph(tanimoto_matrix(fp), k=K)
        rec = {
            "dataset": m.dataset, "cluster": m.dataset, "receptor_class": m.dataset,
            "n_molecules": len(y),
            "assortativity": target_assortativity(g, y).value,
            "rank_assortativity": rank_assortativity(g, y).value,
            "skill_pointwise": 1.0 - res.pointwise.rmse_all / base,
            "skill_knn_floor": 1.0 - res.knn_floor.rmse_all / base,
            "skill_pairwise": 1.0 - res.pairwise.rmse_all / base,
            "provenance": "computed",
        }
        rows.append(rec)
        save_checkpoint(PARTIAL, rows)
        print(f"  {m.dataset:22s} n={len(y):5d} pt={rec['skill_pointwise']:+.3f} "
              f"floor={rec['skill_knn_floor']:+.3f} pair={rec['skill_pairwise']:+.3f} "
              f"[{time.time()-t0:.0f}s]", flush=True)

    t = pd.DataFrame(rows)
    t.to_csv("results/separation_chembl.csv", index=False)

    def band(label, y):
        b = cluster_bootstrap_spearman(t.rank_assortativity, y, t.cluster, DRAWS, 0)
        half = (b.hi - b.lo) / 2
        flag = "EXCLUDES 0" if b.excludes_zero else "covers 0 "
        print(f"  {label:44s} rho={b.rho:+.3f}  [{b.lo:+.3f}, {b.hi:+.3f}]  {flag}  "
              f"half-width {half:.3f}")
        return b, half

    print()
    print("=" * 96)
    print(f"THE CONFIRMED HALF: each arm's own skill, n = {len(t)}")
    print("=" * 96)
    levels = {a: band(f"skill of the {a} arm", t[f"skill_{a}"])
              for a in ("pointwise", "knn_floor", "pairwise")}

    print()
    print("=" * 96)
    print("THE HALF THAT WAS NEVER ESTABLISHED: differences between arms")
    print("=" * 96)
    diffs = {
        "pointwise minus pairwise": band("pointwise minus pairwise",
                                         t.skill_pointwise - t.skill_pairwise),
        "floor minus pointwise": band("floor minus pointwise",
                                      t.skill_knn_floor - t.skill_pointwise),
        "floor minus pairwise": band("floor minus pairwise",
                                     t.skill_knn_floor - t.skill_pairwise),
    }

    print()
    print("=" * 96)
    print("VERDICT against the frozen rule")
    print("=" * 96)
    all_levels = all(b.excludes_zero and b.rho > 0 for b, _ in levels.values())
    any_diff_fires = any(b.excludes_zero for b, _ in diffs.values())
    all_narrow = all(h < MAX_HALF_WIDTH for _, h in diffs.values())
    print(f"  every arm's own skill tracks the statistic:        {all_levels}")
    print(f"  no difference excludes zero:                       {not any_diff_fires}")
    print(f"  every difference interval is narrower than {MAX_HALF_WIDTH}:   {all_narrow}")
    print()
    if any_diff_fires:
        print("  THE SEPARATION IS WRONG AS STATED. A difference excludes zero, so the statistic")
        print("  does carry something about model choice. The project says so.")
    elif all_levels and all_narrow:
        print("  SEPARATION ESTABLISHED: every level tracks the statistic, every difference is")
        print("  bounded near zero. The claim is now complete on both halves.")
    else:
        print("  STILL NOT ESTABLISHED. The intervals cover zero but do not bound the effect, so")
        print("  this remains absence of evidence. The project keeps saying not established.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

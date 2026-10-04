"""Post-hoc sensitivity analysis of the spent TDC holdout.

WHAT THIS IS NOT. The holdout was opened once, under `prereg/increment1.yaml`, and its
pre-registered form is invalid: the plan failed to require a dimensionless gap, and it failed to
require a statistic robust to a skewed label. A holdout cannot be re-run as a confirmatory test
after its result is known, and nothing here is reported as one.

WHAT IT IS. One diagnostic question with a checkable answer: was the reported +0.817 a property
of units rather than of graph structure? That is answered by changing only the label's scale and
re-measuring, which is why three scales are run rather than one.

The log rule takes no threshold: a label is logged when its domain allows it, which on these nine
endpoints is exactly the five that are in raw units. The rank-normal scale applies one rule to all
nine and leaves every task with unit variance, so it needs no admissibility branch at all.
"""
from __future__ import annotations

import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from molace.analysis.correlate import cluster_bootstrap_spearman
from molace.analysis.prereg import load_prereg, prereg_fingerprint
from molace.analysis.transforms import log10_if_positive, rank_to_normal
from molace.data.tdc_admet import TDC_REGRESSION, assert_spine_recorded, load_tdc
from molace.graphs import knn
from molace.graphs.fingerprints import ecfp4, tanimoto_matrix
from molace.measures.assortativity import target_assortativity
from molace.models.gap import evaluate_target

SCALES = ("raw", "log_if_admissible", "rank_normal")


def rescale(y: np.ndarray, scale: str) -> tuple[np.ndarray, bool]:
    if scale == "raw":
        return y, True
    if scale == "log_if_admissible":
        return log10_if_positive(y)
    if scale == "rank_normal":
        return rank_to_normal(y), True
    raise KeyError(scale)


def main() -> int:
    assert_spine_recorded()
    k = load_prereg()["graphs"]["primary"]["k"]
    rows = []
    for name in TDC_REGRESSION:
        df = load_tdc(name)
        y_raw = df["y"].to_numpy(dtype=float)
        # the graph is built from structure alone, so it is identical across scales
        g = knn.knn_graph(tanimoto_matrix(ecfp4(df["smiles"].tolist())), k=k)
        for scale in SCALES:
            t0 = time.time()
            y, applied = rescale(y_raw, scale)
            d = df.assign(y=y)
            try:
                r = evaluate_target(d, seed=0)
            except Exception as exc:  # recorded, not swallowed
                print(f"  FAILED {name} {scale}: {type(exc).__name__}: {exc}", flush=True)
                continue
            sd = float(np.std(y, ddof=1))
            rows.append({
                "dataset": name, "scale": scale, "transform_applied": applied,
                "n": len(df), "label_sd": sd,
                "assortativity": target_assortativity(g, y).value,
                "gap": r.gap, "access": r.access, "correction": r.correction,
                "gap_sd": r.gap / sd, "access_sd": r.access / sd,
                "correction_sd": r.correction / sd,
            })
            print(f"  {name:26s} {scale:18s} applied={str(applied):5s} "
                  f"a={rows[-1]['assortativity']:+.3f} gap_sd={rows[-1]['gap_sd']:+.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv("results/holdout_sensitivity.csv", index=False)

    print()
    print("prereg of the spent holdout:", prereg_fingerprint())
    print()
    print("THIS IS A POST-HOC SENSITIVITY ANALYSIS, NOT A HOLDOUT RESULT.")
    print("The holdout was opened once and is spent. What follows changes only the label's scale")
    print("and asks whether the reported +0.817 survives. It cannot restore a confirmatory test.")
    print()
    for scale in SCALES:
        s = t[t.scale == scale]
        if s.empty:
            print(f"{scale}: no rows"); continue
        n_applied = int(s.transform_applied.sum())
        print(f"--- {scale}  (transform applied to {n_applied} of {len(s)} tasks) ---")
        for col in ("gap_sd", "correction_sd", "access_sd"):
            b = cluster_bootstrap_spearman(s.assortativity, s[col], s.dataset, 10000, 0)
            print(f"  assortativity vs {col:14s} rho={b.rho:+.3f}  95% CI [{b.lo:+.3f}, {b.hi:+.3f}]"
                  f"  excludes 0: {b.excludes_zero}")
        b = cluster_bootstrap_spearman(s.assortativity, s.label_sd, s.dataset, 10000, 0)
        print(f"  CONFOUND CHECK, assortativity vs label_sd: rho={b.rho:+.3f} "
              f"CI [{b.lo:+.3f}, {b.hi:+.3f}]")
        print()
    print("Read the confound check before the correlation above it. The pre-registered holdout's")
    print("defect was that the statistic and the gap both tracked the label's scale; a scale on")
    print("which that correlation is gone is a scale on which the main number means what it says.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

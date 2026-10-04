"""Re-specify the failed positive control as a level and measure it. Post-hoc diagnostic.

The pre-registered control asked target assortativity to predict `access`, the error difference
between the pointwise arm and the kNN floor. It read +0.043 with an interval covering zero. The
reasoning behind it -- nearest-neighbour regression works when the label is smooth on the
nearest-neighbour graph -- is sound about the floor's own accuracy and says nothing about a
difference between two arms that both inherit the same smoothness.

This script measures what the control should have been: each arm's **skill** over the no-graph
baseline, which is predicting the training mean on the test split. It refits nothing; it reads
`results/spine.csv` and the labels. Run it to reproduce the numbers quoted in
`results/results_increment1.md`.
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from molace.analysis.correlate import cluster_bootstrap_spearman
from molace.data import moleculeace as ma

# 10,000, matching the draw count the frozen plan uses for every Spearman in the spine, so the
# pre-registered `access` row below reproduces results/report_spine.txt exactly rather than
# landing a few thousandths away from it on a different draw count.
DRAWS = 10000


def main() -> int:
    s = pd.read_csv("results/spine.csv")
    base = {}
    for name in ma.DATASETS:
        d = ma.load_target(name)
        tr = d["split"].to_numpy() == "train"
        y = d["y"].to_numpy(dtype=float)
        base[name] = float(np.sqrt(np.mean((y[~tr] - y[tr].mean()) ** 2)))
    s["rmse_meanbase"] = s.dataset.map(base)
    s["skill_floor"] = 1.0 - s.rmse_knn_floor / s.rmse_meanbase
    s["skill_pointwise"] = 1.0 - s.rmse_pointwise / s.rmse_meanbase

    def show(label: str, y: pd.Series) -> None:
        b = cluster_bootstrap_spearman(s.assortativity, y, s.receptor_class, DRAWS, 0)
        print(f"  {label:48s} rho={b.rho:+.3f}  95% CI [{b.lo:+.3f}, {b.hi:+.3f}]  "
              f"excludes 0: {b.excludes_zero}")

    print("POST-HOC DIAGNOSTIC, not a pre-registered result. Computed after the primary was known,")
    print("to explain a failed control rather than to support a claim.")
    print()
    print("as pre-registered -- a difference of two arms:")
    show("assortativity vs access", s.access)
    show("assortativity vs raw error, floor", s.rmse_knn_floor)
    show("assortativity vs raw error, pointwise", s.rmse_pointwise)
    print()
    print("as it should have been -- a level, skill over the training-mean baseline:")
    show("assortativity vs floor skill", s.skill_floor)
    show("assortativity vs pointwise skill", s.skill_pointwise)
    show("assortativity vs difference of the two skills", s.skill_floor - s.skill_pointwise)
    print()
    print(f"floor skill across targets: {s.skill_floor.min():.3f} to {s.skill_floor.max():.3f}, "
          f"median {s.skill_floor.median():.3f}")
    print()
    print("Read it as: the statistic predicts how accurately a target can be predicted at all, and")
    print("almost nothing about which arm wins, because both arms inherit the same smoothness.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

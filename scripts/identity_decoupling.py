"""Experiment B of increment 5: the prediction this project put in its own talk without testing it.

Pre-registration: prereg/increment5_identity.yaml, blob c2dcd4ef750f579a89bdfeb5e043b381307abdfe.

The identity weights shared-neighbour agreement at (k-1)/k, about +0.9, against assortativity's -2.
On the 40 ChEMBL targets at k = 10 the two statistics correlate at +0.988, so assortativity alone
nearly matches the full identity for RANKING. The talk turns that into a prediction: assortativity
alone must stop working where the two decouple. That prediction was made from one construction, and
a prediction made on the data it was found in is not a test.

This runs the 10 frozen constructions of scripts/construction_variance.py, which move clustering by
changing the fingerprint, the metric and k one factor at a time, and asks whether assortativity alone
degrades where the decoupling is largest while the full identity does not.

A limit of reusing a frozen grid rather than designing one: k varies inside the grid, so k is
confounded with the construction. Reported either way, as the plan requires.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

from molace.graphs import knn
from molace.graphs.constructions import fingerprint, similarity
from molace.measures.assortativity import target_assortativity
from molace.measures.neighbourhood import predicted_floor_error, shared_neighbour_correlation
from molace.models import anchors, knn_floor

PREREG = "c2dcd4ef750f579a89bdfeb5e043b381307abdfe"
DATA = Path("data/raw/chembl_targets")
GRID = (
    [("ecfp4", "tanimoto", k) for k in (5, 10, 20, 30)]
    + [(f, "tanimoto", 10) for f in ("ecfp6", "maccs", "rdkit", "atompair")]
    + [("ecfp4", m, 10) for m in ("dice", "cosine")]
)
SEED, DRAWS = 0, 10000


def main() -> int:
    meta = pd.read_csv(DATA / "selected.csv")
    print(f"pre-registration blob: {PREREG}")
    print(f"{len(meta)} targets x {len(GRID)} frozen constructions, nothing trained")
    print()

    rows = []
    for _, m in meta.iterrows():
        t0 = time.time()
        d = pd.read_csv(DATA / f"{m.dataset}.csv")
        smiles, y = d.smiles.tolist(), d.y.to_numpy(dtype=float)
        rng = np.random.default_rng(SEED)
        tr = rng.random(len(y)) < 0.8
        base = float(np.sqrt(np.mean((y[~tr] - y[tr].mean()) ** 2)))
        # One similarity matrix at a time. Caching all seven would hold 3.8 GB at once on the
        # largest target, so each (fingerprint, metric) is built, used for every k the grid asks of
        # it, and dropped.
        by_sim: dict[tuple[str, str], list[int]] = {}
        for fp_kind, metric, k in GRID:
            by_sim.setdefault((fp_kind, metric), []).append(k)
        for (fp_kind, metric), ks in by_sim.items():
            sim = similarity(fingerprint(smiles, fp_kind), metric)
            sub = sim[np.ix_(~tr, tr)]
            for k in ks:
                g = knn.knn_graph(sim, k=k)
                r = target_assortativity(g, y).value
                rho = shared_neighbour_correlation(g, y).value
                pred = predicted_floor_error(g, y, k)
                idx = anchors.select(sub, m=k)
                rmse = float(np.sqrt(np.mean((y[~tr] - knn_floor.predict(y[tr], idx)) ** 2)))
                meas = (rmse / base) ** 2
                rows.append({
                    "dataset": m.dataset, "cluster": m.dataset,
                    "construction": f"{fp_kind}/{metric}/k{k}", "fingerprint": fp_kind,
                    "metric": metric, "k": k, "n": len(y),
                    "assortativity": r, "rho_shared_neighbour": rho, "decoupling": abs(r - rho),
                    "mean_degree": float(np.mean([dd for _, dd in g.degree()])),
                    "predicted_rel_mse": pred, "measured_rel_mse": meas, "error": pred - meas,
                })
            del sim, sub
        print(f"  {m.dataset:22s} n={len(y):5d} {len(GRID)} cells "
              f"decoupling {min(x['decoupling'] for x in rows[-len(GRID):]):.3f}"
              f"-{max(x['decoupling'] for x in rows[-len(GRID):]):.3f} "
              f"[{time.time()-t0:.0f}s]", flush=True)

    t = pd.DataFrame(rows)
    t.to_csv("results/identity_decoupling.csv", index=False)

    lo_c, hi_c = t.decoupling.quantile([1 / 3, 2 / 3])
    t["tercile"] = np.where(t.decoupling <= lo_c, "low",
                            np.where(t.decoupling >= hi_c, "high", "mid"))

    print()
    print("=" * 100)
    print(f"{len(t)} cells, decoupling |r - rho_nn| split at {lo_c:.4f} and {hi_c:.4f}")
    print("=" * 100)
    print(f"  {'tercile':>8s} {'cells':>6s} {'decoupling':>11s} {'identity |err|':>15s} "
          f"{'1-r alone, rho':>15s} {'identity rho':>13s}")
    sp = {}
    for name in ("low", "mid", "high"):
        s = t[t.tercile == name]
        alone = stats.spearmanr(1 - s.assortativity, s.measured_rel_mse).statistic
        full = stats.spearmanr(s.predicted_rel_mse, s.measured_rel_mse).statistic
        sp[name] = (alone, full, s.error.abs().mean())
        print(f"  {name:>8s} {len(s):6d} {s.decoupling.mean():11.4f} "
              f"{s.error.abs().mean():15.4f} {alone:15.3f} {full:13.3f}")

    # paired bootstrap over targets, tercile membership fixed on the full pooled data
    targets = t.dataset.unique()
    rng = np.random.default_rng(SEED)
    diffs_alone, diffs_full = [], []
    for _ in range(DRAWS):
        pick = rng.choice(targets, size=len(targets), replace=True)
        s = pd.concat([t[t.dataset == p] for p in pick], ignore_index=True)
        hi, lo = s[s.tercile == "high"], s[s.tercile == "low"]
        if len(hi) < 8 or len(lo) < 8:
            continue
        diffs_alone.append(stats.spearmanr(1 - hi.assortativity, hi.measured_rel_mse).statistic
                           - stats.spearmanr(1 - lo.assortativity, lo.measured_rel_mse).statistic)
        diffs_full.append(stats.spearmanr(hi.predicted_rel_mse, hi.measured_rel_mse).statistic
                          - stats.spearmanr(lo.predicted_rel_mse, lo.measured_rel_mse).statistic)
    a_lo, a_hi = np.percentile(diffs_alone, [2.5, 97.5])
    f_lo, f_hi = np.percentile(diffs_full, [2.5, 97.5])
    obs_alone = sp["high"][0] - sp["low"][0]
    obs_full = sp["high"][1] - sp["low"][1]

    print()
    print("=" * 100)
    print(f"high-minus-low tercile difference in ranking power, paired bootstrap over targets, "
          f"{len(diffs_alone)} usable draws")
    print("=" * 100)
    print(f"  assortativity alone   {obs_alone:+.3f}  [{a_lo:+.3f}, {a_hi:+.3f}]  "
          f"{'EXCLUDES 0' if a_lo > 0 or a_hi < 0 else 'covers 0'}")
    print(f"  the full identity     {obs_full:+.3f}  [{f_lo:+.3f}, {f_hi:+.3f}]  "
          f"{'EXCLUDES 0' if f_lo > 0 or f_hi < 0 else 'covers 0'}")

    print()
    print("=" * 100)
    print("VERDICT against the frozen rule")
    print("=" * 100)
    identity_stable = (sp["high"][2] - sp["low"][2]) <= 0.02
    # ranking power is a correlation of (1-r) with ERROR, so degrading means moving toward zero
    alone_degrades = abs(sp["high"][0]) < abs(sp["low"][0]) and (a_lo > 0 or a_hi < 0)
    print(f"  identity's |error| in the top tercile is no more than 0.02 worse: {identity_stable} "
          f"({sp['high'][2]:.4f} against {sp['low'][2]:.4f})")
    print(f"  assortativity alone ranks worse where the two decouple, interval excludes zero: "
          f"{alone_degrades}")
    print()
    if identity_stable and alone_degrades:
        print("  THE PREDICTION HOLDS. Assortativity alone loses ranking power exactly where")
        print("  shared-neighbour agreement decouples from edge agreement, and the full identity")
        print("  does not. The talk's claim is now tested rather than asserted.")
    elif not identity_stable:
        print("  THE IDENTITY ITSELF DEGRADES with the construction, which is a larger finding than")
        print("  the prediction it was meant to test, and it leads the report.")
    else:
        print("  THE PREDICTION IS NOT SUPPORTED. Assortativity alone does not measurably lose")
        print("  ranking power where the statistics decouple. The talk asserted this without a test")
        print("  and is corrected to say the test was run and found nothing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

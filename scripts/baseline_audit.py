"""Increment 14: the baseline comparison claim 4 requires and the project never ran.

Pre-registration: prereg/increment14_baseline_audit.yaml, blob
179b8bf128302f9c266dd7af9e2815c9049f18f2 (its instrument gate amended before any measurement; the
superseded version, blob 65a408a3..., is recorded inside the file).

Claim 4 measures the kNN floor against mean(hgb, mlp), which excludes the SVM that pointwise.py's own
docstring calls the real bar, while pointwise.select_by_cv exists and is called by no script. The
averaging is correct for the pointwise-minus-pairwise gap, where matched families make the difference a
comparison of architectures, and wrong as the model arm for a floor-versus-best-model comparison. This
script supplies the arm the comparison requires and reports what it does to the sign.

A tie is a positive result here and the plan says so in advance: Janela and Bajorath, Nature Machine
Intelligence 2022, 4:1246-1255, report simple nearest-neighbour analysis meeting or exceeding
state-of-the-art machine learning on this task.
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
from molace.analysis.correlate import cluster_bootstrap_mean
from molace.data import moleculeace as ma
from molace.graphs import knn
from molace.graphs.fingerprints import ecfp4, tanimoto_matrix
from molace.measures.assortativity import rank_assortativity, target_assortativity
from molace.models import anchors, knn_floor, pointwise

PREREG = "179b8bf128302f9c266dd7af9e2815c9049f18f2"
CHEMBL = Path("data/raw/chembl_targets")
PARTIAL = Path("results/baseline_audit_partial.csv")
OUT = Path("results/baseline_audit.csv")
COMMITTED = Path("results/separation_chembl.csv")

K, M, DRAWS, SEED, FOLDS = 10, 10, 10000, 0, 5
MATCHED = ("hgb", "mlp")          # the arm claim 4 currently uses, reported unchanged
TOL_COMPUTED, TOL_RECOVERED = 1e-12, 5e-4      # gate 1, split by the artefact's own precision

REQUIRED = ["collection", "dataset", "n_molecules", "skill_floor", "skill_cv", "selected"]


def collections():
    meta = pd.read_csv(CHEMBL / "selected.csv")
    for m in meta.itertuples():
        d = pd.read_csv(CHEMBL / f"{m.dataset}.csv")
        yield "ChEMBL-40", m.dataset, d.smiles.tolist(), d.y.to_numpy(dtype=float)
    for n in ma.DATASETS:
        d = ma.load_target(n)
        yield "MoleculeACE-30", n, d["smiles"].tolist(), d["y"].to_numpy(dtype=float)


def one_target(coll: str, name: str, smiles: list[str], y: np.ndarray) -> dict:
    rng = np.random.default_rng(SEED)
    tr = rng.random(len(y)) < 0.8                 # identical to scripts/separation_chembl.py:68
    tr_i, te_i = np.flatnonzero(tr), np.flatnonzero(~tr)
    fp = ecfp4(smiles)
    sim = tanimoto_matrix(fp)
    base = float(np.sqrt(np.mean((y[~tr] - y[tr].mean()) ** 2)))
    rmse = lambda p: float(np.sqrt(np.mean((y[te_i] - p) ** 2)))
    skill = lambda p: 1.0 - rmse(p) / base

    idx = anchors.select(sim[np.ix_(te_i, tr_i)], M)
    row = {
        "collection": coll, "dataset": name, "cluster": name, "receptor_class": name,
        "prereg": PREREG, "n_molecules": len(y), "n_train": int(tr.sum()),
        "skill_floor": skill(knn_floor.predict(y[tr_i], idx)),
    }
    g = knn.knn_graph(sim, k=K)
    row["assortativity"] = float(target_assortativity(g, y).value)
    row["rank_assortativity"] = float(rank_assortativity(g, y).value)

    # every learner, fitted on the training split and scored on the test split
    per = {}
    for learner in pointwise.LEARNERS:
        per[learner] = skill(pointwise.fit_predict(learner, fp[tr_i], y[tr_i], fp[te_i], SEED))
        row[f"skill_{learner}"] = per[learner]

    # the arm claim 4 uses: mean of the hgb and mlp RMSEs, not of their skills
    row["skill_matched"] = 1.0 - float(np.mean([base * (1.0 - per[f]) for f in MATCHED])) / base

    # the arm the comparison requires: selection on the training split alone
    winner, scores = pointwise.select_by_cv(fp[tr_i], y[tr_i], SEED, FOLDS)
    row["selected"] = winner
    for learner, s in scores.items():
        row[f"cv_rmse_{learner}"] = float(s)
    # fit_predict is deterministic in (learner, data, seed), so the winner's test skill is the
    # per-learner value already computed; refitting would return the identical prediction.
    row["skill_cv"] = per[winner]

    row["floor_minus_cv"] = row["skill_floor"] - row["skill_cv"]
    row["floor_minus_matched"] = row["skill_floor"] - row["skill_matched"]
    row["cv_minus_matched"] = row["skill_cv"] - row["skill_matched"]
    return row


def gate_1(t: pd.DataFrame) -> bool:
    """The floor is unchanged, so it must reproduce the committed values at the stored precision."""
    if not COMMITTED.is_file():
        print("  gate 1 CANNOT BE EVALUATED: results/separation_chembl.csv is absent")
        return False
    c = pd.read_csv(COMMITTED).set_index("dataset")
    s = t[t.collection == "ChEMBL-40"].set_index("dataset")
    common = s.index.intersection(c.index)
    ok = True
    for prov, tol in (("computed", TOL_COMPUTED), ("recovered_3dp", TOL_RECOVERED)):
        sub = [d for d in common if c.loc[d, "provenance"] == prov]
        if not sub:
            continue
        err = (s.loc[sub, "skill_floor"] - c.loc[sub, "skill_knn_floor"]).abs()
        good = bool(err.max() <= tol)
        ok &= good
        print(f"  gate 1, provenance {prov:14s} n={len(sub):2d}  max |drift| {err.max():.3e}  "
              f"tolerance {tol:.0e}   {'PASS' if good else 'FAIL'}")
        if not good:
            for d in err.sort_values(ascending=False).head(3).index:
                print(f"      {d}: measured {s.loc[d,'skill_floor']:.6f} vs committed "
                      f"{c.loc[d,'skill_knn_floor']:.6f}")
    return ok


def main() -> int:
    print(f"pre-registration blob: {PREREG}")
    print(f"arms: floor | matched mean{MATCHED} | CV-selected of {pointwise.LEARNERS} | each learner")
    print(f"split rng({SEED}) 80/20, m={M}, {FOLDS}-fold CV on the training split, {DRAWS} draws\n")

    rows = load_checkpoint(PARTIAL, REQUIRED)
    done = {(r["collection"], r["dataset"]) for r in rows}
    if done:
        print(f"resuming from {PARTIAL}: {len(done)} targets already done\n")

    for coll, name, smiles, y in collections():
        if (coll, name) in done:
            continue
        t0 = time.time()
        try:
            rec = one_target(coll, name, smiles, y)
        except Exception as exc:
            print(f"  SKIP {coll} {name}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            continue
        rows.append(rec)
        save_checkpoint(PARTIAL, rows)
        print(f"  {coll:15s} {name:22s} n={rec['n_molecules']:5d} floor={rec['skill_floor']:+.3f} "
              f"cv={rec['skill_cv']:+.3f}({rec['selected']:3s}) matched={rec['skill_matched']:+.3f} "
              f"f-cv={rec['floor_minus_cv']:+.3f}  [{time.time()-t0:.0f}s]",
              file=sys.stderr, flush=True)

    t = pd.DataFrame(rows)
    t.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(t)} targets)\n")

    print("=" * 100)
    print("GATE 1, THE INSTRUMENT: the unchanged floor must reproduce the committed values")
    print("=" * 100)
    passed = gate_1(t)
    print()
    if not passed:
        print("  GATE 1 FAILED. The outcome is not reported: this script is not measuring what the")
        print("  committed claim measured, and the discrepancy above is the finding.")
        return 0

    for coll in ("ChEMBL-40", "MoleculeACE-30"):
        s = t[t.collection == coll]
        if len(s) < 8:
            continue
        print("=" * 100)
        print(coll)
        print("=" * 100)
        vc = s.selected.value_counts()
        print(f"  learner chosen by CV on the training split: "
              + ", ".join(f"{k} {v}/{len(s)}" for k, v in vc.items()))
        print(f"  mean skill per learner: "
              + "  ".join(f"{l} {s[f'skill_{l}'].mean():+.4f}" for l in pointwise.LEARNERS))
        print()
        for label, col in (("floor - CV-selected  (PRIMARY)", "floor_minus_cv"),
                           ("floor - matched arm  (as claim 4 has it)", "floor_minus_matched"),
                           ("CV-selected - matched arm", "cv_minus_matched")):
            b = cluster_bootstrap_mean(s[col].to_numpy(), s.receptor_class.to_numpy(), DRAWS, SEED)
            print(f"  {label:42s} {b.mean:+.4f} [{b.lo:+.4f}, {b.hi:+.4f}] "
                  f"{'EXCLUDES' if b.excludes_zero else 'covers  '} 0   "
                  f"floor ahead on {int((s[col] > 0).sum())}/{len(s)}")
        print()
        if coll == "ChEMBL-40":
            b = cluster_bootstrap_mean(s.floor_minus_cv.to_numpy(), s.receptor_class.to_numpy(),
                                       DRAWS, SEED)
            print("  OUTCOME, by the frozen rule:")
            if b.excludes_zero and b.mean > 0:
                print("    CLAIM 4 SURVIVES. The floor beats the CV-selected learner.")
            elif b.excludes_zero and b.mean < 0:
                print("    CLAIM 4 REVERSES. The floor does NOT beat the best fitted model, and the")
                print("    earlier claim rested on an arm that excluded the strongest learner.")
            else:
                print("    THE TWO TIE. Reported as an independent replication of Janela and Bajorath,")
                print("    Nature Machine Intelligence 2022, 4:1246-1255, on 40 independently curated")
                print(f"    targets, with the interval's half-width {(b.hi - b.lo) / 2:.4f} stating what")
                print("    is excluded. This is NOT a null.")
            print()
            n_svm = int((s.selected == "svm").sum())
            print(f"  PREDICTION 1, svm chosen on at least 21 of 40: {n_svm} of {len(s)}  "
                  f"{'HOLDS' if n_svm >= 21 else 'FAILS'}")
            if n_svm < 21:
                print("    The docstring's basis is a MoleculeACE results matrix and does not transfer")
                print("    to independently curated ChEMBL targets. This leads the report.")
            fm = s.floor_minus_matched.mean()
            fc = s.floor_minus_cv.mean()
            print(f"  PREDICTION 2, the floor's advantage shrinks: matched {fm:+.4f} -> CV {fc:+.4f}  "
                  f"{'HOLDS' if fc < fm else 'FAILS'}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

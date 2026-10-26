"""Increment 20: claim 4 under a reweighted floor, with selection symmetric on both sides.

Pre-registration: prereg/increment20_claim4_reweighted.yaml, blob
4e60cc1e95f3d3777311ba47317dc2c84933f847.

NO MODEL IS FITTED. Increments 14 and 16 use the identical split rule, default_rng(0).random(n) < 0.8,
and their floor columns agree to 3.331e-16, so this pairs two committed tables. Section 4a records the
debt in its own words: the audit fixed the model arm and left the floor at m = 10, and a symmetric
pre-registered version is owed.

THREE THINGS IN HERE EXIST BECAUSE THE PLAN'S FIRST DRAFT WAS WRONG ABOUT THEM.

  * GATE 1 NAMES ITS MERGE KEYS, BOTH COLUMNS AND ITS SUBSETS. The first draft quoted one residual for
    a comparison it never specified, and the wrong constant appeared twice. Re-derived here: 3.331e-16
    over all 70 shared targets, 1.110e-16 on ChEMBL-40.
  * PREDICTION 3 IS A TWO-TERM DECOMPOSITION. The first draft compared the kernel's gain over FIXED-m
    uniform against increment 16's committed +0.0155, which was measured over BEST-m uniform -- two
    different baselines. Here the m-term and the weighting term are separated and each gets an
    interval.
  * GATE 2 CAN FAIL. The first draft declared it unable to, which is the defect that has cost this
    project three fatal findings in three increments. It now pins both inputs by git blob hash, pins
    three marginal means to 1e-4, and requires the kernel arm's alpha to be non-zero -- that last
    because an all-zero alpha would silently make this whole comparison a restatement of increment 14.

Both scales are reported. RMSE_arm = base * (1 - skill), so the absolute difference follows from the
committed skills and the per-target baseline, which is read from committed labels under the
deterministic split -- not a recomputed arm. Increment 18 established the paper must carry both scales.
"""
from __future__ import annotations

import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from molace.analysis.correlate import cluster_bootstrap_mean
from molace.data import moleculeace as ma

PREREG = "4e60cc1e95f3d3777311ba47317dc2c84933f847"
AUDIT = Path("results/baseline_audit.csv")
WEIGHT = Path("results/weighting_family.csv")
CHEMBL = Path("data/raw/chembl_targets")
OUT = Path("results/claim4_reweighted.csv")

BLOBS = {AUDIT: "e7a666011e710de2cbf444a585d7e1658cb086ec",
         WEIGHT: "1b136f001844eef17dc57639661e40b0e7a05521"}
LEARNERS_REGISTERED = ("svm", "hgb", "mlp")
M_GRID = (3, 5, 10, 20, 40)
ALPHA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
MARGINALS = {"skill_kernel": 0.3839, "skill_cv": 0.4232, "skill_floor": 0.3478}  # ChEMBL-40
TOL_G1, TOL_MARGINAL = 1e-12, 1e-4
INC14_GAP = -0.0754          # committed mean(skill_floor - skill_cv) on ChEMBL-40
P3_MAX_SPREAD = 0.01
P4_MAX_AHEAD = 8
DRAWS, SEED = 10000, 0
PRIMARY, SECONDARY = "ChEMBL-40", "MoleculeACE-30"


def blob(p: Path) -> str:
    # encoding is explicit because text=True otherwise decodes by the LOCALE, which
    # tests/test_text_io_is_utf8.py flags by name. A blob hash is ASCII hex so nothing here would
    # have broken, but the convention exists to catch the pattern rather than the live failure.
    return subprocess.run(["git", "hash-object", str(p)], capture_output=True,
                          text=True, encoding="utf-8", check=True).stdout.strip()


def labels_of(coll: str, dataset: str) -> np.ndarray:
    """Committed labels. Reading these and re-deriving a deterministic split is not fitting an arm."""
    if coll == PRIMARY:
        return pd.read_csv(CHEMBL / f"{dataset}.csv").y.to_numpy(dtype=float)
    return ma.load_target(dataset)["y"].to_numpy(dtype=float)


def base_rmse(y: np.ndarray) -> float:
    """The baseline every skill in this project is measured against: the training mean on the test."""
    tr = np.random.default_rng(SEED).random(len(y)) < 0.8
    return float(np.sqrt(np.mean((y[~tr] - y[tr].mean()) ** 2)))


def boot(v: pd.Series, clusters: pd.Series):
    r = cluster_bootstrap_mean(np.asarray(v, float), np.asarray(clusters), DRAWS, SEED)
    return r.mean, r.lo, r.hi


def fmt(v, lo, hi, n_pos=None, n=None) -> str:
    s = f"{v:+.4f} [{lo:+.4f}, {hi:+.4f}]"
    return s + (f"  ahead on {n_pos}/{n}" if n_pos is not None else "")


def main() -> int:
    print(f"pre-registration blob: {PREREG}\n")
    a = pd.read_csv(AUDIT, float_precision="round_trip")
    w = pd.read_csv(WEIGHT, float_precision="round_trip")
    j = a.merge(w, on=["collection", "dataset"], suffixes=("_14", "_16"))
    line = "=" * 100

    # ---- GATE 1 ---------------------------------------------------------------------------------
    print(line); print("GATE 1. THE TWO COMMITTED TABLES SIT ON ONE SPLIT"); print(line)
    d = (j.skill_floor - j.g1_skill_mean_m10).abs()
    print(f"  merged on (collection, dataset): {len(j)} shared targets")
    print(f"  baseline_audit.skill_floor against weighting_family.g1_skill_mean_m10:")
    print(f"    all {len(j)} targets   max |difference| = {d.max():.3e}")
    for c in (PRIMARY, SECONDARY):
        s = d[j.collection == c]
        print(f"    {c:16s} ({len(s):2d})  max |difference| = {s.max():.3e}")
    nm = int((j.n_molecules_14 != j.n_molecules_16).sum())
    nt = int((j.n_train_14 != j.n_train_16).sum())
    print(f"  n_molecules disagreements {nm} of {len(j)}; n_train disagreements {nt} of {len(j)}")
    g1 = len(j) == 70 and float(d.max()) <= TOL_G1 and nm == 0 and nt == 0
    print(f"  GATE 1: {'PASS' if g1 else 'FAIL'} at tolerance {TOL_G1:.0e}")
    if not g1:
        print("\n  The plan stops the increment here and the discrepancy IS the report.")
        return 1

    # ---- GATE 2 ---------------------------------------------------------------------------------
    print(); print(line); print("GATE 2. THE INPUTS ARE THE DECLARED ONES, AND THIS GATE CAN FAIL"); print(line)
    ok = True
    for p, want in BLOBS.items():
        got = blob(p)
        hit = got == want
        ok &= hit
        print(f"  {str(p):34s} blob {'matches' if hit else 'DIFFERS'}  {got[:12]}")
    bad_learner = sorted(set(a.selected.dropna()) - set(LEARNERS_REGISTERED))
    print(f"  increment 14 selected learners outside its registered set: {bad_learner or 'none'}")
    ok &= not bad_learner
    bad_m = int((~w.kernel_m.isin(M_GRID)).sum())
    bad_al = int((~w.kernel_alpha.isin(ALPHA_GRID)).sum())
    zero_al = int((w.kernel_alpha == 0.0).sum())
    print(f"  kernel_m outside its frozen grid: {bad_m}; kernel_alpha outside: {bad_al}; "
          f"alpha == 0 on {zero_al} targets")
    ok &= bad_m == 0 and bad_al == 0 and zero_al == 0
    p = j[j.collection == PRIMARY]
    for col, want in MARGINALS.items():
        got = float(p[col].mean())
        hit = abs(got - want) <= TOL_MARGINAL
        ok &= hit
        print(f"  {PRIMARY} mean {col:13s} = {got:+.4f} against committed {want:+.4f}  "
              f"{'ok' if hit else 'DIFFERS'}")
    print(f"  GATE 2: {'PASS' if ok else 'FAIL'}")
    print("  An alpha of zero on every target would make the kernel arm the uniform arm and this")
    print("  whole comparison a restatement of increment 14, which is why that clause is here.")
    if not ok:
        print("\n  The plan stops the increment here and the discrepancy IS the report.")
        return 1

    # ---- the absolute scale, from committed labels under the deterministic split ----------------
    j["base_rmse"] = [base_rmse(labels_of(r.collection, r.dataset)) for r in j.itertuples()]
    for arm in ("skill_kernel", "skill_cv", "skill_floor", "skill_uniform", "g1_skill_mean_m10"):
        j[f"rmse_{arm}"] = j.base_rmse * (1.0 - j[arm])
    j.to_csv(OUT, index=False)
    print(f"\nwrote {OUT} ({len(j)} targets)", file=sys.stderr)

    # ---- the differences ------------------------------------------------------------------------
    DIFFS = [("kernel - cv", "skill_kernel", "skill_cv"),
             ("kernel - floor", "skill_kernel", "skill_floor"),
             ("floor - cv", "skill_floor", "skill_cv"),
             ("m-term: uniform - floor", "skill_uniform", "g1_skill_mean_m10"),
             ("weighting: kernel - uniform", "skill_kernel", "skill_uniform")]

    for coll in (PRIMARY, SECONDARY):
        s = j[j.collection == coll]
        print(); print(line); print(f"{'A' if coll == PRIMARY else 'B'}. {coll}, "
                                   f"{len(s)} targets, BOTH SCALES"); print(line)
        print(f"  {'difference':30s} {'skill':>34s} {'absolute RMSE':>30s}")
        for name, hi, lo in DIFFS:
            dv = s[hi] - s[lo]
            m, l, u = boot(dv, s.cluster_14)
            av = s[f"rmse_{lo}"] - s[f"rmse_{hi}"]       # positive means `hi` has the smaller error
            am, al, au = boot(av, s.cluster_14)
            print(f"  {name:30s} {fmt(m, l, u):>34s} {fmt(am, al, au):>30s}"
                  f"   {int((dv > 0).sum())}/{len(s)}")

        if coll != PRIMARY:
            continue

        # ---- the four predictions, on the primary collection ----------------------------------
        print(); print(line); print("C. THE FOUR PREDICTIONS"); print(line)
        dk = s.skill_kernel - s.skill_cv
        mk, lk, uk = boot(dk, s.cluster_14)
        p1 = mk > INC14_GAP
        print(f"  1  mean(kernel - cv) = {mk:+.4f} against increment 14's committed {INC14_GAP:+.4f}")
        print(f"     PREDICTION 1: {'HOLDS' if p1 else 'FAILS'}   "
              f"DERIVABLE IN ADVANCE, no evidential weight")
        if not p1:
            print("\n     Gate 2 pinned the marginal means, so a failure means the pairing is misread.")
            print("     The plan stops the increment and reports that, not an outcome.")
            return 1

        p2 = uk < 0.0
        print(f"\n  2  the interval is [{lk:+.4f}, {uk:+.4f}]")
        print(f"     PREDICTION 2: {'HOLDS' if p2 else 'FAILS'}   THE PIVOT, not derivable from means")
        print(f"     per target: " + " ".join(f"{v:+.3f}" for v in dk))
        print(f"     the reweighted floor is ahead on {int((dk > 0).sum())}/{len(s)} targets")

        mt = s.skill_uniform - s.g1_skill_mean_m10
        wt = s.skill_kernel - s.skill_uniform
        m_m, m_l, m_u = boot(mt, s.cluster_14)
        w_m, w_l, w_u = boot(wt, s.cluster_14)
        tot = s.skill_kernel - s.g1_skill_mean_m10
        resid = float((mt + wt - tot).abs().max())
        p3 = abs(m_m - w_m) < P3_MAX_SPREAD
        print(f"\n  3  choosing m   {fmt(m_m, m_l, m_u)}")
        print(f"     weighting    {fmt(w_m, w_l, w_u)}")
        print(f"     total        {tot.mean():+.4f}   decomposition residual {resid:.1e}")
        print(f"     PREDICTION 3: {'HOLDS' if p3 else 'FAILS'} (|difference| {abs(m_m-w_m):.4f} "
              f"< {P3_MAX_SPREAD})   DERIVABLE IN ADVANCE, no evidential weight")

        ahead = int((dk > 0).sum())
        p4 = ahead < P4_MAX_AHEAD
        print(f"\n  4  the reweighted floor is ahead on {ahead} of {len(s)}, needs < {P4_MAX_AHEAD}")
        print(f"     PREDICTION 4: {'HOLDS' if p4 else 'FAILS'}")
        print(f"     increment 14 reported the uniform floor ahead on 3 of 40; section 4a's")
        print(f"     unregistered sensitivity on 1 of 31")

        # ---- the outcome ---------------------------------------------------------------------
        print(); print(line); print("D. THE OUTCOME RULE"); print(line)
        print(f"  the paired interval is [{lk:+.4f}, {uk:+.4f}] around {mk:+.4f}")
        if uk < 0.0:
            print("\n  CLAIM 4 SURVIVES, WITH ITS MAGNITUDE REVISED. The gap narrows but does not")
            print(f"  close: from increment 14's {INC14_GAP:+.4f} to {mk:+.4f}, of which choosing m")
            print(f"  recovers {m_m:+.4f} and weighting {w_m:+.4f}. The paper drops any implication")
            print("  that the whole gap is information outside the neighbourhood, and section 4a's")
            print("  unregistered sensitivity is replaced by this registered measurement.")
        elif lk > 0.0:
            print("\n  THE FLOOR WINS, reversing increment 14's reversal. Reported with maximum")
            print("  suspicion of itself, per the plan: the per-target values above, the selected")
            print("  (m, alpha) distribution below, and the search-cost asymmetry, which points")
            print("  AGAINST the floor arm and so cannot explain this.")
        else:
            print("\n  CLAIM 4 CLOSES. The gap does not survive giving the floor its own")
            print("  one-parameter weighting, so its effect was substantially a weighting artefact.")
            print("  Claim 4 is withdrawn in the form increment 14's own reversal took.")

        print(); print(line); print("E. WHAT THE FLOOR ARM SELECTED"); print(line)
        print(f"  m:     {s.kernel_m.value_counts().sort_index().to_dict()}")
        print(f"  alpha: {s.kernel_alpha.value_counts().sort_index().to_dict()}")
        print(f"  increment 16's committed floor-arm selection regret on this collection: "
              f"{s.kernel_regret.mean():.4f}")
        print("  Selection is symmetric in FORM -- both arms choose on the training split alone --")
        print("  but not in SEARCH COST: 40 cross-validation cells against 3. That residual")
        print("  asymmetry points AGAINST the floor arm, the direction that keeps claim 4 alive.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

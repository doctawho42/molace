"""Increment 19: SARI as the third baseline index.

Pre-registration: prereg/increment19_sari.yaml, blob b62cb36b3eb92b6b960040e54c1fd1bb78aaa194.

The raw scores are the originating group's own restatement (Balfer & Bajorath, PLoS One 2015, eqs
1-2), which the `rogi` package implements exactly. The COMBINATION is the published mean,
0.5*(Phi(z_cont) + (1 - Phi(z_disc))), from three independent restatements (Guha 2011 eq. 2, Guha 2012
eq. 3, SPARFlow 2026 eq. 13). rogi 0.1's own combination is a PRODUCT, which is a defect in that
package and not a rival convention: a product of two quantities in (0,1) times 0.5 cannot exceed 0.5,
while published SARI values above 0.5 exist. It is computed here and reported beside the mean as that
package's defect.

FIVE THINGS IN HERE EXIST BECAUSE THE PLAN'S FIRST DRAFT WAS WRONG ABOUT THEM, and one more exists
because the FROZEN plan and the reference implementation cannot both be obeyed literally:

  * THE TARGET IS best_skill FROM modi_baseline.csv. The first draft named frames that do not carry
    what it claimed: a compliant script joined ZERO of 80 rows, and the target it named differs from
    what increment 11 actually measured by up to 0.1786.
  * THE THRESHOLD IS 0.56, the ECFP4 value, not rogi 0.1's MACCS-era hardcoded 0.6.
  * THE REFERENCE SET IS THE 70 POTENCY TARGETS. SARI's weight y_i*y_j/(1+|y_i-y_j|) presumes positive
    log-potency; 8 of the 10 DeepDelta targets carry non-positive labels, down to -25.470, so the
    weight goes negative there and raw_cont is not a weighted mean. Those ten are scored against the
    70's reference set, flagged, and enter no conclusion.
  * THE POTENCY VECTOR IS DECLARED per collection and used as stored, with no centring or rescaling,
    because the weight is a product of two potency values and any shift changes it.
  * NO VECTORISED FAST PATH. The reference path costs about 44 s on the largest target and roughly ten
    minutes over all 80, where a vectorised one would hold about 1.3 GB of upper-triangle arrays.

  * AND THE THRESHOLD TENSION, named rather than papered over. The frozen plan declares 0.56 primary
    AND says the index is computed with the reference implementation's compute_raw_scores(). Those two
    cannot both be obeyed literally, because that method hardcodes 0.6. The resolution: raw_cont comes
    from the reference (it has no threshold), and raw_disc is computed here at BOTH thresholds, with an
    instrument check that this script's raw_disc at 0.6 reproduces the reference's on all 80 targets.
    That check is what makes the substitution legitimate, and it is reported unconditionally.
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from scipy import stats

import rogi
from molace.analysis.correlate import cluster_bootstrap_spearman, incremental_contribution
from molace.data import deepdelta as dd
from molace.data import moleculeace as ma
from molace.graphs.fingerprints import ecfp4, tanimoto_matrix

PREREG = "b62cb36b3eb92b6b960040e54c1fd1bb78aaa194"
CHEMBL = Path("data/raw/chembl_targets")
OUT = Path("results/sari_baseline.csv")

T_PRIMARY, T_SENSITIVITY = 0.56, 0.6
POTENCY = ("ChEMBL-40", "MoleculeACE-30")
OUT_OF_DOMAIN = ("DeepDelta-10",)
# the plan's covariate set for the primary: the other three indices plus the two controls
COVARS = ["rogi", "modi_q2", "sari", "mean_degree", "n_molecules"]
PREDICTOR = "rank_assortativity"
TARGET = "best_skill"
INDICES = [PREDICTOR, "rogi", "modi_q2", "sari"]
P3_MIN_RHO = 0.3
DRAWS, SEED = 10000, 0
TOL = 1e-12


def raw_disc(y: np.ndarray, sim: np.ndarray, t: float) -> tuple[float, int]:
    """Balfer & Bajorath eq. (2) at a DECLARED threshold, which the reference hardcodes at 0.6."""
    iu = np.triu_indices(len(y), k=1)
    s = sim[iu]
    d = np.abs(y[iu[0]] - y[iu[1]])
    m = s > t
    n = int(m.sum())
    return (float((d[m] * s[m]).sum() / n) if n else 1e8), n


def targets():
    meta = pd.read_csv(CHEMBL / "selected.csv")
    for r in meta.itertuples():
        d = pd.read_csv(CHEMBL / f"{r.dataset}.csv")
        yield "ChEMBL-40", r.dataset, d.smiles.tolist(), d.y.to_numpy(float)
    for n in ma.DATASETS:
        d = ma.load_target(n)
        yield "MoleculeACE-30", n, d["smiles"].tolist(), d["y"].to_numpy(float)
    for n in dd.DD_DATASETS:
        d = pd.read_csv(dd.ROOT / "Datasets" / "Benchmarks" / f"{n}.csv")
        yield "DeepDelta-10", n, d.SMILES.tolist(), d.Y.to_numpy(float)


def covariate_frame() -> pd.DataFrame:
    """rogi and mean_degree, which live in three different committed frames. Gate 1 checks this join."""
    a = pd.read_csv("results/rogi_chembl.csv")[["dataset", "rogi", "mean_degree"]]
    s = pd.read_csv("results/rogi_showdown.csv")
    s = s[s.collection == "MoleculeACE"][["dataset", "rogi", "mean_degree"]].copy()
    s["dataset"] = s.dataset.str.replace("^ACE:", "", regex=True)
    d = pd.read_csv("results/separation_deepdelta.csv")[["dataset", "rogi", "mean_degree"]]
    return pd.concat([a, s, d], ignore_index=True)


def main() -> int:
    print(f"pre-registration blob: {PREREG}\n")
    rows = []
    for coll, name, smi, y in targets():
        t0 = time.time()
        sim = tanimoto_matrix(ecfp4(smi)).astype(np.float64)
        ref_cont, ref_disc = rogi.SARI(pKi=y, sim_matrix=sim).compute_raw_scores()
        d_p, n_p = raw_disc(y, sim, T_PRIMARY)
        d_s, n_s = raw_disc(y, sim, T_SENSITIVITY)
        rows.append({"collection": coll, "dataset": name, "n_molecules": len(y),
                     "raw_cont": ref_cont, "ref_raw_disc_0.6": ref_disc,
                     f"raw_disc_{T_PRIMARY}": d_p, f"pairs_{T_PRIMARY}": n_p,
                     f"raw_disc_{T_SENSITIVITY}": d_s, f"pairs_{T_SENSITIVITY}": n_s,
                     "y_min": float(y.min()), "nonpositive_labels": int((y <= 0).sum()),
                     "out_of_domain": bool((y <= 0).any()), "seconds": time.time() - t0})
        print(f"  {coll:15s} {name:24s} n={len(y):5d} cont={ref_cont:.4f} "
              f"disc={d_p:.4f} [{time.time()-t0:.0f}s]", flush=True)
    t = pd.DataFrame(rows)

    # ---- the index, with the reference set as the plan declares it -------------------------------
    ref = t[t.collection.isin(POTENCY)]
    mu_c, sd_c = ref.raw_cont.mean(), ref.raw_cont.std(ddof=0)
    for thr in (T_PRIMARY, T_SENSITIVITY):
        mu_d = ref[f"raw_disc_{thr}"].mean()
        sd_d = ref[f"raw_disc_{thr}"].std(ddof=0)
        zc = stats.norm.cdf((t.raw_cont - mu_c) / sd_c)
        zd = stats.norm.cdf((t[f"raw_disc_{thr}"] - mu_d) / sd_d)
        t[f"sari_{thr}"] = 0.5 * (zc + (1.0 - zd))          # the PUBLISHED mean
        t[f"sari_product_{thr}"] = 0.5 * zc * (1.0 - zd)    # rogi 0.1's defect, reported not used
        t[f"z_cont_{thr}"], t[f"z_disc_{thr}"] = zc, zd
    t["sari"] = t[f"sari_{T_PRIMARY}"]

    mb = pd.read_csv("results/modi_baseline.csv", float_precision="round_trip")
    # receptor_class is carried because incremental_contribution clusters on it internally
    # (src/molace/analysis/correlate.py:154); dropping it crashed the first run.
    mb = mb.rename(columns={"n": "n_mb"})[["collection", "dataset", PREDICTOR, "modi_q2", TARGET,
                                           "cluster", "receptor_class", "n_mb"]]
    f = mb.merge(t.drop(columns=["collection"]), on="dataset", how="left") \
          .merge(covariate_frame(), on="dataset", how="left")
    f.to_csv(OUT, index=False)
    report(f, t)
    return 0


def report(f: pd.DataFrame, t: pd.DataFrame) -> None:
    line = "=" * 100
    print(); print(line); print("INSTRUMENT. THIS SCRIPT'S raw_disc AGAINST THE REFERENCE AT 0.6"); print(line)
    r = (t[f"raw_disc_{T_SENSITIVITY}"] - t["ref_raw_disc_0.6"]).abs()
    print(f"  max |this script - rogi.SARI.compute_raw_scores()| = {r.max():.3e} over {len(t)} targets")
    print(f"  raw_cont comes from the reference directly and carries no threshold")
    print("  The frozen plan declares 0.56 primary AND names compute_raw_scores(), which hardcodes")
    print("  0.6. Both cannot be obeyed literally; this check is what makes computing raw_disc here")
    print("  legitimate, and it is reported whatever it says.")
    instr = float(r.max()) <= TOL
    print(f"  INSTRUMENT: {'PASS' if instr else 'FAIL'} at {TOL:.0e}")
    if not instr:
        print("\n  The substitution is not licensed. The increment stops and this IS the report.")
        return

    print(); print(line); print("GATE 1. THE JOIN IS COMPLETE, AND THIS GATE IS A HARD STOP"); print(line)
    need = [TARGET, "modi_q2", PREDICTOR, "sari", "rogi", "mean_degree", "n_molecules"]
    miss = {c: int(f[c].isna().sum()) for c in need}
    counts = f.collection.value_counts().to_dict()
    print(f"  rows {len(f)}; collections {counts}")
    for c, k in miss.items():
        print(f"  missing {c:20s} {k}")
    g1 = (len(f) == 80 and max(miss.values()) == 0
          and counts.get("ChEMBL-40") == 40 and counts.get("MoleculeACE-30") == 30
          and counts.get("DeepDelta-10") == 10)
    print(f"  GATE 1: {'PASS' if g1 else 'FAIL'}")
    if not g1:
        print("\n  The plan stops the increment here and the discrepancy IS the report.")
        return

    print(); print(line); print("GATE 2. THE THRESHOLD LEAVES PAIRS (ALREADY MEASURED)"); print(line)
    for thr in (T_PRIMARY, T_SENSITIVITY):
        z = int((t[f"pairs_{thr}"] == 0).sum())
        print(f"  targets with zero qualifying pairs at {thr}: {z} of {len(t)}")
    g2 = all(int((t[f'pairs_{thr}'] == 0).sum()) == 0 for thr in (T_PRIMARY, T_SENSITIVITY))
    print(f"  GATE 2: {'PASS' if g2 else 'FAIL'}  (the 1e8 sentinel never fires on this data)")

    print(); print(line); print("GATE 3. THE DOMAIN VIOLATION IS CONFINED"); print(line)
    bad_pot = t[t.collection.isin(POTENCY) & t.out_of_domain]
    ood = t[t.out_of_domain]
    print(f"  potency targets with a non-positive label: {len(bad_pot)} of "
          f"{int(t.collection.isin(POTENCY).sum())}")
    print(f"  out-of-domain targets overall: {len(ood)}  ({', '.join(ood.dataset)})")
    print(f"  label minimum over all targets: {t.y_min.min():.3f}")
    g3 = len(bad_pot) == 0
    print(f"  GATE 3: {'PASS' if g3 else 'FAIL'}")
    print("  SARI's weight y_i*y_j/(1+|y_i-y_j|) presumes positive log-potency; where a label is")
    print("  non-positive the weight goes negative and raw_cont is not a weighted mean.")
    if not g3:
        print("\n  The plan stops the increment here and the discrepancy IS the report.")
        return

    print(); print(line); print("A. THE INDEX, AND THE PACKAGE'S DEFECT BESIDE IT"); print(line)
    print(f"  {'collection':16s} {'SARI (published mean)':>24s} {'rogi 0.1 product':>20s}")
    for c in list(POTENCY) + list(OUT_OF_DOMAIN):
        s = t[t.collection == c]
        flag = "  OUT OF DOMAIN" if c in OUT_OF_DOMAIN else ""
        print(f"  {c:16s} {s[f'sari_{T_PRIMARY}'].min():.4f} .. "
              f"{s[f'sari_{T_PRIMARY}'].max():.4f}{'':>6s} "
              f"{s[f'sari_product_{T_PRIMARY}'].min():.4f} .. "
              f"{s[f'sari_product_{T_PRIMARY}'].max():.4f}{flag}")
    rho_forms = stats.spearmanr(t[f"sari_{T_PRIMARY}"], t[f"sari_product_{T_PRIMARY}"]).statistic
    print(f"\n  Spearman(published mean, rogi 0.1 product) over all {len(t)} = {rho_forms:.4f}")
    print(f"  DISCLOSED: this value was measured before freezing, which is why the plan registers no")
    print(f"  prediction on it. The product is bounded above by 0.5 and published SARI values exceed")
    print(f"  that, so it is this package's defect and not a convention.")
    rho_thr = stats.spearmanr(t[f"sari_{T_PRIMARY}"], t[f"sari_{T_SENSITIVITY}"]).statistic
    print(f"  Spearman(threshold {T_PRIMARY}, threshold {T_SENSITIVITY}) = {rho_thr:.4f}")

    pot = f[f.collection.isin(POTENCY)]
    print(); print(line); print("B. PREDICTION 1. DOES SARI RANK ATTAINABLE ACCURACY"); print(line)
    for c in list(POTENCY) + list(OUT_OF_DOMAIN):
        s = f[f.collection == c]
        b = cluster_bootstrap_spearman(s.sari, s[TARGET], s.cluster, DRAWS, SEED)
        tag = "  (out of domain, descriptive)" if c in OUT_OF_DOMAIN else ""
        print(f"  {c:16s} rho={b.rho:+.4f} [{b.lo:+.4f}, {b.hi:+.4f}]  "
              f"{'EXCLUDES 0' if b.excludes_zero else 'covers 0 '}{tag}")
    ch = f[f.collection == "ChEMBL-40"]
    p1 = cluster_bootstrap_spearman(ch.sari, ch[TARGET], ch.cluster, DRAWS, SEED).excludes_zero
    print(f"  PREDICTION 1: {'HOLDS' if p1 else 'FAILS'} (ChEMBL-40, the primary collection)")

    print(); print(line); print("C. PREDICTION 2, THE PRIMARY. DOES ASSORTATIVITY ADD ANYTHING"); print(line)
    covered = []
    for c in POTENCY:
        s = f[f.collection == c]
        b = incremental_contribution(s, TARGET, COVARS, predictor=PREDICTOR,
                                     n_resamples=DRAWS, seed=SEED)
        covered.append(not b.excludes_zero)
        print(f"  {c:16s} over {{rogi, modi_q2, SARI, mean_degree, n_molecules}}: "
              f"rho={b.rho:+.4f} [{b.lo:+.4f}, {b.hi:+.4f}]  "
              f"{'EXCLUDES 0' if b.excludes_zero else 'covers 0 '}")
        # the same contribution with SARI removed, which the plan requires so a reader can see
        # whether SARI's presence changed anything
        no_sari = [x for x in COVARS if x != "sari"]
        bn = incremental_contribution(s, TARGET, no_sari, predictor=PREDICTOR,
                                      n_resamples=DRAWS, seed=SEED)
        print(f"  {'':16s} with SARI absent from the covariate set: rho={bn.rho:+.4f} "
              f"[{bn.lo:+.4f}, {bn.hi:+.4f}]  "
              f"{'EXCLUDES 0' if bn.excludes_zero else 'covers 0 '}")
    p2 = all(covered)
    print(f"  PREDICTION 2: {'HOLDS' if p2 else 'FAILS'} (covers zero on both potency collections)")

    print(); print(line); print("D. PREDICTION 3. ARE THE FOUR INDICES MUTUALLY REDUNDANT"); print(line)
    print(f"  pairwise |Spearman| on ChEMBL-40, floor {P3_MIN_RHO}")
    worst = 1.0
    for i, a in enumerate(INDICES):
        for b_ in INDICES[i + 1:]:
            rho = abs(stats.spearmanr(ch[a], ch[b_]).statistic)
            worst = min(worst, rho)
            print(f"    {a:20s} {b_:20s} {rho:.4f}{'  <- below the floor' if rho < P3_MIN_RHO else ''}")
    p3 = worst >= P3_MIN_RHO
    print(f"  PREDICTION 3: {'HOLDS' if p3 else 'FAILS'} (weakest pair {worst:.4f})")

    print(); print(line); print("E. THE OUTCOME RULE"); print(line)
    if p2:
        print("  NEAR-INTERCHANGEABLE RANKERS. Claim 2's row is completed with all three published")
        print("  indices named, and the paper stops claiming an incremental win over the baseline")
        print("  line. The row's 'SARI still missing' clause is removed.")
    else:
        print("  ASSORTATIVITY ADDS, on at least one potency collection. Stated per collection and")
        print("  NOT generalised, since increment 11 already found the MODI comparison")
        print("  collection-dependent. Read the SARI-removed contrast above and SARI's own standalone")
        print("  correlation before believing it: a weakened competitor index would produce this")
        print("  spuriously, which is exactly what rogi 0.1's product form would have done.")

    print(); print(line); print("F. EVERY INDEX AGAINST EVERY OTHER, REPORTED UNCONDITIONALLY"); print(line)
    for c in POTENCY:
        s = f[f.collection == c]
        print(f"  {c}:")
        for pred in INDICES:
            others = [x for x in INDICES if x != pred] + ["mean_degree", "n_molecules"]
            b = incremental_contribution(s, TARGET, others, predictor=pred,
                                         n_resamples=DRAWS, seed=SEED)
            al = cluster_bootstrap_spearman(s[pred], s[TARGET], s.cluster, DRAWS, SEED)
            print(f"    {pred:20s} alone {al.rho:+.4f} [{al.lo:+.4f}, {al.hi:+.4f}]   "
                  f"incremental {b.rho:+.4f} [{b.lo:+.4f}, {b.hi:+.4f}]  "
                  f"{'EXCLUDES 0' if b.excludes_zero else 'covers 0'}")


if __name__ == "__main__":
    raise SystemExit(main())

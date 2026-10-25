"""Increment 16: does a one-parameter rank kernel attain the fitted weight optimum.

Pre-registration: prereg/increment16_weighting_family.yaml, blob
0ce65b307616a86cf0b565e854151096501d3f28.

Itskowitz & Tropsha (JCIM 2005, 45(3):777-785) derived the optimal weighting condition for kNN QSAR
as their eq 17 -- a linear system on the weights built from label second moments with a Lagrange
multiplier for the normalisation, which is structurally this project's `fitted` arm -- and then wrote
"Solution of the system of equations (17) ... WILL define the values of F_k" and substituted a tuned
one-parameter kernel, their eq 21, without solving it. Their paper contains both arms, compares
neither against the other, and has no uniform arm at all: its reference is its own kernel at
alpha = 1. No novelty is claimed here for either arm. What is measured is the comparison their own
paper sets up and leaves open, on the rank-indexed family and Tanimoto neighbourhoods this
repository's apparatus supports.

FOUR THINGS IN HERE EXIST BECAUSE THE PLAN'S FIRST DRAFT WAS WRONG ABOUT THEM.

  * `anchors_estimation` masks the diagonal. anchors.select does no masking, so on a square block it
    hands every row ITSELF first at similarity 1.0; then NN_1(v) = v, so C e_1 = c exactly and
    w = C^-1 c = e_1, and the fitted arm silently becomes 1-NN at every m. Measured over 40 ChEMBL
    targets at m=5, the fitted-minus-uniform difference is +0.0159 masked and -0.1018 unmasked. That
    is ten times prediction 3's margin IN THE DIRECTION THAT CONFIRMS THE REGISTERED PREDICTION, so
    the increment would have reported sufficiency off a crippled competitor. The self-exclusion
    assertion beside it is exact (an index-membership test) because a threshold on c_1 does not
    discriminate: unmasked c_1 runs 0.8015 to 1.0024 and only 2 of 40 targets reach 1.0.
  * Gate 1 ties c_first, c_last, skill_mean, skill_rank and cond_C, not the uniform arm's agreement
    with knn_floor.predict. That older statistic compares two computations of the same anchor-set mean
    and is algebraically invariant to the split, the standardisation, the estimation-side
    neighbourhood and the rank order; with the diagonal deliberately unmasked it moves by 0.000e+00
    while c_first moves by 2.515e-01. c_profile cannot serve: it is committed as a 4-decimal string.
  * The conditioning guard is CANDIDATE-level. A per-fold drop would score candidates on CV means
    taken over different numbers of folds, and the drop correlates with fold difficulty, so it would
    flatter the candidate it bites. The size rule is applied the same way, over all folds, for the
    same reason.
  * Gate 3 RE-COMPUTES the selected configuration's CV score from a training split resliced from the
    split rule itself. The first draft checked function signatures, which constrain the callee's
    parameters and not the arguments passed, so it could not fail for the failure it named.

Anchors are selected once per block at max(M_GRID) and sliced per m. Column j is the j-th nearest, so
lexsort(...)[:, :40][:, :m] is lexsort(...)[:, :m] exactly, and gate 1's m=5 and m=10 moments come
free off the refit block rather than costing two more sorts per target.
"""
from __future__ import annotations

import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.model_selection import KFold

from molace.analysis.checkpoint import load_checkpoint, save_checkpoint
from molace.analysis.correlate import cluster_bootstrap_mean
from molace.data import moleculeace as ma
from molace.graphs.fingerprints import ecfp4, tanimoto_matrix
from molace.models import anchors

PREREG = "0ce65b307616a86cf0b565e854151096501d3f28"
CHEMBL = Path("data/raw/chembl_targets")
OUT = Path("results/weighting_family.csv")
PART = Path("results/weighting_family_partial.csv")
COMMITTED = Path("results/local_weights.csv")

M_GRID = (3, 5, 10, 20, 40)
ALPHA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
KERNEL_ARMS = ("kernel", "kernel_scaled")
ALL_ARMS = ("uniform", "kernel", "kernel_scaled", "fitted")
N_FOLDS = 5
SEED = 0
COND_MAX = 100.0           # increment 12's threshold
SIZE_FACTOR = 20           # n_est >= 20 * m(m+1)/2 for the fitted arm
MARGIN = -0.01             # 65 % of increment 12's committed +0.0155
DRAWS = 10000
TOL_G1 = 1e-12
TOL_G2 = 1e-15
P1_ALPHAS = (0.75, 1.0, 1.5)
P1_NEED = 20
KEYS = ["collection", "dataset"]


# ----------------------------------------------------------------- the four weight rules

def w_uniform(m: int) -> np.ndarray:
    return np.full(m, 1.0 / m)


def w_kernel(m: int, alpha: float) -> np.ndarray:
    """Gou's rank weight generalised: w_j proportional to j^-alpha, normalised.

    Computed through its own expression and never by calling w_uniform, so gate 2 compares two
    independent code paths rather than one path with itself.
    """
    w = np.arange(1, m + 1, dtype=float) ** (-alpha)
    return w / w.sum()


def w_fitted(c: np.ndarray, C: np.ndarray) -> np.ndarray:
    return np.linalg.solve(C, c)


def scale_for(shape: np.ndarray, c: np.ndarray, C: np.ndarray) -> float:
    """The least-squares scale for a fixed shape: argmin_b E[(z - b * shape'z_N)^2]."""
    return float((c @ shape) / (shape @ C @ shape))


# ----------------------------------------------------------------- anchors and moments

def anchors_estimation(sim_square: np.ndarray) -> np.ndarray:
    """Estimation-side anchors. The diagonal is masked, and self-exclusion is ASSERTED."""
    s = np.array(sim_square, dtype=np.float64, copy=True)
    np.fill_diagonal(s, -np.inf)
    idx = full_width(anchors.select(s, max(M_GRID)), "estimation")
    hits = int((idx == np.arange(idx.shape[0])[:, None]).any(axis=1).sum())
    if hits:
        raise AssertionError(
            f"{hits} of {idx.shape[0]} estimation rows contain their own index in the anchor set; "
            "the diagonal mask did not take, and the fitted arm would collapse onto rank 1")
    return idx


def anchors_scoring(sim_rect: np.ndarray) -> np.ndarray:
    """Scoring-side anchors over DISJOINT index sets: rectangular, and never masked."""
    return full_width(anchors.select(sim_rect, max(M_GRID)), "scoring")


def full_width(idx: np.ndarray, side: str) -> np.ndarray:
    """anchors.select silently returns min(m, n_train) columns. Fail loudly instead.

    Anchors are selected once at max(M_GRID) and sliced per m, which is exact because column j is
    the j-th nearest. If a block were narrower than the grid's top value, idx[:, :40] would quietly
    become idx[:, :n_train] and the m=40 candidate would be a different, smaller arm wearing the
    label m=40. The smallest training split here is 451 rows, so this cannot fire on the registered
    collections; it exists so that it cannot fire silently on any other.
    """
    if idx.shape[1] != max(M_GRID):
        raise AssertionError(
            f"{side}-side anchors came back {idx.shape[1]} wide, not {max(M_GRID)}: the block has "
            f"too few columns for the declared m grid")
    return idx


def moments(z_est: np.ndarray, idx_est: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
    """Raw second moments of the rank-resolved neighbourhood, exactly increment 12's."""
    zn = z_est[idx_est[:, :m]]
    return (z_est[:, None] * zn).mean(axis=0), (zn.T @ zn) / zn.shape[0]


def ascending(a: np.ndarray, what: str) -> np.ndarray:
    """Every index set handed to anchors.select is ascending; asserted, not assumed."""
    if np.any(np.diff(a) <= 0):
        raise AssertionError(f"{what} is not strictly ascending; the tie-break convention is broken")
    return a


# ----------------------------------------------------------------- one target

def split_of(n: int) -> tuple[np.ndarray, np.ndarray]:
    """The split rule itself. Gate 3 reslices from HERE, never from arrays it was handed."""
    tr = np.random.default_rng(SEED).random(n) < 0.8
    return np.flatnonzero(tr), np.flatnonzero(~tr)


def fold_sizes(n_tr: int) -> list[int]:
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    return [len(ft) for ft, _ in kf.split(np.arange(n_tr))]


def admissible(n_est: int) -> list[int]:
    """The fitted arm's m, by the size rule, on an estimation split of n_est rows."""
    return [m for m in M_GRID if n_est >= SIZE_FACTOR * m * (m + 1) // 2]


def cv_table(sim: np.ndarray, y: np.ndarray, tr_i: np.ndarray) -> tuple[dict, dict, list[int]]:
    """Mean held-out RMSE per (arm, m, alpha), on the training split alone.

    Every candidate is scored on the identical five folds. Candidates the size rule or the
    conditioning guard excludes on ANY fold are absent entirely rather than averaged over fewer.
    """
    n_tr = len(tr_i)
    fitted_ok = admissible(min(fold_sizes(n_tr)))
    acc: dict[tuple, list[float]] = defaultdict(list)
    conds: dict[int, list[float]] = defaultdict(list)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for ft, fv in kf.split(np.arange(n_tr)):
        g_ft = ascending(tr_i[np.sort(ft)], "fold-train")
        g_fv = ascending(tr_i[np.sort(fv)], "fold-validation")
        mu, sd = float(y[g_ft].mean()), float(y[g_ft].std())
        z_est = (y[g_ft] - mu) / sd
        idx_e = anchors_estimation(sim[np.ix_(g_ft, g_ft)])
        idx_s = anchors_scoring(sim[np.ix_(g_fv, g_ft)])
        y_fv = y[g_fv]

        for m in M_GRID:
            if m > len(g_ft) - 1:
                continue
            c, C = moments(z_est, idx_e, m)
            conds[m].append(float(np.linalg.cond(C)))
            zn_s = z_est[idx_s[:, :m]]

            def rmse(w: np.ndarray) -> float:
                return float(np.sqrt(np.mean((y_fv - (mu + sd * (zn_s @ w))) ** 2)))

            acc[("uniform", m, 0.0)].append(rmse(w_uniform(m)))
            for a in ALPHA_GRID:
                shape = w_kernel(m, a)
                acc[("kernel", m, a)].append(rmse(shape))
                acc[("kernel_scaled", m, a)].append(rmse(scale_for(shape, c, C) * shape))
            if m in fitted_ok:
                acc[("fitted", m, 0.0)].append(rmse(w_fitted(c, C)))

    # candidate-level conditioning guard: a breach on ANY fold removes the m for the whole CV
    blocked = {m for m, v in conds.items()
               if not np.all(np.isfinite(v)) or max(v) >= COND_MAX}
    scores = {k: float(np.mean(v)) for k, v in acc.items()
              if len(v) == N_FOLDS and k[1] not in blocked}
    return scores, conds, sorted(set(fitted_ok) - blocked)


def pick(scores: dict, arm: str, allowed: set[int] | None = None) -> tuple[int, float, float]:
    """Argmin mean CV RMSE. Ties break toward the SMALLER m and then the SMALLER alpha."""
    cand = [(k, v) for k, v in scores.items()
            if k[0] == arm and (allowed is None or k[1] in allowed)]
    if not cand:
        return (0, 0.0, float("nan"))
    (_, m, a), s = min(cand, key=lambda kv: (kv[1], kv[0][1], kv[0][2]))
    return m, a, s


def one_target(coll: str, name: str, smiles: list[str], y: np.ndarray) -> dict:
    n = len(y)
    tr_i, te_i = split_of(n)
    sim = tanimoto_matrix(ecfp4(smiles))
    ascending(tr_i, "training split")

    scores, conds, fitted_cv_ok = cv_table(sim, y, tr_i)

    # ---- the final refit: estimate on the whole training split, score the test split
    mu, sd = float(y[tr_i].mean()), float(y[tr_i].std())
    z_tr = (y[tr_i] - mu) / sd
    idx_e = anchors_estimation(sim[np.ix_(tr_i, tr_i)])
    idx_s = anchors_scoring(sim[np.ix_(te_i, tr_i)])
    y_te = y[te_i]
    base = float(np.sqrt(np.mean((y_te - mu) ** 2)))
    fitted_refit_ok = set(admissible(len(tr_i)))

    def skill_at(m: int, arm: str, alpha: float) -> float:
        c, C = moments(z_tr, idx_e, m)
        zn = z_tr[idx_s[:, :m]]
        if arm == "uniform":
            w = w_uniform(m)
        elif arm == "kernel":
            w = w_kernel(m, alpha)
        elif arm == "kernel_scaled":
            shape = w_kernel(m, alpha)
            w = scale_for(shape, c, C) * shape
        else:
            w = w_fitted(c, C)
        return 1.0 - float(np.sqrt(np.mean((y_te - (mu + sd * (zn @ w))) ** 2))) / base

    row = {"collection": coll, "dataset": name, "cluster": name, "prereg": PREREG,
           "n_molecules": n, "n_train": len(tr_i), "n_test": len(te_i),
           "fitted_cv_admissible": " ".join(map(str, fitted_cv_ok)),
           "fitted_refit_admissible": " ".join(map(str, sorted(fitted_refit_ok))),
           "cond_max_over_folds": max(max(v) for v in conds.values()),
           "cond_guard_blocked_m": " ".join(
               str(m) for m, v in sorted(conds.items())
               if not np.all(np.isfinite(v)) or max(v) >= COND_MAX),
           "cond_guard_bound": any(not np.all(np.isfinite(v)) or max(v) >= COND_MAX
                                   for v in conds.values())}

    # ---- own-grid selection, and the matched grid (the fitted arm's refit-admissible set)
    matched = fitted_refit_ok
    for arm in ALL_ARMS:
        allowed = fitted_refit_ok if arm == "fitted" else None
        m, a, s = pick(scores, arm, allowed)
        row[f"{arm}_m"], row[f"{arm}_alpha"], row[f"{arm}_cv"] = m, a, s
        row[f"skill_{arm}"] = skill_at(m, arm, a) if m else float("nan")

        mm, aa, ss = pick(scores, arm, matched)
        row[f"{arm}_m_matched"], row[f"{arm}_alpha_matched"] = mm, aa
        row[f"skill_{arm}_matched"] = skill_at(mm, arm, aa) if mm else float("nan")

        # the cost of honest selection: was the CV choice the test-best choice?
        pool = [k for k in scores if k[0] == arm and (allowed is None or k[1] in allowed)]
        best = max(pool, key=lambda k: skill_at(k[1], arm, k[2])) if pool else None
        row[f"{arm}_cv_was_test_best"] = bool(best == (arm, m, a)) if best else False
        row[f"{arm}_regret"] = (skill_at(best[1], arm, best[2]) - row[f"skill_{arm}"]
                                if best else float("nan"))

    # ---- gate 1, free off the refit block: increment 12's columns at m = 5 and m = 10
    for m in (5, 10):
        c, C = moments(z_tr, idx_e, m)
        zn = z_tr[idx_s[:, :m]]
        row[f"g1_c_first_m{m}"] = float(c[0])
        row[f"g1_c_last_m{m}"] = float(c[-1])
        row[f"g1_cond_C_m{m}"] = float(np.linalg.cond(C))
        for arm, w in (("mean", w_uniform(m)), ("rank", w_fitted(c, C))):
            row[f"g1_skill_{arm}_m{m}"] = 1.0 - float(
                np.sqrt(np.mean((y_te - (mu + sd * (zn @ w))) ** 2))) / base

    # ---- gate 2: alpha = 0 must reproduce the uniform arm, two independent expressions
    row["g2_residual"] = max(float(np.max(np.abs(w_kernel(m, 0.0) - w_uniform(m))))
                             for m in M_GRID)

    # ---- gate 3: reslice from the split rule and recompute the selected CV scores
    tr_again, _ = split_of(n)
    if not np.array_equal(tr_again, tr_i):
        raise AssertionError("the split rule is not deterministic; gate 3 cannot be evaluated")
    again, _, _ = cv_table(sim, y, tr_again)
    row["g3_max_residual"] = max(
        abs(again.get((arm, row[f"{arm}_m"], row[f"{arm}_alpha"]), float("nan")) - row[f"{arm}_cv"])
        for arm in ALL_ARMS if row[f"{arm}_m"])
    return row


# ----------------------------------------------------------------- the run

def collections():
    meta = pd.read_csv(CHEMBL / "selected.csv")
    for m in meta.itertuples():
        d = pd.read_csv(CHEMBL / f"{m.dataset}.csv")
        yield "ChEMBL-40", m.dataset, d.smiles.tolist(), d.y.to_numpy(dtype=float)
    for nm in ma.DATASETS:
        d = ma.load_target(nm)
        yield "MoleculeACE-30", nm, d["smiles"].tolist(), d["y"].to_numpy(dtype=float)


def main() -> int:
    print(f"pre-registration blob: {PREREG}\n")
    rows = load_checkpoint(PART, KEYS)
    done = {(r["collection"], r["dataset"]) for r in rows}
    if done:
        print(f"resuming with {len(done)} target(s) already measured\n", file=sys.stderr)

    for coll, name, smiles, y in collections():
        if (coll, name) in done:
            continue
        t0 = time.time()
        r = one_target(coll, name, smiles, y)
        rows.append(r)
        save_checkpoint(PART, rows)
        print(f"  {coll:14s} {name:24s} n={r['n_molecules']:5d}  "
              f"unif m={r['uniform_m']:<3d} kern m={r['kernel_m']:<3d} a={r['kernel_alpha']:<5} "
              f"fit m={r['fitted_m']:<3d}  skill u/k/f = {r['skill_uniform']:+.4f}/"
              f"{r['skill_kernel']:+.4f}/{r['skill_fitted']:+.4f}  [{time.time()-t0:.0f}s]",
              file=sys.stderr, flush=True)

    t = pd.DataFrame(rows)
    t.to_csv(OUT, index=False)
    report(t)
    return 0


# ----------------------------------------------------------------- the report

def boot(x, clusters):
    r = cluster_bootstrap_mean(np.asarray(x, dtype=float), np.asarray(clusters), DRAWS, SEED)
    return r.mean, r.lo, r.hi


def report(t: pd.DataFrame) -> None:
    line = "=" * 100
    ch = t[t.collection == "ChEMBL-40"]

    print(line); print("GATE 1. THE INSTRUMENT, AGAINST THE COMMITTED INCREMENT-12 ARTEFACT"); print(line)
    com = pd.read_csv(COMMITTED, float_precision="round_trip")
    com = com[com.collection == "ChEMBL-40"]
    worst, counted = {}, {}
    for m in (5, 10):
        c = com[(com.m == m) & com.gate2_pass.fillna(False)].set_index("dataset")
        g = ch.set_index("dataset")
        shared = [d for d in g.index if d in c.index]
        counted[m] = len(shared)
        for col, mine in (("c_first", f"g1_c_first_m{m}"), ("c_last", f"g1_c_last_m{m}"),
                          ("skill_mean", f"g1_skill_mean_m{m}"),
                          ("skill_rank", f"g1_skill_rank_m{m}"), ("cond_C", f"g1_cond_C_m{m}")):
            worst[f"{col} @ m={m}"] = float((g.loc[shared, mine] - c.loc[shared, col]).abs().max())
    for k, v in worst.items():
        print(f"  {k:24s} max |recomputed - committed| = {v:.3e}")
    print(f"\n  checked on {counted[5]} targets at m=5 and {counted[10]} at m=10")
    print("  self-exclusion is asserted at every anchor call and would have raised, not been scored")
    g1 = max(worst.values()) <= TOL_G1 and counted[5] == 40 and counted[10] == 31
    print(f"  GATE 1: {'PASS' if g1 else 'FAIL'} at tolerance {TOL_G1:.0e}")
    if not g1:
        print("\n  The plan stops the increment here and the discrepancy IS the report.")
        return

    print(); print(line); print("GATE 2. ALPHA = 0 AGAINST THE UNIFORM ARM (DECLARED WEAK)"); print(line)
    g2 = float(t.g2_residual.max())
    print(f"  max |kernel(alpha=0) - uniform| over every m and target = {g2:.3e}")
    print(f"  GATE 2: {'PASS' if g2 <= TOL_G2 else 'FAIL'} at tolerance {TOL_G2:.0e}")
    print("  This gate cannot fail for a correct implementation and is NOT evidence the kernel arm")
    print("  is right. It is kept because it would catch a normalisation mismatch between the arms.")
    if g2 > TOL_G2:
        return

    print(); print(line); print("GATE 3. SELECTION IS BLIND, BY RECOMPUTATION FROM THE SPLIT RULE"); print(line)
    g3 = float(t.g3_max_residual.max())
    print(f"  max |recomputed CV score - recorded| over every arm and target = {g3:.3e}")
    g3ok = g3 == 0.0
    print(f"  GATE 3: {'PASS' if g3ok else 'FAIL'}, bit for bit")
    if not g3ok:
        print("\n  The plan stops the increment here and the discrepancy IS the report.")
        return

    print(); print(line); print("A. WHAT EACH ARM SELECTED, AND WHAT IT SCORED"); print(line)
    print(f"  {'arm':14s} {'params':>6s} {'median m':>9s} {'median alpha':>13s} "
          f"{'mean skill':>11s} {'CV was test-best':>17s} {'mean regret':>12s}")
    for arm, p in (("uniform", "0"), ("kernel", "1"), ("kernel_scaled", "2"), ("fitted", "m")):
        s = ch[ch[f"{arm}_m"] > 0]
        a = s[f"{arm}_alpha"].median() if arm in KERNEL_ARMS else float("nan")
        print(f"  {arm:14s} {p:>6s} {s[f'{arm}_m'].median():>9.1f} "
              f"{a if a == a else 0:>13.3f} {s[f'skill_{arm}'].mean():>11.4f} "
              f"{int(s[f'{arm}_cv_was_test_best'].sum()):>12d}/{len(s):<4d} "
              f"{s[f'{arm}_regret'].mean():>12.4f}")

    print(); print(line); print("B. THE PRIMARY STATISTIC, AND PREDICTION 2"); print(line)
    d2 = ch.skill_kernel - ch.skill_uniform
    m2, lo2, hi2 = boot(d2, ch.cluster)
    p2 = lo2 > 0
    print(f"  mean(skill_kernel - skill_uniform), own grids = {m2:+.4f} [{lo2:+.4f}, {hi2:+.4f}]")
    print(f"  positive on {int((d2 > 0).sum())}/{len(d2)} targets")
    print(f"  PREDICTION 2: {'HOLDS' if p2 else 'FAILS'}")
    print("  DISCLOSED: this statistic was measured during the pre-freeze review, so the plan is a")
    print("  pre-registration of procedure and not a blind one. See disclosure (a) in the prereg.")

    print(); print(line); print("C. PREDICTION 3 AND THE SUFFICIENCY INSTRUMENT"); print(line)
    dm = ch.skill_kernel_matched - ch.skill_fitted_matched
    m3, lo3, hi3 = boot(dm, ch.cluster)
    do = ch.skill_kernel - ch.skill_fitted
    mo, loo, hio = boot(do, ch.cluster)
    print(f"  matched grid: mean(skill_kernel - skill_fitted) = {m3:+.4f} [{lo3:+.4f}, {hi3:+.4f}]")
    print(f"  own grids   : mean(skill_kernel - skill_fitted) = {mo:+.4f} [{loo:+.4f}, {hio:+.4f}]")
    print(f"  margin = {MARGIN:+.2f}, which is 65 % of increment 12's committed +0.0155")
    p3 = lo3 > MARGIN
    print(f"  PREDICTION 3: {'HOLDS' if p3 else 'FAILS'} "
          f"(one-sided non-inferiority: lower end {lo3:+.4f} vs {MARGIN:+.2f})")

    print(); print(line); print("D. PREDICTION 1, WHERE ALPHA LANDED"); print(line)
    s = ch[ch.kernel_m > 0]
    freq = s.kernel_alpha.value_counts().sort_index()
    print("     alpha " + " ".join(f"{a:>6}" for a in ALPHA_GRID))
    print("     count " + " ".join(f"{int(freq.get(a, 0)):>6d}" for a in ALPHA_GRID))
    hit = int(s.kernel_alpha.isin(P1_ALPHAS).sum())
    q1, q3 = s.kernel_alpha.quantile(0.25), s.kernel_alpha.quantile(0.75)
    p1 = hit >= P1_NEED
    print(f"  median {s.kernel_alpha.median():.3f}, IQR [{q1:.3f}, {q3:.3f}]")
    print(f"  alpha in {P1_ALPHAS} on {hit}/{len(s)} targets, needs >= {P1_NEED}")
    print(f"  PREDICTION 1: {'HOLDS' if p1 else 'FAILS'}")
    ks = ch[ch.kernel_scaled_m > 0]
    print(f"  kernel_scaled, reported beside and not entering the prediction: median alpha "
          f"{ks.kernel_scaled_alpha.median():.3f}, agrees with kernel on "
          f"{int((ks.kernel_scaled_alpha == ks.kernel_alpha).sum())}/{len(ks)}")

    print(); print(line); print("E. PREDICTION 4, THE SELECTED NEIGHBOURHOOD SIZES"); print(line)
    own = (ch.kernel_m.median(), ch[ch.fitted_m > 0].fitted_m.median())
    mat = (ch.kernel_m_matched.median(), ch[ch.fitted_m_matched > 0].fitted_m_matched.median())
    p4 = own[0] > own[1]
    print(f"  own grids   : kernel median m {own[0]:.1f} against fitted {own[1]:.1f}  "
          f"-> {'HOLDS' if p4 else 'FAILS'}  (this is the test)")
    print(f"  matched grid: kernel median m {mat[0]:.1f} against fitted {mat[1]:.1f}  "
          f"(DESCRIPTION ONLY: the matched grid is the fitted arm's admissible set and truncates")
    print(f"   the kernel harder, deleting the region where its preference for larger m would live)")
    print(f"  PREDICTION 4: {'HOLDS' if p4 else 'FAILS'}")
    print("  Registered AGAINST the project's only committed evidence, which points the other way:")
    print("  increment 12 found uniform prefers m=5 to m=10 while fitted prefers 10 to 5.")

    print(); print(line); print("F. DID THE RESHAPING GAIN SURVIVE SYMMETRIC SELECTION"); print(line)
    dr = ch.skill_fitted_matched - ch.skill_uniform_matched
    mr, lor, hir = boot(dr, ch.cluster)
    print(f"  matched grid: mean(skill_fitted - skill_uniform) = {mr:+.4f} [{lor:+.4f}, {hir:+.4f}]")
    print(f"  increment 12's committed figure, matched two-point grid: +0.0155 [+0.0118, +0.0192]")
    survived = not (lor <= 0 <= hir)
    print(f"  the gain {'SURVIVES' if survived else 'DOES NOT SURVIVE'} selection")

    print(); print(line); print("G. THE OUTCOME RULE, ONE INSTRUMENT AND THREE BRANCHES"); print(line)
    print(f"  instrument: the matched-grid interval [{lo3:+.4f}, {hi3:+.4f}] against {MARGIN:+.2f}")
    if p2 and lo3 > MARGIN:
        print("\n  THE KERNEL SUFFICES. The paper states the reshaping gain is a one-parameter effect,")
        print("  cites Itskowitz & Tropsha 2005 for both the optimum (eq 17) and the kernel form")
        print("  (eq 21) with novelty claimed for neither, and reports the fitted optimum as the")
        print("  upper bound the kernel attains.")
    elif hi3 < MARGIN:
        print("\n  THE FIT IS NEEDED. The m-parameter fit buys what one parameter does not; the paper")
        print("  reports how much and names the parameter count as the price.")
    else:
        print("\n  NOT ESTABLISHED. In increment 12's own wording: sufficiency is not established and")
        print("  this report says not established rather than claiming a null from a failure to")
        print(f"  reject. The difference is {m3:+.4f} and the interval straddles {MARGIN:+.2f}.")
    if not survived:
        print("\n  AND the reshaping gain did not survive symmetric selection, so section 5.2 is")
        print("  downgraded a second time, in the form claim 4's withdrawal took.")

    print(); print(line); print("H. THE SAME ON MoleculeACE-30"); print(line)
    mt = t[t.collection == "MoleculeACE-30"]
    if len(mt):
        dd = mt.skill_kernel - mt.skill_uniform
        mm_, ll, hh = boot(dd, mt.cluster)
        dq = mt.skill_kernel_matched - mt.skill_fitted_matched
        m4, l4, h4 = boot(dq, mt.cluster)
        print(f"  kernel - uniform  = {mm_:+.4f} [{ll:+.4f}, {hh:+.4f}], positive on "
              f"{int((dd > 0).sum())}/{len(dd)}")
        print(f"  kernel - fitted (matched) = {m4:+.4f} [{l4:+.4f}, {h4:+.4f}]")
        print(f"  median selected alpha {mt.kernel_alpha.median():.3f}, "
              f"alpha in {P1_ALPHAS} on {int(mt.kernel_alpha.isin(P1_ALPHAS).sum())}/{len(mt)}")

    print(); print(line); print("I. WHERE THE GUARDS BOUND"); print(line)
    nb = int(t.cond_guard_bound.sum())
    print(f"  the conditioning guard bound on {nb} of {len(t)} targets; "
          f"highest cond(C) over all folds = {t.cond_max_over_folds.max():.3f} against {COND_MAX}")
    for r in t[t.cond_guard_bound].itertuples():
        print(f"    {r.collection:14s} {r.dataset:24s} blocked m = {r.cond_guard_blocked_m}, "
              f"max cond {r.cond_max_over_folds:.3f}")
    print(f"  fitted-arm CV-admissible sets: "
          f"{t.fitted_cv_admissible.value_counts().to_dict()}")
    print(f"  fitted-arm refit-admissible sets: "
          f"{t.fitted_refit_admissible.value_counts().to_dict()}")


if __name__ == "__main__":
    raise SystemExit(main())

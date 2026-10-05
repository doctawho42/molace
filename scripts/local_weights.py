"""Increment 12: what exchangeability deleted, and whether claim 6 survives without it.

Pre-registration: prereg/increment12_local_weights.yaml, blob
32dddd1a8aa1be4b6cad0e24b3e0778fdac436ef.

Claim 6 says equal weights are already near-optimal locally. Under exchangeability that is nearly
vacuous, and provably so: 1 is an eigenvector of the neighbour second-moment matrix and the
cross-moment vector is parallel to it, so every optimal weight is EXACTLY equal, the weight sum is
exactly the regression slope of the target on the neighbourhood mean, the whole gap is
Var(mean) (1 - beta)^2, and fitting one scalar on the plain mean attains the m-dimensional optimum
exactly. The assumption forbids the question the claim appears to answer.

So this script measures the term the assumption deleted. Neighbour RANK is the axis: anchors.select
already ranks by descending similarity, so column j of its output is the j-th nearest neighbour and
no new graph machinery is needed. Three arms separate a scale gain from a shape gain. Everything is
estimated on the training split alone. Weights are never normalised to sum to one, because that
constraint makes the optimum exactly 1/m and would report a null by construction.
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from molace.analysis.correlate import cluster_bootstrap_mean, cluster_bootstrap_spearman
from molace.data import deepdelta as dd
from molace.data import moleculeace as ma
from molace.graphs import knn
from molace.graphs.fingerprints import ecfp4, tanimoto_matrix
from molace.measures.assortativity import target_assortativity
from molace.models import anchors, knn_floor

PREREG = "32dddd1a8aa1be4b6cad0e24b3e0778fdac436ef"
CHEMBL = Path("data/raw/chembl_targets")
PARTIAL = Path("results/local_weights_partial.csv")
OUT = Path("results/local_weights.csv")

K = 10                      # the graph's k, for target assortativity only
M_VALUES = (5, 10, 20)      # frozen
M_PRIMARY = 10              # frozen
DRAWS = 10000               # frozen
SEED = 0                    # frozen
SURROGATES = 200            # frozen
COND_MAX = 100.0            # gate 2, declared in advance
SIZE_FACTOR = 20            # gate 3: n_train >= SIZE_FACTOR * m(m+1)/2


def collections():
    meta = pd.read_csv(CHEMBL / "selected.csv")
    for _, m in meta.iterrows():
        d = pd.read_csv(CHEMBL / f"{m.dataset}.csv")
        yield "ChEMBL-40", m.dataset, d.smiles.tolist(), d.y.to_numpy(dtype=float)
    for n in ma.DATASETS:
        d = ma.load_target(n)
        yield "MoleculeACE-30", n, d["smiles"].tolist(), d["y"].to_numpy(dtype=float)
    for n in dd.DD_DATASETS:
        d = pd.read_csv(dd.ROOT / "Datasets" / "Benchmarks" / f"{n}.csv")
        yield "DeepDelta-10", n, d["SMILES"].tolist(), d["Y"].to_numpy(dtype=float)


def second_moments(z_tr: np.ndarray, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Raw second moments of the rank-resolved neighbourhood.

    Second moments, not Pearson correlations: the expansion 1 - 2 w'c + w'Cw is exact for
    E[(z - w'z_N)^2] only if c and C are raw second moments on the standardised scale.
    """
    zn = z_tr[idx]                                       # (n, m), column j = j-th nearest
    c = (z_tr[:, None] * zn).mean(axis=0)
    C = (zn.T @ zn) / zn.shape[0]
    return c, C


def weights(c: np.ndarray, C: np.ndarray, arm: str) -> np.ndarray:
    """NEVER normalised to sum to one. See the pre-registration's forbidden_normalisation."""
    m = c.size
    if arm == "mean":
        return np.full(m, 1.0 / m)
    if arm == "scaled":                                  # one free parameter
        beta = (c.sum() / m) / (C.sum() / m**2)
        return np.full(m, beta / m)
    if arm == "rank":                                    # m free parameters
        return np.linalg.solve(C, c)
    raise ValueError(arm)


def skill_of(y_te: np.ndarray, pred: np.ndarray, base: float) -> float:
    return 1.0 - float(np.sqrt(np.mean((y_te - pred) ** 2))) / base


def surrogate_shape_gain(z_tr, idx_tr, idx_te, y_te, base, mu, sd, rng) -> float:
    """The null 'exchangeability holds': destroy the rank assignment, keep every neighbour and label.

    Permuted independently in estimation and in application, because under the null the data are
    exchangeable and a permutation changes nothing in distribution. What survives is the noise in
    fitting m weights, which is the quantity a measured shape gain must be compared against.
    """
    m = idx_tr.shape[1]
    p_tr = rng.permuted(np.tile(np.arange(m), (idx_tr.shape[0], 1)), axis=1)
    p_te = rng.permuted(np.tile(np.arange(m), (idx_te.shape[0], 1)), axis=1)
    c, C = second_moments(z_tr, np.take_along_axis(idx_tr, p_tr, axis=1))
    if not np.isfinite(np.linalg.cond(C)) or np.linalg.cond(C) >= COND_MAX:
        return np.nan
    zn = z_tr[np.take_along_axis(idx_te, p_te, axis=1)]
    s_scaled = skill_of(y_te, mu + sd * (zn @ weights(c, C, "scaled")), base)
    s_rank = skill_of(y_te, mu + sd * (zn @ weights(c, C, "rank")), base)
    return s_rank - s_scaled


def one_target(coll: str, name: str, smiles: list[str], y: np.ndarray) -> list[dict]:
    rng = np.random.default_rng(SEED)
    tr = rng.random(len(y)) < 0.8                        # identical split to every earlier increment
    tr_i, te_i = np.flatnonzero(tr), np.flatnonzero(~tr)
    sim = tanimoto_matrix(ecfp4(smiles))
    base = float(np.sqrt(np.mean((y[~tr] - y[tr].mean()) ** 2)))
    mu, sd = float(y[tr].mean()), float(y[tr].std())
    z = (y - mu) / sd
    z_tr = z[tr_i]

    r = float(target_assortativity(knn.knn_graph(sim, k=K), y).value)
    var_y = float(np.var(y))

    s_tt = sim[np.ix_(tr_i, tr_i)].copy()
    np.fill_diagonal(s_tt, -np.inf)                      # never your own neighbour
    s_et = sim[np.ix_(te_i, tr_i)]

    out = []
    for m in M_VALUES:
        need = SIZE_FACTOR * m * (m + 1) // 2
        row = {"collection": coll, "dataset": name, "cluster": name, "receptor_class": name,
               "prereg": PREREG, "m": m, "n_molecules": len(y), "n_train": int(tr.sum()),
               "n_test": int((~tr).sum()), "assortativity": r, "gate3_need": need,
               "gate3_pass": bool(tr.sum() >= need)}
        if not row["gate3_pass"] or len(te_i) < 2:
            out.append(row)
            continue

        idx_tr = anchors.select(s_tt, m)
        idx_te = anchors.select(s_et, m)
        c, C = second_moments(z_tr, idx_tr)
        cond = float(np.linalg.cond(C))
        row |= {"cond_C": cond, "gate2_pass": bool(np.isfinite(cond) and cond < COND_MAX),
                "c_first": float(c[0]), "c_last": float(c[-1]),
                "c_decline": float(c[0] - c[-1]),
                "c_profile": " ".join(f"{v:.4f}" for v in c)}
        if not row["gate2_pass"]:
            out.append(row)
            continue

        zn_te = z_tr[idx_te]
        beta = float((c.sum() / m) / (C.sum() / m**2))
        w_rank = weights(c, C, "rank")
        sk = {}
        for arm in ("mean", "scaled", "rank"):
            sk[arm] = skill_of(y[te_i], mu + sd * (zn_te @ weights(c, C, arm)), base)

        # the mean arm must reproduce the existing floor exactly, or the new code is wrong
        drift = float(np.max(np.abs(
            (mu + sd * (zn_te @ weights(c, C, "mean"))) - knn_floor.predict(y[tr_i], idx_te))))

        rel_mse_mean = float(np.mean((y[te_i] - (mu + sd * (zn_te @ np.full(m, 1.0 / m)))) ** 2)
                             / var_y)
        row |= {"beta": beta, "beta_dev": abs(beta - 1.0),
                "skill_mean": sk["mean"], "skill_scaled": sk["scaled"], "skill_rank": sk["rank"],
                "gain_scale": sk["scaled"] - sk["mean"], "gain_shape": sk["rank"] - sk["scaled"],
                "w_rank_sum": float(w_rank.sum()),
                "w_rank_spread": float(w_rank.max() - w_rank.min()),
                "w_rank": " ".join(f"{v:.4f}" for v in w_rank),
                "floor_drift": drift, "rel_mse_mean": rel_mse_mean,
                "rel_mse_minus_one_minus_r": abs(rel_mse_mean - (1.0 - r))}

        if m == M_PRIMARY:
            srng = np.random.default_rng(SEED)
            s = np.array([surrogate_shape_gain(z_tr, idx_tr, idx_te, y[te_i], base, mu, sd, srng)
                          for _ in range(SURROGATES)], dtype=float)
            s = s[np.isfinite(s)]
            row |= {"surr_n": int(s.size),
                    "surr_p95": float(np.percentile(s, 95)) if s.size else np.nan,
                    "surr_mean": float(np.mean(s)) if s.size else np.nan,
                    "beats_surrogate": bool(s.size and row["gain_shape"] > np.percentile(s, 95))}
        out.append(row)
    return out


def main() -> int:
    print(f"pre-registration blob: {PREREG}")
    print(f"m in {M_VALUES}, primary {M_PRIMARY}; gate 2 cond(C) < {COND_MAX}; "
          f"gate 3 n_train >= {SIZE_FACTOR}*m(m+1)/2; {SURROGATES} rank-shuffled surrogates")
    print()

    rows = pd.read_csv(PARTIAL).to_dict("records") if PARTIAL.exists() else []
    done = {(r["collection"], r["dataset"]) for r in rows}
    if done:
        print(f"resuming from {PARTIAL}: {len(done)} targets already done\n")

    for coll, name, smiles, y in collections():
        if (coll, name) in done:
            continue
        t0 = time.time()
        try:
            got = one_target(coll, name, smiles, y)
        except Exception as exc:
            print(f"  SKIP {coll} {name}: {type(exc).__name__}: {exc}", flush=True)
            continue
        rows.extend(got)
        pd.DataFrame(rows).to_csv(PARTIAL, index=False)
        p = next((g for g in got if g["m"] == M_PRIMARY), got[0])
        msg = (f"  {coll:16s} {name:28s} n={p['n_molecules']:5d} "
               f"gate3={'y' if p['gate3_pass'] else 'n'}")
        if p.get("gate2_pass"):
            msg += (f" decline={p['c_decline']:+.4f} beta={p['beta']:.3f} "
                    f"scale={p['gain_scale']:+.4f} shape={p['gain_shape']:+.4f} "
                    f"surr95={p.get('surr_p95', float('nan')):+.4f} drift={p['floor_drift']:.1e}")
        print(f"{msg}  [{time.time()-t0:.0f}s]", flush=True)

    t = pd.DataFrame(rows)
    t.to_csv(OUT, index=False)
    print(f"\nwrote {OUT} ({len(t)} rows)\n")

    # ---- the implementation check, reported unconditionally -------------------------------------
    d = t[t.get("floor_drift").notna()] if "floor_drift" in t else t.iloc[:0]
    if len(d):
        print(f"mean arm against the existing kNN floor: max absolute drift "
              f"{d.floor_drift.max():.3e} over {len(d)} target-m cells")
        print("  (a non-zero drift would mean the new code is not computing the floor it claims)\n")

    for coll in ("ChEMBL-40", "MoleculeACE-30", "DeepDelta-10"):
        print("=" * 78)
        print(coll)
        print("=" * 78)
        for m in M_VALUES:
            s = t[(t.collection == coll) & (t.m == m)]
            if not len(s):
                continue
            g3 = s[s.gate3_pass]
            print(f"\n  m = {m}: gate 3 retains {len(g3)}/{len(s)} "
                  f"(needs n_train >= {SIZE_FACTOR*m*(m+1)//2})")
            if not len(g3):
                print("    nothing retained; no outcome at this m, as the frozen gate requires")
                continue
            k = g3[g3.gate2_pass.fillna(False)] if "gate2_pass" in g3 else g3.iloc[:0]
            print(f"    gate 2 retains {len(k)}/{len(g3)} (cond(C) < {COND_MAX})")
            if not len(k):
                continue

            gate1 = cluster_bootstrap_mean(k.c_decline, k.receptor_class, DRAWS, SEED)
            print(f"    GATE 1, the premise: mean rank decline c_1 - c_m = {gate1.mean:+.4f} "
                  f"[{gate1.lo:+.4f}, {gate1.hi:+.4f}] "
                  f"{'EXCLUDES' if gate1.excludes_zero else 'covers'} zero")
            print(f"      first-neighbour moment {k.c_first.mean():.4f}, "
                  f"last {k.c_last.mean():.4f}, averaged over targets")
            if not gate1.excludes_zero:
                print("      gate 1 FAILS: exchangeability is not materially wrong here and the")
                print("      relaxation is moot. No outcome reported at this m, per the plan.")
                continue

            sc = cluster_bootstrap_mean(k.gain_scale, k.receptor_class, DRAWS, SEED)
            sh = cluster_bootstrap_mean(k.gain_shape, k.receptor_class, DRAWS, SEED)
            print(f"    gain_scale  = {sc.mean:+.4f} [{sc.lo:+.4f}, {sc.hi:+.4f}] "
                  f"{'excludes' if sc.excludes_zero else 'covers'} zero   "
                  f"(median {k.gain_scale.median():+.4f})")
            print(f"    gain_shape  = {sh.mean:+.4f} [{sh.lo:+.4f}, {sh.hi:+.4f}] "
                  f"{'excludes' if sh.excludes_zero else 'covers'} zero   "
                  f"(median {k.gain_shape.median():+.4f})")
            print(f"    beta: median {k.beta.median():.4f}, median |beta-1| "
                  f"{k.beta_dev.median():.4f}  "
                  f"(prediction 2 holds iff below 0.10: "
                  f"{'HOLDS' if k.beta_dev.median() < 0.10 else 'FAILS'})")
            print(f"    optimal weight sum: median {k.w_rank_sum.median():.4f}; "
                  f"weight spread max-min: median {k.w_rank_spread.median():.4f}")

            if m == M_PRIMARY and "beats_surrogate" in k:
                b = k.beats_surrogate.fillna(False)
                print(f"    against the rank-shuffled surrogate: measured shape gain beats the "
                      f"95th percentile on {int(b.sum())}/{len(k)} targets")
                print(f"      surrogate mean shape gain {k.surr_mean.mean():+.4f}, "
                      f"mean 95th percentile {k.surr_p95.mean():+.4f}")

                survives = (k.gain_shape.median() < 0.01) and (sh.hi < 0.05)
                downgraded = sh.excludes_zero and (k.gain_shape.median() > 0.01) \
                    and (b.sum() > len(k) / 2)
                print()
                print(f"    OUTCOME at the primary m, by the frozen rule:")
                print(f"      claim 6 survives as stated: {survives}")
                print(f"      claim 6 is downgraded:      {downgraded}")
                if not survives and not downgraded:
                    print("      neither: NOT ESTABLISHED, and the report says not established")
                    print("      rather than claiming a null from a failure to reject.")

            if coll == "ChEMBL-40" and m == M_PRIMARY and len(k) > 3:
                p3 = cluster_bootstrap_spearman(k.beta_dev, k.rel_mse_minus_one_minus_r,
                                                k.receptor_class, DRAWS, SEED)
                print()
                print(f"    PREDICTION 3, beta near one collapses the identity to a one-statistic")
                print(f"      floor: Spearman(|beta-1|, |rel_mse - (1-r)|) = {p3.rho:+.4f} "
                      f"[{p3.lo:+.4f}, {p3.hi:+.4f}] "
                      f"{'EXCLUDES' if p3.excludes_zero else 'covers'} zero")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

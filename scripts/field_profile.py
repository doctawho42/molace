"""Increment 10: the decay contrast belongs to the field, not the graph.

Pre-registration: prereg/increment10_field_profile.yaml, blob
5de3f710cb9a89aef11498965b1e3c79bf00cbc6.

Four designs tried to move lambda = rho_nn / r by changing the graph and four design gates refused
them. The arithmetic says they had to: on a stride-d circulant of degree 2 with an exponential field,
r = (1-nu) c(d) and rho_nn = (1-nu) c(d)^2, so lambda = c(d) is pinned to r by the FIELD. A
squared-exponential field gives c(2d) = c(d)^4 and therefore lambda = c(d)^3, so the two families
separate lambda at matched r by construction.

The same arithmetic predicts, quantitatively, that r^2 / rho_nn overstates the signal fraction on a
squared-exponential field by exactly 1 / c(d)^2 at k = 2. That prediction is checked before the gate.
"""
from __future__ import annotations

import sys
import time
import warnings

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

from molace.measures.assortativity import target_assortativity
from molace.measures.neighbourhood import shared_neighbour_correlation

PREREG = "5de3f710cb9a89aef11498965b1e3c79bf00cbc6"
N, DRAWS = 600, 10000
LENGTHS = (2, 4, 8, 16, 32)
NOISES = (0.1, 0.2, 0.3, 0.5, 0.7)
DEGREES = (2, 10)
STRIDES = (1, 3)
SEEDS = (0, 1, 2)
FAMILIES = ("exponential", "squared_exponential")
RHO_T_MIN, MIN_SHARE = 10.0, 0.25


def profile(family: str, delta: np.ndarray, length: float) -> np.ndarray:
    if family == "exponential":
        return np.exp(-delta / length)
    return np.exp(-((delta / length) ** 2))


def ring_field(family: str, length: float, rng: np.random.Generator) -> np.ndarray:
    j = np.arange(N)
    c0 = profile(family, np.minimum(j, N - j).astype(float), length)
    spec = np.maximum(np.fft.fft(c0).real, 0.0)
    w = rng.normal(size=N) + 1j * rng.normal(size=N)
    x = np.real(np.fft.ifft(np.sqrt(spec) * w))
    return (x - x.mean()) / x.std()


def loo_floor_relative_mse(g: nx.Graph, y: np.ndarray) -> float:
    errs = np.empty(len(y))
    for v in sorted(g.nodes()):
        nb = np.fromiter(g.neighbors(v), dtype=np.int64)
        errs[v] = (y[v] - y[nb].mean()) ** 2
    return float(errs.mean() / y.var())


def main() -> int:
    print(f"pre-registration blob: {PREREG}")
    print(f"{len(FAMILIES)*len(LENGTHS)*len(NOISES)*len(DEGREES)*len(STRIDES)*len(SEEDS)} datasets, "
          f"n = {N}, two field families, nothing trained")
    print()
    rows, t0 = [], time.time()
    for k in DEGREES:
        for d in STRIDES:
            offsets = [d * j for j in range(1, k // 2 + 1)]
            g = nx.circulant_graph(N, offsets)
            assert {dd for _, dd in g.degree()} == {k}
            for family in FAMILIES:
                for length in LENGTHS:
                    cd = float(profile(family, np.array([float(d)]), length)[0])
                    for nu in NOISES:
                        for seed in SEEDS:
                            rng = np.random.default_rng(
                                (hash(family) % 9973, k, d, length, int(nu * 100), seed))
                            sig = ring_field(family, length, rng) * np.sqrt(1.0 - nu)
                            noi = rng.normal(size=N)
                            noi = (noi - noi.mean()) / noi.std() * np.sqrt(nu)
                            y = sig + noi
                            nu_true = float(noi.var() / y.var())
                            r = target_assortativity(g, y).value
                            rho = shared_neighbour_correlation(g, y).value
                            meas = loo_floor_relative_mse(g, y)
                            pred = 1.0 + 1.0 / k + (k - 1.0) / k * rho - 2.0 * r
                            sig_hat = r * r / rho if rho != 0 else np.nan
                            pairs = k * (k - 1) // 2 * N
                            rows.append({
                                "family": family, "k": k, "stride": d, "length": length,
                                "construction": f"{family}", "cell": f"{k}_{d}_{length}_{nu}_{seed}",
                                "nu_target": nu, "nu_true": nu_true, "seed": seed, "c_d": cd,
                                "assortativity": r, "rho_shared_neighbour": rho,
                                "rho_t": rho * np.sqrt(pairs),
                                "lambda": rho / r if r > 0 else np.nan,
                                "signal_true": 1.0 - nu_true, "signal_hat": sig_hat,
                                "measured_rel_mse": meas, "predicted_rel_mse": pred,
                                "identity_error": pred - meas,
                            })
            print(f"  k={k:2d} d={d}  {len(rows):4d} datasets  [{time.time()-t0:.0f}s]", flush=True)

    t = pd.DataFrame(rows)
    t.to_csv("results/field_profile.csv", index=False)
    kept = t[t.rho_t.abs() > RHO_T_MIN].copy()
    print()
    print(f"  conditioning guard, declared in the plan: kept {len(kept)} of {len(t)}, "
          f"excluded {len(t)-len(kept)} where rho_nn is within {RHO_T_MIN:g} standard errors of zero")
    print(f"  identity's max |error| over all {len(t)}: {t.identity_error.abs().max():.2e} "
          f"(exact by proof on regular graphs)")

    print()
    print("=" * 100)
    print("PREDICTION 1: on a squared-exponential field the estimator overstates by exactly 1/c(d)^2")
    print("=" * 100)
    k2 = kept[kept.k == 2]
    for family, s in k2.groupby("family"):
        expected = s.signal_true * (1.0 / s.c_d ** 2 if family == "squared_exponential" else 1.0)
        ratio = s.signal_hat / expected
        print(f"  {family:20s} n={len(s):3d}  median ratio to prediction {ratio.median():.4f}  "
              f"IQR [{ratio.quantile(.25):.4f}, {ratio.quantile(.75):.4f}]")
        if family == "squared_exponential":
            naive = (s.signal_hat / s.signal_true).median()
            print(f"  {'':20s}      median ratio if the geometric model were assumed: {naive:.4f}")
            holds = bool(0.9 <= ratio.median() <= 1.1)
    print(f"  PREDICTION 1 holds (median in [0.9, 1.1]): {holds}")
    if not holds:
        print()
        print("  THE ARITHMETIC BEHIND THIS PLAN IS WRONG SOMEWHERE. That leads the report, ahead of")
        print("  everything else, and the gate and outcome below are not interpreted.")

    print()
    print("=" * 100)
    print("THE DESIGN GATE, evaluated before the outcome")
    print("=" * 100)
    gm = kept["lambda"].mean()
    tot = float(((kept["lambda"] - gm) ** 2).sum())
    share = {key: sum(len(g) * (g["lambda"].mean() - gm) ** 2 for _, g in kept.groupby(key)) / tot
             for key in ("family", "k", "length", "nu_target")}
    for family, s in kept.groupby("family"):
        print(f"  {family:20s} mean lambda {s['lambda'].mean():.4f}  sd {s['lambda'].std():.4f}")
    for key, v in share.items():
        print(f"  lambda variance between {key:11s}: {100*v:5.1f} %")
    gate = bool(share["family"] >= MIN_SHARE)
    print(f"  GATE (family needs >= {MIN_SHARE:.0%}): {gate}   "
          f"(four previous designs managed 4.2 %, 0.0 %, 14.1 %, 10.8 %)")
    if not gate:
        print()
        print("  GATE FAILED. The outcome is not reported. Five refusals, and the plan forbids a sixth")
        print("  re-specification: the refusals themselves are the finding.")
        return 0

    print()
    print("=" * 100)
    print("THE OUTCOME: ranking power of assortativity alone, by half of the per-hop decay")
    print("=" * 100)
    med = kept["lambda"].median()
    rank = {}
    for name, s in (("low", kept[kept["lambda"] <= med]), ("high", kept[kept["lambda"] > med])):
        rank[name] = stats.spearmanr(1 - s.assortativity, s.measured_rel_mse).statistic
        print(f"  {name:4s} lambda half: n={len(s):3d} mean lambda {s['lambda'].mean():.3f}  "
              f"Spearman(1-r, measured error) {rank[name]:+.3f}")
    obs = rank["low"] - rank["high"]
    cells = kept.cell.unique()
    rng = np.random.default_rng(0)
    draws = []
    for _ in range(DRAWS):
        s = kept[kept.cell.isin(set(rng.choice(cells, len(cells), replace=True)))]
        a, b = s[s["lambda"] <= med], s[s["lambda"] > med]
        if len(a) > 20 and len(b) > 20:
            v = (stats.spearmanr(1 - a.assortativity, a.measured_rel_mse).statistic
                 - stats.spearmanr(1 - b.assortativity, b.measured_rel_mse).statistic)
            if np.isfinite(v):
                draws.append(v)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    print(f"  low minus high: {obs:+.4f}  [{lo:+.4f}, {hi:+.4f}]  "
          f"{'EXCLUDES 0' if lo > 0 or hi < 0 else 'covers 0'}  half-width {(hi-lo)/2:.4f}")
    print()
    print("  within each degree, because degree moves lambda and the 1/k term together:")
    for k in sorted(kept.k.unique()):
        s = kept[kept.k == k]
        m = s["lambda"].median()
        a, b = s[s["lambda"] <= m], s[s["lambda"] > m]
        ra = stats.spearmanr(1 - a.assortativity, a.measured_rel_mse).statistic
        rb = stats.spearmanr(1 - b.assortativity, b.measured_rel_mse).statistic
        print(f"    k={k:2d}  n={len(s):3d}  low lambda {ra:+.3f}  high lambda {rb:+.3f}  "
              f"difference {ra-rb:+.3f}")
    print()
    if obs < 0 and (lo > 0 or hi < 0):
        print("  SUPPORTED. Assortativity alone ranks the floor's error worse where the per-hop decay")
        print("  is lower, on a contrast that the field sets rather than the graph. The identity's own")
        print("  side is exact here by proof and is not offered as a contrast.")
    else:
        print("  NOT SUPPORTED as stated. The interval covers zero or the sign is wrong, and the width")
        print("  is above. The project does not read that as no effect.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

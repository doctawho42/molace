"""Increment 9: synthetic graphs with a controlled decay, to ask what real data could not.

Pre-registration: prereg/increment9_synthetic.yaml, blob
85f1e3bb67f7aba04753a06b424b7d6716fcc328.

Three real-data designs could not test a structural claim because between-target variance was 77 % to
89 % of everything varied. And increment 6's estimator of the signal fraction, 1 - nu = r^2 / rho_nn,
missed its validity threshold on held-out data with no way to tell whether the decay model or the data
was at fault: real datasets never reveal their true noise fraction.

Here they do. Nodes sit on a ring, the signal is a stationary Gaussian field with covariance
exp(-delta / L) in ring distance, and independent noise takes a prescribed share of the variance. The
graph is a circulant joining node i to i +/- j*d for j = 1 .. k/2, so every edge spans a ring distance
that is a multiple of d. At k = 2 every edge spans exactly d and every shared-neighbour pair exactly
2d, which makes the geometric model EXACT and r^2 / rho_nn equal to 1 - nu whatever d and L are. At
larger k an edge spans several distances at once and the model must degrade. How fast is the question.

These circulants are k-regular, so the floor identity is exact here by the proof increment 5 pinned in
tests. That is stated, not discovered, and the report says so.
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

PREREG = "85f1e3bb67f7aba04753a06b424b7d6716fcc328"
N = 600
LENGTHS = (2, 4, 8, 16, 32)
NOISES = (0.1, 0.2, 0.3, 0.5, 0.7)
DEGREES = (2, 4, 10, 20)
STRIDES = (1, 3)
SEEDS = (0, 1, 2)
DRAWS = 10000
EXACT_CORNER_TOL = 0.05
MIN_CONSTRUCTION_SHARE = 0.25


def ring_field(n: int, length: float, rng: np.random.Generator) -> np.ndarray:
    """A stationary Gaussian field on a ring with covariance exp(-ring distance / length)."""
    j = np.arange(n)
    c0 = np.exp(-np.minimum(j, n - j) / length)
    spec = np.maximum(np.fft.fft(c0).real, 0.0)          # circulant spectrum, clipped
    w = rng.normal(size=n) + 1j * rng.normal(size=n)
    x = np.real(np.fft.ifft(np.sqrt(spec) * w))
    return (x - x.mean()) / x.std()


def loo_floor_relative_mse(g: nx.Graph, y: np.ndarray) -> float:
    """Mean over nodes of (label minus the mean of its graph neighbours) squared, over Var(y)."""
    errs = np.empty(len(y))
    for v in sorted(g.nodes()):
        nb = np.fromiter(g.neighbors(v), dtype=np.int64)
        errs[v] = (y[v] - y[nb].mean()) ** 2
    return float(errs.mean() / y.var())


def main() -> int:
    print(f"pre-registration blob: {PREREG}")
    total = len(LENGTHS) * len(NOISES) * len(DEGREES) * len(STRIDES) * len(SEEDS)
    print(f"{total} synthetic datasets, n = {N} each, nothing trained")
    print()

    rows = []
    t0 = time.time()
    for k in DEGREES:
        for d in STRIDES:
            offsets = [d * j for j in range(1, k // 2 + 1)]
            if max(offsets) >= N // 2:
                print(f"  SKIP k={k} d={d}: offset {max(offsets)} reaches the ring's far side")
                continue
            g = nx.circulant_graph(N, offsets)
            deg = {dd for _, dd in g.degree()}
            assert deg == {k}, f"k={k} d={d} gave degrees {deg}"
            for length in LENGTHS:
                for nu in NOISES:
                    for seed in SEEDS:
                        rng = np.random.default_rng((k, d, length, int(nu * 100), seed))
                        sig = ring_field(N, length, rng) * np.sqrt(1.0 - nu)
                        noi = rng.normal(size=N) * np.sqrt(nu)
                        noi = (noi - noi.mean()) / noi.std() * np.sqrt(nu)
                        y = sig + noi
                        nu_true = float(noi.var() / y.var())
                        r = target_assortativity(g, y).value
                        rho = shared_neighbour_correlation(g, y).value
                        meas = loo_floor_relative_mse(g, y)
                        pred = 1.0 + 1.0 / k + (k - 1.0) / k * rho - 2.0 * r
                        sig_hat = r * r / rho if rho > 0 else np.nan
                        rows.append({
                            "k": k, "stride": d, "construction": f"k{k}d{d}",
                            "length": length, "nu_target": nu, "nu_true": nu_true, "seed": seed,
                            "cell": f"L{length}_nu{nu}_d{d}_s{seed}",
                            "assortativity": r, "rho_shared_neighbour": rho,
                            "lambda": rho / r if r > 0 else np.nan,
                            "signal_true": 1.0 - nu_true, "signal_hat": sig_hat,
                            "recovery_error": sig_hat - (1.0 - nu_true),
                            "measured_rel_mse": meas, "predicted_rel_mse": pred,
                            "identity_error": pred - meas,
                            "skill_floor": 1.0 - np.sqrt(meas),
                            "ceiling_hat": 1.0 - np.sqrt(max(1.0 - sig_hat, 0.0)),
                            "ceiling_true": 1.0 - np.sqrt(nu_true),
                        })
            print(f"  k={k:2d} d={d}  {len(rows):4d} datasets so far  [{time.time()-t0:.0f}s]",
                  flush=True)

    t = pd.DataFrame(rows)
    t.to_csv("results/synthetic_decay.csv", index=False)

    print()
    print("=" * 98)
    print("reported unconditionally: the identity on regular graphs, which must be exact")
    print("=" * 98)
    print(f"  max |identity error| over {len(t)} datasets: {t.identity_error.abs().max():.2e}")
    print("  (exact by the proof increment 5 pinned in tests, not an empirical finding)")

    print()
    print("=" * 98)
    print("QUESTION 1: does r^2 / rho_nn recover the TRUE signal fraction?")
    print("=" * 98)
    print(f"  {'k':>3s} {'mean |recovery err|':>20s} {'mean signed':>12s} {'max |err|':>10s} "
          f"{'mean lambda':>12s} {'ceiling_hat - ceiling_true':>27s}")
    per_k = {}
    for k in sorted(t.k.unique()):
        s = t[t.k == k]
        per_k[k] = s
        print(f"  {k:3d} {s.recovery_error.abs().mean():20.4f} {s.recovery_error.mean():12.4f} "
              f"{s.recovery_error.abs().max():10.4f} {s['lambda'].mean():12.4f} "
              f"{(s.ceiling_hat - s.ceiling_true).mean():27.4f}")

    corner = per_k[min(per_k)].recovery_error.abs().mean()
    corner_ok = bool(corner < EXACT_CORNER_TOL)
    ks = sorted(per_k)
    means = np.array([per_k[k].recovery_error.abs().mean() for k in ks])
    obs = float(stats.spearmanr(ks, means).statistic)
    cells = t.cell.unique()
    rng = np.random.default_rng(0)
    draws = []
    for _ in range(DRAWS):
        pick = rng.choice(cells, len(cells), replace=True)
        s = t[t.cell.isin(set(pick))]
        mm = np.array([s[s.k == k].recovery_error.abs().mean() for k in ks])
        if np.all(np.isfinite(mm)):
            draws.append(stats.spearmanr(ks, mm).statistic)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    grows = bool((lo > 0 or hi < 0) and obs > 0)
    print()
    print(f"  exact corner, k = {min(per_k)}: mean |recovery error| {corner:.4f} "
          f"(threshold {EXACT_CORNER_TOL})   {corner_ok}")
    print(f"  recovery error grows with k: Spearman {obs:+.3f} [{lo:+.3f}, {hi:+.3f}]   {grows}")
    print()
    if corner_ok and grows:
        print("  THE ESTIMATOR IS SOUND WHERE ITS MODEL IS EXACT, AND ITS ERROR IS MIXED HOP LENGTHS.")
        print("  That is the diagnosis increment 6 could not reach: its two held-out violations are")
        print("  the approximation cost of edges that span several ring distances at once, not a")
        print("  broken estimator.")
    elif not corner_ok:
        print("  THE ESTIMATOR FAILS WHERE ITS MODEL IS EXACT. That is a larger finding than")
        print("  increment 6's missed threshold and it leads the report.")
    else:
        print("  THE ESTIMATOR IS SOUND IN THE EXACT CORNER BUT ITS ERROR DOES NOT GROW WITH k, so")
        print("  mixed hop lengths are NOT why increment 6 missed its threshold, and the cause stays")
        print("  unknown rather than being given a second story.")

    print()
    print("=" * 98)
    print("QUESTION 2: the design gate first")
    print("=" * 98)
    gm = t["lambda"].mean()
    tot = float(((t["lambda"] - gm) ** 2).sum())
    share = {key: sum(len(g) * (g["lambda"].mean() - gm) ** 2 for _, g in t.groupby(key)) / tot
             for key in ("construction", "length", "nu_target")}
    for key, v in share.items():
        print(f"  lambda variance between {key:13s}: {100*v:5.1f} %")
    gate = bool(share["construction"] >= MIN_CONSTRUCTION_SHARE)
    print(f"  GATE (construction needs >= {MIN_CONSTRUCTION_SHARE:.0%}): {gate}")

    if not gate:
        print("  GATE FAILED, outcome not reported, on the same grounds as increments 7 and 8.")
        return 0

    med = t["lambda"].median()
    halves = {"low": t[t["lambda"] <= med], "high": t[t["lambda"] > med]}
    print()
    print("=" * 98)
    print("QUESTION 2: ranking power of assortativity alone, by half of the per-hop decay")
    print("=" * 98)
    rank = {}
    for name, s in halves.items():
        rank[name] = stats.spearmanr(1 - s.assortativity, s.measured_rel_mse).statistic
        print(f"  {name:4s} lam half: n={len(s):4d} mean lambda {s['lambda'].mean():.3f}  "
              f"Spearman(1-r, measured error) {rank[name]:+.3f}")
    obs2 = rank["low"] - rank["high"]
    draws2 = []
    for _ in range(DRAWS):
        pick = set(rng.choice(cells, len(cells), replace=True))
        s = t[t.cell.isin(pick)]
        a, b = s[s["lambda"] <= med], s[s["lambda"] > med]
        if len(a) > 20 and len(b) > 20:
            v = (stats.spearmanr(1 - a.assortativity, a.measured_rel_mse).statistic
                 - stats.spearmanr(1 - b.assortativity, b.measured_rel_mse).statistic)
            if np.isfinite(v):
                draws2.append(v)
    lo2, hi2 = np.percentile(draws2, [2.5, 97.5])
    print(f"  low minus high: {obs2:+.4f}  [{lo2:+.4f}, {hi2:+.4f}]  "
          f"{'EXCLUDES 0' if lo2 > 0 or hi2 < 0 else 'covers 0'}  half-width {(hi2-lo2)/2:.4f}")
    print()
    if obs2 < 0 and (lo2 > 0 or hi2 < 0):
        print("  SUPPORTED: assortativity alone ranks the floor's error worse where the per-hop decay")
        print("  is lower. The identity's own side is exact here by construction and carries no")
        print("  information, which is why only this half is reported as a test.")
    else:
        print(f"  NOT SUPPORTED as stated: the interval covers zero or the sign is wrong.")
    print()
    print("  NOT pre-registered, because k moves lambda AND the 1/k term of the identity at once:")
    print("  the same contrast within each k, where that confound is absent.")
    for k in ks:
        s = per_k[k]
        m = s["lambda"].median()
        a, b = s[s["lambda"] <= m], s[s["lambda"] > m]
        ra = stats.spearmanr(1 - a.assortativity, a.measured_rel_mse).statistic
        rb = stats.spearmanr(1 - b.assortativity, b.measured_rel_mse).statistic
        print(f"    k={k:2d}  low lam {ra:+.3f}   high lam {rb:+.3f}   difference {ra-rb:+.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Increment 15: the size dependence of section 4, measured under a frozen plan.

Pre-registration: prereg/increment15_size_dependence.yaml, blob
3d941919c5aeb34e2749cc820e8e3ae55d5ff725.

Section 4 of docs/paper/skeleton.md says modelability is a property of a dataset AND its size, on a
four-row table whose f = 1 row reproduces from results/modi_baseline.csv and whose other three rows
have no committed artefact at all. This script measures all four fractions under the plan, and
measures the decomposition the plan requires instead of declaring the confound: arm A subsamples,
arm B holds the point set and dilutes only which ranks each molecule reads, and arm A-prime is the
bookkeeping quantity that makes

    D_A - D_B  =  (D_A - D_A')  +  (D_A' - D_B)

hold exactly per (target, fraction, chain), with the first term the whole variance-normaliser channel.

Three things in here exist because this plan's own draft was wrong about them, and each is
load-bearing:

  * The subsample index set is numpy.sort(perm[:n_f]) AT EVERY FRACTION. Both neighbour selectors in
    this codebase break similarity ties on the ascending index, and a quarter to a third of rows have
    their 10th and 11th similarity exactly equal, so an unsorted index set moves every gated column by
    up to 1.13e-2. Sorted, f = 1 reproduces the committed row at exactly 0.0.
  * Arm B draws an EXACT uniform subset without replacement, not a Bernoulli thinning. The draft chose
    Bernoulli on a cost argument that was simply wrong: Generator.choice(replace=False) is O(n*n_f)
    and per-molecule Bernoulli is O(n^2), so the exact law is cheaper on the worst-case target.
  * Every sum of squares is accumulated into a full (n,) vector and summed ONCE with numpy.sum.
    Chunked accumulation would not be bit-identical to the committed row, and gate 1 asks for 1e-12.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from molace.analysis.checkpoint import load_checkpoint, save_checkpoint
from molace.analysis.correlate import cluster_groups, cluster_resample
from molace.data import moleculeace as ma
from molace.graphs import knn
from molace.graphs.fingerprints import ecfp4, tanimoto_matrix
from molace.measures.assortativity import rank_assortativity, target_assortativity
from molace.measures.neighbourhood import shared_neighbour_correlation

PREREG = "3d941919c5aeb34e2749cc820e8e3ae55d5ff725"
CHEMBL = Path("data/raw/chembl_targets")
OUT = Path("results/size_dependence.csv")
PART = Path("results/size_dependence_partial.csv")

K = 10
FRACTIONS = (0.125, 0.25, 0.5, 1.0)
CHAINS = (0, 1, 2, 3, 4)
GRID = (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75)
DRAWS, SEED = 10000, 0
TOL = 1e-12
GATE_1B_FRACTION, GATE_1B_CHAIN = 0.5, 0
RANK_TARGETS = ("CHEMBL217_Ki", "CHEMBL4805_Ki")        # largest and smallest, per the plan
KEYS = ["collection", "dataset", "fraction", "chain", "n_f"]


# ----------------------------------------------------------------------------- the measured pieces

def ordering(sim: np.ndarray) -> np.ndarray:
    """Each row's neighbours by descending similarity, ties on ascending ORIGINAL index.

    Identical to scripts/modi_baseline.py:58 -- float32 throughout, the diagonal sent to -inf so a
    molecule is never its own neighbour, and a STABLE argsort so the tie-break is the index order and
    not whatever the sort happens to do. 26.8 % of rows on the largest target and 35.5 % on the
    smallest have their 10th and 11th similarity exactly equal, so this is a first-order choice.

    Returned as int32. At n = 8205 an int64 version would cost 538 MB beside the 269 MB similarity
    matrix; int32 indexes 2^31 molecules, eight thousand times what is here.
    """
    s = sim.copy()
    np.fill_diagonal(s, -np.inf)
    return np.argsort(-s, axis=1, kind="stable").astype(np.int32)


def floor_parts(order: np.ndarray, y: np.ndarray, k: int = K) -> tuple[float, float]:
    """The kNN mean's sum of squares and the label's, kept APART.

    1 - sse/sst is MODI_q2. They come back separately because arm A-prime needs this numerator over
    arm B's denominator, and a ratio computed first cannot be taken apart afterwards.
    """
    yhat = y[order[:, :k]].mean(axis=1)
    return float(np.sum((y - yhat) ** 2)), float(np.sum((y - y.mean()) ** 2))


def graph_stats(sim: np.ndarray, y: np.ndarray) -> dict:
    g = knn.knn_graph(sim, K)
    r = target_assortativity(g, y)
    rho = shared_neighbour_correlation(g, y)
    deg = np.array([d for _, d in g.degree()], dtype=float)
    return {
        "assortativity": r.value,
        "rank_assortativity": rank_assortativity(g, y).value,
        "rho_shared_neighbour": rho.value,
        "identity": 1 + 1 / K + (K - 1) / K * rho.value - 2 * r.value,
        "coverage": r.coverage,
        "n_pairs": rho.n_pairs,
        "mean_degree": float(deg.mean()),
    }


def arm_b(order: np.ndarray, y: np.ndarray, n_f: int, seed: tuple[int, ...],
          want_ranks: bool = False) -> tuple[float, np.ndarray | None]:
    """The rank-dilution null: the same point set, read at the ranks a subsample would leave.

    For each molecule independently, an EXACT uniform subset of exactly n_f - 1 of the other n - 1
    molecules, so the rank of the j-th survivor follows the negative hypergeometric law a real
    subsample induces, with mean j*n/n_f. Everything else is held at full n: the label mean, the
    variance normaliser, and the set of molecules scored.

    Independence across molecules is the point, not a shortcut. One global survival mask would
    reproduce arm A rather than isolate the rank channel from it.
    """
    n = len(y)
    rng = np.random.default_rng(seed)
    yhat = np.empty(n, dtype=float)
    ranks = np.zeros(K, dtype=float) if want_ranks else None
    mask = np.zeros(n, dtype=bool)
    for i in range(n):
        keep = rng.choice(n - 1, size=n_f - 1, replace=False)
        keep = keep + (keep >= i)                       # index the others, skipping i itself
        mask[:] = False
        mask[keep] = True
        row = order[i]
        pos = np.flatnonzero(mask[row])[:K]
        yhat[i] = y[row[pos]].mean()
        if want_ranks:
            ranks += pos + 1.0                          # 1-based rank within the full ordering
    return float(np.sum((y - yhat) ** 2)), (ranks / n if want_ranks else None)


def modi_binary(order: np.ndarray, cls: np.ndarray) -> tuple[float, dict]:
    """The 2014 index, class-weighted 1-NN agreement, with the empty-class hole closed.

    scripts/modi_baseline.py iterates np.unique(cls), so a class absent from a subsample simply does
    not appear and the index silently becomes the mean over one class. On the smallest MoleculeACE
    target the cliff class vanishes with probability 2.4e-04 per draw. That script is increment 11's
    artefact behind a frozen blob and is not edited; the guard lives here, and gate 1 covers this
    reimplementation at f = 1.
    """
    nn = order[:, 0]
    counts = {c: int((cls == c).sum()) for c in (0, 1)}
    if min(counts.values()) == 0:
        return float("nan"), counts
    return float(np.mean([float((cls[nn][cls == c] == c).mean()) for c in (0, 1)])), counts


# ----------------------------------------------------------------------------- the run

def collections():
    meta = pd.read_csv(CHEMBL / "selected.csv")
    for i, m in enumerate(meta.itertuples()):
        d = pd.read_csv(CHEMBL / f"{m.dataset}.csv")
        yield "ChEMBL-40", i, m.dataset, d.smiles.tolist(), d.y.to_numpy(dtype=float), None
    for i, name in enumerate(ma.DATASETS):
        d = ma.load_target(name)
        yield ("MoleculeACE-30", i, name, d["smiles"].tolist(), d["y"].to_numpy(dtype=float),
               d["cliff_mol"].to_numpy(dtype=int))


def measure_target(coll, tix, name, smiles, y, cls, done) -> list[dict]:
    n = len(y)
    todo = [(f, c) for f in FRACTIONS for c in (CHAINS if f < 1.0 else CHAINS[:1])
            if (coll, name, f, c, int(np.floor(f * n))) not in done]
    if not todo:
        return []

    sim = tanimoto_matrix(ecfp4(smiles))
    order_full = ordering(sim)
    _, sst_full = floor_parts(order_full, y)
    rows = []

    for f, chain in todo:
        n_f = int(np.floor(f * n))
        perm = np.random.default_rng((15, tix, chain)).permutation(n)
        S = np.sort(perm[:n_f])                         # SORTED: see the module docstring
        if f == 1.0:
            order_s, y_s, sim_s = order_full, y, sim
        else:
            sim_s = sim[np.ix_(S, S)]
            order_s, y_s = ordering(sim_s), y[S]

        sse_a, sst_a = floor_parts(order_s, y_s)
        row = {
            "collection": coll, "dataset": name, "target_index": tix, "cluster": name,
            "fraction": f, "chain": chain, "n": n, "n_f": n_f,
            "sse_a": sse_a, "sst_a": sst_a,
            "floor_rel_mse_a": sse_a / sst_a,
            "modi_q2": 1.0 - sse_a / sst_a,
            # arm A-prime: arm A's numerator over arm B's denominator, so the normaliser channel
            # separates exactly instead of hiding inside a claim about chemistry.
            "floor_rel_mse_aprime": (sse_a / n_f) / (sst_full / n),
        }
        if cls is None:
            row.update(graph_stats(sim_s, y_s))
            sse_b, ranks = arm_b(order_full, y, n_f, (15, tix, chain, 1),
                                 want_ranks=name in RANK_TARGETS)
            row["sse_b"] = sse_b
            row["floor_rel_mse_b"] = sse_b / sst_full
            if ranks is not None:
                for j, v in enumerate(ranks, start=1):
                    row[f"rank_j{j}"] = v
                    row[f"rank_j{j}_expected"] = j * n / n_f
        else:
            mb, counts = modi_binary(order_s, cls[S])
            row["modi_binary"] = mb
            row["class_0_kept"], row["class_1_kept"] = counts[0], counts[1]

        if cls is None and f == GATE_1B_FRACTION and chain == GATE_1B_CHAIN:
            sim_r = tanimoto_matrix(ecfp4([smiles[i] for i in S]))
            sse_r, sst_r = floor_parts(ordering(sim_r), y_s)
            gr = graph_stats(sim_r, y_s)
            row["g1b_d_modi_q2"] = abs((1 - sse_r / sst_r) - row["modi_q2"])
            for c in ("assortativity", "rank_assortativity", "rho_shared_neighbour"):
                row[f"g1b_d_{c}"] = abs(gr[c] - row[c])

        rows.append(row)
        tail = (f"  relMSE_B={row['floor_rel_mse_b']:.4f}" if cls is None
                else f"  MODI={row['modi_binary']:.4f}")
        print(f"  {coll:14s} {name:22s} f={f:<6} chain={chain}  n_f={n_f:5d}  "
              f"relMSE_A={row['floor_rel_mse_a']:.4f}{tail}", file=sys.stderr, flush=True)
    return rows


def main() -> int:
    print(f"pre-registration blob: {PREREG}\n")
    rows = load_checkpoint(PART, KEYS)
    done = {(r["collection"], r["dataset"], float(r["fraction"]), int(r["chain"]), int(r["n_f"]))
            for r in rows}
    print(f"resuming with {len(rows)} row(s) already measured\n", file=sys.stderr)

    for coll, tix, name, smiles, y, cls in collections():
        new = measure_target(coll, tix, name, smiles, y, cls, done)
        if new:
            rows.extend(new)
            save_checkpoint(PART, rows)

    t = pd.DataFrame(rows)
    t.to_csv(OUT, index=False)
    report(t)
    return 0


# ----------------------------------------------------------------------------- the report

def boot_mean(x, clusters):
    groups = cluster_groups(clusters)
    rng = np.random.default_rng(SEED)
    x = np.asarray(x, dtype=float)
    d = np.array([x[cluster_resample(groups, rng)].mean() for _ in range(DRAWS)])
    return float(x.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def boot_ratio(num, den, clusters):
    """An interval on a RATIO OF TWO MEANS, recomputed inside each draw.

    Declared in the plan before measurement because this has no default: bootstrapping the two means
    separately and dividing is a different statistic, and the two arms are differently noised.
    """
    groups = cluster_groups(clusters)
    rng = np.random.default_rng(SEED)
    num, den = np.asarray(num, float), np.asarray(den, float)
    d = np.empty(DRAWS)
    for b in range(DRAWS):
        i = cluster_resample(groups, rng)
        d[b] = num[i].mean() / den[i].mean()
    return float(num.mean() / den.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def per_target(t, col):
    """Chain means per target: the plan's aggregation, fixed before measurement."""
    return t.groupby(["dataset", "fraction"], sort=False)[col].mean().unstack("fraction")


def report(t: pd.DataFrame) -> None:
    ch = t[t.collection == "ChEMBL-40"]
    line = "=" * 100

    print(line); print("GATE 1. DOES THE PIPELINE REPRODUCE THE COMMITTED ROW AT f = 1"); print(line)
    base = pd.read_csv("results/modi_baseline.csv", float_precision="round_trip")
    base = base[base.collection == "ChEMBL-40"].set_index("dataset")
    f1 = ch[ch.fraction == 1.0].set_index("dataset")
    cols = ["assortativity", "rank_assortativity", "rho_shared_neighbour", "modi_q2"]
    worst = {c: float((f1[c] - base.loc[f1.index, c]).abs().max()) for c in cols}
    for c, v in worst.items():
        print(f"  {c:24s} max |difference| over {len(f1)} targets = {v:.3e}")
    g1 = len(f1) == 40 and max(worst.values()) <= TOL
    print(f"\n  GATE 1: {'PASS' if g1 else 'FAIL'} at tolerance {TOL:.0e}")
    if not g1:
        print("\n  The plan stops the increment here and the discrepancy IS the report.")
        return

    print(); print(line)
    print("GATE 1B. THE SUBSAMPLE ROUTE, AGAINST A ROUTE SHARING NO INDEXING CODE"); print(line)
    gcols = [c for c in ch.columns if c.startswith("g1b_d_")]
    sub = ch.dropna(subset=gcols)
    worst_1b = {c: float(sub[c].abs().max()) for c in gcols}
    for c, v in worst_1b.items():
        print(f"  {c:28s} max = {v:.3e}")
    g1b = len(sub) == 40 and max(worst_1b.values()) <= TOL
    print(f"\n  checked on {len(sub)} targets at f = {GATE_1B_FRACTION}, chain {GATE_1B_CHAIN}")
    print(f"  GATE 1B: {'PASS' if g1b else 'FAIL'} at tolerance {TOL:.0e}")
    if not g1b:
        print("\n  The plan stops the increment here and the discrepancy IS the report.")
        return

    print(); print(line); print("A. THE TABLE SECTION 4 REPORTS, RE-MEASURED"); print(line)
    print(f"  {'fraction':>9s} {'mean n_f':>9s} {'assortativity':>14s} {'floor rel MSE':>14s} "
          f"{'between-chain sd':>17s}")
    for f in FRACTIONS:
        s = ch[ch.fraction == f]
        sd = s.groupby("dataset")["floor_rel_mse_a"].std().mean() if f < 1.0 else 0.0
        print(f"  {f:>9} {s.n_f.mean():>9.1f} {s.assortativity.mean():>14.4f} "
              f"{s.floor_rel_mse_a.mean():>14.4f} {sd:>17.4f}")

    print(); print(line); print("B. THE PRIMARY AND SECONDARY STATISTICS"); print(line)
    ra, fa = per_target(ch, "assortativity"), per_target(ch, "floor_rel_mse_a")
    d_floor, d_assort = fa[0.125] - fa[1.0], ra[1.0] - ra[0.125]
    for nm, d in (("floor_rel_mse(1/8) - floor_rel_mse(1)", d_floor),
                  ("assortativity(1) - assortativity(1/8)", d_assort)):
        m, lo, hi = boot_mean(d.values, d.index.values)
        print(f"  {nm:40s} {m:+.4f} [{lo:+.4f}, {hi:+.4f}]   positive on {int((d > 0).sum())}/40")

    print(); print(line); print("C. GATE 2. THE DECOMPOSITION"); print(line)
    fap, fb = per_target(ch, "floor_rel_mse_aprime"), per_target(ch, "floor_rel_mse_b")
    print(f"  {'fraction':>9s} {'D_A':>9s} {'D_Aprime':>10s} {'D_B':>9s} "
          f"{'D_B / D_Aprime':>28s} {'normaliser':>12s}")
    bands = {}
    for f in (0.125, 0.25, 0.5):
        dA, dAp, dB = fa[f] - fa[1.0], fap[f] - fap[1.0], fb[f] - fb[1.0]
        r, lo, hi = boot_ratio(dB.values, dAp.values, dA.index.values)
        bands[f] = (r, lo, hi)
        print(f"  {f:>9} {dA.mean():>9.4f} {dAp.mean():>10.4f} {dB.mean():>9.4f} "
              f"{r:>12.4f} [{lo:+.4f}, {hi:+.4f}] {(dA - dAp).mean():>12.4f}")
    g2 = all(0.90 <= bands[f][0] <= 1.10 for f in bands)
    print(f"\n  GATE 2: {'PASS' if g2 else 'FAIL'}, band [0.90, 1.10] two-sided at every fraction")
    print("  Reading, per the plan: a pass means the movement is the classical k/n rank-dilution")
    print("  channel (Samworth 2012 Thm 1; near-neighbour distance moments citing Evans, Jones and")
    print("  Schmidt 2002). No novelty is claimed for the mechanism. A low value is an INSTRUMENT")
    print("  failure of arm B or of the fixed-point-set premise, not a discovery about chemistry.")

    print(); print(line); print("D. THE EXACT RESIDUAL DECOMPOSITION"); print(line)
    for f in (0.125, 0.25, 0.5):
        dA, dAp, dB = fa[f] - fa[1.0], fap[f] - fap[1.0], fb[f] - fb[1.0]
        chk = abs((dA - dB).mean() - ((dA - dAp).mean() + (dAp - dB).mean()))
        print(f"  f = {f:<6} D_A - D_B = {(dA - dB).mean():+.5f}  =  normaliser "
              f"{(dA - dAp).mean():+.5f}  +  rank-dilution residual {(dAp - dB).mean():+.5f}"
              f"   (identity check {chk:.2e})")

    print(); print(line); print("E. PREDICTIONS"); print(line)
    p1 = int((ra[1.0] > ra[0.125]).sum()), int((fa[1.0] < fa[0.125]).sum())
    print(f"  1  r rises on {p1[0]}/40 and the floor falls on {p1[1]}/40, both need >= 34  "
          f"-> {'HOLDS' if min(p1) >= 34 else 'FAILS'}")
    med = float((ra[1.0] / ra[0.125]).median())
    small = ra[0.125][ra[0.125] < 0.05]
    print(f"  2  median r(1)/r(1/8) = {med:.3f}, needs [1.70, 2.30]  "
          f"-> {'HOLDS' if 1.70 <= med <= 2.30 else 'FAILS'}")
    print(f"     flagged small denominators (r(1/8) < 0.05): "
          f"{list(small.index) if len(small) else 'none'}")
    print("  3  crossing counts on modi_q2 over the declared grid. 0.65 is Golbraikh's threshold for")
    print("     the BINARY index and its use here is this project's transposition, not the source's.")
    print(f"     {'threshold':>9s} " + " ".join(f"{f:>8}" for f in FRACTIONS) + "   informative")
    strict, ok3 = 0, True
    for tt in GRID:
        cnt = [int((fa[f] < 1 - tt).sum()) for f in FRACTIONS]
        inf = cnt[0] < 40 and cnt[-1] > 0
        if inf:
            ok3 &= cnt[-1] >= cnt[0]
            strict += cnt[-1] > cnt[0]
        print(f"     {tt:>9} " + " ".join(f"{c:>8d}" for c in cnt) + f"   {'yes' if inf else 'no'}")
    print(f"     -> {'HOLDS' if ok3 and strict >= 5 else 'FAILS'} "
          f"(monotone where informative: {ok3}; strict on {strict}, needs >= 5)")
    ident = {f: float((ch[ch.fraction == f].identity
                       - ch[ch.fraction == f].floor_rel_mse_a).abs().mean()) for f in FRACTIONS}
    print("  4  mean |identity - floor| : " + "  ".join(f"f={f}: {v:.4f}" for f, v in ident.items()))
    print(f"     -> {'HOLDS' if ident[0.125] > ident[1.0] else 'FAILS'}")

    print(); print(line); print("F. ARM B RANK DIAGNOSTIC, REPORTED NOT GATED"); print(line)
    for name in RANK_TARGETS:
        s = ch[(ch.dataset == name) & (ch.fraction == 0.125)]
        if not len(s) or "rank_j1" not in s.columns or s.rank_j1.isna().all():
            continue
        r = s.iloc[0]
        dev = max(abs(r[f"rank_j{j}"] / r[f"rank_j{j}_expected"] - 1) for j in range(1, K + 1))
        print(f"  {name:22s} n={int(r.n):5d} f=0.125  max |realised/expected - 1| over j=1..10 "
              f"= {dev:.4%}   (expected = j*n/n_f, the negative hypergeometric mean)")

    print(); print(line)
    print("G. THE BINARY INDEX ON MoleculeACE, WHERE 0.65 IS THE SOURCE'S OWN"); print(line)
    mt = t[t.collection == "MoleculeACE-30"]
    print(f"  {'fraction':>9s} {'mean MODI':>10s} {'above 0.65':>12s} {'not estimable':>14s} "
          f"{'min class kept':>15s}")
    for f in FRACTIONS:
        s = mt[mt.fraction == f]
        est = s.dropna(subset=["modi_binary"])
        print(f"  {f:>9} {est.modi_binary.mean():>10.4f} "
              f"{int((est.modi_binary > 0.65).sum()):>8d}/{len(est):<3d} "
              f"{len(s) - len(est):>14d} "
              f"{int(s[['class_0_kept', 'class_1_kept']].min().min()):>15d}")

    print(); print(line); print("H. THE OUTCOME RULE"); print(line)
    mf, lof, hif = boot_mean(d_floor.values, d_floor.index.values)
    ma_, loa, hia = boot_mean(d_assort.values, d_assort.index.values)
    print(f"  floor difference  {mf:+.4f} [{lof:+.4f}, {hif:+.4f}]")
    print(f"  assortativity     {ma_:+.4f} [{loa:+.4f}, {hia:+.4f}]")
    if not (lof > 0 and loa > 0 and min(p1) >= 34):
        print("\n  SECTION 4 IS WITHDRAWN. An interval covers zero or the sign count failed, so the")
        print("  exploratory numbers did not survive specification, as claim 4's did not.")
    elif g2:
        print("\n  SECTION 4 IS REWRITTEN. The intervals hold and gate 2 passes, so the section leads")
        print("  with the decomposition and the reporting-practice consequence, with the classical")
        print("  k/n citation attached to the mechanism, not with the table as a discovery.")
    else:
        print("\n  SECTION 4 SURVIVES, and gate 2 did not pass. Per the plan this is an instrument")
        print("  failure of arm B or of the fixed-point-set premise, reported as such with no")
        print("  mechanism proposed for the residual.")


if __name__ == "__main__":
    raise SystemExit(main())

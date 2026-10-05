import numpy as np
import pandas as pd
import pytest
from molace.analysis import correlate as co


def test_bootstrap_recovers_a_strong_correlation_and_excludes_zero():
    rng = np.random.default_rng(0)
    x = rng.normal(size=60)
    y = x * 2.0 + rng.normal(scale=0.3, size=60)
    clusters = np.repeat(np.arange(6), 10)
    r = co.cluster_bootstrap_spearman(x, y, clusters, n_resamples=2000, seed=0)
    assert r.rho > 0.9 and r.lo > 0.0 and r.excludes_zero


def test_bootstrap_does_not_exclude_zero_for_pure_noise():
    rng = np.random.default_rng(1)
    x = rng.normal(size=60)
    y = rng.normal(size=60)
    clusters = np.repeat(np.arange(6), 10)
    r = co.cluster_bootstrap_spearman(x, y, clusters, n_resamples=2000, seed=0)
    assert not r.excludes_zero
    assert r.lo < 0.0 < r.hi


def test_resampling_is_over_clusters_not_rows():
    """A cluster bootstrap must be wider than a row bootstrap when the signal lives between clusters.

    x and y here vary only between clusters and are constant within them, so sixty rows carry the
    information of six points and a row bootstrap is overconfident. The row bootstrap is computed in
    the test for contrast rather than assumed, because the claim is a comparison.

    The plan's version used x identical to y, which makes Spearman exactly 1 for every resample that
    retains any variation, so both bootstraps were degenerate and the comparison was untestable.
    """
    from scipy import stats

    cluster_x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    cluster_y = np.array([0.0, 2.0, 1.0, 4.0, 3.0, 5.0])      # loosely monotone across clusters
    clusters = np.repeat(np.arange(6), 10)
    x = cluster_x[clusters]
    y = cluster_y[clusters]

    cl = co.cluster_bootstrap_spearman(x, y, clusters, n_resamples=4000, seed=0)

    draws = []
    for i in range(4000):
        idx = np.random.default_rng(i).integers(0, len(x), size=len(x))
        v = stats.spearmanr(x[idx], y[idx]).statistic
        if np.isfinite(v):
            draws.append(v)
    row_width = float(np.percentile(draws, 97.5) - np.percentile(draws, 2.5))

    assert (cl.hi - cl.lo) > row_width


def test_effective_n_is_about_ten_for_the_real_cluster_sizes():
    clusters = (["GPCR"] * 12 + ["Kinase"] * 6 + ["NR"] * 6 +
                ["Other"] * 3 + ["Protease"] * 2 + ["Transferase"] * 1)
    assert 8.0 < co.effective_n(clusters, rho_intra=0.3) < 12.0


def test_effective_n_equals_n_when_clusters_are_singletons():
    assert co.effective_n(list(range(30)), rho_intra=0.3) == pytest.approx(30.0)


def test_incremental_contribution_is_near_zero_for_a_redundant_predictor():
    rng = np.random.default_rng(2)
    base = rng.normal(size=60)
    frame = pd.DataFrame({
        "gap": base + rng.normal(scale=0.2, size=60),
        "assortativity": base,
        "rogi": base,                      # a perfect duplicate of the signal
        "mean_degree": rng.normal(size=60),
        "n_molecules": rng.integers(600, 3600, size=60),
        "receptor_class": np.repeat(np.arange(6), 10),
    })
    r = co.incremental_contribution(frame, "gap", ["rogi", "mean_degree", "n_molecules"])
    assert not r.excludes_zero


def test_report_puts_the_positive_control_first():
    rng = np.random.default_rng(3)
    frame = pd.DataFrame({
        "dataset": [f"T{i}" for i in range(30)],
        "assortativity": rng.normal(size=30),
        "rogi": rng.normal(size=30),
        "mean_degree": rng.normal(size=30),
        "n_molecules": rng.integers(600, 3600, size=30),
        "gap": rng.normal(size=30),
        "access": rng.normal(size=30),
        "correction": rng.normal(size=30),
        "receptor_class": (["GPCR"] * 12 + ["Kinase"] * 6 + ["NR"] * 6 +
                           ["Other"] * 3 + ["Protease"] * 2 + ["Transferase"] * 1),
    })
    keys = list(co.report(frame))
    assert keys[0] == "positive_control"
    assert "n" in co.report(frame)["meta"] and "n_eff" in co.report(frame)["meta"]


def test_cluster_bootstrap_mean_is_wider_than_a_row_bootstrap_of_the_same_values():
    """The Hodge criterion's interval must resample classes, like every other interval here.

    The runner first bootstrapped the thirty per-target differences as if they were independent,
    which contradicts the pre-registration's own uncertainty rule and the n_eff of about ten that the
    rest of the analysis treats as load-bearing.
    """
    rng = np.random.default_rng(0)
    clusters = np.repeat(np.arange(6), 5)
    offsets = np.array([-0.4, -0.3, -0.1, 0.0, 0.1, 0.2])        # between-class spread
    x = offsets[clusters] + rng.normal(scale=0.01, size=30)      # tight within class

    cl = co.cluster_bootstrap_mean(x, clusters, n_resamples=4000, seed=0)
    row = [float(np.mean(np.random.default_rng(i).choice(x, len(x)))) for i in range(4000)]
    row_width = float(np.percentile(row, 97.5) - np.percentile(row, 2.5))

    assert cl.mean == pytest.approx(float(np.mean(x)), abs=1e-12)
    assert (cl.hi - cl.lo) > row_width
    assert cl.n_clusters == 6


def test_cluster_bootstrap_mean_reports_whether_it_excludes_zero():
    clusters = np.repeat(np.arange(6), 5)
    neg = co.cluster_bootstrap_mean(np.full(30, -0.5), clusters, n_resamples=1000, seed=0)
    assert neg.excludes_zero and neg.hi < 0
    mixed = co.cluster_bootstrap_mean(np.tile([-1.0, 1.0], 15), clusters, n_resamples=1000, seed=0)
    assert not mixed.excludes_zero


def test_cluster_resample_keeps_the_multiplicity_a_bootstrap_depends_on():
    """A cluster drawn twice must contribute its rows twice.

    scripts/field_profile.py and scripts/synthetic_decay.py once resampled as
    `frame.cell.isin(set(rng.choice(cells, len(cells), replace=True)))`. The set() collapsed
    multiplicity, so each draw was a without-replacement subsample of about 1 - 1/e of the clusters
    rather than a bootstrap, and the reported intervals were not the intervals the frozen plans
    required. This pins the difference.
    """
    clusters = np.repeat(np.arange(150), 4)            # synthetic_decay's shape: 150 cells of 4
    groups = co.cluster_groups(clusters)
    rng = np.random.default_rng(0)
    sizes, dup_seen = [], False
    for _ in range(200):
        idx = co.cluster_resample(groups, rng)
        sizes.append(idx.size)
        if len(np.unique(idx)) < idx.size:
            dup_seen = True
    assert set(sizes) == {clusters.size}, "every draw must carry as many rows as the data"
    assert dup_seen, "with replacement, some draw must repeat a row"

    broken = [len(set(rng.choice(np.arange(150), 150, replace=True))) * 4 for _ in range(200)]
    assert np.mean(broken) / clusters.size < 0.70, "the set() form keeps about 1 - 1/e of the rows"
    assert np.mean(sizes) / clusters.size == 1.0


def test_cluster_resample_draws_every_cluster_when_there_is_only_one():
    groups = co.cluster_groups(np.zeros(7, dtype=int))
    idx = co.cluster_resample(groups, np.random.default_rng(0))
    assert np.array_equal(np.sort(idx), np.arange(7))

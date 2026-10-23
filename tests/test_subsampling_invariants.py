"""A subsample is a SET, but this codebase's neighbour selectors read an ORDER.

Both selectors break similarity ties on the ascending index: knn_graph by
`np.lexsort((index, -s))` and the leave-one-out floor by `np.argsort(-s, kind="stable")`. Ties are
not rare on molecular data -- 26.8 % of rows on the largest ChEMBL target and 35.5 % on the smallest
have their 10th and 11th similarity exactly equal -- so which members of a tie land inside the k
depends on how the subsample's rows happen to be ordered inside the submatrix.

Increment 15's draft plan said `perm[:n_f]` and would have failed its own instrument gate by ten
orders of magnitude, on a subsample that is the whole dataset. The convention that fixes it is
`numpy.sort(perm[:n_f])`, and these tests pin it so the next script that subsamples inherits the
convention rather than the bug.

Each test carries a self-check that fails if the hazard it guards has gone away, because a guard
whose premise has quietly become vacuous is worse than no guard: it reports PASS either way.

WHAT THESE TESTS DO NOT DO. They pin the invariant and they pin the one committed script that relies
on it. They cannot police a future call site: a new script that slices a permutation without sorting
it will not fail anything here. A static grep guard was considered and rejected, because this project
already has one that decayed -- the guard against `hash()` inside a seeded generator stopped catching
its own bug the moment the seed moved into a named helper, and had to be widened after the fact. The
honest coverage is: the convention is specified in the frozen plan, demonstrated here, and checked
against the script that implements it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from molace.graphs import knn
from molace.measures.assortativity import target_assortativity

K = 10


def tied_similarity(n: int = 60, levels: int = 4, seed: int = 0) -> np.ndarray:
    """A symmetric similarity matrix on only `levels` distinct values, so ties are everywhere."""
    rng = np.random.default_rng(seed)
    v = (rng.integers(0, levels, size=(n, n)) / levels).astype(np.float32)
    v = np.minimum(v, v.T)
    np.fill_diagonal(v, np.float32(1.0))
    return v


def floor_rel_mse(sim: np.ndarray, y: np.ndarray, k: int = K) -> float:
    s = sim.copy()
    np.fill_diagonal(s, -np.inf)
    idx = np.argsort(-s, axis=1, kind="stable")[:, :k]
    yhat = y[idx].mean(axis=1)
    return float(np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2))


def stats(sim: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    return target_assortativity(knn.knn_graph(sim, K), y).value, floor_rel_mse(sim, y)


@pytest.fixture
def fixture():
    n = 60
    sim = tied_similarity(n)
    y = np.random.default_rng(1).normal(size=n)
    # SELF-CHECK: the fixture must actually contain k-boundary ties, or both tests below are vacuous.
    s = sim.copy()
    np.fill_diagonal(s, -np.inf)
    so = -np.sort(-s, axis=1)
    tied = int((so[:, K - 1] == so[:, K]).sum())
    assert tied > n // 4, f"fixture has only {tied}/{n} tied k-boundaries; the guard would be vacuous"
    return sim, y


def test_a_subsamples_row_order_changes_the_statistics(fixture):
    """The hazard itself: the same SET in a different order gives different numbers.

    This is not a defect to fix -- a deterministic tie-break has to break ties somehow -- it is the
    reason the subsample's order is a specification choice that a frozen plan must make.
    """
    sim, y = fixture
    n = len(y)
    idx = np.sort(np.random.default_rng(2).permutation(n)[: n // 2])
    shuffled = np.random.default_rng(3).permutation(idx)
    assert set(idx) == set(shuffled), "the two index arrays must be the same SET"

    r_sorted, f_sorted = stats(sim[np.ix_(idx, idx)], y[idx])
    r_shuffled, f_shuffled = stats(sim[np.ix_(shuffled, shuffled)], y[shuffled])
    assert (r_sorted, f_sorted) != (r_shuffled, f_shuffled), (
        "row order no longer affects the statistics, so a selector's tie-break has changed; "
        "re-derive the subsampling convention before trusting any increment that relies on it"
    )


def test_b_sorting_the_subsample_recovers_the_full_set_computation(fixture):
    """The convention: at a fraction of 1, sort(perm[:n]) is the identity and nothing moves.

    This is what makes increment 15's gate 1 a test of the pipeline rather than of a permutation.
    Without the sort the committed columns move by about 1e-3 against a 1e-12 tolerance.
    """
    sim, y = fixture
    n = len(y)
    perm = np.random.default_rng(4).permutation(n)

    S = np.sort(perm[:n])
    np.testing.assert_array_equal(S, np.arange(n))
    assert stats(sim[np.ix_(S, S)], y[S]) == stats(sim, y)

    # SELF-CHECK: the unsorted index set is what the draft specified, and it must NOT agree, or this
    # test would pass on the very bug it exists to catch.
    assert stats(sim[np.ix_(perm, perm)], y[perm]) != stats(sim, y), (
        "an unsorted full permutation now reproduces the full-set statistics, so this guard no "
        "longer distinguishes the convention from the bug"
    )


def test_c_exact_subset_thinning_follows_the_negative_hypergeometric_mean():
    """Arm B's rank law: the j-th survivor of a uniform (m)-subset sits at mean rank j*n/n_f.

    Increment 15 draws an EXACT uniform subset without replacement rather than Bernoulli-thinning,
    because Bernoulli matches this mean only to within frac/(f*n) and differs in variance. The draft
    reached for Bernoulli on a cost argument that was wrong in the other direction: choice(replace
    =False) uses a Floyd sampler and is O(n*n_f), while per-molecule Bernoulli draws n-1 uniforms.
    """
    n, n_f, reps = 4000, 500, 400
    rng = np.random.default_rng(5)
    acc = np.zeros(K)
    for _ in range(reps):
        keep = rng.choice(n - 1, size=n_f - 1, replace=False)
        acc += np.sort(keep)[:K] + 1.0
    realised = acc / reps
    expected = np.arange(1, K + 1) * n / n_f
    # 4 % covers the Monte Carlo error at 400 replicates; the systematic part is far smaller.
    np.testing.assert_allclose(realised, expected, rtol=0.04)

    # SELF-CHECK: a WRONG law must be rejected at this tolerance, or the assertion is toothless.
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(realised, np.arange(1, K + 1) * 1.0, rtol=0.04)


def test_d_the_committed_script_uses_the_sorted_convention():
    """The one artefact that relies on this, checked against its own source.

    Narrow on purpose. This guards the script increment 15's numbers came from, not all future code;
    see the note in the module docstring about why a broader static guard was rejected.
    """
    src = Path("scripts/size_dependence.py").read_text(encoding="utf-8")
    assert "np.sort(perm[:n_f])" in src, (
        "scripts/size_dependence.py no longer sorts its subsample index set. Its instrument gate "
        "compares against results/modi_baseline.csv at 1e-12 and would miss by about 1e-3, on a "
        "subsample that is the entire dataset."
    )
    # SELF-CHECK: the unsorted form must be absent, not merely the sorted form present.
    assert "perm[:n_f]\n" not in src.replace("np.sort(perm[:n_f])", "")

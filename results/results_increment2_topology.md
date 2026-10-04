# The topology of a molecular similarity graph, exactly

Produced by `scripts/hodge_census_exact.py` and `scripts/topology.py`; numbers in
`results/report_hodge_census_exact.txt`, `results/report_topology.txt`,
`results/hodge_census_exact.csv` and `results/topology.csv`.

## What is new here

Increment 1 could prove the curl/harmonic split of the cycle space on **3 of 30** targets, because
it reached for rank(B2) by dense SVD and gave up above 5,000 edges. Asking instead for the nullity
of the edge Laplacian returns the same split on **30 of 30**, with a spectral certificate, at
seconds to minutes per target. The three targets the dense SVD could do are reproduced exactly:
110, 124 and 149.

| | min | median | max |
|---|---|---|---|
| harmonic / cycle space | 0.0156 | 0.0488 | 0.1679 |
| harmonic / gradient | 0.0869 | 0.2795 | 1.0301 |
| triangles per edge | 1.635 | 2.384 | 2.771 |
| average clustering | 0.353 | 0.583 | 0.686 |

## The objection the three small graphs invited, answered

The design expected the harmonic component to run **2 to 5 times larger than the gradient
component**, from a synthetic probe. Increment 1 refuted that on the three smallest graphs of
thirty, which invites the obvious reply: harmonic dimension grows with graph size and those three
were unrepresentative.

It does not grow with graph size. Across all 30 targets the harmonic share against |E| is
+0.103 [−0.660, +0.595] and against |V| +0.080 [−0.660, +0.596]; both cover zero. And the
expectation is refuted on **every one of the 30**, not three: the largest harmonic-over-gradient
ratio anywhere is 1.03, against a threshold of 2, so the worst margin is a factor of 1.94, and it
occurs at CHEMBL4203_Ki, not at any of the three originally reported.

## What it does track

| | rho | 95% CI | |
|---|---|---|---|
| average clustering | −0.895 | [−0.971, −0.606] | excludes 0 |
| triangles per edge | −0.883 | [−0.973, −0.593] | excludes 0 |
| mean degree | +0.588 | [+0.100, +0.764] | excludes 0 |
| degree dispersion | +0.048 | [−0.308, +0.214] | covers 0 |

The harmonic share tracks triangle coverage, not size. A molecular similarity graph is triangle-rich
because real chemical series are dense clusters of analogues (a neighbour's neighbour is a
neighbour), and the measured clustering coefficient runs 0.35 to 0.69. The synthetic probe that
produced the design's expectation was built from random sparse bit vectors, where nearest
neighbours are essentially arbitrary; reproducing it gives a clustering coefficient of 0.018 to
0.053, twelve to thirty-five times lower, and 64 triangles where the real graphs have thousands.

## A correlation that looked like chemistry and is not

The graph is built from structure alone, so a correlation between a topological quantity and a
label statistic would be a fact about chemistry rather than a circularity. One appeared: harmonic
share against ROGI, −0.511 [−0.628, −0.026], which excludes zero.

It does not survive. ROGI is not a pure label statistic (it hierarchically clusters molecules by
structure before it ever looks at the property), and ROGI against average clustering is
+0.541 [+0.015, +0.708], which also excludes zero. With the structural covariates removed, ROGI's
contribution to the harmonic share is −0.167 [−0.600, +0.503] and covers zero. It was reading
the structure.

The two genuinely label-side statistics show nothing: target assortativity −0.196 [−0.554, +0.075]
and unbiased homophily on the cliff flag −0.249 [−0.736, +0.150], both covering zero. **No
label-to-topology relationship is established here**, and the one that appeared to be is an
artefact of what ROGI is made of.

## What replaced the estimator, and why it cannot fail the same way

The first census estimated rank(B2) with a randomised range finder, which returns
rank(A·Omega) <= min(rank A, probe width) and so returned exactly the probe width, 2048, against a
true rank of 4,423. Nothing in the sketch's spectrum showed it.

    L1 = B1^T B1 + B2 B2^T,   ker L1 = ker B1 cap ker B2^T = the harmonic space
    dim harmonic = nullity(L1),   rank B2 = |E| - (|V| - c) - nullity(L1)

A shift-invert Lanczos solve returns the eigenvalues nearest zero, so one clearly positive
eigenvalue among them proves every zero was found. When no such eigenvalue appears, or the solver
fails, `HarmonicDim` reports `UNCERTIFIED` and no count at all, because the saturating number is
the specific thing being avoided. See `src/molace/hodge/nullity.py` and
`tests/test_hodge_nullity.py`, which pins the crash cases (a singular Laplacian is exactly the case
the module is for), the uncertified path, dense-against-iterative agreement, and rank(B2) against an
independent dense rank.

# Hodge dimension census, 30 MoleculeACE targets

Graph: kNN at k = 10 on ECFP4 Tanimoto, the pre-registered primary construction.
Pre-registration: see `prereg/increment1.yaml`; the hash travels in `results/hodge_census.csv`.

## What is exact on all 30 targets

| quantity | range across targets |
|---|---|
| edges | 4,295 - 25,032 |
| triangles | 8,515 - 57,661 |
| gradient dimension, \|V\| - c | 614 - 3,656 |
| cycle-space dimension, \|E\| - \|V\| + c | 3,665 - 21,376 |

**Every target has triangles.** The design's precondition for measuring curl at all -- that the
comparison graph is not triangle-free -- holds everywhere at k = 10, with a minimum of 8,515
triangles. No target is disqualified.

## What the curl-against-harmonic split says, and on how little

The split needs rank(B2) and is computed exactly only where a dense decomposition is affordable:
**3 of 30 targets**, and they are the three smallest graphs by edge count, ranks 1, 2 and 3 of 30.

| target | nodes | edges | gradient | cycle space | curl | harmonic |
|---|---|---|---|---|---|---|
| CHEMBL2047_EC50 | 631 | 4,295 | 630 | 3,665 | 3,555 | 110 |
| CHEMBL1871_Ki | 659 | 4,326 | 657 | 3,669 | 3,545 | 124 |
| CHEMBL2835_Ki | 615 | 4,335 | 614 | 3,721 | 3,572 | 149 |

On those three: harmonic over gradient has median **0.189**, harmonic is **3.4%** of the cycle space
(median), and harmonic exceeds gradient on **0 of 3**.

## This refutes the expectation the design was built on

The design document's section 10 expected the harmonic component to run **2 to 5 times larger than
the gradient component**, from a synthetic probe at n = 600 with ECFP-like sparse fingerprints. On real
targets where the rank is exactly computable it runs at about **a fifth** of the gradient, and the cycle
space is almost entirely curl. The probe pointed the wrong way and the design already marked its
numbers as not quotable; this table is what replaces them.

**What this does not license.** Three targets, and the three smallest. Harmonic dimension may well grow
with graph size, and nothing here measures that. The statement is about these three graphs, not about
the thirty, and the across-target claim stays with the energy fractions, which need no rank and run
everywhere.

## A measurement that was nearly a false finding

The first version of the census estimated rank(B2) with a randomised range finder above the exact
threshold. A range finder cannot return a rank above its probe width: rank(A @ Omega) <= min(rank A,
probe width), with equality almost surely for a Gaussian Omega. So on CHEMBL1862_Ki it returned a curl
dimension of exactly 2,048 -- the probe width -- and put the harmonic part at 2,866, pointing in
exactly the direction the project was looking for. Nothing in the sketch's own spectrum gave it away:
the smallest singular value was 0.141 of the largest, against a cut at 1e-8 of the largest, so all
2,048 cleared the tolerance by seven orders of magnitude. The only tell was the estimate equalling the
probe width exactly.

**How badly it was wrong, measured rather than guessed.** An earlier version of this file said "a
factor of twenty" and the project's talk notes said seventeen. Both compared DIFFERENT GRAPHS: the
inflated 2,866 (58.3% of the cycle space) is CHEMBL1862_Ki, while the roughly 130 (3.4%) is the median
of three other, 25% smaller targets whose rank the dense SVD could do. No one had computed the true
rank on the target where the estimator actually fired, so there was no within-target factor to quote.

It is now computed. CHEMBL1862_Ki has dim_harmonic = 491 exactly, hence rank(B2) = 4,423 and a
harmonic share of 9.99%. The estimator's 2,866 is therefore an inflation of **5.8x**, and "the exact
rank on comparable targets is about 3,550" was a poor proxy: this target's own rank is 4,423.

**What replaced the estimator.** The mistake was asking for the large number. rank(B2) runs to
thousands here and every cheap route to it saturates silently. The nullity of the edge Hodge Laplacian
is the small number and certifies itself:

    L1 = B1^T B1 + B2 B2^T,   ker L1 = ker B1 cap ker B2^T = the harmonic space
    dim harmonic = nullity(L1),   rank B2 = |E| - (|V| - c) - nullity(L1)

A shift-invert Lanczos solve returns the eigenvalues nearest zero, so a clearly positive one among
them proves every zero was found -- the certificate the range finder lacked. Validated against all
three dense-SVD targets, reproducing 110, 124 and 149 exactly with spectral gaps of 0.017 to 0.033.
See `src/molace/hodge/nullity.py`, `tests/test_hodge_nullity.py` and
`scripts/hodge_census_exact.py`; the full census it produces is in `results/hodge_census_exact.csv`
and supersedes the "unavailable" rows of `results/hodge_census.csv`.

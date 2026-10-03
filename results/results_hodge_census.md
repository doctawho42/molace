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
threshold. A range finder cannot return a rank above its probe width, so it returned a curl dimension
of exactly 2,048 -- the probe width -- where the exact rank on comparable targets is about 3,550, and
the harmonic part was inflated from roughly 130 to 2,866. That is a factor of twenty, pointing in
exactly the direction the project was looking for. The estimator is removed; the split is reported as
unavailable where it cannot be proved.

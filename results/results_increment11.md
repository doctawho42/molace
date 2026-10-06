# MODI as the baseline the venue requires, and the closed form that reproduces it

Pre-registration: `prereg/increment11_modi.yaml`, blob
`acdba0f77ff9c8fc9149dc9cd8ed39db94122ae1`, frozen before any MODI value was computed. Reproduce with
`uv run python scripts/modi_baseline.py`; output in `results/report_modi_baseline.txt` and
`results/modi_baseline.csv`.

The prior-art pass for a JCIM submission found that MODI (Golbraikh, Muratov and Tropsha, JCIM 2014)
is edge homophily on a 1-NN graph, and that its continuous form MODI_q2 is leave-one-out q2 of
similarity search, which is this project's kNN floor evaluated by LOO. The paper is not submittable
to that journal without MODI measured. 80 targets across three collections.

## A. The identity reproduces MODI_q2 without running it

| collection | targets | mean absolute difference | max | Pearson |
|---|---|---|---|---|
| ChEMBL-40 | 40 | **0.0103** | 0.0436 | 0.9975 |
| MoleculeACE-30 | 30 | **0.0083** | 0.0238 | 0.9960 |
| DeepDelta-10 | 10 | **0.0123** | 0.0362 | 0.9975 |

The frozen threshold was 0.05 on every collection. It holds everywhere with a factor of four to
spare.

This is the paper's central claim, and it is about cost as much as about structure. MODI_q2 requires a
leave-one-out sweep: one prediction per compound, so eight thousand predictions on the largest target
here. The right-hand side of the identity is two sparse matrix products and no prediction at all.
A published and well-cited index turns out to have a closed form in two graph statistics, accurate to
about one point of q2, and in that form it is visible what the index is made of: a first moment, which
is what gets reported, and a second moment, which appears in no published version.

## B and C. What each index says that the other does not

Rank-residualised partial Spearman against the best attained skill.

| collection | assortativity over MODI_q2 | MODI_q2 over assortativity |
|---|---|---|
| ChEMBL-40 | +0.240 [-0.048, +0.497] covers 0 | **+0.305 [+0.026, +0.539] excludes 0** |
| MoleculeACE-30 | +0.101 [-0.245, +0.438] covers 0 | +0.263 [-0.092, +0.580] covers 0 |
| DeepDelta-10 | +0.018 [-0.711, +0.925] covers 0 | +0.200 [-0.590, +0.963] covers 0 |

The plan expected both columns to come back empty. One cell did not, and the asymmetry is the
expected direction rather than a surprise: MODI_q2 carries the second moment implicitly, because it
evaluates the estimator whose error the identity expands, while assortativity is the first moment
alone. So MODI is the richer statistic of the two, and the identity says exactly why.

The size matters more than the sign. The interval clears zero by 0.026 on the largest collection only,
and covers zero on the other two. Reported as a small effect visible at n = 40, not as a separation.

## D. What the second moment adds

| collection | assortativity alone | full identity | difference |
|---|---|---|---|
| ChEMBL-40 | 0.928 | 0.943 | +0.0156 [-0.0171, +0.0600] covers 0 |
| MoleculeACE-30 | 0.842 | 0.854 | +0.0120 [-0.0542, +0.0849] covers 0 |
| DeepDelta-10 | 0.758 | 0.806 | +0.0485 [-0.1132, +0.3376] covers 0 |

Covers zero on all three, as the plan said to expect: increment 5 measured the two statistics
correlating at +0.988 on these graphs, so the second moment has little room to act here. The honest
statement is that the second moment is necessary for the identity and not demonstrably useful for
ranking on similarity graphs of this kind. Where it would act is the regime increments 7 to 10 showed
cannot be reached at this sample size.

## The 2014 threshold does not discriminate on a modern benchmark

The binary index of the original paper, computed on MoleculeACE's own activity-cliff flag:

| | |
|---|---|
| range over 30 targets | 0.693 to 0.809 |
| above the paper's modelability threshold of 0.65 | **30 of 30** |

Every target in the benchmark is "modelable" by the published criterion. Over those same 30 targets
the attained skill of the best of three arms runs from +0.147 to +0.626, a spread of 0.479, and
MODI_q2 runs from +0.394 to +0.831.

So the binary index saturates where the continuous quantities still separate targets by a factor of
four in attained accuracy. That is not a criticism of the 2014 paper, whose threshold was calibrated
on the datasets of its time for a correct-classification-rate target; it is a statement about what a
binary 1-NN agreement can resolve on a benchmark built specifically to contain activity cliffs.

## What this does to the paper

- Claim 1 of the skeleton is a replication of Golbraikh 2014 at larger scale under a frozen plan, and
  is labelled as one.
- Claim 5 becomes the paper's centre and is now measured against the published index rather than
  argued: the modelability index has a closed form, reproduced to 0.008 to 0.012 across 80 targets.
- Claim 2's baseline table now carries MODI beside ROGI, which the venue requires.
- The second moment is reported as structurally necessary and empirically quiet, with its intervals.

## A correction: these numbers come from a re-measurement with a pinned tie-break

Found on 2026-10-05 by a sweep for process-dependent results, after the seeding fix recorded at the
foot of `results_increment9_10.md`. Two defects in `scripts/modi_baseline.py`, neither of which changes
a verdict.

`modi_q2` selected its k nearest neighbours with `np.argpartition`, which guarantees nothing about
which of several equal similarities lands inside the k. Tanimoto on 2048-bit ECFP4 is a ratio of small
integers, so exact ties are dense: on the largest target, 2203 of 8205 rows have the kth and (k+1)th
similarity exactly equal. The selection was therefore a property of the NumPy build rather than of the
data. It is now `np.argsort(..., kind="stable")`, which breaks ties on ascending index by definition
and which was checked to agree with the project's `lexsort` pattern on every row of that target.

Section D's bootstrap generator was also constructed once before the per-collection loop, so each
collection's interval depended on how many draws the collections before it had consumed. It is now
constructed per collection, matching what `cluster_bootstrap_spearman` already did for sections B
and C. ChEMBL-40 was first in the loop, so its interval is unaffected; the other two moved.

| figure | before | after |
|---|---|---|
| section A, ChEMBL-40 mean absolute difference | 0.0108 | 0.0103 |
| section A, DeepDelta-10 | 0.0134 | 0.0123 |
| section A, largest single difference | 0.0440 | 0.0436 |
| MODI over assortativity, ChEMBL-40 | +0.293 [+0.011, +0.530] | +0.305 [+0.026, +0.539] |
| section D, MoleculeACE-30 interval | [-0.0554, +0.0844] | [-0.0542, +0.0849] |
| section D, DeepDelta-10 interval | [-0.1299, +0.3396] | [-0.1132, +0.3376] |

Every verdict holds and the headline improves: the closed form now reproduces MODI_q2 slightly more
closely on two of three collections, the frozen 0.05 threshold still holds everywhere with a factor of
four to spare, and the one interval that excluded zero still excludes it, by 0.026 rather than 0.011.
`prereg/increment11_modi.yaml` was not touched.

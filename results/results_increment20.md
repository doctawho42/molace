# Increment 20: claim 4 survives at half its magnitude

Pre-registration: `prereg/increment20_claim4_reweighted.yaml`, blob
`4e60cc1e95f3d3777311ba47317dc2c84933f847`, frozen before the script existed. Script
`scripts/claim4_reweighted.py`, committed at `54a1468` before it had produced a value. Artefacts:
`results/claim4_reweighted.csv` (70 targets) and `results/report_claim4_reweighted.txt`.

**No model was fitted.** Increments 14 and 16 use the identical split rule and their floor columns
agree to 3.331e-16, so this pairs two committed tables. Section 4a recorded the debt in its own words:
the audit fixed the model arm and left the floor at `m = 10`, and a symmetric pre-registered version
was owed.

**Verdict: claim 4 survives, at 52 % of its reported magnitude.** Giving the floor its own symmetric
selection recovers almost exactly half the gap on both collections, and the half it recovers splits
almost exactly evenly between choosing the neighbourhood size and choosing the weighting shape. The
gap does not close.

## Both gates pass, and the second one could have failed

| gate | what it ties | result |
|---|---|---|
| 1 | the two tables sit on one split: `skill_floor` against `g1_skill_mean_m10`, merged on (collection, dataset) | **3.331e-16** over 70; 1.110e-16 on ChEMBL-40 |
| 2 | both inputs by git blob hash, the learner set, the (m, α) grids, α ≠ 0, and three marginal means to 1e-4 | all pass |

Gate 2's α ≠ 0 clause is the one with teeth: an all-zero α would make the floor arm the uniform arm,
and this whole comparison a restatement of increment 14. The first draft declared a version of this
gate unable to fail, which is the defect that has cost this project a fatal finding in each of the last
four increments.

## The result, on both scales

**ChEMBL-40, 40 targets:**

| difference | skill | absolute RMSE | floor ahead |
|---|---|---|---|
| **reweighted floor − CV-selected learner** | **−0.0394** [−0.0501, −0.0289] | −0.0449 [−0.0564, −0.0339] | **5 / 40** |
| uniform floor at m=10 − learner (increment 14) | −0.0754 [−0.0865, −0.0639] | −0.0852 [−0.0983, −0.0720] | 3 / 40 |
| what the floor's own selection recovers | **+0.0361** [+0.0305, +0.0415] | +0.0402 [+0.0340, +0.0462] | 38 / 40 |
| — of which, choosing `m` | +0.0179 [+0.0114, +0.0242] | +0.0197 [+0.0129, +0.0263] | 31 / 40 |
| — of which, choosing the weighting | +0.0181 [+0.0146, +0.0218] | +0.0205 [+0.0163, +0.0250] | 37 / 40 |

**MoleculeACE-30, 30 targets:**

| difference | skill | absolute RMSE | floor ahead |
|---|---|---|---|
| **reweighted floor − learner** | **−0.0474** [−0.0586, −0.0360] | −0.0596 [−0.0741, −0.0453] | **3 / 30** |
| uniform floor at m=10 − learner | −0.0949 [−0.1084, −0.0806] | −0.1171 [−0.1356, −0.0986] | 1 / 30 |
| recovered | +0.0474 [+0.0412, +0.0536] | +0.0575 [+0.0494, +0.0661] | 30 / 30 |
| — choosing `m` | +0.0263 [+0.0182, +0.0342] | +0.0325 [+0.0227, +0.0425] | 26 / 30 |
| — choosing the weighting | +0.0211 [+0.0171, +0.0257] | +0.0250 [+0.0205, +0.0297] | 30 / 30 |

Both scales are reported because increment 18 established that the paper must carry both.

## Two things the numbers say that a mean would hide

**The recovered fraction is the same on both collections.** 0.0361 of 0.0754 is 48 %; 0.0474 of 0.0949
is 50 %. Two collections, different sizes, different task framing, and the floor's own selection buys
back half the gap on each.

**The two halves of that recovery are equal.** On ChEMBL-40, choosing `m` is worth +0.0179 and choosing
the weighting shape +0.0181 — a difference of **0.0002**, with a decomposition residual of exactly
0.0e+00. The project had treated these as one effect ("giving the floor its better `m`") and then as
another ("the reshaping gain is +0.0155"); they are the same size, and the weighting term here
reproduces increment 16's committed primary statistic of +0.0181 to four decimals, which independently
confirms the two tables sit on one split.

## All four predictions hold, and two of them were derivable in advance

| | holds | note |
|---|---|---|
| 1. the gap narrows | ✓ −0.0394 against −0.0754 | **derivable from marginal means, no evidential weight** |
| 2. it narrows but does not close | ✓ [−0.0501, −0.0289] | **the pivot**, not derivable from means |
| 3. the two halves are of a size | ✓ 0.0002 apart | **derivable, no evidential weight** |
| 4. the floor still loses on the large majority | ✓ ahead on 5 of 40, needed < 8 | not derivable |

Predictions 1 and 3 are labelled derivable-in-advance in the frozen plan and carry no evidential
weight: both are differences of committed marginal means that any reader of the repository could have
computed. The evidential content is prediction 2's paired interval, which depends on the per-target
covariance of the two arms, and prediction 4's count.

## Against this increment

**Every point estimate was derivable before the plan was written.** An earlier draft claimed the
paired interval was unobtainable from the repository; that was wrong, since both tables carry
per-target columns. What is new is the registration, the intervals, the sign counts and the
decomposition.

**Selection is symmetric in form but not in search cost.** The floor arm searches 40 cross-validation
cells (5 values of `m` times 8 of α) against the model arm's 3, and increment 16's committed floor-arm
selection regret is 0.0040 on ChEMBL-40. That residual asymmetry points **against** the floor arm,
which is the direction that keeps claim 4 alive, so it is stated rather than left to a reader. Nor is
the design symmetric in capacity: a one-parameter rank kernel over `m` neighbours and a choice among
three learners are not matched families. That asymmetry is the question, not a defect.

**The quoted instrument constant had to be re-derived.** The pre-freeze review offered 2.776e-16 over
the 70 shared targets and 1.943e-16 on ChEMBL-40; direct measurement gives 3.331e-16 and 1.110e-16.
Its diagnosis was right — name the merge keys, the columns and the subset — and its numbers were not.
This is the third time in three increments that a reviewer's replacement figure did not reproduce,
which is why the standing rule is to re-derive rather than paste.

**DeepDelta-10 is out of scope**, being absent from both input frames.

## What the floor arm chose

`m`: 5 on 7 targets, 10 on 23, 20 on 2, 40 on 8. α: 0.5 on 1, 0.75 on 5, **1.0 on 26**, 1.5 on 8 — Gou's
rank weight on 26 of 40, consistent with increment 16.

## What this does to the paper

Claim 4 keeps its direction and loses half its magnitude, and the mechanism it attributed the gap to
loses half its scope:

> The kNN floor loses to a cross-validated learner by **−0.0394 [−0.0501, −0.0289]** of skill on 40
> ChEMBL targets and **−0.0474 [−0.0586, −0.0360]** on 30 MoleculeACE targets, when both arms select on
> the training split alone. Half of the gap previously reported is the floor's own unexercised freedom:
> **+0.0179** for choosing how many neighbours to average and **+0.0181** for choosing how to weight
> them, two effects of equal size. What remains cannot be attributed to the neighbourhood's weighting,
> and the paper no longer implies that the whole gap is information outside the neighbourhood.

Section 4a's unregistered sensitivity — which found −0.0596 [−0.0708, −0.0482] with `m` alone — is
replaced by this registered measurement.

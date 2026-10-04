# molace increment 1 results

Pre-registration `prereg/increment1.yaml`, blob `864ec513`; amendment history and the one stamp
mismatch are in `docs/prereg-amendments.md`. 30 MoleculeACE targets, kNN graph at k = 10 on ECFP4
Tanimoto, three seeds averaged per target, n_eff about 10 at an assumed intra-receptor-class
correlation of 0.3. All bootstrap intervals resample receptor classes, not rows, and all kept
10,000 of 10,000 draws.

## The pre-registered primary claim is not supported

| | rho | 95% cluster-bootstrap CI | excludes 0 |
|---|---|---|---|
| positive control: assortativity vs `access` | +0.043 | [−0.456, +0.530] | no |
| **primary**: assortativity vs `gap` | **−0.187** | **[−0.579, +0.063]** | **no** |
| incremental over ROGI, degree and size | −0.187 | [−0.410, +0.080] | no |
| decisive secondary: assortativity vs `correction` | −0.153 | [−0.571, +0.416] | no |

The decision rule required the primary **and** the incremental contribution to exclude zero. Neither
does. Target assortativity does not predict which model class wins on this benchmark.

### The positive control failed, and the fault is in the control

The control asked assortativity to predict `access`, the error difference between the pointwise arm
and the kNN floor, on the reasoning that nearest-neighbour regression works precisely when the label
is smooth on the nearest-neighbour graph. That reasoning is right about the floor and wrong about the
difference: the pointwise models benefit from a smooth label too, and by almost the same amount.
Measured, assortativity against each arm's own raw error gives −0.236 for the floor and −0.161 for
the pointwise arm, both with intervals covering zero. A difference of two quantities that respond
alike carries no signal, which is why the control reads +0.043.

So the control was mis-specified, not the pipeline. **Post hoc, the control this should have been is
measured and it works.** Expressing each arm's accuracy as skill over the no-graph baseline
(predicting the training mean on the test split) rather than as a raw error, assortativity against
the kNN floor's skill gives +0.866, 95% CI [+0.694, +0.951], and against the pointwise arm's skill
+0.813 [+0.550, +1.000]. Both exclude zero decisively; their difference is +0.055 [−0.427, +0.541]
and does not.

That is the whole mechanism in three numbers. The statistic is strongly informative about **how
accurately a target can be predicted at all**, which is what a positive control needs to show, and
it shows it at rho about +0.87. It carries almost nothing about **which arm wins**, because both arms
inherit the same smoothness. The pre-registered control asked for the second thing while the
reasoning that motivated it established only the first, so it could not have fired for the stated
reason. A control specified as a **level** passes; the same control specified as a **difference of two
arms** is empty by construction.

This paragraph is a post-hoc diagnostic, not a pre-registered result: it was computed after the
primary was known, and it is reported to explain a failed control rather than to support a claim.
The numbers come from `results/spine.csv` plus the per-target training-mean baseline, which needs no
refitting. Reproduce them with `uv run python scripts/control_recheck.py`; its output is kept at
`results/report_control_recheck.txt`. It uses the frozen plan's 10,000 draws, so the pre-registered
`access` row in its output reproduces `results/report_spine.txt` exactly.

### A units caveat that limits the magnitudes, not the conclusion

The gap is expressed in the label's own units. Across these 30 targets the label standard deviation
spans 2.1x and correlates with assortativity at +0.645, so part of the primary's magnitude is scale
rather than structure. **Post-hoc**, standardising the gap by the label standard deviation moves the
primary from −0.187 to −0.058 (CI [−0.396, +0.157]) and the secondary from −0.153 to −0.058. Both are
nulls, so the pre-registered conclusion stands while its magnitude does not.

## What did come out, robustly, and it is not what the project was looking for

| quantity | sign | in label SDs | 95% cluster CI | excludes 0 |
|---|---|---|---|---|
| `gap` = pointwise − pairwise | 28/30 negative | **−0.082** | [−0.101, −0.049] | **yes** |
| `correction` = floor − pairwise | 28/30 negative | **−0.076** | [−0.093, −0.042] | **yes** |
| `access` = pointwise − floor | 16/30 negative | −0.006 | [−0.013, −0.0002] | yes, but see below |

Best arm per target: pointwise 15 of 30, kNN floor 14, pairwise 1.

Read in order:

- **The pairwise arm loses.** It is worse than the pointwise arm on 28 of 30 targets, by about a
  twelfth of a label standard deviation, with an interval excluding zero in both label units and
  standardised units.
- **The learned pairwise function subtracts value.** It is worse than simply averaging the ten nearest
  training labels, by about the same margin. So the half-1 statement of the restated second hypothesis
  comes out stronger than equality: the learned function does not merely fail to add expressivity over
  anchor averaging, it costs accuracy.
- **Test-time label access buys almost nothing.** `access` technically excludes zero at three seeds,
  but the effect is six thousandths of a label standard deviation with the sign split 16 to 14. That is
  detectable and negligible, and it should be read as "the nearest-neighbour mean and the fitted
  pointwise models are interchangeable here", not as a finding.

**What this does not license.** Our pairwise arm is probably under-tuned rather than representative.
It ships without model selection, with a linear kernel slot because the radial one does not finish on
58,480 pairs, with `concat_difference` deferred, and with hyperparameters resized for cost. DeepDelta's
*published* pairwise model wins on several of its ten datasets; ours wins on one of thirty. The honest
claim is about this configuration, not about pairwise learning.

## The second half agrees with the first, through a different instrument

Validity checks first, because nothing below them means anything otherwise:

- Pointwise-floor curl fraction, maximum over 30 targets: 2.0e-33. Differencing a trained pointwise
  model on the edges gives an exactly gradient flow, as it must.
- Worst Pythagorean residual of any decomposition: 1.3e-13. The three components stay mutually
  orthogonal and reconstruct the flow.
- Neither check can detect wrong orientation signs. A pure gradient flow leaves a zero residual, so
  the curl solve returns zero whatever the triangle operator holds. The signs are enforced instead in
  `hodge.complex.build`, which asserts `B1 @ B2 == 0` on every complex it returns.

| | median over 30 targets |
|---|---|
| gradient fraction of the trained flow | **0.967** |
| curl fraction | 0.0256 |
| harmonic fraction | 0.0077 |
| curl fraction, same architecture on shuffled labels | 0.0288 |

**A trained pairwise flow is 97% gradient.** The restated second hypothesis, that such a model carries
little beyond a node potential, is confirmed by flow energy, and separately by held-out error through
the negative `correction`. Two instruments give the same answer on the same claim.

And the circulation carries no label information: the shuffled control's curl fraction (0.0288) is
indistinguishable from the trained model's (0.0256), and marginally higher.

### The second half's own criterion fails decisively

Per-edge curl had to predict that edge's held-out error better than plain anchor dispersion, which is
published three times. It does not:

- mean rank-correlation difference (curl minus dispersion): −0.0769
- 95% cluster-bootstrap CI over receptor classes: [−0.1024, −0.0278], which excludes zero on the
  wrong side
- negative on 27 of 30 targets

So the Hodge residual is a worse uncertainty signal than the dispersion it was meant to improve on.
This is a pre-registered null, reported rather than omitted.

### The dimension census refutes the design's own expectation

The design expected the harmonic component to run two to five times the gradient component, from a
synthetic probe. On the three targets where the rank is exactly computable it runs at about a fifth of
the gradient and makes up 3.4% of the cycle space, which is almost entirely curl. Three targets, and
the three smallest of thirty, so this is about those graphs; see `results_hodge_census.md`. Every one
of the 30 targets has triangles, so the curl precondition holds everywhere.

## The holdout is invalid as pre-registered

| | rho | 95% CI |
|---|---|---|
| assortativity vs `gap` | +0.817 | [+0.259, +1.000] |
| assortativity vs `correction` | +0.800 | [+0.174, +1.000] |
| assortativity vs `access` | −0.583 | [−0.892, +0.123] |

**Do not read these as support for the hypothesis.** The nine TDC endpoints mix log and raw scales:
label standard deviations run from 0.78 to 81.8, a factor of about a hundred. The gap is in label
units, so a gap of −30.6 on `vdss_lombardo` and +0.52 on `caco2_wang` are not commensurable. Both the
statistic and the gap track that scale (assortativity against label SD gives −0.700 and the gap
against label SD gives −0.883), so the correlation is a property of units rather than of graph
structure, and the low-assortativity targets are exactly the raw-scale ones where the pairwise arm
blows up.

The pre-registration failed to require a dimensionless gap. The amendment the next increment owes is
in `docs/prereg-amendments.md`.

### Post-hoc: the result does not survive the repair, and the diagnosis above was half wrong

A spent holdout cannot be made confirmatory again, so what follows answers one diagnostic question
only (does +0.817 survive putting the label on a comparable scale?) and is reported as a
sensitivity analysis, not as a result. The transform rule carries no free parameter: log10 is
admissible exactly where the label is strictly positive, which on these nine endpoints is exactly the
five in raw units; the other four go negative because they already are logarithms. Run it with
`uv run python scripts/holdout_sensitivity.py`; output in `results/report_holdout_sensitivity.txt`
and `results/holdout_sensitivity.csv`.

| scale of the label | assortativity vs standardised gap | 95% CI |
|---|---|---|
| raw, as pre-registered | +0.883 | [+0.474, +1.000] |
| log10 where admissible (5 of 9 refitted) | **+0.017** | [−0.737, +0.722] |
| rank-to-normal, one rule for all 9 | **+0.250** | [−0.541, +0.846] |

The holdout's interval, the only one the PRE-REGISTERED analysis produced that excluded zero, stops
excluding it as soon as the label is put on a comparable scale. Nothing should be built on it.
(Post-hoc analyses elsewhere in this file do exclude zero: the level-based control at +0.866 and
+0.813, and increment 2's H1. Those are different claims on different quantities.)

**Two corrections to the paragraph above.** First, the stated mechanism was the smaller half.
Standardising only the gap, which is what "the gap is in label units" implies, makes the correlation
*stronger* (+0.817 to +0.883), because the confound lives in the statistic as much as in the gap:
target assortativity is a Pearson correlation across edge endpoints, and on `vdss_lombardo` (label
0.01 to 700 L/kg, skewness 27) the raw value is 0.082 while the same graph with the label's ranks
gives 0.426 (`rank_assortativity`). Two rank-based quantities appear in this project and they are
not the same: `rank_assortativity` is Pearson of the plain ranks, while the `rank_normal` SCALE in
`scripts/holdout_sensitivity.py` is a van der Waerden transform of the label before refitting, which
gives 0.391 on the same graph. The sensitivity table reports the second; this sentence is about the
first. Replacing Pearson with ranks alone, refitting nothing and leaving the gap in raw label units,
already drops the headline from +0.817 to +0.550 [-0.185, +0.948], an interval covering zero.
Doing both (ranks and a standardised gap) gives +0.533 [-0.263, +1.000]. An earlier version of
this paragraph quoted the second number for the first change. This is a statement about what was
measured here, not a general law: outliers can inflate a Pearson correlation as readily as deflate
it.

Second, the repair is not clean and should not be sold as one. The confound check (assortativity
against label SD) does not vanish under the transforms; it changes sign, from −0.700 raw to +0.883
under log. And the log rule is not uniformly the right transform: on `ppbr_az`, a plasma protein
binding *percentage* bounded above, log moves the statistic the wrong way (0.187 to 0.148) where the
rank versions move it up (`rank_assortativity` 0.260, van der Waerden 0.243); a bounded proportion
wants a logit. The defensible conclusion is
the weak one and it is enough: the result is not stable under the choice of scale.

## The external replication disagrees in sign and settles nothing

DeepDelta's published per-fold predictions, ten ADMET datasets, no retraining:

| | rho | 95% CI |
|---|---|---|
| assortativity vs gap against Random Forest | +0.430 | [−0.296, +0.887] |
| assortativity vs gap against ChemProp | +0.345 | [−0.459, +0.988] |

Positive point estimates against our negative one, both intervals covering zero. At n = 10 this is
uninformative, and these gaps are in label units across mixed scales too, so the same caveat applies.
The disagreement in sign is worth stating plainly: it is the instability that withdrew arXiv 2606.10249,
whose authors found their own correlation vanished after a pipeline fix.

## In one paragraph

Target assortativity does not predict which model class wins, and the pre-registered test says so with
both required intervals covering zero. The project's cheap positive control was mis-specified and is
replaced for next time. What the run does establish, robustly and in dimensionless units, is that this
pairwise configuration loses to both pointwise models and to a plain nearest-neighbour mean, that a
trained pairwise flow is 97% gradient so it carries almost nothing beyond a node potential, and that
its circulation is a worse uncertainty signal than the anchor dispersion already in the literature.
The out-of-domain holdout is invalid on a units defect the pre-registration failed to exclude, and the
published external replication points the other way with an interval that covers zero. Two of those
findings are nulls the design predicted might happen; the holdout defect and the harmonic refutation
are ones it did not.

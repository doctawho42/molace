# What target assortativity predicts, and what it does not

Pre-registration: `prereg/increment2_separation.yaml`, blob
`419d0556420388d3bdfc21bde907941de2a526bc`, frozen before the confirmatory set was opened for this
question. Reproduce with `uv run python scripts/separation.py`; output in
`results/report_separation.txt` and `results/separation_deepdelta.csv`.

## The claim

**H1.** Target assortativity predicts **attainable accuracy**: a model's skill over the no-graph
baseline, where skill = 1 - RMSE_model / RMSE_baseline and the baseline predicts the evaluation
set's mean.

**H2.** It does **not** predict **which model family wins**: the difference of two families'
skills is uncorrelated with it.

Found post hoc on the 30 MoleculeACE targets while diagnosing why increment 1's positive control
failed. That set is the discovery set and cannot confirm the claim.

## Confirmatory set: DeepDelta, 10 benchmarks, three published families, no refitting

The models are not mine, the splits are not mine, and none of the three families averages
neighbours, which is what keeps H1 from restating the kNN floor's definition. Primary statistic is
the rank (Spearman across edge endpoints) version, fixed in the plan before any number was seen.

| | rho | 95% CI | |
|---|---|---|---|
| skill of RandomForest | +0.770 | [+0.329, +1.000] | excludes 0 |
| skill of ChemProp50 | +0.818 | [+0.373, +1.000] | excludes 0 |
| skill of DeepDelta5 | +0.758 | [+0.245, +0.987] | excludes 0 |
| RandomForest minus ChemProp50 | −0.152 | [−0.862, +0.636] | covers 0 |
| RandomForest minus DeepDelta5 | −0.503 | [−0.985, +0.234] | covers 0 |
| ChemProp50 minus DeepDelta5 | −0.321 | [−0.976, +0.509] | covers 0 |

**H1 supported.** The Pearson version agrees (+0.758, +0.806, +0.733), so it does not rest on the
choice of statistic.

**H2 is NOT supported, and the pre-registered rule that says otherwise is a bad rule.** It declares
H2 supported when no pairwise difference excludes zero, which is accepting a null from a failure
to reject, and at n = 10 the test has almost no power to reject anything. One of the three
differences has a point estimate of -0.503 inside an interval nearly a full unit wide. That is
not evidence that assortativity fails to predict which family wins; it is the absence of evidence
either way. Writing an accept-the-null rule into a pre-registration does not convert a null into a
result, and the same power objection this document raises against the ROGI increment applies here
with more force.

What does bound the effect is the discovery set, where n = 30 and the intervals are tight enough to
say something: pointwise minus pairwise is +0.073 [-0.150, +0.457] and floor minus pointwise
+0.055 [-0.427, +0.541]. The first of those rules out an effect above about 0.46. That is
exploratory, on the set the claim was found on, and it is the honest basis for the separation
language, not the confirmatory H2 rows.

## What this does to increment 1's null

Increment 1 asked whether assortativity predicts the pointwise-minus-pairwise **gap** and found
nothing: −0.187, interval covering zero. That null now has a mechanism rather than being a bare
absence. The statistic carries a great deal about **how accurately a target can be predicted at
all** and almost nothing about **which model class wins**, because both classes inherit the same
label smoothness. Increment 1 asked for the second thing. Its own positive control failed for
exactly this reason (it was defined as a difference of two arms that respond alike) and that
failure is what led here.

## Where it fails: the increment over the roughness baseline

The plan required the claim to beat ROGI, the cheminformatics roughness index, with mean degree and
task size residualised out. **On the confirmatory set it does not**, and that is the pre-registered
answer:

| DeepDelta, n = 10 | rho | 95% CI | |
|---|---|---|---|
| assortativity incremental over ROGI, RandomForest | +0.212 | [−0.610, +0.790] | covers 0 |
| assortativity incremental over ROGI, ChemProp50 | +0.273 | [−0.459, +0.826] | covers 0 |
| assortativity incremental over ROGI, DeepDelta5 | +0.248 | [−0.469, +0.733] | covers 0 |
| ROGI alone vs skill, RandomForest | −0.818 | [−1.000, −0.231] | excludes 0 |
| ROGI alone vs skill, ChemProp50 | −0.855 | [−1.000, −0.337] | excludes 0 |
| ROGI alone vs skill, DeepDelta5 | −0.806 | [−0.988, −0.231] | excludes 0 |

On these ten datasets ROGI predicts attainable accuracy at least as well as the graph statistic, and
the graph statistic adds nothing that can be distinguished from zero.

**The discovery set says the opposite, and the two have not been reconciled.** On the 30 MoleculeACE
targets, exploratory:

| MoleculeACE, n = 30, n_eff about 10 | rho | 95% CI | |
|---|---|---|---|
| assortativity incremental over ROGI, pointwise | +0.707 | [+0.401, +0.988] | excludes 0 |
| assortativity incremental over ROGI, pairwise | +0.647 | [+0.302, +0.950] | excludes 0 |
| assortativity incremental over ROGI, kNN floor | +0.785 | [+0.567, +0.952] | excludes 0 |
| ROGI incremental over assortativity, pointwise | +0.102 | [−0.411, +0.455] | covers 0 |
| ROGI incremental over assortativity, kNN floor | +0.212 | [−0.163, +0.412] | covers 0 |
| ROGI alone vs skill, pointwise | −0.200 | [−0.363, +0.093] | covers 0 |

There the graph statistic dominates and ROGI adds nothing over it, the reverse of the confirmatory
set on every line.

**What is honest to say about the disagreement.** Three differences could produce it and this work
separates none of them: n = 10 against n = 30 with three covariates residualised, which leaves the
confirmatory increment with very little power; a skill defined on pairwise deltas against one
defined on absolute values; and ROGI itself behaving differently on the two collections (−0.82
against skill on the ten, −0.20 on the thirty). The pre-registered answer is the answer:
**the increment over the roughness baseline is not established.** Claiming otherwise from the
discovery set would be reporting the set the claim was found on.

## What a reader should take

Confirmed on independent data and third-party models: **target assortativity predicts how
accurately a molecular regression dataset can be predicted at all**, by models that are not mine and
that average no neighbours. That half is solid.

The other half, that it does *not* predict which model class wins, is **not** confirmed. The
confirmatory set cannot reject anything at n = 10, and what supports the separation language is the
discovery set's tighter interval, which is exploratory. A reader should take the first sentence as
established and the separation as a hypothesis with one supporting exploratory bound.

Not established: that it says anything a cheminformatics roughness index does not already say. On
the confirmatory set it does not. Settling that needs more datasets than ten, and it is the first
thing the next increment should do.


## Deviations from the frozen plan, recorded

The plan is frozen at blob `419d0556420388d3bdfc21bde907941de2a526bc` and is not edited. Four places
where `scripts/separation.py` does not do what it says, found by review:

1. **A secondary set that was never computed.** The plan declares nine TDC tasks as a caveated
   second set. The script has no TDC path and the report has no such section. Not run; not reported;
   listed here rather than quietly dropped.
2. **Two baselines under one name.** The plan defines the baseline as "predicting the mean of the
   evaluation set's true values". The confirmatory arm does that. The discovery arm uses the
   **training** mean, because an evaluation-set mean is an oracle no predictor could use. Measured
   both ways on the discovery set, the difference is immaterial: pointwise skill correlates +0.813
   against the training mean and +0.810 against the evaluation mean; the floor, +0.866 against
   +0.867.
3. **A wider covariate set than declared.** The plan says `incremental_over: [rogi]`; the script
   residualises against ROGI, mean degree and task size. That is more conservative than frozen, but
   it is not what was frozen, and the "where it fails" verdict depends on the covariate set.
4. **The primary statistic's definition is loose.** The plan says "Spearman across edge endpoints";
   `rank_assortativity` ranks the |V| node labels and then takes Pearson across endpoints, which is
   the degree-unweighted ranking. The two agree to about 0.001 on every graph here (vdss: 0.4259
   against 0.4247) and no downstream number changes, but the code is not the frozen wording.

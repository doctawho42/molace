# Increment 18: the disagreement with Janela & Bajorath is a disagreement about scale

Pre-registration: `prereg/increment18_jb_reanalysis.yaml`, blob
`07c64acc2d48b320b1366b3e5a6251b1b4993a9f`, frozen before the script existed. Script
`scripts/jb_reanalysis.py`, committed at `9a9a96d` before it had produced a value, with one bugfix at
`86fad16` described below. Artefacts: `results/jb_reanalysis.csv` (400 cells) and
`results/report_jb_reanalysis.txt`. The last of the five blocking experiments.

**No model was trained.** Their per-compound predictions are published, so this is a reanalysis of
their own artefacts: their table is recomputed from their own predictions, and then this project's
metric is computed on the same predictions. Their files were read as data and never executed.

**Verdict: the two literatures measured the same thing and reported it on two scales.** On their 8
targets, their 10 splits and their 5 arms, the margin they call "essentially negligible" is 0.0516 in
MAE and **+0.0757 [+0.0601, +0.0913] in skill, positive on 8 of 8 targets**. Their published numbers
reproduce exactly; nothing here says they are wrong.

## All three gates pass, and the first one is unusually strong

| gate | what it ties | result |
|---|---|---|
| 1 | all 1920 rows of their published `performance_test.csv`, recomputed from their `predictions_test.csv` | max **5.551e-16** |
| 2 | training set identical across their five arms, and disjoint from the test set, in all 80 cells | 0 and 0 |
| 3 | their constant arm has one distinct value per cell and equals `numpy.median` of the training labels | 80/80, **3.553e-15** |

Gate 1 is the increment's foundation: it means every number below is computed off a file whose
contents provably agree with their own published table, so no statistic here can be a parsing error
wearing their name. Its tolerance was checked for satisfiability before freezing, which increments 13
and 14 did not do.

## Their scale, and ours, on the same predictions

| arm | params | MAE | RMSE | R² | r | b = sd(pred)/sd(y) | **skill** |
|---|---|---|---|---|---|---|---|
| MR (training median) | 0 | 1.0255 | 1.2258 | −0.0136 | −0.0000 | 0.0000 | **−0.0051** |
| 1-NN | — | 0.6666 | 0.9331 | 0.4041 | 0.7017 | 0.9864 | **0.2330** |
| kNN (k ∈ {1,3}, Jaccard) | — | 0.6094 | 0.8314 | 0.5282 | 0.7380 | 0.8398 | **0.3167** |
| RFR | — | 0.5905 | 0.7691 | 0.5967 | 0.7807 | 0.6787 | **0.3680** |
| SVR (Tanimoto kernel) | — | 0.5578 | 0.7396 | 0.6276 | 0.7936 | 0.7786 | **0.3924** |

| pair | MAE | **skill** | R² | r² |
|---|---|---|---|---|
| SVR − kNN | +0.0516 | **+0.0757** | +0.0993 | +0.0836 |
| SVR − 1-NN | +0.1088 | **+0.1594** | +0.2235 | +0.1349 |
| SVR − MR | +0.4677 | **+0.3974** | +0.6412 | n/a |
| RFR − kNN | +0.0190 | +0.0513 | +0.0685 | +0.0638 |
| RFR − 1-NN | +0.0761 | +0.1350 | +0.1927 | +0.1151 |

Their claim, from *Sci. Rep.* 2023, 13:17816: "differences in mean prediction errors of all methods
were confined to ~ 0.1 units and thus essentially negligible". For the pair this increment compares
that reproduces with room to spare — 0.0516, about half their stated bound — and **prediction 1
holds**. Their sentence covers all five arms including the median regressor and that wider claim is
not tested here.

The same 0.0516 of MAE is **0.0757 of skill**, which is larger than every scientific effect this paper
reports: the reshaping gain is +0.0155, the second moment's ranking contribution +0.0156, the MODI
reproduction error 0.0103.

## The number that was not registered, and is reported as an observation

| | mean | interval | sign |
|---|---|---|---|
| this increment, their data, SVR − kNN | **+0.0757** | [+0.0601, +0.0913] | 8 / 8 |
| increment 14, ChEMBL-40, floor − CV-selected learner | **−0.0754** | [−0.0865, −0.0639] | 3 / 40 ahead |

The magnitudes agree to three decimal places. The signs differ only by direction convention:
increment 14 reported floor minus learner, this reports complex minus control, so both say the complex
model leads by about 0.0755 — on 8 targets with their control and their models, and on 40 different
targets with this project's.

**This agreement was not pre-registered and is not evidence of anything by itself.** What the plan
registered was prediction 2, an order-of-magnitude band of [0.025, 0.226], a factor of three either
side of increment 14's committed figure. Any gap anywhere in a nine-fold range would have satisfied it,
so "prediction 2 holds" is weak on its own. The three-decimal coincidence is an observation on top of
it, reported as such, and a reader should treat it as one target set agreeing with another rather than
as a confirmed hypothesis.

## Why the metric choice has teeth, stated as algebra rather than measured

On any test set, with `b = sd(pred)/sd(y)` and `d = (mean y − mean pred)/sd(y)`,

    R² = r² − (r − b)² − d²

so `r² = R² + penalty` with `penalty = (r − b)² + d² ≥ 0`. Verified against their own published R²
column to **1.887e-15** over all 379 non-constant cells, with zero sign violations (penalty runs
3.369e-05 to 0.2476, median 0.0139).

The squared Pearson correlation is therefore the coefficient of determination **plus** a non-negative
calibration term, and being invariant to any affine rescaling of the prediction it cannot see that
term. Their Methods report `r²`; their repository's `performance_test.csv` carries the coefficient of
determination on all 400 cells including the constant arm. Their Results say "r2 calculations did not
lead to a larger separation between the performance of different models" — and on these data the R²
separation for SVR against kNN is 0.0993 against r²'s 0.0836, a difference of 0.0157 which is exactly
`penalty_kNN − penalty_SVR = 0.0192 − 0.0035`.

**This is not evidence about their modelling choices and the plan forbids reading it as such.** Their
per-arm mean penalty is 1-NN 0.0921, kNN 0.0192, RFR 0.0145, SVR 0.0035 — a 26-fold spread — and it
follows from construction: an unweighted neighbour rule predicts individual training labels, so its
predictions are as dispersed as the labels and `b ≈ 1` (measured 0.9864 for 1-NN), while a regularised
kernel machine shrinks toward the mean and drives `b` toward `r`, which is the penalty-minimising
value. A calibration gap between those two arms was certain before the data were read.

That certainty is why this is algebra here and not a prediction. **The plan's first draft registered
exactly this comparison as the prediction the increment turned on**, and the pre-freeze review proved
it fires in 100.0 % of synthetic replicates when both arms are given the *identical* correlation, so
neither is more accurate and only their calibration differs. That made the "no finding" branch
unreachable. Increment 16's first draft had a prediction cleared by noise 88 % of the time; this one
was cleared by construction.

## The internal scale check

`mean |skill_MR|` = **0.005981** [−0.000655, 0.012617] against a registered threshold of 0.01. The
constant arm predicts the training median while this project's baseline predicts the training mean, so
its skill is near zero but not zero, and the threshold was derived from the measured 0.006058 rather
than asserted. **Prediction 3 holds.** Signed `skill_MR` is positive on 23 of 80 cells; nothing is
claimed about its sign.

The first draft registered 0.05 here — ten times the derived value and two thirds of the effect under
test. It could not have failed.

## Against this increment

**Gate 3 failed on the first run, and the failure was mine.** The plan says the constant arm must have
"exactly one distinct predicted value"; the script tested `pred_sd < 1e-15` instead. `numpy`'s
`ndarray.std` on a constant array of magnitude ~8 returns a few times 1e-15, because `sum/n` need not
return the value itself, so the threshold failed on 25 of 80 cells that are exactly constant —
`nunique` is 1 on all 80. `pandas`' `groupby.std` returns exactly 0.0 on the same data, which is how
the pre-freeze check missed it and reported all 80 cells constant. The frozen plan was right and the
script had deviated from it; the fix at `86fad16` brought the script to the plan, and both commits are
in the history. This is the second time in this project that a tolerance was set finer than the
arithmetic allows, after increment 14's 1e-12 against three stored decimals.

**Eight targets is few.** The t(7) interval's coverage is about 0.95 under symmetry and about 0.88
under strong skew, and a percentile cluster bootstrap at this cluster count covers about 0.87, which is
why the plan forbids one here and makes the 8 per-target values the primary report. They are
+0.0806, +0.0583, +0.0812, +0.0663, +0.0493, +0.0729, +0.0871, +0.1098 — all positive, spread 0.0493
to 0.1098, so the conclusion does not rest on one outlying target.

**This cannot separate their control from their data from their splits.** It holds all three fixed at
theirs. A gap of the same order as increment 14's is consistent with the scale explanation and does not
prove it: their control is at most three Jaccard-nearest neighbours and this project's is a uniform
mean over ten Tanimoto-nearest ones, and that comparison is a different experiment.

**Nothing is said** about their deep-learning arm, which lives in their 2022 repository and not in the
table reanalysed here, nor about whether an absolute or a normalised scale is the right one for their
question. If a 0.1 log-unit margin sits below assay error, absolute scale is the right scale for
"should a medicinal chemist prefer the complex model", and a normalised restatement answers a different
question. This increment establishes that the two scales give different-looking answers on identical
predictions, not which question matters.

## What this does to the paper

Section 4a's open question closes. The honest statement:

> The simple-control-ties-complex-ML result and this paper's floor-loses-to-fitted-models result are
> the same measurement on two scales. Recomputed from Janela & Bajorath's own published
> per-compound predictions, their SVR beats their kNN control by 0.0516 in mean absolute error — about
> half the ~0.1 they describe as essentially negligible — and by 0.0757 in baseline-normalised skill,
> on 8 of 8 activity classes. The coefficient of determination that would show this sits in their
> repository; their text reports the squared Pearson correlation, which is invariant to affine
> rescaling and so cannot see a predictor's calibration. From here this paper reports both scales for
> every model-against-floor comparison, and makes no claim about which scale answers which question.

## Method note

The plan's first draft went through an adversarial pre-freeze review across three lenses — gate and
prediction feasibility, inferential fairness to another research group, and statistical validity —
which raised **28 findings, of which 18 survived verification and three were fatal**, and whose own
proposed fixes were then adversarially checked and corrected in nine places. Ten findings were refuted.

The three fatal ones were the pivot being decided by construction; the metric label being wrong in five
places, where the draft said "squared Pearson" while the pre-freeze check that reproduced the column
had computed it unsquared, so a script following the draft's text would have failed gate 1 on all 320
`r` rows and stopped the increment on the plan's own error; and the outcome rule having one uncovered
state and one double-covered state with no precedence rule. The repairs were mostly removals — the
algebraic pivot out of the predictions, the percentile bootstrap out of the uncertainty, the
distinctness test replaced by an equality — which is why the second draft was frozen without a fourth
review.

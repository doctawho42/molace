# The baseline audit: claim 4 reverses

Pre-registration: `prereg/increment14_baseline_audit.yaml`, blob
`179b8bf128302f9c266dd7af9e2815c9049f18f2`. Its instrument gate was amended once, before any
measurement, and the superseded version (blob `65a408a3...`) is recorded inside the file with its
reason: the first tolerance of 1e-12 was unsatisfiable against 28 rows stored at three decimals.
Reproduce with `uv run python scripts/baseline_audit.py`; output in `results/report_baseline_audit.txt`
and `results/baseline_audit.csv`. 70 targets.

## Gate 1: the instrument is the same one

The floor arm is unchanged from every earlier increment, so it must reproduce the committed
`skill_knn_floor`. It does:

| provenance of the committed row | targets | max absolute drift | tolerance |
|---|---|---|---|
| `computed` | 12 | **5.551e-17** | 1e-12 |
| `recovered_3dp` | 28 | **4.277e-04** | 5e-04 |

Machine precision on the rows stored at full precision. So the split, the fingerprints, the anchors and
the baseline in this script are the ones claim 4 was measured with, and the comparison below is valid.

## The outcome: claim 4 reverses

| ChEMBL-40, 40 targets | mean | interval | floor ahead on |
|---|---|---|---|
| **floor − CV-selected arm (primary)** | **−0.0754** | **[−0.0865, −0.0639]** | **3 / 40** |
| floor − matched arm, as claim 4 has it | +0.0271 | [+0.0118, +0.0450] | 26 / 40 |
| CV-selected − matched arm | **+0.1025** | [+0.0899, +0.1156] | 40 / 40 |

By the frozen rule the primary interval is negative and excludes zero, so **claim 4 reverses**: the kNN
floor does not beat the best fitted model, and the published claim rested on a model arm that excluded
the strongest learner.

Independently on the secondary collection:

| MoleculeACE-30, 30 targets | mean | interval | floor ahead on |
|---|---|---|---|
| floor − CV-selected arm | **−0.0949** | [−0.1084, −0.0806] | 1 / 30 |
| floor − matched arm | +0.0056 | [−0.0114, +0.0234] covers 0 | 18 / 30 |
| CV-selected − matched arm | +0.1005 | [+0.0843, +0.1182] | 30 / 30 |

**And note the second row.** On MoleculeACE the floor's advantage over the *original* arm already covered
zero. The published claim therefore rested on ChEMBL-40 alone; the collection where the project's
pointwise learners were tuned never supported it.

## Why the error was large

The mean skill of each learner, fitted on the training split and scored on the test split:

| | svm | hgb | mlp |
|---|---|---|---|
| ChEMBL-40 | **+0.4221** | +0.3969 | **+0.2446** |
| MoleculeACE-30 | **+0.4524** | +0.4289 | **+0.2751** |

The matched arm is the mean of `hgb` and `mlp`: the second-best learner and the worst one, averaged,
with the best excluded. Cross-validation on the training split picks `svm` on **34 of 40** ChEMBL
targets and **30 of 30** MoleculeACE targets. The MLP sits 0.18 below the SVM and supplies half the
average, which is where the 0.1025 comes from.

**The arm choice is worth more than any effect this project has reported.** +0.1025 against the
reshaping gain's +0.0155, the second moment's +0.0156, and MODI's reproduction error of 0.0103.

## Both registered predictions held

**Prediction 1**, that `select_by_cv` picks `svm` on at least 21 of 40: **34 of 40**, and 30 of 30 on
MoleculeACE. So `pointwise.py`'s docstring was right that the SVM is the real bar, and its basis — a
MoleculeACE results matrix — transfers to independently curated ChEMBL targets.

**Prediction 2**, that the floor's advantage shrinks: **+0.0271 → −0.0754**. Direction only was
registered, with no threshold, because the sign was the question.

## The same error, pointed at me

The audit fixed the model arm's baseline and left the floor arm at a fixed `m = 10`, declared in the
plan as "the project's fixed constant". That is the same class of error in the other direction, and
increment 12's correction already knew it: uniform `m = 5` beats uniform `m = 10` on 28 of 31 targets.

Measured, **not pre-registered and labelled as a sensitivity**, on the 31 overlapping targets:

| | mean | interval | floor ahead on |
|---|---|---|---|
| floor at `m = 10` vs CV arm | −0.0824 | — | — |
| **floor at its better `m` vs CV arm** | **−0.0596** | **[−0.0708, −0.0482]** | **1 / 31** |

The conclusion is robust to the symmetric fix: the gap narrows by a quarter and still excludes zero.
A pre-registered version belongs in the next increment, with both arms selected on the training split.

## What this does to the paper, and what it does not

**Claim 4 is withdrawn as published.** The replacement is sharper and less flattering: on 70
independently curated targets the best fitted model beats the kNN mean by 0.060 to 0.095 skill
depending on how the floor's own `m` is chosen, and the earlier claim to the contrary came from
averaging the second- and worst-performing learners while excluding the one cross-validation selects on
64 of 70 targets.

**This is NOT the replication the plan hoped for, and the plan's tie branch did not fire.** Janela and
Bajorath, *Nature Machine Intelligence* 2022, 4:1246–1255, report that simple nearest-neighbour analysis
meets or exceeds state-of-the-art machine learning on compound potency prediction. This increment does
not reproduce that tie: here the simple control loses, clearly and on both collections.

**The honest reading of the disagreement is that it may not be one.** Their control is 1-NN and kNN
with their own choices; this project's floor is the uniform mean of a test compound's 10 nearest
*training* neighbours at a fixed `m`, on a different data selection and a single random 80/20 split.
Any of those could carry the difference, and the sensitivity above shows the floor's own `m` is worth
0.023 of the 0.075. The project cannot adjudicate this without running their protocol on their data,
which is public at `github.com/TiagoJanela/ML-for-compound-potency-prediction`. That is the obvious next
increment and it is a better one than anything else currently queued, because it tests a published
result on its author's own terms rather than on ours.

**Unaffected.** The pointwise-minus-pairwise gap keeps the matched arm, where averaging over the same
two families is correct and makes the difference a comparison of architectures rather than of learner
sets. Nothing in increments 1, 5, 11 or 12 depends on claim 4.

# Increment 19: SARI is the third baseline, and two of the four indices do not rank

Pre-registration: `prereg/increment19_sari.yaml`, blob `b62cb36b3eb92b6b960040e54c1fd1bb78aaa194`,
frozen before the script existed. Script `scripts/sari_baseline.py`, committed at `edfca4f` before it
had produced a value, with two bugfixes described below. Artefacts: `results/sari_baseline.csv` (80
targets) and `results/report_sari_baseline.txt`.

**Verdict: claim 2's row is completed as near-interchangeable rankers — but the phrase applies to two
of the four indices, not four.** Assortativity and MODI_q2 rank attainable accuracy at +0.90 and +0.92
and are nearly the same statistic; ROGI and SARI cover zero on the primary collection. And no index
adds anything incrementally over the other three, on either potency collection.

## The instrument and all three gates pass

| check | what it ties | result |
|---|---|---|
| instrument | this script's `raw_disc` against `rogi.SARI.compute_raw_scores()` at 0.6 | **1.799e-14** |
| gate 1 | all 80 targets join to a SARI value and every covariate, 40 / 30 / 10 | 0 missing in 7 columns |
| gate 2 | every target has a pair above the threshold | 0 of 80 with none, at either |
| gate 3 | the domain violation is confined to the flagged collection | **0 of 70** potency targets violate |

The instrument check exists because the frozen plan declares 0.56 primary while the reference
implementation hardcodes 0.6, and both cannot be obeyed literally. `raw_cont` comes from the reference,
which has no threshold; `raw_disc` is computed here at both thresholds, and that check is what makes
the substitution legitimate. It is printed before the gates, whatever it says.

## The result

**ChEMBL-40, 40 targets. Each index alone against `best_skill`, and incrementally over the other three
plus `mean_degree` and `n_molecules`:**

| index | alone | incremental |
|---|---|---|
| **rank_assortativity** | **+0.8956** [+0.7428, +0.9711] | +0.1749 [−0.1172, +0.4379] covers 0 |
| **modi_q2** | **+0.9233** [+0.8070, +0.9756] | +0.2161 [−0.0836, +0.4657] covers 0 |
| rogi | −0.3071 [−0.6140, +0.0440] **covers 0** | +0.0407 [−0.2738, +0.3523] covers 0 |
| sari | −0.2224 [−0.5367, +0.1313] **covers 0** | −0.0265 [−0.3190, +0.2754] covers 0 |

**MoleculeACE-30, 30 targets:**

| index | alone | incremental |
|---|---|---|
| rank_assortativity | +0.8096 [+0.5393, +0.9508] | +0.0638 [−0.3169, +0.4114] covers 0 |
| modi_q2 | +0.8563 [+0.6294, +0.9656] | +0.1315 [−0.2420, +0.4541] covers 0 |
| rogi | −0.2369 [−0.5787, +0.1811] covers 0 | −0.1711 [−0.5628, +0.2591] covers 0 |
| sari | **−0.5026** [−0.7172, −0.1995] excludes 0 | −0.2832 [−0.6362, +0.1442] covers 0 |

## The three predictions: one holds, two fail

**Prediction 2 holds — the primary.** Assortativity's incremental contribution over
`{rogi, modi_q2, SARI, mean_degree, n_molecules}` covers zero on both potency collections: +0.1749
[−0.1172, +0.4379] on ChEMBL-40 and +0.0638 [−0.3169, +0.4114] on MoleculeACE-30. It was registered in
the direction that costs this project its claim-2 story, and it holds in that direction.

**And SARI's presence in the covariate set changes nothing.** With SARI removed, the same contribution
is +0.1700 against +0.1749 on ChEMBL-40 and +0.0937 against +0.0638 on MoleculeACE-30 — differences of
0.005 and 0.030. The plan required this contrast precisely so that a reader could see whether SARI was
doing work. It is not.

**Prediction 1 fails.** SARI does not rank attainable accuracy on the primary collection: −0.2224
[−0.5367, +0.1313], covering zero. The plan said in advance that a failure here "would be a finding
about SARI, not about this project", and that is how it is reported. On MoleculeACE-30 it does rank,
at −0.5026 [−0.7172, −0.1995].

**Prediction 3 fails, and its failure qualifies the outcome branch.** The four indices are **not**
mutually redundant. Pairwise |Spearman| on ChEMBL-40:

| pair | ρ |
|---|---|
| rank_assortativity — modi_q2 | **0.9448** |
| rogi — sari | 0.6358 |
| rogi — modi_q2 | 0.4163 |
| rank_assortativity — rogi | 0.3405 |
| rank_assortativity — sari | 0.1732 **below the 0.3 floor** |
| modi_q2 — sari | **0.1467 below the floor** |

So the honest completion of claim 2's row is narrower than "four near-interchangeable rankers":
**assortativity and MODI_q2 are near-interchangeable with each other**, at 0.9448 mutual and +0.90 and
+0.92 standalone. SARI is not redundant with either — it measures something different — but that
something does not predict attainable accuracy here.

## The sign, reported and not resolved

SARI's association with attainable accuracy is **negative** on both potency collections. Its
components, on ChEMBL-40 and MoleculeACE-30 respectively:

| component | ρ with `best_skill` |
|---|---|
| `raw_cont` = 1 − weighted mean similarity | −0.1657 / −0.5244 |
| `raw_disc` | **+0.2422 / +0.2209** |

The published `raw_cont` is one *minus* a similarity-weighted mean, so a high value means a
*dis*continuous SAR; and `raw_disc` correlates **positively** with attainable accuracy, which is the
opposite of the naive reading. Complementing the continuity term does not rescue a positive
association — it gives −0.0159 and +0.0861, destroying what signal there was rather than flipping it.

**This increment does not resolve the orientation.** The primary source is unread, and the plan forbids
claiming anything new about SARI itself. What is reported is the measured sign and the named question:
on these 70 potency targets, SARI as published by three independent restatements is negatively
associated with attainable accuracy, and complementing its continuity term does not reverse that.

## The implementation defect, measured

`rogi` 0.1 computes the combination as a product rather than the published mean. The product is bounded
above by 0.5, while published SARI values above 0.5 exist in two independent papers, and three
independent sources give the mean — so it is that package's defect, not a rival convention, and the
discrepancy is isolated to the one line the originating group's own paper does not restate.

Measured here: the two forms rank these 80 targets at **Spearman 0.9464**. On the pre-freeze run over
all 80 as the reference set the same quantity was 0.9343; the frozen plan pools only the 70 potency
targets, and the normalising set itself moves the agreement between the two forms. That is the
self-referentiality the plan declared as a limit, and it is measurable.

The threshold matters much less: Spearman between 0.56 and 0.6 is **0.9974**.

## Against this increment

**The first draft was comprehensively wrong and the review earned its cost.** Five fatal findings on
this plan alone. A script following the first draft would have joined **zero of 80 rows**: it named
`rogi_showdown.csv` for DeepDelta-10, which that file does not contain (it holds MoleculeACE-30 and
TDC-9, sharing no dataset name with DeepDelta), and `rogi_chembl.csv` by a `collection` key that file
does not have. Its target variable differed from what increment 11 actually measured by up to 0.1786,
so its promise to continue increment 11 was false on its own target. It made the product combination
primary. Its claim that SARI "collapses to about 0" without a reference set was false arithmetic — the
sum form spans 0.4748 to 0.6332 unnormalised, compressing rather than collapsing. It kept a MACCS-era
threshold for an ECFP4 project. And it never declared the potency vector, which hid a domain violation.

**Two of my own bugs cost two runs.** The first dropped `receptor_class`, which
`incremental_contribution` clusters on internally; the second was introduced by my own cleanup commit,
which replaced `{{...}}` with `{...}` inside an f-string and turned a literal brace pair into an
expression. Both crashed after the instrument check and all three gates and before any primary
statistic, so no outcome was contaminated — a consequence of the plan's ordering, not of the script's
care. The verification step now compiles the module rather than only parsing it, which is what would
have caught the second.

**DeepDelta-10 is outside SARI's domain** and enters no conclusion. 8 of its 10 targets carry
non-positive labels, the label minimum over all 80 targets being −25.470, and SARI's weight
`y_i·y_j/(1+|y_i−y_j|)` goes negative there, so `raw_cont` is not a weighted mean. Its values are in
the artefact, flagged, and its descriptive standalone ρ is −0.5515 [−0.9630, +0.2000].

**SARI's absolute values here are self-referential**, the reference set being this project's 70 potency
targets, so no value is comparable to a published SARI number or threshold. Only the ranking is used.

**The target variable is `best_skill`**, a maximum over model families, so this increment inherits that
arm — not increment 14's CV-selected arm, which does not exist for DeepDelta-10.

## What this does to the paper

Claim 2's row loses its "SARI still missing" clause and gains a sharper statement:

> Of the four indices, two rank attainable accuracy and two do not. Target assortativity (+0.8956) and
> MODI_q2 (+0.9233) are nearly the same statistic, at 0.9448 mutual Spearman; ROGI (−0.3071) and SARI
> (−0.2224) cover zero on 40 ChEMBL targets. No index adds anything incrementally over the other
> three on either potency collection, so the honest claim is that assortativity and MODI_q2 are
> interchangeable rankers rather than that either beats the baseline line. SARI is not redundant with
> them — its mutual Spearman with MODI_q2 is 0.1467 — so it measures something else, which on these
> targets does not predict attainable accuracy.

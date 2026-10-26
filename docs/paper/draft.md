# The nearest-neighbour floor in closed form: what a modelability index measures, and why it is not a property of a dataset

**Draft 1, 2026-10-26.** Written from `skeleton.md` after all five blocking experiments and the fifth
prior-art pass were discharged. Every number below is traceable to a committed artefact under
`results/` and to a pre-registration frozen by git blob hash before the measurement existed; the
pointers are in the Methods section rather than inline, except where a figure's provenance is part of
the argument.

Target venue: *Journal of Chemical Information and Modeling*. The modelability programme is native
there, and so is its critique.

---

## Abstract

Modelability indices predict whether a dataset is worth modelling before a model is fitted. We take
the simplest estimator in that family — the mean activity of a compound's `k` nearest neighbours on a
molecular similarity graph — and write its error exactly:

    MSE / σ²  =  1 + 1/k + ((k−1)/k)·ρ_nn  −  2r

where `r` is the target assortativity of the activity across graph edges and `ρ_nn` the correlation
between the activities of two compounds sharing a neighbour. Three consequences follow, each measured
under a plan frozen in advance.

First, the published continuous modelability index is this expression. MODI_q2 is reproduced to a mean
absolute difference of 0.0083–0.0123 across 80 targets in three collections, at a Pearson correlation
of 0.996–0.998. The gain is interpretive rather than computational — we withdraw an earlier claim that
the closed form is the cheaper route, which timing shows to be wrong in the opposite direction — and
it consists in naming the two graph statistics the index is a function of, so that one can ask what
happens when either moves.

Second, every published index in the line reports the first moment and omits the second. Dropping it
costs a factor of 2.5 in the reproduced level — mean absolute error 0.0257 against 0.0103 — while
adding **nothing detectable** to the ranking: the incremental contribution's point estimate is positive
on all three collections (+0.0156, +0.0120, +0.0485) and every interval covers zero, with upper ends
wide enough to contain effects larger than this paper's other gains. On these data the two moments
correlate at +0.9895, so this design cannot separate their contributions to ranking and we do not
claim a null. The honest statement is that the second moment is necessary for the level and that its
ranking contribution is unresolved here.

Third, and this is the consequence for practice: the quantity these indices estimate is not a property
of a dataset but of a dataset **and its size**. Over an eightfold change in sample size the floor's
relative error moves by +0.2698 [+0.2477, +0.2922] and assortativity by +0.2599 [+0.2385, +0.2807], on
40 of 40 targets, and the 2014 binary index crosses its own published threshold on 46 % of cells by
growing alone. The sample dependence of a roughness index is **not news to this literature** — the
ROGI authors devote a supplementary figure to its convergence and bootstrap an interval on it — and we
say so rather than claim otherwise. What is new is that the movement is the classical `k/n`
rank-dilution channel, reproducible from the full-sample neighbour ordering alone rather than only
observable by resampling; that it has not been reported for the MODI family, nor at the sizes real
activity datasets reach; and that the literature reads it as convergence towards a true value while
continuing to tabulate one number per dataset and to apply a fixed threshold to it.

We also report two negative results about our own earlier claims, and one reanalysis of another
group's published predictions which shows that a prominent disagreement in this literature is a
disagreement about scale rather than about substance.

---

## 1. Introduction

The modelability programme asks a question worth asking: given a dataset, can one tell before fitting
anything whether a model will work? MODI (Golbraikh, Muratov, Fourches & Tropsha, *JCIM* 2014,
54(1):1–4) answers it with an activity-class-weighted agreement of 1-nearest-neighbour pairs and a
threshold of 0.65 calibrated on more than a hundred datasets. Its continuous forms, MODI_q2 and
MODI_ssR2, are the leave-one-out and leave-group-out q² of similarity search (Golbraikh et al., 2014
and 2021). ROGI (Aldeghi, Graff et al., *JCIM* 2022, 62(19):4660) is a roughness scalar, with ROGI-XD
(Graff et al., *Digital Discovery* 2023, 2(5):1452) correcting its representation dependence. RMODI
(Luque Ruiz & Gómez-Nieto, *JCIM* 2018, 58(10):2069) is the regression analogue. Kernel target
alignment serves the same role in closed form (Marcou, Horvath & Varnek, *JCIM* 2016, 56(1):6). SARI
(Peltason & Bajorath, *J. Med. Chem.* 2007, 50(23):5571) scores the continuity of a
structure–activity relationship from potency differences over similar pairs.

### 1.1 What these indices do, measured against each other

We computed four of them on the same 80 targets, with the same fingerprints and the same similarity,
and correlated each against the accuracy actually attained. Two rank it and two do not:

| index | Spearman with attained accuracy, ChEMBL-40 | incrementally over the other three |
|---|---|---|
| **target assortativity** | **+0.8956** [+0.7428, +0.9711] | +0.1749 [−0.1172, +0.4379] |
| **MODI_q2** | **+0.9233** [+0.8070, +0.9756] | +0.2161 [−0.0836, +0.4657] |
| ROGI | −0.3071 [−0.6140, +0.0440] | +0.0407 [−0.2738, +0.3523] |
| SARI | −0.2224 [−0.5367, +0.1313] | −0.0265 [−0.3190, +0.2754] |

Assortativity and MODI_q2 sit at 0.9448 mutual Spearman and are, for this purpose, nearly the same
statistic. **No index adds anything over the other three on either potency collection.**

**ROGI's weak showing here replicates its authors' own result.** They report that on these same 30
van Tilborg ChEMBL activity classes "ROGI displayed moderate-to-strong (r = 0.56–0.89) correlations
with model error when representing molecules with physico-chemical descriptors, but only **weak
correlations (r = 0.15–0.39) when representing molecules with fingerprints**", explaining it by "the
much smaller range of both ROGI values and model errors obtained". Our ECFP4 figure of −0.3071 against
attained accuracy is +0.31 against error and sits inside their reported fingerprint range. We
reproduce their result on their data, and the right reading is theirs.

SARI is not redundant with these — its mutual Spearman with MODI_q2 is 0.1467 — so it measures
something else. On ChEMBL-40 that something does not predict attained accuracy, but **on
MoleculeACE-30 it does**: its standalone Spearman there is −0.5026 [−0.7172, −0.1995], excluding zero,
so the failure is on the primary collection and not on both. Its association is negative, and its
discontinuity component is positively associated with accuracy, which we report and do not resolve,
the 2007 source being paywalled and unread by us. We note in passing that the `rogi` package's SARI
combines the two component scores as a product where three independent restatements give a mean, and
that the product is bounded above by 0.5 while published SARI values exceed it; we use the published
mean and report the product as that package's defect.

### 1.2 What no index in the line reports

Every one of these is published, in practice, as a **bare point estimate per dataset** — but it is
important to state precisely what the literature does and does not say, because an earlier version of
this paragraph got it wrong in a way its own citation refutes.

The ROGI authors address both uncertainty and sample size directly. On uncertainty: *"While we do not
provide a statistical estimator of ROGI uncertainty, a general approach to estimate uncertainty is by
bootstrap. Indeed, we find that bootstrap estimates of ROGI uncertainty correlate with its error,
despite a general tendency to"* underestimate it — and their Figure S15 reports bootstrapped 95 %
confidence intervals on subsets of size 10, 100, 1000, 2000 and 5000. On sample size: their Figure S2
is titled *"Convergence of ROGI values with respect to dataset size on toy surfaces"*, their
Discussion states that *"rougher landscapes are expected to require larger datasets for ROGI to
converge to its true value"*, and their Figures S13 and S14 report that *"1000 molecules were
sufficient to accurately estimate ROGI, while 100 were enough only for some properties"*.

So the field is not unaware, and this paper does not claim it is. What goes unreported is narrower and,
we think, more damning. First, the dependence is read throughout as **convergence towards a
well-defined value** — ROGI's own phrase is that the index *"has a well-defined value, even though the
ROGI computed from a small sample is only an estimate"* — rather than as a reason a published point
estimate is uninterpretable without its `n`. Second, nothing in the line reports it for the **MODI
family**, which is the index this paper shows equals the closed form, nor at the sizes real activity
datasets have; ROGI's own sweep stops at 5000 molecules on curated libraries, where the largest target
here carries 8205. Third, and this is what the closed form adds: the movement is not merely empirical
convergence but the **classical `k/n` rank-dilution channel**, reproducible from the full-sample
neighbour ordering alone. And fourth, the practice has not changed: the dependence lives in a
supplementary figure and a discussion sentence, while the index is still tabulated as one number per
dataset, and the 2014 binary index's threshold of 0.65 is still applied to it.

This paper takes one estimator and writes its error exactly. The point is not that the estimator is
good — it is not the best predictor available, and we measure by how much it loses — but that writing
it down exactly makes three things visible that the indices' own form hides.

---

## 2. Theory: the floor in closed form

Standardise the activity to zero mean and unit variance. For the mean of a compound's `k` nearest
neighbours on a similarity graph,

    MSE / σ²  =  1 + 1/k + ((k−1)/k)·ρ_nn  −  2r                                    (1)

with `r` the target assortativity of the activity across graph edges and `ρ_nn` the correlation
between the activities of two compounds that share a neighbour. Equivalently, writing `A_norm = D⁻¹A` for the row-normalised adjacency and standardising `y` to `z`,

    ‖(I − A_norm)z‖² / ‖z‖²  =  1 − 2m₁ + m₂ ,
       m₁ = zᵀA_norm z / zᵀz ,      m₂ = ‖A_norm z‖² / zᵀz                           (2)

**Both terms must be written carefully, and an earlier version of this paper wrote both wrongly.**
`A_norm` is not symmetric on a union-symmetrised kNN graph — on a mid-sized target here its asymmetry
reaches 9.5 × 10⁻² and its degrees run from 1 to 45 — so `A_normᵀA_norm ≠ A_norm²` and the second term
is the **Gram form** `‖A_norm z‖²`, not `zᵀA_norm²z`. With the Gram form equation (2) reproduces its
own left-hand side to 0.0 × 10⁰ on that target; with the square form it misses by 6.7 × 10⁻³. For the
same reason `m₁` equals Newman's target assortativity **only on regular graphs**: on that target
`m₁ = 0.661339` against `r = 0.653292`, a gap of 8.0 × 10⁻³. This is the degree correction that
section 2.2 discusses, and the two statements must agree.

### 2.1 Where it is exact, where it is an approximation, and where it decays

Equation (1) is **exact on k-regular graphs**. We verify it to 3.58 × 10⁻¹⁵ over 1200 synthetic
circulants, and the suite pins it at an absolute tolerance of 10⁻¹², so a broken derivation fails the
tests rather than producing a plausible number.

On the union-symmetrised k-nearest-neighbour graphs real data produce, it is an approximation — and
**against which measured floor matters, because there are two and an earlier version of this paper
called both by one name.** The *out-of-sample* floor is the relative MSE of the kNN mean on a held-out
20 % split; the *in-sample* floor is its leave-one-out relative MSE over every compound, which is
MODI_q2 by construction, that index being the leave-one-out q² of similarity search. The identity
misses the first by a mean absolute 0.0256 at `k = 10`, maximum 0.0837, stable in `k` at 0.0269,
0.0252, 0.0256, 0.0286 and 0.0306 for `k = 3, 5, 10, 20, 40`; it reproduces the second to 0.0103,
maximum 0.0436. The factor of 2.5 between those two numbers is a difference of target, not of
accuracy, and section 2.3's table is against the second.

Against the in-sample floor the identity also **degrades as the sample shrinks**, which bounds where it
may be used and which we measured for the first time here: 0.0103, 0.0133, 0.0168 and 0.0221 at the
full set, a half, a quarter and an eighth of each target, a factor of 2.1, monotone. The reason is structural rather than mysterious — the identity is exact only on k-regular
graphs, and union symmetrisation is further from regular when `k` is a larger fraction of `n`.

### 2.2 What is not ours

The first-moment term is a **Dirichlet energy**, `E_D(x) = xᵀLx = ½Σ_{(i,j)∈E}(x_i − x_j)²`, a standard
homophily and smoothness measure in the heterophilic-graph literature. The right-hand side as a whole
is not, because it also carries `ρ_nn`. That the kNN-mean predictor is the **harmonic** minimiser of
that energy is Zhu, Ghahramani & Lafferty (ICML 2003), who state the harmonic property as "the value
of f at each unlabeled data point is the average of f at neighboring points"; they minimise the energy
to impute labels, we measure the average's error. As a risk the expression is also the Wiener filter
and the kriging system, and we say so here rather than leave it to a reviewer.

Further, `m₁` **is the network Moran index** (Arthur, *EPJ Data Science* 2025, 14:1), and the
relationship to Newman's target assortativity is not identity. Arthur's own statement is that "unlike
assortativity, the Moran index does not incorporate the null model into its definition": written out
for unbinned real-valued data, assortativity subtracts the configuration-null expectation
`k_i k_j / 2E` of each adjacency entry, and the Moran index does not. He adds that the Moran index "is
normalised by the variance of the data rather than the variance of the node degree". Those two
differences are why `m₁` and `r` separate on an irregular graph — measured 0.661339 against 0.653292,
a gap of 8.0 × 10⁻³, on the target used above — and why equation (2) must say *Moran index* where
equation (1) says *assortativity*. An earlier version of this paper glossed Arthur's statement as a
"degree correction" and asserted an agreement he does not claim. The spatial-autocorrelation
literature owns the object either way, and `1 − m₁` is Geary's C in the same family.

### 2.3 What is ours: the published index is this expression

What remains is the accounting: that `m₂` is the shared-neighbour correlation, that the two moments
compose into the floor exactly, and that **the published continuous modelability index is this
expression**. Measured against MODI_q2 on 80 targets under a plan frozen before any value existed:

| collection | targets | mean abs. difference | max | Pearson |
|---|---|---|---|---|
| ChEMBL-40 | 40 | 0.0103 | 0.0436 | 0.9975 |
| MoleculeACE-30 | 30 | 0.0083 | 0.0238 | 0.9960 |
| DeepDelta-10 | 10 | 0.0123 | 0.0362 | 0.9975 |

The frozen threshold was 0.05 and it held everywhere with a factor of four to spare.

**The closed form is not the cheap route, and an earlier version of this paper claimed it was.** A
leave-one-out prediction for the kNN *mean* refits nothing: given the neighbour ordering it is one
gather and one row mean. Timed on this project's own code at `n = 2272`, all 1813 leave-one-out
predictions take **0.2 ms**, against **130 ms** for two dense products of the normalised adjacency —
the "expensive" route is some six hundred times faster, and a sparse implementation of the closed form
would narrow that gap without closing it, because 0.2 ms is already the cost of a single gather. The
claim was wrong by nearly three orders of magnitude and in the opposite direction, it was never timed
before being made, and it is withdrawn.

What the closed form buys is therefore **interpretive, not computational**: it says which two graph
statistics the index is a function of, which lets one ask what happens when either moves — the
question sections 3 and 4 are about — and it supplies the `k`-dependence and the sample dependence in
a form that can be reasoned about rather than only measured.

---

## 3. The term every published index omits

`m₁` alone is what the neighbour-averaging family reports. The second moment is absent from MODI,
MODI_q2, RMODI, kernel target alignment, and from Cui et al.'s Label Incoherence Index
`LII = (1/n)Σ_i(1/k)Σ_{j∈N_i}|y_j − y_i|`, a first-moment L1 index on the same kNN graph.

**ROGI is not in that list, and an earlier version of this paper wrongly put it there.** ROGI is not a
moment of labels over graph edges at all: it coarse-grains the dataset by complete-linkage clustering
at a sequence of distance thresholds and integrates a dispersion over that sequence. It reports
neither moment, so "omits the second" is the wrong criticism of it, and the comparison in section 1.1
is between indices of different construction.

Dropping it costs a **factor of 2.5 in level**: reproducing MODI_q2 from `1 − r` alone gives a mean
absolute error of 0.0257 against the full identity's 0.0103 on ChEMBL-40.

**The limit, stated here and not buried.** On ECFP4/Tanimoto graphs the two moments are nearly
collinear, at a Pearson correlation of +0.9895 across the 40 ChEMBL targets, so this design cannot
separate their contributions to ranking.

Two measurements bear on it, and they are **not** the same statistic, which an earlier version of this
paper presented as though they were. Against the out-of-sample floor, Spearman is +0.9345 for `1 − r`
and +0.9497 for the full identity, a difference of +0.0152 with no interval attached. Separately, the
incremental ranking contribution of the second moment — a rank-residual statistic against attained
accuracy, with a cluster bootstrap — has point estimates +0.0156, +0.0120 and +0.0485 on the three
collections, with intervals [−0.0171, +0.0600], [−0.0542, +0.0849] and [−0.1132, +0.3376]. Every
interval covers zero and every point estimate is positive, and the upper ends are wide enough to
contain effects larger than this paper's other gains.

So the claim is narrower than the factor of 2.5 suggests, and narrower than a null: **the second
moment is necessary for the level, and its ranking contribution is unresolved at this collinearity and
this sample size.** We do not claim it adds nothing; we claim this design cannot tell.

---

## 4. Modelability is not a property of a dataset

### 4.1 The measurement

We subsampled each of the 40 ChEMBL targets to an eighth, a quarter and a half of its compounds, five
independent chains at each fraction, and recomputed both statistics:

| fraction | mean *n* | assortativity | floor rel. MSE | between-chain s.d. |
|---|---|---|---|---|
| 1/8 | 345 | 0.2957 | 0.6915 | 0.0456 |
| 1/4 | 690 | 0.3855 | 0.5919 | 0.0355 |
| 1/2 | 1380 | 0.4752 | 0.4992 | 0.0186 |
| all | 2760 | **0.5556** | **0.4217** | — |

The floor's relative error falls by **+0.2698 [+0.2477, +0.2922]** from an eighth of the data to all of
it, and assortativity rises by **+0.2599 [+0.2385, +0.2807]**, each on **40 of 40 targets**. The median
per-target ratio `r(n)/r(n/8)` is 1.988: assortativity nearly doubles over an eightfold change in size.

Two instrument gates read exactly zero before any of this was read, the second against a
recomputation route sharing no indexing code with the first.

### 4.2 The mechanism is classical, and we cite it

A null arm that holds the point set, every pairwise similarity, the label distribution, the variance
normaliser and the scored set at full `n`, and changes only *which ranks* of its own ordering each
molecule reads, reproduces the measured movement at every fraction: ratios 1.0060, 1.0271 and 0.9848
against a two-sided band of [0.90, 1.10] declared in advance.

That `n` enters a kNN estimator only through `k/n`, by way of neighbour-rank order statistics, is
published — and the right citation for a *regression* estimator is not the one an earlier version of
this paper gave. Samworth (*Ann. Statist.* 2012, 40(5):2733) is titled "Optimal weighted nearest
neighbour **classifiers**", and its Theorem 1 decomposes the excess risk of a weighted
nearest-neighbour **classifier** under 0–1 loss; the paper itself remarks that "although the 0–1
classification loss function is quite different from the squared error loss often used in regression
problems, we nevertheless obtain such an asymptotic decomposition". So it is an **analogy** here, not
an authority, and we cite it as one. Within it, the bias term carries the factor `(k/n)^{2/d}` — for
the unweighted rule the weighted sum of `α_i = i^{1+2/d} − (i−1)^{1+2/d}` telescopes to `k^{2/d}` — so
the *squared*-bias term carries `(k/n)^{4/d}`; an earlier version of this paper called `(k/n)^{2/d}`
the squared-bias factor, dropping the square.

The statement that applies to squared error directly, and is loss-agnostic, is the near-neighbour
distance moment bound `E[ε_k^a] ≤ C(k/n)^{a/D}` of Evans, Jones & Schmidt (*Proc. R. Soc. A* 2002,
458:2839), and that is what the regression claim rests on. **We claim no novelty for the mechanism.**

One accounting detail is worth stating because it would otherwise be misreported. The residual between
the measured arm and the null decomposes exactly, to 8.7 × 10⁻¹⁹, into a variance-normaliser term and a
rank-dilution term. At an eighth of the data the normaliser term is +0.00753 and the rank-dilution
residual −0.00159: 4.7 times larger and of the opposite sign. Reported without that split, the residual
reads +0.00594 and looks like sample dependence beyond rank dilution, in the direction that flatters
this section. It is not.

### 4.3 The consequence for practice, on the index whose threshold is its own

On MoleculeACE, the one collection carrying an activity-class flag, the 2014 binary MODI with the 2014
threshold of 0.65 declares **all 30 targets modelable at full `n` and only 54 % of (target, chain)
cells at `n/8`** (mean 0.7592 against 0.6514). Same targets, same chemistry; the published criterion
crosses by growing alone. One caveat belongs directly under that row, where the measuring plan placed
it: at an eighth of the data 8 of the 150 cells rest on fewer than ten molecules in one activity
class, the thinnest on four, and a value resting on four molecules is quantised in steps of 0.125 in
that class's term. No class emptied on any of the 480 cells, and both per-class counts travel with
every value in the artefact. On ChEMBL the same monotonicity holds at all eight thresholds of a declared
grid, and we report it inside that grid rather than at 0.65 alone, because 0.65 is the binary index's
threshold and not the continuous one's.

So the claim is exactly this, and it is deliberately not the claim that the field is unaware. The
sample dependence of a roughness index is documented in the literature that introduced one: ROGI's own
Figure S2 is titled "Convergence of ROGI values with respect to dataset size", its Discussion says
rougher landscapes need larger datasets to converge, and its Figure S15 bootstraps a 95 % confidence
interval on subsets from 10 to 5000 molecules. What this section adds is that the movement is the
**classical `k/n` channel** rather than only empirical convergence, so it is predictable from the
full-sample neighbour ordering without resampling; that it is of a size — 0.27 in relative error — that
crosses the field's own published threshold; that it has not been reported for the MODI family; and
that despite all of the above, the practice of tabulating one number per dataset and comparing it to a
fixed cut-off is unchanged. **The gap is between what the literature knows and what it reports.**

**And the reading of that dependence is contested, three ways.** ROGI's authors read it as convergence
towards a value the landscape genuinely has. The Gamma-test line contains an **M-test** that recomputes
the statistic over growing samples exactly as this section does and likewise reads the result as
convergence towards a truth. Janela & Bajorath (*Sci. Rep.* 2023, 13:17816) run the same sweep on prediction error
over nine training-set sizes and report errors "constantly small, independent of training set size" in
the sub-range that dominates their data. Of the three, only ROGI reports an index as a function of `n`, and on
synthetic surfaces and curated libraries rather than on activity data at the sizes measured here. We
state all the readings and assert that none is settled by this measurement.

### 4.4 What we deliberately did not do

We fitted no scaling exponent and no power law, and the reason is evidential. With three points and a
two-parameter family, a deliberately unrelated quadratic in `log(1/f)` fits the measured movement as
well as the classical power law does, and the implied dimension swings from 9.5 to 94.8 under
perturbations that the measured between-chain spread of 0.0456 makes ordinary. A pre-registered
exponent comparison would have manufactured a confirmation.

---

## 5. The floor against fitted models, and a disagreement about scale

### 5.1 A baseline audit on our own claim, which reversed it

We had claimed the kNN floor beats fitted models. It does not, and the audit that showed this is
reported here rather than quietly dropped, because the subject is the one Janela & Bajorath's **2022**
title names: "Simple nearest-neighbour analysis meets the accuracy of compound potency predictions
using complex machine learning models" (*Nature Mach. Intell.* 4:1246–1255). Their 2023 *Sci. Rep.*
paper, which section 5.3 reanalyses, is titled differently and concludes something different.

The model arm we had used averaged histogram gradient boosting with a multilayer perceptron and
**excluded the support-vector machine** that our own code comments called the real bar. Per-learner
mean skill on ChEMBL-40 is SVM +0.4221, gradient boosting +0.3969, MLP +0.2446, and cross-validation on
the training split alone selects the SVM on 34 of 40 ChEMBL and 30 of 30 MoleculeACE targets. The arm
averaged the second-best learner with the worst and left out the one that wins. **The choice of
baseline was worth +0.1025 [+0.0899, +0.1156] of skill — larger than every scientific effect this paper
reports** — and nothing in the repository recorded that a choice had been made.

Against the corrected arm. The instrument gate confirming that the floor reproduces our earlier
numbers has **two rows, not one**: on the 12 targets whose earlier values were computed it agrees to
5.6 × 10⁻¹⁷, and on the 28 whose values were recoverable only at three stored decimals it agrees to
4.3 × 10⁻⁴ against a tolerance of 5 × 10⁻⁴ — the looser row a limit of what the earlier artefact
stored, not of the recomputation. An earlier version of this paper quoted only the tighter row.

| | skill | absolute RMSE | floor ahead on |
|---|---|---|---|
| ChEMBL-40, floor − cross-validated learner | −0.0754 [−0.0865, −0.0639] | −0.0852 | 3 / 40 |
| MoleculeACE-30 | −0.0949 [−0.1084, −0.0806] | −0.1171 | 1 / 30 |

Both scales appear because the pre-registration that produced section 5.2 forbids reporting the
normalised scale without the absolute one beside it, and because section 5.3 is about exactly that
choice.

### 5.2 Half of that gap was the floor's own unexercised freedom

The audit fixed the model arm and left the floor at a fixed `m = 10`, which is the same error pointed
the other way. Giving the floor its own symmetric selection — the one-parameter rank kernel of
section 6, with `m` and α chosen by cross-validation on the training split, exactly as the model arm
chooses its learner — recovers half the gap and does not close it:

| | skill | absolute RMSE | floor ahead |
|---|---|---|---|
| uniform floor at `m = 10` | −0.0754 [−0.0865, −0.0639] | −0.0852 | 3 / 40 |
| **reweighted floor, both arms selecting on train** | **−0.0394** [−0.0501, −0.0289] | −0.0449 | 5 / 40 |
| recovered, of which choosing `m` | +0.0179 [+0.0114, +0.0242] | +0.0197 | 31 / 40 |
| recovered, of which choosing the weighting | +0.0181 [+0.0146, +0.0218] | +0.0205 | 37 / 40 |

On MoleculeACE-30, −0.0949 becomes −0.0474 [−0.0586, −0.0360] in skill and −0.1171 becomes −0.0596
[−0.0741, −0.0453] in absolute RMSE, with the floor ahead on 3 of 30. **48 % and 50 % of the gap
recovered**
on two collections of different size and framing, and the recovery splits almost exactly evenly between
the two freedoms — 0.0002 apart, with a decomposition residual of exactly zero.

So half of the gap we reported was the floor's own unexercised freedom, and we no longer imply that the
whole of it is information outside the neighbourhood. What remains excludes zero on both collections.
Selection is symmetric in *form* here but not in *search cost* — the floor arm searches 40
cross-validation cells against the model arm's 3, with a measured selection regret of 0.0040 — and that
residual asymmetry points against the floor arm, so it cannot explain what survives.

### 5.3 Their own conclusion and ours agree, once the scale is named

Janela & Bajorath report that "differences in mean prediction errors of all methods were confined to
~ 0.1 units and thus essentially negligible". **Their conclusion is not that the methods are
equivalent**, and an earlier version of this paper attributed that to them wrongly: they conclude that
"conventional benchmark settings do not provide a realistic assessment of compound potency prediction
methods", and they attribute the flatness to the intermediate-potency compounds near each dataset's
median, whose errors are small for every method and which dominate the aggregate.

That is a claim about benchmark design, and it points the same way as what follows. They say the
benchmark hides differences between methods; we show one scale on which a difference is visible in
their own predictions. We do not reproduce their numerical tie on that scale. Their per-compound predictions are
published, so we recomputed their own table from their own predictions — all 1920 published rows
reproduce to 5.6 × 10⁻¹⁶ — and then computed a variance-normalised error on the same predictions:

| pair, their 8 targets × 10 splits | their MAE | normalised skill |
|---|---|---|
| **SVR − kNN** | **+0.0516** | **+0.0757** [+0.0601, +0.0913], 8 / 8 targets |
| SVR − 1-NN | +0.1088 | +0.1594 |
| SVR − median regressor | +0.4677 | +0.3974 |

For the pair we compare, their claim reproduces with room to spare: 0.0516 is about half their stated
bound. The same 0.0516 of mean absolute error is 0.0757 of normalised skill — larger than every
scientific effect in this paper. Our own gap on 40 different targets with a different control and a different model arm, **as section
5.1 measured it with the floor at a fixed `m = 10`**, was −0.0754; the magnitudes agree to three
decimal places and the signs differ only by direction convention. Two cautions attach to that
coincidence. It was **not pre-registered**, so we report it as an observation rather than a confirmed
hypothesis. And −0.0754 is the figure section 5.2 has just **superseded**: our current estimate, with
the floor given the same selection freedom as the model arm, is −0.0394, and against that the
magnitudes no longer coincide. The comparable quantity is theirs with a fixed control against ours
with a fixed control, which is what is tabulated above; a reader who prefers the symmetric comparison
on both sides should take 0.0757 against 0.0394 and conclude only that the two are of the same order.

**Why the metric choice matters.** On any test set, with `b = sd(pred)/sd(y)` and
`d = (mean y − mean pred)/sd(y)`,

    R² = r² − (r − b)² − d²                                                          (3)

identically — verified against their own published R² column to 1.9 × 10⁻¹⁵ on all 379 non-constant
cells. So `r² = R² + penalty` with `penalty ≥ 0`: the squared Pearson correlation is the coefficient of
determination plus a non-negative calibration term, and being invariant to affine rescaling of the
prediction it cannot see that term.

This decomposition is **not ours**: it is Murphy's mean-square-error skill-score decomposition, long
standard in forecast verification, and we cite it as such rather than present it as apparatus.

Their Results report that `r²` — which their Methods define as the squared Pearson coefficient — "did
not lead to a larger separation between the performance of different models". The coefficient of
determination, which can see the calibration term, **does appear in their Results, but as an
alternative training cost function** whose outcome they report as nearly identical, not as a reported
performance value; it is also in their repository on all 400 cells. An earlier version of this paper
said it was absent from their text, which is wrong. The narrower and correct statement is that the
quantity reported as *performance* throughout is the affine-invariant one.

**None of this says their numbers are wrong** — they reproduce exactly — nor is the calibration spread
across their arms evidence about their modelling choices: an unweighted neighbour rule has `b ≈ 1` by
construction (measured 0.9864) and a regularised kernel machine shrinks `b` toward `r`, so the 26-fold
penalty spread across their arms was certain before the data were read. Nor do we claim which scale
answers which question: if a 0.1 log-unit margin sits below assay error, the absolute scale is the
right one for "should a medicinal chemist prefer the complex model", and a normalised restatement
answers something else. **What is established is that the two scales give different-looking answers on
identical predictions**, and we therefore report both for every model-against-floor comparison above.

---

## 6. What the neighbourhood's internal structure buys

### 6.1 A vacuity theorem, and a 1978 result beside it

With `c_i = E[z_v z_{u_i}]` and `C_ij = E[z_{u_i} z_{u_j}]` over a compound's `k` neighbours, the best
linear predictor is `w* = C⁻¹c` with `MSE* = 1 − cᵀC⁻¹c`. If the neighbourhood is **second-order
exchangeable**, then `c = r·1` and `C = (1−ρ)I + ρ11ᵀ`, so `1` is an eigenvector of `C` and `c` is
parallel to it. Hence, with `β = kr/S` and `S = 1 + (k−1)ρ`:

- every optimal weight is **exactly equal**, `w* = (r/S)·1`;
- the weight sum is the regression slope of the target on the neighbourhood mean, `Σw*_i = β`;
- the whole gap is a squared scale error, `MSE_mean − MSE* = Var(mean)(1−β)²`;
- and fitting one scalar on the plain mean attains the `m`-dimensional optimum exactly.

Two independent derivations and a numerical sweep over 2000 random `(k, r, ρ)` agree to 1.7 × 10⁻¹⁵.

A neighbouring result belongs here, which four of our prior-art passes missed and the fifth found.
Bailey & Jain (*IEEE Trans. SMC* 1978, 8:311) proved that for **fixed k** the **asymptotic
classification** error of the **unweighted** kNN rule is minimal among weighted kNN rules, for all
distributions. The two statements share the conclusion that weighting cannot help and differ in what
is assumed and concluded: theirs is asymptotic, about classification, distribution-free at fixed `k`;
ours is finite-sample, about the squared error of a linear predictor, under second-order
exchangeability. We state both and do not assert that either subsumes the other.

**Scope, stated because a reader will press here.** The theorem bounds predictors linear in the labels
of a fixed neighbour set under a fixed metric. It says nothing about a model reading the query
molecule's own features, and a two-line counterexample makes that explicit: with a latent shared
coordinate, exchangeability holds exactly and the optimal weights are equal to 3 × 10⁻³, `MSE*` is
0.5466 against the plain mean's 0.5502 — one scalar of headroom, as the theorem says — while a
predictor reading the query's own features attains zero. **The theorem is therefore not an explanation
of why fitted models tie nearest-neighbour controls**, and we do not claim it is. What it does explain
is narrower and real: why the rank axis was never fitted. Write the natural exchangeable model and the
question "should nearer neighbours get more weight" cannot be posed, because the answer is provably
"no, they are already equal". The residual then reads as a null.

### 6.2 Neighbourhoods are not exchangeable

Measured on molecular data, the second moment between a compound's activity and its `j`-th nearest
neighbour's, averaged over 31 ChEMBL targets at `k = 10`:

| rank | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| `c_j` | 0.714 | 0.660 | 0.629 | 0.597 | 0.572 | 0.542 | 0.523 | 0.513 | 0.499 | 0.476 |

The mean decline is +0.2384 [+0.2165, +0.2603], excluding zero on every collection. The exchangeability
hypothesis is materially false, and nothing we found reports this profile on molecular data.

### 6.3 The whole effect is bought by one parameter

Reweighting by rank is worth +0.0155 [+0.0118, +0.0192] out of sample against the best
training-selected uniform `m` in each arm, positive on 30 of 31 targets. We first reported +0.0379
against uniform at a fixed `m = 10`; **59 %** of that was the effect of fixing `m` — a difference of
0.0224 in 0.0379 — since uniform at `m = 5` beats uniform at `m = 10` on 28 of 31 targets. The corrected figure is the one above.

Four arms, each selected by five-fold cross-validation on the training split alone, with all three
instrument gates reading exactly zero:

| arm | parameters | median `m` | median α | mean skill |
|---|---|---|---|---|
| uniform | 0 | 5 | — | 0.3658 |
| **`w_j ∝ j^−α`** | **1** | **10** | **1.000** | **0.3839** |
| the same times a fitted scale | 2 | 10 | 1.000 | 0.3841 |
| fitted `w = C⁻¹c` | *m* | 10 | — | 0.3823 |

The one-parameter kernel beats uniform weights by +0.0181 [+0.0146, +0.0218] on 37 of 40 ChEMBL targets
and +0.0211 [+0.0171, +0.0257] on 30 of 30 MoleculeACE targets when each arm selects on its own grid,
and by +0.0170 on 39 of 40 and +0.0185 on 28 of 30 on the matched grid; the pre-registration requires
both, because a comparison on each arm's own grid can reward the arm with the larger grid.

Against the `m`-parameter fitted optimum on the matched grid the kernel gives up 0.0009
[−0.0048, +0.0025] of skill, so a pre-registered non-inferiority margin of −0.01 holds with the
interval's lower end less than half the margin from zero. **Three qualifications belong with that, and
an earlier version of this paper carried none of them.** The result is an equivalence, not a win: the
kernel leads on only 19 of 40 targets on the matched grid and 23 of 40 on the own grids, a coin flip.
The deficit exceeds the margin on 3 of 40 targets, worst −0.0565 on `CHEMBL614358_EC50`. And the kernel
pays a larger selection cost than its competitor — 40 cross-validation cells against the fitted arm's
admissible set, with a measured regret of 0.0040 against 0.0006 — so the protocol charges the arm we
report as sufficient more than the one it is compared against. Cross-validation, given eight exponents and never shown a test label, selects **α = 1 on 26 of 40
targets**, with α ∈ {0.75, 1, 1.5} on **39 of 40** and an interquartile range of [1.000, 1.000]. The
two-parameter arm adds +0.0002, positive on 20 of 40: the scale channel is empty, shown rather than
asserted.

### 6.4 Neither arm is ours, and the prior art here is heavy

Itskowitz & Tropsha (*JCIM* 2005, 45(3):777) derived the optimality condition for kNN QSAR as a linear
system on the weights built from label second moments with a Lagrange multiplier for the normalisation
— structurally the fitted arm above, twenty years earlier, in this paper's target venue, by the group
whose index this whole line is built on — and their eq. (21), a normalised `exp(−αd/D)` with α tuned
for performance, is structurally the one-parameter arm. **What their paper does not do** is compare
them: the sentence after their eq. (17) is in the future tense and the next section substitutes the
chosen exponential without solving it, so the optimum is derived and abandoned. Nor does the paper
contain a uniform arm; its reference is its own kernel at α = 1.

Three more attributions belong here, and two of them correct statements we had made. `w_j ∝ j^−α` is
**not** Dudani's rule: Dudani (*IEEE Trans. SMC* 1976, 6(4):325) weights by distance,
`w_i = (d_k − d_i)/(d_k − d_1)`, and the rank reciprocal `1/j` is Gou, Xiong & Kuang
(*J. Computers* 2011, 6(5):833). Hechenbichler & Schliep (2004) contains no rank-indexed kernel at all,
every kernel in it being a function of standardised distance; we cite it for the distance-kernel family
we deliberately do not test. And the comparison against uniform weights is not ours either: Nigsch
et al. (*JCIM* 2006, 46(6):2412) compared arithmetic and geometric averaging against inverse-distance
and exponential weighting on 4119 organic molecules under 25-fold Monte Carlo cross-validation. The
*direction* is older still — Cleveland & Devlin (*JASA* 1988, 83(403):596) head a section "Weight
Functions and the Poor Performance of the Uniform", and the same section reports that the form of the
weight function does not much matter, which is the honest frame for the one-parameter result above. The
closest adjacent prior to that result is Anava & Levy (NIPS 2016), who prove that optimal local weights
depend on a single parameter, the Lipschitz-to-noise ratio; their parameter is derived inside an
assumed model rather than cross-validated, and they compare against no fitted weight vector.

**What survives as ours on this axis** is narrower than we first wrote and still real: the measurement
of the rank profile on molecular similarity neighbourhoods, and the comparison of a one-parameter
kernel against the *fitted* optimum, which no source on either side runs.

---

## 7. Methods

### 7.1 Data and representation

Three collections, 80 targets: 40 ChEMBL targets selected under a frozen plan, the 30 MoleculeACE
targets, and the 10 DeepDelta benchmarks. Compounds are represented by ECFP4 Morgan fingerprints,
radius 2, 2048 bits, and similarity is Tanimoto throughout; the graph is the union-symmetrised
k-nearest-neighbour graph. Where a model is fitted, the split is a single random 80/20 per target from
one declared generator seed, identical across every increment so that per-target values from different
measurements can be paired.

### 7.2 Pre-registration

Every increment carries a YAML pre-registration committed and referenced by its **git blob hash**,
frozen before any measurement and never edited afterwards. Scripts carry the blob in a module constant
and print it first. Design gates are evaluated and reported before any outcome is read, and a failing
gate stops the increment with the discrepancy as the report. The repository carries 442 tests,
including static guards that pin the tie-breaking conventions on which several measurements depend, the
explicit-encoding rule for all text I/O, and undefined-name resolution across every module.

Where a measurement was not blind, the plan says so in its own text and the result repeats it. Two
cases arose. In one, an adversarial reviewing pass measured an increment's primary statistic before the
plan was frozen; the plan records the leak, and its decision margin is anchored to an older committed
figure rather than to the leaked one. In the other, two point estimates were derivable in advance from
already-published columns, and the plan labels them as carrying no evidential weight, placing it on the
paired intervals and sign counts instead.

### 7.3 Six refusals

The project has **six times refused a verdict its own frozen rule granted**, because the design could
not earn it. Five were design gates on separating the two moments of equation (1), with a proved
structural cause: on a stride-`d` circulant with an exponential field, `λ = ρ_nn/r = c(d)` is pinned to
`r` by the field, so no graph construction can separate them. The sixth was refused before freezing,
when the plan's apparatus gate was found already failed by the project's own committed data.

Three further plans were rewritten before freezing after adversarial review found a defect that would
have confirmed the registered direction for the wrong reason. In one, a subsample index set left
unsorted would have missed an instrument gate by ten orders of magnitude, because both neighbour
selectors break similarity ties on the ascending index and a quarter to a third of rows have an exact
tie at the `k`-th neighbour. In another, a plan that failed to exclude a training molecule from its own
neighbourhood would have crippled the competitor arm by 0.1005 — ten times the decision margin, in the
direction that confirmed the prediction. In the third, a pivot prediction was shown to fire in 100 % of
synthetic replicates by construction, and was replaced by the algebraic identity it actually expressed.

---

## 8. Limitations

**The object is classical.** The identity is a Dirichlet energy, the risk is the Wiener and kriging
form, the first moment is Moran's I, the weighted-kNN result is bounded above by Itskowitz & Tropsha
2005 and adjacent to Anava & Levy 2016, and the noise-floor route is the Gamma test. Eleven
adversarially-tested attempts to build a larger claim on this quantity failed, in three recurring ways:
the phenomenon involved models outside the theorem's class, or the extension landed in kriging, or the
question was already answered inside the class. We state this because it bounds what a reader should
expect, and it is the honest reason this is a methods paper.

**One domain, one representation, and the check is weaker than it looks.** Everything is
ECFP4/Tanimoto on molecular activity, and **the index itself was never recomputed on another
representation**: in the one check we ran, the graph and the statistic stay on ECFP4/Tanimoto
throughout and only the *model's* features change, so what varies is the accuracy the index is
correlated against, not the index. Spearman of rank assortativity against attained skill is +0.9154
when the model uses atom pairs, +0.8934 on MACCS and +0.7255 on RDKit physicochemical descriptors,
against +0.8889 on ECFP4 — so the index predicts accuracy for models built on other features, which is
one more than zero and **does not make the claim representation-free**. An earlier version of this
paper stated this more strongly than its own artefact supports, and the artefact's own verdict says so.

**The size result uses random subsampling**, not a realistic acquisition order. Real datasets grow by
medicinal-chemistry series, which plausibly grows homophily less than random growth does. Under uniform
subsampling the movement is fully accounted for by rank dilution, so nothing in section 4 bears on how
a real acquisition order would behave.

**The two moments are nearly collinear here** (Pearson +0.9895), so this data cannot separate their
contributions to ranking, and the regime where it could is the one five refused gates showed is
unreachable at this sample size.

**The floor does not beat the best fitted model.** We make no claim that a simple control suffices, and
the reanalysis in section 5.3 establishes a disagreement about scale rather than a tie.

**SARI's values here are self-referential.** Its definition z-normalises against a reference set, and
ours is this project's own 70 potency targets, so only its ranking is used and no value is comparable
to a published one. Its weight `y_i·y_j/(1+|y_i−y_j|)` also presumes positive log-potency, and 8 of
the 10 DeepDelta targets carry non-positive labels — the minimum over all 80 targets is −25.47 —
where the weight goes negative and the continuity score is not a weighted mean. We therefore excluded
that collection of 10 from the reference set and from every conclusion, and report its values flagged.

---

## 9. Conclusions

The error of the k-nearest-neighbour mean on a molecular similarity graph has an exact closed form in
two graph statistics, and the published continuous modelability index is that form to within about one
percent. One of the two statistics is reported throughout the field and the other is reported nowhere;
omitting it costs a factor of 2.5 in level, and whether it costs anything in ranking is a question this
data cannot answer, the two moments correlating at +0.9895 here.

The quantity all of these indices estimate is not a property of a dataset. It moves by roughly 0.26 in
both statistics over an eightfold change in sample size, on every target we measured, and the 2014
binary index crosses its own published threshold on nearly half of its cells by growing alone. That a
roughness index depends on sample size is documented in the literature that introduced one, which
reports a convergence figure and a bootstrapped interval; the movement here is the classical `k/n`
rank-dilution channel, for which we claim nothing. What we claim is the gap between knowing and
reporting: the dependence is of a size that crosses the field's own threshold, it is predictable from
the full-sample ordering without resampling, it has not been reported for the MODI family, and the
practice of tabulating one number per dataset against a fixed cut-off is unchanged.

Along the way we withdrew one of our own claims entirely, halved another, and found that a prominent
disagreement in this literature is a disagreement about which scale to report rather than about what
the data say. We report those as results because they were produced by the same frozen-plan procedure
as everything else, and because a methods paper whose method only ever confirms its author is not
demonstrating a method.

---

## Data and code availability

All pre-registrations are in `prereg/`, referenced by git blob hash and never edited after freezing.
All measurements are in `results/`, each with the blob of the plan that produced it. The reanalysis of
section 5.3 uses another group's published artefacts, retrieved and documented but not redistributed.

## Author contributions and provenance note

This draft was produced with substantial assistance from a large language model operating under the
frozen-plan procedure described in section 7.2. The procedure exists because the model's unassisted
output repeatedly confirmed its own hypotheses: of the last four plans it drafted, three contained a
defect that adversarial review showed would have produced a confirmation for the wrong reason. Every
number in this paper comes from a script committed before it produced a value, under a plan frozen
before the script existed. Readers who find that insufficient are invited to check the blob hashes.

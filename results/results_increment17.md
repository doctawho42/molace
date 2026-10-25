# Increment 17: the fifth prior-art pass, remaining scope

Blocking experiment 4 of `docs/paper/skeleton.md`, and items 3 and 5 of `prior-art.md`'s
"Required before drafting". **No measurement; this is a reading pass**, run adversarially across three
lenses with every reported collision verified against the primary source by a second reader.
**21 collisions adjudicated, 16 survived, 5 refuted.**

**Headline: no fatal collision. Claim 5 survives, and so does section 5.2's measurement.** The cost is
five corrected attributions, two of them to sentences this project wrote yesterday.

## The lens the project itself called its highest-stakes unsearched collision came back clean

`prior-art.md` item 3 reads: "One more pass specifically for closed-form or spectral treatments of
MODI, which is the single collision that would cost claim 5." It has now been run, and the clearance
is positive rather than merely empty — **MODI's own paper was read in full**:

> Golbraikh, Muratov, Fourches & Tropsha, *Dataset Modelability by QSAR*, JCIM 2014, 54(1):1-4, at the
> open NIH deposit PMC3984298. Its only equation is `MODI = (1/K) Σ_i N_i^same / N_i^total`. No
> algebraic or closed-form expression, no autocorrelation, homophily, graph, Laplacian or spectral
> language anywhere, no formula for prediction error — it only notes that `QSAR_CCR` "could be also
> estimated from a formula similar to" eq. 1 — and no claim or analysis of dependence on dataset size.
> The originating lab's own package (`github.com/molecularmodelinglab/MODI`) adds only the equivalence
> to the uniform-class average of leave-one-out 1-NN accuracy.

So the accounting in section 2 is not pre-empted, and section 4's size-dependence result is not
pre-empted from this direction either.

### But claim 5's framing takes four corrections

**`m1` is Moran's I, and the project should say so.** Section 2 currently says "`m1` *is* Newman's
target assortativity, the statistic graph-learning benchmarks already report". Arthur,
*EPJ Data Science* 2025, 14:1, eq. (2), gives Moran's I as `I = (N/|W|) · (z·z̃)/|z|²` with `z̃_i` "the
average value of `z` over the neighbours of `i`", and states that for a row-normalised adjacency
matrix `|W| = N` so the constant cancels. That is `m1`. The paper also says "Unlike assortativity, the
Moran index does not include [the degree correction]", so the two are **not** the same statistic in
general and the identification needs its condition stated. The spatial-autocorrelation literature owns
this object, `1 − m1` is Geary's C in the same family, and the paper should cite it rather than present
the identification as its own observation.

**The Dirichlet-energy concession is imprecise.** Section 2 concedes that "the right-hand side is a
Dirichlet energy". The Heterophilic Graph Learning Handbook (arXiv:2407.09618 §2.1) defines
`E_D(x) = xᵀLx = ½ Σ_{(i,j)∈E} (x_i − x_j)²`, which is the **first-moment** object. The identity's
right-hand side carries `ρ_nn` as well, so the concession as written gives away less than it should and
is also not exactly true of the whole expression. State which term is the Dirichlet energy.

**The graph-average predictor is the harmonic energy minimiser, and that is 2003 SSL.** Zhu,
Ghahramani & Lafferty, ICML 2003, state the harmonic property as "the value of f at each unlabeled data
point is the average of f at neighboring points", `f(j) = (1/d_j) Σ_{i~j} w_ij f(i)`, and that "the
minimum energy function is harmonic" with `Δ = D − W`. So "the kNN-mean predictor minimises a Dirichlet
energy" is standard in semi-supervised learning. Adjacent rather than pre-empting — they minimise the
energy to *impute* labels, the project *measures* the error of the average — but it must be cited.

**The published-index enumeration is incomplete.** Cui et al.'s activity-cliff-aware graph learning
paper defines a **Label Incoherence Index**,
`LII = (1/n) Σ_i (1/k) Σ_{j∈N_i} |y_j − y_i|`, on the kNN graph at k=10 with uniform weights,
"excluding `x_i` itself". Sections 1 and 3 enumerate MODI, MODI_q2, RMODI, ROGI and the kernel
variants; LII belongs in that list. It is an L1 first-moment index, so section 3's claim that the
**second** moment is absent from the published indices survives — LII does not contain it either — but
the enumeration must name it.

Refuted on this lens: kernel-target alignment (Cristianini, Shawe-Taylor, Elisseeff & Kandola, NIPS 14)
as a closed-form modelability collision.

## Two sentences this project wrote yesterday are wrong

`prior-art.md` section 7, written 2026-10-25, says increment 16 contributes "the uniform baseline
neither 2005 nor 2011 provides", and `results_increment16.md` says under prediction 2 that "nobody in
the kNN-QSAR line has measured this". **Both are false.**

> Nigsch, Bender, van Buuren, Tissen, Nigsch & Mitchell, *Melting point prediction employing
> k-nearest neighbor algorithms and genetic parameter optimization*, **JCIM 2006, 46(6):2412-2422**:
> "To compute the prediction on the basis of the melting temperatures of the nearest neighbors, we used
> four different methods (**arithmetic and geometric average**, inverse distance weighting, and
> exponential weighting), of which the exponential weighting scheme yielded the best results." On 4119
> organic molecules plus 277 drugs, under 25-fold Monte Carlo cross-validation, with the number of
> neighbours also varied.

That is a uniform arm, in the kNN-QSAR line, in this paper's target journal, compared against three
weighting schemes, twenty years ago. Increment 16's prediction 2 is therefore a **replication on
potency data of a melting-point result**, not a first measurement, and the corrected statement is that
nobody in the *modelability* line had measured it and nobody in either line had compared a
one-parameter kernel against the **fitted** optimum.

## And a kernel beating uniform weights is 1988 statistics

> Cleveland & Devlin, *JASA* 1988, 83(403):596-610, §10.4, whose heading is literally **"Weight
> Functions and the Poor Performance of the Uniform"**: "the uniform weight function, with the
> discontinuity at 1, performs poorly", while "asymptotic results for nonparametric regression show
> that the overall form of the weight function does not have an appreciable effect with respect to mean
> squa[red error]".

Both halves matter to section 5.2. The first says the kernel-beats-uniform direction was known. The
second says the *insensitivity to the kernel's form* was also known — which is the honest frame for
increment 16's finding that one parameter attains the fitted optimum. The same insensitivity is
reported in QSAR by Gajewicz-Skretna et al.'s kernel-weighted local polynomial regression: "the
particular form of the function, K has only a relatively small effect on estimation accuracy."

## The closest thing to increment 16's headline, and it is 2016

> Anava & Levy, *k\*-Nearest Neighbors: From Global to Local*, **NIPS 2016** (arXiv:1701.07266): "we
> offer a simple approach to locally weighted regression/classification, where we make the
> bias-variance tradeoff explicit. Our formulation enables us to phrase a notion of optimal weights,
> and to efficiently find these weights as well as the optimal number of neighbors". And §3: "the
> optimal weights depend on a **single parameter** `L/C`, namely the Lipschitz to noise ratio."

Increment 16's headline is that one parameter attains the m-parameter optimum. Anava & Levy prove that
the optimal local weights are *determined* by one parameter, under a Lipschitz-plus-noise model. This
is **adjacent, not pre-empting** — their one parameter is a derived ratio in an assumed model and is
not an exponent fitted by cross-validation, and they neither measure molecular neighbourhoods nor
compare against a least-squares-fitted weight vector — but it is the paper a reviewer will raise, and
section 5.2 must cite it and say exactly what differs. The rank-indexed analogue is Demirkaya, Fan,
Gao, Lv, Vossler & Wang, *JASA* 2024, 119(545):297-307, whose distributional-NN estimator assigns
monotone rank weights fixed by a single subsampling scale `s`.

Also adjacent, and now cited: ALL-QSAR (Zhang, Golbraikh, Oloff, Kohn & Tropsha), a Gaussian
**distance** kernel `exp(−d²/2K²)` with the width swept 0.01 to 1.00 and no fixed neighbour count; and
GenRA read-across (Helman, Shah & Patlewicz, *Comput. Toxicol.* 2018), similarity-weighted averaging
with Jaccard similarity at a fixed k = 10.

### What nothing pre-empts

The lens searched specifically for a prior statement of section 5.2's **measurement** and found none:
no source reports the rank-resolved cross-moment `c_j = E[z_v z_{u_j}]` as a function of `j` on
molecular data, and none states that molecular neighbourhoods fail second-order exchangeability.
Searched: locally weighted regression, lazy learning and local QSAR; the similarity-property-principle
literature with rank-resolved framing; and the activity-cliff and activity-landscape geometry
literature.

## Janela & Bajorath: the protocol differences are now specific, and the metric is the big one

Nine collisions reported on this lens, **four refuted**. What survived changes section 4a from an open
question with a vague answer to an open question with named differences.

**Their metric is unnormalised MAE.** *Pharmaceuticals* 2023, 16(4):530 §4.5 defines one metric: "the
mean absolute error (MAE) was calculated by comparing predicted and experimental test compound potency
values", and that is the only metric on all 376 activity classes. *Sci. Rep.* 2023 adds RMSE and
**squared Pearson** `r²` on eight classes. Neither paper reports a variance-normalised error as a
performance value. This project's skill is `1 − RMSE_arm / RMSE_baseline` with the baseline predicting
the training mean, so the two literatures report on **different scales**, and a margin that is
"negligible" in absolute potency units need not be negligible after normalisation.

**Stated carefully, because the first version of this finding overreached.** Converting between the two
needs an assumed error distribution and a potency spread, so it is an order-of-magnitude statement and
not an equality. Their own paired figures supply the conversion factor rather than borrowing it:
*Sci. Rep.* reports "Lowest prediction errors detected were ~0.4 and ~0.5 for MAE and RMSE" and "MAE
and RMSE values generally smaller than 0.8 and 1.0", both ratios 1.25, which is the Gaussian
`RMSE/MAE = √(π/2) = 1.2533`. Under that factor a 0.1 MAE margin maps to 0.125 of skill at a potency
standard deviation of 1.0 and 0.084 at 1.5, a range that brackets increment 14's measured gap of
0.0754 to 0.0949. **So the two gaps are of the same order.** Three things are deliberately not claimed:
that this resolves the comparison, that the project's skill equals sklearn's `r2_score` — it does not,
being a root-error ratio against a train-mean baseline rather than a squared-error ratio against the
test mean — and any particular potency standard deviation, since the figure the first version quoted
could not be reproduced from their published per-class statistics.

**Their 2022 control used Euclidean distance on binary fingerprints.** Their public code builds it as
`neighbors.KNeighborsRegressor()` with a grid of `{"n_neighbors": [1, 3, 5]}` only, under
`GridSearchCV(..., cv=10, scoring="neg_mean_absolute_error")`, on
`GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)`. With no metric argument scikit-learn defaults to
`minkowski, p=2` — Euclidean, the square root of Hamming distance on binary vectors — and
`weights='uniform'`. Their **2023** repository instead sets `metric='jaccard'` explicitly. So the
control in the paper whose tie this project does not reproduce is an unweighted kNN over at most five
Euclidean-nearest neighbours, against this project's uniform mean over ten Tanimoto-nearest ones.

**Their model arm was done right.** *Pharmaceuticals* 2023 §4.3.1: "For SVR, the regularization
hyper-parameter C was determined by testing (0.001, ..., 10,000) values. SVR models were derived using
the **Tanimoto kernel**", and their 2022 code adds a deep arm with a 54-point grid. So the arm error
increment 14 found in this project's own earlier work — averaging gradient boosting with an MLP and
excluding the SVM — is not an error their line makes, and section 4a must not imply the field generally
makes it.

**Their mechanism is label-marginal, not graph-structural**, so it does not overlap sections 2 or 4.
*Sci. Rep.* 2023: "in the intermediate potency sub-range, predictions yielded artificially low errors,
due to narrow local potency value distributions around the median. In original activity classes, the
majority of compounds fell into the intermediate potency range, which strongly dominated global potency
predictions."

**They do report a size curve, and it reads the other way.** *Sci. Rep.* 2023 built "nine training sets
of increasing size ... 6, 12, 18, 30, 48, 78, 126, 204, and 330 compounds" over ten trials, and found
that in the dominant sub-range "the prediction errors were constantly small, **independent of training
set size**". `prior-art.md` section 6 says nobody in the modelability line reports an index as a
function of `n`; that stands, since they report prediction error rather than an index, but this is the
second time a cited line turns out to run the same sample-size sweep as section 4 with the opposite
reading, the first being the Gamma test's M-test. Section 4 should name both.

**Clean, and specifically so.** Searching the full text of both open-access papers for "modelab",
"MODI", "ROGI", "Golbraikh", "Tropsha", "assortativ", "homophil", "Dirichlet", "roughness" and
"activity cliff" returns **zero occurrences of every one of them in both papers**. Neither computes,
cites or alludes to a modelability index, a roughness index, graph assortativity, edge homophily or a
Dirichlet energy. Sections 1, 2, 3 and 5 are untouched by this line.

Refuted on this lens: that they report SVR ahead rather than a tie; that their `k` grid contains this
project's CV choice; that their nearest-neighbour-removal intervention is unaddressed here; and that
the roles of their two 2023 papers are reversed in increment 14's pre-registration.

## What this costs and what it leaves

| | before the pass | after |
|---|---|---|
| claim 5, the identity's accounting | "object known, accounting new" | **survives**, with `m1` re-attributed to Moran's I and the Dirichlet term named precisely |
| section 5.2, the measurement | first on molecular data | **survives**, nothing found |
| section 5.2, kernel beats uniform | "nobody in the kNN-QSAR line had measured this" | **wrong**: Nigsch 2006, and the direction is Cleveland & Devlin 1988 |
| section 5.2, one parameter suffices | own contribution | **adjacent prior**: Anava & Levy 2016 derive one-parameter optimal local weights |
| section 4a, the unreproduced tie | "their control is 1-NN/kNN with their choices" | **named**: MAE not skill, Euclidean not Tanimoto, k ≤ 5 not 10 |
| section 1 and 3, index enumeration | five indices | **incomplete**: add the Label Incoherence Index |

The fifth blocking experiment — running their protocol on their public data at
`github.com/TiagoJanela/ML-for-compound-potency-prediction` — remains the only thing that settles
section 4a, and it now has a specification: it must report MAE beside skill, so the comparison is
available on their scale as well as this project's.

## Method note

Three lenses, each required to quote the deciding sentence from a primary source, with every reported
collision handed to a second reader whose instruction was to refute it. Five were refuted, including
two whose quotations were genuine but came from the wrong paper for the claim they were aimed at. The
verification caught three things in the surviving findings that would have gone into the paper as
errors: an equation of this project's skill with sklearn's `r2_score`, a potency standard deviation
that could not be reproduced from the source's own published statistics, and two code locations that
could not be found in the repository they were attributed to. None of the three is carried above.

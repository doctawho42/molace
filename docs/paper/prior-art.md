# Prior art, for a JCIM submission

Compiled 2026-10-05. Venue fixed as JCIM, no co-authorship. Four search passes across
cheminformatics, geostatistics, graph signal processing and the graph-homophily line.

The pass changed the paper twice. Read the verdict table at the end before anything else.

## 1. Cheminformatics: the modelability line. This is the one that matters

**Golbraikh, Muratov, Tropsha and co-workers, "Dataset Modelability by QSAR", JCIM 2014, 129
citations.** Introduces MODI, "an activity class-weighted ratio of the number of nearest-neighbor
pairs of compounds with the same activity class versus the total number of pairs", and reports a
threshold of 0.65 separating modelable from non-modelable datasets over 100+ datasets.

MODI is **edge homophily on a 1-NN graph**. The project's target assortativity is the continuous
analogue of the same quantity. This is published in the target venue, by the people who defined the
area, twelve years ago.

**Golbraikh et al., "Modelability Criteria", 2014.** Extends MODI to continuous endpoints:
`MODI_q2` is leave-one-out q2 of similarity search with varying numbers of nearest neighbours, and
the paper reports "high correlation between the external predictivity of QSAR models (QSAR_R2) and
each of the MODI_q2 and MODI_ssR2".

`MODI_q2` **is the project's kNN floor skill**, and its correlation with what models achieve **is the
project's claim 1**. Published 2014.

**Ruiz et al., JCIM 2018** (weighted modelability and rivality indexes) report r2 above 0.9 between a
weighted modelability index and the achieved classification rate, and **Ruiz et al., Molecules 2018**
extend the same indexes to applicability domain. **Kausar and Falcao, J. Cheminform. 2018** build
modelability scoring into an automated QSAR pipeline to skip datasets not worth modelling.

**Consequence.** "A cheap neighbourhood statistic tells you whether modelling is worth it" is a
mature, named programme in this journal. The project cannot present it as a finding. Everything in
the paper has to be positioned relative to MODI, and MODI has to become a measured baseline, not a
citation.

**What is not in that line, as far as four passes went:** nobody writes MODI down in closed form.
The index is defined operationally and validated empirically. The identity

    MSE_floor / sigma^2  =  1 - 2 m1 + m2,     m_j = y' A_norm^j y / y'y

says the continuous modelability index *is* a Dirichlet energy, that `m1` is exactly target
assortativity, and that a second moment is missing from every version of the index. That is a
contribution to the MODI line rather than a competitor to it.

**Measured, increment 11.** MODI_q2 was computed on all 80 targets and compared against the identity's
right-hand side. Mean absolute difference 0.0103 on ChEMBL-40, 0.0083 on MoleculeACE-30, 0.0123 on
DeepDelta-10; Pearson 0.9975 / 0.9960 / 0.9975. So the closed form is not an analogy: it reproduces
the published index to about one point of q2 without running a single leave-one-out prediction. The
2014 binary index, computed on MoleculeACE's own cliff flag, runs 0.693 to 0.809 and clears the
published 0.65 threshold on 30 of 30 targets, while attained skill over those same targets spans
+0.147 to +0.626 — the binary threshold saturates where the continuous quantities still separate.

## 2. Cheminformatics: landscape roughness

ROGI (Aldeghi, Graff et al., JCIM 2022) is already the project's declared competing baseline and is
handled: assortativity over ROGI +0.846 [+0.678, +0.928], ROGI over assortativity covers zero.
SALI and SARI are the older per-pair and global roughness indices. Activity cliffs: **Dablander et
al., J. Cheminform. 2023** confirm QSAR models systematically fail on cliffs and that giving the
model one of the two activities changes everything, which is the project's label-access argument in
another form.

**Consequence.** ROGI alone is no longer a sufficient baseline table. The paper needs ROGI *and*
MODI, and ideally SARI.

## 3. Cheminformatics: is the kNN baseline actually weak?

**Wu et al., Brief. Bioinform. 2020**, 16 algorithms on 14 datasets, rank the field
rbf-SVM > XGBoost > rbf-GPR > Cubist > GBM > DNN > RF > pca-ANN > MARS > linear-GPR ~ KNN > ... So
the received view is that kNN sits near the bottom. Lenselink et al. 2017 and Koutsoukas et al. 2017
reach the same conclusion in classification.

**Consequence, and it is favourable.** The project's claim 4, that the kNN mean beats fitted models
on the hardest third 12 times out of 13, runs *against* the received ranking. It survives because it
is conditional: the ranking above is an average over datasets, and the claim is about the low-smooth
end. That conditional structure is the novelty, and the literature above is the foil that makes it
worth stating.

## 4. Geostatistics: the noise floor is the nugget

The two-statistic estimate `1 - nu = r^2 / rho_nn` extrapolates label correlation at hop 1 and hop 2
back to hop 0. That is nugget estimation, and the known cautions are the project's own findings.

- **Atkinson, Int. J. Remote Sensing 1997**: the nugget is the most appropriate estimator of
  measurement error, but "it is impossible to account for the form of the variogram near the
  ordinate when selecting a mathematical model", so it is less reliable than previously thought.
  The project's geometric-decay assumption is exactly such a choice of form, and the measured
  mixed-hop bias of -0.222 at k = 10 is that caution quantified.
- **Kim et al., J. Nonparametric Statistics 2010**: asymptotics for a nonparametric nugget estimator,
  with an explicit regime of strong dependency where it degrades. Structurally the project's
  ill-conditioning.
- **Zimmerman, Environmetrics 1991**: network design criteria aimed specifically at estimating the
  nugget-to-sill ratio precisely. The project's five refused design gates, thirty-five years earlier.
- **The Gamma test, and its M-test.** Added 2026-10-23 after an adversarial prior-art pass on
  increment 15's draft plan. The paper already cites the Gamma test twice in section 7 as the
  training-free route to a noise floor (doi 10.1098/rspa.2002.1010, "A proof of the Gamma test";
  companion 10.1098/rspa.2002.1011 on asymptotic near-neighbour distance moments, which is the better
  citation for the rank law section 4's null arm realises). What the pass found is the **M-test**: the
  same line ships a named, implemented, documented procedure for recomputing the statistic over
  growing samples, which is precisely increment 15's operation. The R package `sr` (Haythorn,
  crediting Antonia Jones as the Gamma test's principal creator) documents `get_Mlist` as
  "investigates the effect of sample size by calculating Gamma on larger and larger samples", and
  reads the result as convergence: "the graph will become stable when the sample size is large
  enough", "Gamma will converge on the true noise in the relationship as sampling density on the
  function increases". **Its standard reading is the opposite of section 4's.** The M-test treats the
  sample-size curve as approach towards a truth; section 4 treats it as evidence that the quantity is
  not a property of the data. Section 4 now states both readings and asserts that neither is settled
  by the measurement.

**Consequence.** Section 5 of the paper is a replication, in a setting with hop counts instead of
distances, of a known estimator and its known failure modes. It is still worth reporting, because the
graph-ML and cheminformatics literatures do not cite this one, but it must be framed as replication
and must cite these three.

## 5. Graph signal processing and the homophily line

Dirichlet energy as a homophily and smoothness measure is standard; it appears as a workhorse metric
in the Heterophilic Graph Learning Handbook (arXiv 2407.09618), with the usual reading that low
energy means the signal is mostly low-frequency. Adjacency-polynomial filters and their spectral
analysis are classical.

Platonov et al. (adjusted homophily, NeurIPS 2023) and Mironov and Prokhorenkova (unbiased homophily,
LoG 2024) are the measures the project actually uses; GraphLand and GraphPFN are where target
assortativity is reported for regression datasets.

Bridging work exists but points the other way: GNNs applied to kriging (arXiv 2401.12681), Moran's I
used as an auxiliary signal in GNN training, spatial heterophily (Zhou et al., KDD 2023). The graphs
to spatial-statistics direction is travelled; the measure-to-nugget-theory direction is not.

**Consequence.** The identity's mathematics is a known object in different notation. Claim it as
accounting, not as theory.

## 6. Nonparametric statistics: the sample-size dependence of a kNN estimator

Added 2026-10-23 after an adversarial prior-art pass on increment 15's draft plan, which proposed to
lead section 4 with rank dilution as the project's own closed-form account of why a modelability index
moves with n. It is not the project's. **That n enters a kNN estimator only through `k/n`, by way of
neighbour-rank order statistics, is a standard result**, and the plan now forbids claiming novelty for
the mechanism.

- **Samworth, Annals of Statistics 2012, 40(5):2733** (arXiv:1101.5783), Theorem 1: the regret is
  `gamma_n(w){1+o(1)}` with `gamma_n(w) = B1 sum_i w_ni^2 + B2 (sum_i alpha_i w_ni / n^{2/d})^2` and
  `alpha_i = i^{1+2/d} - (i-1)^{1+2/d}`, which the paper calls the variance and squared-bias
  contributions. For the unweighted k-NN rule the alpha sum telescopes to `k^{1+2/d}`, so the
  squared-bias factor is exactly `(k/n)^{2/d}`: n appears only through `k/n`, that is only through
  which ranks are read. This is the channel increment 15's null arm isolates. The same paper is
  already the theory citation for section 5.2's weighted-kNN result, so the collision is with a
  reference the paper carries rather than with an unknown one.
- **Near-neighbour distance moments**: `E[eps_k^a(x)] <= C (k/(gamma n))^{a/D}` for the k-th
  nearest-neighbour distance, citing Evans, Jones & Schmidt, *Proc. R. Soc. A* 2002, 458:2839-2849 for
  the same moments. The distance-side statement of the identical `k/n` dependence.

**Consequence.** Section 4 keeps the measurement, the accounting and the consequence for reporting
practice, and cites the mechanism. What survives as the project's own is narrower and still real: that
the field publishes this index as a bare point estimate per dataset although its sample dependence is
predictable in closed form, and that the 2014 binary criterion crosses its own published threshold on
46 % of cells by growing alone. Nobody in the modelability line reports an index as a function of n,
and four prior-art passes before this one did not find anyone who does.

## 7. The fifth pass: weighted kNN. Run 2026-10-25, owed since the weighted-kNN result existed

Items 1 and 5 of "Required before drafting" and blocking experiment 2 of the skeleton both scheduled
a pass on weighted kNN, and four earlier passes had not run it. It was run as part of increment 16's
pre-freeze review. **Every citation in that increment's draft plan was wrong about what it cites, and
the pass found the paper that matters.**

- **Itskowitz, P.; Tropsha, A. k Nearest Neighbors QSAR Modeling as a Variational Problem: Theory and
  Applications. J. Chem. Inf. Model. 2005, 45(3), 777-785** (doi 10.1021/ci049628+). **PRE-EMPTS BOTH
  OF INCREMENT 16'S ARMS.** Their eq (3) predicts `y_i = Σ_j y_j F(d_ij)` with "a weighting function
  dependent only on distance", normalised by eq (4). Their eq (17) is the discrete variational
  optimum, `-2u_k + Σ_k' F_k' V_kk' - μ = 0`, "a system of linear equations with respect to `F_k`"
  with `u_k = Σ_i (ȳφ)_ik` and `V_kk' = Σ_i (ȳφ)_ik (ȳφ)_ik'` — a linear system on the weights built
  from label second moments with a Lagrange multiplier for the normalisation, which is structurally
  this project's fitted arm `w = C^-1 c`, twenty years earlier, in this paper's target venue, by the
  group whose modelability index this whole line is built on. Their eq (21),
  `F(d_ij) = exp(-α d_ij/D) / Σ_j exp(-α d_ij/D)`, is a normalised one-parameter kernel with α tuned
  for performance: structurally increment 16's kernel arm. **No novelty is claimed for either.**
  What the paper does NOT do, and this is the opening increment 16 occupies: the sentence after eq
  (17) is future tense — "Solution of the system of equations (17) ... WILL define the values of `F_k`"
  — and the next section says "We chose the weighting function in the form" (21). The optimum is
  derived and abandoned, and the two are never compared. Nor is there a uniform arm anywhere in the
  paper: the reference is q² at α = 1, its own kernel at a fixed parameter (Table 1 reports q²(α)
  against q²(α=1); mean improvement 6.7 % in q² and 5.9 % in R², maximum 28.4 % and 44.9 %). Their k
  runs over 1 to 7 and their distances are Euclidean in a descriptor subspace.
- **Gou, J.; Xiong, T.; Kuang, Y. A Novel Weighted Voting for K-Nearest Neighbor Rule. J. Computers
  2011, 6(5), 833-840.** The correct citation for `w_j` proportional to `1/j`. Increment 16's family
  is declared on it.
- **Dudani, S. A. IEEE Trans. SMC 1976, 6(4), 325-327.** The origin of distance-weighted kNN, and
  **not** the rank family this project had been attributing to it: Dudani's weight is
  `w_i = (d_k - d_i)/(d_k - d_1)`, linear in DISTANCE. The misattribution mattered, because the
  defence "the family was declared on a 1976 citation so it was not chosen after seeing the fitted
  profile" is only valid against a citation that contains the family.
- **Bailey, T.; Jain, A. K. A note on distance-weighted k-nearest neighbor rules. IEEE Trans. SMC
  1978, 8, 311-313.** For **fixed k** the **asymptotic classification** error of the **unweighted**
  kNN rule is minimal among weighted kNN rules, for all distributions. **This is a collision with
  section 5.1**, whose "no collision found" claim rested on four passes that did not have this paper
  and has been withdrawn from the skeleton. The two statements share the conclusion that weighting
  cannot help and differ in assumptions and conclusion: Bailey & Jain is asymptotic, about
  classification, distribution-free at fixed k; section 5.1 is finite-sample, about the squared error
  of a linear predictor, under second-order exchangeability. The skeleton now states both and
  explicitly declines to assert that either subsumes the other. Claiming the 1978 theorem as the
  asymptotic form of section 5.1 without doing that work would be a guess dressed as a citation.
- **Hechenbichler, K.; Schliep, K. Weighted k-Nearest-Neighbor Techniques and Ordinal Classification.
  Discussion Paper 399, SFB 386, LMU Munich, 2004.** **Contains no rank-indexed kernel of any kind**:
  every kernel in it is a function of the standardised distance `D(x, x_(i)) = d(x, x_(i)) /
  d(x, x_(k+1))` — rectangular, triangular, Epanechnikov, quartic, triweight, cosine, Gauss,
  inversion. This project had cited it as theory for a rank family. It is now cited for the
  distance-kernel family increment 16 deliberately does not test.
- **Lenz, O. U.; Bollaert, D.; Cornelis, C. arXiv:2311.16872.** The rank-weight and distance-weight
  taxonomy, and the largest published comparison, 85 classification datasets. Its section 5.1 finds
  distance-weights better than rank-weights **kernel by kernel within the NN classifier**, with
  exceptions, while its overall verdict for NN keeps rank weights. Cited at that strength and no
  stronger, for the axis increment 16 does not test.

**What survives as this project's own on the weighting axis**, after the pass: the comparison of a
one-parameter rank kernel against the **fitted** optimum with both arms cross-validated on the
training split, and the measurement that cross-validation selects Gou's α = 1 on 39 of 40 ChEMBL
targets within the registered set. That is narrower than the draft claimed and it is real.

**CORRECTED 2026-10-25 by the pass in section 8, the same day this section was written.** This
paragraph originally also claimed "the uniform baseline neither 2005 nor 2011 provides". That is
false. Nigsch et al., JCIM 2006, 46(6):2412-2422, compared "arithmetic and geometric average, inverse
distance weighting, and exponential weighting" on 4119 organic molecules under 25-fold Monte Carlo
cross-validation. A uniform arm against weighting schemes exists in the kNN-QSAR line, in this
paper's target journal, from 2006. See section 8.

## 8. The fifth pass, remaining scope: Janela & Bajorath, locally weighted regression, closed-form MODI. Run 2026-10-25

Blocking experiment 4 of the skeleton, and items 3 and 5 of "Required before drafting" below. Three
lenses, every reported collision verified against the primary source by a second reader whose
instruction was to refute it: **21 adjudicated, 16 survived, 5 refuted**. Full record in
`results/results_increment17.md`. **No fatal collision. Claim 5 survives.**

### Item 3 is discharged, and positively

"One more pass specifically for closed-form or spectral treatments of MODI, which is the single
collision that would cost claim 5." **Golbraikh, Muratov, Fourches & Tropsha, JCIM 2014, 54(1):1-4 was
read in full** (open NIH deposit PMC3984298). Its only equation is `MODI = (1/K) Σ_i N_i^same /
N_i^total`. No algebraic or closed-form expression, no autocorrelation, homophily, graph, Laplacian or
spectral language anywhere, no formula for prediction error, and no analysis of dependence on dataset
size. The originating lab's own package (`github.com/molecularmodelinglab/MODI`) adds only the
equivalence to the uniform-class average of leave-one-out 1-NN accuracy. Claim 5's accounting is not
pre-empted. Four corrections to its framing:

- **`m1` is Moran's I.** Arthur, *EPJ Data Science* 2025, 14:1, eq. (2): `I = (N/|W|) · (z·z̃)/|z|²`
  with `z̃_i` "the average value of `z` over the neighbours of `i`", and for a row-normalised adjacency
  matrix `|W| = N` so the constant cancels. The paper also states that "unlike assortativity, the Moran
  index does not include" the degree correction, so the two coincide only under that normalisation and
  the skeleton's bare "`m1` *is* Newman's target assortativity" needs its condition. `1 − m1` is
  Geary's C in the same family. The spatial-autocorrelation literature owns the object.
- **The Dirichlet energy is the first-moment term, not the whole right-hand side.** The Heterophilic
  Graph Learning Handbook (arXiv:2407.09618, §2.1) — the project's own citation here — defines
  `E_D(x) = xᵀLx = ½ Σ_{(i,j)∈E}(x_i − x_j)²`. The identity's right-hand side also carries `ρ_nn`.
- **The graph-average predictor is the harmonic energy minimiser, and that is 2003.** Zhu, Ghahramani
  & Lafferty, ICML 2003: "the value of f at each unlabeled data point is the average of f at
  neighboring points", `f(j) = (1/d_j) Σ_{i~j} w_ij f(i)`, and "the minimum energy function is
  harmonic" with `Δ = D − W`. Adjacent — they minimise the energy to impute, the project measures the
  average's error — and must be cited.
- **The index enumeration is incomplete.** Cui et al.'s activity-cliff-aware graph learning paper
  defines a **Label Incoherence Index**, `LII = (1/n) Σ_i (1/k) Σ_{j∈N_i} |y_j − y_i|`, on the kNN
  graph at k = 10 with uniform weights and `x_i` excluded. Sections 1 and 3 must name it. Section 3's
  claim that the **second** moment is absent from the published indices survives: LII is L1 and
  first-moment, so it does not contain it either.

Refuted here: kernel-target alignment (Cristianini, Shawe-Taylor, Elisseeff & Kandola, NIPS 14, 2001)
as a closed-form modelability collision.

### Item 5's remaining half: locally weighted regression. Seven collisions, none refuted

- **Nigsch, Bender, van Buuren, Tissen, Nigsch & Mitchell, JCIM 2006, 46(6):2412-2422.** **OVERLAPS,
  and corrects section 7 above.** "To compute the prediction on the basis of the melting temperatures
  of the nearest neighbors, we used four different methods (arithmetic and geometric average, inverse
  distance weighting, and exponential weighting), of which the exponential weighting scheme yielded
  the best results", on 4119 organic molecules plus 277 drugs under 25-fold Monte Carlo
  cross-validation with the neighbour count varied. Increment 16's prediction 2 is a **replication on
  potency data of a melting-point result**.
- **Cleveland & Devlin, JASA 1988, 83(403):596-610, §10.4**, whose heading is "Weight Functions and
  the Poor Performance of the Uniform": "the uniform weight function, with the discontinuity at 1,
  performs poorly", and "asymptotic results for nonparametric regression show that the overall form of
  the weight function does not have an appreciable effect with respect to mean squa[red error]". Both
  halves bear on section 5.2: the direction was known in 1988, and so was the insensitivity to the
  kernel's form, which is the honest frame for increment 16's one-parameter result.
- **Anava & Levy, k\*-Nearest Neighbors: From Global to Local, NIPS 2016** (arXiv:1701.07266). The
  closest adjacent prior to increment 16's headline: "Our formulation enables us to phrase a notion of
  optimal weights, and to efficiently find these weights as well as the optimal number of neighbors",
  and §3, "the optimal weights depend on a **single parameter** `L/C`, namely the Lipschitz to noise
  ratio". Adjacent and not pre-empting — their parameter is a derived ratio in an assumed
  Lipschitz-plus-noise model, not a cross-validated exponent, and they neither measure molecular
  neighbourhoods nor compare against a least-squares-fitted weight vector — but it is the paper a
  reviewer raises, and section 5.2 must cite it and state the difference.
- **Demirkaya, Fan, Gao, Lv, Vossler & Wang, JASA 2024, 119(545):297-307** (arXiv:1808.08469). The
  rank-indexed analogue: the distributional-NN estimator "automatically assigns monotonic weights to
  the nearest neighbors", fixed by a single subsampling scale `s`, with an L-statistic representation.
- **Zhang, Golbraikh, Oloff, Kohn & Tropsha, ALL-QSAR.** A Gaussian **distance** kernel
  `w = exp(−d²/2K²)` with the width optimised "from 0.01 to 1.00 with an increment of 0.01", and "no
  fixed number of nearest neighbors". Adjacent: no uniform arm and no fitted weight vector.
- **Gajewicz-Skretna, Kar, Piotrowska & Leszczynski, kernel-weighted local polynomial regression for
  QSAR/QSAAR.** "However, the particular form of the function, K has only a relatively small effect on
  estimation accuracy" — the Cleveland & Devlin insensitivity, restated in QSAR.
- **Helman, Shah & Patlewicz, GenRA, Comput. Toxicol. 2018, 8:34-50.** Similarity-weighted read-across
  with Jaccard similarity at a fixed k = 10: similarity-weighted averaging is standard practice in
  read-across, and the only fitted parameters are in neighbour selection.

**Clean, and specifically.** No source reports the rank-resolved cross-moment `c_j = E[z_v z_{u_j}]` as
a function of `j` on molecular data, and none states that molecular neighbourhoods fail second-order
exchangeability. Section 5.2's **measurement** is not pre-empted. Searched: locally weighted
regression, lazy learning and local QSAR; the similarity-property-principle literature with
rank-resolved framing; and the activity-cliff and activity-landscape geometry literature.

### Blocking experiment 4's own target: Janela & Bajorath. Nine reported, four refuted

- **Their metric is unnormalised MAE.** *Pharmaceuticals* 2023, 16(4):530 §4.5 defines exactly one
  metric for all 376 activity classes: "the mean absolute error (MAE) was calculated by comparing
  predicted and experimental test compound potency values." *Sci. Rep.* 2023 adds RMSE and squared
  Pearson `r²` on eight classes. Neither reports a variance-normalised error as a performance value.
  This project's skill is `1 − RMSE_arm/RMSE_baseline` against a train-mean baseline, so the two
  literatures report on different scales. Their own paired figures give the conversion factor —
  "~0.4 and ~0.5 for MAE and RMSE" and "generally smaller than 0.8 and 1.0", both ratios 1.25, the
  Gaussian `√(π/2)` — under which a 0.1 MAE margin maps to 0.125 of skill at a potency standard
  deviation of 1.0 and 0.084 at 1.5, bracketing increment 14's 0.0754 to 0.0949. **Same order.** Not
  claimed: that this resolves the comparison, that skill equals sklearn's `r2_score` (it does not), or
  any particular potency spread.
- **Their 2022 control used Euclidean distance on binary fingerprints.** Their public code builds it
  as `neighbors.KNeighborsRegressor()` with `{"n_neighbors": [1, 3, 5]}` only, under
  `GridSearchCV(..., cv=10, scoring="neg_mean_absolute_error")`, on
  `GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)`. With no metric argument scikit-learn defaults
  to `minkowski, p=2` with `weights='uniform'`. Their **2023** repository sets `metric='jaccard'`
  explicitly. The control is at most five Euclidean-nearest neighbours against this project's uniform
  mean over ten Tanimoto-nearest ones.
- **Their model arm was done right**, so section 4a must not imply the field generally gets it wrong.
  *Pharmaceuticals* 2023 §4.3.1: "the regularization hyper-parameter C was determined by testing
  (0.001, ..., 10,000) values. SVR models were derived using the **Tanimoto kernel**", with a 54-point
  deep-arm grid in the 2022 code.
- **Their mechanism is label-marginal, not graph-structural**, so it touches neither section 2 nor
  section 4: "in the intermediate potency sub-range, predictions yielded artificially low errors, due
  to narrow local potency value distributions around the median."
- **They run a sample-size sweep and read it the other way.** *Sci. Rep.* 2023: "nine training sets of
  increasing size ... 6, 12, 18, 30, 48, 78, 126, 204, and 330 compounds" over ten trials, finding
  errors "constantly small, independent of training set size" in the dominant sub-range. Section 6's
  statement that nobody in the modelability line reports an **index** as a function of `n` stands, but
  this is the second cited line that runs section 4's sweep with the opposite reading, after the Gamma
  test's M-test. Section 4 should name both.
- **Clean, and specifically.** The full text of both open-access papers contains **zero occurrences**
  of "modelab", "MODI", "ROGI", "Golbraikh", "Tropsha", "assortativ", "homophil", "Dirichlet",
  "roughness" or "activity cliff". Sections 1, 2, 3 and 5 are untouched by this line.

Refuted here: that they report SVR ahead rather than a tie; that their `k` grid contains this
project's CV choice; that their nearest-neighbour-removal intervention is unaddressed; and that the
roles of their two 2023 papers are reversed in increment 14's pre-registration.

## Verdict on each claim in the skeleton

| # | Claim | Verdict after the pass |
|---|---|---|
| 1 | Assortativity predicts attainable accuracy | **Known.** MODI_q2 vs QSAR_R2, Golbraikh 2014. Reported as replication at larger scale with a frozen plan. Increment 11 adds the comparison: MODI over assortativity is +0.305 [+0.026, +0.539] on ChEMBL-40 and covers zero on the other two collections, so MODI is the slightly richer of the two statistics |
| 2 | It beats ROGI incrementally | **Survives**, and MODI is now measured (increment 11). Assortativity over MODI covers zero on all three collections, so the honest statement is that the two are near-interchangeable rankers, not that ours wins. SARI still missing |
| 3 | Not an artefact of one fingerprint | **Survives.** No collision found |
| 4 | It also says which model class wins | **Survives and is strengthened**, because the field's own rankings put kNN near the bottom on average and this is the conditional exception |
| 5 | Exact identity, equals a Dirichlet energy | **Object known, accounting new, and the fifth pass cleared it — see section 8.** MODI's own paper was read in full and contains no closed form, no graph or spectral language and no error formula, so the single collision that would have cost this claim does not exist. 0.0083 to 0.0123 mean absolute error reproducing MODI_q2 on 80 targets. Four framing corrections stand: `m1` is **Moran's I** (Arthur 2025) and coincides with Newman's assortativity only under row normalisation; the Dirichlet energy is the **first-moment term**, not the whole right-hand side; the graph-average-as-harmonic-energy-minimiser is Zhu, Ghahramani & Lafferty 2003; and the index enumeration must add Cui et al.'s **Label Incoherence Index**. The second moment's own ranking contribution covers zero on all three collections, so it is necessary for the identity and empirically quiet here |
| 6 | Equal weights are near-optimal locally | **Refuted by increment 12, corrected by its own audit, and measured to the end by increment 16.** Under exchangeability the statement is a theorem about calibration and says nothing about weighting. Without the assumption, reweighting by neighbour rank is worth **+0.0155 [+0.0118, +0.0192]** once each arm gets its better `m` — not the +0.038 to +0.050 first reported, 62 % of which was the effect of fixing `m` — and increment 16 showed the whole effect is bought by **one parameter**, `w_j ∝ j^-α` with α = 1 selected on 39 of 40 targets, non-inferior to the fitted `C^-1 c` optimum at a pre-registered margin. The claim becomes "the neighbourhood mean is calibrated in total and wrong in shape, and the shape has one degree of freedom". The weighting passes it asked for are **done**: sections 7 and 8. Prior art on the weighting axis is heavy — Itskowitz & Tropsha 2005 own both arms, Nigsch 2006 owns the uniform-against-kernel comparison in QSAR, Cleveland & Devlin 1988 own the direction, Anava & Levy 2016 own one-parameter optimal local weights — and what survives is the measurement on molecular neighbourhoods and the comparison against the fitted optimum |
| 7 | The noise floor fails out of sample, with a diagnosis | **Replication.** Must cite Atkinson, Kim, Zimmerman and claim the graph transport, not the estimator |
| 8 | A structural limit on the whole approach | **Survives**, and is sharper once framed as the graph version of Zimmerman's design problem |

## What the paper is now

**"The modelability index has a closed form, and it is missing a term."**

Take MODI, a JCIM-native and well-cited idea that is defined operationally. Show that its continuous
form is exactly the Dirichlet energy of the activity on the similarity graph, that the reported
statistic is its first moment, and that a second moment is absent from every published version. Show
on 110k compounds across three collections what the second moment buys, that equal weights are
already within a thousandth of the best local linear predictor, and that the natural extension to a
noise floor reproduces nugget estimation including its failures.

Claims 5, 6 and 4 carry the paper. Claims 1 and 7 are replications and are labelled as such. Claim 2
needs a new baseline run before anything is written.

## Required before drafting

1. ~~Run MODI on all three collections~~ **done, increment 11.** `prereg/increment11_modi.yaml`,
   `results/results_increment11.md`. The baseline table has MODI beside ROGI.
2. ~~Relax exchangeability in claim 6~~ **done, increment 12**, and it reversed the claim. The hop
   profile turned out to be the wrong axis: the repository had no measured profile on real targets at
   all, and the axis that matters for the kNN mean is neighbour RANK, not hop count.
3. ~~One more pass specifically for closed-form or spectral treatments of MODI~~ **DONE,
   increment 17, section 8.** MODI's own paper read in full: no closed form, no graph or spectral
   language, no error formula, no size analysis. **Claim 5 survives.** Four framing corrections, none
   of them a refutation.
4. **SARI** as a third baseline, nice to have rather than required.
5. ~~A fifth pass on weighted kNN in QSAR~~ **DONE in two halves: sections 7 and 8.** The
   distance- and rank-weighted half ran with increment 16 and found that three of this project's own
   citations were wrong about what they cite. The similarity-weighted and locally-weighted-regression
   half ran as increment 17 and found seven collisions, none refuted, including the one that corrects
   section 7's own text: Nigsch 2006 compared a uniform arm against three weighting schemes in this
   paper's target journal.
6. **Re-examine claim 4 under the reshaped floor**, with its own frozen plan. Claim 4 attributes the
   floor-against-fitted-model gap to information outside the neighbourhood; **+0.0155** of that gap is
   now known to be weighting, and known to be reachable with one parameter (increment 16). Note the
   figure: this entry originally said +0.04, from the pre-correction reading of increment 12.
7. **Run Janela & Bajorath's protocol on their public data**, which section 8 has now specified: their
   metric is MAE, their 2022 control is unweighted kNN over at most five **Euclidean**-nearest
   neighbours on 2048-bit Morgan vectors, and their model arm is a CV-tuned **Tanimoto-kernel** SVR.
   The run must report MAE beside skill so the comparison exists on their scale too. This is the only
   thing that settles section 4a.

## What increment 11 did to the plan

The pre-registration expected both partial correlations in questions B and C to come back empty,
reasoning that under the identity assortativity and MODI_q2 are the same quantity up to the second
moment. One cell contradicted that: MODI over assortativity excludes zero on ChEMBL-40. The direction
is what the identity predicts, since MODI_q2 evaluates the estimator whose error the identity expands
and so carries the second moment implicitly, while assortativity is the first moment alone. The
tension worth stating in the draft is that the second moment's *own* contribution (question D) covers
zero on all three collections while the statistic that contains it clears zero on one. Both are small
effects at this sample size and the paper reports them that way.

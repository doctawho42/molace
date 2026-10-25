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
one-parameter rank kernel against the fitted optimum with both arms cross-validated on the training
split, the uniform baseline neither 2005 nor 2011 provides, and the measurement that
cross-validation selects Gou's α = 1 on 39 of 40 ChEMBL targets within the registered set. That is
narrower than the draft claimed and it is real.

## Verdict on each claim in the skeleton

| # | Claim | Verdict after the pass |
|---|---|---|
| 1 | Assortativity predicts attainable accuracy | **Known.** MODI_q2 vs QSAR_R2, Golbraikh 2014. Reported as replication at larger scale with a frozen plan. Increment 11 adds the comparison: MODI over assortativity is +0.305 [+0.026, +0.539] on ChEMBL-40 and covers zero on the other two collections, so MODI is the slightly richer of the two statistics |
| 2 | It beats ROGI incrementally | **Survives**, and MODI is now measured (increment 11). Assortativity over MODI covers zero on all three collections, so the honest statement is that the two are near-interchangeable rankers, not that ours wins. SARI still missing |
| 3 | Not an artefact of one fingerprint | **Survives.** No collision found |
| 4 | It also says which model class wins | **Survives and is strengthened**, because the field's own rankings put kNN near the bottom on average and this is the conditional exception |
| 5 | Exact identity, equals a Dirichlet energy | **Object known, accounting new, and now measured against the published index.** 0.0083 to 0.0123 mean absolute error reproducing MODI_q2 on 80 targets. The second moment's own ranking contribution covers zero on all three collections, so it is necessary for the identity and empirically quiet here |
| 6 | Equal weights are near-optimal locally | **Refuted by increment 12, and replaced.** Under exchangeability the statement is a theorem about calibration and says nothing about weighting; measured without the assumption, reweighting by neighbour rank is worth +0.038 to +0.050 skill out of sample against a rank-shuffled null of -0.002. The claim becomes "the neighbourhood mean is calibrated in total and wrong in shape". Needs a fresh pass against distance-weighted kNN in QSAR, which four passes never searched because the claim was not about weighting |
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
3. One more pass specifically for closed-form or spectral treatments of MODI, which is the single
   collision that would cost claim 5.
4. **SARI** as a third baseline, nice to have rather than required.
5. **A fifth pass on weighted kNN in QSAR** — distance-weighted, similarity-weighted and
   locally-weighted regression variants. Claim 6's replacement is a weighting result and this is the
   obvious place for a collision. Four passes never looked, because the claim was not about weighting
   until increment 12 measured it.
6. **Re-examine claim 4 under the reshaped floor**, with its own frozen plan. Claim 4 attributes the
   floor-against-fitted-model gap to information outside the neighbourhood; +0.04 skill of that gap is
   now known to be weighting.

## What increment 11 did to the plan

The pre-registration expected both partial correlations in questions B and C to come back empty,
reasoning that under the identity assortativity and MODI_q2 are the same quantity up to the second
moment. One cell contradicted that: MODI over assortativity excludes zero on ChEMBL-40. The direction
is what the identity predicts, since MODI_q2 evaluates the estimator whose error the identity expands
and so carries the second moment implicitly, while assortativity is the first moment alone. The
tension worth stating in the draft is that the second moment's *own* contribution (question D) covers
zero on all three collections while the statistic that contains it clears zero on one. Both are small
effects at this sample size and the paper reports them that way.

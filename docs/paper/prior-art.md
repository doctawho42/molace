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

## Verdict on each claim in the skeleton

| # | Claim | Verdict after the pass |
|---|---|---|
| 1 | Assortativity predicts attainable accuracy | **Known.** MODI_q2 vs QSAR_R2, Golbraikh 2014. Report as replication at larger scale with a frozen plan, not as a finding |
| 2 | It beats ROGI incrementally | **Survives**, but the baseline table must add MODI and ideally SARI. New experiment required |
| 3 | Not an artefact of one fingerprint | **Survives.** No collision found |
| 4 | It also says which model class wins | **Survives and is strengthened**, because the field's own rankings put kNN near the bottom on average and this is the conditional exception |
| 5 | Exact identity, equals a Dirichlet energy | **Object known, accounting new.** The claim becomes "the modelability index has a closed form, and it is missing a second moment" |
| 6 | Equal weights are near-optimal locally | **Survives, and is the most novel thing in the paper.** No collision found in four passes |
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

1. **Run MODI on all three collections** and add it to the baseline table beside ROGI. Without this
   the paper is not submittable to JCIM.
2. **Relax exchangeability in claim 6** using the measured correlation-versus-hop profile.
3. One more pass specifically for closed-form or spectral treatments of MODI, which is the single
   collision that would cost claim 5.

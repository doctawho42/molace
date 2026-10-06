# From repository to paper: what the prior-art pass found, and what survives

Written 2026-10-05, after Prokhorenkova saw the talk. This is a planning document, not a draft. Its
job is to say what can honestly be claimed, and the answer changed twice while it was being written.

## The short version

**The theory is not ours.** Three checks, three hits, each landing on a different piece of what the
project had been calling its theoretical contribution. What survives is an empirical programme, a
diagnosed negative result, and a transport argument between two literatures that mostly do not cite
each other. That is still a paper. It is not the paper we thought.

## What the checks found

### 1. The identity is a Dirichlet energy

The floor predicts `yhat = A_norm y`, so its residual is `(I - A_norm) y` and its relative error is
`||(I - A_norm) y||^2 / ||y||^2 = 1 - 2 m1 + m2`, with `m_j = y' A_norm^j y / y'y`. Verified to 1e-16
on regular graphs. Target assortativity is exactly `m1`.

Dirichlet energy as a homophily metric is standard in the heterophilic-graph literature; it appears
as a workhorse measure in the Heterophilic Graph Learning Handbook (arXiv 2407.09618), where small
energy is read as "signal is mostly low-frequency". So the mathematics of our identity is a known
object, written in different notation.

**What may still be ours:** the exact accounting. The handbook says Dirichlet energy measures label
smoothness. It does not say "Newman's target assortativity is the first moment, the shared-neighbour
correlation is the second, and the kNN mean's error is `1 - 2 m1 + m2` exactly". Pinning which
reported statistic is which moment is small, but it is the thing that makes the rest legible.

### 2. The two-statistic noise estimate is nugget estimation

`1 - nu = r^2 / rho_nn` extrapolates the label correlation at hop 1 and hop 2 back to hop 0. In
geostatistics that is the nugget: the variogram's intercept at zero lag, read as measurement error
plus microscale variance. It is standard practice and has been for decades.

Worse for our originality, and better for our confidence: the known cautions are our findings.

- Atkinson 1997 (Int. J. Remote Sensing) calls the nugget the most appropriate estimator of
  measurement error, then says it is less reliable than previously thought because "it is impossible
  to account for the form of the variogram near the ordinate when selecting a mathematical model".
  Our geometric-decay model IS a choice of form near the origin, and our mixed-hop-length bias
  (-0.222 at k = 10) is that caution, measured.
- Kim et al. 2010 (J. Nonparametric Statistics) give asymptotics for the nugget estimator and
  identify a regime where it suffers from strong dependency. Structurally our ill-conditioning.
- Zimmerman 1991 (Environmetrics) designs sampling networks specifically to estimate the
  nugget-to-sill ratio precisely. That is our five refused design gates, thirty-five years earlier.

**What may still be ours:** the transport to a setting with no metric, only hop counts, and the
synthetic separation of the two failure modes with known ground truth. We did not know we were
reproducing geostatistics; a reader will, so we must cite it and claim replication rather than
discovery.

### 3. The bridge is partly built

Moran's I already appears in GNN work; GNNs are already used for inductive kriging (arXiv 2401.12681);
spatial heterophily is already a topic (Zhou et al., KDD 2023). So "graphs meet spatial statistics" is
not virgin ground either.

**What may still be ours:** the specific claim that the graph-ML homophily measures and the
geostatistical nugget-to-sill question are the same question, and that the benchmark practice of
reporting a homophily number per dataset is reporting a first moment with no error bar and no second
moment. That is a position, and positions need evidence; we have it.

## What the paper actually is

A measurement paper with a transport argument, not a theory paper.

**Title, working:** "The modelability index has a closed form: dataset modelability as a Dirichlet
energy of activity on the similarity graph"

The earlier working title, "Homophily measures are a first moment", survives as the framing of section
3 but no longer leads: after the prior-art pass the JCIM-native entry point is MODI, not homophily.

**Claims, each with the evidence that backs it**

| # | Claim | Evidence | Strength |
|---|---|---|---|
| 1 | Target assortativity predicts attainable accuracy | rho +0.875 / +0.909 / +0.810 on 40 independently curated ChEMBL targets; +0.77 / +0.82 / +0.76 on 10 DeepDelta benchmarks with published models | strong, four collections, and a replication of Golbraikh 2014 rather than a finding |
| 2 | It does so beyond the existing roughness index | assortativity over ROGI +0.846 [+0.678, +0.928]; ROGI over assortativity +0.098, covers zero; ROGI alone does not predict. Against MODI the picture is symmetric: assortativity over MODI covers zero everywhere, MODI over assortativity +0.305 [+0.026, +0.539] on ChEMBL-40 | strong against ROGI, near-interchangeable with MODI |
| 3 | The link is not an artefact of one fingerprint | +0.726 on RDKit descriptors; paired drop off ECFP4 +0.163 [+0.022, +0.358] | moderate, one alternative representation |
| 4 | **UNDER CORRECTION.** It also says which model class wins, through the level | floor minus pointwise -0.343 [-0.619, -0.008]; floor wins 26/40, and 12/13 on the hardest third | moderate, interval thin |
| 5 | For the kNN mean the link is exact, and equals a Dirichlet energy — so the published modelability index has a closed form | identity verified to 1e-16 on regular graphs; 0.025-0.031 mean absolute error as a point prediction over k = 3..40; reproduces MODI_q2 to 0.0083-0.0123 mean absolute difference on 80 targets, Pearson 0.996-0.998 | the paper's centre: object known, accounting new, measured against the published index |
| 6 | **CORRECTED to +0.0155 [+0.0118, +0.0192].** The neighbourhood mean is calibrated in total weight and wrong in shape | weight sum 0.97, so rescaling buys +0.0000 [-0.0003, +0.0004] out of sample; reweighting by neighbour rank buys +0.0379 [+0.0314, +0.0442] on 31 ChEMBL targets and +0.0497 [+0.0417, +0.0577] on 14 MoleculeACE ones, beating a rank-shuffled null on 44 of 45; optimal profile 3.49x down to 0.19x | strong, pre-registered, and the opposite sign from what this table asserted before increment 12 |
| 7 | The nugget-style noise estimate fails out of sample, and we know why | 2/30 violations against a pre-registered threshold of 1; synthetic separation into ill-conditioning and mixed hop lengths; 1/c(d)^2 overstatement predicted and measured to 0.07 % | strong as a negative result |
| 8 | A structural limit: the regime where the second statistic matters most is the regime where the estimator built on it does not work | five refused design gates, variance decompositions 4.2 / 0.0 / 14.1 / 10.8 / 0.3 % | honest, and the most quotable sentence in the paper |

**Claim 6 reversed under measurement, and is stronger for it.** The earlier version of this table
said equal weights were within a thousandth of the best local linear predictor, and concluded that the
floor-against-fitted-model gap could not come from weighting the neighbourhood better. Increment 12
shows that conclusion rested on the exchangeability assumption rather than on data. Under
exchangeability `1` is an eigenvector of the neighbour second-moment matrix and the cross-moment vector
is parallel to it, so every optimal weight is *exactly* equal, the whole gap is `Var(mean)(1-beta)^2`,
and fitting one scalar on the plain mean attains the k-dimensional optimum exactly. The assumption
deletes the effect and leaves a calibration residue that reads as a null.

Measured without it, the rank-resolved second moment falls from 0.714 to 0.476 across ten ranks, and
reweighting is worth +0.038 to +0.050 skill out of sample against a rank-shuffled null of -0.002. So
the paper's claim is now that the mean is calibrated in total and wrong in shape, which is a measured
quantity with a sign rather than an assertion, and the vacuity theorem explains why nobody had looked:
the natural model to write down forbids the question.

**This unsettles claim 4 and that is not resolved.** Claim 4 reads the floor beating fitted models on
the hardest third as evidence about information outside the neighbourhood. If a better-shaped floor is
worth +0.04 skill, part of that gap was weighting after all. The re-examination needs its own frozen
plan and does not exist yet.

## Section plan

1. **Setup.** The dataset as a comparison graph; what is measured; the three arms.
2. **The measure predicts attainable accuracy.** Claims 1-3. Charts, not tables.
3. **What the measure is.** Claim 5: the identity, its reading as a Dirichlet energy, assortativity
   as the first moment, and the second moment nobody reports. Cite the handbook; claim the accounting,
   not the object.
4. **What averaging gets wrong.** Claim 6: the vacuity theorem, the measured rank profile, and the
   scale-against-shape decomposition. Its consequence for claim 4 is stated as open, not as resolved.
5. **A noise floor from two moments, and why it fails.** Claim 7, framed as a graph replication of
   nugget estimation, citing Atkinson, Kim, Zimmerman.
6. **A limit on the whole approach.** Claim 8.
7. **Method.** Pre-registration, the refused gates, what was frozen when.

The Hodge half does not appear. It is a separate, weaker story and it dilutes this one.

## What has to happen before a word of the draft is written

1. ~~A real prior-art pass~~ **done.** `docs/paper/prior-art.md`, four passes. It found MODI, which
   reframed the paper, and confirmed the nugget and Dirichlet collisions.
2. ~~Run MODI as a measured baseline~~ **done, increment 11.** The identity reproduces MODI_q2 to
   0.008-0.013 across 80 targets. `results/results_increment11.md`.
3. ~~Relax exchangeability in claim 6~~ **done, increment 12.** It reversed the claim.
   `prereg/increment12_local_weights.yaml`, `results/results_increment12.md`.
4. One more search pass for closed-form or spectral treatments of MODI, the single collision that
   would cost claim 5. A second pass is now also needed on rank-weighted or distance-weighted kNN in
   QSAR, since claim 6's replacement is a statement about weighting and that literature is old.
7. **Re-examine claim 4 under the reshaped floor**, with a frozen plan. This is the new blocker and it
   is a claim the paper leans on.
5. ~~Decide the venue~~ **done: JCIM**, molecular programme leading, transport as framing.
6. ~~Co-authorship~~ **excluded by decision.**

## Risk register

- **Novelty.** After three checks the theoretical contribution is a known object in new notation. If
  a fourth check finds the exact identity stated somewhere, section 3 becomes a citation and the paper
  leans entirely on sections 2, 4, 5 and 6. It would survive that.
- **Claim 4's interval** clears zero by 0.008. Robust to the recovered-rounding check (200/200), but
  it is thin and a reviewer will say so first.
- **Claim 3** rests on one alternative representation. It is reported as one, not as generality.
- **Claim 6's replacement is a weighting result**, which puts it next to a long line of
  distance-weighted kNN work in QSAR that the four prior-art passes did not search for, because until
  increment 12 the claim was not about weighting. That search has to happen before drafting.
- **Claim 4 is now partly undermined by claim 6** and the paper cannot state both as they stand.
- **Scope.** Thirty ChEMBL targets plus forty plus ten benchmarks is a lot of molecules and one domain.
  Nothing here is shown outside molecular property prediction.


## Two claims under correction, found 2026-10-06

Both were found by adversarial review of a later plan, and both are verified against committed data.

**Claim 4 is measured against an arm that excludes the best learner.** `skill_pointwise` in
`results/separation_chembl.csv` is the mean of the histogram-gradient-boosting and MLP RMSEs, set by
`MATCHED_FAMILIES = ("hgb", "mlp")` in `src/molace/models/pairwise.py` and used at
`src/molace/models/gap.py:109`. SVM is excluded, and `src/molace/models/pointwise.py`'s own docstring
says "SVM leads because it is the real bar... 21 of 30 targets, mean rank 1.60".
`pointwise.select_by_cv` exists, takes no test arguments, and is called by **no script** —
`tests/test_gap.py:97` even pins `assert "select_by_cv" not in src`.

The averaging is **correct** for the pointwise-minus-pairwise gap, because both arms are averaged over
the same two families and the difference is matched. It is **wrong** for claim 4, which compares the
floor against the best a model can do. One number was used for two incompatible purposes. Over all 40
ChEMBL targets the trained arms beat the floor on 14 of 40, median -0.016; the corrected comparison
against the CV-selected learner has not been run and claim 4's sign is therefore not currently known.

**Claim 6's magnitude is 41 % of what this table said**, +0.0155 [+0.0118, +0.0192] rather than
+0.0379, because the uniform baseline was fixed at `m = 10` while uniform at `m = 5` is better on 28 of
31 targets. It still excludes zero and is positive on 30 of 31. See the correction appended to
`results/results_increment12.md`. The optimal profile has cosine 0.9946 with `w_j` proportional to
`1/j`, so Dudani 1976 is the competitor the claim must be stated against, and that comparison is not
yet run.

**Consequence for the paper.** Claim 4 cannot be stated until the baseline audit runs. Claim 6 is
restated at the corrected magnitude. The fifth prior-art pass on weighted kNN, listed as required in
`docs/paper/prior-art.md` and never run, is now the blocking item it was always going to be.

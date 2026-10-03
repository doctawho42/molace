# molace increment 1 — design

Date: 2026-10-03
Status: approved in conversation, pending written-spec review
Target: first falsifiable result before 2026-10-05

## 1. Purpose

Does a **parameter-free statistic of a molecule-similarity graph, computable before any
training**, predict whether pairwise (delta) models beat pointwise models on that task —
over and above what landscape roughness (ROGI-XD) and graph density already explain?

And, separately: **where does the energy of a trained pairwise edge flow live** in the
three-way Hodge decomposition (gradient / harmonic / curl) of the comparison graph?

The graph here is not a molecule. Nodes are molecules; edges are comparisons between them.

## 2. Why this design differs from the original proposal

The source proposal is `~/life_planning/Life/Материалы/Graph ML/petproject_graph_ml.md`
(literature review dated 2026-09-27). A six-dimension adversarial audit on 2026-10-03
refuted four load-bearing claims. The refutations are recorded here because the main risk
to this project is silently re-introducing them.

| Original claim | Verdict | Consequence for this design |
|---|---|---|
| "Граф здесь не молекула... Это и есть необычный ход" | REFUTED | Chemical Space Networks (Maggiora & Bajorath, JCAMD 2014, and the 2014-2023 line) already use compounds-as-nodes / Tanimoto-as-edges over ChEMBL activity datasets, characterised with network measures **including assortativity**, across 36 activity classes, **at controlled edge density**. We position inside that paradigm, cite it, and control density. |
| "Так устроены SQRL и PADRE" (pairwise models are `g(xi) − g(xj)`, hence curl-free) | REFUTED for all four named works | SQRL eq. 2 is `l(f(g(xi) − g(xj)), dy_ij)` with `f` an MLP head; PADRE is a random forest on `[xi, xj, xi−xj]`; DeepDelta concatenates two D-MPNN embeddings; RankRefine has no learned pairwise function. The theorem is restated for a **frozen encoder with a bias-free linear readout**, where it is provable. |
| "Ни одна работа не замечает, что архитектура бесциркуляционна"; label-free uncertainty from non-closing triangles is new | REFUTED | Twin Neural Network Regression (Wetzel et al., arXiv 2012.14873 / 2106.06124 / 2301.01383) states and uses the loop condition `F(x1,x2) + F(x2,x3) + F(x3,x1) = 0` and derives uncertainty from its violation, including on unlabelled points. DeepDelta measures triangle additivity (MAE 0.127 +/- 0.043) and uses it as an unsupervised model-quality signal. PADRE and Zhang et al. (J Cheminform 15:75, 2023) publish anchor-dispersion uncertainty. Kramer (JCIM 2019) analyses nonadditivity over MMP cycles. We claim the **orthogonal three-way decomposition, the energy budget, and the observability identity**, not the observation. |
| Exit-experiment 1 correlates against the published pointwise-vs-pairwise gap from SQRL and MoleculeACE | REFUTED as written, partly rescued | SQRL reports only numbers aggregated across tasks ("results are reported aggregated across tasks"); MoleculeACE's 12 benchmarked algorithms are all pointwise. No per-target gap exists in either. DeepDelta **does** report per-benchmark results on 10 ADMET sets and ships per-fold predictions, so a published gap exists on 10 tasks. We compute the gap ourselves on 30 targets and use DeepDelta's 10 as external replication. |

Two further corrections that change numbers rather than claims:

- **Adjusted homophily and label informativeness are defined for categorical labels only.**
  The group's own answer for continuous targets is Newman (2003) scalar assortativity:
  GraphLand states "To measure the similarity of labels of connected nodes for regression
  datasets, we use target assortativity — the Pearson correlation coefficient of target
  values between pairs of connected nodes", and GraphPFN repeats it. GraphLand reports
  label informativeness zero times. There is no published continuous LI and we do not
  invent one.
- **Adjusted homophily has been superseded by unbiased homophily** (Mironov &
  Prokhorenkova, arXiv 2412.09663, LoG 2024), which GraphLand and GraphPFN both use.
  We use unbiased homophily wherever a categorical label is used.

The following items are **not independently verified** (their adversarial second pass
died on network errors) and must be checked before the text is written: arXiv 2409.14500,
2508.20906, 2509.21489, 2601.04507, 2508.16495 were confirmed by title match only, by the
same batch method that mis-described 2509.18893 (which is **graph-level** heterophily with
motif labels, not atom-level — the original proposal mis-describes it too). SALI is named
in the proposal's prose with no bibliography entry.

## 3. Positioning

| Prior art | What stays theirs | What we add |
|---|---|---|
| Chemical Space Networks (Maggiora & Bajorath 2014 onward) | molecules as nodes, Tanimoto edges, network measures, density control | the **label** statistic rather than degree assortativity; the tie to model-class choice |
| ROGI / ROGI-XD (2207.09250, 2305.08238) | roughness to modelability | roughness to **node-vs-edge model choice**, and the translation into the homophily vocabulary |
| TNNR (Wetzel et al. 2021-2023) | antisymmetry by construction, the loop condition, label-free UQ | the **orthogonal** gradient/harmonic/curl split, the energy budget, observability |
| DeepDelta (2023) | additivity MAE as unsupervised quality signal | decomposing **their** flow rather than proposing a new architecture |

A deliberate design consequence: target assortativity on a kNN similarity graph measures
nearly what ROGI measures, differently normalised. We state this in front rather than
hiding it. The headline claim is therefore **incremental over ROGI-XD**, not merely
correlated with the gap.

## 4. Scope

In increment 1:

- 30 MoleculeACE targets as the spine; 9 TDC ADMET regression tasks as a pre-registered
  out-of-domain holdout opened exactly once.
- kNN similarity graphs at fixed degree; threshold graphs as a robustness check.
- Target assortativity (primary), unbiased homophily and LI on `cliff_mol`, ROGI-XD as a
  competing baseline, normalised Dirichlet energy as secondary.
- A self-computed pointwise-vs-pairwise gap on the shipped MoleculeACE split.
- DeepDelta's shipped per-fold predictions as a published external gap on 10 tasks.
- Hodge dimension census on all 30 targets (no training required).
- The frozen-encoder readout ablation.
- Energy budget of a trained pairwise flow (at risk, see §11).

Out of increment 1, explicitly: MMP graph construction; a new joint pair encoder
(DeepDelta already is one); GraphPFN and G2T-FM in-context arms; Polaris (it largely
re-hosts MoleculeACE and TDC) and OpenADMET (its CYP challenge is live with a blinded test
set until 2026-11-03).

## 5. Data layer

MoleculeACE is **git-cloned, not pip-installed**: `Data/results/*` is absent from
`package_data`, and that file carries the per-target per-model performance matrix.

Three measured traps become tests, not footnotes:

1. The canonical label column is `y [pEC50/pKi]`. The column `y` is standardised and
   negative. Using it silently changes every homophily number.
2. `cliff_mol` is read from the per-target CSVs, never from `metadata/datasets.csv`: the
   metadata describes the pre-Correction data (Correction: PubMed 36995229) and disagrees
   with the CSVs on cliff counts for 28 of 30 targets.
3. Integrity golden test: 30 files, 48,714 molecules, min 615, max 3,657.

Holdout, the 9 TDC ADMET **regression** tasks (the other 13 in `admet_benchmark` are
classification): caco2_wang, lipophilicity_astrazeneca, solubility_aqsoldb, ppbr_az,
vdss_lombardo, half_life_obach, clearance_hepatocyte_az, clearance_microsome_az, ld50_zhu.

External anchor: `RekerLab/DeepDelta`, `Results/` — 10 ADMET datasets x 5 folds x
{RandomForest, ChemProp, DeepDelta5} raw predictions. Metrics recomputed from the files;
no retraining. Both families must be checked to be scored on the same delta task before
the gap is formed.

## 6. Graph construction

Two distinct objects. Conflating them in code is the most likely structural bug.

**Similarity graph** (for the label statistics). Fingerprint: binary Morgan, radius 2,
2048 bits (ECFP4). Edges: kNN by Tanimoto, **k = 10 primary**, symmetrised by union
(mutual-kNN disconnects more often). Degree-controlled construction is primary because at a
fixed Tanimoto threshold of 0.5 the measured edge density across the 30 targets spans 0.41% to 40.57%
(98x), mean degree 3.0 to 249.1 (82x), and isolated-node fraction 0.5% to 36.4%; and on
identical labels the threshold-to-kNN switch moves adjusted homophily 12x. A headline
correlation computed across graphs of 98x differing density would be confounded with
density, which independently determines whether a GNN beats an MLP. Robustness: k in
{5, 20}, and the Tanimoto-threshold graph.

Precision about what kNN does and does not fix: union symmetrisation gives every node
degree **at least** `k`, not exactly `k`, because a molecule appearing in many neighbour
lists accumulates extra edges. Mean degree is therefore bounded below by `k` and still
varies across targets with the hubness of the series — far less than the measured 82x
spread of the threshold graph, but not constant. Consequences, both pre-registered:
the measured mean degree of every task graph is reported, and **mean degree enters the
incremental regression of §9 as a covariate** rather than being assumed away. The directed
kNN variant (exactly `k` out-edges, measures computed on the directed edge list) is
recorded as a robustness check for the case where the covariate carries real weight.

Fingerprints are computed in exactly one module. A second fingerprint call site is a bug.

**Comparison graph** (for Hodge). Anchors form a clique, plus spokes to the remaining
molecules. Without the anchor clique the graph is bipartite, contains no triangles, and
`curl == 0` by construction — the second half would then measure zero where zero is
structurally forced rather than empirically found.

`graphs/diagnostics.py` computes density, degree distribution, connected components,
triangle count and the Hodge dimensions, and these are **assertions in tests**, not lines
in a report: `triangles > 0` for any graph fed to the Hodge code; coverage (the fraction
of molecules participating in a measure) is recorded for every reported number; isolated
nodes are never silently dropped.

Triangle budget is hard. Measured: CHEMBL234_Ki at threshold 0.3 has 337,346 edges and
32,992,829 triangles, so the curl operator is about 1.1e13 entries. If the budget is
exceeded the code switches to a sampled cycle basis with a reported estimator variance,
and the switch is logged, never silent.

## 7. Measures

| Statistic | Computed on | Free parameters | Role |
|---|---|---|---|
| Target assortativity (Newman 2003) | raw `y [pEC50/pKi]` | 0 | **primary** |
| Unbiased homophily (arXiv 2412.09663) | `cliff_mol` | 0 in our hands (`alpha` pinned to GraphLand's value) | categorical arm, the group's current measure |
| Label informativeness (2209.06177 eq. 3) | `cliff_mol` | 0 | categorical arm |
| Adjusted homophily (2209.06177 eq. 2) | `cliff_mol` | 0 | reported for continuity with the 2023 vocabulary |
| ROGI-XD (2305.08238) | same fingerprint | reference-implementation defaults, pinned | **competing baseline the primary must beat** |
| Dirichlet energy of the label | `y`, normalised | 1 (normalisation) | secondary; presented as an import, not as "their metric" |

Definitions implemented from the sources, not from memory:

- Target assortativity: Pearson correlation of target values over the symmetrised edge
  list. Validated against `networkx.numeric_assortativity_coefficient`.
- Adjusted homophily, eq. (2) of 2209.06177:
  `h_adj = (h_edge − sum_k (D_k / 2|E|)^2) / (1 − sum_k (D_k / 2|E|)^2)`,
  where `h_edge` is the fraction of edges joining same-class endpoints and `D_k` is the
  summed degree of class-`k` nodes.
- Label informativeness, eq. (3): `LI = I(y_xi, y_eta) / H(y_xi)` over a uniformly random
  edge with random orientation, so the marginal is degree-weighted, `p(k) = D_k / 2|E|`.
- Unbiased homophily: formula to be read from arXiv 2412.09663 in the first task, with the
  `alpha` setting GraphLand uses. **Not to be written from memory.** Cross-checked against
  any reference values recoverable from GraphLand Table 1.

**Binning a continuous label is forbidden in code**: `measures/informativeness.py` raises
on continuous input. This is mechanical enforcement of the pre-registration, not
ergonomics. Measured justification: on CHEMBL2835_Ki, one graph and one bin count,
switching quantile to equal-width bins moves adjusted homophily from +0.050 to +0.375
(7.5x) and LI from 0.0021 to 0.2122 (100x), flipping the target from "no label structure"
to "strongly informative".

Circularity of `cliff_mol`, handled constructively rather than hidden: MoleculeACE defines
cliffs via substructure / scaffold / SMILES similarity at a 0.9 threshold with a 10-fold
activity difference (this wording is **UNVERIFIED** — confirm against the paper before it
is written down), while our graph is ECFP4-kNN. These are different relations. The
difference is stated in the text; an identical-relation version would be partly
tautological.

## 8. The dependent variable

One definition, in `models/gap.py`.

Split: the `split` column shipped with MoleculeACE (public, fixed, identical for both
arms). Metrics: RMSE and RMSE on cliff compounds, as MoleculeACE reports them.

- Pointwise arm: ECFP4 + SVM, + LightGBM, + MLP. SVM because it is the actual bar — from
  MoleculeACE's own shipped numbers, ECFP-SVM takes 21 of 30 targets with mean rank 1.60
  against 3.00 for ECFP-GBM, so naming GBM as the baseline inflates every reported gap.
  MLP because SQRL Table 1 shows tree models gain nothing from pairing (XGBoost 0.79 to
  0.76, RF 0.80 to 0.77), and a tree-only comparison would yield a degenerate gap.
- **kNN floor arm** (label-access floor): `y_hat = (1/m) sum_i y_i` over the same `m`
  nearest training molecules by Tanimoto **on the single fingerprint of §6**, uniform weights,
  no learned function at all.
  Deterministic, no hyperparameter to select, no seed. This arm exists because the pairwise
  arm reads `m` measured labels at test time while the pointwise arm reads none, so a raw
  pointwise-against-pairwise gap partly measures **test-time label access rather than
  architecture**. Uniform weights and the identical anchor set are required, not
  conveniences — see the identity below.
- Pairwise arm: the same three base learners on pair features, in two variants —
  difference only (`xi − xj`, SQRL-like) and concatenation with difference
  (`[xi, xj, xi − xj]`, PADRE-like). Inference by averaging over the same `m` anchors using
  their measured labels.
- Sign convention is fixed explicitly and stated. SQRL's published equations are
  inconsistent: eq. 3 under eq. 2's convention estimates `2 y_i − y_new`. We do not
  reproduce SQRL and we say so.

**Primary gap**, pre-registered: within each arm the model is selected by **5-fold
cross-validation on the training split only**, by mean CV RMSE, then refit on the full
training split and evaluated once on the shipped test split. Anchor count is **m = 10**
nearest training molecules — a separate constant from the similarity graph's `k`, and not to
be tied to it. The three seeds vary model initialisation and the CV fold assignment, and
nothing else.

`gap_t = RMSE_pointwise_selected(t) − RMSE_pairwise_selected(t)`, so a positive gap means
pairwise wins. Secondary, also pre-registered: per-base-learner gaps, and the same
quantities under RMSE on cliff compounds.

**The decomposition, which is exact.** With uniform weights over the same `m` anchors,

```
y_hat_pairwise = (1/m) sum_i [ y_i + f(x_i, x_new) ]
               = (1/m) sum_i y_i  +  (1/m) sum_i f(x_i, x_new)
               = y_hat_kNN        +  mean learned correction
```

so the pairwise prediction is the kNN prediction plus the mean learned correction, as an
algebraic identity rather than an interpretation.

Separately, and trivially, the gap telescopes:

```
gap_t = [ RMSE_pointwise − RMSE_kNN ]  +  [ RMSE_kNN − RMSE_pairwise ]
      =        access_t               +         correction_t
```

These are two different facts and must not be fused. The telescoping is exact for any three
numbers and carries no content by itself. What the prediction identity adds is the
**attribution**: because the pairwise arm differs from the kNN arm by exactly the mean of `f`,
`correction_t` is the error change caused by the learned pairwise function and by nothing else
— same anchors, same weights, same split. RMSE is nonlinear, so `correction_t` is **not** a
variance-like share of anything and must never be reported as a percentage of `gap_t`; it is
an error difference attributable to one named component. `access_t` is what test-time label
access buys; `correction_t` is what the learned pairwise function buys on top of it. Both are pre-registered outcomes. This also makes the identity
and the uniform weighting load-bearing: distance weighting or a different anchor set in
either arm destroys the decomposition and the third arm stops being a floor.

## 9. Analysis plan (frozen before any measurement)

- **Positive control, reported before anything else**: Spearman rho between target
  assortativity and `access_t`. kNN regression works precisely when the label is smooth on
  the kNN graph, so assortativity must predict this component; a 95% interval covering zero
  here indicts the measurement pipeline, not the hypothesis, and nothing downstream is
  interpreted until it is resolved.
- Primary: Spearman rho between target assortativity and the primary gap across the 30
  MoleculeACE targets.
- **Decisive secondary**: Spearman rho between target assortativity and `correction_t`, plus
  the distribution of `correction_t` itself with a cluster-bootstrap interval for its mean.
  `correction_t` indistinguishable from zero is the half-1 statement of hypothesis 2 — that a
  pairwise model is a pointwise model plus anchor averaging — reached through held-out error
  and without any Hodge machinery. §10 tests the same statement through the energy of the
  flow. Agreement is a strong result; disagreement localises which of the two instruments is
  wrong, and either outcome is reported.
- Incremental: gap regressed on {target assortativity, ROGI-XD, mean degree, log n};
  report the incremental contribution of assortativity with its interval. Mean degree is
  included because union-symmetrised kNN bounds degree below by `k` without fixing it
  (§6); the construction removes the threshold graph's 82x spread, it does not remove the
  covariate.
- Uncertainty: cluster bootstrap over the 6 receptor classes (GPCR 12, Kinase 6, NR 6,
  Other 3, Protease 2, Transferase 1), 10,000 resamples, percentile intervals.
- Decision rule: the primary claim is supported iff the 95% cluster-bootstrap CI for rho
  excludes zero **and** the incremental contribution over ROGI-XD has a CI excluding zero.
  There is **no sign test**: at n=5 the exact permutation critical value is |rho| = 1.0,
  and a sign-level null kills a true rho of 0.5-0.7 in 9-20% of samples.
- `n` and `n_eff` are printed together everywhere. The 30 tasks are not independent:
  CHEMBL237_Ki and CHEMBL237_EC50 are the same protein, 26.7% of molecules appear in more
  than one task, JAK1/JAK2 overlap 88.5%. Block-exchangeable `n_eff` is about 10 at an
  intra-class correlation of 0.3, and the assumed correlation is stated with the number.
- Seeds: 3, averaged, for the learned arms only. The kNN floor is deterministic and is
  computed once. No best-seed selection anywhere.
- Holdout: once the primary is computed and recorded, the 9 TDC tasks are run exactly
  once. Sign agreement and rho are reported. No re-tuning afterwards.
- External replication: the same correlation against DeepDelta's published 10-task gap,
  reported with its own interval and its own n.

Pre-registration artifact: `prereg/increment1.yaml`, committed and referenced by commit
hash from every figure, before any measurement is computed. Same discipline as
`data/cage/prospective_prereg.yaml` in fep-surrogate.

## 10. Half 2

**The theorem, in the form that is true.** With the encoder `g` frozen and a bias-free
linear readout `w`, the edge flow is `F_ij = w . g(x_i) − w . g(x_j) = phi_i − phi_j`,
exactly a gradient flow, and the model collapses to a pointwise model plus anchor
averaging. With `g` frozen the linear-head problem is convex, so the collapse is provable
rather than measured. The encoder must **not** be trained end to end: a linear head would
push nonlinearity down into `g` and the ablation would stop isolating the readout. The
proposal's own "one fixed encoder" decision already supports the frozen variant.

The real boundary is linear versus nonlinear readout, not difference-of-encoders versus
joint encoding: by Cauchy, `h` is additive on triangles iff it is linear. Two of these are theorems and are tested as such: a bias-free linear head gives curl
exactly zero (measured 4.1e-17 for a least-squares fit on exactly additive targets, i.e.
machine zero), and an affine head `w.z + c` gives exactly `3c` on every triangle. The
nonlinear values from the same probe (`tanh` +0.168, `relu` +5.913) are **probe-specific
magnitudes, not constants**: the regression test asserts the theorems and asserts only that
the nonlinear heads give curl bounded away from zero, with the probe's generating setup
fixed in the test fixture.

**Dimension census first, no training.** Build `B1` (vertex-edge) and `B2` (edge-triangle),
assert the fundamental identity `B1 @ B2 == 0`, and report per target:
`dim gradient = rank(B1) = |V| − c`, `dim curl = rank(B2)`, and
`dim harmonic = |E| − rank(B1) − rank(B2)`. This is the cheapest deliverable in the
project and it already carries a provisional result. On a **synthetic probe** (n = 600,
ECFP-like sparse fingerprints, kNN — not a MoleculeACE target), k = 5 gave 1,876 edges with
71 triangles against a gradient subspace of dimension 599, and k = 20 gave 6,762 edges with
2,875 triangles, so the harmonic component ran 2-5x larger than the gradient component the
proposal wants to extract while curl had the smallest support of the three. **This provenance
matters: the census on the 30 real targets is the deliverable, and the probe only says what
to expect.** The probe numbers are not quotable as a result. On threshold graphs the picture is worse: CHEMBL234_Ki at
0.7 breaks into 925 components with 637 isolated nodes and a largest component holding
11.9% of nodes, so the gradient potential is fixed only up to a constant **per component**
and "consistent absolute values" does not exist. Harmonic flows are both curl-free and
divergence-free yet globally inconsistent (Jiang, Lim, Yao, Ye 2011), so the proposal's
two-way gradient-plus-circulation language is wrong, not merely incomplete.

**Energy budget.** Orthogonal projections by least squares: gradient
`B1^T phi` with `phi = (B1 B1^T)^+ B1 f`; curl `B2 psi` with `psi = (B2^T B2)^+ B2^T f`;
harmonic as the residual. Report the three fractions of `||f||^2`.

**Controls: three, not one.** The original null is void — the target flow
`dy_ij = y_j − y_i` is the gradient of a node labelling and is therefore curl-free
identically, under true labels **and** under any permutation of them (measured maximum
triangle circulation 0.00e+00 in both arms on five real graphs). And raw-norm comparison
is scale-confounded: shuffling inflates the mean-square edge target by 2.1x to 11.7x.
So the statistic is the scale-invariant curl **fraction**, and the controls are: the
trained model; the same architecture on shuffled labels **with total flow norm matched**;
and an exactly curl-free floor built by differencing a trained pointwise model on the same
edges.

**Cross-check against half 1.** `correction_t` of §8 and the curl-plus-harmonic energy
fraction here are two independent measurements of the same claim: that a trained pairwise
flow carries nothing beyond a node potential. A near-zero `correction_t` with a large
non-gradient energy fraction, or the reverse, is informative about the instruments and is
reported as such rather than reconciled by choosing the friendlier number.

**The question half 2 must answer, or it has no contribution.** Does the per-edge curl
fraction predict that edge's held-out error **better than plain anchor dispersion**?
Anchor dispersion is published three times (PADRE 2021, TNNR, Zhang et al. 2023) and the
comparator for triangle non-closure is DeepDelta's 0.127 +/- 0.043. If it does not beat
anchor dispersion, that is written as a measured null, not omitted.

**Observability.** Transfer `h_e + w_e * Omega_e = 1` from the RBFE cycle-closure preprint
(10.26434/chemrxiv.15007931/v3, verified live in Crossref on 2026-10-03) to state which
edges are detectable given the comparison structure.

## 11. Risks

| Risk | Status |
|---|---|
| The headline reduces to "ROGI predicts the gap" | Addressed by design: the claim is incremental over ROGI-XD, and the translation between vocabularies is a stated contribution rather than a defect. |
| Harmonic dominance makes HodgeRank meaningless on these graphs | This is a finding, not a failure. Reported as the census result. |
| Triangle explosion on the energy budget | Hard budget plus sampled cycle basis with reported variance. The energy budget across all 30 targets is the one piece genuinely at risk before 2026-10-05. |
| n_eff about 10 | Cluster bootstrap, and `n_eff` printed beside `n`. The conclusion is phrased as variance explained with an interval, never as "we showed". |
| A raw pointwise-against-pairwise gap measures test-time label access, not architecture | Removed by the kNN floor arm of §8: the gap decomposes exactly into `access_t` and `correction_t`, and both are pre-registered outcomes. |
| `cliff_mol` circularity | Graph built on a different relation than the cliff definition uses; stated in the text. |
| Unverified bibliography entries | Five entries listed in §2 to be opened by eye before any text is written. |

## 12. Repository architecture

```
molace/
  pyproject.toml                   uv-pinned
  prereg/increment1.yaml           frozen analysis plan, referenced by commit hash
  src/molace/
    data/       moleculeace.py  tdc_admet.py  deepdelta.py
    graphs/     fingerprints.py  knn.py  threshold.py  comparison.py  diagnostics.py
    measures/   assortativity.py  homophily.py  informativeness.py  dirichlet.py  rogi.py
    models/     pointwise.py  anchors.py  knn_floor.py  pairwise.py  gap.py
    hodge/      complex.py  decompose.py  energy.py
    analysis/   correlate.py  figures.py
  tests/
```

Boundaries: `data/` yields tidy per-task frames and knows nothing about graphs;
`graphs/` knows nothing about labels beyond carrying them; `measures/` takes a graph plus
a label vector and returns one number plus its coverage; `models/` never touches the
similarity graph except to draw anchors, and anchor selection lives in exactly one module
(`anchors.py`) because the kNN floor and the pairwise arm must draw the identical set or the
§8 decomposition is not exact; `hodge/` takes a comparison complex plus an edge
flow; `analysis/` takes a table of per-task numbers. Each unit is testable without the
others.

Error handling: every task is computed independently and cached to a per-task JSON, so a
crash on target 23 does not lose the first 22. Coverage is recorded for every reported
number. Budget switches are logged.

## 13. Testing

- Unit: target assortativity against `networkx.numeric_assortativity_coefficient`;
  adjusted homophily and LI against hand-computed two-class examples; unbiased homophily
  against the source paper's stated properties; `B1 @ B2 == 0`; gradient orthogonal to
  curl; the linear-readout curl values of §10 as regression tests.
- Identity: `y_hat_pairwise == y_hat_kNN + mean_i f(x_i, x_new)` to floating-point
  tolerance on a fitted model, and the three RMSE differences telescope to `gap_t` exactly.
  This is the test that keeps the §8 decomposition honest if anyone later changes the anchor
  weighting.
- Property: label shuffling leaves graph diagnostics unchanged; a permutation null for LI
  as a one-line sanity check (the measured observed-to-null ratio ran 33x to 15,510x, so
  no MI-estimator correction work is budgeted).
- Golden: the MoleculeACE integrity numbers of §5.
- Precondition: any graph entering `hodge/` has `triangles > 0`.

## 13a. Preconditions that gate the text (not the code)

These are open by design rather than by omission, and each has an owner task and an
acceptance test. None blocks writing code; each blocks writing a sentence that depends on it.

1. ~~**Unbiased homophily formula** read from arXiv 2412.09663, `alpha` set as GraphLand sets
   it.~~ **CLOSED 2026-10-03.** Equation (3) transcribed from the paper into
   `src/molace/measures/unbiased.py`, recommended form `alpha = 0`, so no free parameter in our
   hands. All four defining properties pass as tests, including minimal agreement. The paper's
   prose leaves the normalisation of `C` implicit; it is both orientations of every edge divided
   by `2|E|`, which is the convention under which the paper's own stated values (0 for a balanced
   random labelling, +1 for perfect separation, -1 for a fully heterophilous one) come out right.
2. **MoleculeACE cliff definition** (substructure / scaffold / SMILES similarity at 0.9
   with a 10-fold activity difference) confirmed against van Tilborg et al. 2022 directly.
   Acceptance: the quoted thresholds appear in the paper. The circularity paragraph of §7
   depends on this and must not be written before it.
3. **Five bibliography entries opened by eye**: arXiv 2409.14500, 2508.20906, 2509.21489,
   2601.04507, 2508.16495. They were batch-confirmed by title match by the same method that
   mis-described 2509.18893. Acceptance: abstract read, description in our text matches it.
4. **SALI** given a real citation (Guha & Van Drie) or dropped from the prose.

## 14. Deliverables before 2026-10-05

1. `prereg/increment1.yaml` committed before any measurement.
2. Half 1 end to end on 30 targets: measures, self-computed gap across **three arms**
   (pointwise, kNN floor, pairwise) with its exact `access` / `correction` decomposition, primary correlation with
   cluster-bootstrap interval, incremental contribution over ROGI-XD, DeepDelta external
   replication, TDC holdout opened once.
3. Hodge dimension census on 30 targets.
4. The frozen-encoder readout ablation.
5. One figure per result, and a README that states what the audit refuted in the original
   proposal.

## 15. References to add that the original proposal lacks

- Platonov, Kuznedelev, Babenko, Prokhorenkova. Characterizing Graph Datasets for Node
  Classification: Homophily-Heterophily Dichotomy and Beyond. arXiv 2209.06177, NeurIPS
  2023. (The defining paper for both measures; absent from the proposal.)
- Mironov, Prokhorenkova. Revisiting Graph Homophily Measures. arXiv 2412.09663, LoG 2024.
- Newman. Mixing patterns in networks. Phys. Rev. E 67, 026126, 2003.
- Maggiora, Bajorath. Chemical space networks: a powerful new paradigm for the description
  of chemical space. J. Comput. Aided Mol. Des. 2014, and the 2014-2023 CSN line.
- Wetzel et al. Twin Neural Network Regression. arXiv 2012.14873; arXiv 2106.06124;
  arXiv 2301.01383.
- Zhang et al. Similarity-based pairing improves efficiency of siamese neural networks for
  regression and uncertainty quantification. J. Cheminform. 15:75, 2023.
- Fralish, Reker. Pairwise learning for molecular property prediction and optimization.
  Front. Drug Discov. 6:1859068, 17 June 2026.
- Kramer. Nonadditivity Analysis. J. Chem. Inf. Model. 2019.
- Mueller et al. Extended Graph Assessment Metrics for Graph Neural Networks.
  arXiv 2307.10112, 2023. (Regression variants of homophily.)
- van Tilborg, Alenicheva, Grisoni. Correction to "Exposing the Limitations of Molecular
  Machine Learning with Activity Cliffs". PubMed 36995229.
- Guha, Van Drie. SALI. (Named in the proposal's prose with no entry.)
- Sharma, Sharma. When Design Rules Break. arXiv 2606.10249, **withdrawn** 2026-08-29
  because "after fixing the pipeline, the main correlation is no longer significant" —
  cited as the cautionary precedent for exactly this correlation design.

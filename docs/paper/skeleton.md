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

**Title, working:** "Homophily measures are a first moment: what graph statistics do and do not say
about attainable accuracy on molecular property data"

**Claims, each with the evidence that backs it**

| # | Claim | Evidence | Strength |
|---|---|---|---|
| 1 | Target assortativity predicts attainable accuracy | rho +0.875 / +0.909 / +0.810 on 40 independently curated ChEMBL targets; +0.77 / +0.82 / +0.76 on 10 DeepDelta benchmarks with published models | strong, four collections |
| 2 | It does so beyond the existing roughness index | assortativity over ROGI +0.846 [+0.678, +0.928]; ROGI over assortativity +0.098, covers zero; ROGI alone does not predict | strong, pre-registered |
| 3 | The link is not an artefact of one fingerprint | +0.726 on RDKit descriptors; paired drop off ECFP4 +0.163 [+0.022, +0.358] | moderate, one alternative representation |
| 4 | It also says which model class wins, through the level | floor minus pointwise -0.343 [-0.619, -0.008]; floor wins 26/40, and 12/13 on the hardest third | moderate, interval thin |
| 5 | For the kNN mean the link is exact, and equals a Dirichlet energy | identity verified to 1e-16 on regular graphs; 0.025-0.031 mean absolute error as a point prediction over k = 3..40 on real targets | strong, but the object is known |
| 6 | Equal weights are already near-optimal locally | best linear predictor on the same neighbourhood beats the mean by +0.0013 skill at k = 10 under exchangeability | strong and new to us; needs the exchangeability assumption relaxed |
| 7 | The nugget-style noise estimate fails out of sample, and we know why | 2/30 violations against a pre-registered threshold of 1; synthetic separation into ill-conditioning and mixed hop lengths; 1/c(d)^2 overstatement predicted and measured to 0.3 % | strong as a negative result |
| 8 | A structural limit: the regime where the second statistic matters most is the regime where the estimator built on it does not work | five refused design gates, variance decompositions 4.2 / 0.0 / 14.1 / 10.8 / 0.9 % | honest, and the most quotable sentence in the paper |

**Claim 6 is the one worth chasing.** If equal weights are within a thousandth of the best local
linear predictor, then the gap between the floor and a fitted model cannot come from weighting the
neighbourhood better. It must come from information outside it. That is a testable reframing of the
paper's own empirical crossover and it is not, as far as the checks went, in either literature.

## Section plan

1. **Setup.** The dataset as a comparison graph; what is measured; the three arms.
2. **The measure predicts attainable accuracy.** Claims 1-3. Charts, not tables.
3. **What the measure is.** Claim 5: the identity, its reading as a Dirichlet energy, assortativity
   as the first moment, and the second moment nobody reports. Cite the handbook; claim the accounting,
   not the object.
4. **What averaging already achieves.** Claim 6 and its consequence for claim 4.
5. **A noise floor from two moments, and why it fails.** Claim 7, framed as a graph replication of
   nugget estimation, citing Atkinson, Kim, Zimmerman.
6. **A limit on the whole approach.** Claim 8.
7. **Method.** Pre-registration, the refused gates, what was frozen when.

The Hodge half does not appear. It is a separate, weaker story and it dilutes this one.

## What has to happen before a word of the draft is written

1. **A real prior-art pass**, not three searches. Geostatistics (nugget estimation, kriging variance,
   network design), graph signal processing (polynomial filters, spectral smoothness), and the
   homophily-measure line (Platonov, Mironov, GraphLand, GraphPFN). Expect more collisions.
2. **Relax exchangeability in claim 6.** Nearer neighbours correlate more; the closed form assumes
   they do not. Redo with the measured correlation profile.
3. **Decide the venue, which decides the paper.** JCIM or Digital Discovery, with the molecular
   programme leading and the transport as framing. Or LoG, with the transport leading and the
   molecules as the testbed. These are different papers and the choice cannot be deferred.
4. **Talk to Prokhorenkova about co-authorship.** Her group knows this literature; two of the three
   collisions above would have been obvious to them in a minute, and there are likely more.

## Risk register

- **Novelty.** After three checks the theoretical contribution is a known object in new notation. If
  a fourth check finds the exact identity stated somewhere, section 3 becomes a citation and the paper
  leans entirely on sections 2, 4, 5 and 6. It would survive that.
- **Claim 4's interval** clears zero by 0.008. Robust to the recovered-rounding check (200/200), but
  it is thin and a reviewer will say so first.
- **Claim 3** rests on one alternative representation. It is reported as one, not as generality.
- **Scope.** Thirty ChEMBL targets plus forty plus ten benchmarks is a lot of molecules and one domain.
  Nothing here is shown outside molecular property prediction.

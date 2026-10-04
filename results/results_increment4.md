# H2 is false, and what replaces it

Pre-registration: `prereg/increment4_separation_and_representation.yaml`, blob
`c8ebac86ffd1f0813144121c84927632fef1ab1e`, frozen before either experiment ran. Reproduce with
`uv run python scripts/separation_chembl.py` and `uv run python scripts/representation.py`; output in
`results/report_separation_chembl.txt`, `results/report_representation.txt`,
`results/separation_chembl.csv` and `results/representation.csv`.

Both experiments use the 40 ChEMBL targets of `prereg/increment3_chembl.yaml`, blob
`7519ef31c4374637483bae167221f9b7567ece7f`. Those targets share no identifier with the 30
MoleculeACE targets the claim was found on, and they are four times the n at which H2 was last
tested.

## Experiment A: the statistic does predict which model class wins

H1 held for a fourth time, on every arm:

| | rho | 95% CI | |
|---|---|---|---|
| skill of the pointwise arm | +0.875 | [+0.717, +0.955] | excludes 0 |
| skill of the kNN floor | +0.909 | [+0.780, +0.968] | excludes 0 |
| skill of the pairwise arm | +0.810 | [+0.624, +0.913] | excludes 0 |

H2 did not:

| | rho | 95% CI | half-width | |
|---|---|---|---|---|
| pointwise minus pairwise | -0.010 | [-0.322, +0.294] | 0.308 | covers 0 |
| **floor minus pointwise** | **-0.343** | **[-0.619, -0.008]** | **0.305** | **excludes 0** |
| floor minus pairwise | -0.253 | [-0.537, +0.080] | 0.309 | covers 0 |

The frozen rule's branch for this case is unambiguous: "the separation claim is WRONG as stated and
the project says so." It is wrong as stated. Every results document and both deliverables that lead
with H2 are corrected to lead with what follows instead.

The interval clears zero by 0.008, so the honest description is that the test fires narrowly. The
effect itself is moderate at rho = -0.343, and the narrow part is the interval's edge, not the
effect.

### What the sign means

The floor beats the pointwise model on 26 of 40 targets, by +0.027 on average, and the advantage
falls as the statistic rises:

| targets, by attainable level | mean pointwise skill | mean floor advantage | floor wins |
|---|---|---|---|
| lowest third | +0.173 | +0.072 | 12 / 13 |
| middle third | +0.327 | +0.015 | 8 / 14 |
| highest third | +0.462 | -0.005 | 6 / 13 |

Where the attainable accuracy is low, averaging ten neighbours beats fitting a model almost every
time. Where it is high, fitting wins the advantage back. The same arithmetic that makes the kNN floor
a positive control predicts this sign: the floor's relative error is

    1 + 1/k + (k-1)/k * rho_between_neighbours - 2 * assortativity

and the 1/k term is a fixed price for averaging only k = 10 noisy neighbours. That price does not
shrink when the signal improves, while the cost of overfitting does, so a fitted model must overtake
the floor as the signal fraction grows.

### Post hoc, not pre-registered: the statistic adds nothing to the level

| | rho | 95% CI | |
|---|---|---|---|
| statistic vs floor advantage, level residualised out | +0.204 | [-0.149, +0.518] | covers 0 |
| level vs floor advantage, statistic residualised out | -0.338 | [-0.594, -0.012] | excludes 0 |

This is consistent with a chain: the statistic predicts the attainable level, the level predicts
which arm wins, and the statistic carries nothing about model choice of its own. It is not evidence
for that chain. The first interval's half-width is 0.33, which is not narrow, so what this shows is
that the data do not separate the chain from a direct link. Increment 2 made exactly this mistake by
reading a wide interval around zero as a null, and this document does not repeat it.

The logical error in H2 was its form rather than its numbers. H2 is a claim about a MARGINAL
relation, and a marginal relation cannot vanish while the level it is measured through predicts the
outcome. If the level predicts which arm wins, then every predictor of the level predicts which arm
wins. H2 could only have been true if the level carried no model-choice information at all.

### A descriptive threshold, selected on the same data

Chosen by maximising the split on these 40 targets, therefore optimistically biased and NOT a
validated cutoff. The ordering is the finding; the number is an orientation.

| split | below | above |
|---|---|---|
| rank assortativity 0.546 | floor wins 79% (19 targets), +0.047 | 52% (21 targets), +0.009 |
| pointwise skill 0.347 | floor wins 86% (22 targets), +0.052 | 39% (18 targets), -0.003 |

## Experiment B: the link survives a different representation, and weakens by a measurable amount

The graph and the statistic stay on ECFP4 Tanimoto throughout. Only the model's features change.

| model's representation | rho with the ECFP4-graph statistic | 95% CI | | skill-similarity to ecfp4 |
|---|---|---|---|---|
| ecfp4 | +0.889 | [+0.750, +0.956] | excludes 0 | 1.000 |
| **rdkit_descriptors** | **+0.726** | **[+0.481, +0.877]** | **excludes 0** | 0.767 |
| maccs | +0.893 | [+0.757, +0.956] | excludes 0 | 0.928 |
| atompair | +0.915 | [+0.805, +0.964] | excludes 0 | 0.970 |

The frozen rule's condition is met: a statistic computed on the ECFP4 graph predicts the skill of a
model that never sees ECFP4. The link is not an artefact of one fingerprint. As the pre-registration
says in its own note, this does not make the claim representation-free; it makes it survive one
genuinely different representation, which is one more than zero.

### Post hoc, not pre-registered: the paired drop is real

The frozen rule asked only whether each correlation excludes zero. The sharper question is whether
the correlation FALLS as the representation moves away from the one the graph was built on. Same 40
targets, so a paired bootstrap, 10000 draws, seed 0:

| | difference | 95% CI | |
|---|---|---|---|
| rho(ecfp4) - rho(rdkit_descriptors) | +0.163 | [+0.022, +0.358] | excludes 0 |
| rho(ecfp4) - rho(maccs) | -0.004 | [-0.087, +0.072] | covers 0 |
| rho(ecfp4) - rho(atompair) | -0.027 | [-0.088, +0.016] | covers 0 |

The drop appears for the physicochemical descriptors and not for the two substructure fingerprints
that are near-duplicates of ECFP4. So the statistic measures a mixture: a part that bounds every
model whatever it reads, and a part specific to the representation the graph was built from.

One prediction of that reading is NOT observed. If the ECFP4-specific part were the whole story,
ECFP4 would carry the largest correlation. Atom-pair edges it out by 0.027, inside the paired
interval, so the ordinal prediction holds and the strict one does not.

## What this costs the project's story, and what it buys

Lost: H2 as stated, after surviving two earlier tests that lacked the power to kill it. The DeepDelta
test at n = 10 could not reject anything, and increment 2 correctly downgraded it from "supported" to
a non-rejection rather than claiming the null. At n = 40 the difference that was hiding there fired.

Gained: a positive claim in place of a negative one. The statistic is computable from the graph alone,
with nothing trained, and it tracks whether fitting a model will beat averaging the neighbours. The
mechanism is arithmetic that was already in the project's own derivation of the kNN floor.

## Operational note: the checkpoint and the recovered rows

Both scripts originally wrote their table only after the last target and were killed by a two-hour
background limit, A at 28 of 40 and B at 30 of 40. Both now checkpoint every target to
`results/*_partial.csv` and resume by skipping what is already there, and the interrupted targets were
recovered from the printed logs rather than recomputed. Recovered rows carry
`provenance = recovered_3dp`: their statistics were joined from `results/rogi_chembl.csv` at full
precision, and their skills were read back from the log at three decimals.

Every decision quantity in the pre-registration is rank-based, so three decimals cannot change a
Spearman or a cluster bootstrap unless they flip a rank. Because experiment A's interval clears zero
by only 0.008, that was checked rather than asserted: 200 replicates perturbing each recovered skill
uniformly within its rounding, 2000 draws each. The interval excluded zero in 200 of 200 replicates,
upper edge ranging over [-0.021, -0.002]. The verdict does not depend on the recovered digits.

## The theory under the finding: the floor's error is an identity, and it was checked

Post hoc, not pre-registered. `scripts/floor_identity.py`, output in
`results/report_floor_identity.txt` and `results/floor_identity.csv`.

Centring the label, with sigma^2 = Var(y) and the floor predicting the mean of a node's k
neighbours, expanding the square gives

    E(y_v - yhat_v)^2 / sigma^2  =  1 + 1/k + (k-1)/k * rho_nn  -  2 * r

with r the mean label correlation across an edge, which IS target assortativity by definition, and
rho_nn the mean label correlation between two nodes sharing a neighbour. No assumption about the data
enters; this is arithmetic. Both statistics are read off the graph with nothing trained, so the
identity is a POINT prediction of one arm's error, falsifiable in one run.

Measured on the 40 ChEMBL targets, with k = 10, the floor's own averaging width:

| | |
|---|---|
| mean absolute error | **0.0256**, against a predicted quantity ranging 0.108 to 0.819 |
| mean signed error | -0.0127 |
| max absolute error | 0.0837 |
| Pearson(predicted, measured) | **+0.983** |
| Spearman(predicted, measured) | +0.950 |

The sign of the bias matches a reason stated before the run: the measured floor averages the k nearest
TRAINING molecules of a test molecule while both statistics are computed over the whole graph, so the
real neighbours are fewer and worse and the measured error must exceed the predicted one. It does.

Two consequences.

**For the project.** On this arm, "assortativity predicts attainable accuracy" is a theorem rather
than a finding, which is exactly why the arm is a positive control: a negative result there would
have meant a broken pipeline, not a refuted hypothesis.

**For the measure.** Assortativity enters with coefficient -2 and neighbour-to-neighbour agreement
with (k-1)/k, about +0.9, which is comparable. GraphLand reports the first and not the second. On
these 40 targets the two correlate at +0.988, so assortativity alone reaches +0.934 against the full
identity's +0.950: the second statistic is necessary for the identity and nearly redundant for
ranking on graphs of this kind. That yields a testable prediction, untested here: assortativity alone
should stop working where neighbour agreement decouples from edge agreement, which is to say across
graphs of sharply different clustering.

### What does NOT follow

Substituting (1 - r) for the representation's Bayes error gives a tempting parameter-free ceiling,
skill <= 1 - sqrt(1 - r). It is **violated on 28 of 40 targets**, by up to +0.117. The reason is the
same neighbourhood smearing: neighbours do not coincide in x, so Cov(f(x_v), f(x_u)) < Var f(x), the
measured r understates the signal fraction, and the formula lands BELOW the true ceiling rather than
above it. The conjecture holds ordinally, Spearman +0.889, and fails in level. It is reported as a
conjecture.

The crossover between averaging and fitting is also not derived. In the co-location limit the floor's
excess over Bayes is exactly (1 - r)/k, one k-th of the irreducible noise, so the floor wins whenever
a fitted model's excess risk exceeds that. How large a fitted model's excess risk is, the derivation
does not say, and part of it here is the absence of model selection, which this project skipped for
cost. `scripts/floor_identity.py` prints an "implied crossover" with median 0.526 against the
descriptive split measured at 0.546; that near-coincidence is NOT offered as a prediction, because the
right-hand side of it is the pessimistic proxy that the previous paragraph shows to be violated.

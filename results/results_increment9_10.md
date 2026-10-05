# Synthetic graphs with a controlled decay: what they settled, and the tension they exposed

Pre-registrations: `prereg/increment9_synthetic.yaml`, blob
`85f1e3bb67f7aba04753a06b424b7d6716fcc328`; `prereg/increment10_field_profile.yaml`, blob
`5de3f710cb9a89aef11498965b1e3c79bf00cbc6`. Both frozen before any synthetic data existed. Reproduce
with `uv run python scripts/synthetic_decay.py` and `uv run python scripts/field_profile.py`; output in
`results/report_synthetic_decay.txt`, `results/report_synthetic_conditioning.txt`,
`results/report_field_profile.txt`, `results/synthetic_decay.csv` and `results/field_profile.csv`.

Real datasets never reveal their true noise fraction, so increment 6's failure, 2 ceiling violations of
30 where at most 1 was allowed, could not be attributed to the decay model or to the data. Synthetic
graphs can attribute it, and the contrast is set rather than hoped for.

## The setting

Nodes sit on a ring of 600. The signal is a stationary Gaussian field; independent noise takes a
prescribed share `nu` of the variance, and the TRUE fraction is measured from the realisation rather
than assumed, so a clipped spectrum cannot move it silently. Graphs are circulants joining node `i` to
`i +/- j*d` for `j = 1 .. k/2`, so every edge spans a ring distance that is a multiple of `d`.

At `k = 2` every edge spans exactly `d` and every shared-neighbour pair exactly `2d`, which makes the
geometric model exact: `r = (1-nu) c(d)` and `rho_nn = (1-nu) c(d)^2`, so `r^2 / rho_nn = 1 - nu`
whatever `d` and the correlation length are. At larger `k` an edge spans several distances at once and
the model must degrade.

**Reported unconditionally, as both plans required.** These circulants are k-regular, so the floor
identity is exact here by the proof `tests/test_neighbourhood.py` pins. Measured over 1200 synthetic
datasets its largest absolute error is **3.58e-15**. That is the implementation agreeing with its own
proof, not an empirical finding, and it is reported because an implementation that disagreed would
invalidate everything built on it.

This figure read 3.2e-15 until 2026-10-05, which was wrong for a reason unrelated to the seeding fix
described at the foot of this document: it quoted `field_profile`'s own maximum and attached it to all
1200 datasets, while `report_synthetic_decay.txt` had printed 3.58e-15 for its own 600 all along. The
combined maximum is the larger of the two and comes from increment 9, whose data are bit-identical
before and after the fix.

## Increment 9, question 1: the estimator has two distinct failure modes

Against the frozen rule, **the exact corner fails**: at `k = 2` the mean absolute error of
`r^2 / rho_nn` against the realisation's true signal fraction is **0.4150**, far above the 0.05
threshold, with a single dataset off by **40.85**. The frozen branch says the estimator is wrong for a
reason other than mixed hop lengths and that leads the report. It does, and the reason is visible in
that 40.85.

### Mode one: ill-conditioning

`r^2 / rho_nn` divides by a quantity that can be statistically indistinguishable from zero. At `k = 2`
each node contributes exactly one shared-neighbour pair, so `rho_nn` rests on 600 observations and its
standard error is about 0.04; where the true value is 0.015 the ratio explodes.

| k | pairs per node | share of datasets with rho_nn inside 3 standard errors of zero | Spearman(abs recovery error, 1 / rho_nn in its own standard errors) |
|---|---|---|---|
| 2 | 1 | 0.180 | **+0.716** |
| 4 | 6 | 0.073 | +0.528 |
| 10 | 45 | 0.060 | +0.548 |
| 20 | 190 | 0.027 | +0.556 |

The correlation is computed within each degree, because pooling confounds it with `k`; pooled it reads
+0.027, which is a Simpson effect and not the quantity of interest.

### Mode two: mixed hop lengths, once the first mode is excluded

Post hoc, not pre-registered: restricting to datasets where `rho_nn` exceeds ten of its own standard
errors, 439 of 600 survive, and the picture is clean.

| k | datasets | mean abs recovery error | mean signed | share of error that is bias |
|---|---|---|---|---|
| 2 | 59 | **0.0237** | -0.0053 | 0.224 |
| 4 | 120 | 0.1136 | -0.1121 | 0.987 |
| 10 | 129 | 0.2224 | -0.2224 | 1.000 |
| 20 | 131 | 0.3384 | -0.3384 | 1.000 |

So the estimator **is** sound where its model is exact, to 0.0237, below the frozen threshold it failed
unconditionally. And the residual error grows with `k` at Spearman **+1.000**, bootstrap
[+1.000, +1.000] over cells, and turns from scatter into pure bias between `k = 2` and `k = 4`.

**This explains increment 6.** The bias is negative, so the estimated signal fraction is too low, so
the estimated ceiling sits too low, so it is violated more often than it should be. On real kNN graphs
at `k = 10` edges span many distances at once, which is the synthetic `k = 10` row: bias -0.222. Two
held-out violations of 30 is what that looks like.

## Increment 10: the contrast belongs to the field, and prediction 1 confirms the arithmetic

Four designs had tried to move `lambda = rho_nn / r` by changing the graph and four gates refused them.
The arithmetic says they had to: with an exponential field `lambda = c(d) = r / (1-nu)` is pinned to `r`
by the field, and no graph can separate them. A squared-exponential field gives `c(2d) = c(d)^4` and so
`lambda = c(d)^3`, which separates them at matched `r`.

The same arithmetic makes a quantitative prediction: on a squared-exponential field at `k = 2` the
estimator must overstate the signal fraction by exactly `1 / c(d)^2`. Measured, with the conditioning
guard declared in the plan:

| field family | datasets | median ratio to the prediction | IQR |
|---|---|---|---|
| exponential | 61 | **0.9973** | [0.9754, 1.0204] |
| squared-exponential | 76 | **0.9993** | [0.9772, 1.0175] |

So the reasoning that produced the plan is right to within **0.07 %**. Had the geometric model been
assumed instead of corrected, the median ratio would be 1.0363.

These are the regenerated values. The first version of this table read 59 and 75 datasets with medians
0.9947 and 0.9970, "right to within 0.3 %", from a run whose seeds could not be reproduced; see the
foot of this document. The prediction holds either way, and holds more tightly now.

## And the fifth refusal, which is the finding

The design gate asked whether the field family moves `lambda`. It accounts for **0.3 %** of its
variance; the correlation length accounts for 21.4 %, degree for 5.6 % and the noise target for 1.1 %.
The conditioning guard kept 392 of 600.

| design | share of lambda's (or the decoupling's) variance the construction moved |
|---|---|
| increment 5, ten fingerprint/metric/k constructions | 4.2 % |
| increment 7, spread neighbours at fixed k | 0.0 % |
| increment 8, same data, contrast corrected to lambda | 14.1 % |
| increment 9, synthetic circulants | 10.8 % |
| increment 10, two field families | **0.3 %** |

Gate failed, outcome not reported, and the plan forbids a sixth re-specification.

**The five refusals have one cause, and it is structural rather than a run of bad designs.** The
conditioning guard keeps only datasets where `rho_nn` is well determined, which means `rho_nn` is not
small, which means `lambda` is near one. A low per-hop decay IS the ill-conditioned regime: small
`rho_nn` is simultaneously what makes `lambda` low and what makes `r^2 / rho_nn` unusable. The two
cannot be had together at a fixed sample size, because `rho_nn`'s standard error falls as
`1 / sqrt(n * pairs per node)` and buying pairs means raising the degree, which brings mode two's bias
back.

So the question the talk had been asserting an answer to is not merely untested, it is **not testable
in that form**: the regime where the second statistic should matter most is the regime where the
estimator built on it does not work. That is a sharper statement than five null results, and it is what
the synthetic setting was for.

## What a usable version of the estimator would need

Named, not attempted, and each follows from a measured failure above rather than from taste:

- a conditioning guard as part of the estimator rather than as part of the analysis, refusing to
  report a signal fraction when `rho_nn` is within a stated number of standard errors of zero;
- a decay model that uses the actual distribution of edge hop lengths instead of a single `lambda`,
  since mode two's bias is entirely explained by edges spanning several distances at once;
- and, for the low-decay regime, far more pairs per node than a 600-node graph at degree 2 can supply.


## A correction: increment 10's numbers come from a regenerated run

Found on 2026-10-05 while writing increment 12. `scripts/field_profile.py` seeded each synthetic field
with

    np.random.default_rng((hash(family) % 9973, k, d, length, int(nu * 100), seed))

and Python randomises `hash` on a `str` per process unless `PYTHONHASHSEED` is pinned. So the 600 rows
this increment originally wrote were a valid realisation of the frozen design — within one process the
mapping was constant, and every cell of the grid was generated exactly as the plan specified — but a
realisation that could never be drawn again. In a project whose whole method is a frozen plan and a
reproducible measurement, a number nobody can recompute cannot be checked against its plan.

The seed is now `FAMILIES.index(family)`, an explicit index into the declared grid, lifted into a named
`cell_seed` function so a test can call it in two processes and compare. `tests/test_seeding_is_reproducible.py`
pins both the property and the class: no generator may call `hash()` at all.

**What changed.** The old seeds are unrecoverable, so the data had to be regenerated and every
increment-10 figure moved. Nothing qualitative moved with them.

| figure | before | after |
|---|---|---|
| conditioning guard kept | 391 of 600 | 392 of 600 |
| prediction 1, median ratio, squared-exponential | 0.9970 | 0.9993 |
| prediction 1, accuracy of the plan's arithmetic | 0.3 % | **0.07 %** |
| lambda variance from the field family | 0.9 % | 0.3 % |
| lambda variance from the correlation length | 28.0 % | 21.4 % |
| field_profile's own max identity error | 3.21e-15 | 2.97e-15 |

Prediction 1 still holds and holds more tightly. The design gate still fails, and by a wider margin:
0.3 % against the 25 % the plan demanded. The fifth refusal stands.

**Increment 9 is unaffected and this was verified rather than assumed.** `scripts/synthetic_decay.py`
seeded on `(k, d, length, int(nu * 100), seed)`, all numeric, so it never had the defect.
Regenerating it reproduced `results/synthetic_decay.csv` **bit-identically** — same md5,
`55cb03606f48f4edee6adea33023f4d9` — and its report identically apart from one elapsed-time line. So
every figure in this document that comes from increment 9 is reproducible and unchanged: the 0.4150
mean absolute recovery error at `k = 2`, the single 40.85 outlier, the 0.180 ill-conditioned share, and
the mixed-hop bias series 0.0237 / 0.1136 / 0.2224 / 0.338. That file's `cell_seed` refactor is proved
behaviour-preserving by the identical md5.

No pre-registration was touched. `prereg/increment9_synthetic.yaml` and
`prereg/increment10_field_profile.yaml` remain frozen at their committed blob hashes, and nothing in
either plan's grid, guard, threshold or decision rule changed — only the arithmetic that turns the
declared grid into a seed.

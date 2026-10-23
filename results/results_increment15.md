# Increment 15: the size dependence of section 4, measured under a frozen plan

Pre-registration: `prereg/increment15_size_dependence.yaml`, blob
`3d941919c5aeb34e2749cc820e8e3ae55d5ff725`, frozen before any number under it existed.
Script `scripts/size_dependence.py`, committed at `dbae4b3` before it had produced a value.
Artefacts: `results/size_dependence.csv` (1120 rows) and `results/report_size_dependence.txt`.

**Verdict: section 4's claim survives, and section 4 is REWRITTEN.** The index moves with n, on 40 of
40 targets, with an interval far from zero. The movement is entirely rank dilution, which is the
classical k/n channel, so the paper claims the consequence for reporting practice and not the
mechanism.

## Both instrument gates read exactly zero

| gate | what it checks | worst over 40 targets |
|---|---|---|
| 1 | the pipeline against committed `modi_baseline.csv` at f = 1, four columns | **0.000e+00** each |
| 1b | the subsample route against a route sharing no indexing code, f = 0.5 | **0.000e+00** each |

Gate 1 reads zero because the plan specifies `S_f = numpy.sort(perm[:n_f])`. The draft of the plan
said `perm[:n_f]`, which at f = 1 is a full permutation, and because both neighbour selectors break
similarity ties on the ascending index — 26.8 % of rows on the largest target and 35.5 % on the
smallest have their 10th and 11th similarity exactly equal — that moved every gated column by up to
1.13e-2, ten orders above tolerance. The error was caught before freezing and is recorded in the
plan's header as specification error (3).

Gate 1b exists because gate 1 cannot reach the subsample route: with `S_1 = arange(n)` the f = 1
submatrix *is* the full matrix. It recomputes all four statistics from `ecfp4(smiles[S])` forward, a
path sharing no indexing code with the submatrix route, and so would catch a label-against-similarity
misalignment that a `np.ix_` identity check cannot.

## The table, re-measured

| fraction | mean n | assortativity | floor rel. MSE | between-chain sd |
|---|---|---|---|---|
| 1/8 | 344.7 | 0.2957 | 0.6915 | 0.0456 |
| 1/4 | 689.8 | 0.3855 | 0.5919 | 0.0355 |
| 1/2 | 1379.9 | 0.4752 | 0.4992 | 0.0186 |
| all | 2760.4 | **0.5556** | **0.4217** | — |

**The exploratory probe was sound.** Its four rows, which had no committed artefact anywhere in the
repository, reproduce to within 0.0055 at worst and 0.0003 at best. That is luckier than it looks: the
probe reported a single draw, and the between-chain standard deviation at f = 1/8 is 0.0456, with
0.1016 on the worst target. Five chains were not a refinement; a single draw could have landed
anywhere in a band twenty times wider than the discrepancy.

## The primary and secondary statistics

| | mean | interval | sign count |
|---|---|---|---|
| floor_rel_mse(1/8) − floor_rel_mse(1) | **+0.2698** | [+0.2477, +0.2922] | 40 / 40 |
| assortativity(1) − assortativity(1/8) | **+0.2599** | [+0.2385, +0.2807] | 40 / 40 |

Both intervals exclude zero in the declared direction and both sign counts are 40 of 40 against a
registered requirement of 34. Section 4's claim is not withdrawn.

## Gate 2: the movement is rank dilution, and nothing else

| fraction | D_A | D_A′ | D_B | D_B / D_A′ | normaliser channel |
|---|---|---|---|---|---|
| 1/8 | 0.2698 | 0.2623 | 0.2639 | **1.0060** [0.9731, 1.0536] | +0.0075 |
| 1/4 | 0.1702 | 0.1652 | 0.1697 | **1.0271** [0.9873, 1.0710] | +0.0050 |
| 1/2 | 0.0776 | 0.0770 | 0.0758 | **0.9848** [0.9236, 1.0401] | +0.0006 |

Arm B holds the point set, every pairwise similarity, the label distribution, the variance normaliser
and the scored set at full n, and changes only which ranks of its own full ordering each molecule
reads. It reproduces the measured movement at every fraction, inside the two-sided band [0.90, 1.10]
declared in advance.

**This was close to forced and the plan said so.** A pilot on committed data already put the ratio at
0.978, and gate 2 is declared in the plan as an instrument check on a closed-form prediction rather
than a test of an open question, precisely so that this pass could not be written up as a discovery.
What was not forced is the per-fraction consistency and the magnitude with an interval over 40
targets.

**No novelty is claimed for the mechanism.** That n enters a kNN estimator only through k/n, by way of
neighbour-rank order statistics, is published: Samworth, *Annals of Statistics* 2012, 40(5):2733,
Theorem 1, whose squared-bias factor for the unweighted rule is exactly (k/n)^{2/d}; and the
near-neighbour distance moments E[eps_k^a] <= C (k/n)^{a/D} citing Evans, Jones and Schmidt,
*Proc. R. Soc. A* 2002, 458:2839. Both were added to the plan after an adversarial prior-art pass on
its draft, which is also where the Gamma test's **M-test** was found — the same operation as this
increment, in a line the paper already cites, with the opposite standard reading (convergence towards
a truth rather than a property that is not the data's).

## Arm A′ was load-bearing, and this is the finding that would otherwise have been misreported

The exact decomposition `D_A − D_B = (D_A − D_A′) + (D_A′ − D_B)` holds to 8.67e-19:

| fraction | D_A − D_B | = normaliser channel | + rank-dilution residual | ratio |
|---|---|---|---|---|
| 1/8 | +0.00594 | **+0.00753** | **−0.00159** | 4.7x |
| 1/4 | +0.00049 | +0.00497 | −0.00448 | 1.1x |
| 1/2 | +0.00176 | +0.00059 | +0.00117 | 0.5x |

At f = 1/8 the variance-normaliser channel is **4.7 times the rank-dilution residual and points the
opposite way**. Arm A divides by the subsample's own mean and variance; arms A′ and B divide by the
full set's. Without arm A′ the reported residual would have been +0.00594, read as "sample dependence
beyond rank dilution", and all of it would have been a normaliser artefact pointing in the direction
that flatters section 4. The true residual at that fraction is negative.

Arm A′ costs no new measurement, seed or thinning: it is arm A's numerator over arm B's denominator,
`(SSE_sub / n_f) / (SST_full / n)`. It was added to the plan after the draft proposed subtracting an
asymptotic `1/n_f` correction instead, which recovers only about 37 % of the channel and would have
labelled the problem solved.

## Registered predictions: all four hold

1. **HOLDS.** r rises on 40/40 and the floor falls on 40/40, against a registered 34 of 40.
2. **HOLDS.** Median per-target ratio r(1)/r(1/8) = **1.988** against the exploratory 1.986 and a
   registered band of [1.70, 2.30].
3. **HOLDS.** Monotone at all eight declared thresholds and strictly increasing at all eight.
4. **HOLDS**, and this one was genuinely unknown.

### Prediction 3: the crossing counts, over a grid and not one cut point

| threshold on modi_q2 | 1/8 | 1/4 | 1/2 | all |
|---|---|---|---|---|
| 0.40 | 8 | 17 | 33 | 38 |
| 0.45 | 6 | 15 | 29 | 35 |
| 0.50 | 3 | 9 | 20 | 32 |
| 0.55 | 3 | 4 | 12 | 29 |
| 0.60 | 2 | 3 | 9 | 15 |
| 0.65 | 2 | 2 | 5 | 10 |
| 0.70 | 2 | 2 | 3 | 6 |
| 0.75 | 2 | 2 | 2 | 3 |

The 0.65 row reproduces the exploratory "2 of 40" and "10 of 40" exactly. **It is reported inside a
grid because 0.65 is not this index's threshold.** Golbraikh, Muratov, Fourches and Tropsha, *JCIM*
2014, 54(1):1-4, calibrate 0.65 on the activity-class-weighted binary MODI, and this project's own
`scripts/modi_baseline.py:185` applies it to `modi_binary`. The ChEMBL collection has no activity
classes here, so the published threshold cannot be evaluated on the primary collection at all, and
transposing it onto q2 is this project's step. The claim is monotone at every threshold, so it does
not rest on the transposition.

### Prediction 4: the closed form degrades at small n, and now the bound is measured

| fraction | mean abs. difference, identity against measured floor |
|---|---|
| all | 0.0103 |
| 1/2 | 0.0133 |
| 1/4 | 0.0168 |
| 1/8 | **0.0221** |

Section 2's closed form is exact only on k-regular graphs, and union symmetrisation is further from
regular when k is a larger fraction of n. Its error grows monotonically as n falls, by a factor of
2.1 from full n to n/8. This bounds where the identity may be used and had never been measured below
f = 1. It is a result about the paper's own instrument, not about section 4's claim.

## The published index, on the collection where its threshold is the source's own

| fraction | mean binary MODI | cells above 0.65 | min class kept | cells with < 10 in a class |
|---|---|---|---|---|
| 1/8 | 0.6514 | **81 / 150** | 4 | 8 / 150 |
| 1/4 | 0.6882 | 123 / 150 | 13 | 0 |
| 1/2 | 0.7291 | 147 / 150 | 24 | 0 |
| all | **0.7592** | **30 / 30** | 60 | 0 |

This is the strongest form of the claim and it is on the right index. The 2014 binary MODI, with the
2014 threshold, declares every one of the 30 MoleculeACE targets modelable at full n and only 54 % of
(target, chain) cells at n/8. Same targets, same chemistry.

**Against this row:** 8 of the 150 cells at f = 1/8 rest on fewer than ten molecules in one class, the
thinnest on four. The empty-class guard written into this increment's script never fired — no class
emptied on any of the 480 cells — but a MODI value resting on four molecules is quantised in steps of
0.125 in that class's term, and the row should be read with that in mind. Both per-class surviving
counts are carried beside every cell in `size_dependence.csv` for exactly this reason. The guard lives
in the new script because `scripts/modi_baseline.py` is increment 11's artefact behind a frozen blob
and was not edited.

## The one target the plan flagged in advance

`CHEMBL614358_EC50` has r(1/8) = 0.0168 in the chain mean, below the 0.05 the plan declared as a flag,
with per-chain values of −0.0165, −0.0121, +0.0041, +0.0287 and +0.0800. Two of five chains are
**negative**. Its ratio is 6.72, the maximum over the 40 targets against a median of 1.988. It is
flagged and reported, not excluded, as the plan requires; the median moves from 1.988 to 1.979 without
it, so prediction 2 does not turn on it. This vindicates the plan's stability note: a ratio of
correlation coefficients is unstable near zero, and on this data it blows up on exactly one target.

## What this does to the paper

Per the plan's outcome rule, the intervals hold and gate 2 passes, so **section 4 is rewritten**. It
leads with the decomposition and the consequence for reporting practice, with the classical k/n
citation attached to the mechanism, rather than with the four-row table as though the movement were a
discovery about chemistry. The honest statement is:

> Every index in this line is published as a bare point estimate per dataset. The quantity those
> indices estimate moves by 0.27 in relative MSE and 0.26 in assortativity over an eightfold change in
> sample size, on 40 of 40 targets, and that movement is predictable in closed form from the full-n
> neighbour ordering alone. The 2014 binary index crosses its own published threshold on 46 % of cells
> by growing alone. Reporting such an index without its n is reporting a point on a curve whose
> position was never in doubt.

## What this increment did not do

No scaling exponent, power law or extrapolation was fitted. That ban is in the plan with an evidential
reason rather than a stylistic one: a check on the exploratory values showed the comparison cannot
discriminate. Baselining on f = 1 leaves three points and a two-parameter family, so one degree of
freedom; pinned on f = 1/8 and f = 1/2, `D(f) = C(f^-p - 1)` predicts D(1/4) = 0.17005 against a
measured 0.17000, an apparent confirmation to 5e-5 implying d = 19.5 — while a deliberately unrelated
quadratic in log(1/f), pinned on the same two points, predicts 0.17067 and is equally "confirmed" at
any precision this design reports. The implied exponent swings from 9.5 to 94.8 under perturbations
the measured between-chain spread of 0.0456 makes ordinary. A pre-registered exponent comparison would
have manufactured a confirmation.

No mechanism is offered for the residual arm B leaves. Nothing is claimed about a realistic
medicinal-chemistry acquisition order; the subsampling here is uniform, and the limit stands as
section 7 states it. Neither the M-test's convergence reading nor section 4's reading is asserted to
be settled by this measurement.

## Method note

The draft of this plan went through an adversarial pre-freeze pass, because two of the last three
gates in this project were broken at freeze time: increment 13's was already failed by the project's
own committed data, and increment 14's was unsatisfiable against rows stored at three decimals.
Twenty-three findings were raised and sixteen survived verification. Six changed the plan, and three
of those would not have been found from the inside: the unsorted index set, the wrongly-costed
Bernoulli approximation in arm B, and the tautology in gate 2 whose one-sided threshold would have
committed the project to publishing a refutation of its own premise as a confirmation of it. One
proposed fix was rejected on evidence, and the reason is now in the plan.

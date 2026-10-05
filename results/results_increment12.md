# What exchangeability deleted: claim 6 downgraded by the project's own rule

Pre-registration: `prereg/increment12_local_weights.yaml`, blob
`32dddd1a8aa1be4b6cad0e24b3e0778fdac436ef`, frozen before any number below existed. Reproduce with
`uv run python scripts/local_weights.py`; output in `results/report_local_weights.txt` and
`results/local_weights.csv`.

Claim 6 of `docs/paper/skeleton.md` said equal weights are already near-optimal locally, on the
evidence of a `+0.0013` skill gap "under exchangeability". The prior-art pass called it the most novel
thing in the paper. It had no pre-registration, no committed script and no results document anywhere in
this repository; the number existed only in the claim table. It is now measured, and the frozen rule
downgrades it.

## The theorem, reported unconditionally

Standardise the label and write `c_i = E[z_v z_{u_i}]`, `C_ij = E[z_{u_i} z_{u_j}]`. The best linear
predictor is `w* = C^-1 c` with `MSE* = 1 - c' C^-1 c`. Under exchangeability `c = r 1` and
`C = (1-rho) I + rho 11'`, so `1` is an eigenvector of `C` with eigenvalue `S = 1 + (k-1) rho` and `c`
is parallel to it. Therefore, with `m_bar` the plain neighbourhood mean and `beta = k r / S`:

    w*                    = (r / S) 1            every optimal weight is EXACTLY equal
    sum_i w*_i            = beta                  the weight sum IS the regression slope on m_bar
    MSE_mean - MSE*       = Var(m_bar) (1-beta)^2 the whole gap is a squared scale error
    MSE(best alpha m_bar) = MSE*                  exactly, for all k, r, rho

The last line is the decisive one: the k-dimensional optimal weighting has **precisely zero** advantage
over fitting one scalar on the plain mean. So claim 6's `+0.0013` measured how badly calibrated the
neighbourhood mean is, and said nothing about whether weighting nearer neighbours more would help —
exchangeability forbids that question by assuming every neighbour correlates with the target at the
same `r`. Checked three ways before the plan was written: two independent derivations
(Sherman-Morrison, and permutation symmetry with a one-dimensional reduction) and a numerical sweep
over 2000 random `(k, r, rho)` agreeing to 1.7e-15.

A corollary that is also an implementation trap: imposing `sum w = 1`, as a softmax or any
normalisation does, makes the constrained optimum exactly `1/k`. A normalised-weights arm reports a
null by construction. This experiment never normalises.

## The implementation computes the floor it claims

The mean arm against the existing `knn_floor.predict`: **max absolute drift 3.553e-15** over 128
target-`m` cells. Reported first because a non-zero drift would void everything after it.

## Gate 1: exchangeability is materially false, and by a lot

The premise. Second moment against neighbour rank, averaged over targets, at the primary `m = 10`:

    rank  1     2     3     4     5     6     7     8     9    10
    c_j  0.714 0.660 0.629 0.597 0.572 0.542 0.523 0.513 0.499 0.476

| collection, m = 10 | mean decline `c_1 - c_m` |
|---|---|
| ChEMBL-40 (31 targets) | **+0.2384 [+0.2165, +0.2603]** excludes zero |
| MoleculeACE-30 (14 targets) | **+0.2289 [+0.2022, +0.2561]** excludes zero |

The gate passes on every collection at every `m`. Exchangeability assumes this curve is flat; it falls
by a third of its value across ten ranks. This is the first measurement of a neighbour-rank
correlation profile in the project — `scripts/field_profile.py` is entirely synthetic and its `c_d`
column is an analytic input rather than a measurement, so the project's whole prior "profile" was two
lags fitting one parameter.

## The verdict: scale gains nothing, shape gains a lot

Out of sample, on held-out compounds, with everything estimated on the training split alone.

| collection, m = 10 | `gain_scale` | `gain_shape` |
|---|---|---|
| ChEMBL-40, 31 targets | +0.0000 [-0.0003, +0.0004] **covers zero** | **+0.0379 [+0.0314, +0.0442]** excludes zero |
| MoleculeACE-30, 14 targets | -0.0002 [-0.0006, +0.0000] **covers zero** | **+0.0497 [+0.0417, +0.0577]** excludes zero |

`gain_scale` is the entire content of claim 6 as it stands, and out of sample it is zero. `gain_shape`
is the term exchangeability deletes by theorem, and it is thirty to fifty times larger than the claim's
own reported effect.

Against the rank-shuffled surrogate, which destroys the rank assignment while keeping every neighbour
and every label: the measured shape gain beats the surrogate's 95th percentile on **30 of 31** ChEMBL
targets and **14 of 14** MoleculeACE targets. The surrogate's own mean shape gain is **-0.0023** —
fitting `m` weights on rank-scrambled data costs skill out of sample, as it should — and its mean 95th
percentile is +0.0022. So the measured +0.038 is about seventeen times the noise ceiling, not inside it.

Per target the shape gain runs **-0.0126 to +0.0784** on ChEMBL-40, negative on 1 of 31, and +0.0256 to
+0.0779 on MoleculeACE-30, negative on 0 of 14.

**By the frozen rule: claim 6 survives as stated = False. Claim 6 is downgraded = True.** On both
collections where the primary `m` is reachable.

## What the optimal weights actually look like

Normalised so that 1.0 is the weight the plain mean gives every neighbour. ChEMBL-40, `m = 10`,
31 targets, mean across targets with the between-target standard deviation:

    rank   1     2     3     4     5     6     7     8     9    10
    w*m  3.487 1.937 1.244 0.934 0.696 0.393 0.227 0.338 0.270 0.187
    sd   0.535 0.304 0.284 0.214 0.228 0.225 0.248 0.249 0.172 0.206

The nearest neighbour deserves **3.5 times** the weight the mean gives it and the tenth deserves a
fifth. Negative weights appear on 15 of 31 targets. The weight sum is 0.97, which is why the scale gain
is nothing: the mean is already almost perfectly calibrated in total, and badly wrong in shape.

## The claim's own number, reproduced, as the scale gain

The skeleton reported `+0.0013`. The measured median `gain_scale` on ChEMBL-40 is **+0.00132 at m = 5**
and **+0.00000 at m = 10**.

The provenance of the original figure cannot be checked, because no script produced it. What can be
said is that `+0.0013` is exactly the size of the scale gain, which is the only improvement
exchangeability permits, and that at the primary `m` that gain is zero to four decimals while the
shape gain is +0.042.

## The trend in m, and where the gates bit

| m | ChEMBL-40 retained | median beta | median `gain_scale` | median `gain_shape` |
|---|---|---|---|---|
| 5 | 40/40 | 0.946 | +0.00132 | +0.0127 |
| 10 | 31/40 | 0.994 | +0.00000 | +0.0421 |
| 20 | 4/40 | 1.164 | +0.00460 | +0.1114 |

The shape gain grows with `m`, which is what more rank structure should buy. Gate 3, the sample-size
guard `n_train >= 20 m(m+1)/2`, was declared in advance and bit hard: it retains 79 of 80 target cells
at `m = 5`, 45 of 80 at `m = 10`, and **4 of 80** at `m = 20`. So the `m = 20` row rests on four ChEMBL
targets and nothing else, and MoleculeACE and DeepDelta report no outcome at `m = 10` and `m = 20`
respectively. Gate 2, `cond(C) < 100`, excluded nothing anywhere: the rank-resolved second-moment
matrix is well conditioned on real targets, unlike the two-statistic estimator of increments 9 and 10.

## The two registered predictions

**Prediction 2 holds.** `beta` sits near one on the primary collection at the primary `m`: median
0.9940, median `|beta - 1|` **0.0149**, against a threshold of 0.10. That is the registered explanation
for why the exchangeable gap was `+0.0013` rather than ten times that, and it is confirmed. It fails
on two non-primary cells and this is reported rather than filed away: median `|beta - 1|` is 0.1638 at
`m = 20` (4 targets) and 0.1290 on DeepDelta-10 at `m = 5`.

**Prediction 3 fails.** The registered claim was that where `beta` is near one the floor's relative
error approaches `1 - r`, because `beta = 1` makes the second moment cancel. Measured:
Spearman(`|beta-1|`, `|rel_mse - (1-r)|`) = **-0.0698 [-0.4643, +0.3286]**, covering zero and with the
wrong sign. Not supported.

Why, offered as a post-hoc reading and labelled as one: `|beta - 1|` has almost no spread to correlate
against (median 0.0149), and `rel_mse - (1-r)` on these graphs is dominated by the union-symmetrised
kNN graph's irregularity, where the identity is an approximation with 0.0256 mean absolute error,
rather than by `beta`. The prediction was testable and it did not survive; that is recorded here and
the one-statistic-floor idea is not carried into the paper.

## What this does to the paper

Claim 6 was the paper's most novel claim and it is now a measured quantity with the opposite sign from
the one the skeleton asserted. The replacement is stronger, not weaker:

> On molecular similarity neighbourhoods the plain mean is almost perfectly calibrated in total weight
> and badly wrong in shape. Rescaling it buys nothing out of sample. Reweighting by neighbour rank buys
> +0.038 to +0.050 skill, beats a rank-shuffled null on 44 of 45 targets, and the optimal profile gives
> the nearest neighbour three and a half times the weight the mean gives it.

And the vacuity theorem travels with it, because it explains why nobody had measured this: the
exchangeable model, which is the natural one to write down, deletes the entire effect by assumption and
leaves a calibration residue that looks like a null result.

The consequence for claim 4 has to be re-examined and is NOT settled here. Claim 4 reads the floor
beating fitted models on the hardest third as evidence about information outside the neighbourhood.
If a better-shaped floor is worth +0.04 skill, part of that gap was weighting after all, and the
re-examination needs its own frozen plan.

# The identity as a function of k, and a prediction the talk made without a test

Pre-registration: `prereg/increment5_identity.yaml`, blob
`c2dcd4ef750f579a89bdfeb5e043b381307abdfe`, frozen and committed before either experiment ran.
Reproduce with `uv run python scripts/identity_k_sweep.py` and
`uv run python scripts/identity_decoupling.py`; output in `results/report_identity_k_sweep.txt`,
`results/report_identity_decoupling.txt`, `results/identity_k_sweep.csv` and
`results/identity_decoupling.csv`.

Increment 4 checked the identity once, at k = 10, on one construction, and got 0.0256 mean absolute
error. One point is not a functional form and one construction is not a range of graphs.

## The identity is exact on a regular graph, and that is now a test rather than a claim

Before either experiment, the derivation itself was pinned. On a k-REGULAR graph the identity

    mean_v (y_v - mean_{u in N(v)} y_u)^2 / sigma^2  =  1 + 1/k + (k-1)/k * rho_nn - 2 r

is exact to machine precision, because when every degree is k both multisets the correlations are
taken over have the label's own mean and variance: an edge endpoint is uniform, and a shared-neighbour
endpoint is uniform because every node is counted k(k-1) times.
`tests/test_neighbourhood.py::test_the_identity_is_exact_on_a_regular_graph` asserts that at
k = 3, 4, 6, 8 with `abs=1e-12`. A broken derivation now fails the suite instead of surviving inside a
results document, and the two statistics live in `src/molace/measures/neighbourhood.py` rather than in
a script.

## Experiment A: the identity holds across a thirteenfold range of k

| k | mean abs error | mean signed error | max abs error | Pearson | Spearman | mean degree | rho_nn | r |
|---|---|---|---|---|---|---|---|---|
| 3 | 0.0269 | -0.0074 | 0.1166 | 0.983 | 0.956 | 4.2 | 0.576 | 0.646 |
| 5 | 0.0252 | -0.0114 | 0.0983 | 0.983 | 0.925 | 6.8 | 0.545 | 0.615 |
| 10 | 0.0256 | -0.0127 | 0.0837 | 0.983 | 0.949 | 13.5 | 0.487 | 0.556 |
| 20 | 0.0286 | -0.0183 | 0.0890 | 0.981 | 0.942 | 26.7 | 0.409 | 0.481 |
| 40 | 0.0306 | -0.0241 | 0.1161 | 0.978 | 0.964 | 53.7 | 0.318 | 0.392 |

The frozen threshold was mean absolute error below 0.05 at every k. The worst is 0.0306 at k = 40, so
the functional form holds. It is worth saying what else moved while it held: r runs from 0.646 down to
0.392 and rho_nn from 0.576 to 0.318, so the identity is not accurate only in the regime it was first
checked in.

The mean degree is also worth a line. Union symmetrisation gives degree at least k, so at k = 40 the
graph's mean degree is 53.7. The prediction uses the floor's own averaging width, 40, not the graph's
degree, and that is the right choice on these numbers as well as in the derivation; increment 4
measured substituting the mean degree and it was worse.

### The residual, and a deviation in how it was tested

The plan said, before the run, that the identity should UNDER-predict the error, because the measured
floor averages the k nearest TRAINING molecules of a test molecule while both statistics are computed
over the whole graph, and that the mismatch should GROW with k.

The mean signed error is negative at every k and its modulus is strictly increasing:
0.0074, 0.0114, 0.0127, 0.0183, 0.0241. The frozen statistic is Spearman(k, |mean signed error|),
which is +1.000, with a cluster bootstrap over targets of [+0.300, +1.000], excluding zero. Both
conditions hold, so the explanation of the residual holds too.

**The deviation.** `scripts/identity_k_sweep.py` as first run computed
Spearman(k, |per-cell error|) = +0.083 [-0.064, +0.228] instead, which covers zero, and printed a
verdict saying the explanation failed. That is a different statistic: it pools 200 cells and is
dominated by differences between targets rather than by k. The frozen text reads
"Spearman(k, |mean signed error|)", the modulus of the mean, and is in the committed blob above. So
the plan was not changed after seeing a result; the script disagreed with the plan and the script was
corrected. Both numbers are printed by the corrected script, and the secondary one is labelled as not
being the frozen statistic.

## Experiment B: the frozen rule says the prediction holds, and I am not banking that

400 cells, 40 targets by the 10 frozen constructions. Decoupling `d = |r - rho_nn|` terciles split at
0.0597 and 0.0794.

| tercile | cells | mean decoupling | identity mean abs error | assortativity alone, Spearman | full identity, Spearman |
|---|---|---|---|---|---|
| low | 134 | 0.0435 | 0.0265 | 0.979 | 0.980 |
| mid | 132 | 0.0694 | 0.0294 | 0.924 | 0.940 |
| high | 134 | 0.0913 | 0.0303 | 0.917 | 0.931 |

High-minus-low difference in ranking power, paired bootstrap over targets, 10000 draws:

| | difference | 95% CI | |
|---|---|---|---|
| assortativity alone | -0.062 | [-0.119, -0.009] | excludes 0 |
| the full identity | -0.049 | [-0.106, +0.010] | covers 0 |

Against the frozen rule both conditions are met: the identity's mean absolute error in the top
tercile exceeds the bottom by 0.0038, well under the 0.02 allowed, and assortativity alone ranks worse
in the top tercile with an interval excluding zero. The script prints THE PREDICTION HOLDS.

**The project does not accept that verdict.** Two reasons, each sufficient on its own.

### The design cannot test what the prediction is about

The prediction is about graph structure: assortativity alone should fail where neighbour agreement
decouples from edge agreement. Decomposing the decoupling's variance over the 400 cells:

| source | share of variance |
|---|---|
| between targets | **77.0 %** |
| between constructions | **4.2 %** |

Across the ten constructions the mean decoupling runs from 0.0578 (MACCS) to 0.0741 (ECFP4 at k = 30),
a spread of 0.016 against a cell-level standard deviation of 0.0218. The frozen grid barely moves the
quantity the terciles stratify on, so the terciles sort TARGETS. Whatever appears in the high tercile
is as well explained by "harder targets rank worse" as by anything structural. Reusing a frozen grid
rather than designing one was named as a limit in the plan; the measurement shows the limit is fatal
rather than cosmetic.

### The frozen rule tested the wrong contrast

The prediction implies that assortativity alone degrades MORE than the full identity does. The rule
tested each degradation separately and read "one interval excludes zero, the other covers it" as the
answer. That is the inference increment 2 got wrong and corrected: an interval covering zero is not
evidence of no effect.

The contrast the prediction actually implies, post hoc because the plan never asked for it:

| | difference of the two degradations | 95% CI | |
|---|---|---|---|
| (assortativity alone) minus (full identity) | **-0.0132** | [-0.0501, +0.0268] | **covers 0** |

Both degrade, by -0.062 and -0.049, and the data do not separate them.

### Verdict

**The prediction remains untested.** The talk's claim that assortativity alone must fail where the two
statistics decouple is neither confirmed nor refuted here; the experiment that was meant to settle it
could not, and the talk is corrected to say exactly that.

What a real test needs, for a later increment with its own frozen plan: constructions that move
clustering on purpose rather than incidentally. Threshold graphs are the obvious candidate, since the
project already measured their density spanning 98x across targets at a fixed Tanimoto cut, and
explicit rewiring at fixed degree is another. Both change neighbourhood overlap directly, which is
what the decoupling is made of.

## What the pairwise arm is, when its flow is a gradient

Not in the pre-registration: a derivation and its test, no measurement.

The project already has two pieces. Algebra: with uniform weights over the same anchors,
`yhat_pair = floor + mean_a f(x_a, x_new)`. Measurement: the trained flow is 96.6 % gradient, curl
2.6 %, harmonic 0.8 %, and shuffling the labels leaves curl where it was.

Together they say something sharper. A gradient flow is `f(x_a, x_b) = phi_b - phi_a` for a node
potential phi, so the mean learned correction is not free:

    yhat_pair  =  phi_new  +  (mean_a y_a  -  mean_a phi_a)

a pointwise predictor phi plus the floor residual of phi. That is the precise version of "the model
collapses into a pointwise one plus anchor averaging", which is how the talk put it, and it explains
without further measurement why the pairwise arm tracks the floor as closely as it does: the floor
residual is literally one of its two terms. `src/molace/models/gradient_collapse.py`, with
`tests/test_gradient_collapse.py` pinning it at `abs=1e-13`.

**A caveat the tests forced into the open.** What survives the read-out is the MEAN of the
non-gradient part over the anchors, not its energy. A flow can be far from gradient and predict
exactly as a gradient one does, if its non-gradient part averages to zero over the anchors; the test
`test_a_non_gradient_flow_with_zero_mean_leaves_the_prediction_untouched` is that case. So "96.6 %
gradient" bounds the energy and not the prediction gap, and the two must not be conflated. This cuts
both ways: the collapse may be more complete than the energy figure suggests, and the energy figure is
not evidence for the collapse being incomplete either. `non_gradient_residual` is the quantity that
actually moves a prediction, and it has not been measured on the trained model; doing that needs the
flow on test-to-anchor edges, which lie outside the training graph the Hodge census was computed on.

---

# Increments 7 and 8: two more designs, two failed gates, and the reason

Pre-registrations: `prereg/increment7_decoupling.yaml`, blob
`0018a8760d7881322b23dc6dfe6b8465917ab225`; `prereg/increment8_lambda_contrast.yaml`, blob
`8bf30344d87e0a9e54cd77d794ca77848baeff96`. Output in
`results/report_decoupling_designed.txt`, `results/report_lambda_contrast.txt` and
`results/decoupling_designed.csv`.

After increment 5's experiment B was refused, increment 7 built a construction designed to move
neighbourhood overlap on purpose at fixed k, and put a **design gate before the outcome**: if the
construction does not move the quantity the contrast is built on, the outcome is not read at all. That
rule exists because increment 5 nearly banked a result its design had not earned.

## Increment 7: the construction worked, the contrast variable did not

| construction | clustering | per-hop decay lambda | decoupling abs(r - rho_nn) | mean degree |
|---|---|---|---|---|
| nearest (k most similar) | 0.557 | 0.864 | 0.0690 | 13.5 |
| spread (least overlapping k of the 4k nearest) | **0.223** | 0.797 | 0.0695 | 17.9 |

Clustering fell by a factor of 2.5 and the per-hop decay by 0.067, which cleared that half of the
gate. The decoupling moved by **0.0005**: 0.0 % of its variance between constructions against 89.3 %
between targets. Gate failed, outcome not read, and the script returns before computing it.

The reason is arithmetic and it indicts both earlier plans: `abs(r - rho_nn) = r (1 - lambda)`, and the
spread construction lowers lambda **and** lowers r, so the product barely moves. The contrast variable
was wrong in increment 5 and wrong again in increment 7. What carries the claim is lambda itself, the
decay model's own parameter.

## Increment 8: the contrast corrected, and the same wall

Increment 8 re-specified the contrast as lambda, disclosed as written on data already collected: the
cells existed, their gate statistics were known, and the lambda-based outcome had never been computed,
printed or looked at. That is pre-registration with respect to the outcome and not to the data, which
is weaker than increments 3 to 7 and is labelled so wherever it appears.

Its additional gate asked whether the construction separates lambda more than the targets do:

| source | share of lambda's variance |
|---|---|
| between constructions | **14.1 %** (needed 25 %) |
| between targets | 77.0 % |

Gate failed. Outcome not read. The plan forbids re-specifying a third time, so the question is left
open rather than re-rolled.

## What three attempts actually established

The same number keeps appearing: **between-target variance is 77 % to 89 % of everything**, whatever is
varied. Increment 5's ten-construction grid, increment 7's deliberate 2.5-fold change in clustering and
increment 8's corrected contrast all hit it. A 2.5-fold change in clustering bought 14.1 % of lambda's
variance.

So the honest conclusion is about feasibility, not about the claim: **a structural claim of this kind
cannot be tested on 40 real targets with a handful of constructions**, because target identity swamps
construction. What it needs is synthetic graphs with a controlled decay and a known noise fraction,
where the contrast is set rather than hoped for, exactly as the identity's own exactness was pinned on
regular graphs rather than argued from chemistry. That is the design for a later increment and it is
not attempted here.

## What the attempts did deliver, unconditionally

Both plans required the identity's accuracy to be reported whatever the gate did, because an identity
that stopped describing the estimator under a different construction would be a larger finding than the
contrast it was meant to test:

| construction | mean abs error | mean signed error | Pearson |
|---|---|---|---|
| nearest | 0.0256 | -0.0127 | 0.983 |
| spread | **0.0247** | -0.0031 | 0.980 |

The identity holds under a construction with **2.5 times less clustering**, 17.9 mean degree against
13.5, and a quite different neighbour-selection rule. Its residual also nearly vanishes there, -0.0031
against -0.0127, which is consistent with the train-only-neighbour explanation of increment 5: spread
anchors reach further down the candidate pool, so the whole-graph statistics describe them better. That
is a direction, not a tested claim.

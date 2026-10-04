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

## Experiment B: the prediction about decoupling

PENDING; this section is written when the run finishes.

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

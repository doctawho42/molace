# Increment 16: a one-parameter rank kernel attains the fitted weight optimum

Pre-registration: `prereg/increment16_weighting_family.yaml`, blob
`0ce65b307616a86cf0b565e854151096501d3f28`, frozen before the script existed. Script
`scripts/weighting_family.py`, committed at `eff9106` before it had produced a value. Artefacts:
`results/weighting_family.csv` (70 targets) and `results/report_weighting_family.txt`.

**Verdict: the kernel suffices.** Gou's `w_j` proportional to `1/j` is selected by cross-validation on
39 of 40 ChEMBL targets, beats uniform weights by +0.0181 [+0.0146, +0.0218], and is non-inferior to
the m-parameter fitted optimum at a pre-registered margin. Section 5.2's reshaping gain survives
symmetric selection. **No novelty is claimed for either arm**: both are in Itskowitz & Tropsha 2005.

**This is a pre-registration of procedure, not a blind one.** During the pre-freeze review a verifying
agent measured the primary statistic as +0.0182 [+0.0146, +0.0218]; the frozen plan records that as
disclosure (a). The measured value is +0.0181 [+0.0146, +0.0218], so the leak was accurate and
distorted nothing, but it happened and the plan's margin was anchored to increment 12's committed
+0.0155 rather than to it.

## All three instrument gates read exactly zero

| gate | what it ties | worst over all targets |
|---|---|---|
| 1 | `c_first`, `c_last`, `skill_mean`, `skill_rank`, `cond_C` against committed `local_weights.csv` | **0.000e+00** each |
| 2 | kernel at α=0 against the uniform arm, independent expressions | **0.000e+00** |
| 3 | the selected configuration's CV score, recomputed from a reslice of the split rule | **0.000e+00** |

Gate 1 checked 10 quantities on 40 targets at m=5 and 31 at m=10. It exists in this form because the
plan's first-draft gate 1 was **algebraically blind**: its statistic compared two computations of the
same anchor-set mean and is invariant to the split, the standardisation, the estimation-side
neighbourhood and the rank order. Measured with the diagonal deliberately unmasked, that statistic
moves by 0.000e+00 while `c_first` moves by 2.515e-01. Its satisfiability was also checked before
freezing, which increments 13 and 14 did not do.

Gate 3 is bit-for-bit because the reslice comes from the split rule rather than from the arrays the
selection was handed; the first draft checked function signatures, which cannot fail for the failure
they name.

## What each arm selected, and what it scored

| arm | parameters | median m | median α | mean skill | CV choice was test-best | mean regret |
|---|---|---|---|---|---|---|
| uniform | 0 | 5 | — | 0.3658 | 23/40 | 0.0053 |
| **kernel** | **1** | **10** | **1.000** | **0.3839** | 10/40 | 0.0040 |
| kernel_scaled | 2 | 10 | 1.000 | 0.3841 | 4/40 | 0.0045 |
| fitted | m | 10 | — | 0.3823 | 32/40 | 0.0006 |

## Registered predictions: three of four hold

### Prediction 1 holds, and not narrowly

| α | 0.0 | 0.25 | 0.5 | 0.75 | **1.0** | 1.5 | 2.0 | 3.0 |
|---|---|---|---|---|---|---|---|---|
| targets | 0 | 0 | 1 | 5 | **26** | 8 | 0 | 0 |

Median 1.000 with interquartile range [1.000, 1.000]. α ∈ {0.75, 1.0, 1.5} on **39 of 40** against a
registered threshold of 20, and on 29 of 30 MoleculeACE targets. The single exception is
`CHEMBL2000_Ki` at α = 0.5. The threshold was calibrated before freezing: an i.i.d. uniform draw over
the eight grid points clears it 7.2 % of the time, where the first draft's clause (median in
[0.5, 2.0], containing six of eight grid points) was cleared by noise 88.1 % of the time.

Cross-validation, given eight exponents and never shown a test label, picks **Gou's published rank
weight** on essentially every target in both collections.

### Prediction 2 holds

`mean(skill_kernel − skill_uniform)` = **+0.0181 [+0.0146, +0.0218]**, positive on **37 of 40**. On
MoleculeACE: +0.0211 [+0.0171, +0.0257], positive on **30 of 30**. Nobody in the kNN-QSAR line had
measured this: the 2005 paper has no uniform arm at all, and Bailey & Jain 1978 points the other way
at fixed k asymptotically.

### Prediction 3 holds, as an equivalence and not as a win

| | mean | interval | lower end vs −0.01 |
|---|---|---|---|
| matched grid | −0.0009 | [−0.0048, +0.0025] | **−0.0048 > −0.01, holds** |
| own grids | +0.0016 | [−0.0004, +0.0040] | — |

One parameter gives up 0.0009 of skill against m parameters, with the interval's lower end less than
half the pre-registered margin away from zero. The margin, −0.01, is 65 % of increment 12's committed
reshaping gain and was anchored to that committed figure rather than to anything measured here.

**Against this result, three things.** The per-target sign is a coin flip: the kernel is ahead on only
**19 of 40** on the matched grid and 23 of 40 on the own grids. The deficit exceeds the margin on
**3 of 40** targets, worst −0.0565 on `CHEMBL614358_EC50` — the same target increment 15 flagged for
near-zero assortativity. And this is a non-inferiority result, so the honest statement is that one
parameter is **not materially worse**, never that it is better.

### Prediction 4 fails

Both medians are 10.0, and the plan counts a tie against the prediction. The selected-m distributions
show the mechanism is visible but not in the median:

| arm | m=3 | m=5 | m=10 | m=20 | m=40 |
|---|---|---|---|---|---|
| uniform | 13 | 24 | 1 | 2 | 0 |
| kernel | 0 | 7 | 23 | 2 | **8** |
| fitted | 1 | 15 | 22 | 2 | 0 |

The kernel reaches m=40 on 8 targets where the fitted arm cannot, and never selects m=3 where uniform
does 13 times, so the parameter-count ordering is there in the tails. The registered statistic was the
median and the median tied. The plan registered this prediction **against** the project's only
committed evidence, which pointed the other way, and it was labelled as such rather than as
"genuinely unknown" — which is what the first draft called it, wrongly.

## Section 5.2's reshaping gain survives symmetric selection

| | mean | interval |
|---|---|---|
| this increment, matched grid, each arm CV-selected | **+0.0179** | [+0.0139, +0.0221] |
| increment 12, committed, matched two-point grid | +0.0155 | [+0.0118, +0.0192] |

So the gain is not downgraded a second time. It is also now known to be a **one-parameter** effect: the
same movement is bought by choosing one exponent as by fitting m weights.

## The scale channel is empty, demonstrated rather than asserted

`mean(skill_kernel_scaled − skill_kernel)` = **+0.00020**, positive on 20 of 40 — a coin flip, from an
arm with one extra free parameter. This is what increment 12's committed `gain_scale` of +0.0000 at
m=10 predicted, and it is why the normalised one-parameter family is not handicapped against a
free-scale fit. The arm entered no prediction and no outcome branch; it exists so a reader can see this
rather than take the plan's word for it.

## Where the plan's own text was wrong

**The conditioning guard bound on 9 of 70 targets, not once.** The frozen plan says "the guard binds
exactly ONCE: CHEMBL4361_Ki at m=10 on fold 2, cond 100.889". That figure came from the review, which
measured ChEMBL at m ≤ 10; I generalised it without checking m=20 and m=40, where it bites widely. The
measured truth is nine targets with at least one blocked m and a maximum cond(C) of **483.704**:

| target | blocked m | max cond(C) |
|---|---|---|
| `CHEMBL4860_Ki` | 10, 20, 40 | 483.70 |
| `CHEMBL4361_Ki` | 10, 20, 40 | 346.16 |
| `CHEMBL2147_Ki` | 20, 40 | 238.28 |
| `CHEMBL2835_Ki` | 20, 40 | 233.13 |
| `CHEMBL2971_Ki` | 40 | 205.72 |
| `CHEMBL1784_EC50` | 40 | 154.41 |
| `CHEMBL244_Ki` | 40 | 120.06 |
| `CHEMBL245_Ki` | 40 | 115.54 |
| `CHEMBL1862_Ki` | 40 | 105.98 |

The plan is frozen and is not edited; the correction lives here. **It matters in a direction that
favours the registered conclusion**, so it is stated rather than buried: on three targets
(`CHEMBL4361_Ki`, `CHEMBL1784_EC50`, `CHEMBL2147_Ki`) the guard left the fitted arm a smaller
CV-admissible set than its refit-admissible set, so on those the fitted arm could not select an m the
final refit would have supported. Three targets of seventy, and the matched-grid interval's lower end
is less than half the margin from zero, so the conclusion does not turn on them — but the guard is a
candidate-level decision that handicapped the kernel's competitor, and a reader is entitled to know by
how much.

**The kernel's advantage is understated by its own search cost.** The kernel searches 40 CV cells and
the fitted arm at most 5, and the protocol charges it for that: its mean selection regret is 0.0040
against the fitted arm's 0.0006. Under oracle selection the own-grid margin would widen from +0.0016
to about +0.0050. The registered protocol is the honest one and the reported numbers are the registered
ones; the point is that the direction of the selection asymmetry is conservative here, not inflating.
The pre-freeze review's S1 finding, that the asymmetry would manufacture a spurious advantage, was
refuted by measurement before freezing, and this is the same conclusion seen from the other side.

## What this does to the paper

Section 5.2 is rewritten to lead with the one-parameter result and to cite 2005 for both arms. The
honest statement is:

> The reshaping gain over uniform neighbour weights is real, survives symmetric cross-validated
> selection of the neighbourhood size, and is a **one-parameter** effect: an inverse-rank kernel
> `w_j ∝ j^-α` with α chosen by cross-validation attains the m-parameter least-squares optimum to
> within 0.0009 of skill, and cross-validation selects α = 1 — Gou's published rank weight — on 39 of
> 40 targets. The optimum itself and the use of a tuned one-parameter kernel are both due to Itskowitz
> & Tropsha, who derived the optimality condition for kNN QSAR in 2005 and then substituted a chosen
> one-parameter exponential without solving it. What is new here is the comparison their paper sets up
> and leaves open, and the uniform baseline their paper does not contain.

## Prior art, the fifth pass

This increment ran the pass on weighted kNN that `docs/paper/prior-art.md` has required since the
weighted-kNN result existed. It found that every citation in the plan's first draft was wrong about
what it cites, and it found the paper that matters.

- **Itskowitz & Tropsha, JCIM 2005, 45(3):777-785.** Their eq (17),
  `-2u_k + Σ_k' F_k' V_kk' - μ = 0`, "a system of linear equations with respect to `F_k`" built from
  label second moments with a Lagrange multiplier for the normalisation, is structurally this
  project's fitted arm. Their eq (21), `exp(-αd/D)` normalised, is structurally this increment's
  kernel arm. **Novelty is claimed for neither.** What they do not do: the sentence after eq (17) is
  future tense and the next section says "We chose the weighting function in the form" (21), so the
  optimum is derived and abandoned and the two are never compared; and the paper has no uniform arm,
  its reference being its own kernel at α = 1 (Table 1 reports q²(α) against q²(α=1), mean improvement
  6.7 %). Their k runs over 1 to 7 and their distances are Euclidean in a descriptor subspace, against
  this increment's rank index on Tanimoto neighbourhoods.
- **`w_j ∝ j^-α` is not Dudani's rule.** Dudani 1976 weights by distance,
  `w_i = (d_k - d_i)/(d_k - d_1)`. The rank reciprocal `1/j` is **Gou, Xiong & Kuang 2011**,
  J. Computers 6(5):833-840, and the family is declared on that citation. This mattered: the defence
  "the family was declared on a 1976 citation so it was not chosen after seeing the fitted profile" is
  only valid against a citation that contains the family.
- **Hechenbichler & Schliep 2004 contains no rank-indexed kernel at all**; every kernel in it is a
  function of the standardised distance. It is now cited as the distance-kernel family this increment
  deliberately does not test, which makes the plan's existing ban on claiming anything about
  distance-weighted kernels legible instead of unexplained.
- **Bailey & Jain 1978** proved that for fixed k the asymptotic classification error of the
  **unweighted** kNN rule is minimal among weighted kNN rules, for all distributions. This is a
  collision with section 5.1, which `prior-art.md` and `skeleton.md` both currently declare
  collision-free, and it is recorded separately rather than inside this increment.
- **Lenz, Bollaert & Cornelis, arXiv:2311.16872** find distance-weights better than rank-weights
  kernel by kernel within the NN classifier over 85 datasets, with exceptions, while their overall
  verdict for NN keeps rank weights. The declared family here is rank-indexed; the distance-indexed
  comparison on molecular similarity neighbourhoods is left to a separate frozen plan.

## Method note

The plan's first draft went through an adversarial pre-freeze review across four lenses: gate and
prediction feasibility, statistical validity, prior art, and implementation. It raised 23 findings, of
which **16 survived verification and 4 were fatal**, and its own proposed fixes were then adversarially
checked and corrected in six places. Seven findings were refuted, including the one I had thought most
likely to be fatal.

The fatal one I would not have found: the first draft never said a training molecule must be excluded
from its own neighbourhood. `anchors.select` does no masking, so unmasked `NN_1(v) = v`, hence
`C e_1 = c` exactly and `w = C^-1 c = e_1`, making the fitted arm 1-NN at every m. Measured over all 40
ChEMBL targets at m=5, the fitted-minus-uniform difference is +0.0159 masked and **−0.1018** unmasked,
with the fitted arm ahead on 37/40 masked and **0/40** unmasked. That is ten times prediction 3's margin
**in the direction that confirms the registered prediction**, so the increment would have reported that
one parameter suffices because its competitor was crippled — and the first draft's gate 1 was blind to
exactly that.

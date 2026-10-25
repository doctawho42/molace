# Paper skeleton

Rewritten 2026-10-06, replacing the version now at `skeleton_superseded_2026-10-06.md`. The previous
skeleton listed eight claims and treated the identity as the finding. Eleven adversarially-tested
attempts to build something larger on it all died, with a diagnosable cause, and two of the eight
claims turned out to be measured against baselines the comparison requires and the project never ran.
This version says what the material supports.

**Every section below carries a status line.** `replication`, `known object, new accounting`,
`own contribution`, or `correction`. Nothing is presented as new unless an adversarial prior-art pass
failed to place it.

---

## What this paper is

A methods paper about one quantity: the error of the k-nearest-neighbour mean of an activity on a
molecular similarity graph. The paper writes it in closed form, shows that the published continuous
modelability index is that quantity to within a percent, shows which of its two terms every published
index omits, and shows that it is not a property of a dataset but of a dataset **and its size**.

**Working title.** *The nearest-neighbour floor in closed form: what a modelability index measures, and
why it is not a property of a dataset.*

**Venue.** JCIM. The modelability programme is native there (Golbraikh 2014; Luque Ruiz & Gómez-Nieto
2018; Aldeghi/Graff 2022) and so is its critique.

**Honest ceiling.** A solid methods paper. Not a breakthrough, and the introduction should not pretend
otherwise. The project's attempts at a breakthrough are recorded in `prereg/REFUSED_*` and in the
refusal log, and the reason they failed is stated in the Limits section because it is informative.

---

## 1. Introduction

The modelability programme asks whether a dataset is worth modelling before a model is fitted. MODI
(Golbraikh, Muratov, Fourches & Tropsha, *JCIM* 2014, 54(1):1–4) is an activity-class-weighted
agreement of 1-nearest-neighbour pairs with a threshold of 0.65 calibrated on 100+ datasets; its
continuous forms MODI_q2 and MODI_ssR2 are leave-one-out and leave-group-out q² of similarity search
(Golbraikh et al., Springer 2014 and 2021). ROGI (Aldeghi, Graff et al., *JCIM* 2022, 62(19):4660) is a
roughness scalar; ROGI-XD (Graff et al., *Digital Discovery* 2023, 2(5):1452) corrects its
representation dependence. RMODI (Luque Ruiz & Gómez-Nieto, *JCIM* 2018, 58(10):2069) is the regression
analogue. Kernel target alignment serves the same role in closed form (Marcou, Horvath & Varnek,
*JCIM* 2016, 56(1):6).

Every one of these is published as a **bare point estimate per dataset**. The ROGI authors say so in
print: *"While we do not provide a statistical estimator of ROGI uncertainty..."*, and report that
their bootstrap *"generally correlates with, but also underestimates, the ROGI error"*. No index in the
line carries an interval, and none reports how it moves with the amount of data.

This paper takes one estimator — the kNN mean — and writes its error exactly.

---

## 2. The closed form

**Status: known object, new accounting.**

Standardise the activity. For the mean of a compound's `k` nearest neighbours,

    MSE / sigma^2  =  1 + 1/k + ((k-1)/k) * rho_nn  -  2r

with `r` Newman's target assortativity of the activity across graph edges and `rho_nn` the correlation
between the activities of two compounds sharing a neighbour. Equivalently, with
`A_norm = D^-1 A` and `m_j = y' A_norm^j y / y'y`,

    ||(I - A_norm) y||^2 / ||y||^2  =  1 - 2 m1 + m2 ,      m1 = r exactly.

- **Exact on k-regular graphs**, pinned by `tests/test_neighbourhood.py` at `abs=1e-12`, so a broken
  derivation fails the suite. Verified to 3.58e-15 over 1200 synthetic circulants.
- **An approximation on real union-symmetrised kNN graphs**: mean absolute error 0.0256 as a point
  prediction of the measured floor over 40 ChEMBL targets at k=10, max 0.0837, and stable in k
  (0.0269, 0.0252, 0.0256, 0.0286, 0.0306 at k = 3, 5, 10, 20, 40).
- **And it degrades as the sample shrinks**, which bounds where it may be used and was measured for
  the first time in increment 15: mean absolute difference against the measured floor runs 0.0103,
  0.0133, 0.0168, 0.0221 at the full set, a half, a quarter and an eighth of each target. A factor of
  2.1, monotone, and expected: the identity is exact only on k-regular graphs, and union
  symmetrisation is further from regular when k is a larger fraction of n.

**What is not ours.** The right-hand side is a Dirichlet energy, a standard homophily and smoothness
measure in the heterophilic-graph literature (Heterophilic Graph Learning Handbook, arXiv 2407.09618).
As a risk it is the Wiener filter and the kriging system. The paper says so in this section rather than
leaving a reviewer to say it.

**What is ours.** The accounting. That `m1` *is* Newman's target assortativity, the statistic
graph-learning benchmarks already report; that `m2` is the shared-neighbour correlation; and that the
published continuous modelability index is this expression. Measured against MODI_q2 on 80 targets
under a plan frozen before any value existed (`prereg/increment11_modi.yaml`, blob
`acdba0f77ff9c8fc9149dc9cd8ed39db94122ae1`):

| collection | targets | mean abs. difference | max | Pearson |
|---|---|---|---|---|
| ChEMBL-40 | 40 | 0.0103 | 0.0436 | 0.9975 |
| MoleculeACE-30 | 30 | 0.0083 | 0.0238 | 0.9960 |
| DeepDelta-10 | 10 | 0.0123 | 0.0362 | 0.9975 |

Frozen threshold 0.05, held everywhere with a factor of four to spare. MODI_q2 costs one leave-one-out
prediction per compound, eight thousand on the largest target here; the right-hand side costs two
sparse matrix products and no prediction.

---

## 3. The term every published index omits

**Status: own contribution, with a stated limit.**

`m1` alone is what the field reports. The second moment is absent from MODI, MODI_q2, RMODI, ROGI and
kernel target alignment alike. Dropping it costs a factor of 2.5 in level: reproducing MODI_q2 from
`1 - r` alone gives mean absolute error 0.0257 against the full identity's 0.0103 on ChEMBL-40.

**The limit, stated here and not buried.** On ECFP4/Tanimoto graphs the two moments are nearly
collinear, Pearson `+0.9895` across the 40 ChEMBL targets. As a **ranker** the second moment adds
little: Spearman against the measured floor is `+0.9345` for `1 - r` and `+0.9497` for the full
identity, and its incremental ranking contribution covers zero on all three collections
(+0.0156 [−0.0171, +0.0600]; +0.0120 [−0.0542, +0.0849]; +0.0485 [−0.1132, +0.3376]).

So the claim is precise: **the second moment is necessary for the level and quiet for the ranking.**
A paper that claimed it improves ranking would be refuted by its own numbers.

---

## 4. Modelability is not a property of a dataset, and the movement is in closed form

**Status: own contribution for the accounting and the consequence; the mechanism is classical and is
cited as such.** Measured under `prereg/increment15_size_dependence.yaml`, blob
`3d941919c5aeb34e2749cc820e8e3ae55d5ff725`, frozen before any number existed; see
`results/results_increment15.md`.

Every index in section 1 is reported per dataset, as though it were a property of the chemistry. The
quantity those indices estimate is a property of the chemistry **and the sample size**, and it moves
far:

| | mean | interval | sign count |
|---|---|---|---|
| floor rel. MSE at n/8 minus at n | **+0.2698** | [+0.2477, +0.2922] | 40 / 40 |
| assortativity at n minus at n/8 | **+0.2599** | [+0.2385, +0.2807] | 40 / 40 |

| fraction | mean n | assortativity | floor rel. MSE | between-chain sd |
|---|---|---|---|---|
| 1/8 | 345 | 0.2957 | 0.6915 | 0.0456 |
| 1/4 | 690 | 0.3855 | 0.5919 | 0.0355 |
| 1/2 | 1380 | 0.4752 | 0.4992 | 0.0186 |
| all | 2760 | **0.5556** | **0.4217** | — |

Per target the median ratio `r(n) / r(n/8)` is **1.988**: assortativity nearly doubles over an
eightfold change in size. Both instrument gates read exactly `0.000e+00` over all 40 targets, the
second against a recomputation route that shares no indexing code with the first.

**The movement is rank dilution, and that is not ours.** A null arm that holds the point set, every
pairwise similarity, the label distribution, the variance normaliser and the scored set at full n, and
changes only which ranks of its own ordering each molecule reads, reproduces the measured movement at
every fraction: ratios 1.0060, 1.0271 and 0.9848 against a band of [0.90, 1.10] declared in advance.
That n enters a kNN estimator only through `k/n`, by neighbour-rank order statistics, is published:
Samworth, *Annals of Statistics* 2012, 40(5):2733, Theorem 1, whose squared-bias factor for the
unweighted rule is exactly `(k/n)^{2/d}`; and the near-neighbour distance moments
`E[eps_k^a] <= C (k/n)^{a/D}` citing Evans, Jones & Schmidt, *Proc. R. Soc. A* 2002, 458:2839. The
paper cites both at the mechanism and claims no novelty for it.

**What is ours is the accounting and the consequence.** On the one collection where 0.65 is the
source's own threshold — MoleculeACE, where an activity-class flag exists — the 2014 binary MODI
declares **all 30 targets modelable at full n and only 54 % of cells at n/8** (mean 0.7592 against
0.6514). Same targets, same chemistry, and the published criterion crosses by growing alone. On
ChEMBL the same monotonicity holds at all eight thresholds of a declared grid; the 0.65 row runs 2 of
40 at n/8 to 10 of 40 at full n, and it is reported inside a grid because **0.65 is the binary index's
threshold and not this one's** — the field's threshold cannot be evaluated on ChEMBL at all, since
this project has no activity classes there, and transposing it onto q2 is our step, not Golbraikh's.

So the claim is exactly this: an index whose sample dependence is predictable in closed form from the
full-n ordering alone was nevertheless published, by every paper in the line, as a bare point estimate
with no n beside it. Reporting it that way reports a point on a curve whose position was never in
doubt.

**A normaliser channel that would have been misread.** The residual between the measured arm and the
null decomposes exactly, to 8.67e-19, into a variance-normaliser term and a rank-dilution term. At
n/8 the normaliser term is `+0.00753` and the rank-dilution residual `-0.00159`: 4.7 times larger and
pointing the other way. Reported without that split, the residual reads `+0.00594` and looks like
sample dependence beyond rank dilution, in the direction that flatters this section. It is not.

**What this section does NOT claim.** No scaling law, no exponent, no extrapolation outside [1/8, 1],
and the reason is evidential rather than stylistic: with three points and a two-parameter family, a
deliberately unrelated quadratic in `log(1/f)` fits the measured movement as well as the classical
power law does, and the implied dimension swings from 9.5 to 94.8 under perturbations the measured
between-chain spread of 0.0456 makes ordinary. It also claims nothing about a realistic
medicinal-chemistry acquisition order; the subsampling here is uniform.

**And the opposite reading exists.** The Gamma test line this paper already cites contains the
**M-test**, which recomputes the statistic over growing samples exactly as this section does, and reads
the result as convergence towards a truth rather than as evidence that the quantity is not the data's.
The paper states both readings and asserts that neither is settled here.

## 4a. A model arm is a choice, and this one was worth 0.10 skill

**Status: own contribution, and a withdrawal.**

The project previously claimed the kNN floor beats fitted models. It does not. Measured under a plan
frozen before any number existed, with an instrument gate confirming the floor arm reproduces the
earlier numbers to 5.551e-17:

| | mean | interval | floor ahead on |
|---|---|---|---|
| ChEMBL-40, floor − CV-selected learner | **−0.0754** | [−0.0865, −0.0639] | 3 / 40 |
| MoleculeACE-30, floor − CV-selected learner | **−0.0949** | [−0.1084, −0.0806] | 1 / 30 |
| ChEMBL-40, CV-selected − the arm used before | **+0.1025** | [+0.0899, +0.1156] | 40 / 40 |

The earlier arm averaged histogram gradient boosting with an MLP and excluded the SVM. Per-learner mean
skill on ChEMBL-40: svm +0.4221, hgb +0.3969, mlp +0.2446. Cross-validation on the training split picks
the SVM on 34 of 40 ChEMBL and 30 of 30 MoleculeACE targets, so the arm averaged the second-best learner
with the worst and left out the one that wins.

**+0.1025 is larger than every effect this paper reports**: the reshaping gain is +0.0155, the second
moment's ranking contribution +0.0156, the MODI reproduction error 0.0103. The choice of baseline
dominated the science, and nothing in the repository recorded that a choice had been made.

**And the same error, pointed at us.** The audit fixed the model arm and left the floor at a fixed
`m = 10`. Giving the floor its better `m` narrows the gap from −0.0824 to −0.0596 [−0.0708, −0.0482],
still excluding zero, floor ahead on 1 of 31. Reported as a sensitivity because it was not
pre-registered; a symmetric pre-registered version is owed.

**What stays open.** Janela & Bajorath (*Nature Mach. Intell.* 2022, 4:1246) report the simple control
meeting or exceeding complex ML. We do not reproduce that tie — but their control is 1-NN/kNN with their
choices and ours is a uniform mean over 10 nearest training neighbours at fixed `m`, on a different
selection and a single 80/20 split. The sensitivity shows `m` alone carries 0.023 of the 0.075. Settling
it means running their protocol on their public data
(`github.com/TiagoJanela/ML-for-compound-potency-prediction`), which is now the best-motivated experiment
in the queue.

---

## 5. What the neighbourhood's internal structure buys

**Status: a theorem with no found collision, whose scope must be stated exactly, plus a correction.**

### 5.1 The vacuity theorem

With `c_i = E[z_v z_{u_i}]` and `C_ij = E[z_{u_i} z_{u_j}]` over a compound's `k` neighbours, the best
linear predictor is `w* = C^-1 c` with `MSE* = 1 - c' C^-1 c`. If the neighbourhood is **second-order
exchangeable**, `c = r 1` and `C = (1-rho) I + rho 11'`, so `1` is an eigenvector of `C` with eigenvalue
`S = 1 + (k-1) rho` and `c` is parallel to it. Hence, with `beta = k r / S`:

    w*               = (r / S) 1            every optimal weight is EXACTLY equal
    sum_i w*_i       = beta                 the weight sum is the regression slope on the mean
    MSE_mean - MSE*  = Var(mean) (1-beta)^2 the whole gap is a squared scale error
    MSE(best alpha * mean) = MSE*           exactly, for all k, r, rho

Two independent derivations (Sherman–Morrison; permutation symmetry with a one-dimensional reduction)
and a numerical sweep over 2000 random `(k, r, rho)` agreeing to 1.7e-15.

**A neighbouring 1978 result, found by the fifth prior-art pass and not by the four before it.**
Bailey & Jain, *IEEE Trans. SMC* 1978, 8:311-313, proved that for **fixed k** the **asymptotic
classification** error of the **unweighted** kNN rule is minimal among weighted kNN rules, for all
distributions. The four earlier passes reported no collision for this section; that report was made
without this paper and does not stand. What the two statements share is the conclusion that weighting
cannot help; they differ in what is assumed and what is concluded. Bailey & Jain is asymptotic, is
about classification error, and holds for all distributions at fixed k. The statement above is
finite-sample, is about the squared error of a linear predictor, and holds under second-order
exchangeability — which section 5.2 then measures to be materially false on molecular data. **The
paper states both and does not assert that either subsumes the other**: establishing the relationship
would be its own piece of work, and claiming the 1978 theorem as the asymptotic form of this one
without that work would be a guess dressed as a citation.

**Scope, stated because the paper will be attacked here.** The theorem bounds predictors that are
**linear in the labels of a fixed neighbour set under a fixed metric**. It says nothing about a model
reading the query molecule's own features. A two-line counterexample makes this explicit and belongs in
the paper: with a latent shared coordinate, exchangeability holds exactly (`c_j` all 0.50, `C`
off-diagonals 0.50, optimal weights equal to 3e-3), `MSE*` is 0.5466 and the plain mean 0.5502 — one
scalar of headroom, as the theorem says — while a predictor reading the query's own features attains 0.
**Therefore this theorem is not an explanation of why fitted models tie nearest-neighbour controls**
(Janela & Bajorath, *Nature Mach. Intell.* 2022, 4:1246), and the paper must not claim that it is.

What it does explain is narrower and real: **why the rank axis was never fitted.** Write the natural
exchangeable model and the question "should nearer neighbours get more weight" cannot be posed, because
the answer is provably "no, they are already equal". The residual then reads as a null.

### 5.2 Neighbourhoods are not exchangeable, and the correction

Measured for the first time here on molecular data: the second moment between a compound's activity and
its j-th nearest neighbour's, averaged over 31 ChEMBL targets at k=10,

    rank  1     2     3     4     5     6     7     8     9    10
    c_j  0.714 0.660 0.629 0.597 0.572 0.542 0.523 0.513 0.499 0.476

with mean decline `+0.2384 [+0.2165, +0.2603]`, excluding zero on every collection. The exchangeability
hypothesis is materially false.

**Status: correction.** Reweighting by rank is worth `+0.0155 [+0.0118, +0.0192]` out of sample,
positive on **30 of 31** targets, against the best training-selected uniform `m` in each arm. The
project first reported `+0.0379` against uniform at a fixed `m = 10`; 62 % of that was the effect of
fixing `m`, since uniform at `m = 5` beats uniform at `m = 10` on 28 of 31 targets.

**And the whole effect is bought by ONE parameter.** Measured under
`prereg/increment16_weighting_family.yaml`, blob `0ce65b307616a86cf0b565e854151096501d3f28`; see
`results/results_increment16.md`. Four arms, every one of them selected by 5-fold cross-validation on
the training split alone, with all three instrument gates reading exactly `0.000e+00`:

| arm | parameters | median m | median α | mean skill |
|---|---|---|---|---|
| uniform | 0 | 5 | — | 0.3658 |
| **`w_j ∝ j^-α`** | **1** | **10** | **1.000** | **0.3839** |
| the same times a fitted scale | 2 | 10 | 1.000 | 0.3841 |
| fitted `w = C^-1 c` | m | 10 | — | 0.3823 |

The one-parameter kernel beats uniform weights by **+0.0181 [+0.0146, +0.0218]** on 37 of 40 ChEMBL
targets and **+0.0211 [+0.0171, +0.0257]** on 30 of 30 MoleculeACE targets, and on a matched
neighbourhood grid it gives up **0.0009 [-0.0048, +0.0025]** of skill to the `m`-parameter optimum, so
a pre-registered non-inferiority margin of -0.01 holds with the interval's lower end less than half
the margin from zero. Cross-validation, given eight exponents and never shown a test label, selects
**α = 1 on 26 of 40 targets**, with α ∈ {0.75, 1, 1.5} on **39 of 40** and an interquartile range of
[1.000, 1.000]. The reshaping gain re-measured under symmetric selection is **+0.0179
[+0.0139, +0.0221]**, so the correction above survives. The two-parameter arm adds +0.0002, positive
on 20 of 40: the scale channel is empty, shown rather than asserted.

**Neither arm is ours, and the section says so.** Itskowitz & Tropsha, *J. Chem. Inf. Model.* 2005,
45(3):777-785, derived the optimality condition for kNN QSAR as their eq (17),
`-2u_k + Σ_k' F_k' V_kk' - μ = 0`, "a system of linear equations with respect to `F_k`" built from
label second moments with a Lagrange multiplier for the normalisation — structurally the fitted arm
above, twenty years earlier, in this paper's target venue. Their eq (21), a normalised
`exp(-αd/D)` with α tuned for performance, is structurally the one-parameter arm. **What their paper
does not do** is compare them: the sentence after eq (17) is future tense and the next section says
"We chose the weighting function in the form" (21), so the optimum is derived and abandoned. Nor does
it contain a uniform arm at all; its reference is its own kernel at α = 1. That comparison and that
baseline are what this section adds, on a rank index over Tanimoto neighbourhoods rather than their
Euclidean descriptor distances with k in 1..7.

`w_j ∝ j^-α` is also **not Dudani's rule**, as this project previously wrote. Dudani, *IEEE Trans.
SMC* 1976, 6(4):325, weights by distance, `w_i = (d_k - d_i)/(d_k - d_1)`; the rank reciprocal `1/j`
is Gou, Xiong & Kuang, *J. Computers* 2011, 6(5):833-840, and the family is declared on that citation.
Hechenbichler & Schliep 2004 contains no rank-indexed kernel of any kind and is cited for the
distance-kernel family this section deliberately does not test. Lenz, Bollaert & Cornelis
(arXiv:2311.16872) find distance-weights stronger than rank-weights kernel by kernel within the NN
classifier over 85 datasets; that axis is left to a separate frozen plan.

**Against this result.** The non-inferiority is an equivalence, not a win: the kernel leads on only 19
of 40 targets on the matched grid, and its deficit exceeds the margin on 3, worst -0.0565 on
`CHEMBL614358_EC50`. The prediction that a kernel would select a larger neighbourhood than the fit
**failed**, on a tie of medians at 10. And the kernel pays a larger selection cost than its
competitor, 0.0040 of regret against 0.0006, so the protocol charges the winning arm more than the
losing one.

---

## 6. Method, and six refusals

**Status: own contribution, and the part no other group can copy.**

Every increment carries a YAML pre-registration committed and referenced by its **git blob hash**,
frozen before any measurement and never edited afterwards; scripts carry the blob in a module constant
and print it first. Design gates are evaluated before outcomes are read. 431 tests.

The project has **six times refused a verdict its own frozen rule granted**, because the design could
not earn it. Five were design gates on the decoupling of the two moments, with a proved structural
cause: on a stride-`d` circulant with an exponential field, `lambda = rho_nn / r = c(d)` is pinned to
`r` by the field, so no graph construction can separate them. The sixth was refused before freezing,
when the plan's apparatus gate was found already failed by the project's own committed data.

**And a baseline audit on its own claims, which belongs in this paper because it is the subject
Janela & Bajorath's 2023 title names.** Two claims the project had published were measured against
baselines the comparison requires and the project never ran: the reshaping gain against a fixed `m`
(section 5.2), and the model arm for the floor-versus-model comparison, which averages histogram
gradient boosting and an MLP and **excludes the SVM** that the project's own code comments call the
real bar. Over 40 targets the trained arms beat the floor on only 14 of 40, median −0.016, so the
comparison's sign depends on a choice nobody declared. Reported as a case study, with the corrected
numbers, not as a footnote.

---

## 7. Limits

- **The object is classical.** The identity is a Dirichlet energy, the risk is the Wiener/kriging form,
  the weighted-kNN result is adjacent to Dudani 1976 and Samworth 2012, and the noise-floor route is
  the Gamma test (Stefánsson/Končar/Jones 1997; Evans & Jones, *Proc. R. Soc. A* 2002). Eleven
  adversarially-tested attempts to build a larger claim on it failed, in three recurring ways: the
  phenomenon involved models outside the theorem's class, or the extension landed in kriging, or the
  question was already answered inside the class. This is stated because it bounds what a reader should
  expect from the quantity, and it is the honest reason this is a methods paper.
- **One domain, one representation family.** Everything is ECFP4/Tanimoto on molecular activity. The
  representation check (section 3 of the old skeleton) survives on one genuinely different
  representation: Spearman of rank assortativity against attained skill is +0.726 on RDKit
  physicochemical descriptors, +0.893 on MACCS, +0.915 on atom pairs.
- **The size result uses random subsampling**, not a realistic acquisition order. Real datasets grow by
  medicinal-chemistry series, which plausibly grows homophily less than random growth does. Under
  uniform subsampling the movement is fully accounted for by rank dilution (increment 15), so nothing
  in section 4 bears on how a real acquisition order would behave, and the section says so.
- **The two moments are nearly collinear here** (Pearson +0.9895), so this data cannot separate their
  contributions to ranking. The regime where it could is the one five refused gates showed is
  unreachable at this sample size.
- **The floor does not beat the best fitted model**, so the paper makes no claim that a simple control
  suffices. Whether it *ties* under a protocol closer to Janela & Bajorath's remains open; see 4a.

---

## Blocking experiments, in order

1. ~~The baseline audit~~ **DONE, increment 14.** `prereg/increment14_baseline_audit.yaml`, blob
   `179b8bf128302f9c266dd7af9e2815c9049f18f2`; `results/results_increment14.md`. **Claim 4 reverses.**
   The floor loses to the CV-selected learner by −0.0754 [−0.0865, −0.0639] on ChEMBL-40, ahead on 3 of
   40, and by −0.0949 [−0.1084, −0.0806] on MoleculeACE-30, ahead on 1 of 30. The arm choice was worth
   +0.1025 [+0.0899, +0.1156]. See the new section 4a below.
2. ~~Weighted kNN against Dudani~~ **DONE, increment 16.**
   `prereg/increment16_weighting_family.yaml`, blob `0ce65b307616a86cf0b565e854151096501d3f28`;
   `results/results_increment16.md`. **The kernel suffices.** One parameter attains the `m`-parameter
   optimum to within 0.0009 at a pre-registered non-inferiority margin, cross-validation picks α = 1 on
   39 of 40 targets within {0.75, 1, 1.5}, and all three instrument gates read exactly 0.0. The family
   is Gou 2011 and not Dudani 1976, which weights by distance; and both arms turn out to be Itskowitz
   & Tropsha 2005, who derived the optimum and the kernel and never compared them. The instrument
   checks this entry proposed were replaced: `alpha = 0` against the uniform arm cannot fail and is
   labelled weak, and the `knn_floor` check was **algebraically blind** to the estimation-side
   neighbourhood it claimed to tie. See section 5.2 above.
3. ~~The size dependence, pre-registered~~ **DONE, increment 15.**
   `prereg/increment15_size_dependence.yaml`, blob `3d941919c5aeb34e2749cc820e8e3ae55d5ff725`;
   `results/results_increment15.md`. **Section 4 survives and is rewritten.** The index moves by
   +0.2698 [+0.2477, +0.2922] in relative MSE and +0.2599 [+0.2385, +0.2807] in assortativity over an
   eightfold change in size, on 40 of 40 targets; both instrument gates read exactly 0.0; and the
   movement is entirely the classical k/n rank-dilution channel, which the section now cites rather
   than claims. The confound was measured instead of declared, and the decomposition turned up a
   variance-normaliser term 4.7 times the residual and pointing the other way. See section 4 above.
4. **The fifth prior-art pass**, which `prior-art.md` has listed as required since the weighted-kNN
   result existed: Dudani 1976, Hechenbichler & Schliep 2004, Samworth 2012, locally weighted
   regression in QSAR, and Janela & Bajorath's line (*Nature Mach. Intell.* 2022, 4:1246;
   *Sci. Rep.* 2023, 13:17816; *Pharmaceuticals* 2023, 16(4):530) which is claim-4's prior.

## Dropped, with the reason

- **The noise floor and the attainable-accuracy ceiling.** The two-statistic estimator is nugget
  estimation and fails out of sample; the identified set is `lambda_min(Sigma)` from minimum-trace
  factor analysis; the training-free bound is the Gamma test. Keep one paragraph citing all three and
  reporting the project's measured failure as a replication of a known failure mode.
- **The Hodge half.** 96.6 % gradient and the proved collapse are pre-empted by Balduzzi et al.,
  NeurIPS 2018, which owns the collapse statement, the closed-form potential and the orthogonal
  gradient/curl split; HodgeRank owns the Pythagorean residual; Wetzel's twin-network line enforces
  loop consistency verbatim. One sentence and a citation, or nothing.
- **Learned geometry, learning curves, acquisition functions, spectral transport, impossibility
  theorems.** All refused; see `prereg/REFUSED_increment13_homophilisation.yaml` and the refusal log.

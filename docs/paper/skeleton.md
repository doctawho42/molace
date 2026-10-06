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

## 4. Modelability is not a property of a dataset

**Status: own contribution. The paper's sharpest empirical section.**

Every index in section 1 is reported per dataset, as though it were a property of the chemistry. It is
a property of the chemistry **and the sample size**. On the same 40 ChEMBL targets, under a nested
random subsample chain:

| fraction of the data | mean n | assortativity r | kNN floor (rel. MSE) |
|---|---|---|---|
| 1/8 | 345 | 0.292 | 0.697 |
| 1/4 | 690 | 0.383 | 0.592 |
| 1/2 | 1380 | 0.474 | 0.501 |
| all | 2760 | **0.556** | **0.422** |

Per target, `r(n) / r(n/8)` has median **1.986** — assortativity nearly doubles over an eightfold
change in size — and the floor falls by a median 0.275 in relative MSE.

**The consequence for the field's one published threshold.** MODI's 0.65 is crossed by growing alone:
at `n/8` **2 of 40** targets clear it; at full `n`, **10 of 40**. Same targets, same chemistry, five
times as many declared modelable because there is more data. "This dataset is not modelable" is, in
part, "I do not have enough of this dataset yet".

**What this section does NOT claim.** It does not claim a scaling law. Extrapolating the curve from
small subsamples is partly circular: for a fixed point set the j-th nearest neighbour in a subsample is
the i-th in the full set with i negative-hypergeometric, so much of the n-dependence is order
statistics rather than chemistry. The claim is the **observation** that the index moves, which is
enough to invalidate reporting it without a size, and which no paper in the line reports.

**Pre-registration note.** The numbers above come from an exploratory probe and must be re-measured
under a frozen plan before they enter the paper. See the blocking experiments below.

---

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
and a numerical sweep over 2000 random `(k, r, rho)` agreeing to 1.7e-15. An adversarial prior-art pass
that killed six other proposals found no collision for this statement.

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

**And the optimal profile is a 1976 rule.** Its cosine similarity with `w_j` proportional to `1/j` is
**0.9946**, against 0.7002 with uniform. Dudani, *IEEE Trans. SMC* 1976, 6(4):325, is therefore the
competitor this section is stated against, with Hechenbichler & Schliep 2004 and Samworth,
*Annals of Statistics* 2012, as the theory. Whether the fitted `m`-parameter optimum beats the
one-parameter inverse-rank kernel is **not settled** and is a blocking experiment.

---

## 6. Method, and six refusals

**Status: own contribution, and the part no other group can copy.**

Every increment carries a YAML pre-registration committed and referenced by its **git blob hash**,
frozen before any measurement and never edited afterwards; scripts carry the blob in a module constant
and print it first. Design gates are evaluated before outcomes are read. 422 tests.

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
  medicinal-chemistry series, which plausibly grows homophily less than random growth does.
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
2. **Weighted kNN against Dudani.** The inverse-rank-power family `w_j ∝ j^-alpha`, declared on the 1976
   citation so the family is not chosen after seeing the profile, with `m` and `alpha` selected by
   5-fold CV on the training split; against uniform-at-best-`m` and against the fitted `C^-1 c`
   optimum. Instrument checks: `alpha = 0` must reproduce the uniform arm to machine precision, and the
   uniform `m=10` arm must reproduce `knn_floor.predict` to the 3.553e-15 already measured.
3. **The size dependence, pre-registered.** Re-measure section 4 under a frozen plan, with the
   subsampling scheme, the sizes, the seeds and the decision rule declared first, and with the
   order-statistics confound stated as a limit rather than tested as a law.
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

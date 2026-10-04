# A two-statistic estimate of attainable accuracy: better by an order of magnitude, and still not a bound

Pre-registration: `prereg/increment6_ceiling.yaml`, blob
`be0e9415878b85f76994ea75fc0338488e284dcc`, frozen and committed before the held-out set was read for
this question. Reproduce with `uv run python scripts/ceiling_holdout.py`; output in
`results/report_ceiling_holdout.txt` and `results/ceiling_holdout.csv`.

## The idea

The project has wanted a ceiling on attainable accuracy from the graph alone and kept failing to get
one. Substituting `(1 - r)` for the representation's Bayes error gives `skill <= 1 - sqrt(1 - r)`, and
increment 4 measured it violated on 28 of 40 targets, because neighbours do not coincide in feature
space so `r` understates the signal fraction.

The identity of increment 5 supplies a second statistic. Model the label correlation as decaying
geometrically with hop distance, `rho_d = (1 - nu) * lambda^d`. Target assortativity is the one-hop
value and the shared-neighbour correlation the two-hop value, so two equations fix the one parameter:

    r = (1 - nu) * lambda,   rho_nn = (1 - nu) * lambda^2    =>    1 - nu = r^2 / rho_nn

and therefore

    skill  <=  1 - sqrt(1 - r^2 / rho_nn)

Nothing is trained and nothing is fitted. This is a ONE-PARAMETER MODEL solved from two equations, and
the pre-registration forbids calling it a theorem anywhere in the project's prose.

## Found on the 40 ChEMBL targets, which therefore cannot confirm it

| | one statistic | two statistics |
|---|---|---|
| violated on | 29 / 40 | **0 / 40** |
| mean slack | -0.0120 | +0.0528 |

`r^2 / rho_nn` stayed inside (0, 1] on all 40, so the decay model is self-consistent there.

## The held-out test: the 30 MoleculeACE targets

Those 30 are the discovery set for the project's ORIGINAL claim, and the 40 ChEMBL targets were
curated to exclude their identifiers. For the ceiling formula they had never been read, because the
formula did not exist when they were last touched. The three arms are not refitted: their RMSEs come
from `results/spine.csv` and only the baseline is recomputed to turn them into skills.

| | |
|---|---|
| inadmissible targets | 0 / 30 |
| **two-statistic ceiling violated on** | **2 / 30** |
| one-statistic ceiling violated on | 19 / 30 |
| mean slack | +0.0567, range [-0.0486, +0.1380] |
| targets within 0.02 below their ceiling | 3 / 30 |
| Spearman(ceiling, best attained skill) | **+0.866** [+0.613, +0.995], excludes zero |

The two violations:

| target | exceeds the ceiling by | ceiling | best arm | class | n |
|---|---|---|---|---|---|
| CHEMBL237_EC50 | +0.0486 | 0.383 | 0.431 | GPCR | 955 |
| CHEMBL4792_Ki | +0.0259 | 0.392 | 0.418 | GPCR | 1471 |

## Verdict against the frozen rule

The frozen threshold for validity was **at most one violation**. There are two. So:

**It is not a valid bound out of sample.** The frozen branch for this case says the formula was a
within-set fit and the talk carries it as an idea with a measured failure rather than as a result.
That is what the project does. The threshold is not revisited after the fact, and the two violations
are not argued away.

The other three conditions all hold, and they are worth stating in the same breath, because the
honest summary is neither "it works" nor "it failed":

- not vacuous: mean slack +0.0567 with three targets inside 0.012 of their ceiling;
- informative: it ranks attained accuracy at +0.866 with an interval excluding zero;
- the second statistic adds a great deal: 19 violations become 2.

So what the increment delivers is a **predictor** of attainable accuracy, computable from two graph
statistics with nothing trained, that is roughly an order of magnitude closer to being a bound than
assortativity alone, and that is not a bound.

## What the failure is probably not, and what it might be

Post hoc, not pre-registered, offered as a direction and not a finding: both violations are GPCR
targets and both are small. Whether the geometric decay model is systematically wrong on GPCR panels
is a question for another increment with its own frozen plan. It is NOT a reason to adjust the
formula here, because the pre-registration forbids fitting lambda or nu to the held-out data, and any
correction chosen after seeing which targets broke would be exactly that.

What the failure is not: an admissibility problem. `r^2 / rho_nn` stayed in (0, 1] on every one of the
30 targets, so the model never had to be rescued, it simply came out slightly too low twice.

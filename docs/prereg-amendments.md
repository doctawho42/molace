# Pre-registration history

`prereg/increment1.yaml` is anchored by its git blob hash, which every result file and figure caption
carries. That stamp exists so a number produced under a different plan is visible rather than
arguable, which means a stamp mismatch has to be explained rather than shrugged at.

The file's own rule: amendments before any measurement are permitted; changes after the sweep require
a new file with a new name, never an edit of this one.

## Amendments, all before any measurement ran

| when | blob | what changed | analysis fields touched |
|---|---|---|---|
| 2026-10-03, after Task 7 | — | recorded the roughness flavour actually shipped (plain ROGI, with the reason) | none |
| 2026-10-03, after Task 10 | — | both `learners` lists `lightgbm` to `hgb`, plus a `learner_note` giving the reason | `arms` only |
| 2026-10-03, after Task 12 | — | `selection: NONE`, `aggregation`, `matched_families`, `reported_but_not_matched`, `feature_maps`, `kernel_slot`, measured hyperparameter costs | `arms`, `dependent_variables` |
| 2026-10-03, after Task 15 | `8e26372e` | `census_estimated` became `census_split`: the randomised rank estimator was removed | `hodge` only |

## One edit after the first measurement, and why the stamps differ

| file | stamp | note |
|---|---|---|
| `results/spine.csv`, `results/hodge_census.csv` | `8e26372e…` | written before the edit below |
| `results/report_spine.txt`, `results/report_hodge.txt`, figure captions | `864ec513…` | written after it |

The edit is commit `efe85f0` and it changed **prose only**: the `circularity_note` under
`independent_variables.categorical_arm`, correcting the claim that the cliff label is defined on a
different relation from our graph. It was wrong; see `docs/preconditions.md`. **No analysis field
moved** — not the statistic, not the gap, not the decision rule, not the clustering, not the holdout.

So the mismatch is a prose correction that happened to land between two writes, and it is recorded
here rather than left for a reader to discover. A reviewer flagged the unexplained mismatch, which is
exactly what the stamp is for.

## The amendment the next increment owes

**The gap must be dimensionless before anything is measured.** It is currently expressed in the
label's own units, which invalidated the holdout outright: the nine TDC endpoints mix log and raw
scales, their label standard deviations span a factor of about a hundred, and both the statistic and
the gap track that scale, so the correlation the holdout reports is a property of units. The spine
escaped only partly — its label spread varies about twofold and correlates with the statistic at
about +0.65, and a post-hoc standardised primary falls from -0.22 to -0.07, which is also a null, so
the conclusion stands while the magnitude does not.

Define the gap in label standard deviations in the next pre-registration, and it cannot happen again.
Doing it now, after seeing the result, would be tuning.

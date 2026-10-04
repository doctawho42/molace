# The roughness baseline, settled on targets the claim was not found on

Pre-registration: `prereg/increment3_chembl.yaml`, blob
`7519ef31c4374637483bae167221f9b7567ece7f`, frozen before a single activity row was downloaded.
Reproduce with `scripts/fetch_chembl_targets.py` then `scripts/rogi_chembl.py`; numbers in
`results/report_rogi_chembl.txt` and `results/rogi_chembl.csv`.

## What was open

Increment 2 confirmed that target assortativity predicts attainable accuracy, and could not show
that it adds anything over ROGI, the cheminformatics roughness index. Without that, the
contribution is a bridge between two fields rather than a capability: a reviewer would say the graph
statistic re-describes something chemistry already has.

Two attempts failed. The confirmatory set (DeepDelta, n = 10) had no power. The pooled set (n = 40)
was three quarters discovery set, so a positive result there is the claim reproducing on itself.
Two pre-registered explanations for the disagreement between them -- sample size, and the
absolute-versus-delta task -- were both refuted.

## What was done

Forty regression targets curated from ChEMBL by activity count: human single proteins, pchembl
values with an exact relation, the dominant assay type per target, deduplicated by canonical SMILES
with the median as the label. **Every ChEMBL id appearing in MoleculeACE was excluded**, so no
target is shared with the set the claim was found on.

The selection rule was frozen before any download and contains nothing an outcome could steer: rank
by count, exclude the discovery ids, take the top 40 with at least 200 unique compounds. No target
was dropped by the size rule, so the threshold did not shape the sample. Compounds per target run
583 to 8,205, median 1,984 -- larger than MoleculeACE's 615 to 4,000.

Everything downstream matches the earlier increments line for line: kNN at k = 10 on ECFP4
Tanimoto, the pointwise arm averaged over its three matched families, skill against predicting the
training mean, mean degree and task size residualised out, cluster bootstrap at 10,000 draws.

## Result

| | rho | 95% CI | |
|---|---|---|---|
| rank assortativity, incremental over ROGI | **+0.846** | [+0.678, +0.928] | excludes 0 |
| ROGI, incremental over rank assortativity | +0.098 | [−0.230, +0.417] | covers 0 |
| rank assortativity alone | +0.889 | [+0.750, +0.955] | excludes 0 |
| Newman target assortativity alone | +0.908 | [+0.793, +0.959] | excludes 0 |
| ROGI alone | −0.228 | [−0.540, +0.114] | covers 0 |

Exactly one direction excludes zero, which is what the frozen rule calls settled. **Target
assortativity carries information about attainable accuracy that the roughness index does not, and
the roughness index carries nothing beyond it.** On these forty targets ROGI alone does not predict
attainable accuracy at all, though its point estimate has the expected sign.

The mechanism is visible in the extremes rather than only in the aggregate. The worst target by
skill, CHEMBL614358_EC50 at −0.18, also has the lowest rank assortativity of the forty, 0.135. The
two best, at +0.68 and +0.67, have 0.841 and 0.850. ROGI shows no such ordering: 0.050 at the worst
target against 0.055 and 0.035 at the two best.

## What this does and does not license

**Does.** The project's headline is no longer a re-description. A reader can be told that the
statistic predicts how accurately a molecular regression dataset can be predicted at all, that this
was confirmed on third-party models, and that it is not something the cheminformatics roughness
index already provides -- the last part on forty targets chosen before anything was measured and
sharing nothing with the set where the claim was found.

**Does not.** Both measures are computed from the same ECFP4 fingerprint and the same label, and so
are the models whose skill is being predicted. The claim is therefore about prediction *in that
representation*: a label smooth over ECFP4 neighbourhoods is one that ECFP4-based models predict
well. That is not vacuous -- ROGI sees the identical fingerprint and the identical label and does
not capture it -- but it is not a representation-free statement, and a different fingerprint is an
experiment this does not run.

Nor is this a random sample of drug targets. Ranking by activity count selects the best-measured
proteins in ChEMBL, which are plausibly also the best-behaved ones. The direction of that bias is
unknown and nothing here bounds it.

One target of forty has negative skill, meaning the models lose to predicting the mean. It is kept,
because dropping it after seeing it is exactly what the pre-registration forbids.

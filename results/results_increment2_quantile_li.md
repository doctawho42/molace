# A candidate continuous label informativeness, and why it is redundant

Produced by `scripts/quantile_li_study.py` and `scripts/continuous_li.py`; numbers in
`results/report_quantile_li.txt`, `results/quantile_li.csv` and `results/report_continuous_li.txt`.
40 datasets on the frozen graph (ECFP4, Tanimoto, kNN k = 10).

## The obstacle, measured rather than asserted

LI = I(y_xi, y_eta) / H(y_xi). Substituting a continuous label breaks exactly one part.

**The numerator survives.** Mutual information is defined for continuous variables and is invariant
under a smooth invertible change of either argument.

**The denominator does not.** Differential entropy is not an entropy: h(cY) = h(Y) + log|c|. On
`CHEMBL2047_EC50`, one graph, changing only the label's scale: LI reads −0.039 at x0.5, −0.049 as
shipped, −0.068 at x2 and **−0.588** at x10. A pole where h = 0 always exists, because h is
continuous in log c and unbounded below.

**Where the pole is cannot be quoted.** On a tied sample the entropy estimate needs a constant to
replace zero neighbour distances, and the level moves several nats with it (measured: h from −0.22
to −7.67 as the constant goes from 1e-2 to 1e-15). What survives the constant is the *difference*
between two scales, which is exactly log c. An earlier version of this work quoted a pole location
to six figures; those figures were the constant.

**Two further obstacles, neither anticipated.** The marginal LI is defined over is degree-weighted,
so each molecule's label enters deg(v) times: the sample is tied by construction, at multiplicity
20 to 48 here, and a nearest-neighbour entropy is undefined on it. And the labels are not continuous
to begin with -- 667 molecules carry 193 distinct half-life values, 631 carry 411 distinct pEC50
values, because assays report to a fixed precision.

On `half_life_obach` the mutual-information estimate comes out **negative at every scale**, which no
mutual information is. That target's numerator is broken by its ties, and no LI value from it is
reportable; an earlier version of this work reported four.

## The candidate

Bin the label into **quantile** bins and apply the categorical measure. This does not fix the
objection the pre-registration raised -- the value still moves with the bin count -- but it answers
the two that make a binning arbitrary:

* the cut POSITIONS are not chosen, so there is no boundary to tune after seeing a result;
* the result is invariant under every strictly monotone relabelling, which is the units failure that
  invalidated this project's holdout, fixed by construction rather than by a transform chosen later.

One declared integer is left. Whether that is survivable is a question about ORDERINGS, since a
measure for comparing datasets does not need a stable value.

| | |
|---|---|
| per-dataset value range across bin counts | median 1.9x, max 5.3x |
| ordering, Spearman between adjacent bin counts | worst **+0.967** |
| ordering, worst pair anywhere (b = 2 against b = 32) | **+0.853** |

**The value moves and the ordering does not.** By the standard this measure exists to meet --
comparing datasets -- a quantile-binned LI with a declared bin count is usable.

## And it is redundant

| | rho against rank assortativity |
|---|---|
| LI_q2 | +0.961 |
| LI_q4 | +0.986 |
| LI_q8 | +0.984 |
| LI_q16 | +0.951 |
| LI_q32 | +0.874 |

It predicts attainable accuracy about as well as the continuous measure (+0.76 to +0.80 against
pointwise skill on the 30 MoleculeACE targets, all excluding zero) and adds nothing over it:
incrementally over rank assortativity, ROGI, mean degree and task size, +0.137, +0.098 and +0.065,
all covering zero.

The most telling line is the first. **LI_q2** -- above or below the median, a single binary split --
already correlates +0.961 with rank assortativity. The dependence between a molecule's label and its
neighbours' on these graphs is essentially monotone, and a monotone dependence is what the
continuous measure already captures.

## What this suggests, carefully

A continuous label informativeness may be absent not because it is hard to define but because, for a
continuous label, there is nothing left for it to add over the continuous measure that already
exists. Label informativeness earns its place on categorical labels by capturing dependence that a
correlation cannot see; on these graphs there is no such dependence to capture.

That is a claim about **molecular similarity graphs with continuous assay labels**, on 40 datasets,
not about continuous labels in general. A domain whose label-neighbour dependence is genuinely
non-monotone would be the place to look for the counterexample, and this work does not supply one.

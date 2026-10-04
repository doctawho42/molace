# How much of a homophily measure is the data, and how much is the graph you built

Produced by `scripts/construction_variance.py`; numbers in
`results/report_construction_variance.txt` and `results/construction_variance.csv`.
40 datasets (30 MoleculeACE targets, 10 DeepDelta benchmarks) x 10 constructions. No model is
fitted anywhere.

## Why it is worth measuring

Homophily measures are defined for a **given** graph. In molecular machine learning there is no
given graph: an analyst picks a fingerprint, a similarity and a neighbourhood rule, and the graph
appears. The same is true of much of GraphLand, where edges are constructed rather than observed.
So a number reported as a dataset's homophily is partly a property of the dataset and partly a
property of the choice, and the split is not reported anywhere.

The grid is **one factor at a time** around increment 1's frozen choice (ECFP4, Tanimoto, kNN at
k = 10): k in {5, 10, 20, 30}; fingerprint in {ECFP6, MACCS, RDKit, atom-pair} at k = 10; metric in
{Dice, cosine} at k = 10. Every alternative is a choice an analyst might defensibly have made.

This varies exactly what increment 1 froze, and the two disciplines are not in conflict: tuning is
choosing a construction because of what it does to your answer; this reports what every reasonable
choice does, before picking any.

## The split

| | target assortativity | rank assortativity |
|---|---|---|
| spread BETWEEN datasets (sd of each dataset's mean) | 0.1608 | 0.1539 |
| spread WITHIN a dataset (mean sd across constructions) | 0.0633 | 0.0624 |
| share of variance that is the dataset | **0.866** | **0.859** |
| per-dataset range across constructions, median | 0.2087 | 0.2047 |
| per-dataset range across constructions, max | 0.4041 | 0.4108 |
| full range of dataset means, for comparison | 0.6235 | 0.6024 |

Read the last three rows together. 87% of the variance being the dataset sounds reassuring, and
then the median dataset still moves **0.21** across constructions against a total between-dataset
spread of **0.62**. One dataset, holding its data fixed and changing only the fingerprint and k,
traverses a third of the whole scale. The worst case, DeepDelta's FreeSolv, moves 0.40.

## The ordering survives

Spearman of the 40-dataset ranking, frozen construction against each alternative:

| | rho |
|---|---|
| ECFP4 / Dice / k=10 | +1.000 |
| ECFP4 / cosine / k=10 | +0.997 |
| ECFP6 / Tanimoto / k=10 | +0.994 |
| ECFP4 / Tanimoto / k=20 | +0.992 |
| ECFP4 / Tanimoto / k=5 | +0.987 |
| ECFP4 / Tanimoto / k=30 | +0.983 |
| atom-pair / Tanimoto / k=10 | +0.942 |
| RDKit / Tanimoto / k=10 | +0.896 |
| MACCS / Tanimoto / k=10 | +0.864 |

Worst +0.864, median +0.987, and the rank version agrees line for line. Tanimoto against Dice is
+1.000 exactly: that choice carries no information at all.

## What to take

**A homophily number quoted without its construction is not a dataset property.** It can be moved a
third of the between-dataset range by choices nobody writes down.

**The ordering is a dataset property.** Across every alternative tried, the ranking of datasets by
homophily is stable to rho >= 0.86, and within the Morgan-like family to >= 0.94. Comparative
statements survive; absolute ones do not.

For a benchmark that reports homophily per dataset, the actionable version is: publish the
construction with the number, and make comparative claims rather than absolute ones.

## What this does not establish

Ten constructions around one centre, two collections of molecular data, one node-feature type. It
says nothing about graphs that are observed rather than built, and nothing about classification
measures -- unbiased homophily on the cliff flag was not swept here because the flag exists only on
the MoleculeACE half. A wider sweep could find a construction that breaks the ordering; none of
these did.

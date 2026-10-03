# molace

Does a parameter-free, pre-training statistic of a molecule-similarity graph predict when pairwise
models beat pointwise ones? Nodes are molecules; edges are comparisons between them.

## What this is

Increment 1 of a pet project. The design is in
`docs/superpowers/specs/2026-10-03-molace-increment1-design.md`; the frozen analysis plan is
`prereg/increment1.yaml` and every result and figure carries its git blob hash. Decisions taken during
execution, with what each costs if wrong, are in the implementation ledger and in the commit messages.

## What the audit refuted before any code was written

The original proposal is recorded in the spec's §2 together with these verdicts, so the claims cannot
quietly return:

- **Molecules-as-nodes with Tanimoto edges is not a new move.** Chemical Space Networks (Maggiora and
  Bajorath, 2014 onward) are exactly that construction, characterised with network measures including
  assortativity across 36 ChEMBL activity classes, and that literature controls edge density. This
  project works inside that paradigm and controls density.
- **No named pairwise architecture is curl-free.** SQRL applies a nonlinear MLP head to the difference
  of representations, PADRE is a random forest on concatenated features plus their difference,
  DeepDelta concatenates two message-passing embeddings, and RankRefine has no learned pairwise
  function at all. The theorem here is restated for a frozen encoder with a bias-free linear readout,
  where it is provable, and the real boundary is linear against nonlinear readout.
- **Triangle closure as a label-free consistency check is published.** Twin Neural Network Regression
  states and uses the loop condition and derives uncertainty from its violation; DeepDelta measures the
  additivity residual at 0.127 +/- 0.043 and uses it as an unsupervised quality signal. What is claimed
  here is the orthogonal three-way decomposition, the energy budget and the observability identity, not
  the observation.
- **Neither SQRL nor MoleculeACE publishes a per-target pointwise-against-pairwise gap.** The gap is
  computed here on MoleculeACE's own shipped split; DeepDelta's per-fold predictions supply the one
  genuinely published external anchor, on ten tasks.

Two further corrections: adjusted homophily and label informativeness are categorical-only, and the
group's own choice for a continuous target is Newman target assortativity, which carries no binning
parameter; and adjusted homophily was superseded by unbiased homophily in 2024, which is what their
current benchmarks report.

We do not reproduce SQRL's numbers. Its published equations are inconsistent: its inference rule under
its own training convention estimates twice the anchor label minus the query.

## What building it refuted in turn

Execution changed three claims the design was making. Each is recorded where it was made, not only here.

- **The harmonic component does not dominate.** The design expected it to run two to five times the
  gradient component, from a synthetic probe. On the three targets where the rank is exactly computable
  it runs at about a fifth of the gradient and makes up three and a half percent of the cycle space,
  which is almost entirely curl. Three targets, and the three smallest of thirty, so the statement is
  about those graphs; see `results/results_hodge_census.md`.
- **The cliff label is not built on a different relation from our graph.** The spec said it was. Read
  from the code that generated the shipped flags, one of the three similarity channels defining a cliff
  is Morgan radius-two Tanimoto, the same relation our graph uses. The circularity is real; what limits
  it is the threshold, and between 0.02% and 2.75% of our edges reach the similarity a cliff edge needs.
  See `docs/preconditions.md`.
- **The cost model was wrong.** Cross-validated selection in both arms is a twenty-hour sweep at measured
  fit times, so selection is removed and the primary averages matched learner families instead, which is
  also symmetric in a way selection was not. The pre-registration records this, amended before any
  measurement ran.

The roughness baseline ships as plain ROGI rather than ROGI-XD, which the design named: ROGI-XD's only
implementation needs a separate environment on an older Python with a CUDA-pinned torch, git submodules
and pretrained encoders. Recorded in the module, the pre-registration and here rather than substituted
silently.

## Reproducing

```bash
uv sync
./scripts/fetch_data.sh
uv run pytest
uv run python -c "from molace.analysis.sweep import run_sweep; run_sweep().to_csv('results/spine.csv', index=False)"
```

`uv sync` must keep the `rdkit-pypi` override in `pyproject.toml`: PyTDC depends on that deprecated
wheel, it installs over the same package directory, and its extension modules are built against the
previous NumPy generation, so every `import rdkit` segfaults once it lands.

## Results

`results/report_spine.txt`, `results/report_external.txt`, `results/report_holdout.txt`,
`results/report_hodge.txt`, `results/hodge_census.csv`, `results/results_hodge_census.md`, and
`results/figures/`.

## Known deviations from the spec

Recorded rather than hidden: spec §2 for what the audit refuted, the "Deviations From The Spec" section
of the implementation plan for the four taken at planning time, `docs/preconditions.md` for what
checking the text preconditions changed, and the commit messages for the rest.

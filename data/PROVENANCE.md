# Where the inputs come from, and what pins them

Every number in `results/` is a function of three input corpora. Until 2026-10-05 none of them was
pinned: a fresh checkout would fetch whatever upstream had that day and could not reproduce the
committed results. This file records what each input is and what now fixes it.

## MoleculeACE, 30 targets

Cloned from `github.com/molML/MoleculeACE`, pinned in `scripts/fetch_data.sh` to
**`7e6de0bd2968c56589c580f2a397f01c531ede26`** (2025-02-15). The script now checks out that exact
revision and refuses to continue if the working copy is at a different one. It must be cloned rather
than pip-installed: `Data/results/*` is absent from the package's `package_data`.

## DeepDelta, 10 benchmarks

Cloned from `github.com/RekerLab/DeepDelta`, pinned to
**`cd9b131559d54be639fcb9455e4c057a3dfd2c6b`** (2024-03-11), same mechanism.

## ChEMBL, 40 targets — the one that cannot be re-fetched

`scripts/fetch_chembl_targets.py` pages the live EBI REST API at
`https://www.ebi.ac.uk/chembl/api/data/activity.json` and then takes the top 40 targets by activity
count. Both the activity rows and the ranking move as the database grows, so a re-fetch on any later
date yields a *different collection*, not merely more rows of the same one. There is no release
parameter in the query and no release was recorded at fetch time, so the release the data came from
cannot now be recovered. The fetch date is 2026-10-04, from the files' own timestamps.

What pins it instead is the data itself:

- `data/chembl_targets.tar.gz` — the exact bytes every committed ChEMBL number was computed from,
  8.1 MB compressed, 60.4 MB extracted. `bash scripts/restore_chembl.sh` unpacks it to
  `data/raw/chembl_targets/` and verifies it.
- `data/chembl_manifest.csv` — per-file SHA-256, byte count and row count for all 43 files.
  `uv run python scripts/verify_chembl.py` checks a working copy against it and fails loudly on any
  mismatch. 739,407 activity rows across 40 targets, plus `selected.csv`.

`fetch_chembl_targets.py` is kept for provenance and for anyone who wants a fresh collection, but it
is no longer the path that reproduces the results. A collection it fetches today will not match the
manifest, and that is a property of the source rather than a bug.

## What is still not pinned

- **TDC**, used by `src/molace/data/tdc_admet.py`, downloads over the network at import. No result in
  `results/` currently depends on it.
- **The Python stack is pinned**: `uv.lock` fixes exact versions with hashes and `.python-version`
  fixes 3.11, so `uv sync` gives a deterministic numpy / scipy / scikit-learn / rdkit / pandas
  environment.

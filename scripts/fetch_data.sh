#!/usr/bin/env bash
# Clones the two upstream data repositories into data/raw (gitignored), AT PINNED REVISIONS.
# MoleculeACE must be CLONED, not pip-installed: Data/results/* is absent from package_data.
#
# The revisions are pinned because they are inputs to every committed number. Until 2026-10-05 this
# script did `git clone --depth 1` of each upstream default branch, so whatever HEAD happened to be
# on the day of the clone became the input, and a fresh checkout could not reproduce the results.
# These two are the revisions every result in results/ was computed from.
set -euo pipefail

MOLECULEACE_REV=7e6de0bd2968c56589c580f2a397f01c531ede26   # 2025-02-15
DEEPDELTA_REV=cd9b131559d54be639fcb9455e4c057a3dfd2c6b     # 2024-03-11

mkdir -p data/raw

clone_at() {                      # repo_url, target_dir, revision
  if [ ! -d "$2" ]; then
    git clone --no-checkout "$1" "$2"
    git -C "$2" checkout --detach "$3"
  fi
  have=$(git -C "$2" rev-parse HEAD)
  if [ "$have" != "$3" ]; then
    echo "ERROR: $2 is at $have, expected $3." >&2
    echo "       Remove it and re-run, or the inputs no longer match the committed results." >&2
    exit 1
  fi
  echo "$2 pinned at $3"
}

clone_at https://github.com/molML/MoleculeACE.git data/raw/MoleculeACE "$MOLECULEACE_REV"
clone_at https://github.com/RekerLab/DeepDelta.git data/raw/DeepDelta "$DEEPDELTA_REV"

echo "benchmark csvs: $(ls data/raw/MoleculeACE/MoleculeACE/Data/benchmark_data/*.csv | wc -l)"

# The 40-target ChEMBL collection is NOT re-fetched by default: it came from a live REST query whose
# result moves as the database grows. Restore the exact bytes every result used instead.
if [ ! -d data/raw/chembl_targets ]; then
  echo "data/raw/chembl_targets is missing; run: bash scripts/restore_chembl.sh"
fi

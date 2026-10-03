#!/usr/bin/env bash
# Clones the two upstream data repositories into data/raw (gitignored).
# MoleculeACE must be CLONED, not pip-installed: Data/results/* is absent from package_data.
set -euo pipefail
mkdir -p data/raw
[ -d data/raw/MoleculeACE ] || git clone --depth 1 https://github.com/molML/MoleculeACE.git data/raw/MoleculeACE
[ -d data/raw/DeepDelta ]   || git clone --depth 1 https://github.com/RekerLab/DeepDelta.git data/raw/DeepDelta
echo "benchmark csvs: $(ls data/raw/MoleculeACE/MoleculeACE/Data/benchmark_data/*.csv | wc -l)"

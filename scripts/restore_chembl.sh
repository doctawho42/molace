#!/usr/bin/env bash
# Restores the 40-target ChEMBL collection from the copy committed in this repository, and verifies
# every file against data/chembl_manifest.csv.
#
# Why a committed copy rather than a re-fetch: scripts/fetch_chembl_targets.py pages the live EBI
# REST API and then takes the top 40 targets by activity count. Both the activity rows and the
# ranking move as the database grows, so a re-fetch on any later date yields a different collection
# and every ChEMBL number in results/ becomes unreproducible. The fetch script is kept for
# provenance and for anyone who wants a fresh collection; this script is what reproduces the results.
set -euo pipefail

ARCHIVE=data/chembl_targets.tar.gz
TARGET=data/raw/chembl_targets

[ -f "$ARCHIVE" ] || { echo "ERROR: $ARCHIVE is missing." >&2; exit 1; }
mkdir -p data/raw
tar -xzf "$ARCHIVE" -C data/raw
echo "extracted to $TARGET"
uv run python scripts/verify_chembl.py

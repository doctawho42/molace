"""Verify the restored ChEMBL collection against the committed manifest.

The collection came from a live REST query, so it cannot be re-fetched reproducibly. What makes it
checkable instead is a per-file SHA-256: a checkout that passes this has byte-identical inputs to the
ones every committed ChEMBL number was computed from.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd

MANIFEST = Path("data/chembl_manifest.csv")
ROOT = Path("data/raw/chembl_targets")


def main() -> int:
    man = pd.read_csv(MANIFEST)
    bad, missing = [], []
    for r in man.itertuples():
        p = ROOT / r.file
        if not p.is_file():
            missing.append(r.file)
            continue
        if hashlib.sha256(p.read_bytes()).hexdigest() != r.sha256:
            bad.append(r.file)
    extra = sorted({p.name for p in ROOT.glob("*.csv")} - set(man.file))
    for label, items in (("missing", missing), ("checksum mismatch", bad), ("not in manifest", extra)):
        if items:
            print(f"{label}: {len(items)}")
            for i in items[:10]:
                print(f"  {i}")
    if missing or bad:
        print(f"\nFAILED: the inputs do not match the manifest, so results computed from them will "
              f"not match the committed ones.")
        return 1
    print(f"all {len(man)} files match data/chembl_manifest.csv "
          f"({man.bytes.sum() / 1048576:.1f} MB, {man[man.file != 'selected.csv'].rows.sum()} "
          f"activity rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

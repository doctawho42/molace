"""Access to the frozen analysis plan.

The plan is anchored by the git blob hash of the YAML, which is stamped into every result file
and every figure. A result whose stamp does not match the committed plan was produced under a
different plan, and that is visible rather than arguable.
"""
from __future__ import annotations

import functools
import subprocess
from pathlib import Path

import yaml

PREREG = Path(__file__).resolve().parents[3] / "prereg" / "increment1.yaml"


@functools.lru_cache(maxsize=1)
def load_prereg() -> dict:
    if not PREREG.is_file():
        raise FileNotFoundError(f"{PREREG} is missing; the plan must exist before measurement")
    with PREREG.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@functools.lru_cache(maxsize=1)
def prereg_fingerprint() -> str:
    """The git blob hash of the pre-registration file."""
    out = subprocess.run(
        ["git", "hash-object", str(PREREG)],
        capture_output=True, text=True, check=True, encoding="utf-8",
    )
    return out.stdout.strip()

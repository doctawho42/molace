"""Seeds must be pure functions of the declared grid, across processes.

scripts/field_profile.py once seeded its synthetic fields with `hash(family) % 9973`. Python
randomises `hash` on a str per process unless PYTHONHASHSEED is pinned, so the 600 rows of
results/field_profile.csv were a valid realisation of the design that could never be regenerated.
Every pre-registration in this project forbids post-hoc changes, which is worth nothing if the
numbers cannot be reproduced to check against.

Two tests: the property, end to end across real processes with different hash seeds, and a static
guard that catches the whole class rather than the one instance.
"""
from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GENERATORS = ("scripts/field_profile.py", "scripts/synthetic_decay.py")

PROBE = """
import importlib.util, numpy as np
spec = importlib.util.spec_from_file_location("gen", {path!r})
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
s = m.cell_seed({args})                       # the script's OWN seed construction, not the test's
rng = np.random.default_rng(s)
print(repr(tuple(int(v) for v in s)), float(rng.normal(size=8).sum()))
"""


def _probe(path: str, args: str, hash_seed: str) -> str:
    env = dict(os.environ, PYTHONHASHSEED=hash_seed)
    out = subprocess.run(
        [sys.executable, "-c", PROBE.format(path=str(ROOT / path), args=args)],
        capture_output=True, text=True, encoding="utf-8", check=True, cwd=ROOT, env=env,
    )
    return out.stdout.strip()


@pytest.mark.parametrize(
    "path, args",
    [
        ("scripts/field_profile.py", "'squared_exponential', 10, 3, 16, 0.3, 2"),
        ("scripts/synthetic_decay.py", "10, 3, 16, 0.3, 2"),
    ],
)
def test_a_cell_draws_the_same_field_under_a_different_hash_seed(path, args):
    """The end-to-end property: same grid cell, different PYTHONHASHSEED, same seed and realisation.

    Calls the script's own cell_seed, so a generator that goes back to hash() fails here even if it
    keeps the tuple's shape.
    """
    a = _probe(path, args, "0")
    b = _probe(path, args, "12345")
    assert a == b, f"{path} draws a different field when PYTHONHASHSEED changes: {a} != {b}"


@pytest.mark.parametrize("path", GENERATORS)
def test_no_randomised_hash_anywhere_in_a_generator(path):
    """The static guard: no call to hash() anywhere in a synthetic-data generator.

    Deliberately broader than "inside a default_rng() argument". The first version of this test
    checked only that narrower condition and silently stopped catching the bug the moment the seed
    was lifted into a helper function - the hash call was then one frame away and the guard passed.
    These scripts generate data from a declared grid and have no legitimate use for a per-process
    hash, so the honest rule is none at all.
    """
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    lines = [n.lineno for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "hash"]
    assert not lines, (
        f"{path} calls hash() at line(s) {lines}; hash() on a str is randomised per process, so "
        "anything derived from it cannot be regenerated. Use an explicit index into the declared "
        "grid instead."
    )

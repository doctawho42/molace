"""No module may reference a name that is not defined.

WHY THIS AND NOT A COMPILE GUARD. The bug that prompted this test was an f-string whose literal
brace pair had been turned into an expression:

    print(f"  {c:16s} over {rogi, modi_q2, SARI}: ")

`ast.parse`, `compile()` and `py_compile` ALL accept that -- it is syntactically a valid f-string
containing a tuple -- and it fails only when the line executes, as NameError. Measured, before this
test was written: all three accept it and only `exec` raises. So a compile-only guard would have been
useless against the very error it was proposed for, and the check has to be static NAME RESOLUTION.

ruff's F821 is that check. It is a declared dev dependency rather than an optional convenience,
because a guard that skips when its tool is missing reports PASS on a broken tree, which is the
defect this project has been bitten by in four consecutive increments.

What this found on its first run, before it was committed: scripts/synthetic_decay.py referenced
`cells`, defined nowhere in the file, inside a bootstrap loop. Two defects stacked -- an undefined
name, and the `set(rng.choice(...))` resampling pattern that analysis.correlate.cluster_groups
documents as the bug its own refactor replaced. It sat behind a design gate that failed in the
committed run, so it never executed and no published number was affected.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHECKED = ("src", "scripts", "tests")
RULE = "F821"


def _ruff() -> str:
    """The ruff executable. A missing tool is a FAILURE, never a skip."""
    for cand in (Path(sys.prefix) / "bin" / "ruff", Path(sys.prefix) / "Scripts" / "ruff.exe"):
        if cand.is_file():
            return str(cand)
    found = shutil.which("ruff")
    if found:
        return found
    pytest.fail(
        "ruff is not installed, so this guard cannot run. It is declared in pyproject.toml's dev "
        "group; run `uv sync --group dev`. This fails rather than skips because a skipped guard "
        "reports PASS on a tree it never examined.")


def _run(*paths: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [_ruff(), "check", "--isolated", "--no-cache", "--select", RULE,
         "--output-format", "concise", *paths],
        capture_output=True, text=True, encoding="utf-8", cwd=ROOT)


def test_the_guard_itself_detects_an_undefined_name(tmp_path):
    """A positive control, so the guard cannot silently become vacuous.

    If ruff renames or retires F821, or if the invocation stops selecting it, this fails instead of
    the real test passing on an empty findings list. It uses the exact shape of the bug that
    prompted the guard, which no compile-based check flags.
    """
    bad = tmp_path / "bug.py"
    bad.write_text('c = "x"\nprint(f"  {c:16s} over {rogi, modi_q2, SARI}: ")\n', encoding="utf-8")
    r = _run(str(bad))
    assert RULE in r.stdout, (
        f"the guard did not flag a known undefined name, so it is not testing anything. "
        f"ruff said: {r.stdout or r.stderr!r}")


def test_no_module_references_an_undefined_name():
    r = _run(*CHECKED)
    # 0 is clean and 1 is findings; anything else means ruff itself failed, which must not read as a
    # pass. Emptiness of stdout is NOT the signal: ruff prints "All checks passed!" on a clean tree
    # even in concise output, so a test asserting empty stdout fails on success, as the first version
    # of this one did.
    assert r.returncode in (0, 1), f"ruff failed to run (exit {r.returncode}): {r.stderr!r}"
    assert r.returncode == 0, (
        f"{RULE} findings under {', '.join(CHECKED)}:\n" + r.stdout.rstrip()
        + "\n\nAn undefined name is not caught by ast.parse, compile() or py_compile; it raises "
          "NameError only when the line executes, which may be behind a gate that rarely fires.")

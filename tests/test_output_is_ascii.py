"""Nothing a script prints may be non-ASCII.

scripts/modi_baseline.py printed its whole analysis in Russian. On a stdout whose encoding is not
UTF-8 - a LANG=en_US.ISO8859-1 locale, or any PYTHONIOENCODING that is not UTF-8 - the first such
print raises UnicodeEncodeError. That script wrote results/modi_baseline.csv only after its
per-target loop, so the crash lost the measurement as well as the report, and it would have hit the
first person to run this repository under a different locale rather than anyone here.

Reproduced before fixing:
    PYTHONIOENCODING=iso-8859-1 python -c "print('\\u0442\\u043e\\u0436\\u0434\\u0435\\u0441\\u0442\\u0432\\u043e')"
    UnicodeEncodeError: 'latin-1' codec can't encode characters in position 0-8

Comments and docstrings are exempt: they are never encoded to stdout, and four files legitimately
carry a section sign or an em dash in prose.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCES = sorted((ROOT / "scripts").glob("*.py")) + sorted((ROOT / "src").rglob("*.py"))


def _printed_literals(tree: ast.AST) -> list[tuple[int, str]]:
    """Every string constant that reaches a print(), including inside f-strings."""
    out = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "print"):
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.Constant) and isinstance(inner.value, str):
                out.append((getattr(inner, "lineno", node.lineno), inner.value))
    return out


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: str(p.relative_to(ROOT)))
def test_no_script_prints_a_non_ascii_literal(path):
    offenders = [(ln, s) for ln, s in _printed_literals(ast.parse(path.read_text(encoding="utf-8")))
                 if not s.isascii()]
    assert not offenders, (
        f"{path.relative_to(ROOT)} prints non-ASCII at line(s) "
        f"{[ln for ln, _ in offenders]}; this raises UnicodeEncodeError wherever stdout is not UTF-8"
    )


def test_the_check_would_catch_a_cyrillic_print(tmp_path):
    """The guard must fail on the thing it is guarding against, or it guards nothing."""
    p = tmp_path / "bad.py"
    p.write_text('print(f"  identity={1.0:.3f}  тождество")\n', encoding="utf-8")
    offenders = [(ln, s) for ln, s in _printed_literals(ast.parse(p.read_text(encoding="utf-8")))
                 if not s.isascii()]
    assert offenders, "the AST walk must see string constants inside an f-string in a print call"

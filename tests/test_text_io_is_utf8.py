"""Text file I/O must name its encoding.

open(), Path.open(), Path.read_text() and Path.write_text() default to
locale.getpreferredencoding(False), not UTF-8, so on a machine whose locale is not UTF-8 they read
and write different bytes than they do here. Python source is defined to be UTF-8, the YAML
pre-registrations and the JSON measurement cache are written as UTF-8, and nothing in this project
wants a locale to decide how they are decoded.

Verified reproducer on macOS: LC_ALL=en_US.ISO8859-1 makes
locale.getpreferredencoding(False) return ISO8859-1. LC_ALL=C does NOT - CPython coerces it to UTF-8
under PEP 538 - so `C` is not a reproducer and the latin-1 locale is the one to test with.

Every file these call sites touch is pure ASCII today, so this was a latent hazard rather than a
live break, with one exception already fixed: tests/test_output_is_ascii.py writes Cyrillic, and
without an explicit encoding that write raised UnicodeEncodeError under the latin-1 locale.

pandas is deliberately NOT covered: pandas.io.common.get_handle does `encoding = encoding or "utf-8"`,
so read_csv and to_csv are already UTF-8 on every locale regardless of the signature default.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "scripts").glob("*.py"))
           + sorted((ROOT / "tests").glob("*.py")))

TEXT_METHODS = {"read_text", "write_text"}
BINARY_METHODS = {"read_bytes", "write_bytes"}


def _offenders(tree: ast.AST) -> list[tuple[int, str]]:
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        kwargs = {kw.arg for kw in node.keywords}
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)

        if name in BINARY_METHODS:
            continue
        if name in TEXT_METHODS and "encoding" not in kwargs:
            out.append((node.lineno, f"{name}() without encoding="))
        elif name == "open" and "encoding" not in kwargs:
            # Binary mode needs no encoding. The mode is positional argument 1 for the builtin
            # open(file, mode) but argument 0 for Path.open(mode), so the two cannot share an index:
            # reading the first string constant instead picks up the FILENAME and calls
            # open("e", "rb") a text-mode offender, which is what the first version of this did.
            at = 0 if isinstance(fn, ast.Attribute) else 1
            mode = node.args[at].value if (len(node.args) > at
                                           and isinstance(node.args[at], ast.Constant)) else None
            if mode is None:
                kw = next((k for k in node.keywords if k.arg == "mode"), None)
                mode = kw.value.value if (kw is not None
                                          and isinstance(kw.value, ast.Constant)) else None
            if not (isinstance(mode, str) and "b" in mode):
                out.append((node.lineno, "open() in text mode without encoding="))
        elif name == "run" and "text" in kwargs and "encoding" not in kwargs:
            out.append((node.lineno, "subprocess.run(text=True) without encoding="))
    return out


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: str(p.relative_to(ROOT)))
def test_no_text_io_relies_on_the_locale(path):
    bad = _offenders(ast.parse(path.read_text(encoding="utf-8")))
    assert not bad, (
        f"{path.relative_to(ROOT)} leaves the encoding to the locale at: "
        + "; ".join(f"line {ln}: {what}" for ln, what in bad)
    )


def test_the_check_sees_each_form_it_claims_to(tmp_path):
    """The guard must fail on what it guards against, and pass what is legitimately exempt."""
    p = tmp_path / "sample.py"
    p.write_text(
        "from pathlib import Path\n"
        "import subprocess\n"
        "Path('a').read_text()\n"                      # 3: offender
        "Path('b').write_text('x')\n"                  # 4: offender
        "open('c')\n"                                  # 5: offender
        "subprocess.run(['x'], text=True)\n"           # 6: offender
        "Path('d').read_bytes()\n"                     # exempt
        "open('e', 'rb')\n"                            # exempt
        "Path('f').read_text(encoding='utf-8')\n"      # exempt
        "Path('g').open('rb')\n"                       # exempt: Path.open puts mode first
        "Path('h').open()\n",                          # 11: offender
        encoding="utf-8")
    lines = [ln for ln, _ in _offenders(ast.parse(p.read_text(encoding="utf-8")))]
    assert lines == [3, 4, 5, 6, 11], lines

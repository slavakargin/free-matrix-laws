"""Guard against Markdown/MathJax pitfalls in docstrings and how-to pages.

The docs are built with mkdocstrings + pymdownx.arithmatex.  Arithmatex only
recognizes a ``$$ ... $$`` display block when it is a paragraph of its own,
i.e. separated from the surrounding text by blank lines.  Otherwise the
block is left to Markdown, which turns ``a^* ... b^*`` into emphasis,
collapses ``\\\\`` to ``\\`` and eats ``_..._``, and the formula renders as
garbage.  Sphinx roles such as ``:func:`name``` are not understood by
mkdocstrings either; use ``[`name`][free_matrix_laws.module.name]``.
"""
import ast
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCUMENTED_MODULES = ["brown", "transforms", "quadrature", "singularity",
                      "opvalued", "ensembles"]
MD_FILES = sorted((ROOT / "docs").glob("**/*.md"))


def _docstrings(path):
    tree = ast.parse(path.read_text())
    nodes = [tree] + [n for n in ast.walk(tree)
                      if isinstance(n, (ast.FunctionDef, ast.ClassDef,
                                        ast.AsyncFunctionDef))]
    for node in nodes:
        doc = ast.get_docstring(node, clean=True)
        if doc:
            yield getattr(node, "name", path.stem), doc


def _unseparated_display_math(text):
    """Return line numbers of $$-blocks not separated by blank lines."""
    lines = text.split("\n")
    bad, i = [], 0
    while i < len(lines):
        s = lines[i].strip()
        if not s.startswith("$$"):
            i += 1
            continue
        j = i if (len(s) > 2 and s.endswith("$$")) else i + 1
        while j < len(lines) and not lines[j].rstrip().endswith("$$"):
            j += 1
        if i > 0 and lines[i - 1].strip():
            bad.append(i + 1)
        if j + 1 < len(lines) and lines[j + 1].strip():
            bad.append(j + 2)
        i = j + 1
    return bad


ROLE = re.compile(r":(?:func|class|meth|mod|data|attr):`")


@pytest.mark.parametrize("module", DOCUMENTED_MODULES)
def test_docstrings_markdown(module):
    path = ROOT / "src" / "free_matrix_laws" / f"{module}.py"
    problems = []
    for name, doc in _docstrings(path):
        for ln in _unseparated_display_math(doc):
            problems.append(f"{name}: $$ block needs blank lines (docstring line {ln})")
        if ROLE.search(doc):
            problems.append(f"{name}: Sphinx role; use a mkdocstrings cross-reference")
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("path", MD_FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_markdown_pages(path):
    text = path.read_text()
    bad = _unseparated_display_math(text)
    assert not bad, f"$$ block needs blank lines at lines {bad}"
    assert not ROLE.search(text)

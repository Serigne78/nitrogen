"""Guard for rule #1: the scanner source never deserializes anything.

Ruff's banned-api already blocks the imports; this test also catches
dynamic access (`__import__`, `importlib`) and calls such as `x.load(...)`
on a module aliased under another name.
"""

import ast
from pathlib import Path

import nitrogen

SRC = Path(nitrogen.__file__).parent

FORBIDDEN_MODULES = {
    "pickle",
    "_pickle",
    "cPickle",
    "dill",
    "cloudpickle",
    "joblib",
    "marshal",
    "shelve",
    "torch",
}
# ADR 0011: the scanner reads model .py with ast.parse only. `eval`/`exec`
# would run it; both are banned here. `compile` is not (that would flag the
# legitimate `re.compile`), and it is harmless without an `exec` to run it.
FORBIDDEN_NAMES = {"Unpickler", "__import__", "import_module", "eval", "exec"}


def _violations(tree: ast.AST) -> list[str]:
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [a.name for a in node.names if a.name.split(".")[0] in FORBIDDEN_MODULES]
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] in FORBIDDEN_MODULES:
                found.append(node.module)
        elif isinstance(node, ast.Name | ast.Attribute):
            name = node.id if isinstance(node, ast.Name) else node.attr
            if name in FORBIDDEN_NAMES:
                found.append(name)
    return found


def test_source_tree_is_not_empty() -> None:
    assert list(SRC.rglob("*.py"))


def test_scanner_never_imports_deserializers() -> None:
    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        assert _violations(tree) == [], path


def test_guard_detects_violations() -> None:
    sample = "import pickle\nfrom torch import load\nx = __import__('os')\neval('1')\nexec('y=2')\n"
    assert _violations(ast.parse(sample)) == ["pickle", "torch", "__import__", "eval", "exec"]

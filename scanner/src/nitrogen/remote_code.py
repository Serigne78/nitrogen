"""Remote-code analysis (step 10, ADR 0011).

A model can ship executable Python that runs at load time under
`trust_remote_code`: `.py` files wired in through `auto_map` / `custom_pipelines`
in the JSON configs. The pickle analysis never sees this vector.

We read the configs as JSON (data, never code) and the `.py` files with
`ast.parse` only — a scanned file is NEVER imported, executed, compiled-and-run
or `importlib`-ed (rule 1).

Fail-closed (reviewer, ADR 0011 amended): the verdict never rides on a fragile
string match. A config that *declares* local custom code is a `risk` on its own,
whether or not we can locate the exact `.py` — the user who passes
`trust_remote_code=True` runs it either way. Locating the file, following the
config's directory, matching case: all only enrich the message. The AST walk is
a best-effort denylist; a referenced `.py`, or a config declaring custom code,
is a risk even when the AST looks clean.
"""

from __future__ import annotations

import ast
import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path, PurePosixPath

from nitrogen.findings import BoundedFindings, Finding
from nitrogen.inventory import FileEntry

_KNOWN_CONFIGS = frozenset(
    {
        "config.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
        "feature_extractor_config.json",
        "processor_config.json",
        "image_processor_config.json",
    }
)
_REF_KEYS = ("auto_map", "custom_pipelines")

# Bounded reads: a crafted config or source must not make us allocate without
# limit. Real configs are a few KB; a real modeling file a few tens of KB. Over
# the cap is itself a finding, never a silently truncated parse.
_MAX_CONFIG_BYTES = 8 * 1024 * 1024
_MAX_SOURCE_BYTES = 4 * 1024 * 1024
_MAX_DEPTH = 64  # bound the walk of a parsed config (no stack overflow)

# Sensitive calls the AST flags (ADR 0011). Exact dotted names, or a prefix for
# a whole module's API. A denylist: code is too varied for an allowlist.
_DANGEROUS_EXACT = frozenset(
    {
        "eval", "exec", "compile", "__import__",
        "builtins.eval", "builtins.exec", "builtins.compile", "builtins.__import__",
        "os.system", "os.popen",
        "urllib.request.urlopen", "urllib.urlopen",
        "pickle.loads", "pickle.load",
        "base64.b64decode", "base64.decodebytes",
        "importlib.import_module",
    }
)  # fmt: skip
_DANGEROUS_PREFIX = ("subprocess.", "socket.", "requests.", "ctypes.", "os.exec", "os.spawn")
# Modules rarely legitimate in model code: importing them (even aliased) is a
# signal in itself.
_DANGEROUS_IMPORTS = frozenset({"subprocess", "socket", "ctypes", "pty", "requests"})


def analyze_remote_code(entries: list[FileEntry], root: Path) -> list[FileEntry]:
    """Add remote-code findings: auto_map wiring on the configs, executable-code
    findings on the `.py` files. Never runs any code."""
    base = root if root.is_dir() else root.parent
    present = {str(e.path).casefold() for e in entries}  # case-insensitive FS safe
    size_of = {e.path: e.size for e in entries}
    extra: dict[PurePosixPath, list[Finding]] = defaultdict(list)
    referenced: set[str] = set()  # casefolded relative paths of located .py

    for entry in entries:
        if not _is_config(entry.path):
            continue
        if size_of[entry.path] > _MAX_CONFIG_BYTES:
            # Padding a config past the cap must not hide an auto_map. A config
            # this large is abnormal, and we cannot rule out custom code.
            extra[entry.path].append(
                Finding(
                    "remote_code.config_unreadable", "config too large to analyze safely", "risk"
                )
            )
            continue
        data = _read_json(base / entry.path, size_of[entry.path])
        if data is None:
            # Malformed JSON has no execution path: transformers fails to parse
            # it too, so nothing runs (ADR 0008). Not flagged — only the
            # oversized case above, valid for transformers but not for us.
            continue
        for ref in _collect_refs(data):
            if "--" in ref:
                extra[entry.path].append(
                    Finding(
                        "remote_code.external", f"auto_map references another repo: {ref}", "risk"
                    )
                )
                continue
            module = ref.rsplit(".", 1)[0] if "." in ref else ref
            located = _locate(module, entry.path.parent, present)
            referenced |= located
            where = f" -> {sorted(located)[0]}" if located else " (declared, file not located)"
            # Fail-closed: the declaration of local custom code is the danger,
            # located or not.
            extra[entry.path].append(
                Finding("remote_code.auto_map", f"config wires custom code: {ref}{where}", "risk")
            )

    for entry in entries:
        if entry.path.suffix.casefold() != ".py":
            continue
        is_referenced = str(entry.path).casefold() in referenced
        extra[entry.path].append(
            Finding(
                "remote_code.python_file",
                "auto-loaded via trust_remote_code"
                if is_referenced
                else "executable Python file present (not auto-wired)",
                "risk" if is_referenced else "warning",
            )
        )
        if size_of[entry.path] > _MAX_SOURCE_BYTES:
            extra[entry.path].append(
                Finding("remote_code.python_file", "Python file too large to analyze", "warning")
            )
            continue
        source = _read_text(base / entry.path, size_of[entry.path])
        if source is None:
            extra[entry.path].append(
                Finding(
                    "remote_code.python_file",
                    "Python file could not be decoded: not analyzed",
                    "warning",
                )
            )
        else:
            extra[entry.path].extend(scan_python_source(source))

    return [
        replace(e, findings=e.findings + tuple(extra[e.path])) if e.path in extra else e
        for e in entries
    ]


def _locate(module: str, config_dir: PurePosixPath, present: set[str]) -> set[str]:
    """Casefolded relative paths of the `.py` a reference resolves to, relative
    to the config's own directory (transformers resolves auto_map from there)."""
    stem = (config_dir / module.replace(".", "/")).as_posix()
    return {c.casefold() for c in (f"{stem}.py", f"{stem}/__init__.py") if c.casefold() in present}


def scan_python_source(source: str) -> list[Finding]:
    """Dangerous calls/imports in Python source, read with ast.parse only. An
    unparseable source yields no finding; it never raises."""
    try:
        tree = ast.parse(source)
    except Exception:
        return []
    aliases: dict[str, str] = {}  # local name -> real top module (follows `as`)
    dangerous_locals: set[str] = set()  # names bound to a dangerous callable
    findings = BoundedFindings()

    for node in ast.walk(tree):  # pass 1: imports and the names they bind
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                aliases[alias.asname or root] = root
                if root in _DANGEROUS_IMPORTS:
                    findings.append(_call_finding(f"import {alias.name}", node))
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                member = f"{node.module}.{alias.name}"
                if _is_dangerous(member) or node.module.split(".")[0] in _DANGEROUS_IMPORTS:
                    dangerous_locals.add(alias.asname or alias.name)
                    findings.append(_call_finding(f"from {node.module} import {alias.name}", node))

    for node in ast.walk(tree):  # pass 2: calls, resolved through the aliases
        if isinstance(node, ast.Call):
            dotted = _dotted_name(node.func)
            if dotted is None:
                continue
            resolved = _resolve(dotted, aliases)
            if (
                _is_dangerous(resolved)
                or dotted in dangerous_locals
                or resolved in dangerous_locals
            ):
                findings.append(_call_finding(resolved, node))
    return findings.result()


def _call_finding(name: str, node: ast.AST) -> Finding:
    return Finding(
        "remote_code.dangerous_call", f"{name} (line {getattr(node, 'lineno', 0)})", "risk"
    )


def _resolve(dotted: str, aliases: dict[str, str]) -> str:
    head, _, tail = dotted.partition(".")
    base = aliases.get(head, head)
    return f"{base}.{tail}" if tail else base


def _is_dangerous(dotted: str) -> bool:
    return dotted in _DANGEROUS_EXACT or dotted.startswith(_DANGEROUS_PREFIX)


def _dotted_name(node: ast.expr) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def _is_config(path: PurePosixPath) -> bool:
    name = path.name.casefold()
    return name.endswith(".json") and ("config" in name or name in _KNOWN_CONFIGS)


def _collect_refs(data: object, depth: int = 0) -> list[str]:
    if depth > _MAX_DEPTH:
        return []
    refs: list[str] = []
    if isinstance(data, dict):
        for key, value in data.items():
            if key in _REF_KEYS:
                refs.extend(_strings(value, depth + 1))
            else:
                refs.extend(_collect_refs(value, depth + 1))
    elif isinstance(data, list):
        for item in data:
            refs.extend(_collect_refs(item, depth + 1))
    return refs


def _strings(value: object, depth: int = 0) -> list[str]:
    if depth > _MAX_DEPTH:
        return []
    if isinstance(value, str):
        return [value] if ("." in value or "--" in value) else []
    if isinstance(value, dict):
        return [s for v in value.values() for s in _strings(v, depth + 1)]
    if isinstance(value, list):
        return [s for v in value for s in _strings(v, depth + 1)]
    return []


def _read_json(path: Path, size: int) -> object | None:
    try:
        with path.open("rb") as fh:
            raw = fh.read(min(size, _MAX_CONFIG_BYTES) + 1)  # bounded: never loads a huge file
        parsed: object = json.loads(raw)
        return parsed
    except (OSError, ValueError, RecursionError):
        return None


def _read_text(path: Path, size: int) -> str | None:
    try:
        with path.open("rb") as fh:
            raw = fh.read(min(size, _MAX_SOURCE_BYTES) + 1)
    except OSError:
        return None
    for encoding in ("utf-8", "latin-1"):  # latin-1 fallback: a `# coding:` trick still parses
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return None

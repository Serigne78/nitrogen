"""Chat-template analysis (step 11, ADR 0013).

Chat templates are Jinja2, rendered at inference to format the conversation. A
poisoned template escapes the sandbox and runs code (CVE-2024-34359). We parse
each template with jinja2's OWN parser (`Environment().parse()`) and walk the
AST — we NEVER render or compile it (rule 1).

Fail-closed and not a dodgeable denylist (reviewer): any dunder/gadget string
is flagged wherever it appears (so `map('attr','__class__')`, a `{% set %}`
variable, an item key all count), computed keys are constant-folded
(`'__cla'+'ss__'`), the `generation` tag transformers adds is parsed (not a
blind spot), and a file too large to read is a finding, never a silent skip.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import replace
from pathlib import Path, PurePosixPath

import jinja2
from jinja2 import nodes
from jinja2.ext import Extension
from jinja2.parser import Parser

from nitrogen.findings import BoundedFindings, Finding
from nitrogen.inventory import FileEntry

# tokenizer_config.json can legitimately reach a few MB (inline added tokens);
# match the remote_code config cap. Over it, we cannot rule out a hidden
# template -> a finding, never a silent miss. A real template itself is a few KB.
_MAX_FILE_BYTES = 8 * 1024 * 1024
_LONG_TEMPLATE = 50_000

# Introspection attributes a chat template never legitimately touches; reaching
# one is the sandbox-escape vector (CVE-2024-34359). Any dunder counts, plus
# frame / coroutine / traceback gadgets not spelled with leading/trailing "__".
_GADGET_ATTRS = frozenset(
    {
        "func_globals", "gi_frame", "gi_code", "gi_yieldfrom",
        "cr_frame", "cr_await", "ag_frame",
        "f_globals", "f_builtins", "f_locals", "f_back",
        "tb_frame", "tb_next",
    }
)  # fmt: skip
_URL = re.compile(r"https?://", re.IGNORECASE)
_BASE64_RUN = re.compile(r"[A-Za-z0-9+/]{64,}={0,2}")

_JSON_FILES = frozenset({"tokenizer_config.json", "chat_template.json"})


class _GenerationExtension(Extension):
    """Parse transformers' `{% generation %}...{% endgeneration %}` block as a
    no-op wrapper, so a real template is not a false `unparseable` and an escape
    hidden inside it is still walked (reviewer I2). Mirrors what the loader sees."""

    tags = {"generation"}  # noqa: RUF012 - jinja2 Extension API expects a set here

    def parse(self, parser: Parser) -> nodes.Node | list[nodes.Node]:
        next(parser.stream)  # consume 'generation'
        return parser.parse_statements(("name:endgeneration",), drop_needle=True)


# Parse-only environment, created once. We never call .render()/.compile().
_ENV = jinja2.Environment(extensions=[_GenerationExtension])  # noqa: S701 - never renders


def analyze_chat_templates(entries: list[FileEntry], root: Path) -> list[FileEntry]:
    """Add chat-template findings. Templates are parsed, never rendered."""
    base = root if root.is_dir() else root.parent
    extra: dict[PurePosixPath, list[Finding]] = defaultdict(list)
    for entry in entries:
        kind = _candidate_kind(entry.path)
        if kind is None:
            continue
        if entry.size > _MAX_FILE_BYTES:
            # Too large to read fully: a hidden template cannot be ruled out.
            extra[entry.path].append(
                Finding(
                    "chat_template.oversized", "file too large to analyze — review manually", "risk"
                )
            )
            continue
        for template in _templates_in(base, entry, kind):
            extra[entry.path].extend(scan_template(template))
    return [
        replace(e, findings=e.findings + tuple(extra[e.path])) if e.path in extra else e
        for e in entries
    ]


def scan_template(source: str) -> list[Finding]:
    """Findings for one Jinja template, parsed with jinja2 and never rendered.
    An unparseable template yields a finding; it never raises."""
    findings = BoundedFindings()
    if len(source) > _LONG_TEMPLATE:
        findings.append(
            Finding(
                "chat_template.obfuscated",
                f"template abnormally long ({len(source)} chars)",
                "warning",
            )
        )
    try:
        tree = _ENV.parse(source)
    except Exception as exc:
        findings.append(
            Finding(
                "chat_template.unparseable",
                f"template not parseable by jinja2 ({type(exc).__name__})",
                "warning",
            )
        )
        return findings.result()

    # A dunder/gadget reached by attribute access, by a (folded) computed key,
    # or named in ANY string constant (attr()/map()/selectattr args, set vars).
    for getattr_node in tree.find_all(nodes.Getattr):
        if _dangerous(getattr_node.attr):
            findings.append(_escape(f"attribute access {getattr_node.attr}"))
    for getitem in tree.find_all(nodes.Getitem):
        if not isinstance(getitem.arg, nodes.Const):  # literal keys caught by the Const walk
            key = _as_const(getitem.arg)
            if isinstance(key, str) and _dangerous(key):
                findings.append(_escape(f"computed item key {key!r}"))
    for const in tree.find_all(nodes.Const):
        if isinstance(const.value, str) and _dangerous(const.value):
            findings.append(_escape(f"dunder/gadget string {const.value!r}"))
    for filter_node in tree.find_all(nodes.Filter):
        if filter_node.name == "attr":
            findings.append(_escape("attr() filter: dynamic attribute access"))

    for text in _literals(tree):
        if _URL.search(text):
            findings.append(
                Finding("chat_template.hidden_instructions", "template embeds a URL", "warning")
            )
        if _BASE64_RUN.search(text):
            findings.append(
                Finding(
                    "chat_template.obfuscated", "long base64-like string in template", "warning"
                )
            )
    return findings.result()


def _escape(detail: str) -> Finding:
    return Finding("chat_template.sandbox_escape", detail, "risk")


def _dangerous(name: str) -> bool:
    return (name.startswith("__") and name.endswith("__")) or name in _GADGET_ATTRS


def _as_const(node: nodes.Expr) -> object:
    """Fold a constant expression (`'__cla' + 'ss__'`), or None if not constant."""
    try:
        return node.as_const()
    except nodes.Impossible:
        return None


def _literals(tree: nodes.Template) -> list[str]:
    out = [n.data for n in tree.find_all(nodes.TemplateData) if isinstance(n.data, str)]
    out += [n.value for n in tree.find_all(nodes.Const) if isinstance(n.value, str)]
    return out


def _candidate_kind(path: PurePosixPath) -> str | None:
    name = path.name.casefold()
    if name == "chat_template.jinja":
        return "jinja"
    if name in _JSON_FILES:
        return "json"
    return None


def _templates_in(base: Path, entry: FileEntry, kind: str) -> list[str]:
    if kind == "jinja":
        text = _read_text(base / entry.path, entry.size)
        return [text] if text is not None else []
    return _extract(_read_json(base / entry.path, entry.size))


def _extract(data: object) -> list[str]:
    """`chat_template` from a config: a string, or a list of {name, template}."""
    if not isinstance(data, dict):
        return []
    value = data.get("chat_template")
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if isinstance(item, dict) and isinstance(item.get("template"), str):
                out.append(item["template"])
            elif isinstance(item, str):
                out.append(item)
        return out
    return []


def _read_json(path: Path, size: int) -> object | None:
    try:
        with path.open("rb") as fh:
            raw = fh.read(min(size, _MAX_FILE_BYTES) + 1)
        parsed: object = json.loads(raw)
        return parsed
    except (OSError, ValueError, RecursionError):
        return None


def _read_text(path: Path, size: int) -> str | None:
    try:
        with path.open("rb") as fh:
            raw = fh.read(min(size, _MAX_FILE_BYTES) + 1)
    except OSError:
        return None
    for encoding in ("utf-8", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return None

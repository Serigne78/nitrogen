"""Minimal YAML policy (ADR 0005): `allow_formats` only, an allowlist.

    allow_formats: [safetensors, unknown]   # "safetensors only"

Every format of every file must be allowed; a file with no recognised format
carries the pseudo-format `unknown`, which must be allowed by name. Without
`allow_formats`, no format restriction applies.

Read with a SafeLoader that also refuses duplicate keys: the policy may come
from a pull request, and a bad merge must not quietly widen it. Anything
unexpected (unknown key or format, wrong type, invalid YAML, oversized file)
is a PolicyError, hence exit code 2, never silently ignored.
"""

import dataclasses
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from nitrogen.findings import Finding
from nitrogen.formats import Format
from nitrogen.inventory import FileEntry

# A policy is a few lines; anything bigger is a mistake.
MAX_POLICY_BYTES = 64 * 1024
UNKNOWN = "unknown"  # pseudo-format: no supported loader reads the file
FORMAT_NAMES = frozenset({f.value for f in Format} | {UNKNOWN})
# Severities a policy may escalate: never "risk" (always blocking already).
ESCALATABLE = frozenset({"warning", "info"})
_KEYS = frozenset({"allow_formats", "fail_on", "allow"})


class PolicyError(Exception):
    """The policy file cannot be applied as written."""


@dataclass(frozen=True)
class Policy:
    allow_formats: frozenset[str] | None = None  # None: no format restriction
    fail_on: frozenset[str] = frozenset()  # severities made blocking (e.g. warning)
    allow: frozenset[str] = frozenset()  # finding rules to neutralise
    sha256: str | None = None  # of the policy file, recorded in the AI-BOM


class _StrictLoader(yaml.SafeLoader):
    """SafeLoader (no Python tags) that also refuses duplicate keys."""


def _construct_mapping(loader: _StrictLoader, node: yaml.MappingNode) -> dict[Any, Any]:
    keys = set()
    for key_node, _value in node.value:
        key = loader.construct_object(key_node)
        try:
            if key in keys:
                raise PolicyError(f"duplicate key {key!r} in policy")
            keys.add(key)
        except TypeError as exc:  # a list or dict used as a key
            raise PolicyError("policy keys must be plain names") from exc
    return loader.construct_mapping(node)


_StrictLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping,
)


def load_policy(path: Path) -> Policy:
    with path.open("rb") as fh:
        raw = fh.read(MAX_POLICY_BYTES + 1)  # never read a huge file whole
    if len(raw) > MAX_POLICY_BYTES:
        raise PolicyError(f"policy larger than {MAX_POLICY_BYTES} bytes")
    try:
        document = yaml.load(raw, Loader=_StrictLoader)  # noqa: S506 - SafeLoader subclass
    except yaml.YAMLError as exc:
        raise PolicyError(f"invalid YAML: {exc}") from exc
    except RecursionError as exc:
        raise PolicyError("policy nested too deeply") from exc
    sha256 = hashlib.sha256(raw).hexdigest()
    if document is None:
        return Policy(sha256=sha256)
    if not isinstance(document, dict):
        raise PolicyError("the policy must be a mapping, e.g. 'allow_formats: [safetensors]'")
    unknown = set(document) - _KEYS
    if unknown:
        raise PolicyError(f"unknown policy key(s): {sorted(map(str, unknown))}")
    allow_formats = _formats(document["allow_formats"]) if "allow_formats" in document else None
    return Policy(
        allow_formats=allow_formats,
        fail_on=_fail_on(document.get("fail_on")),
        allow=_str_list("allow", document.get("allow")),
        sha256=sha256,
    )


def _str_list(key: str, value: object) -> frozenset[str]:
    if value is None:
        return frozenset()
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise PolicyError(f"{key} must be a list of strings")
    return frozenset(value)


def _fail_on(value: object) -> frozenset[str]:
    severities = _str_list("fail_on", value)
    bad = sorted(severities - ESCALATABLE)
    if bad:
        raise PolicyError(
            f"fail_on severities {bad}; allowed: {sorted(ESCALATABLE)} (risk always blocks)"
        )
    return severities


def _formats(value: object) -> frozenset[str]:
    # Strict: null, false, {}, "" or 0 must not read as "nothing to check".
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise PolicyError("allow_formats must be a list of format names")
    bad = sorted(set(value) - FORMAT_NAMES)
    if bad:
        raise PolicyError(f"unknown format(s) {bad}; known: {sorted(FORMAT_NAMES)}")
    return frozenset(value)


def apply_policy(policy: Policy, entries: list[FileEntry]) -> list[FileEntry]:
    """Add a `policy.format-not-allowed` finding per format not allowed.

    `allow` is NOT applied here: allowed findings stay in the entry (and so in
    the signed AI-BOM, traceably). They are excluded from the verdict by
    `is_blocking` instead — a risk neutralised by policy must never vanish from
    the evidence the webhook relies on."""
    if policy.allow_formats is None:
        return entries
    out = []
    for entry in entries:
        names = {f.value for f in entry.formats} or {UNKNOWN}
        extra = tuple(
            Finding("policy.format-not-allowed", f"format {name} is not allowed by policy")
            for name in sorted(names - policy.allow_formats)
        )
        out.append(dataclasses.replace(entry, findings=entry.findings + extra) if extra else entry)
    return out


def is_allowed(finding: Finding, policy: Policy) -> bool:
    """The policy explicitly accepts this rule (still shown and recorded)."""
    return finding.rule in policy.allow


def is_blocking(finding: Finding, policy: Policy) -> bool:
    """A finding blocks (exit 1) unless the policy allows its rule, and either
    it is a risk or its severity is escalated by `fail_on`."""
    if is_allowed(finding, policy):
        return False
    return finding.severity == "risk" or finding.severity in policy.fail_on

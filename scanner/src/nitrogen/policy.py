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
_KEYS = frozenset({"allow_formats"})


class PolicyError(Exception):
    """The policy file cannot be applied as written."""


@dataclass(frozen=True)
class Policy:
    allow_formats: frozenset[str] | None = None  # None: no format restriction
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
    if "allow_formats" not in document:
        return Policy(sha256=sha256)
    return Policy(_formats(document["allow_formats"]), sha256)


def _formats(value: object) -> frozenset[str]:
    # Strict: null, false, {}, "" or 0 must not read as "nothing to check".
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise PolicyError("allow_formats must be a list of format names")
    bad = sorted(set(value) - FORMAT_NAMES)
    if bad:
        raise PolicyError(f"unknown format(s) {bad}; known: {sorted(FORMAT_NAMES)}")
    return frozenset(value)


def apply_policy(policy: Policy, entries: list[FileEntry]) -> list[FileEntry]:
    """Add a `policy.format-not-allowed` finding per format not allowed."""
    if policy.allow_formats is None:
        return entries
    out = []
    for entry in entries:
        names = {f.value for f in entry.formats} or {UNKNOWN}
        refused = sorted(names - policy.allow_formats)
        extra = tuple(
            Finding("policy.format-not-allowed", f"format {name} is not allowed by policy")
            for name in refused
        )
        out.append(dataclasses.replace(entry, findings=entry.findings + extra) if extra else entry)
    return out

"""CycloneDX 1.6 AI-BOM (ADR 0005), built by our own code with the standard
library only: no production dependency for a supply-chain security tool. The
tests validate it against the official schema.

Layout:
- metadata.component: the scanned model (`machine-learning-model`), hashed
  with the manifest digest, carrying the verdict;
- components: one `file` per scanned file, with its SHA-256 and formats;
- vulnerabilities: one per finding, pointing at its file (`affects`).
"""

import hashlib
import json
import os
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nitrogen import __version__
from nitrogen.inventory import ANALYZED_FORMATS, FileEntry
from nitrogen.policy import Policy

_MODEL_REF = "model"
DIGEST_ALGORITHM = "nitrogen-manifest-v1"

# Nitrogen's three severity tiers (ADR 0008) mapped to CycloneDX ratings. The
# tier lives on the finding, so there is no per-rule table to keep in sync.
_CYCLONEDX_SEVERITY = {"risk": "high", "warning": "medium", "info": "info"}


def _path_bytes(entry: FileEntry) -> bytes:
    # surrogateescape gives back the original bytes of a non-UTF-8 Linux name;
    # surrogatepass (WTF-8, as os.fsencode on Windows) covers the unpaired
    # UTF-16 a Windows name can hold, which surrogateescape refuses.
    path = str(entry.path)
    try:
        return path.encode("utf-8", "surrogateescape")
    except UnicodeEncodeError:
        return path.encode("utf-8", "surrogatepass")


def manifest_digest(entries: list[FileEntry]) -> str:
    """nitrogen-manifest-v1 (ADR 0005): SHA-256 of the compact, ASCII-escaped
    JSON list [[path, sha256], ...] sorted on the UTF-8 bytes of the path.
    JSON quoting makes the encoding injective: no file name (newline, quotes,
    spaces) can make two different file sets produce the same manifest."""
    pairs = [[str(e.path), e.sha256] for e in sorted(entries, key=_path_bytes)]
    canonical = json.dumps(pairs, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def build_aibom(
    model_name: str, entries: list[FileEntry], policy: Policy | None = None
) -> dict[str, Any]:
    policy = policy or Policy()
    refs = {e.path: f"file:{e.path}" for e in entries}
    vulnerabilities = []
    for index, (entry, finding) in enumerate(
        ((e, f) for e in entries for f in e.findings), start=1
    ):
        vuln: dict[str, Any] = {
            "bom-ref": f"finding-{index}",
            "id": finding.rule,
            "source": {"name": "Nitrogen"},
            "ratings": [
                {
                    "severity": _CYCLONEDX_SEVERITY.get(finding.severity, "unknown"),
                    "method": "other",
                }
            ],
            "description": finding.message,
            "affects": [{"ref": refs[entry.path]}],
        }
        if finding.rule in policy.allow:
            # Kept in the signed AI-BOM, marked: a risk accepted by policy must
            # stay visible to whoever verifies the attestation (reviewer I2).
            vuln["analysis"] = {"state": "resolved", "detail": "allowed by policy"}
        vulnerabilities.append(vuln)

    # The verdict ignores allowed findings, exactly like the exit code.
    severities = {f.severity for e in entries for f in e.findings if f.rule not in policy.allow}
    if "risk" in severities:
        verdict = "risk"
    elif "warning" in severities:
        verdict = "compliant-with-warnings"
    else:
        verdict = "compliant"
    bom: dict[str, Any] = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.6",
        "serialNumber": f"urn:uuid:{uuid.uuid4()}",
        "version": 1,
        "metadata": {
            "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "tools": {
                "components": [{"type": "application", "name": "nitrogen", "version": __version__}]
            },
            "component": {
                "type": "machine-learning-model",
                "bom-ref": _MODEL_REF,
                "name": model_name,
                "hashes": [{"alg": "SHA-256", "content": manifest_digest(entries)}],
                "properties": [
                    {"name": "nitrogen:verdict", "value": verdict},
                    {"name": "nitrogen:digest", "value": DIGEST_ALGORITHM},
                    *_policy_properties(policy),
                ],
            },
        },
        "components": [_file_component(e, refs[e.path]) for e in entries],
        "dependencies": [{"ref": _MODEL_REF, "dependsOn": [refs[e.path] for e in entries]}],
    }
    if vulnerabilities:
        bom["vulnerabilities"] = vulnerabilities
    return bom


def _file_component(entry: FileEntry, ref: str) -> dict[str, Any]:
    component: dict[str, Any] = {
        "type": "file",
        "bom-ref": ref,
        "name": str(entry.path),
        "hashes": [{"alg": "SHA-256", "content": entry.sha256}],
    }
    # ADR 0006: false only when a model format Nitrogen does not analyze is
    # present; a file with no model format (config, tokenizer) has none to analyze.
    analyzed = entry.formats <= ANALYZED_FORMATS
    properties = [{"name": "nitrogen:format", "value": str(f)} for f in sorted(entry.formats)]
    properties.append({"name": "nitrogen:analyzed", "value": str(analyzed).lower()})
    properties.append({"name": "nitrogen:size", "value": str(entry.size)})
    component["properties"] = properties
    return component


def _policy_properties(policy: Policy) -> list[dict[str, str]]:
    """Which policy the verdict holds under: a "compliant" without the policy
    it was judged against cannot be told from one under a stricter policy."""
    if policy.sha256 is None:
        return [{"name": "nitrogen:policy", "value": "none"}]
    allowed = "any" if policy.allow_formats is None else ",".join(sorted(policy.allow_formats))
    props = [
        {"name": "nitrogen:policy.sha256", "value": policy.sha256},
        {"name": "nitrogen:policy.allow_formats", "value": allowed},
    ]
    # Record what the policy neutralised or hardened, so a "compliant" verdict
    # under an allow/fail_on policy is never indistinguishable from a clean one.
    if policy.allow:
        props.append({"name": "nitrogen:policy.allow", "value": ",".join(sorted(policy.allow))})
    if policy.fail_on:
        props.append({"name": "nitrogen:policy.fail_on", "value": ",".join(sorted(policy.fail_on))})
    return props


def write_aibom(path: Path, bom: dict[str, Any]) -> None:
    """Write atomically: a temp file in the same directory, then a rename, so a
    half-written AI-BOM can never be picked up and signed."""
    # Short fixed prefix: a 255-character output name must still fit NAME_MAX.
    fd, tmp = tempfile.mkstemp(prefix=".nitrogen-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="ascii") as fh:
            # ASCII-escaped: a non-UTF-8 file name cannot make the write fail
            # and turn a risk into an execution error without an AI-BOM.
            json.dump(bom, fh, indent=2, ensure_ascii=True)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())  # on disk before the rename
        Path(tmp).chmod(0o644)  # mkstemp makes it 0600; cosign may run as another user
        Path(tmp).replace(path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise

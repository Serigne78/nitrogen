"""AI-BOM (step 6, ADR 0005): CycloneDX 1.6, written by our own code and
validated here against the official schema shipped by cyclonedx-python-lib
(a development dependency only, never installed for users)."""

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

import pytest
from cyclonedx.schema import SchemaVersion
from cyclonedx.validation.json import JsonStrictValidator

from nitrogen.aibom import build_aibom, manifest_digest, write_aibom
from nitrogen.findings import Finding
from nitrogen.formats import Format
from nitrogen.inventory import FileEntry

VALIDATOR = JsonStrictValidator(SchemaVersion.V1_6)

CLEAN = [
    FileEntry(PurePosixPath("config.json"), 2, "a" * 64, frozenset()),
    FileEntry(PurePosixPath("model.safetensors"), 80, "b" * 64, frozenset({Format.SAFETENSORS})),
]
RISKY = [
    *CLEAN,
    FileEntry(
        PurePosixPath("sub/evil.pkl"),
        40,
        "c" * 64,
        frozenset({Format.PICKLE}),
        (Finding("pickle.import", "os.system"),),
    ),
]


def _valid(bom: dict[str, Any]) -> None:
    error = VALIDATOR.validate_str(json.dumps(bom))
    assert error is None, error


def _component(bom: dict[str, Any], name: str) -> dict[str, Any]:
    matches: list[dict[str, Any]] = [c for c in bom["components"] if c["name"] == name]
    [component] = matches
    return component


def _props(item: dict[str, Any]) -> dict[str, list[str]]:
    props: dict[str, list[str]] = {}
    for prop in item.get("properties", []):
        props.setdefault(prop["name"], []).append(prop["value"])
    return props


@pytest.mark.parametrize("entries", [CLEAN, RISKY, []], ids=["clean", "risky", "empty"])
def test_valid_against_official_cyclonedx_1_6_schema(entries: list[FileEntry]) -> None:
    _valid(build_aibom("my-model", entries))


def test_model_component_carries_the_manifest_digest() -> None:
    bom = build_aibom("my-model", CLEAN)
    model = bom["metadata"]["component"]
    assert model["type"] == "machine-learning-model"
    assert model["name"] == "my-model"
    assert model["hashes"] == [{"alg": "SHA-256", "content": manifest_digest(CLEAN)}]


def test_manifest_digest_follows_the_v1_spec() -> None:
    # ADR 0005: sha256 of compact, ASCII-escaped JSON [[path, sha256], ...]
    # sorted on the UTF-8 bytes of the path.
    canonical = json.dumps(
        [["config.json", "a" * 64], ["model.safetensors", "b" * 64]],
        ensure_ascii=True,
        separators=(",", ":"),
    )
    assert manifest_digest(CLEAN) == hashlib.sha256(canonical.encode("ascii")).hexdigest()
    props = _props(build_aibom("m", CLEAN)["metadata"]["component"])
    assert props["nitrogen:digest"] == ["nitrogen-manifest-v1"]


def test_newline_in_a_file_name_cannot_forge_a_manifest() -> None:
    # Reviewer C1: one forged name must not collide with two real files.
    h1, evil = "1" * 64, "e" * 64
    forged = [FileEntry(PurePosixPath(f"config.json\n{evil}  model.bin"), 1, h1, frozenset())]
    real = [
        FileEntry(PurePosixPath("config.json"), 1, h1, frozenset()),
        FileEntry(PurePosixPath("model.bin"), 1, evil, frozenset()),
    ]
    assert manifest_digest(forged) != manifest_digest(real)


def test_manifest_is_sorted_on_utf8_bytes_not_path_parts() -> None:
    # "." (0x2E) sorts before "/" (0x2F) as bytes; PurePosixPath sorts the other way.
    a_dot = FileEntry(PurePosixPath("a.b"), 1, "1" * 64, frozenset())
    a_slash = FileEntry(PurePosixPath("a/b"), 1, "2" * 64, frozenset())
    canonical = json.dumps([["a.b", "1" * 64], ["a/b", "2" * 64]], separators=(",", ":"))
    expected = hashlib.sha256(canonical.encode()).hexdigest()
    assert manifest_digest([a_slash, a_dot]) == manifest_digest([a_dot, a_slash]) == expected


def test_manifest_digest_changes_with_any_file() -> None:
    changed = [CLEAN[0], FileEntry(CLEAN[1].path, 80, "d" * 64, CLEAN[1].formats)]
    renamed = [CLEAN[0], FileEntry(PurePosixPath("other.safetensors"), 80, "b" * 64, frozenset())]
    digests = {manifest_digest(e) for e in (CLEAN, changed, renamed, CLEAN[:1])}
    assert len(digests) == 4


def test_non_utf8_file_name_is_hashed_and_written(tmp_path: Path) -> None:
    # A Linux name that is not UTF-8 arrives with surrogateescape characters;
    # it must neither crash the digest nor the write (reviewer I2).
    odd = FileEntry(PurePosixPath("bad\udcff.bin"), 1, "1" * 64, frozenset())
    digest = manifest_digest([odd])
    write_aibom(tmp_path / "bom.json", build_aibom("m", [odd]))
    assert digest and VALIDATOR.validate_str((tmp_path / "bom.json").read_text()) is None


def test_windows_lone_surrogate_name_is_hashed() -> None:
    # NTFS accepts unpaired UTF-16, decoded as "\ud800": no error either.
    assert manifest_digest([FileEntry(PurePosixPath("a\ud800.bin"), 1, "1" * 64, frozenset())])


# Reference vectors for nitrogen-manifest-v1 (ADR 0005): any reimplementation
# (the Go webhook) must reproduce these digests byte for byte. An empty
# expected value is checked against the spec formula only.
REFERENCE_VECTORS: list[tuple[list[tuple[str, str]], str]] = [
    ([], "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"),
    ([("config.json", "a" * 64)], ""),
    ([('dir/été "q" <&>\n.bin', "b" * 64), ("a.b", "c" * 64)], ""),
]


def test_reference_vectors() -> None:
    for pairs, expected in REFERENCE_VECTORS:
        entries = [FileEntry(PurePosixPath(p), 1, h, frozenset()) for p, h in pairs]
        canonical = json.dumps(
            sorted([[p, h] for p, h in pairs], key=lambda x: x[0].encode("utf-8")),
            ensure_ascii=True,
            separators=(",", ":"),
        )
        assert manifest_digest(entries) == hashlib.sha256(canonical.encode()).hexdigest()
        if expected:
            assert manifest_digest(entries) == expected


def test_one_file_component_per_file_with_hash_and_formats() -> None:
    bom = build_aibom("my-model", RISKY)
    assert [c["name"] for c in bom["components"]] == [str(e.path) for e in RISKY]
    evil = _component(bom, "sub/evil.pkl")
    assert evil["type"] == "file"
    assert evil["hashes"] == [{"alg": "SHA-256", "content": "c" * 64}]
    assert _props(evil)["nitrogen:format"] == ["pickle"]
    assert "nitrogen:format" not in _props(_component(bom, "config.json"))


def test_findings_are_vulnerabilities_affecting_their_file() -> None:
    bom = build_aibom("my-model", RISKY)
    [vuln] = bom["vulnerabilities"]
    assert vuln["id"] == "pickle.import"
    assert vuln["description"] == "os.system"
    assert vuln["source"]["name"] == "Nitrogen"
    assert vuln["affects"] == [{"ref": _component(bom, "sub/evil.pkl")["bom-ref"]}]


def test_clean_model_has_no_vulnerabilities_and_compliant_verdict() -> None:
    bom = build_aibom("my-model", CLEAN)
    assert bom.get("vulnerabilities", []) == []
    assert _props(bom["metadata"]["component"])["nitrogen:verdict"] == ["compliant"]
    assert _props(build_aibom("m", RISKY)["metadata"]["component"])["nitrogen:verdict"] == ["risk"]


def test_scanner_is_recorded_as_the_tool() -> None:
    from nitrogen import __version__

    [tool] = build_aibom("m", CLEAN)["metadata"]["tools"]["components"]
    assert (tool["name"], tool["version"]) == ("nitrogen", __version__)


def test_serial_numbers_are_unique() -> None:
    assert build_aibom("m", CLEAN)["serialNumber"] != build_aibom("m", CLEAN)["serialNumber"]


def test_write_is_atomic_and_leaves_no_temp_file(tmp_path: Path) -> None:
    target = tmp_path / "bom.cdx.json"
    target.write_text("old")
    write_aibom(target, build_aibom("m", CLEAN))
    assert json.loads(target.read_text())["bomFormat"] == "CycloneDX"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["bom.cdx.json"]


def test_every_file_says_whether_it_was_analyzed() -> None:
    # ADR 0006: false only for a model format Nitrogen does not analyze.
    h5 = FileEntry(PurePosixPath("model.h5"), 1, "d" * 64, frozenset({Format.HDF5}))
    bom = build_aibom("m", [*RISKY, h5])
    assert _props(_component(bom, "config.json"))["nitrogen:analyzed"] == ["true"]
    assert _props(_component(bom, "sub/evil.pkl"))["nitrogen:analyzed"] == ["true"]
    assert _props(_component(bom, "model.h5"))["nitrogen:analyzed"] == ["false"]


def test_the_applied_policy_is_recorded() -> None:
    from nitrogen.policy import Policy

    none = _props(build_aibom("m", CLEAN)["metadata"]["component"])
    assert none["nitrogen:policy"] == ["none"]
    policy = Policy(allow_formats=frozenset({"unknown", "safetensors"}), sha256="f" * 64)
    applied = _props(build_aibom("m", CLEAN, policy)["metadata"]["component"])
    assert applied["nitrogen:policy.sha256"] == ["f" * 64]
    assert applied["nitrogen:policy.allow_formats"] == ["safetensors,unknown"]


def test_finding_severity_maps_to_cyclonedx_rating() -> None:
    # ADR 0008: the tier lives on the finding (risk/warning/info), mapped to a
    # CycloneDX rating; no silent fallback.
    entries = [
        FileEntry(
            PurePosixPath("m.zip"),
            1,
            "a" * 64,
            frozenset({Format.ZIP}),
            (
                Finding("pickle.import", "os.system", "risk"),
                Finding("archive.encrypted", "x: encrypted member", "warning"),
                Finding("findings.truncated", "3 more pickle.import finding(s) not listed", "info"),
            ),
        )
    ]
    ratings = {
        v["id"]: v["ratings"][0]["severity"] for v in build_aibom("m", entries)["vulnerabilities"]
    }
    assert ratings == {
        "pickle.import": "high",
        "archive.encrypted": "medium",
        "findings.truncated": "info",
    }

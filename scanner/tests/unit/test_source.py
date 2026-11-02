"""Remote source & integrity (step 12, ADR 0014).

The scanner can take `hf://org/model@revision`: download to a cache (never
loaded), resolve the revision to a commit sha, and verify each LFS file's
SHA-256 against what the Hub publishes. These tests stay OFFLINE — the network
(`resolve` of an hf:// target) is injected. The real download is an opt-in
script, never part of pytest (rule 5).
"""

import types
from pathlib import Path, PurePosixPath

import pytest

from nitrogen.inventory import FileEntry
from nitrogen.source import (
    HfSource,
    LocalSource,
    Resolved,
    check_integrity,
    integrity_coverage,
    missing_files,
    parse_target,
    resolve,
)

SHA = "a" * 40


# --- parsing the target -----------------------------------------------------


def test_parse_local_path() -> None:
    assert parse_target("./model") == LocalSource(Path("./model"))


def test_parse_hf_pinned() -> None:
    src = parse_target(f"hf://org/model@{SHA}")
    assert src == HfSource("org/model", SHA)
    assert src.pinned


@pytest.mark.parametrize("target", ["hf://org/model", "hf://org/model@main", "hf://org/model@v1.0"])
def test_parse_hf_unpinned(target: str) -> None:
    src = parse_target(target)
    assert isinstance(src, HfSource)
    assert not src.pinned


@pytest.mark.parametrize(
    "target",
    ["hf://model", "hf://org/model/extra", "hf://org/model@", "hf://", "hf:///x"],
    ids=lambda t: t,
)
def test_parse_invalid_hf_is_an_error(target: str) -> None:
    with pytest.raises(ValueError, match="hf://"):
        parse_target(target)


# --- integrity --------------------------------------------------------------


def _entry(name: str, sha: str) -> FileEntry:
    return FileEntry(PurePosixPath(name), 1, sha, frozenset())


def test_integrity_match_is_silent() -> None:
    entries = [_entry("model.safetensors", SHA)]
    assert check_integrity(entries, {"model.safetensors": SHA}) == []


def test_integrity_mismatch_is_a_risk() -> None:
    entries = [_entry("model.safetensors", "b" * 40)]
    [(path, finding)] = check_integrity(entries, {"model.safetensors": SHA})
    assert path == PurePosixPath("model.safetensors")
    assert (finding.rule, finding.severity) == ("integrity.mismatch", "risk")


def test_integrity_ignores_files_the_hub_does_not_publish() -> None:
    # A non-LFS file (no published SHA-256) is simply not verified here.
    entries = [_entry("config.json", "c" * 40)]
    assert check_integrity(entries, {}) == []


def test_injected_file_not_published_is_unlisted() -> None:
    # Reviewer I1: a local file the Hub does not list at all is injected.
    entries = [_entry("model.safetensors", SHA), _entry("evil.onnx", "e" * 40)]
    results = check_integrity(entries, {"model.safetensors": SHA}, frozenset({"model.safetensors"}))
    assert [(str(p), f.rule, f.severity) for p, f in results] == [
        ("evil.onnx", "integrity.unlisted", "warning")
    ]


def test_missing_published_file_is_flagged() -> None:
    entries = [_entry("a.txt", SHA)]
    findings = missing_files(frozenset({"a.txt", "weights.bin"}), entries)
    assert [(f.rule, f.severity) for f in findings] == [("integrity.missing", "warning")]
    assert "weights.bin" in findings[0].message


def test_local_source_has_no_completeness_findings() -> None:
    # published is None for a local source: no injected/missing checks at all.
    entries = [_entry("whatever.bin", SHA)]
    assert check_integrity(entries, {}, None) == []
    assert missing_files(None, entries) == []


def test_coverage_reports_verified_over_published() -> None:
    hf = Resolved(Path("x"), "hf://o/m", SHA, {"a": "h", "b": "h"}, published=frozenset("abc"))
    assert integrity_coverage(hf) == "2/3"
    assert integrity_coverage(Resolved(Path("x"), None, None, {})) is None


# --- resolve (network injected) ---------------------------------------------


def _fake_info(sha: str, siblings: dict[str, str | None]) -> object:
    sibs = [
        types.SimpleNamespace(rfilename=name, lfs=types.SimpleNamespace(sha256=h) if h else None)
        for name, h in siblings.items()
    ]
    return types.SimpleNamespace(sha=sha, siblings=sibs)


def test_resolve_pinned_has_no_warning(tmp_path: Path) -> None:
    src = HfSource("org/model", SHA)
    resolved = resolve(
        src,
        model_info=lambda repo, revision: _fake_info(SHA, {"model.safetensors": "d" * 64}),
        snapshot_download=lambda **kw: str(tmp_path),
    )
    assert resolved.revision == SHA
    assert resolved.findings == []
    assert resolved.expected_sha256 == {"model.safetensors": "d" * 64}
    assert resolved.source_uri == "hf://org/model"


def test_resolve_unpinned_warns_and_records_resolved_sha(tmp_path: Path) -> None:
    src = HfSource("org/model", None)  # unpinned
    resolved = resolve(
        src,
        model_info=lambda repo, revision: _fake_info("c" * 40, {}),
        snapshot_download=lambda **kw: str(tmp_path),
    )
    assert resolved.revision == "c" * 40  # resolved to the commit sha
    assert [(f.rule, f.severity) for f in resolved.findings] == [
        ("source.unpinned_revision", "warning")
    ]


def test_resolve_refuses_a_non_commit_resolution(tmp_path: Path) -> None:
    # I3: if the Hub yields no commit sha, we must not record a mutable ref
    # (a branch/tag) as the pinned revision — fail closed.
    src = HfSource("org/model", "main")
    with pytest.raises(ValueError, match="commit sha"):
        resolve(
            src,
            model_info=lambda repo, revision: _fake_info("", {}),
            snapshot_download=lambda **kw: str(tmp_path),
        )


def test_resolve_local_source_is_offline(tmp_path: Path) -> None:
    def _no_info(repo: str, revision: str | None) -> object:
        raise AssertionError("a local source must not touch the network")

    def _no_download(**kw: object) -> str:
        raise AssertionError("a local source must not touch the network")

    resolved = resolve(LocalSource(tmp_path), model_info=_no_info, snapshot_download=_no_download)
    assert resolved.local_dir == tmp_path
    assert resolved.revision is None
    assert resolved.source_uri is None
    assert resolved.expected_sha256 == {}

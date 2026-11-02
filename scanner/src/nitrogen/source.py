"""Remote source resolution & integrity (step 12, ADR 0014).

The scan target is a local path or `hf://org/model[@revision]`. A pluggable
source abstraction (hf:// now; s3:// / registries later) resolves a remote
target to a local cache directory — downloaded, NEVER loaded — plus the commit
revision and the per-file SHA-256 the Hub publishes for integrity.

Only an `hf://` target touches the network, and the network calls are injected
so the test suite stays offline (rule 5); the real download is opt-in.
"""

from __future__ import annotations

import re
import shutil
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from nitrogen.findings import Finding
from nitrogen.inventory import FileEntry

# hf://<org>/<model>[@<revision>] — exactly one org/model, an optional revision.
_HF = re.compile(r"hf://(?P<repo>[^@/\s]+/[^@/\s]+)(?:@(?P<rev>[^@/\s]+))?")
_COMMIT_SHA = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class LocalSource:
    path: Path


@dataclass(frozen=True)
class HfSource:
    repo: str
    revision: str | None  # None, a branch or a tag = unpinned

    @property
    def pinned(self) -> bool:
        return self.revision is not None and _COMMIT_SHA.fullmatch(self.revision) is not None


Source = LocalSource | HfSource


@dataclass
class Resolved:
    local_dir: Path
    source_uri: str | None  # e.g. "hf://org/model", or None for a local path
    revision: str | None  # resolved commit sha, or None for a local path
    expected_sha256: dict[str, str]  # repo-relative path -> Hub-published LFS sha256
    published: frozenset[str] | None = None  # all files the Hub lists; None for a local source
    findings: list[Finding] = field(default_factory=list)  # source-level (unpinned revision)


def parse_target(target: str) -> Source:
    """A local path, or an `hf://org/model[@rev]` target. An hf:// that is not a
    clean org/model[@rev] is an error, never a silent local-path fallback."""
    if target.startswith("hf://"):
        match = _HF.fullmatch(target.strip())
        if match is None:
            raise ValueError(f"invalid hf:// target (expected hf://org/model[@revision]): {target}")
        return HfSource(match["repo"], match["rev"])
    return LocalSource(Path(target))


def resolve(
    source: Source,
    *,
    dest: Path | None = None,
    model_info: Callable[[str, str | None], object] | None = None,
    snapshot_download: Callable[..., str] | None = None,
) -> Resolved:
    """Resolve a source to a local directory. A local source is returned as-is,
    offline. An hf:// source is resolved to its commit sha and downloaded; the
    network calls are injected (tests stay offline)."""
    if isinstance(source, LocalSource):
        return Resolved(source.path, None, None, {})

    info_fn, download_fn = _network(model_info, snapshot_download)
    info = info_fn(source.repo, source.revision)
    commit = str(getattr(info, "sha", None) or source.revision or "")
    if _COMMIT_SHA.fullmatch(commit) is None:
        # ADR 0014: a revision must resolve to a commit sha. Refuse to record a
        # mutable ref (a branch/tag) as the pinned revision — fail closed.
        raise ValueError(f"{source.repo}@{source.revision}: did not resolve to a commit sha")
    findings: list[Finding] = []
    if not source.pinned:
        findings.append(
            Finding(
                "source.unpinned_revision", f"unpinned revision resolved to {commit}", "warning"
            )
        )
    siblings = getattr(info, "siblings", None)
    expected = _lfs_hashes(siblings)
    published = _published_files(siblings)
    local = Path(
        download_fn(repo_id=source.repo, revision=commit, local_dir=str(dest) if dest else None)
    )
    # local_dir leaves a .cache/huggingface/ tracking tree (not model content):
    # drop it so it is neither scanned nor listed in the AI-BOM.
    shutil.rmtree(local / ".cache", ignore_errors=True)
    return Resolved(
        local, f"hf://{source.repo}", commit, expected, published=published, findings=findings
    )


def check_integrity(
    entries: Iterable[FileEntry],
    expected_sha256: dict[str, str],
    published: frozenset[str] | None = None,
) -> list[tuple[PurePosixPath, Finding]]:
    """Per-file integrity findings (ADR 0014). A file whose local SHA-256 differs
    from the Hub-published one was altered → `integrity.mismatch` (risk). A local
    file the Hub does not list at all is injected → `integrity.unlisted`
    (warning). `published` is None for a local source (no completeness check)."""
    out: list[tuple[PurePosixPath, Finding]] = []
    for entry in entries:
        name = str(entry.path)
        want = expected_sha256.get(name)
        if want is not None:
            if entry.sha256 != want:
                out.append(
                    (
                        entry.path,
                        Finding(
                            "integrity.mismatch",
                            f"{name}: SHA-256 differs from the Hub (local {entry.sha256[:12]}…)",
                            "risk",
                        ),
                    )
                )
        elif published is not None and name not in published:
            out.append(
                (
                    entry.path,
                    Finding("integrity.unlisted", f"{name}: not published by the Hub", "warning"),
                )
            )
    return out


def missing_files(published: frozenset[str] | None, entries: Iterable[FileEntry]) -> list[Finding]:
    """Files the Hub publishes but that are absent locally (incomplete or
    tampered download) — source-level `integrity.missing` (warning, ADR 0014)."""
    if published is None:
        return []
    local = {str(e.path) for e in entries}
    return [
        Finding("integrity.missing", f"{name}: published by the Hub but absent locally", "warning")
        for name in sorted(published - local)
    ]


def integrity_coverage(resolved: Resolved) -> str | None:
    """`verified/published`: how many published files have a Hub SHA-256 we
    checked, over the total the Hub lists. None for a local source."""
    if resolved.published is None:
        return None
    return f"{len(resolved.expected_sha256)}/{len(resolved.published)}"


def _network(
    model_info: Callable[[str, str | None], object] | None,
    snapshot_download: Callable[..., str] | None,
) -> tuple[Callable[[str, str | None], object], Callable[..., str]]:
    if model_info is not None and snapshot_download is not None:
        return model_info, snapshot_download
    if model_info is not None or snapshot_download is not None:
        # All-or-nothing: injecting only one would silently reach the network
        # through the other, in a module whose point is offline tests.
        raise ValueError("inject both model_info and snapshot_download, or neither")
    from huggingface_hub import HfApi  # lazy: keeps the import off the offline path
    from huggingface_hub import snapshot_download as hf_download

    api = HfApi()

    def info(repo: str, revision: str | None) -> object:
        return api.model_info(repo, revision=revision, files_metadata=True)

    def download(repo_id: str, revision: str | None = None, local_dir: str | None = None) -> str:
        # Download the real files into local_dir (flat, no blob symlinks), which
        # avoids the Windows "no symlink support" failure. Never loaded.
        return str(hf_download(repo_id=repo_id, revision=revision, local_dir=local_dir))

    return info, download  # both injected together, or both real (guarded above)


def _published_files(siblings: Iterable[Any] | None) -> frozenset[str]:
    return frozenset(
        name for s in siblings or [] if isinstance(name := getattr(s, "rfilename", None), str)
    )


def _lfs_hashes(siblings: Iterable[Any] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for sibling in siblings or []:
        lfs = getattr(sibling, "lfs", None)
        if lfs is None:
            continue
        sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        name = getattr(sibling, "rfilename", None)
        if isinstance(sha, str) and isinstance(name, str):
            out[name] = sha
    return out

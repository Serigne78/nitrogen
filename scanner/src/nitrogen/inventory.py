"""Inventory of a model path: every regular file with its size, SHA-256, formats
and findings.

Fail-closed: a path that cannot be read (broken symlink, permission denied,
FIFO, device...) raises `InventoryError`, never a silent skip.

Each file is analyzed through the descriptor it was just hashed with: the same
file, not one swapped in between. A concurrent writer could still change its
bytes between the two reads; a scanned tree is assumed not to be written to.
"""

import hashlib
import os
import stat
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO

from nitrogen.findings import Finding
from nitrogen.formats import Format, detect_formats
from nitrogen.safetensors import check_safetensors

CHUNK_SIZE = 1024 * 1024

_ANALYZERS: dict[Format, Callable[[BinaryIO, int], list[Finding]]] = {
    Format.SAFETENSORS: check_safetensors,
}
# Formats with an analyzer. A file with no format (config, tokenizer...) has
# nothing a supported loader would read, hence nothing to analyze.
ANALYZED_FORMATS = frozenset(_ANALYZERS)

# O_NONBLOCK: opening a FIFO must not wait for a writer (we reject it right
# after). O_BINARY: no newline translation on Windows.
_OPEN_FLAGS = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)


class InventoryError(Exception):
    """A path the scan cannot read, hence cannot vouch for."""


@dataclass(frozen=True)
class FileEntry:
    path: PurePosixPath  # relative to the scanned directory, or the file name
    size: int
    sha256: str
    formats: frozenset[Format]  # empty: no supported loader would read it
    findings: tuple[Finding, ...] = ()


def inventory(root: Path) -> list[FileEntry]:
    """List every file under `root` (or `root` itself), sorted by path."""
    if root.is_dir():
        files = list(_walk(root))
    elif root.exists():
        files = [(PurePosixPath(root.name), root)]
    else:
        raise InventoryError(f"no such file or directory: {root}")
    return sorted((_entry(rel, path) for rel, path in files), key=lambda e: e.path)


def _walk(root: Path) -> Iterator[tuple[PurePosixPath, Path]]:
    """Walk `root` without following directory links.

    File symlinks are followed wherever they point: Hugging Face cache
    snapshots are symlinks to `../../blobs`. A directory link (symlink or
    junction) inside `root` is skipped, its target being walked under its real
    path anyway, which also rules out loops. One leaving `root` is an error:
    skipping it would hide files a loader follows, walking it could mean `/`.
    """
    boundary = Path(os.path.realpath(root))
    for dirpath, dirnames, filenames in root.walk(on_error=_raise):
        for name in [*dirnames, *filenames]:
            path = dirpath / name
            if not (path.is_symlink() or path.is_junction()) or not path.is_dir():
                continue
            if not Path(os.path.realpath(path)).is_relative_to(boundary):
                raise InventoryError(f"directory link leaves the scanned tree: {path}")
            if name in dirnames:
                dirnames.remove(name)
            else:
                filenames.remove(name)
        for name in filenames:
            path = dirpath / name
            yield PurePosixPath(path.relative_to(root).as_posix()), path


def _raise(exc: OSError) -> None:
    raise InventoryError(f"cannot list {exc.filename}: {exc.strerror}") from exc


def _entry(rel: PurePosixPath, path: Path) -> FileEntry:
    try:
        fd = os.open(path, _OPEN_FLAGS)
    except OSError as exc:
        raise InventoryError(f"cannot open {path}: {exc.strerror}") from exc
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise InventoryError(f"not a regular file: {path}")
        with os.fdopen(fd, "rb") as fh:
            fd = -1  # now owned by fh
            digest = hashlib.sha256()
            size = 0
            while chunk := fh.read(CHUNK_SIZE):
                digest.update(chunk)
                size += len(chunk)
            formats = detect_formats(fh)
            # Every interpretation is analyzed, in a fixed order (ADR 0003).
            findings = tuple(
                finding
                for fmt in sorted(formats)
                if (analyzer := _ANALYZERS.get(fmt))
                for finding in analyzer(fh, size)
            )
    except OSError as exc:
        raise InventoryError(f"cannot read {path}: {exc.strerror}") from exc
    finally:
        if fd != -1:
            os.close(fd)
    return FileEntry(rel, size, digest.hexdigest(), formats, findings)

import hashlib
import os
import struct
import sys
from pathlib import Path, PurePosixPath

import pytest

from nitrogen.formats import Format
from nitrogen.inventory import FileEntry, InventoryError, inventory

EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def _symlink(link: Path, target: Path, *, directory: bool = False) -> None:
    try:
        link.symlink_to(target, target_is_directory=directory)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not supported here")


def test_empty_file(tmp_path: Path) -> None:
    (tmp_path / "empty").write_bytes(b"")
    assert inventory(tmp_path) == [
        FileEntry(PurePosixPath("empty"), 0, EMPTY_SHA256, Format.UNKNOWN)
    ]


def test_hash_and_format(tmp_path: Path) -> None:
    data = b"\x80\x02K\x01."
    (tmp_path / "weights.bin").write_bytes(data)
    [entry] = inventory(tmp_path)
    assert entry.sha256 == hashlib.sha256(data).hexdigest()
    assert entry.size == len(data)
    assert entry.format == Format.PICKLE


def test_hash_of_multi_chunk_file(tmp_path: Path) -> None:
    data = os.urandom(3 * 1024 * 1024 + 7)
    (tmp_path / "big").write_bytes(data)
    [entry] = inventory(tmp_path)
    assert entry.sha256 == hashlib.sha256(data).hexdigest()


def test_nested_paths_are_relative_and_sorted(tmp_path: Path) -> None:
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "c").mkdir()
    (tmp_path / "b" / "c" / "z.json").write_bytes(b"{}")
    (tmp_path / "b" / "y").write_bytes(b"")
    (tmp_path / "a").write_bytes(b"")
    paths = [e.path for e in inventory(tmp_path)]
    assert paths == [PurePosixPath("a"), PurePosixPath("b/c/z.json"), PurePosixPath("b/y")]


def test_empty_directory(tmp_path: Path) -> None:
    assert inventory(tmp_path) == []


def test_single_file(tmp_path: Path) -> None:
    raw = b'{"__metadata__": {}}'
    model = tmp_path / "model.safetensors"
    model.write_bytes(struct.pack("<Q", len(raw)) + raw)
    [entry] = inventory(model)
    assert entry.path == PurePosixPath("model.safetensors")
    assert entry.format == Format.SAFETENSORS


def test_missing_root(tmp_path: Path) -> None:
    with pytest.raises(InventoryError):
        inventory(tmp_path / "nope")


def test_file_symlink_is_followed(tmp_path: Path) -> None:
    # Hugging Face cache snapshots are symlinks to blobs.
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    (blobs / "abc").write_bytes(b"K\x01.")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    _symlink(snapshot / "pytorch_model.bin", blobs / "abc")
    [entry] = inventory(snapshot)
    assert entry.path == PurePosixPath("pytorch_model.bin")
    assert entry.format == Format.PICKLE


def test_directory_symlink_inside_root_is_listed_once(tmp_path: Path) -> None:
    # Its target is walked anyway under its real path.
    (tmp_path / "real").mkdir()
    (tmp_path / "real" / "f").write_bytes(b"")
    _symlink(tmp_path / "linked", tmp_path / "real", directory=True)
    assert [e.path for e in inventory(tmp_path)] == [PurePosixPath("real/f")]


def test_directory_symlink_outside_root_is_error(tmp_path: Path) -> None:
    # Neither silently skipped (a loader would follow it) nor walked (DoS, `/`).
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "payload.bin").write_bytes(b"K\x01.")
    root = tmp_path / "root"
    root.mkdir()
    _symlink(root / "linked", outside, directory=True)
    with pytest.raises(InventoryError):
        inventory(root)


def test_root_itself_may_be_a_symlink(tmp_path: Path) -> None:
    (tmp_path / "real").mkdir()
    (tmp_path / "real" / "f").write_bytes(b"")
    _symlink(tmp_path / "model", tmp_path / "real", directory=True)
    assert [e.path for e in inventory(tmp_path / "model")] == [PurePosixPath("f")]


def test_directory_symlink_loop_terminates(tmp_path: Path) -> None:
    (tmp_path / "f").write_bytes(b"")
    _symlink(tmp_path / "loop", tmp_path, directory=True)
    assert [e.path for e in inventory(tmp_path)] == [PurePosixPath("f")]


def test_broken_symlink_is_error(tmp_path: Path) -> None:
    _symlink(tmp_path / "dangling", tmp_path / "missing")
    with pytest.raises(InventoryError):
        inventory(tmp_path)


def test_unreadable_directory_is_error_not_skipped(tmp_path: Path) -> None:
    if sys.platform == "win32" or os.geteuid() == 0:
        pytest.skip("needs POSIX permissions and a non-root user")
    locked = tmp_path / "locked"
    locked.mkdir()
    (locked / "payload.bin").write_bytes(b"K\x01.")
    locked.chmod(0)
    try:
        with pytest.raises(InventoryError):
            inventory(tmp_path)
    finally:
        locked.chmod(0o755)


def test_fifo_is_error_not_a_hang(tmp_path: Path) -> None:
    mkfifo = getattr(os, "mkfifo", None)  # absent on Windows (and in its typeshed stubs)
    if mkfifo is None:
        pytest.skip("no FIFOs on this OS")
    mkfifo(tmp_path / "fifo")
    with pytest.raises(InventoryError):
        inventory(tmp_path)

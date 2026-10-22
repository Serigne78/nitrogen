"""Analyze pickles carried inside zip and tar archives (ADR 0004).

Every member whose bytes are a pickle is analyzed, not only `data.pkl`, in a
PyTorch zip, a generic zip or a legacy torch tar. Members are read from the
archive into memory, never written to disk (no path-traversal), and the amount
read is capped (no decompression bomb). Nested archives are flagged, not
recursed, in v0.1.
"""

import io
import tarfile
import zipfile
from typing import BinaryIO

from nitrogen.findings import BoundedFindings, Finding
from nitrogen.formats import Format, detect_formats
from nitrogen.pickle_scan import scan_pickle

# Per-member read cap: a bigger member is flagged, never fully read.
MAX_MEMBER_BYTES = 64 * 1024 * 1024
# Total read across all members, so an archive of many members cannot add up
# to an unbounded read.
MAX_TOTAL_BYTES = 512 * 1024 * 1024


def check_zip(fh: BinaryIO, size: int, *, sole_format: bool = True) -> list[Finding]:
    fh.seek(0)
    try:
        with zipfile.ZipFile(fh) as zf:
            members = [
                (i.filename, i.file_size, _is_encrypted(i)) for i in zf.infolist() if not i.is_dir()
            ]
            return _scan_members(members, lambda name: zf.read(name))
    except (zipfile.BadZipFile, OSError, EOFError, ValueError) as exc:
        return [Finding("pickle.unanalyzable", f"unreadable zip: {exc}")]
    finally:
        fh.seek(0)


def _is_encrypted(info: zipfile.ZipInfo) -> bool:
    """Encryption is read from the central directory, never from an error
    message: a member name is attacker-controlled and must not decide severity.
    Bit 0 = traditional ZipCrypto, bit 6 = strong encryption, method 99 = AES."""
    return bool(info.flag_bits & 0x1) or bool(info.flag_bits & 0x40) or info.compress_type == 99


def check_tar(fh: BinaryIO, size: int, *, sole_format: bool = True) -> list[Finding]:
    fh.seek(0)
    try:
        with tarfile.open(fileobj=fh, mode="r:") as tar:
            files = [m for m in tar.getmembers() if m.isfile()]
            members = [(m.name, m.size, False) for m in files]  # tar has no encryption

            def read(name: str) -> bytes:
                extracted = tar.extractfile(name)
                return extracted.read(MAX_MEMBER_BYTES + 1) if extracted else b""

            return _scan_members(members, read)
    except (tarfile.TarError, OSError, EOFError, ValueError) as exc:
        return [Finding("pickle.unanalyzable", f"unreadable tar: {exc}")]
    finally:
        fh.seek(0)


def _scan_members(members: list[tuple[str, int, bool]], read: object) -> list[Finding]:
    # Bounded while walking: an archive of many members cannot pile up
    # millions of findings before the per-file cap applies.
    findings = BoundedFindings()
    total = 0
    for name, declared_size, encrypted in members:
        if encrypted:
            # No loader opens an encrypted member without the password: a blind
            # spot (warning), not a proven risk (ADR 0008). Decided from the
            # central directory, never from a read-error message.
            findings.append(Finding("archive.encrypted", f"{name}: encrypted member", "warning"))
            continue
        if declared_size > MAX_MEMBER_BYTES or total > MAX_TOTAL_BYTES:
            findings.append(Finding("pickle.unanalyzable", f"{name}: member too large to scan"))
            continue
        try:
            data = read(name)[: MAX_MEMBER_BYTES + 1]  # type: ignore[operator]
        except Exception as exc:
            # Corrupt / unsupported feature: we cannot rule out that some
            # loader reads it. Stays a risk, whatever the member is named.
            findings.append(Finding("pickle.unanalyzable", f"{name}: cannot read member ({exc})"))
            continue
        total += len(data)
        if len(data) > MAX_MEMBER_BYTES:
            findings.append(Finding("pickle.unanalyzable", f"{name}: member too large to scan"))
            continue
        formats = detect_formats(io.BytesIO(data))
        if formats & {Format.ZIP, Format.PYTORCH_ZIP, Format.TAR}:
            findings.append(Finding("pickle.unanalyzable", f"{name}: nested archive not scanned"))
        elif Format.PICKLE in formats:
            findings.absorb(scan_pickle(data, sole_format=True, source=name))
    return findings.result()

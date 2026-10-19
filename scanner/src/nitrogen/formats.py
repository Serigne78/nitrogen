"""Model file format detection from content (rule #4).

Only the bytes decide: file names and extensions are never consulted.
Nothing here deserializes; pickles are probed with `pickletools.genops`.

A file gets every format a supported loader would accept, with no priority
between them (ADR 0003): a polyglot cannot pick which analysis runs. No
format at all means no supported loader would read it.
"""

import io
import pickletools
import re
import struct
import tarfile
import zipfile
from enum import StrEnum
from typing import BinaryIO

# Bytes read to probe a pickle without magic (protocol 0/1). The probe runs on
# an in-memory copy: genops on the file itself would honour attacker-chosen
# lengths (a BINUNICODE announcing 4 GiB allocates 4 GiB).
PROBE_SIZE = 64 * 1024

# Local file header (torch.load's zip test), or end of central directory for an
# empty archive (which torch sends down its legacy path, where the plain
# magic-number unpickler stops at once on "P").
_ZIP_MAGICS = (b"PK\x03\x04", b"PK\x05\x06")


class Format(StrEnum):
    PICKLE = "pickle"
    PYTORCH_ZIP = "pytorch-zip"
    ZIP = "zip"
    TAR = "tar"  # pre-1.6 torch.save: torch.load still unpickles its members
    SAFETENSORS = "safetensors"


def detect_formats(fh: BinaryIO) -> frozenset[Format]:
    """Every format of a seekable binary stream, which is left rewound to 0."""
    fh.seek(0)
    try:
        return _detect(fh)
    finally:
        fh.seek(0)


def _detect(fh: BinaryIO) -> frozenset[Format]:
    prefix = fh.read(PROBE_SIZE)
    file_continues = fh.read(1) != b""
    size = fh.seek(0, io.SEEK_END)
    formats: set[Format] = set()
    if prefix[:4] in _ZIP_MAGICS:
        # torch.load takes its zip path. A plain unpickler stops at once on
        # "P" (PERSID) without a persistent_load: no pickle reading to analyze.
        formats.add(_zip_format(fh))
    elif _has_proto_magic(prefix) or _reads_as_pickle(prefix, file_continues=file_continues):
        formats.add(Format.PICKLE)
    if _is_tar(fh):
        formats.add(Format.TAR)
    if _is_safetensors(prefix, size):
        formats.add(Format.SAFETENSORS)
    return frozenset(formats)


def _is_tar(fh: BinaryIO) -> bool:
    """torch.load's own test: tarfile.open(mode="r:") reads a first header.

    Only that header block is parsed (checksum included); nothing is extracted.
    """
    fh.seek(0)
    try:
        with tarfile.open(fileobj=fh, mode="r:", format=tarfile.PAX_FORMAT):
            return True
    except (tarfile.TarError, OSError, ValueError, EOFError, UnicodeError):
        return False


def _has_proto_magic(prefix: bytes) -> bool:
    # PROTO opcode: pickle protocol 2 to 5 (also the legacy non-zip torch.save).
    return len(prefix) >= 2 and prefix[0] == 0x80 and 2 <= prefix[1] <= 5


def _zip_format(fh: BinaryIO) -> Format:
    """Tell a PyTorch archive (`<name>/data.pkl`) from any other zip.

    Only the central directory is read; no entry is decompressed. An archive
    that cannot be listed is still a zip: the pickle analysis reports it.
    """
    fh.seek(0)
    try:
        with zipfile.ZipFile(fh) as zf:
            names = zf.namelist()
    except (zipfile.BadZipFile, OSError, ValueError, EOFError, NotImplementedError):
        return Format.ZIP
    if any(re.split(r"[/\\]", name)[-1] == "data.pkl" for name in names):
        return Format.PYTORCH_ZIP
    return Format.ZIP


def _is_safetensors(prefix: bytes, size: int) -> bool:
    """8-byte little-endian header size, then a JSON object.

    Matches what the reference loader accepts, not what a typical file looks
    like: JSON whitespace before "{" is allowed, so that such a file reaches
    the safetensors parser (which reports it) instead of falling to UNKNOWN.
    And the header must fit in the file, as every loader requires: read as a
    size, 8 bytes of text are ~10**18, so "EMOJI = {" is no safetensors.
    Deliberately loose otherwise: JSON and offsets are validated there.
    """
    if len(prefix) < 9:
        return False
    (header_size,) = struct.unpack("<Q", prefix[:8])
    if header_size > size - 8:
        return False
    start = prefix[8:].lstrip(b" \t\n\r")
    # Whitespace up to the probe limit: the "{" may come later.
    return header_size >= 2 and (start[:1] == b"{" or (start == b"" and len(prefix) == PROBE_SIZE))


def _reads_as_pickle(prefix: bytes, *, file_continues: bool) -> bool:
    """Would an unpickler read this stream? Protocol 0/1 have no magic.

    Yes if it resolves a global, reaches STOP, or stays valid up to the probe
    limit. An unpickler acts on each opcode as it reads it, so what counts is
    what it does before the stream breaks: a global resolved before garbage or
    a missing STOP is still executed. Pickles may be concatenated (legacy
    torch.save), so probing goes on after STOP.
    """
    stream = io.BytesIO(prefix)
    reached_stop = False
    while stream.tell() < len(prefix):
        ops = pickletools.genops(stream)
        str_pushes = 0  # per pickle: STACK_GLOBAL needs two names on the stack
        while True:
            start = stream.tell()
            try:
                opcode, arg, _pos = next(ops)
            except StopIteration:
                break
            except ValueError:
                if _loader_reads_int(prefix, start):
                    return True
                if recoverable_global(prefix, start) is not None:
                    return True  # GLOBAL/INST line the loader reads without a final newline
                if prefix[start : start + 1] == b"P" and not reached_stop:
                    return False  # PERSID whose line runs past the probe: see below
                return reached_stop or (file_continues and stream.tell() >= len(prefix))
            if opcode.name == "STOP" and start == 0:
                # An empty first pickle: STOP pops an empty stack and every
                # loader fails here, before any later pickle is read.
                return False
            if opcode.name in ("PERSID", "BINPERSID") and not reached_stop:
                # The first pickle is read by a plain unpickler (pickle.load,
                # torch's magic-number load): no persistent_load, it stops here.
                return False
            if opcode.name in ("GLOBAL", "INST") and not _importable(arg) and not reached_stop:
                # A module/name that cannot exist (a CSV line, prose) makes the
                # loader fail right here: nothing after it runs, no pickle.
                return False
            if _resolves_global(opcode.name, arg, str_pushes):
                return True
            str_pushes += opcode.name in _MAY_PUSH_STR
        reached_stop = True
    return reached_stop


def recoverable_int_line(data: bytes, start: int) -> int | None:
    """Known genops/unpickler gap: INT and LONG lines are read with int(s) by
    genops but int(s, 0) by the unpickler, which also takes "0x1", "0o7"...
    Where genops stops and the loader goes on, the stream is a pickle. Returns
    the offset just past the line so a walk can resume, or None if the loader
    would fail here too (then genops and the loader agree: a real break)."""
    code = data[start : start + 1]
    end = data.find(b"\n", start)
    if code not in (b"I", b"L") or end == -1:
        return None
    line = data[start + 1 : end]
    try:
        int(line.removesuffix(b"L") if code == b"L" else line, 0)
    except ValueError:
        return None
    return end + 1


def _loader_reads_int(prefix: bytes, start: int) -> bool:
    return recoverable_int_line(prefix, start) is not None


def recoverable_global(data: bytes, start: int) -> tuple[str, str] | None:
    """Another genops/loader gap: GLOBAL (c) and INST (i) read two lines with
    readline()[:-1], which drops the final byte even with no trailing newline
    at end of file. So `cos\\nsystemX` resolves os.system (the X is eaten),
    which genops rejects for the missing newline. Returns (module, name) the
    loader would resolve, or None when it would fail here too."""
    if data[start : start + 1] not in (b"c", b"i"):
        return None
    nl = data.find(b"\n", start + 1)
    if nl == -1:
        return None  # no module line: the loader cannot resolve anything
    rest = data[nl + 1 :]
    name = rest[: rest.index(b"\n")] if b"\n" in rest else rest[:-1]  # readline()[:-1]
    if not name:
        return None
    try:
        module, name_str = data[start + 1 : nl].decode("utf-8"), name.decode("utf-8")
    except UnicodeDecodeError:
        return None  # a non-UTF-8 name: the loader's decode fails too
    # Only importable names resolve; a CSV line "ity,population"/"Paris,4" fails
    # to import, so the loader runs nothing (keeps text files out).
    if _is_dotted_identifier(module) and _is_dotted_identifier(name_str):
        return module, name_str
    return None


def _resolves_global(name: str, arg: object, str_pushes: int) -> bool:
    if name == "STACK_GLOBAL":
        # Alone (say, the low byte of a safetensors header size) it has nothing
        # to resolve and the loader fails. EXT1/2/4 are left out: the copyreg
        # extension registry is empty unless the loading process fills it.
        return str_pushes >= 2
    if name in ("GLOBAL", "INST"):
        return _importable(arg)
    return False


def _importable(arg: object) -> bool:
    """genops gives a GLOBAL/INST argument as "module name". Only dotted
    identifiers can be imported; anything else makes the loader fail."""
    parts = str(arg).split(" ")
    return len(parts) == 2 and all(_is_dotted_identifier(p) for p in parts)


# Opcodes that may leave a str on the stack, directly or from the memo.
_MAY_PUSH_STR = frozenset(
    {
        "STRING",
        "BINSTRING",
        "SHORT_BINSTRING",
        "UNICODE",
        "SHORT_BINUNICODE",
        "BINUNICODE",
        "BINUNICODE8",
        "GET",
        "BINGET",
        "LONG_BINGET",
        "DUP",
    }
)


def _is_dotted_identifier(text: str) -> bool:
    return all(part.isidentifier() for part in text.split("."))

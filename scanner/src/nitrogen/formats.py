"""Model file format detection from content (rule #4).

Only the bytes decide: file names and extensions are never consulted.
Nothing here deserializes; pickles are probed with `pickletools.genops`.
"""

import io
import pickletools
import re
import struct
import zipfile
from enum import Enum, StrEnum, auto
from typing import BinaryIO

# Bytes read to probe a pickle without magic (protocol 0/1). The probe runs on
# an in-memory copy: genops on the file itself would honour attacker-chosen
# lengths (a BINUNICODE announcing 4 GiB allocates 4 GiB).
PROBE_SIZE = 64 * 1024

# Local file header, or end of central directory for an empty archive. Same
# offset-0 rule as torch.load, so a file torch reads as a zip is a zip here.
_ZIP_MAGICS = (b"PK\x03\x04", b"PK\x05\x06")


class Format(StrEnum):
    PICKLE = "pickle"
    PYTORCH_ZIP = "pytorch-zip"
    ZIP = "zip"
    SAFETENSORS = "safetensors"
    UNKNOWN = "unknown"


def detect_format(fh: BinaryIO) -> Format:
    """Detect the format of a seekable binary stream, left rewound to offset 0."""
    fh.seek(0)
    try:
        return _detect(fh)
    finally:
        fh.seek(0)


def _detect(fh: BinaryIO) -> Format:
    prefix = fh.read(PROBE_SIZE)
    # PROTO opcode: pickle protocol 2 to 5 (also the legacy non-zip torch.save).
    if len(prefix) >= 2 and prefix[0] == 0x80 and 2 <= prefix[1] <= 5:
        return Format.PICKLE
    if prefix[:4] in _ZIP_MAGICS:
        return _zip_format(fh)
    probe = _probe_headless_pickle(prefix, file_continues=fh.read(1) != b"")
    # A pickle loader resolving a global wins over the loose safetensors check:
    # a safetensors/pickle polyglot runs code under torch.load.
    if probe is _Probe.RESOLVES_GLOBAL:
        return Format.PICKLE
    if _is_safetensors(prefix):
        return Format.SAFETENSORS
    if probe is _Probe.PLAUSIBLE:
        return Format.PICKLE
    return Format.UNKNOWN


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


def _is_safetensors(prefix: bytes) -> bool:
    """8-byte little-endian header size, then a JSON object.

    Deliberately loose: header size, JSON and offsets are validated by the
    safetensors parser, which reports a malformed file as a risk.
    """
    if len(prefix) < 9:
        return False
    (header_size,) = struct.unpack("<Q", prefix[:8])
    return header_size >= 2 and prefix[8:9] == b"{"


class _Probe(Enum):
    NOT_PICKLE = auto()
    PLAUSIBLE = auto()  # reaches STOP, or valid opcodes up to the probe limit
    RESOLVES_GLOBAL = auto()  # a loader would import something: always a pickle


def _probe_headless_pickle(prefix: bytes, *, file_continues: bool) -> _Probe:
    """Protocol 0/1 pickles have no magic: walk their opcodes instead.

    An unpickler acts on each opcode as it reads it, so what counts is what it
    does before the stream breaks, not whether the stream is well formed: a
    global resolved before garbage or a missing STOP is still executed. Pickles
    may be concatenated (legacy torch.save), so probing goes on after STOP.
    """
    stream = io.BytesIO(prefix)
    result = _Probe.NOT_PICKLE
    try:
        while stream.tell() < len(prefix):
            str_pushes = 0  # per pickle: STACK_GLOBAL needs two names on the stack
            for opcode, arg, _pos in pickletools.genops(stream):
                if _resolves_global(opcode.name, arg, str_pushes):
                    return _Probe.RESOLVES_GLOBAL
                str_pushes += opcode.name in _MAY_PUSH_STR
            result = _Probe.PLAUSIBLE
    except ValueError:
        if file_continues and stream.tell() >= len(prefix):
            return _Probe.PLAUSIBLE
    return result


def _resolves_global(name: str, arg: object, str_pushes: int) -> bool:
    if name == "STACK_GLOBAL":
        # Alone (say, the low byte of a safetensors header size) it has nothing
        # to resolve and the loader fails. EXT1/2/4 are left out: the copyreg
        # extension registry is empty unless the loading process fills it.
        return str_pushes >= 2
    if name in ("GLOBAL", "INST"):
        # genops gives "module name". Only importable names can run anything;
        # requiring them keeps text files starting with "c" or "i" out.
        parts = str(arg).split(" ")
        return len(parts) == 2 and all(_is_dotted_identifier(p) for p in parts)
    return False


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

"""Strict safetensors validation: header, tensor entries and data layout.

Layout: 8-byte little-endian header size N, N bytes of UTF-8 JSON (padded
with spaces), then the tensor data. Tensor data is never read: only the
header is, and its size is capped.

Aligned with the reference implementation (safetensors 0.8), and stricter
where a looser reading leaves room for two parsers to disagree: duplicate
keys, unknown fields in a tensor entry, header not starting with "{", and
padding other than spaces are all reported.
"""

import json
import struct
from typing import BinaryIO

from nitrogen.findings import Finding

# Same cap as the reference implementation.
MAX_HEADER_SIZE = 100_000_000

_U64_MAX = 2**64 - 1

# Bits per element. Sub-byte types must fill whole bytes.
_DTYPE_BITS = {
    "BOOL": 8,
    "U8": 8,
    "I8": 8,
    "F8_E5M2": 8,
    "F8_E4M3": 8,
    "F8_E8M0": 8,
    "I16": 16,
    "U16": 16,
    "F16": 16,
    "BF16": 16,
    "I32": 32,
    "U32": 32,
    "F32": 32,
    "I64": 64,
    "U64": 64,
    "F64": 64,
    "C64": 64,
    "F4": 4,
    "F6_E2M3": 6,
    "F6_E3M2": 6,
}

_TENSOR_FIELDS = {"dtype", "shape", "data_offsets"}


class _Malformed(Exception):
    def __init__(self, rule: str, message: str) -> None:
        super().__init__(message)
        self.rule = rule


def check_safetensors(fh: BinaryIO, size: int) -> list[Finding]:
    """Validate a seekable safetensors stream of `size` bytes, left rewound."""
    fh.seek(0)
    try:
        header = _read_header(fh, size)
        tensors = _parse_header(header)
        _check_layout(tensors, data_size=size - 8 - len(header))
    except _Malformed as exc:
        return [Finding(f"safetensors.{exc.rule}", str(exc))]
    finally:
        fh.seek(0)
    return []


def _read_header(fh: BinaryIO, size: int) -> bytes:
    size_field = fh.read(8)
    if len(size_field) < 8:
        raise _Malformed("header-size", "file shorter than the 8-byte header size")
    (header_size,) = struct.unpack("<Q", size_field)
    # Checked before reading: an announced size is never trusted for a read.
    if header_size > MAX_HEADER_SIZE:
        raise _Malformed("header-size", f"header of {header_size} bytes exceeds the 100 MB cap")
    if header_size > size - 8:
        raise _Malformed("header-size", f"header of {header_size} bytes exceeds the file")
    header = fh.read(header_size)
    if len(header) != header_size:
        raise _Malformed("header-size", "file shorter than its header (changed while scanned?)")
    return header


def _parse_header(header: bytes) -> list[tuple[str, int, int]]:
    """Return (name, begin, end) for each tensor."""
    try:
        # Decoded first: json.loads would accept UTF-16/32 bytes.
        text = header.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise _Malformed("header-encoding", f"header is not UTF-8: {exc.reason}") from exc
    body = text.rstrip(" ")
    if not body.startswith("{") or not body.endswith("}"):
        raise _Malformed("header-json", "header must be a JSON object padded with spaces only")
    try:
        root = json.loads(
            body,
            object_pairs_hook=_checked_object,
            parse_constant=_no_constant,
            parse_int=_unsigned_int,
        )
    except _Malformed:
        raise
    except (ValueError, RecursionError) as exc:
        # ValueError covers invalid JSON and integers past the str-digits limit;
        # RecursionError, deeply nested arrays.
        raise _Malformed("header-json", f"invalid JSON header: {exc}") from exc
    # A body framed by "{" and "}" that parses is an object: root is a dict.

    tensors = []
    for name, entry in root.items():
        if name == "__metadata__":
            _check_metadata(entry)
        else:
            tensors.append((name, *_check_tensor(name, entry)))
    return tensors


def _checked_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    seen: dict[str, object] = {}
    for key, value in pairs:
        # Python keeps the last duplicate, other readers may keep the first.
        if key in seen:
            raise _Malformed("header-json", f"duplicate key {key!r} in header")
        # Python keeps a lone surrogate such as "\ud800"; UTF-8 readers refuse it.
        for text in (key, value) if isinstance(value, str) else (key,):
            if not _is_unicode(text):
                raise _Malformed("header-json", f"invalid escaped unicode in {text!r}")
        seen[key] = value
    return seen


def _is_unicode(text: str) -> bool:
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _no_constant(name: str) -> object:
    raise _Malformed("header-json", f"non-standard JSON constant {name}")


def _unsigned_int(literal: str) -> int:
    # Every integer in a header is unsigned. Python reads "-0" as 0, the
    # reference refuses it: no sign at all.
    if literal.startswith("-"):
        raise _Malformed("header-json", f"negative integer {literal[:20]} in header")
    return int(literal)


def _check_metadata(metadata: object) -> None:
    if not isinstance(metadata, dict) or not all(isinstance(v, str) for v in metadata.values()):
        raise _Malformed("metadata", "__metadata__ must map strings to strings")


def _check_tensor(name: str, entry: object) -> tuple[int, int]:
    """Validate one tensor entry and return its (begin, end) data offsets."""

    def bad(reason: str) -> _Malformed:
        return _Malformed("tensor", f"tensor {name!r}: {reason}")

    if not isinstance(entry, dict) or entry.keys() != _TENSOR_FIELDS:
        raise bad(f"expected exactly the fields {sorted(_TENSOR_FIELDS)}")
    dtype, shape, offsets = entry["dtype"], entry["shape"], entry["data_offsets"]
    # isinstance first: a list or dict dtype is unhashable and would raise.
    if not isinstance(dtype, str) or dtype not in _DTYPE_BITS:
        raise bad(f"unknown dtype {_short(dtype)}")
    if not isinstance(shape, list) or not all(_is_u64(d) for d in shape):
        raise bad("shape must be a list of integers between 0 and 2**64 - 1")
    if not (isinstance(offsets, list) and len(offsets) == 2 and all(map(_is_u64, offsets))):
        raise bad("data_offsets must be two integers between 0 and 2**64 - 1")
    begin, end = offsets
    if begin > end:
        raise bad(f"data_offsets [{begin}, {end}] end before they begin")
    elements = 1
    for dim in shape:
        # Bounded as we go: a huge shape list would otherwise build an integer
        # of millions of digits, one slow multiplication at a time.
        elements *= dim
        if elements > _U64_MAX:
            raise bad(f"shape {_short(shape)} holds more than 2**64 - 1 elements")
    bits = elements * _DTYPE_BITS[dtype]
    if bits % 8:
        raise bad(f"{elements} x {dtype} does not fill whole bytes")
    if end - begin != bits // 8:
        raise bad(
            f"shape {_short(shape)} x {dtype} needs {bits // 8} bytes, offsets give {end - begin}"
        )
    return begin, end


def _short(value: object, limit: int = 80) -> str:
    # Findings end up in terminals and AI-BOMs: never echo 100 MB of header.
    text = repr(value)
    return text if len(text) <= limit else text[:limit] + "..."


def _is_u64(value: object) -> bool:
    # type() rather than isinstance(): True and False are ints in Python.
    return type(value) is int and 0 <= value <= _U64_MAX


def _check_layout(tensors: list[tuple[str, int, int]], data_size: int) -> None:
    """Tensors must tile the data section exactly: from 0, no gap, no overlap."""
    position = 0
    for name, begin, end in sorted(tensors, key=lambda t: (t[1], t[2])):
        if begin != position:
            kind = "gap before" if begin > position else "overlap with"
            raise _Malformed(
                "layout", f"tensor {name!r} starts at {begin}, expected {position} ({kind} it)"
            )
        position = end
    if position > data_size:
        raise _Malformed(
            "layout", f"tensors need {position} bytes, the data section has {data_size}"
        )
    if position < data_size:
        raise _Malformed(
            "layout", f"{data_size - position} byte(s) after the last tensor belong to no tensor"
        )

"""Strict safetensors header validation (step 3).

Healthy files come from the reference serializer; malformed ones are built
byte by byte. The reference library only parses, it never executes anything.
"""

import io
import json
import struct

import numpy as np
import pytest
from safetensors.numpy import save

from nitrogen.findings import Finding
from nitrogen.safetensors import MAX_HEADER_SIZE, check_safetensors


def _check(blob: bytes) -> list[Finding]:
    return check_safetensors(io.BytesIO(blob), len(blob))


def _rules(blob: bytes) -> list[str]:
    return [f.rule for f in _check(blob)]


def _build(header: bytes, data: bytes = b"") -> bytes:
    return struct.pack("<Q", len(header)) + header + data


def _json(obj: object) -> bytes:
    return json.dumps(obj, separators=(",", ":")).encode()


def _tensor(
    begin: int, end: int, shape: list[object] | None = None, dtype: str = "F32"
) -> dict[str, object]:
    return {"dtype": dtype, "shape": [2] if shape is None else shape, "data_offsets": [begin, end]}


ONE_TENSOR = _json({"a": _tensor(0, 8)})


# --- healthy files ------------------------------------------------------------


def test_reference_file_passes() -> None:
    blob = save(
        {
            "w": np.zeros((2, 3), dtype=np.float32),
            "b": np.zeros(3, dtype=np.uint8),
            "empty": np.zeros((0, 4), dtype=np.float16),
            "scalar": np.zeros((), dtype=np.int64),
        },
        metadata={"format": "pt"},
    )
    assert _check(blob) == []


def test_minimal_files_pass() -> None:
    assert _check(_build(b"{}")) == []
    assert _check(_build(_json({"__metadata__": {"k": "v"}}))) == []
    assert _check(_build(ONE_TENSOR, bytes(8))) == []


def test_space_padding_passes() -> None:
    assert _check(_build(ONE_TENSOR + b"     ", bytes(8))) == []


def test_tensors_in_any_order_pass() -> None:
    header = _json({"b": _tensor(8, 16), "a": _tensor(0, 8)})
    assert _check(_build(header, bytes(16))) == []


@pytest.mark.parametrize(
    ("dtype", "count", "nbytes"),
    [
        ("BOOL", 3, 3),
        ("U8", 3, 3),
        ("I8", 3, 3),
        ("F8_E5M2", 3, 3),
        ("F8_E4M3", 3, 3),
        ("F8_E8M0", 3, 3),
        ("I16", 3, 6),
        ("U16", 3, 6),
        ("F16", 3, 6),
        ("BF16", 3, 6),
        ("I32", 3, 12),
        ("U32", 3, 12),
        ("F32", 3, 12),
        ("I64", 3, 24),
        ("U64", 3, 24),
        ("F64", 3, 24),
        ("C64", 3, 24),
        ("F4", 4, 2),
        ("F6_E2M3", 4, 3),
        ("F6_E3M2", 4, 3),
    ],
)
def test_every_known_dtype_passes(dtype: str, count: int, nbytes: int) -> None:
    header = _json({"a": _tensor(0, nbytes, [count], dtype)})
    assert _check(_build(header, bytes(nbytes))) == []


def test_stream_is_rewound() -> None:
    fh = io.BytesIO(_build(ONE_TENSOR, bytes(8)))
    fh.seek(3)
    check_safetensors(fh, len(fh.getvalue()))
    assert fh.tell() == 0


# --- header size ----------------------------------------------------------------


def test_file_shorter_than_size_field() -> None:
    assert _rules(b"\x02\x00\x00") == ["safetensors.header-size"]


def test_header_size_beyond_file() -> None:
    assert _rules(struct.pack("<Q", 100) + b"{}") == ["safetensors.header-size"]


def test_header_size_over_cap_is_not_read() -> None:
    # Checked before any read: a 100 MB+ header is never loaded in memory.
    blob = struct.pack("<Q", MAX_HEADER_SIZE + 1) + b"{}"
    fh = io.BytesIO(blob)
    assert [f.rule for f in check_safetensors(fh, MAX_HEADER_SIZE + 100)] == [
        "safetensors.header-size"
    ]


def test_stream_shorter_than_announced_size() -> None:
    # The file shrank between hashing and analysis.
    blob = _build(ONE_TENSOR, bytes(8))
    assert [f.rule for f in check_safetensors(io.BytesIO(blob[:20]), len(blob))] == [
        "safetensors.header-size"
    ]


# --- header encoding and JSON ---------------------------------------------------


def test_non_utf8_header() -> None:
    assert _rules(_build(b'{"__metadata__":{"k":"\xff"}}')) == ["safetensors.header-encoding"]


def test_utf16_header() -> None:
    # json.loads would happily decode UTF-16 bytes; the reference does not.
    assert _rules(_build('{"a":1}'.encode("utf-16"))) == ["safetensors.header-encoding"]


@pytest.mark.parametrize(
    "header",
    [
        b"",  # empty
        b"{",  # truncated
        b"[]",  # not an object
        b" " + ONE_TENSOR,  # must start with "{"
        b'{"__metadata__":{"k":NaN}}',  # non-standard constant
        b'{"__metadata__":{"k":"1"},"__metadata__":{"k":"2"}}',  # duplicate key
        b'{"a":' + _json(_tensor(0, 8)) + b',"a":' + _json(_tensor(0, 8)) + b"}",
        b'{"__metadata__":' + b"[" * 100_000 + b"]" * 100_000 + b"}",  # RecursionError
        b'{"__metadata__":{"k":' + b"9" * 5000 + b"}}",  # int beyond str-digits limit
    ],
    ids=[
        "empty",
        "truncated",
        "not-object",
        "leading-space",
        "nan",
        "dup-metadata",
        "dup-tensor",
        "deep-nesting",
        "huge-int",
    ],
)
def test_invalid_json(header: bytes) -> None:
    assert _rules(_build(header, bytes(8))) == ["safetensors.header-json"]


@pytest.mark.parametrize("padding", [b"\n", b"\t", b"\r", b"\x00", b" x"])
def test_padding_other_than_spaces(padding: bytes) -> None:
    # The reference serializer pads with spaces only.
    assert _rules(_build(ONE_TENSOR + padding, bytes(8))) == ["safetensors.header-json"]


# --- metadata -------------------------------------------------------------------


@pytest.mark.parametrize(
    "header",
    [
        b'{"a":{"dtype":"F32","shape":[-0,2],"data_offsets":[0,0]}}',
        b'{"a":{"dtype":"F32","shape":[-2],"data_offsets":[0,8]}}',
        b'{"a":{"dtype":"F32","shape":[2],"data_offsets":[-0,8]}}',
        b'{"\\ud800":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}',
        b'{"__metadata__":{"k":"\\udc00"}}',
        b'{"__metadata__":{"\\udc00":"v"}}',
    ],
    ids=[
        "minus-zero-shape",
        "negative-shape",
        "minus-zero-offset",
        "surrogate-name",
        "surrogate-value",
        "surrogate-key",
    ],
)
def test_what_the_reference_rejects_is_reported(header: bytes) -> None:
    # Python reads "-0" as 0 and keeps lone surrogates; the reference refuses both.
    # Data sized to the header, so the layout check cannot be what fires.
    blob = _build(header, bytes(8) if b"[0,8]" in header or b"[-0,8]" in header else b"")
    assert _rules(blob) == ["safetensors.header-json"]


@pytest.mark.parametrize("metadata", ["x", {"k": 1}, {"k": None}, {"k": ["v"]}, [], None])
def test_metadata_must_map_strings_to_strings(metadata: object) -> None:
    assert _rules(_build(_json({"__metadata__": metadata}))) == ["safetensors.metadata"]


# --- tensor entries -------------------------------------------------------------


@pytest.mark.parametrize(
    "entry",
    [
        "x",  # not an object
        ["F32", [2], [0, 8]],  # array form (the reference accepts it, we do not)
        _tensor(0, 8, dtype=[]),  # type: ignore[arg-type]  # unhashable dtype
        _tensor(0, 8, dtype={}),  # type: ignore[arg-type]  # unhashable dtype
        _tensor(0, 8, dtype={"F32": None}),  # type: ignore[arg-type]  # enum as a map
        {"dtype": "F32", "shape": [2]},  # missing data_offsets
        {**_tensor(0, 8), "extra": 1},  # unknown field (the reference ignores it)
        _tensor(0, 8, dtype="F128"),  # unknown dtype
        _tensor(0, 8, dtype="f32"),  # dtypes are case sensitive
        _tensor(0, 8, shape=[2.0]),  # float dimension
        _tensor(0, 8, shape=[True, 2]),  # bool is not an int
        _tensor(0, 8, shape=[2**63, 4]),  # more than 2**64 - 1 elements
        {"dtype": "F32", "shape": 2, "data_offsets": [0, 8]},  # shape not a list
        {"dtype": "F32", "shape": [2], "data_offsets": [0]},  # one offset
        {"dtype": "F32", "shape": [2], "data_offsets": [0.0, 8.0]},  # float offsets
        {"dtype": "F32", "shape": [2], "data_offsets": [False, 8]},  # bool offset
        _tensor(8, 0),  # begin after end
        _tensor(0, 4),  # 2 x F32 needs 8 bytes
        _tensor(0, 1, shape=[1], dtype="F4"),  # 4 bits do not fill a byte
        _tensor(0, 0, shape=[0, 2**64]),  # dimension beyond u64
        _tensor(0, 2**64, shape=[2**62]),  # offset beyond u64
    ],
)
def test_malformed_tensor_entry(entry: object) -> None:
    assert _rules(_build(_json({"a": entry}), bytes(8))) == ["safetensors.tensor"]


# --- data layout ----------------------------------------------------------------


def test_data_must_start_at_zero() -> None:
    assert _rules(_build(_json({"a": _tensor(4, 12)}), bytes(12))) == ["safetensors.layout"]


def test_gap_between_tensors() -> None:
    header = _json({"a": _tensor(0, 8), "b": _tensor(12, 20)})
    assert _rules(_build(header, bytes(20))) == ["safetensors.layout"]


def test_overlapping_tensors() -> None:
    header = _json({"a": _tensor(0, 8), "b": _tensor(4, 12)})
    assert _rules(_build(header, bytes(12))) == ["safetensors.layout"]


def test_tensor_beyond_end_of_file() -> None:
    assert _rules(_build(ONE_TENSOR, bytes(4))) == ["safetensors.layout"]


def test_trailing_bytes_after_last_tensor() -> None:
    # Bytes no tensor accounts for: room for a hidden payload.
    assert _rules(_build(ONE_TENSOR, bytes(8) + b"payload")) == ["safetensors.layout"]


def test_trailing_bytes_without_tensors() -> None:
    assert _rules(_build(b"{}", b"x")) == ["safetensors.layout"]


def test_finding_message_names_the_tensor() -> None:
    header = _json({"a": _tensor(0, 8), "b": _tensor(12, 20)})
    [finding] = _check(_build(header, bytes(20)))
    assert "'b'" in finding.message

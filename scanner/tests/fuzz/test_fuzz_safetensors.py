"""Safetensors parser on random input, and against the reference implementation.

Soundness: whatever we accept, the reference accepts too (we may be stricter,
never looser). No false positive: whatever the reference serializer writes,
we accept.
"""

import io
import json
import struct

import numpy as np
import safetensors
from hypothesis import given, settings
from hypothesis import strategies as st
from safetensors.numpy import save

from nitrogen.formats import Format, detect_formats
from nitrogen.safetensors import _DTYPE_BITS, check_safetensors

DTYPES = [*_DTYPE_BITS, "X"]  # every dtype we accept, plus an unknown one
NUMPY_DTYPES = [np.bool_, np.uint8, np.int8, np.int16, np.float16, np.int32, np.float32]


def _accepted(blob: bytes) -> bool:
    return check_safetensors(io.BytesIO(blob), len(blob)) == []


def _reference_accepts(blob: bytes) -> bool:
    try:
        safetensors.deserialize(blob)  # type: ignore[no-untyped-call]  # untyped library
    except Exception:
        return False
    return True


@given(st.binary(max_size=2048))
def test_random_bytes_never_raise(data: bytes) -> None:
    check_safetensors(io.BytesIO(data), len(data))


@given(st.integers(min_value=0, max_value=2**64 - 1), st.binary(max_size=512))
def test_random_size_field_never_raises(size: int, data: bytes) -> None:
    blob = struct.pack("<Q", size) + data
    check_safetensors(io.BytesIO(blob), len(blob))


JSON = st.recursive(
    st.none() | st.booleans() | st.integers() | st.floats() | st.text(max_size=8),
    lambda children: (
        st.lists(children, max_size=4) | st.dictionaries(st.text(max_size=8), children, max_size=4)
    ),
    max_leaves=20,
)


@given(st.dictionaries(st.text(max_size=8), JSON, max_size=4), st.binary(max_size=64))
def test_random_json_headers_never_raise(header: dict[str, object], data: bytes) -> None:
    raw = json.dumps(header).encode()
    blob = struct.pack("<Q", len(raw)) + raw + data
    check_safetensors(io.BytesIO(blob), len(blob))


TENSOR = st.fixed_dictionaries(
    {
        "dtype": st.sampled_from(DTYPES),
        "shape": st.lists(st.integers(min_value=-1, max_value=4), max_size=3),
        "data_offsets": st.lists(st.integers(min_value=-1, max_value=40), min_size=2, max_size=2),
    }
)


@settings(max_examples=300)
@given(
    st.dictionaries(st.sampled_from(["a", "b", "c"]), TENSOR, max_size=3),
    st.integers(min_value=0, max_value=48),
    st.sampled_from([b"", b" ", b"    "]),
)
def test_we_never_accept_what_the_reference_rejects(
    tensors: dict[str, object], data_size: int, padding: bytes
) -> None:
    raw = json.dumps(tensors, separators=(",", ":")).encode() + padding
    blob = struct.pack("<Q", len(raw)) + raw + bytes(data_size)
    if _accepted(blob):
        assert _reference_accepts(blob)


@settings(max_examples=100)
@given(
    st.dictionaries(
        st.text(min_size=1, max_size=8).filter(lambda s: s != "__metadata__"),
        st.tuples(st.sampled_from(NUMPY_DTYPES), st.lists(st.integers(0, 4), max_size=3)),
        max_size=4,
    ),
    st.none() | st.dictionaries(st.text(max_size=8), st.text(max_size=8), max_size=3),
)
def test_reference_files_are_accepted(
    arrays: dict[str, tuple[type, list[int]]], metadata: dict[str, str] | None
) -> None:
    blob = save({k: np.zeros(shape, dtype=dt) for k, (dt, shape) in arrays.items()}, metadata)
    # safetensors 0.8 writes an invalid header for no tensor + empty metadata
    # ('{},"__metadata__":{}}') and then rejects it itself: only what the
    # reference reads back is required to pass.
    if _reference_accepts(blob):
        assert _accepted(blob)


def test_reference_bug_on_empty_file_is_rejected_by_both() -> None:
    blob = save({}, {})
    assert not _reference_accepts(blob)
    assert not _accepted(blob)


@settings(max_examples=300)
@given(
    st.dictionaries(st.sampled_from(["a", "b"]), TENSOR, max_size=2),
    st.sampled_from([b"", b" ", b"\t", b"\n", b"\r\n "]),
    st.integers(min_value=0, max_value=1500),
    st.sampled_from([b" ", b"\t", b"\n"]),
    st.integers(min_value=0, max_value=48),
)
def test_whatever_the_reference_loads_is_detected_as_safetensors(
    tensors: dict[str, object], lead: bytes, pad: int, pad_char: bytes, data_size: int
) -> None:
    # Pipeline-level differential (review C1/C2): detection must route every
    # file the reference accepts to the safetensors checks. Padding moves the
    # size field over pickle and zip magic values (e.g. 0x280 = PROTO 2).
    raw = lead + json.dumps(tensors, separators=(",", ":")).encode() + pad_char * pad
    blob = struct.pack("<Q", len(raw)) + raw + bytes(data_size)
    if _reference_accepts(blob):
        assert Format.SAFETENSORS in detect_formats(io.BytesIO(blob))

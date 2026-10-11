"""Format detection on random or malformed input: never raises, never hangs."""

import io

from hypothesis import given
from hypothesis import strategies as st

from nitrogen.formats import Format, detect_formats


@given(st.binary(max_size=4096))
def test_never_raises(data: bytes) -> None:
    assert detect_formats(io.BytesIO(data)) <= set(Format)


@given(
    st.sampled_from([b"\x80\x02", b"PK\x03\x04", b"PK\x05\x06", b"\x10" + b"\x00" * 7 + b"{"]),
    st.binary(max_size=4096),
)
def test_magic_prefixes_never_raise(prefix: bytes, data: bytes) -> None:
    assert detect_formats(io.BytesIO(prefix + data)) <= set(Format)

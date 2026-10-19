"""The pickle analyzer on random or malformed input: never raises, never hangs,
never allocates on an attacker-declared length."""

import io

from hypothesis import given, settings
from hypothesis import strategies as st

from nitrogen.findings import Finding
from nitrogen.pickle_scan import check_pickle, scan_pickle

OPCODES = bytes(range(0x20, 0x98))  # the printable + binary opcode range


@given(st.binary(max_size=4096))
def test_random_bytes_never_raise(data: bytes) -> None:
    result = scan_pickle(data)
    assert all(isinstance(f, Finding) for f in result)


@given(st.lists(st.sampled_from(OPCODES), max_size=200).map(bytes))
def test_opcode_soup_never_raises(data: bytes) -> None:
    scan_pickle(b"\x80\x04" + data)


@settings(max_examples=50)
@given(st.integers(min_value=0, max_value=2**64 - 1))
def test_declared_length_never_allocates(declared: int) -> None:
    # BINUNICODE8 announcing up to 2**64 bytes, with no data behind it: the
    # analyzer reads a bounded buffer, so this is a finding, not an allocation.
    blob = b"\x80\x04\x8d" + declared.to_bytes(8, "little")
    result = scan_pickle(blob)
    assert all(isinstance(f, Finding) for f in result)


@given(st.binary(max_size=2048))
def test_detected_pickle_returning_clean_reached_stop(data: bytes) -> None:
    # The invariant that would have caught C1: if the file is detected as a
    # (sole) pickle and the analyzer finds nothing, genops must have reached a
    # STOP. A clean result on a broken-but-executable stream is a false negative.
    from nitrogen.formats import Format, detect_formats

    if detect_formats(io.BytesIO(data)) != {Format.PICKLE}:
        return
    if scan_pickle(data) == []:
        assert _reaches_stop(data)


def _reaches_stop(data: bytes) -> bool:
    import pickletools

    stream = io.BytesIO(data)
    try:
        while stream.tell() < len(data):
            for opcode, _arg, _pos in pickletools.genops(stream):
                if opcode.name == "STOP":
                    break
            else:
                return False
    except ValueError:
        return False
    return True


def test_huge_file_is_capped_not_loaded() -> None:
    # A declared-small but very long stream: check_pickle reads a bounded prefix.
    data = b"\x80\x02" + b"K\x01" * (20 * 1024 * 1024) + b"."
    result = check_pickle(io.BytesIO(data), len(data))
    assert all(isinstance(f, Finding) for f in result)

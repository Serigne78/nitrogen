"""Direct unit tests for the pickle analyzer's mechanisms."""

import pytest

from nitrogen.pickle_scan import scan_pickle


def _rules(data: bytes) -> list[str]:
    return [f.rule for f in scan_pickle(data)]


def _details(data: bytes) -> list[str]:
    return [f.message for f in scan_pickle(data)]


def test_harmless_pickle_has_no_finding() -> None:
    assert scan_pickle(b"\x80\x02K\x01.") == []  # PROTO, BININT1, STOP


def test_allowlisted_global_passes() -> None:
    assert scan_pickle(b"\x80\x02ccollections\nOrderedDict\n)R.") == []


def test_dangerous_global_is_flagged() -> None:
    assert _details(b"cos\nsystem\n(S'x'\ntR.") == ["os.system"]


def test_unresolved_stack_global_is_flagged() -> None:
    # STACK_GLOBAL with no concrete strings beneath it: cannot vouch for it.
    assert _rules(b"\x80\x04K\x01K\x02\x93.") == ["pickle.import"]
    assert _details(b"\x80\x04K\x01K\x02\x93.") == ["unresolved dynamic global"]


@pytest.mark.parametrize("code", [b"\x82\x01", b"\x83\x01\x00", b"\x84\x01\x00\x00\x00"])
def test_ext_opcodes_are_flagged(code: bytes) -> None:
    assert _rules(b"\x80\x02" + code + b".") == ["pickle.import"]


def test_appends_then_dangerous_stack_global_still_flagged() -> None:
    # APPENDS modifies a list below a mark; the stack must stay aligned so the
    # STACK_GLOBAL after it still resolves os.system (reviewer M1).
    data = (
        b"\x80\x04]("  # EMPTY_LIST, MARK
        b"\x8c\x01a\x8c\x01be"  # two strings, APPENDS
        b"\x8c\x02os\x8c\x06system\x93"  # STACK_GLOBAL os.system
        b"."
    )
    assert _details(data) == ["os.system"]


def test_concatenated_proto_then_int_gap() -> None:
    # A clean first pickle, then a headless second one using the int(s,0) gap.
    data = b"\x80\x02K\x01." + b"I0x1\ncos\nsystem\n(S'x'\ntR."
    assert _details(data) == ["os.system"]


def test_malformed_only_with_proto_or_import() -> None:
    assert _rules(b"\x80\x02(K\x01") == ["pickle.malformed"]  # proto magic, broken
    assert scan_pickle(b"]q\x00(K\x01e") == []  # headless, breaks, not a pickle


def test_polyglot_parse_error_is_suppressed() -> None:
    # sole_format=False: a broken pickle reading of a file valid in another
    # format is not reported, but a resolved import still is.
    assert scan_pickle(b"\x80\x02(K\x01", sole_format=False) == []
    assert [f.message for f in scan_pickle(b"cos\nsystem\n(S'x'\ntR.", sole_format=False)] == [
        "os.system"
    ]


def test_stack_global_on_empty_stack_resolves_nothing() -> None:
    # The loader crashes on STACK_GLOBAL with an empty stack before resolving
    # anything: no import finding (reviewer step 7, I1).
    assert "unresolved dynamic global" not in _details(b"\x93.")
    assert "unresolved dynamic global" not in _details(b"\x80\x04\x93.")

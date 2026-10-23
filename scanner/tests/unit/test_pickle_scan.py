"""Direct unit tests for the pickle analyzer's mechanisms."""

import pytest

from nitrogen.pickle_scan import scan_pickle


def _rules(data: bytes) -> list[str]:
    return [f.rule for f in scan_pickle(data)]


def _details(data: bytes) -> list[str]:
    return [f.message for f in scan_pickle(data)]


def test_harmless_pickle_has_no_finding() -> None:
    assert scan_pickle(b"\x80\x02K\x01.") == []  # PROTO, BININT1, STOP


def test_trailing_storage_after_a_complete_pickle_is_not_malformed() -> None:
    # ADR 0010: legacy torch .bin appends raw tensor storage after its pickles.
    # Once a pickle has completed (STOP), bytes that do not even begin a pickle
    # are that storage, not a malformed pickle. "opcode b'\x00' unknown" was the
    # false positive on gpt2 and t5-small.
    assert scan_pickle(b"\x80\x02K\x01." + b"\x00" * 256) == []


def test_trailing_storage_after_a_dangerous_pickle_still_reports_the_import() -> None:
    # Suppressing the trailing-bytes "malformed" must not drop a real finding
    # from the pickle that preceded it.
    assert [f.rule for f in scan_pickle(b"cos\nsystem\n(S'x'\ntR." + b"\x00" * 64)] == [
        "pickle.import"
    ]


def test_truncated_sole_pickle_is_still_malformed() -> None:
    # No pickle completed before the break: a genuinely truncated pickle. The
    # trailing-storage rule must not swallow this (it only applies after a STOP).
    assert _rules(b"\x80\x02(K\x01") == ["pickle.malformed"]  # MARK, BININT1, no STOP


def test_truncated_second_pickle_is_still_malformed() -> None:
    # A complete pickle, then a second one that starts (makes progress) but is
    # truncated before STOP: that is malformed, not trailing storage.
    assert _rules(b"\x80\x02K\x01." + b"\x80\x02(K\x01") == ["pickle.malformed"]


def test_payload_after_a_complete_pickle_and_a_non_opcode_is_an_accepted_blind_spot() -> None:
    # Reviewer frontier case: a complete pickle, then a byte that is not a pickle
    # opcode (\x00), then a dangerous pickle. No standard loader executes the
    # third: pickle.load stops at the first STOP; a concatenation loader crashes
    # on the \x00 and never resyncs; legacy torch reads the rest as storage by
    # offset. So this is clean (ADR 0010). If a loader that resyncs is ever in
    # scope, this must be revisited.
    assert scan_pickle(b"\x80\x02K\x01." + b"\x00" + b"cos\nsystem\n(S'x'\ntR.") == []


def test_dangerous_first_pickle_before_the_non_opcode_is_still_flagged() -> None:
    # The suppression only hides what comes AFTER a clean pickle; a dangerous
    # import in the pickle that precedes the storage is always reported.
    assert _rules(b"cos\nsystem\n(S'x'\ntR." + b"\x00" + b"\x80\x02K\x01.") == ["pickle.import"]


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

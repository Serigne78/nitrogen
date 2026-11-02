"""The target parser on random input: never raises anything but ValueError,
never hangs. A local path is always accepted; an hf:// target is either a valid
HfSource or a ValueError."""

from hypothesis import given
from hypothesis import strategies as st

from nitrogen.source import HfSource, LocalSource, parse_target


@given(st.text(max_size=200))
def test_parse_target_never_crashes(target: str) -> None:
    try:
        result = parse_target(target)
    except ValueError:
        return
    assert isinstance(result, (LocalSource, HfSource))


@given(st.from_regex(r"hf://[\w.-]+/[\w.-]+(@[\w.-]+)?", fullmatch=True))
def test_valid_hf_targets_parse(target: str) -> None:
    assert isinstance(parse_target(target), HfSource)

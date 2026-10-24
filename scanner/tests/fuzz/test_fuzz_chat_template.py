"""The Jinja template analyzer on random or malformed source: never raises,
never hangs, and a template jinja2 cannot parse yields a finding, not an
exception. The template is parsed with Environment().parse() only — never
rendered or compiled (ADR 0013)."""

from hypothesis import given, settings
from hypothesis import strategies as st
from nitrogen.chat_template import scan_template

from nitrogen.findings import Finding

_TOKENS = [
    "{{",
    "}}",
    "{%",
    "%}",
    "__class__",
    ".",
    "|attr(",
    "'",
    "x",
    " ",
    "for",
    "in",
    "messages",
]


@settings(max_examples=300)
@given(st.text(max_size=2000))
def test_random_text_never_raises(source: str) -> None:
    assert all(isinstance(f, Finding) for f in scan_template(source))


@given(st.lists(st.sampled_from(_TOKENS), max_size=200).map("".join))
def test_jinja_like_soup_never_raises(source: str) -> None:
    assert all(isinstance(f, Finding) for f in scan_template(source))

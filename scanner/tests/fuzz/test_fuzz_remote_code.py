"""The .py AST analyzer on random or malformed source: never raises, never
hangs, and an unparseable file yields a finding, not an exception. The code is
parsed with ast.parse only — never imported or executed (rule 1, ADR 0011)."""

from hypothesis import given, settings
from hypothesis import strategies as st
from nitrogen.remote_code import scan_python_source

from nitrogen.findings import Finding

_TOKENS = [
    "import",
    "os",
    "system",
    "subprocess",
    "eval",
    "exec",
    "(",
    ")",
    "\n",
    ".",
    "=",
    " ",
    "x",
]


@settings(max_examples=300)
@given(st.text(max_size=2000))
def test_random_text_never_raises(source: str) -> None:
    result = scan_python_source(source)
    assert all(isinstance(f, Finding) for f in result)


@given(st.lists(st.sampled_from(_TOKENS), max_size=200).map("".join))
def test_python_like_soup_never_raises(source: str) -> None:
    assert all(isinstance(f, Finding) for f in scan_python_source(source))

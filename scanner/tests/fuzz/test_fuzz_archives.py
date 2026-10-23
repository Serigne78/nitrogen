"""The archive layer on random member names and bytes: never raises, never
hangs, and the storage-skip rule (ADR 0010) stays deterministic. The step-9
false positives all came from this layer, so it earns its own fuzzing."""

import io
import zipfile

from hypothesis import given, settings
from hypothesis import strategies as st

from nitrogen.archives import check_zip
from nitrogen.findings import Finding

# A member name built from the pieces that matter to the skip rule: roots,
# `data`, separators (including the back-slash the rule must not normalize),
# digits, dotted names and traversal.
_PART = st.sampled_from(["archive", "data", "data.pkl", "0", "12", "5x", "..", "a", "b", "\\", "/"])
_NAME = (
    st.lists(_PART, min_size=1, max_size=6).map("".join).filter(lambda n: n and not n.endswith("/"))
)
_CONTENT = st.binary(max_size=64)


def _zip(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in members.items():
            zf.writestr(name, content)
    return buf.getvalue()


@settings(max_examples=200)
@given(st.dictionaries(_NAME, _CONTENT, min_size=1, max_size=6))
def test_check_zip_never_raises_on_random_members(members: dict[str, bytes]) -> None:
    result = check_zip(io.BytesIO(_zip(members)), 0)
    assert all(isinstance(f, Finding) for f in result)


@given(st.dictionaries(_NAME, _CONTENT, min_size=1, max_size=6))
def test_storage_skip_is_deterministic(members: dict[str, bytes]) -> None:
    data = _zip(members)
    first = [(f.rule, f.message) for f in check_zip(io.BytesIO(data), 0)]
    second = [(f.rule, f.message) for f in check_zip(io.BytesIO(data), 0)]
    assert first == second

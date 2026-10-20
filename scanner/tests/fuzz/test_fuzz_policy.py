"""The policy parser on random input: only ever a Policy or a PolicyError."""

import contextlib

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from nitrogen.policy import Policy, PolicyError, load_policy


@settings(max_examples=200)
@given(data=st.binary(max_size=512))
def test_random_bytes_give_policy_or_policy_error(
    data: bytes, tmp_path_factory: pytest.TempPathFactory
) -> None:
    path = tmp_path_factory.mktemp("p") / "policy.yaml"
    path.write_bytes(data)
    with contextlib.suppress(PolicyError):
        assert isinstance(load_policy(path), Policy)


_YAML_VALUES = st.recursive(
    st.none() | st.booleans() | st.integers() | st.text(max_size=12),
    lambda inner: (
        st.lists(inner, max_size=4) | st.dictionaries(st.text(max_size=8), inner, max_size=3)
    ),
    max_leaves=12,
)


@settings(max_examples=200)
@given(document=st.dictionaries(st.sampled_from(["allow_formats", "x"]), _YAML_VALUES, max_size=2))
def test_structured_yaml_gives_policy_or_policy_error(
    document: dict[str, object], tmp_path_factory: pytest.TempPathFactory
) -> None:
    import yaml

    path = tmp_path_factory.mktemp("p") / "policy.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with contextlib.suppress(PolicyError):
        assert isinstance(load_policy(path), Policy)

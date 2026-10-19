"""The allowlist is a closed set, never an open suffix match (reviewer I1)."""

import pytest

from nitrogen.allowlist import is_allowed


@pytest.mark.parametrize(
    ("module", "name"),
    [
        ("collections", "OrderedDict"),
        ("torch._utils", "_rebuild_tensor_v2"),
        ("torch", "FloatStorage"),
        ("torch.storage", "UntypedStorage"),
        ("torch.storage", "TypedStorage"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "_reconstruct"),  # private spelling
        ("numpy", "ndarray"),
    ],
)
def test_allowed(module: str, name: str) -> None:
    assert is_allowed(module, name)


@pytest.mark.parametrize(
    ("module", "name"),
    [
        ("os", "system"),
        ("builtins", "eval"),
        ("torch", "load"),
        ("torch.serialization", "load"),
        ("torch", "a.b.CfgStorage"),  # dotted name must not pass the suffix
        ("torch", "_HackStorage"),  # private, not a real storage class
        ("torch", "Storage"),  # bare "Storage" is not a concrete class we expect
        ("torch.nn", "FloatStorage"),  # wrong module
        ("numpy._core.evil", "system"),  # the _core prefix must not whitelist it
        ("pydoc", "pipepager"),
    ],
)
def test_rejected(module: str, name: str) -> None:
    assert not is_allowed(module, name)

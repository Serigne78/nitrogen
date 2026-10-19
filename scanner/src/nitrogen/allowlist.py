"""The imports a pickle may resolve without being flagged (rule #2).

Allowlist, never denylist: only the names a real PyTorch or NumPy checkpoint
needs are here, and everything else is a finding. This list is deliberately
small and reviewed on every addition — a name that can reach arbitrary code
(torch.load, eval, getattr...) must never be added.
"""

# Tensor and array reconstruction helpers, plus the containers that hold them.
# These build data structures; none of them runs caller-supplied code.
ALLOWED_GLOBALS = frozenset(
    {
        "collections.OrderedDict",
        "collections.defaultdict",
        "torch._utils._rebuild_tensor",
        "torch._utils._rebuild_tensor_v2",
        "torch._utils._rebuild_parameter",
        "torch._utils._rebuild_sparse_tensor",
        "numpy.core.multiarray._reconstruct",
        "numpy.core.multiarray.scalar",
        "numpy.ndarray",
        "numpy.dtype",
    }
)

# Storage classes live under two module paths depending on the torch version
# (torch.FloatStorage vs torch.storage.FloatStorage).
_STORAGE_MODULES = frozenset({"torch", "torch.storage"})
# A closed set of storage class names, not an open "ends with Storage": the
# latter would let a dotted or private name through, since STACK_GLOBAL resolves
# dotted paths (e.g. torch.a.b.CfgStorage).
_STORAGE_CLASSES = frozenset(
    {"TypedStorage", "UntypedStorage"}
    | {
        f"{kind}Storage"
        for kind in (
            "Byte", "Char", "Short", "Int", "Long", "Half", "Float", "Double",
            "Bool", "BFloat16", "ComplexFloat", "ComplexDouble",
            "QInt8", "QUInt8", "QInt32",
        )
    }
)  # fmt: skip


def is_allowed(module: str, name: str) -> bool:
    if f"{module}.{name}" in ALLOWED_GLOBALS:
        return True
    # numpy._core is the private spelling of numpy.core; accept both as a prefix.
    if module.startswith("numpy._core"):
        return is_allowed(module.replace("numpy._core", "numpy.core", 1), name)
    return module in _STORAGE_MODULES and name in _STORAGE_CLASSES

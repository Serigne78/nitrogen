"""Generate the test corpus, reproducibly, from hand-written opcodes.

Rule #6: the corpus is only ever harmless (every payload is `echo`), it is
produced by this script, and it is NEVER loaded. Pickles are assembled byte by
byte here; `pickle.dumps` is banned (it would run on build machines and is
forbidden by ruff TID251). The bytes are validated with `pickletools.genops`,
which reads opcodes without executing them.

Run `python generate.py` to (re)write the corpus next to this file. The suite
checks that the committed files match what this script produces.
"""

import io
import pickletools
import struct
import zipfile
from pathlib import Path

CORPUS = Path(__file__).parent
PAYLOAD = b"echo nitrogen"  # the only command any malicious case ever carries


# --- opcode helpers (hand-written, never pickle.dumps) ------------------------


def proto(version: int) -> bytes:
    return b"\x80" + bytes([version])


def global_op(module: str, name: str) -> bytes:
    """GLOBAL (protocol 0): c<module>\\n<name>\\n."""
    return b"c" + module.encode() + b"\n" + name.encode() + b"\n"


def inst(module: str, name: str) -> bytes:
    """INST: i<module>\\n<name>\\n (old-style instance, resolves a global too)."""
    return b"i" + module.encode() + b"\n" + name.encode() + b"\n"


def short_binunicode(text: str) -> bytes:
    raw = text.encode()
    return b"\x8c" + bytes([len(raw)]) + raw


def string0(text: str) -> bytes:
    """STRING (protocol 0): S'...'\\n."""
    return b"S'" + text.encode() + b"'\n"


MARK, TUPLE, TUPLE1, EMPTY_TUPLE = b"(", b"t", b"\x85", b")"
REDUCE, STOP, POP, MEMOIZE, STACK_GLOBAL = b"R", b".", b"0", b"\x94", b"\x93"
NEWOBJ, BUILD, EMPTY_DICT, SETITEM, NEWFALSE, BINPERSID = b"\x81", b"b", b"}", b"s", b"\x89", b"Q"


def binget(index: int) -> bytes:
    return b"h" + bytes([index])


def short_binstring(text: str) -> bytes:
    """SHORT_BINSTRING (protocol 1)."""
    raw = text.encode()
    return b"U" + bytes([len(raw)]) + raw


def short_binbytes(data: bytes) -> bytes:
    """SHORT_BINBYTES (protocol 3)."""
    return b"C" + bytes([len(data)]) + data


def _memo_put(kind: str, index: int) -> bytes:
    if kind == "memoize":
        return MEMOIZE  # index is implicit and auto-incremented
    if kind == "text":
        return b"p" + str(index).encode() + b"\n"  # PUT (protocol 0)
    if kind == "bin":
        return b"q" + bytes([index])  # BINPUT (protocol 1)
    return b"r" + struct.pack("<I", index)  # LONG_BINPUT


def _memo_get(kind: str, index: int) -> bytes:
    if kind in ("memoize", "bin"):
        return b"h" + bytes([index])  # BINGET
    if kind == "text":
        return b"g" + str(index).encode() + b"\n"  # GET
    return b"j" + struct.pack("<I", index)  # LONG_BINGET


def call0(module: str, name: str, arg: str) -> bytes:
    """Protocol 0: resolve <module>.<name> and REDUCE it on (arg,)."""
    return global_op(module, name) + MARK + string0(arg) + TUPLE + REDUCE + STOP


def call_stack(module: str, name: str, arg: str) -> bytes:
    """Protocol 4: STACK_GLOBAL from inline strings, then REDUCE on (arg,)."""
    return (
        proto(4)
        + short_binunicode(module)
        + short_binunicode(name)
        + STACK_GLOBAL
        + short_binunicode(arg)
        + TUPLE1
        + REDUCE
        + STOP
    )


def call_stack_via_memo(module: str, name: str, arg: str, kind: str = "memoize") -> bytes:
    """STACK_GLOBAL whose module/name are read back from the memo (rule #3).

    The names are pushed, stored in the memo, popped, then fetched back: a
    scanner that does not rebuild the memo sees two bare gets and misses the
    import. `kind` picks which store/fetch opcode pair carries the names, since
    rule #3 lists MEMOIZE/PUT/BINPUT/LONG_BINPUT and their get counterparts.
    """
    # Two distinct slots; "long" uses indices past 255 to exercise LONG_BINGET.
    i0, i1 = (300, 301) if kind == "long" else (0, 1)
    return (
        proto(4)
        + short_binunicode(module)
        + _memo_put(kind, i0)
        + POP
        + short_binunicode(name)
        + _memo_put(kind, i1)
        + POP
        + _memo_get(kind, i0)
        + _memo_get(kind, i1)
        + STACK_GLOBAL
        + short_binunicode(arg)
        + TUPLE1
        + REDUCE
        + STOP
    )


def framed(body: bytes) -> bytes:
    """Wrap a protocol-5 body in a FRAME opcode."""
    return proto(5) + b"\x95" + struct.pack("<Q", len(body)) + body


def pytorch_zip(data_pkl: bytes) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in [
            ("archive/data.pkl", data_pkl),
            ("archive/version", b"3\n"),
            ("archive/byteorder", b"little"),
        ]:
            # Fixed timestamp and host system: both are stamped by zipfile and
            # would otherwise make the bytes differ across runs and OSes
            # (create_system is 0 on Windows, 3 on Unix).
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 0
            zf.writestr(info, content)
    return buf.getvalue()


def safetensors(header: bytes, data: bytes = b"") -> bytes:
    return struct.pack("<Q", len(header)) + header + data


# --- the corpus ---------------------------------------------------------------

# A safe pickle references only names a real PyTorch checkpoint uses; these are
# the minimum the step-5 allowlist must accept, or every model is a false
# positive. OrderedDict() is built; the rebuild helper is only referenced.
_SAFE_PICKLE = (
    proto(2)
    + global_op("collections", "OrderedDict")
    + EMPTY_TUPLE
    + REDUCE
    + global_op("torch._utils", "_rebuild_tensor_v2")
    + POP
    + STOP
)

# A realistic state_dict: OrderedDict mapping a tensor name to a tensor rebuilt
# from a storage loaded through BINPERSID, exactly as torch.save emits. This is
# the true shape the step-5 allowlist is calibrated against (torch.FloatStorage,
# torch._utils._rebuild_tensor_v2, BINPERSID), not the minimal pickle above.
_REALISTIC_TORCH_PICKLE = (
    proto(2)
    + global_op("collections", "OrderedDict")
    + EMPTY_TUPLE
    + REDUCE  # od = OrderedDict()
    + short_binunicode("weight")
    + global_op("torch._utils", "_rebuild_tensor_v2")
    + MARK
    + MARK  # persistent id tuple for the storage
    + short_binunicode("storage")
    + global_op("torch", "FloatStorage")
    + short_binunicode("0")  # storage key
    + short_binunicode("cpu")  # location
    + b"K\x08"  # numel = 8
    + TUPLE
    + BINPERSID  # -> storage object
    + b"K\x00"  # storage offset
    + EMPTY_TUPLE  # size (placeholder)
    + EMPTY_TUPLE  # stride
    + NEWFALSE  # requires_grad
    + TUPLE
    + REDUCE  # _rebuild_tensor_v2(storage, 0, (), (), False)
    + SETITEM  # od["weight"] = tensor
    + STOP
)


def corpus() -> dict[str, bytes]:
    files = {
        # --- dangerous imports, one mechanism each ---------------------------
        "os_system_global.pkl": call0("os", "system", PAYLOAD.decode()),
        "subprocess_popen.pkl": call0("subprocess", "Popen", PAYLOAD.decode()),
        "subprocess_call.pkl": call_stack("subprocess", "call", PAYLOAD.decode()),
        "builtins_eval.pkl": call0("builtins", "eval", PAYLOAD.decode()),
        "builtins_exec.pkl": call_stack("builtins", "exec", PAYLOAD.decode()),
        "builtins_import.pkl": call0("builtins", "__import__", "os"),
        "webbrowser_open.pkl": call0("webbrowser", "open", PAYLOAD.decode()),
        # --- opcode mechanisms -----------------------------------------------
        "posix_system_inst.pkl": MARK + string0(PAYLOAD.decode()) + inst("posix", "system") + STOP,
        "os_system_stack.pkl": call_stack("os", "system", PAYLOAD.decode()),
        "os_system_framed.pkl": framed(
            global_op("os", "system") + MARK + string0(PAYLOAD.decode()) + TUPLE + REDUCE + STOP
        ),
        # --- memo bypasses, every store/fetch pair of rule #3 ----------------
        "os_system_memo.pkl": call_stack_via_memo("os", "system", PAYLOAD.decode()),  # MEMOIZE
        "os_system_memo_put_get.pkl": call_stack_via_memo("os", "system", PAYLOAD.decode(), "text"),
        "os_system_memo_binput.pkl": call_stack_via_memo("os", "system", PAYLOAD.decode(), "bin"),
        "os_system_memo_long.pkl": call_stack_via_memo("os", "system", PAYLOAD.decode(), "long"),
        # --- construction opcodes: import, not REDUCE, is what is flagged -----
        "os_system_newobj.pkl": global_op("os", "system") + EMPTY_TUPLE + NEWOBJ + STOP,
        "os_system_build.pkl": global_op("os", "system") + EMPTY_DICT + BUILD + STOP,
        # --- protocol variety and binary string opcodes ----------------------
        "proto1_binstring.pkl": (
            global_op("os", "system")
            + MARK
            + short_binstring(PAYLOAD.decode())
            + TUPLE
            + REDUCE
            + STOP
        ),
        "proto3_binbytes.pkl": (
            proto(3) + global_op("os", "system") + short_binbytes(PAYLOAD) + TUPLE1 + REDUCE + STOP
        ),
        # --- dotted names and submodules -------------------------------------
        "importlib_import_module.pkl": call_stack("importlib", "import_module", "os"),
        "torch_serialization_load.pkl": call_stack("torch.serialization", "load", PAYLOAD.decode()),
        # --- allowlist-bypass gadgets ----------------------------------------
        "getattr_gadget.pkl": call0("builtins", "getattr", "os"),
        "runpy_module.pkl": call_stack("runpy", "_run_module_code", PAYLOAD.decode()),
        "pydoc_pipepager.pkl": call_stack("pydoc", "pipepager", PAYLOAD.decode()),
        "functools_partial.pkl": call_stack("functools", "partial", PAYLOAD.decode()),
        # --- containers and concatenation ------------------------------------
        "pytorch_zip_os_system.pt": pytorch_zip(call0("os", "system", PAYLOAD.decode())),
        "concatenated.pkl": (proto(2) + b"K\x01." + call0("os", "system", PAYLOAD.decode())),
        # --- malformed: reported, never ignored, never a crash ---------------
        "truncated.pkl": proto(2) + b"(K\x01",  # MARK, BININT1, no STOP
        "oversized_length.pkl": proto(4)
        + b"\x8d\xff\xff\xff\xff\x00\x00\x00\x00",  # BINUNICODE8 4 GiB
        # --- safe (no finding, no false positive) ----------------------------
        "safe_allowlist.pkl": _SAFE_PICKLE,
        "safe_pytorch_zip.pt": pytorch_zip(_SAFE_PICKLE),
        "safe_pytorch_model.pt": pytorch_zip(_REALISTIC_TORCH_PICKLE),
        "safe_model.safetensors": safetensors(
            b'{"w":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}', bytes(8)
        ),
    }
    return files


# These are built to break a parser on purpose; genops need not accept them.
MALFORMED = {"truncated.pkl", "oversized_length.pkl"}


def _validate(data: bytes) -> None:
    """genops must read the stream (it decodes opcodes, never executes)."""
    list(pickletools.genops(io.BytesIO(data)))


def main() -> None:
    for name, data in corpus().items():
        if name.endswith(".pkl") and name not in MALFORMED:
            _validate(data)
        if name.endswith(".pt"):  # the data.pkl inside the zip must parse too
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                _validate(zf.read("archive/data.pkl"))
        (CORPUS / name).write_bytes(data)
    print(f"wrote {len(corpus())} corpus files to {CORPUS}")


if __name__ == "__main__":
    main()

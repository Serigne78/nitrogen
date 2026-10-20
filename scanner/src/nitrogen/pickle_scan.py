"""Static pickle analysis (rule #1: opcodes are read, never executed).

Every opcode that resolves a callable is checked against the allowlist:
GLOBAL, INST, STACK_GLOBAL (including names rebuilt through the memo), and the
copyreg extension opcodes EXT1/2/4. Anything not explicitly allowed is a
finding (rule #2). All concatenated pickles in a stream are walked, and reads
are bounded: a declared length beyond the buffer is a finding, never a giant
allocation.
"""

import io
import pickletools
from typing import BinaryIO

from nitrogen.allowlist import is_allowed
from nitrogen.findings import MAX_MESSAGE, BoundedFindings, Finding
from nitrogen.formats import recoverable_global, recoverable_int_line

# A pickle is read into memory before genops walks it, so an attacker-declared
# length can never allocate more than this. Real data.pkl members are tiny;
# tensor bytes live in separate archive members, not in the pickle.
MAX_PICKLE_BYTES = 16 * 1024 * 1024

# Opcodes whose argument is a string pushed on the stack.
_STRING_OPS = frozenset(
    {
        "STRING", "BINSTRING", "SHORT_BINSTRING",
        "UNICODE", "SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8",
    }
)  # fmt: skip

_MARK = object()  # mark sentinel on the simulated stack
_OPAQUE = object()  # a value we do not track (int, tuple, reduced object...)


def check_pickle(fh: BinaryIO, size: int, *, sole_format: bool = True) -> list[Finding]:
    """Analyze a pickle stream. `sole_format` is False when the file is also a
    valid other format (a polyglot): a mere parse error is then not reported,
    since the bytes are a real file of that other format (ADR 0003)."""
    fh.seek(0)
    data = fh.read(MAX_PICKLE_BYTES + 1)
    fh.seek(0)
    capped = len(data) > MAX_PICKLE_BYTES
    return scan_pickle(data[:MAX_PICKLE_BYTES], sole_format=sole_format, capped=capped)


def scan_pickle(
    data: bytes, *, sole_format: bool = True, capped: bool = False, source: str = ""
) -> list[Finding]:
    findings = BoundedFindings()  # dedup + cap while walking: memory stays bounded
    # A broken stream is only trustworthy as "malformed" when the bytes are
    # unambiguously a pickle: a PROTO header, or an import already resolved.
    # A headless stream that breaks early (a RECORD file, a ruff cache blob) is
    # just not a pickle, so its parse error is not a finding.
    has_proto = data[:1] == b"\x80"
    stream = io.BytesIO(data)
    while stream.tell() < len(data):
        try:
            _walk_one(stream, findings, source)
        except (ValueError, IndexError) as exc:
            if sole_format and (has_proto or findings):
                findings.append(_finding("pickle.malformed", str(exc), source))
            return findings.result()
    if capped and not findings:
        findings.append(
            _finding("pickle.unanalyzable", f"pickle larger than {MAX_PICKLE_BYTES} bytes", source)
        )
    return findings.result()


def _walk_one(stream: io.BytesIO, findings: BoundedFindings, source: str) -> None:
    """Walk a single pickle, up to and including its STOP."""
    stack: list[object] = []
    memo: dict[int, object] = {}

    def push(value: object) -> None:
        stack.append(value)

    def pop() -> object:
        return stack.pop() if stack else _OPAQUE

    def pop_to_mark() -> None:
        while stack and stack.pop() is not _MARK:
            pass

    data = stream.getvalue()
    ops = pickletools.genops(stream)
    while True:
        start = stream.tell()
        try:
            opcode, arg, _pos = next(ops)
        except StopIteration:
            raise ValueError("pickle exhausted before seeing STOP") from None
        except ValueError:
            # genops reads INT/LONG with int(s); the real loader uses int(s, 0)
            # and runs past "0x1", "0o7"... Where only genops stops, we skip the
            # line and keep walking, so the import after it is still seen.
            resume = recoverable_int_line(data, start)
            if resume is not None:
                stream.seek(resume)
                push(_OPAQUE)  # the int the loader pushed
                ops = pickletools.genops(stream)
                continue
            # GLOBAL/INST whose name line has no final newline at EOF: the
            # loader still resolves it (readline()[:-1]). Flag the import.
            resolved = recoverable_global(data, start)
            if resolved is not None:
                _check_global(resolved[0], resolved[1], findings, source)
                return
            raise
        name = opcode.name
        if name == "STOP":
            return
        if name in _STRING_OPS:
            push(_as_text(arg))
        elif name == "MARK":
            push(_MARK)
        elif name == "MEMOIZE":
            memo[len(memo)] = stack[-1] if stack else _OPAQUE
        elif name in ("PUT", "BINPUT", "LONG_BINPUT"):
            memo[int(arg)] = stack[-1] if stack else _OPAQUE  # type: ignore[arg-type]
        elif name in ("GET", "BINGET", "LONG_BINGET"):
            push(memo.get(int(arg), _OPAQUE))  # type: ignore[arg-type]
        elif name == "DUP":
            push(stack[-1] if stack else _OPAQUE)
        elif name in ("GLOBAL", "INST"):
            _check_global_arg(arg, findings, source)
            _apply_stack_effect(opcode, stack, pop_to_mark)
        elif name == "STACK_GLOBAL":
            n, module = pop(), pop()
            _check_global(module, n, findings, source)
            push(_OPAQUE)
        elif name in ("EXT1", "EXT2", "EXT4"):
            findings.append(_finding("pickle.import", f"copyreg extension {arg}", source))
            push(_OPAQUE)
        else:
            _apply_stack_effect(opcode, stack, pop_to_mark)


def _apply_stack_effect(
    opcode: pickletools.OpcodeInfo, stack: list[object], pop_to_mark: object
) -> None:
    """Approximate the net stack effect from pickletools' own opcode metadata,
    enough to keep STACK_GLOBAL's two operands aligned in the common case."""
    before = opcode.stack_before
    after = opcode.stack_after
    if pickletools.markobject in before:
        # e.g. APPENDS: [list, mark, slice] -> [list]. Items below the mark are
        # preserved; only push results beyond them (TUPLE: nothing below -> +1).
        preserved = before.index(pickletools.markobject)
        pop_to_mark()  # type: ignore[operator]
        after = after[preserved:]
    else:
        for _ in before:
            if stack:
                stack.pop()
    for item in after:
        stack.append(_MARK if item is pickletools.markobject else _OPAQUE)


def _check_global_arg(arg: object, findings: BoundedFindings, source: str) -> None:
    # genops gives a GLOBAL/INST argument as "module name".
    parts = str(arg).split(" ", 1)
    if len(parts) == 2:
        _check_global(parts[0], parts[1], findings, source)
    else:
        findings.append(_finding("pickle.import", f"malformed global {arg!r}", source))


def _check_global(module: object, name: object, findings: BoundedFindings, source: str) -> None:
    if not isinstance(module, str) or not isinstance(name, str):
        # STACK_GLOBAL whose names we could not rebuild: cannot vouch for it.
        findings.append(_finding("pickle.import", "unresolved dynamic global", source))
    elif not is_allowed(module, name):
        # Truncate before formatting: a 1 MB memoized name reused a million
        # times must not be copied a million times.
        detail = f"{module[:MAX_MESSAGE]}.{name[:MAX_MESSAGE]}"
        findings.append(_finding("pickle.import", detail, source))


def _as_text(arg: object) -> object:
    if isinstance(arg, (bytes, bytearray)):
        try:
            return bytes(arg).decode("utf-8")
        except UnicodeDecodeError:
            return _OPAQUE
    if not isinstance(arg, str):
        return _OPAQUE
    # pickletools decodes with surrogatepass: a lone surrogate is no name any
    # loader imports, and would break UTF-8 output later. Treat it as opaque,
    # so a STACK_GLOBAL using it is still flagged ("unresolved").
    try:
        arg.encode("utf-8")
    except UnicodeEncodeError:
        return _OPAQUE
    return arg


def _finding(rule: str, detail: str, source: str) -> Finding:
    return Finding(rule, f"{source}: {detail}" if source else detail)

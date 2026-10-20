"""Format detection by magic bytes (rule #4): the extension is never consulted.

Pickles are hand-written opcode bytes; they are written to memory or disk and
never loaded.
"""

import io
import json
import struct
import tarfile
import zipfile
from pathlib import Path

import pytest

from nitrogen.formats import PROBE_SIZE, Format, detect_formats

PICKLE_P2 = b"\x80\x02K\x01."  # PROTO 2, BININT1 1, STOP
PICKLE_P5 = b"\x80\x05\x95\x03\x00\x00\x00\x00\x00\x00\x00K\x01."  # with FRAME
PICKLE_P0 = b"I1\n."  # INT 1, STOP
PICKLE_P1 = b"K\x01."  # BININT1 1, STOP
PICKLE_P0_GLOBAL = b"cos\nsystem\n(S'echo'\ntR."  # protocol 0, no magic


def _detect(data: bytes) -> frozenset[Format]:
    return detect_formats(io.BytesIO(data))


def _safetensors(header: dict[str, object] | None = None) -> bytes:
    raw = json.dumps(header or {"__metadata__": {}}).encode()
    return struct.pack("<Q", len(raw)) + raw


def _safetensors_sized(header_size: int, data: bytes = b"\x00" * 4) -> bytes:
    """A valid safetensors file whose JSON header is exactly `header_size` bytes.

    The header size is the first 8 bytes, so it picks the opcode a pickle
    loader would read at offset 0.
    """
    tensor = {"dtype": "U8", "shape": [len(data)], "data_offsets": [0, len(data)]}
    header = {"__metadata__": {"pad": ""}, "t": tensor}
    raw = json.dumps(header, separators=(",", ":")).encode()
    header["__metadata__"] = {"pad": "x" * (header_size - len(raw))}
    raw = json.dumps(header, separators=(",", ":")).encode()
    assert len(raw) == header_size
    return struct.pack("<Q", header_size) + raw + data


def _zip(names: list[str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name in names:
            zf.writestr(name, b"x")
    return buf.getvalue()


def test_pickle_protocols_with_magic() -> None:
    assert _detect(PICKLE_P2) == {Format.PICKLE}
    assert _detect(PICKLE_P5) == {Format.PICKLE}


def test_pickle_protocols_without_magic() -> None:
    assert _detect(PICKLE_P0) == {Format.PICKLE}
    assert _detect(PICKLE_P1) == {Format.PICKLE}
    assert _detect(PICKLE_P0_GLOBAL) == {Format.PICKLE}
    assert _detect(b"(dp0\n.") == {Format.PICKLE}


def test_pickle_followed_by_trailing_data() -> None:
    assert _detect(PICKLE_P1 + b"junk") == {Format.PICKLE}


def test_large_protocol_0_pickle_beyond_probe() -> None:
    # A valid pickle whose STOP lies past the probed prefix is still a pickle.
    data = b"(" + b"I1\n" * (PROBE_SIZE // 3 + 10) + b"l."
    assert len(data) > PROBE_SIZE
    assert _detect(data) == {Format.PICKLE}


def test_truncated_harmless_pickle_is_not_a_pickle() -> None:
    # Nothing resolves a global before the stream breaks: a loader just fails.
    assert _detect(b"(I1\n") == set()


def test_truncated_pickle_resolving_a_global_is_a_pickle() -> None:
    # The unpickler runs GLOBAL + REDUCE before noticing the missing STOP.
    assert _detect(PICKLE_P0_GLOBAL.removesuffix(b".")) == {Format.PICKLE}


def test_pickle_resolving_a_global_then_garbage_is_a_pickle() -> None:
    data = PICKLE_P0_GLOBAL.removesuffix(b".") + b"\xff" + b"junk" * 10
    assert _detect(data) == {Format.PICKLE}


def test_inst_and_stack_global_count_as_resolving() -> None:
    assert _detect(b"(S'echo'\nios\nsystem\n") == {Format.PICKLE}  # INST, no STOP
    stack_global = b"\x8c\x02os\x8c\x06system\x93"  # SHORT_BINUNICODE x2, STACK_GLOBAL
    assert _detect(stack_global) == {Format.PICKLE}


def test_stack_global_through_the_memo_counts_as_resolving() -> None:
    # SHORT_BINUNICODE + MEMOIZE twice, then BINGET 0, BINGET 1, STACK_GLOBAL.
    data = b"\x8c\x02os\x94\x8c\x06system\x94h\x00h\x01\x93"
    assert _detect(data) == {Format.PICKLE}


def test_concatenated_pickles_are_all_probed() -> None:
    # Legacy torch.save chains several pickles; the second one carries the global.
    assert _detect(PICKLE_P1 + PICKLE_P0_GLOBAL.removesuffix(b".")) == {Format.PICKLE}


def test_csv_starting_with_c_is_not_a_pickle() -> None:
    # GLOBAL needs importable names: "ity,population" cannot be one.
    assert _detect(b"city,population\nParis,2100000\n") == set()


def test_huge_declared_length_does_not_allocate() -> None:
    # BINUNICODE announcing 4 GiB: the probe reads a bounded prefix only.
    data = b"X\xff\xff\xff\xff" + b"a" * (PROBE_SIZE * 2)
    assert _detect(data) == {Format.PICKLE}


def test_pytorch_zip() -> None:
    data = _zip(["archive/data.pkl", "archive/version", "archive/data/0"])
    assert _detect(data) == {Format.PYTORCH_ZIP}


def test_pytorch_zip_any_archive_name() -> None:
    assert _detect(_zip(["resnet18/data.pkl"])) == {Format.PYTORCH_ZIP}


def test_generic_zip() -> None:
    assert _detect(_zip(["readme.txt"])) == {Format.ZIP}


def test_empty_zip() -> None:
    assert _detect(_zip([])) == {Format.ZIP}


def test_corrupt_zip_is_still_zip() -> None:
    assert _detect(b"PK\x03\x04" + b"\x00" * 64) == {Format.ZIP}


def test_safetensors() -> None:
    assert _detect(_safetensors()) == {Format.SAFETENSORS}


def test_safetensors_with_malformed_header_is_still_detected() -> None:
    # Step 3 reports it; detection must not let it slip to UNKNOWN.
    assert _detect(struct.pack("<Q", 8) + b"{garbage") == {Format.SAFETENSORS}


@pytest.mark.parametrize(
    "text",
    [
        b'EMOJI = {\n    "1st_place_medal": "\xf0\x9f\xa5\x87",\n}\n',  # rich/_emoji_codes.py
        b"prefix=${pcfiledir}/../../..\nincludedir=${prefix}/include\n",  # numpy.pc
        b'{\n  "R": {\n    "Version": "4.2.3"\n  }\n}\n',  # pre-commit renv.lock
        b'{"abc": {"def": 1}}',  # any JSON with an object at byte 8
    ],
    ids=["python", "pkgconfig", "json-indented", "json-compact"],
)
def test_text_with_brace_at_byte_8_is_not_safetensors(text: bytes) -> None:
    # Read as a size, 8 bytes of text are ~10**18: no loader reads a header
    # past the end of the file. False positives seen in CI on .venv.
    assert Format.SAFETENSORS not in _detect(text)


def test_safetensors_whose_size_bytes_are_pickle_opcodes() -> None:
    # Low byte "." is STOP, "V" opens a line. A pickle loader would read these
    # bytes too, so both interpretations are kept (ADR 0003); the pickle
    # analysis decides whether that reading resolves anything.
    assert _detect(_safetensors_sized(0x12E)) == {Format.SAFETENSORS}  # empty pickle: unreadable
    assert _detect(_safetensors_sized(0x156)) == {Format.SAFETENSORS}  # line never ends
    big_header = _safetensors_sized(0x20056)  # "V" line running past the probe
    assert _detect(big_header) == {Format.PICKLE, Format.SAFETENSORS}


def test_safetensors_is_detected_whatever_its_size_bytes() -> None:
    # The size field may look like any opcode or magic: it must never hide
    # the safetensors interpretation. Every size a serializer can write
    # (multiples of 8) up to 64 KiB, and every low byte.
    sizes = [*range(0x80, 0x200), *range(0x200, 0x10000, 8), 0x9393, 0x1293]
    for size in sizes:
        assert Format.SAFETENSORS in _detect(_safetensors_sized(size)), hex(size)


@pytest.mark.parametrize("size", [0x280, 0x380, 0x480, 0x580, 0x10280])
def test_size_field_colliding_with_pickle_magic(size: int) -> None:
    # Starts with PROTO 2..5: both a pickle and a safetensors (review C2).
    assert _detect(_safetensors_sized(size)) == {Format.PICKLE, Format.SAFETENSORS}


def test_zip_magic_size_beyond_the_file_is_only_a_zip() -> None:
    # A 67 MB header size reads "PK\x03\x04". Without the 67 MB that follow,
    # no safetensors loader reads it; the real polyglot is tested below.
    blob = struct.pack("<Q", 0x04034B50) + b'{"__metadata__":{}}'
    assert _detect(blob) == {Format.ZIP}


def test_safetensors_pytorch_zip_polyglot() -> None:
    # A valid safetensors whose single tensor holds a PyTorch zip: torch.load
    # reads the zip (central directory at the end), safetensors the tensors.
    # Neither interpretation may hide the other.
    payload = _zip(["archive/data.pkl"])
    header_size = 0x04034B50
    tensor = {"dtype": "U8", "shape": [len(payload)], "data_offsets": [0, len(payload)]}
    raw = json.dumps({"__metadata__": {"p": ""}, "t": tensor}, separators=(",", ":"))
    pad = "x" * (header_size - len(raw))
    raw = json.dumps({"__metadata__": {"p": pad}, "t": tensor}, separators=(",", ":"))
    blob = struct.pack("<Q", header_size) + raw.encode() + payload
    assert blob[:4] == b"PK\x03\x04"
    assert _detect(blob) == {Format.PYTORCH_ZIP, Format.SAFETENSORS}


@pytest.mark.parametrize("whitespace", [b" ", b"\t", b"\n", b"\r", b" \n\t "])
def test_safetensors_with_leading_whitespace(whitespace: bytes) -> None:
    # The reference loader skips JSON whitespace before "{": such a file must
    # reach the safetensors checks (which report it), never fall to UNKNOWN.
    header = whitespace + b'{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}'
    assert _detect(struct.pack("<Q", len(header)) + header + bytes(8)) == {Format.SAFETENSORS}


def test_whitespace_only_prefix_is_safetensors() -> None:
    blob = struct.pack("<Q", PROBE_SIZE * 2) + b" " * (PROBE_SIZE * 2) + b"{}"
    assert _detect(blob) == {Format.SAFETENSORS}


def test_safetensors_pickle_polyglot_has_both_formats() -> None:
    # Size low byte "V" (UNICODE) swallows the JSON header up to the newline
    # placed in the tensor data, then a pickle loader resolves os.system.
    polyglot = _safetensors_sized(0x156, b"\n" + PICKLE_P0_GLOBAL)
    assert _detect(polyglot) == {Format.PICKLE, Format.SAFETENSORS}


def test_payload_after_an_empty_first_pickle_is_unreachable() -> None:
    # Size bytes "." then "V": the first pickle is empty, and STOP on an empty
    # stack makes every loader fail before the second pickle is ever read.
    polyglot = _safetensors_sized(0x562E, b"\n" + PICKLE_P0_GLOBAL)
    assert _detect(polyglot) == {Format.SAFETENSORS}


def test_unknown_files() -> None:
    assert _detect(b"") == set()
    assert _detect(b'{"architectures": ["Bert"]}') == set()
    assert _detect(b"README\n") == set()
    # A README that merely starts with "GGUF" is not a GGUF file (ADR 0006).
    assert _detect(b"GGUF quantized versions of this model\n") == set()


def test_extension_is_ignored(tmp_path: Path) -> None:
    disguised_pickle = tmp_path / "model.safetensors"
    disguised_pickle.write_bytes(PICKLE_P0_GLOBAL)
    disguised_safetensors = tmp_path / "weights.bin"
    disguised_safetensors.write_bytes(_safetensors())
    with disguised_pickle.open("rb") as fh:
        assert detect_formats(fh) == {Format.PICKLE}
    with disguised_safetensors.open("rb") as fh:
        assert detect_formats(fh) == {Format.SAFETENSORS}


def test_detection_rewinds_stream() -> None:
    fh = io.BytesIO(_zip(["archive/data.pkl"]))
    fh.seek(5)
    assert detect_formats(fh) == {Format.PYTORCH_ZIP}
    assert fh.tell() == 0


@pytest.mark.parametrize(
    "head",
    [b"I0x1\n", b"I0b1\n", b"I0o7\n", b"L0x1L\n"],
    ids=["int-hex", "int-bin", "int-oct", "long-hex"],
)
def test_int_in_another_base_is_still_a_pickle(head: bytes) -> None:
    # The unpickler reads INT/LONG with int(s, 0), genops with int(s): genops
    # stops where the loader goes on to run the payload.
    assert _detect(head + PICKLE_P0_GLOBAL) == {Format.PICKLE}


@pytest.mark.parametrize(
    "text",
    [b"Introduction\nThis model...", b"Licensed under the Apache License\n", b"I\n"],
)
def test_text_starting_with_i_or_l_is_not_a_pickle(text: bytes) -> None:
    assert _detect(text) == set()


def test_legacy_torch_tar_is_detected() -> None:
    # torch.load still reads the pre-1.6 tar format and unpickles its members.
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:") as tar:
        member = tarfile.TarInfo("pickle")
        member.size = len(PICKLE_P1)
        tar.addfile(member, io.BytesIO(PICKLE_P1))
    assert _detect(buf.getvalue()) == {Format.TAR}


def test_persid_first_is_not_a_pickle() -> None:
    # A plain unpickler stops on PERSID at once (no persistent_load): nothing
    # runs. A large safetensors whose size starts with "P" stays safetensors.
    assert _detect(b"Pid\n" + PICKLE_P0_GLOBAL) == set()
    assert _detect(_safetensors_sized(0x20050)) == {Format.SAFETENSORS}


@pytest.mark.parametrize("size", [200, PROBE_SIZE * 2], ids=["small", "beyond-probe"])
def test_wheel_record_file_is_not_a_pickle(size: int) -> None:
    # dist-info/RECORD lines start with a package name, so "c" reads as GLOBAL.
    # A loader stops at once on a module name that cannot exist: no format.
    for line in (
        b"coverage/__init__.py,sha256=Zm9vYmFy,1234\n",
        b"../../Scripts/x.exe,sha256=Zm9v,9\n",
    ):
        record = (line * (size // len(line) + 1))[:size]
        assert _detect(record) == set(), line


def test_text_starting_with_a_dot_is_not_a_pickle() -> None:
    # "." is STOP: an empty pickle, which no loader can read (stack underflow).
    assert _detect(b".. -*- rest -*-\r\n\r\nAPI changes\n") == set()
    assert _detect(b".") == set()


@pytest.mark.parametrize(
    ("data", "fmt"),
    [
        (b"\x89HDF\r\n\x1a\n" + bytes(100), Format.HDF5),
        (b"GGUF\x03\x00\x00\x00" + bytes(100), Format.GGUF),
        (b"\x93NUMPY\x01\x00v\x00{'descr': '<f8'}" + bytes(100), Format.NPY),
        (b"\x08\x07\x12\x07pytorch\x1a\x032.1" + bytes(100), Format.ONNX),
    ],
    ids=["hdf5", "gguf", "npy", "onnx"],
)
def test_unanalyzed_model_formats_are_recognised(data: bytes, fmt: Format) -> None:
    # ADR 0006: recognised by magic, not analyzed, so "unknown" no longer hides them.
    assert fmt in _detect(data)


def test_onnx_heuristic_needs_an_exporter_trace() -> None:
    assert Format.ONNX not in _detect(b"\x08\x07" + bytes(200))

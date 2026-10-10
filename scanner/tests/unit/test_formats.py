"""Format detection by magic bytes (rule #4): the extension is never consulted.

Pickles are hand-written opcode bytes; they are written to memory or disk and
never loaded.
"""

import io
import json
import struct
import zipfile
from pathlib import Path

from nitrogen.formats import PROBE_SIZE, Format, detect_format

PICKLE_P2 = b"\x80\x02K\x01."  # PROTO 2, BININT1 1, STOP
PICKLE_P5 = b"\x80\x05\x95\x03\x00\x00\x00\x00\x00\x00\x00K\x01."  # with FRAME
PICKLE_P0 = b"I1\n."  # INT 1, STOP
PICKLE_P1 = b"K\x01."  # BININT1 1, STOP
PICKLE_P0_GLOBAL = b"cos\nsystem\n(S'echo'\ntR."  # protocol 0, no magic


def _detect(data: bytes) -> Format:
    return detect_format(io.BytesIO(data))


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
    assert _detect(PICKLE_P2) == Format.PICKLE
    assert _detect(PICKLE_P5) == Format.PICKLE


def test_pickle_protocols_without_magic() -> None:
    assert _detect(PICKLE_P0) == Format.PICKLE
    assert _detect(PICKLE_P1) == Format.PICKLE
    assert _detect(PICKLE_P0_GLOBAL) == Format.PICKLE
    assert _detect(b"(dp0\n.") == Format.PICKLE


def test_pickle_followed_by_trailing_data() -> None:
    assert _detect(PICKLE_P1 + b"junk") == Format.PICKLE


def test_large_protocol_0_pickle_beyond_probe() -> None:
    # A valid pickle whose STOP lies past the probed prefix is still a pickle.
    data = b"(" + b"I1\n" * (PROBE_SIZE // 3 + 10) + b"l."
    assert len(data) > PROBE_SIZE
    assert _detect(data) == Format.PICKLE


def test_truncated_harmless_pickle_is_not_a_pickle() -> None:
    # Nothing resolves a global before the stream breaks: a loader just fails.
    assert _detect(b"(I1\n") == Format.UNKNOWN


def test_truncated_pickle_resolving_a_global_is_a_pickle() -> None:
    # The unpickler runs GLOBAL + REDUCE before noticing the missing STOP.
    assert _detect(PICKLE_P0_GLOBAL.removesuffix(b".")) == Format.PICKLE


def test_pickle_resolving_a_global_then_garbage_is_a_pickle() -> None:
    data = PICKLE_P0_GLOBAL.removesuffix(b".") + b"\xff" + b"junk" * 10
    assert _detect(data) == Format.PICKLE


def test_inst_and_stack_global_count_as_resolving() -> None:
    assert _detect(b"(S'echo'\nios\nsystem\n") == Format.PICKLE  # INST, no STOP
    stack_global = b"\x8c\x02os\x8c\x06system\x93"  # SHORT_BINUNICODE x2, STACK_GLOBAL
    assert _detect(stack_global) == Format.PICKLE


def test_stack_global_through_the_memo_counts_as_resolving() -> None:
    # SHORT_BINUNICODE + MEMOIZE twice, then BINGET 0, BINGET 1, STACK_GLOBAL.
    data = b"\x8c\x02os\x94\x8c\x06system\x94h\x00h\x01\x93"
    assert _detect(data) == Format.PICKLE


def test_concatenated_pickles_are_all_probed() -> None:
    # Legacy torch.save chains several pickles; the second one carries the global.
    assert _detect(PICKLE_P1 + PICKLE_P0_GLOBAL.removesuffix(b".")) == Format.PICKLE


def test_csv_starting_with_c_is_not_a_pickle() -> None:
    # GLOBAL needs importable names: "ity,population" cannot be one.
    assert _detect(b"city,population\nParis,2100000\n") == Format.UNKNOWN


def test_huge_declared_length_does_not_allocate() -> None:
    # BINUNICODE announcing 4 GiB: the probe reads a bounded prefix only.
    data = b"X\xff\xff\xff\xff" + b"a" * (PROBE_SIZE * 2)
    assert _detect(data) == Format.PICKLE


def test_pytorch_zip() -> None:
    data = _zip(["archive/data.pkl", "archive/version", "archive/data/0"])
    assert _detect(data) == Format.PYTORCH_ZIP


def test_pytorch_zip_any_archive_name() -> None:
    assert _detect(_zip(["resnet18/data.pkl"])) == Format.PYTORCH_ZIP


def test_generic_zip() -> None:
    assert _detect(_zip(["readme.txt"])) == Format.ZIP


def test_empty_zip() -> None:
    assert _detect(_zip([])) == Format.ZIP


def test_corrupt_zip_is_still_zip() -> None:
    assert _detect(b"PK\x03\x04" + b"\x00" * 64) == Format.ZIP


def test_safetensors() -> None:
    assert _detect(_safetensors()) == Format.SAFETENSORS


def test_safetensors_with_malformed_header_is_still_detected() -> None:
    # Step 3 reports it; detection must not let it slip to UNKNOWN.
    assert _detect(struct.pack("<Q", 10**12) + b"{garbage") == Format.SAFETENSORS


def test_safetensors_whose_size_bytes_are_pickle_opcodes() -> None:
    # Low byte "." is STOP, "V" opens a line: harmless prefixes of real files.
    assert _detect(_safetensors_sized(0x12E)) == Format.SAFETENSORS
    assert _detect(_safetensors_sized(0x156)) == Format.SAFETENSORS
    big_header = _safetensors_sized(0x20056)  # "V" line running past the probe
    assert _detect(big_header) == Format.SAFETENSORS


def test_safetensors_whatever_its_size_bytes() -> None:
    # The size field is read as opcodes by the probe: no value may turn a
    # genuine safetensors into a pickle (e.g. a lone STACK_GLOBAL or EXT1).
    for size in [*range(0x80, 0x100), *range(0x100, 0x200), 0x9393, 0x1293]:
        assert _detect(_safetensors_sized(size)) == Format.SAFETENSORS, hex(size)


def test_safetensors_pickle_polyglot_is_a_pickle() -> None:
    # Size low byte "V" (UNICODE) swallows the JSON header up to the newline
    # placed in the tensor data, then a pickle loader resolves os.system.
    polyglot = _safetensors_sized(0x156, b"\n" + PICKLE_P0_GLOBAL)
    assert _detect(polyglot) == Format.PICKLE


def test_safetensors_pickle_polyglot_after_a_stop_is_a_pickle() -> None:
    # Size bytes "." then "V": an empty first pickle, the payload in the second.
    polyglot = _safetensors_sized(0x562E, b"\n" + PICKLE_P0_GLOBAL)
    assert _detect(polyglot) == Format.PICKLE


def test_unknown_files() -> None:
    assert _detect(b"") == Format.UNKNOWN
    assert _detect(b'{"architectures": ["Bert"]}') == Format.UNKNOWN
    assert _detect(b"README\n") == Format.UNKNOWN
    assert _detect(b"GGUF\x03\x00\x00\x00") == Format.UNKNOWN


def test_extension_is_ignored(tmp_path: Path) -> None:
    disguised_pickle = tmp_path / "model.safetensors"
    disguised_pickle.write_bytes(PICKLE_P0_GLOBAL)
    disguised_safetensors = tmp_path / "weights.bin"
    disguised_safetensors.write_bytes(_safetensors())
    with disguised_pickle.open("rb") as fh:
        assert detect_format(fh) == Format.PICKLE
    with disguised_safetensors.open("rb") as fh:
        assert detect_format(fh) == Format.SAFETENSORS


def test_detection_rewinds_stream() -> None:
    fh = io.BytesIO(_zip(["archive/data.pkl"]))
    fh.seek(5)
    assert detect_format(fh) == Format.PYTORCH_ZIP
    assert fh.tell() == 0

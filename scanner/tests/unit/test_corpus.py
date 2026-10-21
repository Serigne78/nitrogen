"""Integrity of the test corpus (step 4).

Step 4 builds the corpus and freezes the expected result of each case; the
step-5 analyzer is what will be run against it. These tests guard the corpus
itself: it stays in sync with generate.py and cases.yaml, its formats are what
we claim, and every payload is harmless (rule #6).
"""

import io
import pickletools
from pathlib import Path
from typing import Any

import pytest
import yaml

from nitrogen.formats import Format, detect_formats

CORPUS = Path(__file__).resolve().parents[1] / "corpus"
CASES: list[dict[str, Any]] = yaml.safe_load((CORPUS / "cases.yaml").read_text())
BY_FILE = {case["file"]: case for case in CASES}


def _corpus_files() -> list[Path]:
    return sorted(
        p for p in CORPUS.iterdir() if p.suffix in (".pkl", ".pt", ".zip", ".safetensors", ".npy")
    )


def test_cases_and_files_match() -> None:
    on_disk = {p.name for p in _corpus_files()}
    declared = {case["file"] for case in CASES}
    assert on_disk == declared, f"orphan files {on_disk - declared}, missing {declared - on_disk}"


def test_no_duplicate_cases() -> None:
    files = [case["file"] for case in CASES]
    assert len(files) == len(set(files))


def _load_generator() -> Any:
    import importlib.util

    spec = importlib.util.spec_from_file_location("corpus_generate", CORPUS / "generate.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generate_is_reproducible() -> None:
    for name, data in _load_generator().corpus().items():
        assert (CORPUS / name).read_bytes() == data, f"{name} is stale; rerun generate.py"


def test_corpus_has_both_verdicts() -> None:
    verdicts = {case["verdict"] for case in CASES}
    assert verdicts == {"safe", "risk"}
    assert sum(c["verdict"] == "risk" for c in CASES) >= 15  # roadmap: 15-20 cases


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["file"])
def test_case_schema(case: dict[str, Any]) -> None:
    assert case["verdict"] in ("safe", "risk")
    if case["verdict"] == "risk":
        assert case["finding"] in ("pickle.import", "pickle.malformed", "pickle.unanalyzable")
        if case["finding"] == "pickle.import":
            assert case["detail"]  # the dangerous name must be named
    else:
        assert "finding" not in case


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["file"])
def test_declared_format_is_detected(case: dict[str, Any]) -> None:
    with (CORPUS / case["file"]).open("rb") as fh:
        formats = detect_formats(fh)
    assert Format(case["format"]) in formats


# Tokens that betray a real command. "echo nitrogen" contains none; module and
# attribute names contain none either. Anything carrying one is not harmless.
_DANGEROUS_TOKENS = (
    "rm ",
    "rmdir",
    "del ",
    "curl",
    "wget",
    "http",
    "nc ",
    "ncat",
    "bash",
    "/bin",
    "/etc",
    "sh -",
    "powershell",
    "cmd",
    "chmod",
    "chown",
    "mkfifo",
    "eval(",
    ";",
    "|",
    "&",
    "$(",
    "`",
    ">",
    "<",
    "\n",
    "..",
)


def test_payloads_are_harmless() -> None:
    """No corpus pickle carries anything but `echo nitrogen`. Every string and
    bytes argument, across concatenated pickles and every zip member, is read
    with genops and checked; the corpus is never loaded (rule #6)."""
    for path in _corpus_files():
        for value in _arguments_in_pickles(path.read_bytes()):
            text = value.decode("latin-1") if isinstance(value, bytes) else value
            lowered = text.lower()
            if "echo nitrogen" in lowered or lowered == "nitrogen":
                continue  # the only payload we ever ship
            for token in _DANGEROUS_TOKENS:
                assert token not in lowered, f"{path.name}: suspicious argument {text!r}"


def _arguments_in_pickles(data: bytes) -> list[str | bytes]:
    """Collect str AND bytes arguments across all concatenated pickles in a
    blob, over every zip member (not just .pkl). Reads opcodes only."""
    found: list[str | bytes] = []
    for blob in _pickle_streams(data):
        stream = io.BytesIO(blob)
        try:
            while stream.tell() < len(blob):
                for _op, arg, _pos in pickletools.genops(stream):
                    if isinstance(arg, (str, bytes, bytearray)):
                        found.append(bytes(arg) if isinstance(arg, bytearray) else arg)
        except ValueError:
            pass  # malformed case: check what parsed
    return found


def _pickle_streams(data: bytes) -> list[bytes]:
    import zipfile

    if zipfile.is_zipfile(io.BytesIO(data)):
        # Every member, not only *.pkl: a payload could hide in any entry.
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            return [zf.read(n) for n in zf.namelist()]
    import tarfile

    try:
        # Legacy torch tar: every member, read in memory, never extracted.
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as tar:
            files = [m for m in tar.getmembers() if m.isfile()]
            return [tar.extractfile(m).read() for m in files]  # type: ignore[union-attr]
    except tarfile.TarError:
        pass
    if len(data) >= 8 and data[8:9] == b"{":
        return []  # safetensors: no embedded pickle
    if data.startswith(b"\x93NUMPY"):
        return []  # a .npy array holds no pickle (its \x93 is just the magic)
    return [data]

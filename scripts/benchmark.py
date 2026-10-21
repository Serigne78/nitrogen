"""Benchmark Nitrogen against picklescan and ModelScan on the test corpus.

Each tool is run through its CLI in a subprocess, one corpus file at a time —
no tool ever loads a file, and the corpus (harmless `echo` payloads) is never
executed. Ground truth is tests/corpus/cases.yaml. The script prints a Markdown
table of detection (malicious caught) and false positives (healthy flagged).

Run from the scanner directory (picklescan and modelscan are development
dependencies, pulled in by `uv sync`):
    uv run python ../scripts/benchmark.py

This script is not part of the pytest suite.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import yaml

SCANNER = Path(__file__).resolve().parents[1] / "scanner"
CORPUS = SCANNER / "tests" / "corpus"
CASES = yaml.safe_load((CORPUS / "cases.yaml").read_text())


def _detects(tool: str, *args: str) -> bool:
    """True when `tool` flags the file. Fixed tool names and corpus paths —
    no untrusted input — run in a subprocess that never loads the file."""
    exe = shutil.which(tool)
    if exe is None:
        raise SystemExit(f"{tool} not found; install the bench dev dependencies")
    # Exit 1 = detection; 2 (our error) is not a detection.
    return subprocess.run([exe, *args], capture_output=True).returncode == 1  # noqa: S603


def nitrogen(path: Path) -> bool:
    return _detects("nitrogen", "scan", str(path))


def picklescan(path: Path) -> bool:
    return _detects("picklescan", "-p", str(path))


def modelscan(path: Path) -> bool:
    return _detects("modelscan", "-p", str(path))


TOOLS = {"Nitrogen": nitrogen, "picklescan": picklescan, "ModelScan": modelscan}


def main() -> None:
    malicious = [c for c in CASES if c["verdict"] == "risk"]
    healthy = [c for c in CASES if c["verdict"] == "safe"]
    rows = {}
    for name, run in TOOLS.items():
        caught = sum(run(CORPUS / c["file"]) for c in malicious)
        false_pos = sum(run(CORPUS / c["file"]) for c in healthy)
        rows[name] = (caught, false_pos)

    print(f"Corpus: {len(malicious)} malicious, {len(healthy)} healthy\n")
    print(f"| Tool | Detected (of {len(malicious)}) | False positives (of {len(healthy)}) |")
    print("|---|---|---|")
    for name, (caught, false_pos) in rows.items():
        print(f"| {name} | {caught} | {false_pos} |")


if __name__ == "__main__":
    main()

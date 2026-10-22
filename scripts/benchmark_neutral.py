"""Neutral benchmark (ADR 0007): run Nitrogen, picklescan and ModelScan on a
third-party corpus we did not design — picklescan's own test fixtures, vendored
under scanner/tests/vendor/picklescan/.

Two axes:
- depth: on the vendored pickle corpus, who detects the malicious files and who
  false-positives on the benign ones;
- breadth: which model formats each tool actually analyzes (a capability matrix,
  since the competitors generate their non-pickle fixtures at test time).

Each tool runs through its CLI in a subprocess; no file is ever loaded. Run from
the scanner directory:
    uv run python ../scripts/benchmark_neutral.py
"""

import shutil
import subprocess
from pathlib import Path

VENDOR = Path(__file__).resolve().parents[1] / "scanner" / "tests" / "vendor" / "picklescan"


def _detects(tool: str, *args: str) -> bool:
    exe = shutil.which(tool)
    if exe is None:
        raise SystemExit(f"{tool} not found; install the dev dependencies")
    return subprocess.run([exe, *args], capture_output=True).returncode == 1  # noqa: S603


TOOLS = {
    "Nitrogen": lambda p: _detects("nitrogen", "scan", str(p)),
    "picklescan": lambda p: _detects("picklescan", "-p", str(p)),
    "ModelScan": lambda p: _detects("modelscan", "-p", str(p)),
}


def _is_malicious(name: str) -> bool:
    lowered = name.lower()
    return lowered.startswith(("malicious", "bad", "broken")) or "sploit" in lowered


# Which formats each tool analyzes (not just recognizes). From each project's
# docs/scanners; the competitors' non-pickle fixtures are generated at runtime,
# so breadth is a capability matrix, not a detection count.
BREADTH = {
    "pickle / PyTorch zip / tar": {"Nitrogen": "yes", "picklescan": "yes", "ModelScan": "yes"},
    "safetensors": {"Nitrogen": "yes", "picklescan": "no", "ModelScan": "no"},
    "TensorFlow SavedModel": {"Nitrogen": "no", "picklescan": "no", "ModelScan": "yes"},
    "Keras / HDF5": {"Nitrogen": "no", "picklescan": "no", "ModelScan": "yes"},
    "remote code (.py, trust_remote_code)": {
        "Nitrogen": "no",
        "picklescan": "no",
        "ModelScan": "no",
    },
}


def main() -> None:
    files = sorted(p for p in VENDOR.iterdir() if p.suffix != ".md" and p.name != "LICENSE")
    malicious = [p for p in files if _is_malicious(p.name)]
    benign = [p for p in files if not _is_malicious(p.name)]

    counts = f"{len(malicious)} malicious, {len(benign)} benign"
    print(f"## Depth — picklescan's own corpus ({counts})\n")
    print("| Tool | Detected | Missed | False positives |")
    print("|---|---|---|---|")
    for name, run in TOOLS.items():
        caught = [p for p in malicious if run(p)]
        fp = [p for p in benign if run(p)]
        missed = ", ".join(p.name for p in malicious if p not in caught) or "—"
        print(f"| {name} | {len(caught)}/{len(malicious)} | {missed} | {len(fp)}/{len(benign)} |")

    print("\n## Breadth — formats analyzed\n")
    print("| Format | " + " | ".join(TOOLS) + " |")
    print("|---|" + "---|" * len(TOOLS))
    for fmt, row in BREADTH.items():
        print(f"| {fmt} | " + " | ".join(row[t] for t in TOOLS) + " |")


if __name__ == "__main__":
    main()

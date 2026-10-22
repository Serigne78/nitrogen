"""Measure false positives on real Hugging Face models (ADR 0007).

Downloads a few small, popular, trusted models and scans them: every one must
come back COMPLIANT. This needs the network, so it is a manual script, never
part of the pytest suite (which stays offline). The models are not versioned;
only the result goes in the README. Scanning never loads a model.

    uv run python ../scripts/false_positives.py
"""

import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

BASE = "https://huggingface.co"
# (repo, file) of small, widely-used, trusted models — a real pickle zip and
# real safetensors are the cases most likely to trip a scanner.
MODELS = [
    ("prajjwal1/bert-tiny", "pytorch_model.bin"),
    ("prajjwal1/bert-tiny", "config.json"),
    ("hf-internal-testing/tiny-random-bert", "model.safetensors"),
    ("hf-internal-testing/tiny-random-gpt2", "model.safetensors"),
]


def _download(repo: str, name: str, into: Path) -> Path | None:
    url = f"{BASE}/{repo}/resolve/main/{name}"
    target = into / repo.replace("/", "__") / name
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 - fixed https host
            target.write_bytes(resp.read())
    except Exception as exc:
        print(f"  skip {repo}/{name}: {exc}")
        return None
    return target


def main() -> None:
    nitrogen = __import__("shutil").which("nitrogen")
    if nitrogen is None:
        raise SystemExit("nitrogen not found; run from the scanner directory under uv")
    false_positives = 0
    scanned = 0
    with tempfile.TemporaryDirectory() as tmp:
        for repo, name in MODELS:
            path = _download(repo, name, Path(tmp))
            if path is None:
                continue
            scanned += 1
            code = subprocess.run([nitrogen, "scan", str(path)], capture_output=True).returncode  # noqa: S603
            verdict = {0: "COMPLIANT", 1: "RISK", 2: "ERROR"}.get(code, str(code))
            print(f"  {repo}/{name}: {verdict}")
            false_positives += code != 0
    print(f"\n{scanned} files scanned, {false_positives} false positive(s).")
    sys.exit(1 if false_positives else 0)


if __name__ == "__main__":
    main()

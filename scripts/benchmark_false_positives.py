"""False-positive benchmark on real Hugging Face models (roadmap step 9).

A security scanner that cries wolf is ignored. This benchmark takes a frozen
list of popular, trusted models — each pinned to an exact commit revision so the
run is reproducible — downloads them and scans each one. Every finding is, by
construction, a false positive: these models are benign, so a RISK verdict means
the scanner is wrong, and the published rate is the number that matters.

Downloading needs the network, so this is opt-in and never part of the pytest
suite (which stays offline, roadmap principle 5). Only the scanner's exit code
and printed verdict are read; a model is never loaded or deserialized.

    uv run python ../scripts/benchmark_false_positives.py \
        --models ../scripts/models.txt \
        --out ../docs/benchmark-faux-positifs.md

The run is resumable: results are written to a state file after every model, so
an interrupted run (or a later one that lifts --max-bytes to reach the large
models) continues where it stopped. Pinned models over --max-bytes are recorded
as skipped, not silently dropped.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Quieter, gentler downloads: no progress-bar spam in the log, no symlink
# warning on Windows (we download into a flat local_dir anyway).
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Files the scanner analyses in v0.1: safetensors, pickle zips (.bin/.pt/.pth/
# .ckpt) and raw pickles, plus the JSON config that comes with every model.
# Downloading only these keeps a run tractable and measures exactly the formats
# v0.1 claims to cover.
WEIGHT_PATTERNS = ["*.safetensors", "*.bin", "*.pt", "*.pth", "*.ckpt", "*.pkl", "*.json"]
DEFAULT_MAX_BYTES = 2 * 1024**3  # 2 GiB of weights per model, by default
_SHA = re.compile(r"[0-9a-f]{40}")
# A verdict/finding line printed by `nitrogen scan`, e.g. "[risk] [pickle.import]".
_FINDING = re.compile(r"\[(?:risk|warning|info)\] \[([^\]]+)\]")


class ModelListError(ValueError):
    """The frozen model list is malformed or not actually pinned."""


@dataclass(frozen=True)
class ModelPin:
    repo: str
    revision: str

    @classmethod
    def parse(cls, spec: str) -> ModelPin:
        parts = spec.split("@")
        if len(parts) != 2:
            raise ModelListError(f"expected 'org/model@<sha>': {spec!r}")
        repo, revision = parts[0].strip(), parts[1].strip()
        segments = repo.split("/")
        if len(segments) != 2 or not all(segments):
            raise ModelListError(f"repo must be 'org/model': {spec!r}")
        if not _SHA.fullmatch(revision):
            raise ModelListError(f"revision must be a full 40-hex commit sha (pinned): {spec!r}")
        return cls(repo, revision)


@dataclass
class Result:
    repo: str
    revision: str
    status: str  # compliant | warning | risk | error | skipped
    exit_code: int | None
    rules: list[str] = field(default_factory=list)
    size_bytes: int | None = None
    note: str = ""


@dataclass
class Summary:
    compliant: int
    warnings: int
    false_positives: int
    errors: int
    skipped: int
    evaluated: int
    false_positive_rate: float


# --- pure logic (covered by tests/unit/test_benchmark_fp.py, offline) -------


def parse_model_list(text: str) -> list[ModelPin]:
    """Parse the frozen list: one `org/model@<sha>` per line, `#` comments and
    blank lines ignored. A line that is not pinned to a full commit sha, or a
    repo pinned twice, is an error — a frozen list that is not frozen would make
    the published number unreproducible."""
    pins: list[ModelPin] = []
    seen: set[str] = set()
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        pin = ModelPin.parse(line)
        if pin.repo in seen:
            raise ModelListError(f"duplicate repo: {pin.repo}")
        seen.add(pin.repo)
        pins.append(pin)
    return pins


def summarize(results: list[Result]) -> Summary:
    def count(status: str) -> int:
        return sum(r.status == status for r in results)

    compliant, warnings = count("compliant"), count("warning")
    risks, errors, skipped = count("risk"), count("error"), count("skipped")
    evaluated = compliant + warnings + risks + errors
    rate = risks / evaluated if evaluated else 0.0
    return Summary(compliant, warnings, risks, errors, skipped, evaluated, rate)


def _human_size(n: int | None) -> str:
    if n is None:
        return "—"
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


_VERDICT = {"compliant": "COMPLIANT", "warning": "COMPLIANT (warnings)", "risk": "RISK",
            "error": "ERROR", "skipped": "skipped"}  # fmt: skip


def render_report(results: list[Result], summary: Summary) -> str:
    """Render the Markdown report. Pure and deterministic (no clock, no
    network): the generation date is added by main(), not here, so the report
    diffs cleanly when only the date would change."""
    pct = f"{summary.false_positive_rate * 100:.1f}%"
    lines = [
        "# False positives on real models",
        "",
        "Nitrogen run against a frozen list of popular, trusted Hugging Face",
        "models, each pinned to an exact commit revision. Every model is benign,",
        "so any RISK verdict is a false positive.",
        "",
        "## Result",
        "",
        f"- Models evaluated: **{summary.evaluated}**",
        f"- Compliant: {summary.compliant}",
        f"- Compliant with warnings: {summary.warnings}",
        f"- **False positives (RISK on a trusted model): {summary.false_positives}**",
        f"- Scanner errors (robustness): {summary.errors}",
        f"- Not evaluated (over size budget / unavailable): {summary.skipped}",
        f"- **False-positive rate: {pct}**",
        "",
    ]

    flagged = [r for r in results if r.status == "risk"]
    if flagged:
        lines += ["## False positives", "", "| Model | Revision | Findings |", "|---|---|---|"]
        lines += [f"| {r.repo} | `{r.revision[:12]}` | {', '.join(r.rules) or '—'} |"
                  for r in flagged]  # fmt: skip
        lines.append("")

    errored = [r for r in results if r.status == "error"]
    if errored:
        lines += ["## Scanner errors", "", "| Model | Revision | Note |", "|---|---|---|"]
        lines += [f"| {r.repo} | `{r.revision[:12]}` | {r.note or '—'} |" for r in errored]
        lines.append("")

    lines += ["## Per-model detail", "", "| Model | Revision | Size | Verdict | Findings |",
              "|---|---|---|---|---|"]  # fmt: skip
    for r in results:
        findings = ", ".join(r.rules) or ("—" if r.status != "skipped" else r.note or "—")
        lines.append(
            f"| {r.repo} | `{r.revision[:12]}` | {_human_size(r.size_bytes)} "
            f"| {_VERDICT.get(r.status, r.status)} | {findings} |"
        )
    lines.append("")
    return "\n".join(lines)


# --- network / scanning (opt-in, never imported by the test suite) ----------


def _classify(exit_code: int, stdout: str) -> tuple[str, list[str]]:
    rules = _FINDING.findall(stdout)
    if exit_code == 2:
        return "error", rules
    if exit_code == 1:
        return "risk", rules
    # Exit 0 is COMPLIANT; the CLI adds "(N warning(s))" when warnings remain.
    return ("warning" if "COMPLIANT (" in stdout else "compliant"), rules


def _download_size(info: object) -> int:
    total = 0
    for sibling in getattr(info, "siblings", None) or []:
        name = getattr(sibling, "rfilename", "")
        size = getattr(sibling, "size", None)
        if size and any(Path(name).match(pat) for pat in WEIGHT_PATTERNS):
            total += int(size)
    return total


def scan_model(pin: ModelPin, cache_dir: Path, max_bytes: int, nitrogen: str) -> Result:
    """Download the model's v0.1-relevant files at the pinned revision and scan
    the directory. The file is never loaded: nitrogen only reads opcodes."""
    from huggingface_hub import HfApi, snapshot_download  # lazy: keeps import offline
    from huggingface_hub.utils import HfHubHTTPError

    base = Result(pin.repo, pin.revision, "skipped", None)
    try:
        info = HfApi().model_info(pin.repo, revision=pin.revision, files_metadata=True)
    except (HfHubHTTPError, OSError) as exc:
        base.note = f"unavailable: {type(exc).__name__}"
        return base
    size = _download_size(info)
    base.size_bytes = size
    if size == 0:
        base.note = "no v0.1-relevant file"
        return base
    if size > max_bytes:
        base.note = f"over budget ({_human_size(size)} > {_human_size(max_bytes)})"
        return base
    # local_dir downloads the real files into a flat directory (no blob/symlink
    # cache), which the scanner reads directly and which avoids the Windows
    # "no symlink support" finalization failure.
    into = cache_dir / pin.repo.replace("/", "__")
    try:
        local = snapshot_download(
            pin.repo,
            revision=pin.revision,
            allow_patterns=WEIGHT_PATTERNS,
            local_dir=str(into),
            max_workers=2,
        )
    except (HfHubHTTPError, OSError) as exc:
        base.note = f"download failed: {type(exc).__name__}"
        return base
    # local_dir leaves a .cache/huggingface/ tracking tree (empty, sometimes
    # unlistable on Windows); it is not model content, so drop it before the
    # scan sees the directory a user would scan.
    shutil.rmtree(Path(local) / ".cache", ignore_errors=True)
    proc = subprocess.run(  # noqa: S603 - fixed exe, path is a local cache dir
        [nitrogen, "scan", local], capture_output=True, text=True, check=False
    )
    status, rules = _classify(proc.returncode, proc.stdout)
    return Result(pin.repo, pin.revision, status, proc.returncode, rules, size)


def _load_state(path: Path) -> dict[str, Result]:
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {repo: Result(**data) for repo, data in raw.items()}


def _save_state(path: Path, done: dict[str, Result]) -> None:
    path.write_text(
        json.dumps({repo: asdict(r) for repo, r in done.items()}, indent=2), encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--models", type=Path, default=here / "models.txt")
    parser.add_argument("--out", type=Path, default=here.parent / "docs" / "benchmark-faux-positifs.md")  # noqa: E501
    parser.add_argument("--state", type=Path, default=None, help="Resumable results JSON.")
    parser.add_argument("--cache-dir", type=Path, default=None, help="HF download cache.")
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    parser.add_argument("--limit", type=int, default=None, help="Scan at most this many models.")
    args = parser.parse_args(argv)

    nitrogen = shutil.which("nitrogen")
    if nitrogen is None:
        raise SystemExit("nitrogen not found; run from the scanner directory under uv")
    pins = parse_model_list(args.models.read_text(encoding="utf-8"))
    if args.limit is not None:
        pins = pins[: args.limit]

    state_path = args.state or args.out.with_suffix(".state.json")
    done = _load_state(state_path)
    cache_ctx = (
        tempfile.TemporaryDirectory() if args.cache_dir is None else None
    )
    cache_dir = Path(cache_ctx.name) if cache_ctx is not None else args.cache_dir
    try:
        for i, pin in enumerate(pins, 1):
            if pin.repo in done and done[pin.repo].status != "skipped":
                continue  # already evaluated in a previous run
            print(f"[{i}/{len(pins)}] {pin.repo}@{pin.revision[:12]} ...", flush=True)
            result = scan_model(pin, cache_dir, args.max_bytes, nitrogen)
            print(f"    -> {result.status} {result.rules or ''} {result.note}".rstrip())
            done[pin.repo] = result
            _save_state(state_path, done)
    finally:
        if cache_ctx is not None:
            cache_ctx.cleanup()

    results = [done[p.repo] for p in pins if p.repo in done]
    summary = summarize(results)
    args.out.write_text(render_report(results, summary), encoding="utf-8")
    print(f"\n{summary.evaluated} evaluated, {summary.false_positives} false positive(s) "
          f"({summary.false_positive_rate * 100:.1f}%). Report: {args.out}")  # fmt: skip
    return 1 if summary.false_positives else 0


if __name__ == "__main__":
    sys.exit(main())

"""`nitrogen` command line.

The scanner does not know whether it runs on a laptop or in CI: the default
output is a human-readable verdict, the AI-BOM is only written with `-o`, and
the exit code is always meaningful.
"""

from enum import IntEnum
from pathlib import Path
from typing import Annotated

import typer

from nitrogen import __version__
from nitrogen.aibom import build_aibom, write_aibom
from nitrogen.formats import Format
from nitrogen.inventory import FileEntry, inventory
from nitrogen.policy import Policy, apply_policy, load_policy


class ExitCode(IntEnum):
    COMPLIANT = 0
    RISK = 1
    ERROR = 2


app = typer.Typer(
    help="Supply-chain security for AI models.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)


def _version(value: bool) -> None:
    if value:
        typer.echo(f"nitrogen {__version__}")
        raise typer.Exit(ExitCode.COMPLIANT)


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version, is_eager=True, help="Show version."),
    ] = False,
) -> None:
    """Supply-chain security for AI models."""


@app.command()
def scan(
    path: Annotated[Path, typer.Argument(help="Model file or directory to scan.")],
    output: Annotated[
        Path | None,
        typer.Option("-o", "--output", help="Write the CycloneDX AI-BOM to this file."),
    ] = None,
    policy: Annotated[
        Path | None,
        typer.Option("--policy", help="YAML policy file."),
    ] = None,
) -> None:
    """Scan a model without executing it.

    Exit code: 0 = compliant, 1 = risk detected, 2 = execution error.
    """
    try:
        code = _scan(path, output, policy)
    except Exception as exc:
        typer.echo(f"error: {_printable(str(exc))}", err=True)
        code = ExitCode.ERROR
    except SystemExit as exc:
        # A sys.exit() buried in a dependency would otherwise leak its own
        # code, 0 meaning "compliant" for a model nobody checked. Ctrl+C
        # (KeyboardInterrupt) still goes through.
        typer.echo(f"error: unexpected exit ({exc.code}) during the scan", err=True)
        code = ExitCode.ERROR
    raise typer.Exit(code)


def _scan(path: Path, output: Path | None, policy_path: Path | None) -> ExitCode:
    # Fail fast: everything that can fail for sure is checked before the scan,
    # which can take minutes on a large model.
    if not path.exists():
        raise FileNotFoundError(f"no such file or directory: {path}")
    if policy_path is not None and not policy_path.is_file():
        raise FileNotFoundError(f"policy file not found: {policy_path}")
    if output is not None:
        _check_output(path, output)
    policy = load_policy(policy_path) if policy_path is not None else Policy()

    entries = apply_policy(policy, inventory(path))
    if output is not None:
        _check_output_not_scanned(path, output, entries)
    _print_inventory(path, entries)
    findings = [(e.path, f) for e in entries for f in e.findings]
    if findings:
        typer.echo("Findings:")
        for file, finding in findings:
            # The message carries attacker-chosen names: escaped like the path.
            line = _printable(f"[{finding.rule}] {finding.message}")
            typer.echo(f"  {_printable(str(file))}  {line}")
    if output is not None:
        # Written for both verdicts: the evidence of a refusal matters too.
        write_aibom(output, build_aibom(path.resolve().name or str(path), entries, policy))
        typer.echo(f"AI-BOM: {_printable(str(output))}")
    if findings:
        typer.echo(f"Verdict: RISK ({_plural(len(findings), 'finding')})")
        return ExitCode.RISK
    typer.echo("Verdict: COMPLIANT")
    return ExitCode.COMPLIANT


def _check_output(scanned: Path, output: Path) -> None:
    """Refuse, before scanning, an AI-BOM target that would overwrite the
    model or land inside the scanned tree (a later scan would hash a stale
    AI-BOM, and its digest would describe files that no longer exist)."""
    if not output.parent.is_dir():
        raise FileNotFoundError(f"output directory not found: {output.parent}")
    if output.is_dir():
        raise IsADirectoryError(f"output is a directory: {output}")
    target, root = output.resolve(), scanned.resolve()
    inside = target == root or (root.is_dir() and target.is_relative_to(root))
    # samefile also catches the same directory reached another way: different
    # case on a case-insensitive volume, a bind mount.
    if not inside and root.is_dir():
        inside = any(_same(ancestor, root) for ancestor in (target.parent, *target.parents))
    if not inside and target.exists():
        inside = _same(target, root)
    if inside:
        raise ValueError(f"output must be outside the scanned path: {output}")


def _check_output_not_scanned(scanned: Path, output: Path, entries: list[FileEntry]) -> None:
    """After the inventory: refuse a target that is the real file behind a
    scanned symlink (a Hugging Face snapshot points into blobs/)."""
    base = scanned if scanned.is_dir() else scanned.parent
    target = output.resolve()
    if any((base / e.path).resolve() == target for e in entries):
        raise ValueError(f"output is a file of the scanned model: {output}")


def _same(a: Path, b: Path) -> bool:
    try:
        return a.samefile(b)
    except OSError:
        return False


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}{'' if count == 1 else 's'}"


def _print_inventory(path: Path, entries: list[FileEntry]) -> None:
    typer.echo(f"Scanned: {_printable(str(path))} ({_plural(len(entries), 'file')})")
    if not entries:
        return
    rows = [("FILE", "FORMAT", "SIZE", "SHA-256")] + [
        (_printable(str(e.path)), _format_label(e.formats), str(e.size), e.sha256) for e in entries
    ]
    name_w = max(len(r[0]) for r in rows)
    fmt_w = max(len(r[1]) for r in rows)
    size_w = max(len(r[2]) for r in rows)
    for name, fmt, size, sha in rows:
        typer.echo(f"{name:<{name_w}}  {fmt:<{fmt_w}}  {size:>{size_w}}  {sha}")


def _format_label(formats: frozenset[Format]) -> str:
    return "+".join(sorted(formats)) or "unknown"


def _printable(text: str) -> str:
    # A file named "x\nVerdict: COMPLIANT" (or with ANSI escapes) must not
    # forge output lines: anything non-printable is shown escaped.
    return text if text.isprintable() else repr(text)

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
from nitrogen.formats import Format
from nitrogen.inventory import ANALYZED_FORMATS, FileEntry, inventory


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
        typer.echo(f"error: {exc}", err=True)
        code = ExitCode.ERROR
    except SystemExit as exc:
        # A sys.exit() buried in a dependency would otherwise leak its own
        # code, 0 meaning "compliant" for a model nobody checked. Ctrl+C
        # (KeyboardInterrupt) still goes through.
        typer.echo(f"error: unexpected exit ({exc.code}) during the scan", err=True)
        code = ExitCode.ERROR
    raise typer.Exit(code)


def _scan(path: Path, output: Path | None, policy: Path | None) -> ExitCode:
    if not path.exists():
        raise FileNotFoundError(f"no such file or directory: {path}")
    if policy is not None and not policy.is_file():
        raise FileNotFoundError(f"policy file not found: {policy}")
    if output is not None:
        raise NotImplementedError("AI-BOM output (-o) is not implemented yet")

    entries = inventory(path)
    _print_inventory(path, entries)
    findings = [(e.path, f) for e in entries for f in e.findings]
    if findings:
        typer.echo("Findings:")
        for file, finding in findings:
            typer.echo(f"  {_printable(str(file))}  [{finding.rule}] {finding.message}")
        typer.echo(f"Verdict: RISK ({_plural(len(findings), 'finding')})")
        return ExitCode.RISK
    unanalyzed = [e for e in entries if e.formats - ANALYZED_FORMATS]
    if unanalyzed:
        # Pickle analysis comes in step 5: do not let COMPLIANT oversell.
        formats = ", ".join(
            sorted({str(f) for e in unanalyzed for f in e.formats - ANALYZED_FORMATS})
        )
        note = f"{_plural(len(unanalyzed), 'file')} ({formats}) not analyzed yet"
        typer.echo(f"Verdict: COMPLIANT ({note})")
    else:
        typer.echo("Verdict: COMPLIANT")
    return ExitCode.COMPLIANT


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

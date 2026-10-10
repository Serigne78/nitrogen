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
    raise typer.Exit(code)


def _scan(path: Path, output: Path | None, policy: Path | None) -> ExitCode:
    if not path.exists():
        raise FileNotFoundError(f"no such file or directory: {path}")
    if policy is not None and not policy.is_file():
        raise FileNotFoundError(f"policy file not found: {policy}")
    if output is not None:
        raise NotImplementedError("AI-BOM output (-o) is not implemented yet")

    # Inventory and analysis come in the next roadmap steps.
    typer.echo(f"Scanned: {path}")
    typer.echo("Verdict: COMPLIANT (no analyzers enabled yet)")
    return ExitCode.COMPLIANT

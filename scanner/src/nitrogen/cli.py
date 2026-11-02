"""`nitrogen` command line.

The scanner does not know whether it runs on a laptop or in CI: the default
output is a human-readable verdict, the AI-BOM is only written with `-o`, and
the exit code is always meaningful.
"""

import tempfile
from contextlib import ExitStack
from dataclasses import replace
from enum import IntEnum
from pathlib import Path, PurePosixPath
from typing import Annotated

import typer

from nitrogen import __version__
from nitrogen.aibom import build_aibom, write_aibom
from nitrogen.chat_template import analyze_chat_templates
from nitrogen.findings import Finding
from nitrogen.formats import Format
from nitrogen.inventory import FileEntry, inventory
from nitrogen.policy import Policy, apply_policy, is_allowed, is_blocking, load_policy
from nitrogen.remote_code import analyze_remote_code
from nitrogen.source import (
    HfSource,
    LocalSource,
    check_integrity,
    integrity_coverage,
    missing_files,
    parse_target,
    resolve,
)


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
    target: Annotated[
        str,
        typer.Argument(help="Model file or directory, or hf://org/model[@revision]."),
    ],
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
        code = _scan(target, output, policy)
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


def _scan(target: str, output: Path | None, policy_path: Path | None) -> ExitCode:
    # Fail fast: everything that can fail for sure is checked before the scan,
    # which can take minutes on a large model.
    source = parse_target(target)
    if isinstance(source, LocalSource) and not source.path.exists():
        raise FileNotFoundError(f"no such file or directory: {source.path}")
    if policy_path is not None and not policy_path.is_file():
        raise FileNotFoundError(f"policy file not found: {policy_path}")
    policy = load_policy(policy_path) if policy_path is not None else Policy()

    with ExitStack() as stack:
        # An hf:// source is downloaded (never loaded) into a temp dir for the
        # duration of the scan; a local source stays offline, in place.
        dest = None
        if not isinstance(source, LocalSource):
            dest = Path(
                stack.enter_context(
                    # ignore_cleanup_errors: a read-only downloaded file must not
                    # turn a clean verdict into an exit-2 error at teardown (Windows).
                    tempfile.TemporaryDirectory(prefix="nitrogen-hf-", ignore_cleanup_errors=True)
                )
            )
        resolved = resolve(source, dest=dest)
        local = resolved.local_dir
        if output is not None:
            _check_output(local, output)

        scanned = analyze_chat_templates(analyze_remote_code(inventory(local), local), local)
        # Integrity (ADR 0014): a local SHA-256 differing from the Hub's is a
        # risk; a local file the Hub does not list is injected (warning). Both
        # attach to the file they concern. Files the Hub lists but that are
        # absent locally are source-level (integrity.missing).
        per_file: dict[PurePosixPath, list[Finding]] = {}
        for ipath, finding in check_integrity(
            scanned, resolved.expected_sha256, resolved.published
        ):
            per_file.setdefault(ipath, []).append(finding)
        scanned = [
            replace(e, findings=e.findings + tuple(per_file[e.path])) if e.path in per_file else e
            for e in scanned
        ]
        entries = apply_policy(policy, scanned)
        if output is not None:
            _check_output_not_scanned(local, output, entries)
        _print_inventory(local, entries)

        # (label, finding): per-file findings keyed by path; source-level ones
        # (unpinned revision, missing files) by the source URI.
        source_label = resolved.source_uri or "model"
        source_findings = [*resolved.findings, *missing_files(resolved.published, entries)]
        findings = [(str(e.path), f) for e in entries for f in e.findings]
        findings += [(source_label, f) for f in source_findings]
        if findings:
            typer.echo("Findings:")
            for label, finding in findings:
                # The message carries attacker-chosen names: escaped like the path.
                allowed = " (allowed by policy)" if is_allowed(finding, policy) else ""
                line = _printable(
                    f"[{finding.severity}] [{finding.rule}] {finding.message}{allowed}"
                )
                typer.echo(f"  {_printable(label)}  {line}")
        # Make a remote-code risk understandable, not mistaken for noise: say the
        # model runs its own code and list the files to review (ADR 0012).
        code_files = sorted({label for label, f in findings if f.rule.startswith("remote_code.")})
        if code_files:
            typer.echo(
                "Note: this model runs its own code if loaded with trust_remote_code / torch.hub."
            )
            typer.echo(
                "      Review before loading: " + ", ".join(_printable(c) for c in code_files)
            )
        if resolved.source_uri is not None:
            typer.echo(f"Source: {_printable(resolved.source_uri)} @ {resolved.revision}")
        if output is not None:
            model_name = (
                source.repo
                if isinstance(source, HfSource)
                else (local.resolve().name or str(local))
            )
            # Written for every verdict: the evidence of a refusal matters too.
            write_aibom(
                output,
                build_aibom(
                    model_name,
                    entries,
                    policy,
                    source_findings=source_findings,
                    source_uri=resolved.source_uri,
                    revision=resolved.revision,
                    coverage=integrity_coverage(resolved),
                ),
            )
            typer.echo(f"AI-BOM: {_printable(str(output))}")

        blocking = [f for _, f in findings if is_blocking(f, policy)]
        warnings = [
            f
            for _, f in findings
            if f.severity == "warning" and not is_blocking(f, policy) and not is_allowed(f, policy)
        ]
        return _verdict(blocking, warnings)


def _verdict(blocking: list[Finding], warnings: list[Finding]) -> ExitCode:
    if blocking:
        typer.echo(f"Verdict: RISK ({_plural(len(blocking), 'finding')})")
        return ExitCode.RISK
    if warnings:
        typer.echo(f"Verdict: COMPLIANT ({_plural(len(warnings), 'warning')})")
    else:
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

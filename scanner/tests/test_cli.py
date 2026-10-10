from pathlib import Path

from typer.testing import CliRunner

from nitrogen import __version__
from nitrogen.cli import ExitCode, app

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == ExitCode.COMPLIANT
    assert __version__ in result.output


def test_no_args_shows_help() -> None:
    result = runner.invoke(app, [])
    assert "scan" in result.output


def test_scan_directory(tmp_path: Path) -> None:
    result = runner.invoke(app, ["scan", str(tmp_path)])
    assert result.exit_code == ExitCode.COMPLIANT
    assert "Verdict" in result.output


def test_scan_missing_path_is_error(tmp_path: Path) -> None:
    result = runner.invoke(app, ["scan", str(tmp_path / "nope")])
    assert result.exit_code == ExitCode.ERROR


def test_scan_missing_policy_is_error(tmp_path: Path) -> None:
    result = runner.invoke(app, ["scan", str(tmp_path), "--policy", str(tmp_path / "p.yaml")])
    assert result.exit_code == ExitCode.ERROR


def test_scan_accepts_existing_policy(tmp_path: Path) -> None:
    policy = tmp_path / "policy.yaml"
    policy.write_text("{}\n")
    result = runner.invoke(app, ["scan", str(tmp_path), "--policy", str(policy)])
    assert result.exit_code == ExitCode.COMPLIANT


def test_scan_output_not_implemented_is_error(tmp_path: Path) -> None:
    result = runner.invoke(app, ["scan", str(tmp_path), "-o", str(tmp_path / "bom.cdx.json")])
    assert result.exit_code == ExitCode.ERROR
    assert not (tmp_path / "bom.cdx.json").exists()

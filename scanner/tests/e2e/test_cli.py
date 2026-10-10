import hashlib
from pathlib import Path

import pytest
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


def test_scan_lists_file_format_and_hash(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_bytes(b"")
    (tmp_path / "model.safetensors").write_bytes(b"K\x01.")  # a pickle in disguise
    result = runner.invoke(app, ["scan", str(tmp_path)])
    assert result.exit_code == ExitCode.COMPLIANT
    assert "2 files" in result.output
    lines = {line.split()[0]: line.split() for line in result.output.splitlines() if line}
    assert lines["config.json"][1] == "unknown"
    assert lines["config.json"][3] == hashlib.sha256(b"").hexdigest()
    assert lines["model.safetensors"][1] == "pickle"
    assert lines["model.safetensors"][3] == hashlib.sha256(b"K\x01.").hexdigest()


def test_scan_single_file(tmp_path: Path) -> None:
    model = tmp_path / "model.bin"
    model.write_bytes(b"\x80\x02K\x01.")
    result = runner.invoke(app, ["scan", str(model)])
    assert result.exit_code == ExitCode.COMPLIANT
    assert "model.bin" in result.output
    assert "pickle" in result.output


def test_scan_unreadable_entry_is_error(tmp_path: Path) -> None:
    try:
        (tmp_path / "dangling").symlink_to(tmp_path / "missing")
    except OSError:
        pytest.skip("symlinks not supported here")
    result = runner.invoke(app, ["scan", str(tmp_path)])
    assert result.exit_code == ExitCode.ERROR


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

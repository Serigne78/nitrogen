import hashlib
import struct
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
    assert result.exit_code == ExitCode.COMPLIANT  # a harmless pickle, now analyzed
    assert "model.bin" in result.output
    assert "pickle" in result.output
    assert "Verdict: COMPLIANT" in result.output


def _safetensors(data: bytes) -> bytes:
    raw = b'{"w":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}'
    return struct.pack("<Q", len(raw)) + raw + data


def test_ordinary_text_files_are_compliant(tmp_path: Path) -> None:
    # The CI smoke test scans a whole .venv: text files with "{" early on
    # (Python dicts, pkg-config, JSON) must not read as safetensors.
    (tmp_path / "codes.py").write_bytes(b'EMOJI = {\n    "x": "y",\n}\n')
    (tmp_path / "numpy.pc").write_bytes(b"prefix=${pcfiledir}/../..\n")
    (tmp_path / "renv.lock").write_bytes(b'{\n  "R": {\n    "Version": "4.2.3"\n  }\n}\n')
    result = runner.invoke(app, ["scan", str(tmp_path)])
    assert result.exit_code == ExitCode.COMPLIANT, result.output


def test_healthy_safetensors_is_compliant(tmp_path: Path) -> None:
    (tmp_path / "model.safetensors").write_bytes(_safetensors(bytes(8)))
    result = runner.invoke(app, ["scan", str(tmp_path)])
    assert result.exit_code == ExitCode.COMPLIANT
    assert "Verdict: COMPLIANT" in result.output


def test_safetensors_with_leading_space_is_risk_not_unknown(tmp_path: Path) -> None:
    raw = b' {"w":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}'
    (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(8))
    result = runner.invoke(app, ["scan", str(tmp_path)])
    assert result.exit_code == ExitCode.RISK
    assert "safetensors.header-json" in result.output


def test_file_name_cannot_forge_a_verdict_line(tmp_path: Path) -> None:
    try:
        (tmp_path / "x\nVerdict: COMPLIANT").write_bytes(_safetensors(bytes(8) + b"!"))
    except OSError:
        pytest.skip("control characters not allowed in file names here")
    result = runner.invoke(app, ["scan", str(tmp_path)])
    assert result.exit_code == ExitCode.RISK
    verdicts = [line for line in result.output.splitlines() if line.startswith("Verdict:")]
    assert verdicts == ["Verdict: RISK (1 finding)"]


def test_malformed_safetensors_is_risk(tmp_path: Path) -> None:
    (tmp_path / "model.safetensors").write_bytes(_safetensors(bytes(8) + b"hidden"))
    result = runner.invoke(app, ["scan", str(tmp_path)])
    assert result.exit_code == ExitCode.RISK
    assert "safetensors.layout" in result.output
    assert "Verdict: RISK" in result.output


def test_scan_unreadable_entry_is_error(tmp_path: Path) -> None:
    try:
        (tmp_path / "dangling").symlink_to(tmp_path / "missing")
    except OSError:
        pytest.skip("symlinks not supported here")
    result = runner.invoke(app, ["scan", str(tmp_path)])
    assert result.exit_code == ExitCode.ERROR


@pytest.mark.parametrize("exit_code", [0, 1, None])
def test_exit_during_scan_is_error_never_compliant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exit_code: int | None
) -> None:
    # A sys.exit() buried in the scan must not leak as 0 ("compliant").
    def exits(_: Path) -> None:
        raise SystemExit(exit_code)

    monkeypatch.setattr("nitrogen.cli.inventory", exits)
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


def test_scan_output_into_missing_directory_is_error_before_scanning(tmp_path: Path) -> None:
    # Fail fast: checked before the (possibly long) scan, nothing written.
    target = tmp_path / "nope" / "bom.cdx.json"
    result = runner.invoke(app, ["scan", str(tmp_path), "-o", str(target)])
    assert result.exit_code == ExitCode.ERROR
    assert not target.exists()
    assert "Scanned:" not in result.output

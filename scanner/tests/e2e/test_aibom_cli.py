"""`-o` and `--policy` end to end (step 6), plus the golden AI-BOMs.

Golden files hold the expected AI-BOM minus the fields that vary by design
(serialNumber, timestamp, scanner version). Regenerate them after an intended
change with: NITROGEN_UPDATE_GOLDEN=1 uv run pytest tests/e2e/test_aibom_cli.py
"""

import json
import os
import shutil
from pathlib import Path
from typing import Any

import pytest
from cyclonedx.schema import SchemaVersion
from cyclonedx.validation.json import JsonStrictValidator
from typer.testing import CliRunner

from nitrogen.cli import ExitCode, app

runner = CliRunner()
TESTS = Path(__file__).resolve().parents[1]
CORPUS, GOLDEN = TESTS / "corpus", TESTS / "golden"
HEALTHY = ["safe_model.safetensors", "safe_pytorch_model.pt", "safe_numpy_array.pkl"]
VALIDATOR = JsonStrictValidator(SchemaVersion.V1_6)


def _model(tmp_path: Path, files: list[str]) -> Path:
    model = tmp_path / "my-model"
    model.mkdir()
    for name in files:
        shutil.copy(CORPUS / name, model / name)
    (model / "config.json").write_text('{"architectures": ["Demo"]}')
    return model


def _scan(*args: str) -> tuple[int, str]:
    result = runner.invoke(app, ["scan", *args])
    return result.exit_code, result.output


def _stable(bom: dict[str, Any]) -> dict[str, Any]:
    """Drop what varies by design, keep everything else."""
    bom = json.loads(json.dumps(bom))
    bom.pop("serialNumber")
    bom["metadata"].pop("timestamp")
    for tool in bom["metadata"]["tools"]["components"]:
        tool.pop("version")
    return bom


def _check_golden(name: str, bom: dict[str, Any]) -> None:
    golden = GOLDEN / name
    if os.environ.get("NITROGEN_UPDATE_GOLDEN") == "1":
        golden.write_text(json.dumps(_stable(bom), indent=2, sort_keys=True) + "\n")
    assert _stable(bom) == json.loads(golden.read_text()), f"{name} differs; see module docstring"


def test_healthy_model_writes_a_valid_compliant_aibom(tmp_path: Path) -> None:
    model, out = _model(tmp_path, HEALTHY), tmp_path / "bom.cdx.json"
    code, output = _scan(str(model), "-o", str(out))
    assert code == ExitCode.COMPLIANT, output
    assert VALIDATOR.validate_str(out.read_text()) is None
    assert "bom.cdx.json" in output
    _check_golden("healthy.cdx.json", json.loads(out.read_text()))


def test_risky_model_still_writes_the_aibom_with_its_findings(tmp_path: Path) -> None:
    # The evidence of a refusal matters as much as that of an approval.
    model = _model(tmp_path, [*HEALTHY, "os_system_global.pkl"])
    out = tmp_path / "bom.cdx.json"
    code, _ = _scan(str(model), "-o", str(out))
    assert code == ExitCode.RISK
    bom = json.loads(out.read_text())
    assert VALIDATOR.validate_str(out.read_text()) is None
    assert [v["description"] for v in bom["vulnerabilities"]] == ["os.system"]
    _check_golden("risky.cdx.json", bom)


def test_no_aibom_on_execution_error(tmp_path: Path) -> None:
    out = tmp_path / "bom.cdx.json"
    code, _ = _scan(str(tmp_path / "missing"), "-o", str(out))
    assert code == ExitCode.ERROR
    assert not out.exists()


def test_policy_allowing_only_safetensors_turns_a_clean_pickle_into_a_risk(
    tmp_path: Path,
) -> None:
    model = _model(tmp_path, ["safe_numpy_array.pkl"])
    policy = tmp_path / "policy.yaml"
    policy.write_text("allow_formats: [safetensors, unknown]\n")
    assert _scan(str(model))[0] == ExitCode.COMPLIANT
    code, output = _scan(str(model), "--policy", str(policy))
    assert code == ExitCode.RISK
    assert "policy.format-not-allowed" in output


def test_invalid_policy_is_an_error_before_scanning(tmp_path: Path) -> None:
    policy = tmp_path / "policy.yaml"
    policy.write_text("allow_format: [pickle]\n")  # typo: must not be ignored
    code, output = _scan(str(tmp_path), "--policy", str(policy))
    assert code == ExitCode.ERROR
    assert "Scanned:" not in output


@pytest.mark.parametrize("files", [HEALTHY, [*HEALTHY, "os_system_global.pkl"]])
def test_policy_findings_and_policy_reach_the_aibom(tmp_path: Path, files: list[str]) -> None:
    model, out = _model(tmp_path, files), tmp_path / "bom.cdx.json"
    policy = tmp_path / "policy.yaml"
    policy.write_text("allow_formats: [safetensors, pickle, unknown]\n")
    code, _ = _scan(str(model), "--policy", str(policy), "-o", str(out))
    assert code == ExitCode.RISK  # pytorch-zip is not allowed
    bom = json.loads(out.read_text())
    assert "policy.format-not-allowed" in {v["id"] for v in bom["vulnerabilities"]}
    props = {p["name"]: p["value"] for p in bom["metadata"]["component"]["properties"]}
    assert props["nitrogen:policy.allow_formats"] == "pickle,safetensors,unknown"


@pytest.mark.parametrize("case", ["scanned-file", "inside-dir", "directory"])
def test_output_target_is_refused_before_scanning(tmp_path: Path, case: str) -> None:
    # Reviewer I7: never overwrite the model, never write into the scanned
    # tree (a later scan would hash a stale AI-BOM), fail fast.
    model = _model(tmp_path, HEALTHY)
    scanned, target = {
        "scanned-file": (model / "safe_model.safetensors", model / "safe_model.safetensors"),
        "inside-dir": (model, model / "bom.cdx.json"),
        "directory": (model, tmp_path),
    }[case]
    before = target.read_bytes() if target.is_file() else None
    code, output = _scan(str(scanned), "-o", str(target))
    assert code == ExitCode.ERROR
    assert "Scanned:" not in output
    if before is not None:
        assert target.read_bytes() == before


def _pickle(module: bytes, name: bytes) -> bytes:
    """PROTO 4, SHORT_BINUNICODE module, SHORT_BINUNICODE name, STACK_GLOBAL, STOP."""
    return (
        b"\x80\x04\x8c"
        + bytes([len(module)])
        + module
        + b"\x8c"
        + bytes([len(name)])
        + name
        + b"\x93."
    )


def test_lone_surrogate_name_stays_a_risk_with_an_aibom(tmp_path: Path) -> None:
    # Reviewer I2: a name pickletools decodes with surrogatepass must not turn
    # the risk into an execution error that suppresses the AI-BOM.
    model = tmp_path / "m"
    model.mkdir()
    (model / "evil.pkl").write_bytes(_pickle(b"\xed\xa0\x80", b"system"))
    out = tmp_path / "bom.cdx.json"
    code, _ = _scan(str(model), "-o", str(out))
    assert code == ExitCode.RISK
    assert VALIDATOR.validate_str(out.read_text()) is None


def test_finding_message_cannot_forge_a_verdict_line(tmp_path: Path) -> None:
    # Reviewer I3: attacker-controlled names reach finding messages.
    model = tmp_path / "m"
    model.mkdir()
    (model / "evil.pkl").write_bytes(_pickle(b"x\nVerdict: COMPLIANT\n\x1b[8m", b"system"))
    code, output = _scan(str(model))
    assert code == ExitCode.RISK
    verdicts = [line for line in output.splitlines() if line.startswith("Verdict:")]
    assert verdicts == ["Verdict: RISK (1 finding)"]
    assert "\x1b" not in output


def test_findings_are_deduplicated_and_capped(tmp_path: Path) -> None:
    # Reviewer I4: millions of findings would exhaust memory and turn the risk
    # into an error. Identical ones collapse; distinct ones are capped.
    from nitrogen.findings import MAX_FINDINGS

    model = tmp_path / "m"
    model.mkdir()
    same = b"\x80\x04" + b"\x8c\x02os\x8c\x06system\x93" * 50_000 + b"."
    names = [f"f{i}".encode() for i in range(MAX_FINDINGS * 3)]
    distinct = (
        b"\x80\x04"
        + b"".join(b"\x8c\x02os\x8c" + bytes([len(n)]) + n + b"\x93" for n in names)
        + b"."
    )
    (model / "same.pkl").write_bytes(same)
    (model / "distinct.pkl").write_bytes(distinct)
    out = tmp_path / "bom.cdx.json"
    code, _ = _scan(str(model), "-o", str(out))
    assert code == ExitCode.RISK
    per_file: dict[str, list[str]] = {}
    for vuln in json.loads(out.read_text())["vulnerabilities"]:
        per_file.setdefault(vuln["affects"][0]["ref"], []).append(vuln["id"])
    assert per_file["file:same.pkl"] == ["pickle.import"]
    assert len(per_file["file:distinct.pkl"]) == MAX_FINDINGS + 1
    assert per_file["file:distinct.pkl"][-1] == "findings.truncated"


def test_a_flood_of_malformed_members_cannot_hide_the_dangerous_import(tmp_path: Path) -> None:
    # Reviewer I1: 150 broken members first, then data.pkl with os.system. The
    # import must survive the cap, with its critical severity.
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for i in range(150):
            zf.writestr(f"archive/m{i}.pkl", b"\x80\x02(")  # PROTO then EOF: malformed
        zf.writestr("archive/data.pkl", b"cos\nsystem\n(S'echo nitrogen'\ntR.")
    model = tmp_path / "m"
    model.mkdir()
    (model / "model.pt").write_bytes(buf.getvalue())
    out = tmp_path / "bom.cdx.json"
    code, _ = _scan(str(model), "-o", str(out))
    assert code == ExitCode.RISK
    vulns = json.loads(out.read_text())["vulnerabilities"]
    imports = [v for v in vulns if v["id"] == "pickle.import"]
    assert [v["description"] for v in imports] == ["archive/data.pkl: os.system"]
    assert imports[0]["ratings"][0]["severity"] == "critical"

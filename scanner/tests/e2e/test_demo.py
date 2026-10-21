"""The demo transcript in docs/demo.txt is a real, reproducible session: a
clean model, then a trapped one. Regenerate it after an intended change with
NITROGEN_UPDATE_DEMO=1 uv run pytest tests/e2e/test_demo.py (step 7)."""

import os
import re
import shutil
from pathlib import Path

from typer.testing import CliRunner

from nitrogen.cli import app

DEMO = Path(__file__).resolve().parents[3] / "docs" / "demo.txt"
CORPUS = Path(__file__).resolve().parents[1] / "corpus"
runner = CliRunner()


def _render(display: str, model: Path, *args: str) -> str:
    result = runner.invoke(app, ["scan", str(model), *args])
    # Make the transcript stable across runs and OSes: temp paths, the file
    # count and the line endings all vary otherwise.
    output = result.output.replace(str(model), "./bert-model").replace("\\", "/")
    output = output.replace("\r\n", "\n")
    output = re.sub(r"\(\d+ files?\)", "(N files)", output)
    output = re.sub(r"AI-BOM: .*", "AI-BOM: bom.cdx.json", output)
    return f"$ {display}\n{output}(exit {result.exit_code})\n"


def _transcript(tmp_path: Path) -> str:
    clean = tmp_path / "bert-model"
    clean.mkdir()
    (clean / "config.json").write_text('{"architectures": ["BertModel"]}\n')
    shutil.copy(CORPUS / "safe_model.safetensors", clean / "model.safetensors")
    trapped = tmp_path / "trapped"
    trapped.mkdir()
    shutil.copy(CORPUS / "os_system_global.pkl", trapped / "pytorch_model.bin")

    header = "# Nitrogen demo — a clean model passes, a trapped one is caught.\n\n"
    return (
        header
        + "## A healthy model\n\n"
        + _render("nitrogen scan ./bert-model", clean)
        + "\n## A model with a malicious pickle\n\n"
        + _render(
            "nitrogen scan ./bert-model -o bom.cdx.json",
            trapped,
            "-o",
            str(tmp_path / "bom.cdx.json"),
        )
    )


def test_demo_transcript_is_current(tmp_path: Path) -> None:
    transcript = _transcript(tmp_path)
    if os.environ.get("NITROGEN_UPDATE_DEMO") == "1":
        DEMO.parent.mkdir(exist_ok=True)
        DEMO.write_text(transcript, encoding="utf-8", newline="\n")
    current = DEMO.read_text(encoding="utf-8").replace("\r\n", "\n")
    assert transcript == current, "stale; see module docstring"

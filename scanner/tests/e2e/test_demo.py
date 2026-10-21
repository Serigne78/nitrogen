"""docs/demo.txt is a committed, illustrative session: a clean model passes,
a trapped one is caught. The test checks the demo still reflects the real CLI
(the key lines, not a byte-exact golden, to stay stable across OSes) and keeps
docs/demo.txt in sync. Regenerate it with
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

# Lines that must appear for the demo to tell the truth.
CLEAN_MARKERS = ["config.json", "model.safetensors", "Verdict: COMPLIANT", "(exit 0)"]
TRAPPED_MARKERS = ["pytorch_model.bin", "[pickle.import] os.system", "Verdict: RISK", "(exit 1)"]


def _render(display: str, model: Path, *args: str) -> str:
    result = runner.invoke(app, ["scan", str(model), *args])
    output = result.output.replace(str(model), "./bert-model").replace("\\", "/")
    output = output.replace("\r\n", "\n")
    output = re.sub(r"\(\d+ files?\)", "(N files)", output)
    output = re.sub(r"AI-BOM: .*", "AI-BOM: bom.cdx.json", output)
    return f"$ {display}\n{output}(exit {result.exit_code})\n"


def _transcript(tmp_path: Path) -> tuple[str, str]:
    clean = tmp_path / "bert-model"
    clean.mkdir()
    (clean / "config.json").write_text('{"architectures": ["BertModel"]}\n')
    shutil.copy(CORPUS / "safe_model.safetensors", clean / "model.safetensors")
    trapped = tmp_path / "trapped"
    trapped.mkdir()
    shutil.copy(CORPUS / "os_system_global.pkl", trapped / "pytorch_model.bin")

    clean_out = _render("nitrogen scan ./bert-model", clean)
    trapped_out = _render(
        "nitrogen scan ./bert-model -o bom.cdx.json", trapped, "-o", str(tmp_path / "bom.cdx.json")
    )
    return clean_out, trapped_out


def test_demo_reflects_the_real_cli_and_is_in_sync(tmp_path: Path) -> None:
    clean_out, trapped_out = _transcript(tmp_path)
    for marker in CLEAN_MARKERS:
        assert marker in clean_out, marker
    for marker in TRAPPED_MARKERS:
        assert marker in trapped_out, marker

    if os.environ.get("NITROGEN_UPDATE_DEMO") == "1":
        DEMO.parent.mkdir(exist_ok=True)
        DEMO.write_text(
            "# Nitrogen demo — a clean model passes, a trapped one is caught.\n\n"
            "## A healthy model\n\n"
            + clean_out
            + "\n## A model with a malicious pickle\n\n"
            + trapped_out,
            encoding="utf-8",
            newline="\n",
        )
    demo = DEMO.read_text(encoding="utf-8").replace("\r\n", "\n")
    for marker in CLEAN_MARKERS + TRAPPED_MARKERS:
        assert marker in demo, f"docs/demo.txt is stale, missing {marker!r}"

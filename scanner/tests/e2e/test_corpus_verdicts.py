"""Step 5 done-when: the whole corpus is detected, no false positive on the
healthy files. One parametrized test reads cases.yaml; adding a bypass means
adding a file in generate.py and a row in cases.yaml.

The corpus is analyzed through the real pipeline (inventory: detection, then
every matching analyzer) and is never loaded.
"""

from pathlib import Path
from typing import Any

import pytest
import yaml

from nitrogen.inventory import inventory

CORPUS = Path(__file__).resolve().parents[1] / "corpus"
CASES: list[dict[str, Any]] = yaml.safe_load((CORPUS / "cases.yaml").read_text())


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["file"])
def test_corpus_verdict(case: dict[str, Any]) -> None:
    [entry] = inventory(CORPUS / case["file"])
    findings = entry.findings
    if case["verdict"] == "safe":
        assert findings == (), f"false positive: {findings}"
        return
    assert findings, "false negative: no finding"
    assert case["finding"] in {f.rule for f in findings}, findings
    if "detail" in case:
        assert any(case["detail"] in f.message for f in findings), findings

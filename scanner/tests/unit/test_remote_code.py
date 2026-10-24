"""Remote code analysis (step 10, ADR 0011).

A model can ship executable Python that runs at load time under
`trust_remote_code`, with no poisoned pickle. We inventory it from the configs
(`auto_map`, `custom_pipelines`) and the `.py` files, and read the code with
`ast.parse` only — it is never imported or executed.
"""

import json
from pathlib import Path

import pytest

from nitrogen import remote_code
from nitrogen.inventory import inventory
from nitrogen.remote_code import analyze_remote_code


def _scan(tmp_path: Path, files: dict[str, str]) -> dict[str, list[tuple[str, str]]]:
    for name, content in files.items():
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    entries = analyze_remote_code(inventory(tmp_path), tmp_path)
    return {str(e.path): [(f.rule, f.severity) for f in e.findings] for e in entries}


# --- auto_map wiring --------------------------------------------------------


def test_auto_map_to_local_py_is_risk(tmp_path: Path) -> None:
    out = _scan(
        tmp_path,
        {
            "config.json": json.dumps({"auto_map": {"AutoModel": "modeling_x.XModel"}}),
            "modeling_x.py": "import torch\n\nclass XModel:\n    pass\n",
        },
    )
    assert ("remote_code.auto_map", "risk") in out["config.json"]
    assert ("remote_code.python_file", "risk") in out["modeling_x.py"]


def test_auto_map_to_another_repo_is_risk(tmp_path: Path) -> None:
    # A cross-repo reference: code from `evil/repo` runs, and we cannot see it.
    out = _scan(
        tmp_path,
        {"config.json": json.dumps({"auto_map": {"AutoModel": "evil/repo--modeling_x.XModel"}})},
    )
    assert ("remote_code.external", "risk") in out["config.json"]


def test_custom_pipelines_reference_is_detected(tmp_path: Path) -> None:
    out = _scan(
        tmp_path,
        {
            "config.json": json.dumps(
                {"custom_pipelines": {"my-task": {"impl": "pipeline_x.MyPipeline"}}}
            ),
            "pipeline_x.py": "class MyPipeline:\n    pass\n",
        },
    )
    assert ("remote_code.python_file", "risk") in out["pipeline_x.py"]


def test_auto_map_in_tokenizer_config_is_detected(tmp_path: Path) -> None:
    out = _scan(
        tmp_path,
        {
            "tokenizer_config.json": json.dumps(
                {"auto_map": {"AutoTokenizer": ["tokenization_x.XTok", None]}}
            ),
            "tokenization_x.py": "class XTok:\n    pass\n",
        },
    )
    assert ("remote_code.python_file", "risk") in out["tokenization_x.py"]


# --- AST of the .py files ---------------------------------------------------


@pytest.mark.parametrize(
    "source",
    [
        "import os\nos.system('echo nitrogen')\n",
        "import subprocess\nsubprocess.Popen(['echo', 'nitrogen'])\n",
        "eval('1+1')\n",
        "exec('x=1')\n",
        "import socket\ns = socket.socket()\n",
        "import urllib.request\nurllib.request.urlopen('http://x')\n",
        "import base64\nbase64.b64decode('AAAA')\n",
    ],
    ids=["os.system", "subprocess", "eval", "exec", "socket", "urllib", "base64"],
)
def test_dangerous_call_in_unreferenced_py_is_risk(tmp_path: Path, source: str) -> None:
    # ADR 0011 #4: an unreferenced .py is a warning, but a dangerous call in it
    # elevates the verdict to risk (the dangerous_call finding is itself risk).
    out = _scan(tmp_path, {"helper.py": source})
    rules = out["helper.py"]
    assert ("remote_code.dangerous_call", "risk") in rules
    assert ("remote_code.python_file", "warning") in rules


def test_benign_unreferenced_py_is_a_warning_only(tmp_path: Path) -> None:
    out = _scan(
        tmp_path,
        {
            "config.json": json.dumps({"model_type": "bert"}),
            "utils.py": "import torch\nimport math\n\n\ndef f(x):\n    return math.sqrt(x)\n",
        },
    )
    assert out["utils.py"] == [("remote_code.python_file", "warning")]


def test_referenced_but_ast_clean_py_is_still_risk(tmp_path: Path) -> None:
    # Roadmap: a referenced .py is risk by default, even with a clean AST — the
    # static scan cannot prove the code is safe.
    out = _scan(
        tmp_path,
        {
            "config.json": json.dumps({"auto_map": {"AutoModel": "modeling_x.XModel"}}),
            "modeling_x.py": "import torch\n\nclass XModel:\n    pass\n",
        },
    )
    assert ("remote_code.python_file", "risk") in out["modeling_x.py"]
    assert not any(r == "remote_code.dangerous_call" for r, _ in out["modeling_x.py"])


# --- no false positives -----------------------------------------------------


def test_model_without_code_has_no_remote_code_finding(tmp_path: Path) -> None:
    out = _scan(
        tmp_path,
        {
            "config.json": json.dumps({"model_type": "bert", "hidden_size": 768}),
            "README.md": "# a model\n",
        },
    )
    assert not any(rule.startswith("remote_code") for rules in out.values() for rule, _ in rules)


def test_malformed_config_json_does_not_crash(tmp_path: Path) -> None:
    out = _scan(tmp_path, {"config.json": "{ not valid json "})
    assert not any(rule == "remote_code.auto_map" for rules in out.values() for rule, _ in rules)


def test_unparseable_py_is_still_reported_without_crashing(tmp_path: Path) -> None:
    # A .py that does not even parse: still flagged as present (no AST, no crash).
    out = _scan(tmp_path, {"broken.py": "def (= :::\n"})
    assert any(rule == "remote_code.python_file" for rule, _ in out["broken.py"])


# --- reviewer: fail-closed against the critical bypasses ---------------------


def test_model_in_subdirectory_is_still_risk(tmp_path: Path) -> None:
    # CRITIQUE 1: transformers resolves auto_map from the config's directory. A
    # model in a subfolder (e.g. an HF cache snapshot) must not degrade to
    # COMPLIANT. The .py uses an AST-evading payload to prove the verdict does
    # not depend on the AST.
    out = _scan(
        tmp_path,
        {
            "mymodel/config.json": json.dumps({"auto_map": {"AutoModel": "modeling_evil.Evil"}}),
            "mymodel/modeling_evil.py": "from os import system as _s\n_s('echo nitrogen')\n",
        },
    )
    assert ("remote_code.auto_map", "risk") in out["mymodel/config.json"]
    assert ("remote_code.python_file", "risk") in out["mymodel/modeling_evil.py"]


def test_auto_map_declared_but_file_not_located_is_still_risk(tmp_path: Path) -> None:
    # Fail-closed: a config that declares local custom code is risk even when we
    # cannot locate the exact .py — the user still passes trust_remote_code.
    out = _scan(tmp_path, {"config.json": json.dumps({"auto_map": {"AutoModel": "mystery.Model"}})})
    assert ("remote_code.auto_map", "risk") in out["config.json"]


def test_oversized_config_is_flagged_not_silently_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # CRITIQUE 2: padding a config past the cap must not hide its auto_map.
    monkeypatch.setattr(remote_code, "_MAX_CONFIG_BYTES", 100)
    padded = json.dumps({"_pad": "x" * 500, "auto_map": {"AutoModel": "modeling_x.X"}})
    out = _scan(tmp_path, {"config.json": padded})
    assert ("remote_code.config_unreadable", "risk") in out["config.json"]


def test_py_extension_case_is_analyzed_on_insensitive_fs(tmp_path: Path) -> None:
    # IMPORTANT 5: a `.PY` file is still analyzed, and a cased reference still
    # matches on a case-insensitive filesystem.
    out = _scan(
        tmp_path,
        {
            "config.json": json.dumps({"auto_map": {"AutoModel": "Modeling.Model"}}),
            "modeling.PY": "import torch\n",
        },
    )
    assert ("remote_code.python_file", "risk") in out["modeling.PY"]


@pytest.mark.parametrize(
    "source",
    [
        "import os as o\no.system('echo nitrogen')\n",
        "from os import system\nsystem('echo nitrogen')\n",
        "from base64 import b64decode\nb64decode('AAAA')\n",
        "import importlib as il\nil.import_module('os')\n",
        "import builtins as b\nb.eval('1')\n",
    ],
    ids=["import-as", "from-import", "from-base64", "importlib-as", "builtins-as"],
)
def test_ast_evasions_are_caught(tmp_path: Path, source: str) -> None:
    # IMPORTANT 3: aliased imports and from-imports of sensitive names no longer
    # slip past the AST.
    out = _scan(tmp_path, {"helper.py": source})
    assert ("remote_code.dangerous_call", "risk") in out["helper.py"]


def test_deeply_nested_config_does_not_crash(tmp_path: Path) -> None:
    # A crafted config must not overflow the stack: bounded depth, never an
    # error. It is reported as unreadable (rule 5), never a risk auto_map.
    nested = "[" * 5000 + "]" * 5000
    out = _scan(tmp_path, {"config.json": nested})
    assert not any(rule == "remote_code.auto_map" for rules in out.values() for rule, _ in rules)

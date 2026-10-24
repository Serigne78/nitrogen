"""Chat-template analysis (step 11, ADR 0013).

Chat templates are Jinja2, rendered at inference — a poisoned one escapes the
sandbox and runs code (CVE-2024-34359). We parse with jinja2's own parser
(`Environment().parse()`), never render, and walk the AST for sandbox-escape
patterns.
"""

import json
from pathlib import Path

import pytest

from nitrogen.chat_template import analyze_chat_templates, scan_template
from nitrogen.inventory import inventory


def _rules(source: str) -> list[tuple[str, str]]:
    return [(f.rule, f.severity) for f in scan_template(source)]


# --- sandbox escape (risk) --------------------------------------------------


@pytest.mark.parametrize(
    "source",
    [
        "{{ x.__class__.__mro__ }}",
        "{{ x['__class__'] }}",
        "{{ cycler.__init__.__globals__['os'].system('echo nitrogen') }}",
        "{{ x|attr('__class__') }}",
        "{% for c in ().__class__.__base__.__subclasses__() %}{{ c }}{% endfor %}",
        "{{ self.__init__.__globals__.__builtins__.__import__('os') }}",
    ],
    ids=["class-mro", "getitem", "globals-chain", "attr-filter", "subclasses", "builtins-import"],
)
def test_sandbox_escape_is_risk(source: str) -> None:
    assert ("chat_template.sandbox_escape", "risk") in _rules(source)


# --- benign real templates (no false positive) ------------------------------

_CHATML = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)
_LLAMA = (
    "{% set loop_messages = messages %}"
    "{% for message in loop_messages %}"
    "{{ bos_token + '[INST] ' + message['content'] | trim + ' [/INST]' }}"
    "{% endfor %}"
)


@pytest.mark.parametrize(
    "source", [_CHATML, _LLAMA, "{{ messages | tojson }}"], ids=["chatml", "llama", "tojson"]
)
def test_real_templates_are_clean(source: str) -> None:
    assert scan_template(source) == []


# --- reviewer: evasions that must not slip past (C2, I1, I2) ----------------


@pytest.mark.parametrize(
    "source",
    [
        "{{ x['__cla' + 'ss__'] }}",  # C2: concatenated key, constant-folded
        "{% set k = '__class__' %}{{ x[k] }}",  # C2: dunder named in a set-var const
        "{{ (x | map('attr', '__class__') | list)[0] }}",  # I1: map('attr', ...)
        "{{ x | selectattr('__globals__') | list }}",  # I1: selectattr
        "{% generation %}{{ x.__class__.__mro__ }}{% endgeneration %}",  # I2: hidden in generation
    ],
    ids=["concat-key", "set-var-key", "map-attr", "selectattr", "generation-wrapped"],
)
def test_ast_evasions_are_still_risk(source: str) -> None:
    assert ("chat_template.sandbox_escape", "risk") in _rules(source)


def test_generation_tag_alone_is_not_a_false_unparseable() -> None:
    # I2: a legitimate {% generation %} template must parse, not become unparseable.
    source = (
        "{% for m in messages %}{% generation %}{{ m['content'] }}{% endgeneration %}{% endfor %}"
    )
    assert scan_template(source) == []


def test_oversized_config_is_flagged_not_silently_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # C1: padding tokenizer_config.json past the cap must not hide its template.
    import nitrogen.chat_template as ct

    monkeypatch.setattr(ct, "_MAX_FILE_BYTES", 200)
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"_pad": "x" * 500, "chat_template": "{{ x.__class__ }}"}), encoding="utf-8"
    )
    entries = analyze_chat_templates(inventory(tmp_path), tmp_path)
    rules = [(f.rule, f.severity) for e in entries for f in e.findings]
    assert ("chat_template.oversized", "risk") in rules


# --- hidden instructions / obfuscation (warning) ----------------------------


def test_url_in_template_is_a_warning() -> None:
    rules = _rules("{% for m in messages %}{{ m }}{% endfor %}Visit https://evil.example/x now")
    assert ("chat_template.hidden_instructions", "warning") in rules


def test_long_base64_is_a_warning() -> None:
    blob = "QUJDREVG" * 20  # a long contiguous base64 run
    assert ("chat_template.obfuscated", "warning") in _rules("{{ messages }}" + blob)


def test_abnormally_long_template_is_a_warning() -> None:
    assert ("chat_template.obfuscated", "warning") in _rules("{{ messages }}" + "x" * 60000)


def test_malformed_template_is_reported_not_raised() -> None:
    rules = _rules("{{ unclosed ")
    assert rules == [("chat_template.unparseable", "warning")]


def test_deeply_nested_template_does_not_crash() -> None:
    # A pure-Python parser hits RecursionError before any C-level crash; it must
    # surface as a finding, never an exception.
    rules = _rules("{{ " + "(" * 5000)
    assert all(r.startswith("chat_template.") for r, _ in rules)


# --- extraction from the model files ----------------------------------------


def _scan(tmp_path: Path, files: dict[str, str]) -> dict[str, list[tuple[str, str]]]:
    for name, content in files.items():
        (tmp_path / name).write_text(content, encoding="utf-8")
    entries = analyze_chat_templates(inventory(tmp_path), tmp_path)
    return {str(e.path): [(f.rule, f.severity) for f in e.findings] for e in entries}


def test_template_in_tokenizer_config_is_analyzed(tmp_path: Path) -> None:
    out = _scan(
        tmp_path, {"tokenizer_config.json": json.dumps({"chat_template": "{{ x.__class__ }}"})}
    )
    assert ("chat_template.sandbox_escape", "risk") in out["tokenizer_config.json"]


def test_named_template_list_in_tokenizer_config_is_analyzed(tmp_path: Path) -> None:
    out = _scan(
        tmp_path,
        {
            "tokenizer_config.json": json.dumps(
                {"chat_template": [{"name": "default", "template": "{{ x.__globals__ }}"}]}
            )
        },
    )
    assert ("chat_template.sandbox_escape", "risk") in out["tokenizer_config.json"]


def test_chat_template_jinja_file_is_analyzed(tmp_path: Path) -> None:
    out = _scan(tmp_path, {"chat_template.jinja": "{{ ().__class__.__bases__ }}"})
    assert ("chat_template.sandbox_escape", "risk") in out["chat_template.jinja"]


def test_chat_template_json_file_is_analyzed(tmp_path: Path) -> None:
    out = _scan(
        tmp_path, {"chat_template.json": json.dumps({"chat_template": "{{ x|attr('__init__') }}"})}
    )
    assert ("chat_template.sandbox_escape", "risk") in out["chat_template.json"]


def test_real_template_file_has_no_finding(tmp_path: Path) -> None:
    out = _scan(tmp_path, {"tokenizer_config.json": json.dumps({"chat_template": _CHATML})})
    assert out["tokenizer_config.json"] == []

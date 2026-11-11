"""Model metadata extraction (step 13, ADR 0015).

config.json and the README front matter are author-controlled data: read as
JSON / YAML (never code), bounded, and a parse error just means the metadata is
absent — never a crash or a finding.
"""

import json
from pathlib import Path

import pytest

from nitrogen.inventory import inventory
from nitrogen.metadata import ModelMetadata, extract_metadata


def _meta(tmp_path: Path, files: dict[str, str]) -> ModelMetadata:
    for name, content in files.items():
        (tmp_path / name).write_text(content, encoding="utf-8")
    return extract_metadata(inventory(tmp_path), tmp_path)


def test_config_metadata(tmp_path: Path) -> None:
    config = {
        "model_type": "gpt2",
        "architectures": ["GPT2LMHeadModel"],
        "torch_dtype": "float16",
        "quantization_config": {"quant_method": "gptq", "bits": 4},
    }
    m = _meta(tmp_path, {"config.json": json.dumps(config)})
    assert m.model_type == "gpt2"
    assert m.architectures == ["GPT2LMHeadModel"]
    assert m.dtype == "float16"
    assert m.quantization == "gptq"


def test_readme_front_matter(tmp_path: Path) -> None:
    readme = (
        "---\n"
        "license: apache-2.0\n"
        "base_model: meta-llama/Llama-3\n"
        "datasets:\n  - squad\n  - glue\n"
        "---\n# A model\n"
    )
    m = _meta(tmp_path, {"README.md": readme})
    assert m.license == "apache-2.0"
    assert m.base_models == ["meta-llama/Llama-3"]
    assert m.datasets == ["squad", "glue"]


def test_base_model_can_be_a_list(tmp_path: Path) -> None:
    m = _meta(tmp_path, {"README.md": "---\nbase_model:\n  - a/b\n  - c/d\n---\n"})
    assert m.base_models == ["a/b", "c/d"]


def test_malformed_config_and_readme_do_not_crash(tmp_path: Path) -> None:
    m = _meta(tmp_path, {"config.json": "{ not json", "README.md": "---\nnot: [valid yaml\n---\n"})
    assert m.model_type is None
    assert m.license is None


def test_readme_without_front_matter_has_no_metadata(tmp_path: Path) -> None:
    m = _meta(tmp_path, {"README.md": "# Just a title\nno front matter here"})
    assert m.license is None and m.base_models == []


def test_no_metadata_files(tmp_path: Path) -> None:
    assert _meta(tmp_path, {"weights.txt": "data"}) == ModelMetadata()


def test_non_string_scalars_are_ignored(tmp_path: Path) -> None:
    # A crafted README with wrong types must not leak non-strings into the AI-BOM.
    m = _meta(tmp_path, {"README.md": "---\nlicense:\n  nested: 1\ndatasets: 5\n---\n"})
    assert m.license is None
    assert m.datasets == []


@pytest.mark.parametrize("value", ["mit", "MIT", "other"])
def test_license_is_kept_verbatim(tmp_path: Path, value: str) -> None:
    m = _meta(tmp_path, {"README.md": f"---\nlicense: {value}\n---\n"})
    assert m.license == value


def test_yaml_and_json_bombs_do_not_crash(tmp_path: Path) -> None:
    # Reviewer I1: a deeply nested README/config raises RecursionError, which the
    # ADR promises to turn into absent metadata, never an exit-2 crash.
    files = {"README.md": "---\n" + "[" * 2000 + "\n---\n", "config.json": "[" * 100_000}
    assert _meta(tmp_path, files) == ModelMetadata()


def test_subdirectory_files_do_not_shadow_the_root(tmp_path: Path) -> None:
    # Reviewer I2: real loaders read root config.json / README.md only.
    (tmp_path / "README.md").write_text("---\nlicense: gpl-3.0\n---\n", encoding="utf-8")
    sub = tmp_path / "third_party"
    sub.mkdir()
    (sub / "README.md").write_text("---\nlicense: mit\n---\n", encoding="utf-8")
    (sub / "config.json").write_text('{"model_type": "evil"}', encoding="utf-8")
    m = extract_metadata(inventory(tmp_path), tmp_path)
    assert m.license == "gpl-3.0"  # the root wins, not the sub-dir
    assert m.model_type is None  # there is no root config.json


def test_dataset_list_is_capped(tmp_path: Path) -> None:
    # Reviewer I3: a padded model card must not amplify the signed AI-BOM.
    readme = "---\ndatasets:\n" + "".join(f"  - d{i}\n" for i in range(500)) + "---\n"
    m = _meta(tmp_path, {"README.md": readme})
    assert len(m.datasets) == 32


def test_yaml_python_tags_execute_nothing(tmp_path: Path) -> None:
    # safe_load rejects !!python tags: metadata absent, nothing runs.
    readme = "---\nlicense: !!python/object/apply:os.system ['echo nitrogen']\n---\n"
    assert _meta(tmp_path, {"README.md": readme}).license is None


def test_leading_bom_is_tolerated(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("﻿---\nlicense: mit\n---\n", encoding="utf-8")
    assert extract_metadata(inventory(tmp_path), tmp_path).license == "mit"

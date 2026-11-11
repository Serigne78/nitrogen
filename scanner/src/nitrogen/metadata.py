"""Model metadata extraction (step 13, ADR 0015).

Reads author-controlled data — `config.json` (JSON) and the `README.md` model
card front matter (YAML) — never code. Values are bounded and recorded, escaped,
in the AI-BOM. A parse error just means the metadata is absent: never a crash,
never a finding (the license policy aside, handled in policy.py).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import yaml

from nitrogen.inventory import FileEntry

_MAX_BYTES = 1024 * 1024  # a config / model card is a few KB; bound the read
_MAX_STR = 200  # a single metadata string never needs more
_MAX_LIST = 32  # and a list never needs more: bound the AI-BOM against a padded card


@dataclass(frozen=True)
class ModelMetadata:
    model_type: str | None = None
    architectures: list[str] = field(default_factory=list)
    dtype: str | None = None
    quantization: str | None = None
    license: str | None = None
    base_models: list[str] = field(default_factory=list)
    datasets: list[str] = field(default_factory=list)


def extract_metadata(entries: list[FileEntry], root: Path) -> ModelMetadata:
    base = root if root.is_dir() else root.parent
    # Root-level files only: transformers / the model card read config.json and
    # README.md at the repo root, so a file in a sub-directory must not shadow it.
    by_name = {e.path.name.casefold(): e for e in entries if len(e.path.parts) == 1}

    config = _read_json(base, by_name.get("config.json"))
    readme = _front_matter(_read_text(base, by_name.get("readme.md")))

    quant = config.get("quantization_config")
    return ModelMetadata(
        model_type=_as_str(config.get("model_type")),
        architectures=_as_str_list(config.get("architectures")),
        dtype=_as_str(config.get("torch_dtype")),
        quantization=_quantization(quant),
        license=_as_str(readme.get("license")),
        base_models=_as_str_list(readme.get("base_model")),
        datasets=_as_str_list(readme.get("datasets")),
    )


def _quantization(value: object) -> str | None:
    if not isinstance(value, dict):
        return None
    return _as_str(value.get("quant_method")) or "present"


def _as_str(value: object) -> str | None:
    return value[:_MAX_STR] if isinstance(value, str) and value else None


def _as_str_list(value: object) -> list[str]:
    if isinstance(value, str):
        return [value[:_MAX_STR]] if value else []
    if isinstance(value, list):
        return [v[:_MAX_STR] for v in value if isinstance(v, str) and v][:_MAX_LIST]
    return []


def _front_matter(text: str | None) -> dict[str, object]:
    """The YAML front matter of a model card, between leading and closing `---`."""
    if not text:
        return {}
    lines = text.lstrip("﻿").splitlines()  # tolerate a leading UTF-8 BOM
    if not lines or lines[0].strip() != "---":
        return {}
    end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
    if end is None:
        return {}
    try:
        # SafeLoader: data, never code. RecursionError (deeply nested / a YAML
        # bomb) is caught too — the ADR promises "absent metadata, never a crash".
        data = yaml.safe_load("\n".join(lines[1:end]))
    except (yaml.YAMLError, RecursionError):
        return {}
    return data if isinstance(data, dict) else {}


def _read_json(base: Path, entry: FileEntry | None) -> dict[str, object]:
    raw = _read_bytes(base, entry)
    if raw is None:
        return {}
    try:
        data = json.loads(raw)
    except (ValueError, RecursionError):  # malformed, or a deeply nested bomb
        return {}
    return data if isinstance(data, dict) else {}


def _read_text(base: Path, entry: FileEntry | None) -> str | None:
    raw = _read_bytes(base, entry)
    return None if raw is None else raw.decode("utf-8", "replace")


def _read_bytes(base: Path, entry: FileEntry | None) -> bytes | None:
    if entry is None or entry.size > _MAX_BYTES:
        return None
    try:
        with (base / _rel(entry.path)).open("rb") as fh:
            return fh.read(_MAX_BYTES + 1)
    except OSError:
        return None


def _rel(path: PurePosixPath) -> Path:
    return Path(*path.parts)

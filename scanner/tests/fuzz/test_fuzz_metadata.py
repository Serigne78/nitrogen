"""The README front-matter / config parser on random input: never raises, never
hangs; unreadable metadata just yields an empty ModelMetadata."""

import tempfile
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from nitrogen.inventory import inventory
from nitrogen.metadata import ModelMetadata, extract_metadata


@settings(max_examples=200)
@given(st.text(max_size=2000))
def test_random_metadata_files_never_crash(source: str) -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "README.md").write_text(source, encoding="utf-8")
        (root / "config.json").write_text(source, encoding="utf-8")
        result = extract_metadata(inventory(root), root)
    assert isinstance(result, ModelMetadata)

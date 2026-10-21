"""The release version is 0.1.0 and consistent between the package and
pyproject.toml (step 7)."""

import tomllib
from pathlib import Path

import nitrogen

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def test_package_version_is_the_release() -> None:
    assert nitrogen.__version__ == "0.1.0"


def test_pyproject_matches_the_package() -> None:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    assert data["project"]["version"] == nitrogen.__version__

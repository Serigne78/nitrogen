"""Minimal YAML policy (step 6, ADR 0005): allow_formats only. Read with a
SafeLoader that also refuses duplicate keys; anything unexpected is an error,
never silently ignored."""

import hashlib
from pathlib import Path, PurePosixPath

import pytest

from nitrogen.findings import Finding
from nitrogen.formats import Format
from nitrogen.inventory import FileEntry
from nitrogen.policy import MAX_POLICY_BYTES, Policy, PolicyError, apply_policy, load_policy


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "policy.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def _entry(name: str, *formats: Format) -> FileEntry:
    return FileEntry(PurePosixPath(name), 1, "a" * 64, frozenset(formats))


def test_allow_formats(tmp_path: Path) -> None:
    path = _write(tmp_path, "allow_formats: [safetensors, unknown]\n")
    policy = load_policy(path)
    assert policy.allow_formats == frozenset({"safetensors", "unknown"})
    assert policy.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("text", ["", "# only a comment\n", "{}\n"])
def test_policy_without_allow_formats_restricts_nothing(tmp_path: Path, text: str) -> None:
    policy = load_policy(_write(tmp_path, text))
    assert policy.allow_formats is None
    entries = [_entry("a.pkl", Format.PICKLE), _entry("c.json")]
    assert apply_policy(policy, entries) == entries


@pytest.mark.parametrize(
    "text",
    [
        "allow_format: [pickle]\n",  # typo in the key: must not be silently ignored
        "deny_formats: [pickle]\n",  # replaced by allow_formats (ADR 0005 revision)
        "allow_formats: [pikle]\n",  # unknown format
        "allow_formats: pickle\n",  # not a list
        "allow_formats: [1]\n",  # not a string
        # Falsy values of the wrong type must not read as "nothing to check".
        "allow_formats:\n",
        "allow_formats: null\n",
        "allow_formats: {}\n",
        "allow_formats: false\n",
        "allow_formats: no\n",
        "allow_formats: 0\n",
        'allow_formats: ""\n',
        # A bad merge must not quietly widen the policy.
        "allow_formats: [safetensors]\nallow_formats: [safetensors, pickle]\n",
        "- pickle\n",  # not a mapping
        "allow_formats: [pickle\n",  # invalid YAML
        "allow_formats: " + "[" * 5000 + "]" * 5000 + "\n",  # deep nesting
        "!!python/object/apply:os.system ['echo nitrogen']\n",  # tags refused
    ],
    ids=lambda t: t[:30],
)
def test_invalid_policy_is_an_error(tmp_path: Path, text: str) -> None:
    with pytest.raises(PolicyError):
        load_policy(_write(tmp_path, text))


def test_oversized_policy_is_an_error_without_reading_it_all(tmp_path: Path) -> None:
    path = tmp_path / "policy.yaml"
    path.write_bytes(b"#" * (MAX_POLICY_BYTES * 4))
    with pytest.raises(PolicyError):
        load_policy(path)


def test_apply_flags_every_format_not_allowed() -> None:
    policy = Policy(allow_formats=frozenset({"safetensors", "unknown"}))
    out = apply_policy(
        policy,
        [
            _entry("a.pkl", Format.PICKLE),
            _entry("b.safetensors", Format.SAFETENSORS),
            _entry("c.json"),
            _entry("d.pt", Format.PYTORCH_ZIP),
            _entry("e.poly", Format.PICKLE, Format.SAFETENSORS),  # every format must be allowed
        ],
    )
    flagged = {e.path.name: [f.message for f in e.findings] for e in out}
    assert flagged == {
        "a.pkl": ["format pickle is not allowed by policy"],
        "b.safetensors": [],
        "c.json": [],
        "d.pt": ["format pytorch-zip is not allowed by policy"],
        "e.poly": ["format pickle is not allowed by policy"],
    }
    assert {f.rule for e in out for f in e.findings} == {"policy.format-not-allowed"}


def test_unknown_must_be_allowed_by_name() -> None:
    [out] = apply_policy(Policy(allow_formats=frozenset({"safetensors"})), [_entry("x.onnx")])
    assert [f.message for f in out.findings] == ["format unknown is not allowed by policy"]


def test_apply_keeps_existing_findings() -> None:
    entry = FileEntry(
        PurePosixPath("a.pkl"),
        1,
        "a" * 64,
        frozenset({Format.PICKLE}),
        (Finding("pickle.import", "os.system"),),
    )
    [out] = apply_policy(Policy(allow_formats=frozenset({"safetensors"})), [entry])
    assert [f.rule for f in out.findings] == ["pickle.import", "policy.format-not-allowed"]


def test_unanalyzed_model_formats_are_refused_unless_allowed() -> None:
    policy = Policy(allow_formats=frozenset({"safetensors", "unknown"}))
    [h5, cfg] = apply_policy(policy, [_entry("m.h5", Format.HDF5), _entry("config.json")])
    assert [f.message for f in h5.findings] == ["format hdf5 is not allowed by policy"]
    assert cfg.findings == ()


def test_unhashable_key_is_a_policy_error(tmp_path: Path) -> None:
    with pytest.raises(PolicyError):
        load_policy(_write(tmp_path, "? [a]\n: 1\n"))

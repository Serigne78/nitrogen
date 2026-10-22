"""Archive scanning robustness (surfaced by the neutral benchmark, ADR 0007).

An archive member we cannot read — encrypted, corrupt — must be reported as
`pickle.unanalyzable` (a RISK), never raise and turn the scan into an error:
an unreadable member is exactly what an attacker would use to hide a payload."""

import io
import zipfile
from pathlib import Path

from nitrogen.archives import check_zip

VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "picklescan"


def _rules(path: Path) -> list[str]:
    with path.open("rb") as fh:
        return [f.rule for f in check_zip(fh, path.stat().st_size)]


def test_encrypted_zip_member_is_unanalyzable_not_an_error() -> None:
    # picklescan's malicious1_0x*.zip: a password-protected member.
    encrypted = sorted(VENDOR.glob("malicious1_0x*.zip"))
    assert encrypted, "vendored encrypted-zip fixtures missing"
    for path in encrypted:
        assert _rules(path) == ["pickle.unanalyzable"], path.name


def test_unreadable_member_does_not_hide_a_readable_one(tmp_path: Path) -> None:
    # A readable malicious member alongside an unreadable one is still caught.
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("evil.pkl", b"cos\nsystem\n(S'x'\ntR.")
    data = bytearray(buf.getvalue())
    # Corrupt a second appended member is overkill; the readable one must flag.
    rules = check_zip(io.BytesIO(bytes(data)), len(data))
    assert any(f.rule == "pickle.import" for f in rules)

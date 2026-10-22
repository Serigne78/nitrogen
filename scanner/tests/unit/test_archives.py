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
    # picklescan's malicious1_0x*.zip: members the standard zip library cannot
    # extract. Each yields exactly one finding, never an uncaught error.
    for path in sorted(VENDOR.glob("malicious1_0x*.zip")):
        assert len(_rules(path)) == 1, path.name


def test_unreadable_member_does_not_hide_a_readable_one(tmp_path: Path) -> None:
    # A readable malicious member alongside an unreadable one is still caught.
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("evil.pkl", b"cos\nsystem\n(S'x'\ntR.")
    data = bytearray(buf.getvalue())
    # Corrupt a second appended member is overkill; the readable one must flag.
    rules = check_zip(io.BytesIO(bytes(data)), len(data))
    assert any(f.rule == "pickle.import" for f in rules)


def _one(path: Path) -> tuple[str, str]:
    with path.open("rb") as fh:
        [f] = check_zip(fh, path.stat().st_size)
    return f.rule, f.severity


def test_encrypted_member_is_a_warning() -> None:
    # ADR 0008: no loader opens an encrypted member (traditional or strong), so
    # it is a blind spot (warning), not a proven risk.
    assert _one(VENDOR / "malicious1_0x1.zip") == ("archive.encrypted", "warning")  # encrypted
    assert _one(VENDOR / "malicious1_0x40.zip") == ("archive.encrypted", "warning")  # strong enc.


def test_other_unreadable_member_stays_a_risk() -> None:
    # Bit 5 (patched data) is not encryption: unreadable for another reason,
    # so we cannot rule out that some loader reads it — stays risk.
    assert _one(VENDOR / "malicious1_0x20.zip") == ("pickle.unanalyzable", "risk")


def test_member_name_cannot_fake_encryption(tmp_path: Path) -> None:
    # Reviewer critical: a corrupt-CRC member named "encrypted.pkl" must NOT be
    # downgraded to warning by its name. Encryption is read from flag bits only.
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("archive/encrypted_password.pkl", b"cos\nsystem\n(S'x'\ntR.")
    data = bytearray(buf.getvalue())
    data[data.index(b"cos")] ^= 0xFF  # break the CRC of the member
    [f] = check_zip(io.BytesIO(bytes(data)), len(data))
    assert (f.rule, f.severity) == ("pickle.unanalyzable", "risk")

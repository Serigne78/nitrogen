"""Archive scanning robustness (surfaced by the neutral benchmark, ADR 0007).

An archive member we cannot read — encrypted, corrupt — must be reported as
`pickle.unanalyzable` (a RISK), never raise and turn the scan into an error:
an unreadable member is exactly what an attacker would use to hide a payload."""

import io
import zipfile
from pathlib import Path

import pytest

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


def test_zero_tensor_member_is_not_a_nested_archive() -> None:
    # ADR 0009: a zero-initialized tensor storage (member `data/<key>`) is all
    # zeros, which tarfile.open accepts as an empty tar. It is NOT a nested
    # archive and must not be flagged (the false positive measured in step 9).
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("archive/data.pkl", b"\x80\x02}q\x00.")  # empty dict, allowlisted
        zf.writestr("archive/data/0", b"\x00" * 1536)  # a zero tensor storage
    data = buf.getvalue()
    assert check_zip(io.BytesIO(data), len(data)) == []


def test_pytorch_storage_members_are_not_scanned() -> None:
    # ADR 0010: in a PyTorch zip (has data.pkl), the archive/data/<key> members
    # are raw tensor storage. Their bytes often resemble a pickle or an archive
    # (the 64% false-positive rate measured in step 9); torch.load reads them as
    # storage, never unpickles them, so they must not be analyzed.
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("archive/data.pkl", b"\x80\x02}q\x00.")  # empty dict, safe
        zf.writestr("archive/data/0", b"\x00" * 600)  # zeros: read as an empty tar
        zf.writestr(
            "archive/data/1", b"\x80\x02\x8d\xff\xff\xff\xff\x00\x00\x00\x00"
        )  # pickle-like
        zf.writestr("archive/data/2", b"PK\x03\x04" + b"\x00" * 60)  # zip-like
    data = buf.getvalue()
    assert check_zip(io.BytesIO(data), len(data)) == []


def test_malicious_data_pkl_is_still_scanned_with_storage() -> None:
    # Skipping storage must not skip data.pkl: a dangerous import there is caught.
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("archive/data.pkl", b"cos\nsystem\n(S'x'\ntR.")
        zf.writestr("archive/data/0", b"\x00" * 600)
    data = buf.getvalue()
    [f] = check_zip(io.BytesIO(data), len(data))
    assert (f.rule, f.message) == ("pickle.import", "archive/data.pkl: os.system")


def test_non_storage_member_pickle_is_still_scanned() -> None:
    # A member that is NOT tensor storage (not data/<int>) is analyzed as before.
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("archive/data.pkl", b"\x80\x02}q\x00.")
        zf.writestr("archive/extra.pkl", b"cos\nsystem\n(S'x'\ntR.")
    data = buf.getvalue()
    assert any(f.rule == "pickle.import" for f in check_zip(io.BytesIO(data), len(data)))


def test_storage_skipping_needs_a_data_pkl() -> None:
    # Not a PyTorch zip (no data.pkl): a generic zip's data/0 member is still
    # scanned, so a pickle hidden there in a plain zip is not missed.
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("data/0", b"cos\nsystem\n(S'x'\ntR.")
    data = buf.getvalue()
    assert any(f.rule == "pickle.import" for f in check_zip(io.BytesIO(data), len(data)))


def test_real_pickle_in_storage_member_is_an_accepted_blind_spot() -> None:
    # ADR 0010, stated plainly: a genuine os.system pickle inside a storage
    # member (archive/data/0, under a real data.pkl) is NOT flagged. This is the
    # accepted blind spot — torch.load reads data/<key> as raw storage by type
    # and size, it never unpickles it, so there is no execution path. If this
    # ever starts flagging, the trade-off has changed and must be revisited.
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("archive/data.pkl", b"\x80\x02}q\x00.")
        zf.writestr("archive/data/0", b"cos\nsystem\n(S'x'\ntR.")  # a real payload
    data = buf.getvalue()
    assert check_zip(io.BytesIO(data), len(data)) == []


def test_storage_skip_is_anchored_to_the_pickle_root() -> None:
    # Reviewer I1: the skip must follow torch's layout (a single root holding
    # data.pkl), not "data/<n> at any depth under any bait data.pkl". A payload
    # in another root, or nested deeper, is still analyzed.
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("archive/data.pkl", b"\x80\x02}q\x00.")  # the real root is "archive"
        zf.writestr("weights/data/0", b"cos\nsystem\n(S'x'\ntR.")  # a different root
        zf.writestr("a/b/data/1", b"cos\nsystem\n(S'x'\ntR.")  # deeper than one root
    data = buf.getvalue()
    assert [f.rule for f in check_zip(io.BytesIO(data), len(data))] == [
        "pickle.import",
        "pickle.import",
    ]


@pytest.mark.parametrize("name", ["archive/DATA/0", "archive/data/5x", "mydata/0"])
def test_storage_name_variants_are_still_scanned(name: str) -> None:
    # Only an exact <root>/data/<digits> under a matching data.pkl is storage;
    # a near-miss name must stay analyzed, so the regex cannot be a bypass.
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("archive/data.pkl", b"\x80\x02}q\x00.")
        zf.writestr(name, b"cos\nsystem\n(S'x'\ntR.")
    data = buf.getvalue()
    assert any(f.rule == "pickle.import" for f in check_zip(io.BytesIO(data), len(data)))


def test_genuine_nested_archive_member_stays_a_risk() -> None:
    # ADR 0009 does not loosen ADR 0004: a member that really opens as an archive
    # (>=1 entry) is still flagged, since no standard loader recurses into it.
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as zf:
        zf.writestr("payload.pkl", b"\x80\x02K\x00.")
    outer = io.BytesIO()
    with zipfile.ZipFile(outer, "w") as zf:
        zf.writestr("archive/nested.bin", inner.getvalue())  # a real zip inside
    data = outer.getvalue()
    [f] = check_zip(io.BytesIO(data), len(data))
    assert (f.rule, f.severity) == ("pickle.unanalyzable", "risk")
    assert "nested archive" in f.message

from pathlib import Path

import pytest

from scripts import fetch_freemhd_assets as fetch


pytestmark = pytest.mark.unit


def test_is_valid_zip_detects_bad_partial_file(tmp_path: Path):
    bad = tmp_path / "StartingFiles.zip"
    bad.write_bytes(b"PK\x03\x04partial")
    assert fetch._is_valid_zip(bad) is False


def test_is_valid_zip_accepts_real_zip(tmp_path: Path):
    import zipfile

    good = tmp_path / "good.zip"
    with zipfile.ZipFile(good, "w") as archive:
        archive.writestr("a.txt", "hello")
    assert fetch._is_valid_zip(good) is True


def test_download_resumes_invalid_existing_zip_before_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    target = tmp_path / "StartingFiles.zip"
    target.write_bytes(b"PK\x03\x04partial")
    calls: list[list[str]] = []

    def fake_run(command: list[str], check: bool) -> None:
        calls.append(command)
        import zipfile

        with zipfile.ZipFile(target, "w") as archive:
            archive.writestr("a.txt", "hello")

    monkeypatch.setattr(fetch.subprocess, "run", fake_run)
    fetch._download("https://example.invalid/file.zip", target, validate_zip=True)
    assert calls
    assert "-C" in calls[0]
    assert fetch._is_valid_zip(target) is True

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


def test_download_skips_existing_valid_archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import zipfile

    target = tmp_path / "StartingFiles.zip"
    with zipfile.ZipFile(target, "w") as archive:
        archive.writestr("a.txt", "hello")

    monkeypatch.setattr(fetch.subprocess, "run", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected download")))
    fetch._download("https://example.invalid/file.zip", target, validate_zip=True)


def test_fetch_main_clones_and_downloads_requested_archives(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    calls: list[tuple[str, Path, bool]] = []
    commands: list[list[str]] = []

    monkeypatch.setattr(fetch, "_download", lambda url, path, validate_zip=False: calls.append((url, path, validate_zip)))
    monkeypatch.setattr(fetch.subprocess, "run", lambda command, check=True: commands.append(command))
    monkeypatch.setattr(fetch.argparse.ArgumentParser, "parse_args", lambda self: type("Args", (), {"dest": tmp_path, "include_starting_files": True})())

    exit_code = fetch.main()

    assert exit_code == 0
    assert commands[0][:2] == ["git", "clone"]
    assert len(calls) == 2
    assert str(tmp_path.resolve()) in capsys.readouterr().out

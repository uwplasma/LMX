from __future__ import annotations

import json
import struct
import zipfile
from pathlib import Path
import zlib

import pytest

from scripts import inspect_starting_files_archive as archive_tools


pytestmark = pytest.mark.unit


def _write_partial_zip(path: Path, name: str, payload: bytes, compress_type: int = 0) -> None:
    if compress_type == 0:
        compressed = payload
    elif compress_type == 8:
        compressed = zlib.compress(payload)[2:-4]
    else:
        raise ValueError("unsupported test compression")
    header = struct.pack(
        "<4sHHHHHIIIHH",
        archive_tools.LOCAL_FILE_HEADER,
        20,
        0,
        compress_type,
        0,
        0,
        0,
        len(compressed),
        len(payload),
        len(name.encode()),
        0,
    )
    path.write_bytes(header + name.encode() + compressed)


def test_inspect_archive_reports_missing(tmp_path: Path):
    payload = archive_tools.inspect_archive(tmp_path / "missing.zip", "hunt")

    assert payload["status"] == "missing"
    assert payload["entries"] == []


def test_read_partial_entries_stops_on_data_descriptor_flag(tmp_path: Path):
    archive_path = tmp_path / "descriptor.zip"
    name = "Hunt/case.dat"
    header = struct.pack(
        "<4sHHHHHIIIHH",
        archive_tools.LOCAL_FILE_HEADER,
        20,
        0x08,
        0,
        0,
        0,
        0,
        3,
        3,
        len(name.encode()),
        0,
    )
    archive_path.write_bytes(header + name.encode() + b"abc")

    assert archive_tools._read_partial_entries(archive_path) == []


def test_inspect_archive_filters_valid_zip_entries(tmp_path: Path):
    archive_path = tmp_path / "cases.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("Shercliff/caseA.txt", "a")
        archive.writestr("Hunt/caseB.txt", "b")

    payload = archive_tools.inspect_archive(archive_path, "hunt")

    assert payload["status"] == "ok"
    assert payload["entry_count"] == 1
    assert payload["entries"] == ["Hunt/caseB.txt"]


def test_extract_matching_extracts_from_valid_zip(tmp_path: Path):
    archive_path = tmp_path / "cases.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("Hunt/caseB.txt", "b")
        archive.writestr("Hartmann/caseC.txt", "c")

    output = tmp_path / "extract"
    payload = archive_tools.extract_matching(archive_path, "hunt", output)

    assert payload["status"] == "ok"
    assert payload["extracted_count"] == 1
    assert (output / "Hunt" / "caseB.txt").read_text() == "b"


def test_read_partial_entries_reads_local_file_headers(tmp_path: Path):
    archive_path = tmp_path / "partial.zip"
    _write_partial_zip(archive_path, "Hunt/case.dat", b"payload")

    entries = archive_tools._read_partial_entries(archive_path)

    assert len(entries) == 1
    assert entries[0].name == "Hunt/case.dat"
    assert entries[0].uncompressed_size == 7


def test_extract_partial_entry_handles_stored_and_deflated_payloads(tmp_path: Path):
    stored_zip = tmp_path / "stored.zip"
    deflated_zip = tmp_path / "deflated.zip"
    _write_partial_zip(stored_zip, "Hunt/stored.dat", b"abc", compress_type=0)
    _write_partial_zip(deflated_zip, "Hunt/deflated.dat", b"xyz", compress_type=8)

    stored_entry = archive_tools._read_partial_entries(stored_zip)[0]
    deflated_entry = archive_tools._read_partial_entries(deflated_zip)[0]
    output = tmp_path / "extract"

    stored_target = archive_tools._extract_partial_entry(stored_zip, stored_entry, output)
    deflated_target = archive_tools._extract_partial_entry(deflated_zip, deflated_entry, output)

    assert stored_target.read_text() == "abc"
    assert deflated_target.read_text() == "xyz"


def test_extract_partial_entry_rejects_unknown_compression(tmp_path: Path):
    archive_path = tmp_path / "partial.zip"
    _write_partial_zip(archive_path, "Hunt/case.dat", b"payload")
    entry = archive_tools._read_partial_entries(archive_path)[0]
    bad_entry = archive_tools.PartialZipEntry(
        name=entry.name,
        compress_type=99,
        compressed_size=entry.compressed_size,
        uncompressed_size=entry.uncompressed_size,
        data_offset=entry.data_offset,
    )

    with pytest.raises(ValueError, match="Unsupported compression method"):
        archive_tools._extract_partial_entry(archive_path, bad_entry, tmp_path / "out")


def test_extract_matching_uses_partial_zip_fallback(tmp_path: Path):
    archive_path = tmp_path / "partial.zip"
    _write_partial_zip(archive_path, "Hunt/case.dat", b"payload")

    payload = archive_tools.extract_matching(archive_path, "hunt", tmp_path / "out")

    assert payload["status"] == "partial-ok"
    assert payload["extracted_count"] == 1
    assert (tmp_path / "out" / "Hunt" / "case.dat").read_text() == "payload"


def test_main_writes_json_for_inspection(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    archive_path = tmp_path / "cases.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("Hunt/caseB.txt", "b")

    monkeypatch.setattr(
        archive_tools.argparse.ArgumentParser,
        "parse_args",
        lambda self: type(
            "Args",
            (),
            {
                "archive": archive_path,
                "pattern": "hunt",
                "extract": False,
                "output_dir": tmp_path / "unused",
            },
        )(),
    )

    exit_code = archive_tools.main()

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["entry_count"] == 1


def test_main_writes_json_for_extract(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    archive_path = tmp_path / "cases.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("Hunt/caseB.txt", "b")

    monkeypatch.setattr(
        archive_tools.argparse.ArgumentParser,
        "parse_args",
        lambda self: type(
            "Args",
            (),
            {
                "archive": archive_path,
                "pattern": "hunt",
                "extract": True,
                "output_dir": tmp_path / "extract",
            },
        )(),
    )

    exit_code = archive_tools.main()

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["extracted_count"] == 1
    assert (tmp_path / "extract" / "Hunt" / "caseB.txt").read_text() == "b"

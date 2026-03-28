import struct
from pathlib import Path

import pytest

from scripts.inspect_starting_files_archive import extract_matching, inspect_archive


pytestmark = pytest.mark.unit


def test_inspect_archive_handles_partial_zip(tmp_path: Path):
    archive = tmp_path / "StartingFiles.zip"

    def local_header(name: bytes, payload: bytes) -> bytes:
        return (
            b"PK\x03\x04"
            + struct.pack("<HHHHHIIIHH", 20, 0, 0, 0, 0, 0, len(payload), len(payload), len(name), 0)
            + name
            + payload
        )

    archive.write_bytes(local_header(b"StartingFiles/", b"") + local_header(b"StartingFiles/a.txt", b"data"))
    payload = inspect_archive(archive, pattern="StartingFiles")
    assert payload["status"] == "partial-ok"
    assert payload["entry_count"] == 2


def test_extract_matching_recovers_partial_stored_entries(tmp_path: Path):
    archive = tmp_path / "StartingFiles.zip"

    def local_header(name: bytes, payload: bytes) -> bytes:
        return (
            b"PK\x03\x04"
            + struct.pack("<HHHHHIIIHH", 20, 0, 0, 0, 0, 0, len(payload), len(payload), len(name), 0)
            + name
            + payload
        )

    archive.write_bytes(local_header(b"StartingFiles/", b"") + local_header(b"StartingFiles/a.txt", b"data"))
    out = tmp_path / "out"
    payload = extract_matching(archive, "a.txt", out)
    assert payload["status"] == "partial-ok"
    assert (out / "StartingFiles" / "a.txt").read_text() == "data"

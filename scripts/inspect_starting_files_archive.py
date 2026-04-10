#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import struct
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path


LOCAL_FILE_HEADER = b"PK\x03\x04"


@dataclass(frozen=True)
class PartialZipEntry:
    name: str
    compress_type: int
    compressed_size: int
    uncompressed_size: int
    data_offset: int


def _read_partial_entries(path: Path) -> list[PartialZipEntry]:
    entries: list[PartialZipEntry] = []
    filesize = path.stat().st_size
    with path.open("rb") as handle:
        while handle.tell() + 30 <= filesize:
            signature = handle.read(4)
            if len(signature) < 4:
                break
            if signature != LOCAL_FILE_HEADER:
                chunk = signature + handle.read(4096)
                index = chunk.find(LOCAL_FILE_HEADER)
                if index == -1:
                    break
                handle.seek(-(len(chunk) - index), 1)
                continue
            header = handle.read(26)
            if len(header) < 26:
                break
            _, flag, compress_type, _, _, _, compressed_size, uncompressed_size, name_len, extra_len = struct.unpack(
                "<HHHHHIIIHH",
                header,
            )
            name = handle.read(name_len)
            extra = handle.read(extra_len)
            if len(name) != name_len or len(extra) != extra_len:
                break
            data_offset = handle.tell()
            if (flag & 0x08) != 0:
                break
            if data_offset + compressed_size > filesize:
                break
            entries.append(
                PartialZipEntry(
                    name=name.decode("utf-8", "ignore"),
                    compress_type=compress_type,
                    compressed_size=compressed_size,
                    uncompressed_size=uncompressed_size,
                    data_offset=data_offset,
                )
            )
            handle.seek(compressed_size, 1)
    return entries


def inspect_archive(archive_path: str | Path, pattern: str = "") -> dict[str, object]:
    path = Path(archive_path)
    payload: dict[str, object] = {
        "archive_path": str(path),
        "exists": path.exists(),
        "pattern": pattern,
    }
    if not path.exists():
        payload["status"] = "missing"
        payload["entries"] = []
        return payload
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if pattern:
                names = [name for name in names if pattern.lower() in name.lower()]
            payload["status"] = "ok"
            payload["entry_count"] = len(names)
            payload["entries"] = names[:500]
            return payload
    except zipfile.BadZipFile:
        entries = _read_partial_entries(path)
        names = [entry.name for entry in entries]
        if pattern:
            names = [name for name in names if pattern.lower() in name.lower()]
        payload["status"] = "partial-ok"
        payload["entry_count"] = len(names)
        payload["entries"] = names[:500]
        return payload


def _extract_partial_entry(path: Path, entry: PartialZipEntry, output_root: Path) -> Path:
    destination = output_root / entry.name
    if entry.name.endswith("/"):
        destination.mkdir(parents=True, exist_ok=True)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with path.open("rb") as handle:
        handle.seek(entry.data_offset)
        data = handle.read(entry.compressed_size)
    if entry.compress_type == 0:
        payload = data
    elif entry.compress_type == 8:
        payload = zlib.decompress(data, -zlib.MAX_WBITS)
    else:
        raise ValueError(f"Unsupported compression method {entry.compress_type} for {entry.name}")
    destination.write_bytes(payload)
    return destination


def extract_matching(archive_path: str | Path, pattern: str, output_dir: str | Path) -> dict[str, object]:
    path = Path(archive_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(path) as archive:
            names = [name for name in archive.namelist() if pattern.lower() in name.lower()]
            for name in names:
                archive.extract(name, out)
            return {
                "archive_path": str(path),
                "pattern": pattern,
                "output_dir": str(out),
                "status": "ok",
                "extracted_count": len(names),
                "extracted_entries": names[:500],
            }
    except zipfile.BadZipFile:
        entries = [entry for entry in _read_partial_entries(path) if pattern.lower() in entry.name.lower()]
        extracted = [_extract_partial_entry(path, entry, out) for entry in entries]
        return {
            "archive_path": str(path),
            "pattern": pattern,
            "output_dir": str(out),
            "status": "partial-ok",
            "extracted_count": len(extracted),
            "extracted_entries": [str(item.relative_to(out)) for item in extracted[:500]],
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect or selectively extract a recovered benchmark StartingFiles archive.")
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "external" / "StartingFiles.zip",
    )
    parser.add_argument("--pattern", type=str, default="")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/starting_files_extract"))
    args = parser.parse_args()

    if args.extract:
        payload = extract_matching(args.archive, args.pattern, args.output_dir)
    else:
        payload = inspect_archive(args.archive, args.pattern)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

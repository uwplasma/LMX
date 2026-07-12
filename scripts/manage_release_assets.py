#!/usr/bin/env python3
"""Inventory, bundle, and verify generated LMX release assets."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import mimetypes
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from scripts.audit_architecture import ROOT, _release_asset_candidates
except ModuleNotFoundError:  # direct ``python scripts/...`` execution
    from audit_architecture import ROOT, _release_asset_candidates


MANIFEST_PATH = ROOT / "provenance" / "release-assets.json"
RELEASE_TAG = "lmx-research-assets-v1"
ARCHIVE_NAME = f"{RELEASE_TAG}.tar.gz"
DOWNLOAD_URL = (
    f"https://github.com/uwplasma/lmx/releases/download/{RELEASE_TAG}/{ARCHIVE_NAME}"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(root: Path = ROOT) -> dict[str, Any]:
    """Build the canonical source-asset inventory, grouping duplicate content."""

    grouped: dict[tuple[str, int], list[str]] = defaultdict(list)
    for item in _release_asset_candidates(root):
        grouped[(str(item["sha256"]), int(item["bytes"]))].append(str(item["path"]))
    assets = []
    for (digest, size), paths in sorted(grouped.items()):
        media_types = sorted(
            {
                mimetypes.guess_type(path)[0] or "application/octet-stream"
                for path in paths
            }
        )
        assets.append(
            {
                "bytes": size,
                "media_types": media_types,
                "paths": sorted(paths),
                "sha256": digest,
            }
        )
    return {
        "schema_version": 1,
        "generated_by": "scripts/manage_release_assets.py",
        "release": {
            "archive_name": ARCHIVE_NAME,
            "download_url": DOWNLOAD_URL,
            "repository": "uwplasma/lmx",
            "status": "planned",
            "tag": RELEASE_TAG,
        },
        "summary": {
            "logical_bytes": sum(
                asset["bytes"] * len(asset["paths"]) for asset in assets
            ),
            "logical_file_count": sum(len(asset["paths"]) for asset in assets),
            "unique_bytes": sum(asset["bytes"] for asset in assets),
            "unique_content_count": len(assets),
        },
        "assets": assets,
    }


def _canonical(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def write_manifest(path: Path = MANIFEST_PATH, root: Path = ROOT) -> dict[str, Any]:
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("release", {}).get("status") == "uploaded":
            raise ValueError("Uploaded release-asset manifests are immutable")
    payload = build_manifest(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_canonical(payload), encoding="utf-8")
    return payload


def check_manifest(path: Path = MANIFEST_PATH, root: Path = ROOT) -> dict[str, Any]:
    tracked = json.loads(path.read_text(encoding="utf-8"))
    release = tracked.get("release", {})
    if release.get("status") not in {"planned", "uploaded"}:
        raise ValueError("Release status must be 'planned' or 'uploaded'")
    if release.get("status") == "uploaded" and not release.get("archive_sha256"):
        raise ValueError("Uploaded release assets require archive_sha256")
    expected = {
        relative: (int(asset["bytes"]), str(asset["sha256"]))
        for asset in tracked.get("assets", [])
        for relative in asset["paths"]
    }
    current = {str(item["path"]): item for item in _release_asset_candidates(root)}
    unexpected = sorted(set(current) - set(expected))
    if unexpected:
        raise ValueError(f"Untracked generated release assets: {unexpected}")
    missing = []
    for relative, (size, digest) in expected.items():
        source = root / relative
        if not source.is_file():
            missing.append(relative)
            continue
        if source.stat().st_size != size or _sha256(source) != digest:
            raise ValueError(f"Release-asset source drift: {relative}")
    if missing and release.get("status") != "uploaded":
        raise ValueError(f"Planned release assets are missing: {missing}")
    return tracked


def build_archive(output: Path, root: Path = ROOT) -> str:
    """Create a deterministic archive containing every logical source path."""

    manifest = build_manifest(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for asset in manifest["assets"]:
                    for relative in asset["paths"]:
                        path = root / relative
                        info = archive.gettarinfo(str(path), arcname=relative)
                        info.mtime = 0
                        info.uid = 0
                        info.gid = 0
                        info.uname = ""
                        info.gname = ""
                        with path.open("rb") as stream:
                            archive.addfile(info, stream)
    verify_archive(output, manifest)
    return _sha256(output)


def verify_archive(path: Path, manifest: dict[str, Any] | None = None) -> None:
    """Verify archive membership, sizes, and hashes against the manifest."""

    expected = manifest or json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    expected_paths = {
        relative: (int(asset["bytes"]), str(asset["sha256"]))
        for asset in expected["assets"]
        for relative in asset["paths"]
    }
    with tarfile.open(path, mode="r:gz") as archive:
        members = {
            member.name: member for member in archive.getmembers() if member.isfile()
        }
        if set(members) != set(expected_paths):
            raise ValueError("Release archive membership differs from the manifest")
        for relative, (size, digest) in expected_paths.items():
            member = members[relative]
            if member.size != size:
                raise ValueError(f"Release archive size mismatch for {relative}")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"Cannot read release archive member {relative}")
            actual = hashlib.sha256(stream.read()).hexdigest()
            if actual != digest:
                raise ValueError(f"Release archive checksum mismatch for {relative}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true")
    action.add_argument("--check", action="store_true")
    action.add_argument("--build-archive", type=Path)
    action.add_argument("--verify-archive", type=Path)
    action.add_argument("--require-uploaded", action="store_true")
    args = parser.parse_args()
    if args.write:
        payload = write_manifest()
        print(
            f"release assets: {payload['summary']['logical_file_count']} files inventoried"
        )
    elif args.check:
        check_manifest()
        print("release-asset manifest verified")
    elif args.build_archive:
        digest = build_archive(args.build_archive)
        print(f"release archive sha256={digest}")
    elif args.verify_archive:
        verify_archive(args.verify_archive)
        print("release archive verified")
    else:
        payload = check_manifest()
        if payload["release"]["status"] != "uploaded":
            raise SystemExit("release assets have not been uploaded")
        print("release assets are marked uploaded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

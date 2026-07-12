from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.manage_release_assets import (
    build_archive,
    build_manifest,
    check_manifest,
    verify_archive,
    write_manifest,
)


def test_tracked_release_asset_manifest_matches_sources() -> None:
    tracked = json.loads(Path("provenance/release-assets.json").read_text())
    assert tracked["release"]["status"] == "uploaded"
    assert len(tracked["release"]["archive_sha256"]) == 64
    assert tracked["release"]["download_url"].startswith("https://github.com/")
    assert tracked["summary"]["logical_file_count"] > 0
    assert (
        tracked["summary"]["unique_content_count"]
        <= tracked["summary"]["logical_file_count"]
    )
    assert check_manifest() == tracked


def test_release_asset_archive_is_deterministic_and_verified(tmp_path: Path) -> None:
    root = tmp_path / "source"
    generated = root / "docs" / "_static" / "generated"
    generated.mkdir(parents=True)
    (generated / "large.bin").write_bytes(b"a" * (128 * 1024 + 1))
    manifest_path = tmp_path / "manifest.json"
    manifest = write_manifest(manifest_path, root)
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"
    first_digest = build_archive(first, root)
    second_digest = build_archive(second, root)
    assert first_digest == second_digest
    assert first.read_bytes() == second.read_bytes()
    verify_archive(first, manifest)


def test_release_asset_manifest_detects_source_and_status_drift(tmp_path: Path) -> None:
    root = tmp_path / "source"
    generated = root / "docs" / "_static" / "generated"
    generated.mkdir(parents=True)
    asset = generated / "large.bin"
    asset.write_bytes(b"a" * (128 * 1024 + 1))
    manifest_path = tmp_path / "manifest.json"
    write_manifest(manifest_path, root)
    asset.write_bytes(b"b" * (128 * 1024 + 1))
    with pytest.raises(ValueError, match="source drift"):
        check_manifest(manifest_path, root)

    payload = build_manifest(root)
    invalid = copy.deepcopy(payload)
    invalid["release"]["status"] = "uploaded"
    manifest_path.write_text(json.dumps(invalid))
    with pytest.raises(ValueError, match="archive_sha256"):
        check_manifest(manifest_path, root)

#!/usr/bin/env python3
"""Validate and freeze a compact passing Samper Table I campaign artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


MAX_COMPACT_BYTES = 128 * 1024
_MERGED_FIELDS = {"freeze", "mesh_levels", "records"}


def _load(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read Samper campaign artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Samper campaign artifact must be a JSON object")
    return payload


def _validate_passing(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if "active_record" in payload:
        raise ValueError("Cannot freeze an incomplete Samper campaign")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("Samper campaign must contain at least one completed row")
    if not payload.get("research_grade_validation_pass"):
        raise ValueError("Cannot freeze a failing Samper campaign")
    if not all(record.get("finest_level_pass") for record in records):
        raise ValueError("Every frozen Samper row must pass its finest-level gate")
    return records


def _write(payload: dict[str, Any], destination: Path) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    if len(encoded) > MAX_COMPACT_BYTES:
        raise ValueError(
            f"Samper compact evidence is {len(encoded)} bytes; limit is {MAX_COMPACT_BYTES}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(destination)


def freeze_campaign(source: Path, destination: Path) -> dict[str, Any]:
    """Validate a passing campaign and write deterministic compact evidence."""

    payload = _load(source)
    _validate_passing(payload)

    payload.pop("freeze", None)
    payload["freeze"] = {
        "format": "compact-json",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    _write(payload, destination)
    return payload


def merge_campaigns(sources: list[Path], destination: Path) -> dict[str, Any]:
    """Merge compatible passing campaigns while retaining their source hashes."""

    if not sources:
        raise ValueError("At least one Samper campaign is required")
    payloads = [(path, _load(path)) for path in sources]
    common = {
        key: value
        for key, value in payloads[0][1].items()
        if key not in _MERGED_FIELDS
    }
    records: dict[tuple[str, int], dict[str, Any]] = {}
    mesh_levels: set[tuple[int, int]] = set()
    source_hashes: dict[str, str] = {}
    for path, payload in payloads:
        _validate_passing(payload)
        candidate = {key: value for key, value in payload.items() if key not in _MERGED_FIELDS}
        if candidate != common:
            raise ValueError(f"Samper campaigns do not share one contract: {path}")
        if path.name in source_hashes:
            raise ValueError(f"Samper source filenames must be unique: {path.name}")
        source_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        mesh_levels.update(tuple(map(int, level)) for level in payload.get("mesh_levels", []))
        for record in payload["records"]:
            key = (str(record.get("case_kind")), int(record.get("hartmann_number", -1)))
            if key in records:
                raise ValueError(f"Duplicate Samper row: {key[0]} Ha={key[1]}")
            records[key] = record

    merged = {
        **common,
        "mesh_levels": [list(level) for level in sorted(mesh_levels)],
        "records": [records[key] for key in sorted(records)],
        "freeze": {
            "format": "compact-json",
            "source_sha256_by_file": dict(sorted(source_hashes.items())),
        },
    }
    _write(merged, destination)
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    try:
        freeze_campaign(args.source, args.destination)
    except ValueError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

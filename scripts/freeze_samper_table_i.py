#!/usr/bin/env python3
"""Validate and freeze a compact passing Samper Table I campaign artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


MAX_COMPACT_BYTES = 128 * 1024


def _load(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read Samper campaign artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Samper campaign artifact must be a JSON object")
    return payload


def freeze_campaign(source: Path, destination: Path) -> dict[str, Any]:
    """Validate a passing campaign and write deterministic compact evidence."""

    payload = _load(source)
    if "active_record" in payload:
        raise ValueError("Cannot freeze an incomplete Samper campaign")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("Samper campaign must contain at least one completed row")
    if not payload.get("research_grade_validation_pass"):
        raise ValueError("Cannot freeze a failing Samper campaign")
    if not all(record.get("finest_level_pass") for record in records):
        raise ValueError("Every frozen Samper row must pass its finest-level gate")

    payload.pop("freeze", None)
    payload["freeze"] = {
        "format": "compact-json",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    if len(encoded) > MAX_COMPACT_BYTES:
        raise ValueError(
            f"Samper compact evidence is {len(encoded)} bytes; limit is {MAX_COMPACT_BYTES}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(destination)
    return payload


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

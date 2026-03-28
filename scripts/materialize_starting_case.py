#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path


ARCHIVE_NAMES = ("0.tar.gz", "constant.tar.gz", "system.tar.gz")


def materialize_case(case_dir: str | Path) -> dict[str, object]:
    root = Path(case_dir)
    extracted: list[str] = []
    missing: list[str] = []
    for name in ARCHIVE_NAMES:
        archive = root / name
        if not archive.exists():
            missing.append(name)
            continue
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(root, filter="data")
        extracted.append(name)
    return {
        "case_dir": str(root.resolve()),
        "extracted_archives": extracted,
        "missing_archives": missing,
        "has_zero_dir": (root / "0").is_dir(),
        "has_constant_dir": (root / "constant").is_dir(),
        "has_system_dir": (root / "system").is_dir(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Expand a recovered FreeMHD StartingFiles case shell into 0/, constant/, and system/.")
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    payload = materialize_case(args.case_dir)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

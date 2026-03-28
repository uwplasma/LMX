#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lmx.freemhd import freemhd_environment_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect local FreeMHD parity prerequisites.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "external" / "FreeMHD",
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "external" / "FreeMHDPaperAllFigures" / "FreeMHDPaperAllFigures" / "ClosedChannel",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    payload = freemhd_environment_report(args.repo_root, args.reference_root)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

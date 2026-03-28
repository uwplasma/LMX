#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lmx.freemhd import freemhd_container_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect the local FreeMHD Docker bundle and image availability.")
    parser.add_argument(
        "--bundle-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "docker",
        help="Directory containing the generated Dockerfile and runner script.",
    )
    parser.add_argument("--image", type=str, default="lmx-freemhd", help="Expected local runtime image tag.")
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=20,
        help="Timeout for remote base-image manifest resolution.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    payload = freemhd_container_report(
        bundle_root=args.bundle_root,
        image=args.image,
        timeout_seconds=args.timeout_seconds,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

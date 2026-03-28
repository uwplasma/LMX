#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lmx.freemhd import docker_cli_available, docker_daemon_available


def build_freemhd_container(
    image: str,
    bundle_root: str | Path,
    platform: str = "linux/amd64",
) -> subprocess.CompletedProcess[str]:
    bundle_path = Path(bundle_root).resolve()
    command = [
        "docker",
        "build",
        "--progress=plain",
        "--platform",
        platform,
        "-t",
        image,
        str(bundle_path),
    ]
    return subprocess.run(command, text=True, capture_output=True, check=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the local FreeMHD Docker image.")
    parser.add_argument("--image", type=str, default="lmx-freemhd")
    parser.add_argument(
        "--bundle-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "docker",
    )
    parser.add_argument("--platform", type=str, default="linux/amd64")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    if not docker_cli_available():
        payload = {
            "image": args.image,
            "bundle_root": str(args.bundle_root.resolve()),
            "platform": args.platform,
            "docker_cli_available": False,
            "docker_available": False,
            "status": "docker-cli-unavailable",
        }
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, indent=2))
        print(json.dumps(payload, indent=2))
        return 0

    if not docker_daemon_available():
        payload = {
            "image": args.image,
            "bundle_root": str(args.bundle_root.resolve()),
            "platform": args.platform,
            "docker_cli_available": True,
            "docker_available": False,
            "status": "docker-daemon-unavailable",
        }
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, indent=2))
        print(json.dumps(payload, indent=2))
        return 0

    result = build_freemhd_container(args.image, args.bundle_root, platform=args.platform)
    payload = {
        "image": args.image,
        "bundle_root": str(args.bundle_root.resolve()),
        "platform": args.platform,
        "docker_cli_available": True,
        "docker_available": True,
        "status": "ok" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    return 0 if result.returncode == 0 else result.returncode


if __name__ == "__main__":
    raise SystemExit(main())

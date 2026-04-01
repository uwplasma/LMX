#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import tempfile
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lmx.freemhd import docker_cli_available, docker_daemon_available


def _copy_local_freemhd_minimal(source_root: Path, target_root: Path) -> None:
    target_root.mkdir(parents=True, exist_ok=True)
    for name in ("LICENSE", "README.md", ".gitignore"):
        source = source_root / name
        if source.exists():
            shutil.copy2(source, target_root / name)
    source_mhd = source_root / "MHD_Solvers"
    if not source_mhd.is_dir():
        raise FileNotFoundError(f"Expected local FreeMHD tree at {source_root} to contain MHD_Solvers/")
    shutil.copytree(
        source_mhd,
        target_root / "MHD_Solvers",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc", "*.zip"),
    )


def prepare_build_context(bundle_root: str | Path, local_freemhd_root: str | Path | None = None) -> tempfile.TemporaryDirectory[str] | None:
    if local_freemhd_root is None:
        return None
    bundle_path = Path(bundle_root).resolve()
    local_root = Path(local_freemhd_root).resolve()
    temp_dir = tempfile.TemporaryDirectory(prefix="lmx_freemhd_build_")
    context_root = Path(temp_dir.name)
    shutil.copytree(bundle_path, context_root, dirs_exist_ok=True)
    staged_root = context_root / "FreeMHD"
    if staged_root.exists():
        shutil.rmtree(staged_root)
    _copy_local_freemhd_minimal(local_root, staged_root)
    return temp_dir


def build_freemhd_container(
    image: str,
    bundle_root: str | Path,
    platform: str = "linux/amd64",
    local_freemhd_root: str | Path | None = None,
) -> subprocess.CompletedProcess[str]:
    temp_context = prepare_build_context(bundle_root, local_freemhd_root)
    bundle_path = Path(temp_context.name) if temp_context is not None else Path(bundle_root).resolve()
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
    try:
        return subprocess.run(command, text=True, capture_output=True, check=False)
    finally:
        if temp_context is not None:
            temp_context.cleanup()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the local FreeMHD Docker image.")
    parser.add_argument("--image", type=str, default="lmx-freemhd")
    parser.add_argument(
        "--bundle-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "docker",
    )
    parser.add_argument("--platform", type=str, default="linux/amd64")
    parser.add_argument(
        "--local-freemhd-root",
        type=Path,
        default=None,
        help="Optional local FreeMHD checkout to stage into the build context instead of cloning upstream.",
    )
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

    result = build_freemhd_container(
        args.image,
        args.bundle_root,
        platform=args.platform,
        local_freemhd_root=args.local_freemhd_root,
    )
    payload = {
        "image": args.image,
        "bundle_root": str(args.bundle_root.resolve()),
        "platform": args.platform,
        "local_freemhd_root": str(args.local_freemhd_root.resolve()) if args.local_freemhd_root is not None else None,
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

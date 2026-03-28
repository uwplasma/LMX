#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lmx.validation import docker_available


def run_freemhd_case(
    image: str,
    case_dir: str | Path,
    bundle_root: str | Path,
    cores: int = 4,
    solver: str = "epotMultiRegionFoam",
) -> subprocess.CompletedProcess[str]:
    case_path = Path(case_dir).resolve()
    bundle_path = Path(bundle_root).resolve()
    runner = bundle_path / "run_freemhd_case.sh"
    command = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{case_path}:/workspace/case",
        "-v",
        f"{runner}:/opt/lmx/run_freemhd_case.sh:ro",
        image,
        "bash",
        "/opt/lmx/run_freemhd_case.sh",
        "/workspace/case",
        str(cores),
        solver,
    ]
    return subprocess.run(command, text=True, capture_output=True, check=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a local FreeMHD case inside a prepared Docker image.")
    parser.add_argument("--image", required=True, help="Docker image tag built from ./docker/Dockerfile.")
    parser.add_argument("--case-dir", type=Path, required=True, help="Absolute or relative FreeMHD case directory.")
    parser.add_argument(
        "--bundle-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "docker",
        help="Directory containing run_freemhd_case.sh.",
    )
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--solver", type=str, default="epotMultiRegionFoam")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    if not docker_available():
        raise SystemExit("docker is not available on PATH")

    result = run_freemhd_case(
        image=args.image,
        case_dir=args.case_dir,
        bundle_root=args.bundle_root,
        cores=args.cores,
        solver=args.solver,
    )
    payload = {
        "image": args.image,
        "case_dir": str(args.case_dir.resolve()),
        "bundle_root": str(args.bundle_root.resolve()),
        "cores": args.cores,
        "solver": args.solver,
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

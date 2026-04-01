#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lmx.freemhd import (
    control_dict_application,
    decompose_par_subdomains,
    docker_cli_available,
    docker_daemon_available,
    docker_image_available,
)


def run_freemhd_case(
    image: str,
    case_dir: str | Path,
    bundle_root: str | Path,
    cores: int = 4,
    solver: str = "epotMultiRegionFoam",
    platform: str = "linux/amd64",
    end_time: str | None = None,
    write_interval: str | None = None,
    delta_t: str | None = None,
    start_from: str | None = None,
) -> subprocess.CompletedProcess[str]:
    case_path = Path(case_dir).resolve()
    bundle_path = Path(bundle_root).resolve()
    runner = bundle_path / "run_freemhd_case.sh"
    user_spec = f"{os.getuid()}:{os.getgid()}"
    command = [
        "docker",
        "run",
        "--rm",
        "--platform",
        platform,
        "--user",
        user_spec,
        "-e",
        "HOME=/tmp",
        "-e",
        f"LMX_END_TIME={end_time or ''}",
        "-e",
        f"LMX_WRITE_INTERVAL={write_interval or ''}",
        "-e",
        f"LMX_DELTA_T={delta_t or ''}",
        "-e",
        f"LMX_START_FROM={start_from or ''}",
        "--entrypoint",
        "/bin/bash",
        "-v",
        f"{case_path}:/workspace/case",
        "-v",
        f"{runner}:/opt/lmx/run_freemhd_case.sh:ro",
        image,
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
    parser.add_argument("--cores", type=str, default="auto")
    parser.add_argument("--solver", type=str, default="auto")
    parser.add_argument("--platform", type=str, default="linux/amd64")
    parser.add_argument("--end-time", type=str, default=None)
    parser.add_argument("--write-interval", type=str, default=None)
    parser.add_argument("--delta-t", type=str, default=None)
    parser.add_argument("--start-from", type=str, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    resolved_solver = control_dict_application(args.case_dir) if args.solver == "auto" else args.solver
    if resolved_solver is None:
        resolved_solver = "epotMultiRegionFoam"
    if args.cores == "auto":
        resolved_cores = decompose_par_subdomains(args.case_dir) or 4
    else:
        resolved_cores = int(args.cores)

    if not docker_cli_available():
        payload = {
            "image": args.image,
            "case_dir": str(args.case_dir.resolve()),
            "bundle_root": str(args.bundle_root.resolve()),
            "cores": resolved_cores,
            "solver": resolved_solver,
            "platform": args.platform,
            "end_time": args.end_time,
            "write_interval": args.write_interval,
            "delta_t": args.delta_t,
            "start_from": args.start_from,
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
            "case_dir": str(args.case_dir.resolve()),
            "bundle_root": str(args.bundle_root.resolve()),
            "cores": resolved_cores,
            "solver": resolved_solver,
            "platform": args.platform,
            "end_time": args.end_time,
            "write_interval": args.write_interval,
            "delta_t": args.delta_t,
            "start_from": args.start_from,
            "docker_cli_available": True,
            "docker_available": False,
            "status": "docker-daemon-unavailable",
        }
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, indent=2))
        print(json.dumps(payload, indent=2))
        return 0

    if not docker_image_available(args.image):
        payload = {
            "image": args.image,
            "case_dir": str(args.case_dir.resolve()),
            "bundle_root": str(args.bundle_root.resolve()),
            "cores": resolved_cores,
            "solver": resolved_solver,
            "platform": args.platform,
            "end_time": args.end_time,
            "write_interval": args.write_interval,
            "delta_t": args.delta_t,
            "start_from": args.start_from,
            "docker_cli_available": True,
            "docker_available": True,
            "docker_image_available": False,
            "status": "docker-image-unavailable",
        }
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, indent=2))
        print(json.dumps(payload, indent=2))
        return 0

    result = run_freemhd_case(
        image=args.image,
        case_dir=args.case_dir,
        bundle_root=args.bundle_root,
        cores=resolved_cores,
        solver=resolved_solver,
        platform=args.platform,
        end_time=args.end_time,
        write_interval=args.write_interval,
        delta_t=args.delta_t,
        start_from=args.start_from,
    )
    payload = {
        "image": args.image,
        "case_dir": str(args.case_dir.resolve()),
        "bundle_root": str(args.bundle_root.resolve()),
        "cores": resolved_cores,
        "solver": resolved_solver,
        "platform": args.platform,
        "end_time": args.end_time,
        "write_interval": args.write_interval,
        "delta_t": args.delta_t,
        "start_from": args.start_from,
        "docker_cli_available": True,
        "docker_available": True,
        "docker_image_available": True,
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

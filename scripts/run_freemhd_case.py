#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_freemhd_container import build_freemhd_container
from scripts.extract_freemhd_coupled_log import extract_records
from scripts.patch_freemhd_coupled_logging import patch_freemhd_tree
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
    log_coupled_iterations: bool = False,
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
        "-e",
        f"LMX_LOG_COUPLED_ITERATIONS={'true' if log_coupled_iterations else ''}",
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


def write_process_logs(
    output_path: Path | None,
    process_name: str,
    result: subprocess.CompletedProcess[str],
) -> dict[str, str] | None:
    if output_path is None:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path = output_path.with_suffix(f".{process_name}.stdout.log")
    stderr_path = output_path.with_suffix(f".{process_name}.stderr.log")
    stdout_path.write_text(result.stdout)
    stderr_path.write_text(result.stderr)
    return {
        f"{process_name}_stdout_log": str(stdout_path),
        f"{process_name}_stderr_log": str(stderr_path),
    }


def write_diag_records(
    output_path: Path | None,
    process_name: str,
    stdout_log_path: str | None,
    force_write: bool = False,
) -> dict[str, object] | None:
    if output_path is None or stdout_log_path is None:
        return None
    log_path = Path(stdout_log_path)
    if not log_path.exists():
        return None
    records = extract_records(log_path)
    if not records and not force_write:
        return None
    diag_path = output_path.with_suffix(f".{process_name}.diag.json")
    diag_path.write_text(json.dumps({"records": records}, indent=2))
    last_diag_time = None
    if records:
        times = [float(record["time"]) for record in records if "time" in record]
        if times:
            last_diag_time = max(times)
    return {
        f"{process_name}_diag_json": str(diag_path),
        f"{process_name}_diag_record_count": len(records),
        f"{process_name}_diag_last_time": last_diag_time,
    }


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
    parser.add_argument(
        "--local-freemhd-root",
        type=Path,
        default=None,
        help="Optional local FreeMHD checkout to auto-build the image if it is missing.",
    )
    parser.add_argument(
        "--patch-local-freemhd-logging",
        action="store_true",
        help="Patch the local FreeMHD checkout with LMX coupled-logging diagnostics before auto-building.",
    )
    parser.add_argument("--end-time", type=str, default=None)
    parser.add_argument("--write-interval", type=str, default=None)
    parser.add_argument("--delta-t", type=str, default=None)
    parser.add_argument("--start-from", type=str, default=None)
    parser.add_argument(
        "--log-coupled-iterations",
        action="store_true",
        help="Force logCoupledMhdIterations true in the case controlDict before running the solver.",
    )
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
        auto_build_payload: dict[str, object] | None = None
        if args.local_freemhd_root is not None:
            patched_files: list[str] = []
            if args.patch_local_freemhd_logging:
                patched_files = [str(path) for path in patch_freemhd_tree(args.local_freemhd_root)]
            build_result = build_freemhd_container(
                args.image,
                args.bundle_root,
                platform=args.platform,
                local_freemhd_root=args.local_freemhd_root,
                no_cache=False,
            )
            build_log_paths = write_process_logs(args.output, "build", build_result)
            auto_build_payload = {
                "attempted": True,
                "local_freemhd_root": str(args.local_freemhd_root.resolve()),
                "patch_local_freemhd_logging": args.patch_local_freemhd_logging,
                "patched_files": patched_files,
                "status": "ok" if build_result.returncode == 0 else "failed",
                "returncode": build_result.returncode,
                "stdout_tail": build_result.stdout[-4000:],
                "stderr_tail": build_result.stderr[-4000:],
            }
            if build_log_paths is not None:
                auto_build_payload.update(build_log_paths)
            if build_result.returncode == 0 and docker_image_available(args.image):
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
                    log_coupled_iterations=args.log_coupled_iterations,
                )
                run_log_paths = write_process_logs(args.output, "run", result)
                payload = {
                    "image": args.image,
                    "case_dir": str(args.case_dir.resolve()),
                    "bundle_root": str(args.bundle_root.resolve()),
                    "cores": resolved_cores,
                    "solver": resolved_solver,
                    "platform": args.platform,
                    "local_freemhd_root": str(args.local_freemhd_root.resolve()),
                    "patch_local_freemhd_logging": args.patch_local_freemhd_logging,
                    "end_time": args.end_time,
                    "write_interval": args.write_interval,
                    "delta_t": args.delta_t,
                    "start_from": args.start_from,
                    "log_coupled_iterations": args.log_coupled_iterations,
                    "docker_cli_available": True,
                    "docker_available": True,
                    "docker_image_available": True,
                    "image_auto_built": True,
                    "auto_build": auto_build_payload,
                    "status": "ok" if result.returncode == 0 else "failed",
                    "returncode": result.returncode,
                    "stdout_tail": result.stdout[-4000:],
                    "stderr_tail": result.stderr[-4000:],
                }
                if run_log_paths is not None:
                    payload.update(run_log_paths)
                run_diag_payload = write_diag_records(
                    args.output,
                    "run",
                    payload.get("run_stdout_log"),
                    force_write=args.log_coupled_iterations,
                )
                if run_diag_payload is not None:
                    payload.update(run_diag_payload)
                if args.output is not None:
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(json.dumps(payload, indent=2))
                print(json.dumps(payload, indent=2))
                return 0 if result.returncode == 0 else result.returncode

        payload = {
            "image": args.image,
            "case_dir": str(args.case_dir.resolve()),
            "bundle_root": str(args.bundle_root.resolve()),
            "cores": resolved_cores,
            "solver": resolved_solver,
            "platform": args.platform,
            "local_freemhd_root": str(args.local_freemhd_root.resolve()) if args.local_freemhd_root is not None else None,
            "patch_local_freemhd_logging": args.patch_local_freemhd_logging,
            "end_time": args.end_time,
            "write_interval": args.write_interval,
            "delta_t": args.delta_t,
            "start_from": args.start_from,
            "log_coupled_iterations": args.log_coupled_iterations,
            "docker_cli_available": True,
            "docker_available": True,
            "docker_image_available": False,
            "image_auto_built": False,
            "auto_build": auto_build_payload,
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
        log_coupled_iterations=args.log_coupled_iterations,
    )
    run_log_paths = write_process_logs(args.output, "run", result)
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
        "log_coupled_iterations": args.log_coupled_iterations,
        "docker_cli_available": True,
        "docker_available": True,
        "docker_image_available": True,
        "status": "ok" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }
    if run_log_paths is not None:
        payload.update(run_log_paths)
    run_diag_payload = write_diag_records(
        args.output,
        "run",
        payload.get("run_stdout_log"),
        force_write=args.log_coupled_iterations,
    )
    if run_diag_payload is not None:
        payload.update(run_diag_payload)
        if result.returncode != 0 and int(run_diag_payload.get("run_diag_record_count", 0)) > 0:
            payload["status"] = "partial-failed"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    return 0 if result.returncode == 0 else result.returncode


if __name__ == "__main__":
    raise SystemExit(main())

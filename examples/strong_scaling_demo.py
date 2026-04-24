from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile

from lmx.plotting import write_strong_scaling_plots
from lmx.scaling import summarize_strong_scaling_records, write_strong_scaling_summary_table


def _run_worker(
    *,
    python_executable: str,
    repo_root: Path,
    output_path: Path,
    platform: str,
    benchmark_kind: str,
    nx: int | None,
    num_devices: int,
    ny: int,
    nz: int,
    iterations: int,
    repeats: int,
    profile_dir: Path | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, object]:
    command = [
        python_executable,
        str(repo_root / "scripts" / "run_strong_scaling_worker.py"),
        "--benchmark-kind",
        benchmark_kind,
        "--platform",
        platform,
        "--num-devices",
        str(num_devices),
        *(["--nx", str(nx)] if nx is not None else []),
        "--ny",
        str(ny),
        "--nz",
        str(nz),
        "--iterations",
        str(iterations),
        "--repeats",
        str(repeats),
        "--output",
        str(output_path),
        *(["--profile-dir", str(profile_dir)] if profile_dir is not None else []),
    ]
    worker_env = os.environ.copy()
    if env is not None:
        worker_env.update(env)
    existing_pythonpath = worker_env.get("PYTHONPATH")
    worker_env["PYTHONPATH"] = str(repo_root) if not existing_pythonpath else f"{repo_root}:{existing_pythonpath}"
    subprocess.run(command, check=True, cwd=repo_root, env=worker_env)
    return json.loads(output_path.read_text())


def run_local_cpu_scaling(
    *,
    repo_root: Path,
    out_dir: Path,
    device_counts: tuple[int, ...],
    benchmark_kind: str,
    nx: int | None,
    ny: int,
    nz: int,
    iterations: int,
    repeats: int,
    python_executable: str,
    profile_dir: Path | None = None,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for count in device_counts:
        env = os.environ.copy()
        env["JAX_PLATFORMS"] = "cpu"
        env["OMP_NUM_THREADS"] = "1"
        env["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={count}"
        output_path = out_dir / f"cpu_{count}.json"
        record = _run_worker(
            python_executable=python_executable,
            repo_root=repo_root,
            output_path=output_path,
            platform="CPU",
            benchmark_kind=benchmark_kind,
            nx=nx,
            num_devices=count,
            ny=ny,
            nz=nz,
            iterations=iterations,
            repeats=repeats,
            env=env,
            profile_dir=profile_dir / f"cpu_{count}" if profile_dir is not None else None,
        )
        records.append(record)
    return records


def _sync_repo_to_remote(*, repo_root: Path, remote_host: str, remote_dir: str) -> None:
    with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as handle:
        archive_path = Path(handle.name)
    try:
        with tarfile.open(archive_path, "w") as archive:
            archive.add(repo_root / "lmx", arcname="lmx")
            archive.add(repo_root / "scripts" / "run_strong_scaling_worker.py", arcname="scripts/run_strong_scaling_worker.py")
        subprocess.run(
            [
                "ssh",
                remote_host,
                f"rm -rf {shlex.quote(remote_dir)} && mkdir -p {shlex.quote(Path(remote_dir).parent.as_posix())}",
            ],
            check=True,
        )
        subprocess.run(
            ["scp", str(archive_path), f"{remote_host}:{remote_dir}.tar"],
            check=True,
        )
        subprocess.run(
            [
                "ssh",
                remote_host,
                (
                    f"rm -rf {shlex.quote(remote_dir)} && mkdir -p {shlex.quote(remote_dir)} && "
                    f"tar -xf {shlex.quote(remote_dir)}.tar -C {shlex.quote(remote_dir)} && "
                    f"rm -f {shlex.quote(remote_dir)}.tar"
                ),
            ],
            check=True,
        )
    finally:
        archive_path.unlink(missing_ok=True)


def _query_remote_gpu_indices(remote_host: str) -> list[str]:
    result = subprocess.run(
        [
            "ssh",
            remote_host,
            "nvidia-smi --query-gpu=index --format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _default_visible_devices(remote_host: str, count: int) -> str:
    indices = _query_remote_gpu_indices(remote_host)
    if count < 1 or count > len(indices):
        raise ValueError(f"Requested {count} GPU devices, but remote host exposes {len(indices)}.")
    chosen = indices[-count:]
    return ",".join(chosen)


def run_remote_gpu_scaling(
    *,
    repo_root: Path,
    out_dir: Path,
    remote_host: str,
    remote_dir: str,
    device_counts: tuple[int, ...],
    benchmark_kind: str,
    nx: int | None,
    ny: int,
    nz: int,
    iterations: int,
    repeats: int,
    python_executable: str = "python3",
    profile_dir: Path | None = None,
) -> list[dict[str, object]]:
    _sync_repo_to_remote(repo_root=repo_root, remote_host=remote_host, remote_dir=remote_dir)
    local_records: list[dict[str, object]] = []
    for count in device_counts:
        visible_devices = _default_visible_devices(remote_host, count)
        remote_json = f"{remote_dir}/artifacts/strong_scaling/gpu_{count}.json"
        profile_arg = ""
        if profile_dir is not None:
            remote_profile = Path(remote_dir) / "artifacts" / "strong_scaling" / f"profile_gpu_{count}"
            profile_arg = f" --profile-dir {shlex.quote(str(remote_profile))}"
        remote_command = (
            f"cd {shlex.quote(remote_dir)} && "
            f"PYTHONPATH={shlex.quote(remote_dir)} CUDA_VISIBLE_DEVICES={shlex.quote(visible_devices)} JAX_PLATFORMS=cuda "
            f"{shlex.quote(python_executable)} scripts/run_strong_scaling_worker.py "
            f"--benchmark-kind {shlex.quote(benchmark_kind)} --platform GPU --num-devices {count} "
            f"{'' if nx is None else f'--nx {nx} '}--ny {ny} --nz {nz} "
            f"--iterations {iterations} --repeats {repeats} --output {shlex.quote(remote_json)}"
            f"{profile_arg}"
        )
        subprocess.run(["ssh", remote_host, remote_command], check=True)
        local_output = out_dir / f"gpu_{count}.json"
        subprocess.run(["scp", f"{remote_host}:{remote_json}", str(local_output)], check=True)
        local_records.append(json.loads(local_output.read_text()))
    return local_records


def run_strong_scaling_demo(
    *,
    out_dir: Path,
    benchmark_kind: str = "extruded3d",
    cpu_counts: tuple[int, ...] = (1, 2, 4),
    gpu_counts: tuple[int, ...] = (1, 2),
    cpu_problem: tuple[int, int, int] = (2048, 64, 64),
    gpu_problem: tuple[int, int, int] = (6144, 96, 96),
    cpu_iterations: int = 1024,
    gpu_iterations: int = 2048,
    repeats: int = 2,
    python_executable: str = sys.executable,
    remote_host: str | None = None,
    remote_dir: str = "/home/rjorge/tmp/lmx_scaling_repo",
    profile_dir: Path | None = None,
) -> dict[str, object]:
    repo_root = Path(__file__).resolve().parents[1]
    out_dir.mkdir(parents=True, exist_ok=True)
    records = run_local_cpu_scaling(
        repo_root=repo_root,
        out_dir=out_dir,
        device_counts=cpu_counts,
        benchmark_kind=benchmark_kind,
        nx=cpu_problem[0],
        ny=cpu_problem[1],
        nz=cpu_problem[2],
        iterations=cpu_iterations,
        repeats=repeats,
        python_executable=python_executable,
        profile_dir=profile_dir,
    )
    if remote_host is not None:
        records.extend(
            run_remote_gpu_scaling(
                repo_root=repo_root,
                out_dir=out_dir,
                remote_host=remote_host,
                remote_dir=remote_dir,
                device_counts=gpu_counts,
                benchmark_kind=benchmark_kind,
                nx=gpu_problem[0],
                ny=gpu_problem[1],
                nz=gpu_problem[2],
                iterations=gpu_iterations,
                repeats=repeats,
                profile_dir=profile_dir,
            )
        )

    plots = write_strong_scaling_plots(records, out_dir, case_title="LMX strong scaling")
    table_path = write_strong_scaling_summary_table(records, out_dir / "strong_scaling_table.csv")
    diagnostics = summarize_strong_scaling_records(records)
    diagnostics_path = out_dir / "strong_scaling_diagnostics.json"
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2))
    summary = {
        "records": records,
        "plots": [path.name for path in plots],
        "table": table_path.name,
        "diagnostics": diagnostics,
        "diagnostics_path": diagnostics_path.name,
    }
    (out_dir / "strong_scaling_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the LMX strong-scaling demo.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/examples/strong_scaling"))
    parser.add_argument("--benchmark-kind", choices=("stencil2d", "extruded3d", "extruded_solve"), default="extruded3d")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--remote-host", type=str, default=None)
    parser.add_argument("--remote-dir", type=str, default="/home/rjorge/tmp/lmx_scaling_repo")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--cpu-iterations", type=int, default=None)
    parser.add_argument("--gpu-iterations", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--cpu-counts", type=str, default="1,2,4")
    parser.add_argument("--gpu-counts", type=str, default="1,2")
    parser.add_argument("--cpu-nx", type=int, default=2048)
    parser.add_argument("--cpu-ny", type=int, default=64)
    parser.add_argument("--cpu-nz", type=int, default=64)
    parser.add_argument("--gpu-nx", type=int, default=6144)
    parser.add_argument("--gpu-ny", type=int, default=96)
    parser.add_argument("--gpu-nz", type=int, default=96)
    parser.add_argument("--profile", action="store_true", help="Collect a JAX trace for the first repeat of each worker.")
    args = parser.parse_args(argv)

    shared_iterations = args.iterations
    cpu_iterations = args.cpu_iterations if args.cpu_iterations is not None else shared_iterations
    gpu_iterations = args.gpu_iterations if args.gpu_iterations is not None else shared_iterations
    if cpu_iterations is None:
        cpu_iterations = 1024
    if gpu_iterations is None:
        gpu_iterations = 2048

    cpu_problem = (args.cpu_nx, args.cpu_ny, args.cpu_nz)
    gpu_problem = (args.gpu_nx, args.gpu_ny, args.gpu_nz)
    if args.benchmark_kind == "extruded_solve":
        if cpu_problem == (2048, 64, 64):
            cpu_problem = (48, 24, 24)
        if gpu_problem == (6144, 96, 96):
            gpu_problem = (96, 32, 32)
        if args.iterations is None and args.cpu_iterations is None:
            cpu_iterations = 12
        if args.iterations is None and args.gpu_iterations is None:
            gpu_iterations = 12

    run_strong_scaling_demo(
        out_dir=args.output,
        benchmark_kind=args.benchmark_kind,
        cpu_counts=tuple(int(value) for value in args.cpu_counts.split(",") if value),
        gpu_counts=tuple(int(value) for value in args.gpu_counts.split(",") if value),
        cpu_problem=cpu_problem,
        gpu_problem=gpu_problem,
        cpu_iterations=cpu_iterations,
        gpu_iterations=gpu_iterations,
        python_executable=args.python,
        remote_host=args.remote_host,
        remote_dir=args.remote_dir,
        repeats=args.repeats,
        profile_dir=args.output / "profiles" if args.profile else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

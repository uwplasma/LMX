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


def _run_worker(
    *,
    python_executable: str,
    repo_root: Path,
    output_path: Path,
    platform: str,
    num_devices: int,
    ny: int,
    nz: int,
    iterations: int,
    repeats: int,
    env: dict[str, str] | None = None,
) -> dict[str, object]:
    command = [
        python_executable,
        str(repo_root / "scripts" / "run_strong_scaling_worker.py"),
        "--platform",
        platform,
        "--num-devices",
        str(num_devices),
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
    ny: int,
    nz: int,
    iterations: int,
    repeats: int,
    python_executable: str,
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
            num_devices=count,
            ny=ny,
            nz=nz,
            iterations=iterations,
            repeats=repeats,
            env=env,
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
    ny: int,
    nz: int,
    iterations: int,
    repeats: int,
    python_executable: str = "python3",
) -> list[dict[str, object]]:
    _sync_repo_to_remote(repo_root=repo_root, remote_host=remote_host, remote_dir=remote_dir)
    local_records: list[dict[str, object]] = []
    for count in device_counts:
        visible_devices = _default_visible_devices(remote_host, count)
        remote_json = f"{remote_dir}/artifacts/strong_scaling/gpu_{count}.json"
        remote_command = (
            f"cd {shlex.quote(remote_dir)} && "
            f"PYTHONPATH={shlex.quote(remote_dir)} CUDA_VISIBLE_DEVICES={shlex.quote(visible_devices)} JAX_PLATFORMS=cuda "
            f"{shlex.quote(python_executable)} scripts/run_strong_scaling_worker.py "
            f"--platform GPU --num-devices {count} --ny {ny} --nz {nz} "
            f"--iterations {iterations} --repeats {repeats} --output {shlex.quote(remote_json)}"
        )
        subprocess.run(["ssh", remote_host, remote_command], check=True)
        local_output = out_dir / f"gpu_{count}.json"
        subprocess.run(["scp", f"{remote_host}:{remote_json}", str(local_output)], check=True)
        local_records.append(json.loads(local_output.read_text()))
    return local_records


def run_strong_scaling_demo(
    *,
    out_dir: Path,
    cpu_counts: tuple[int, ...] = (1, 4, 8),
    gpu_counts: tuple[int, ...] = (1, 2),
    cpu_problem: tuple[int, int] = (2048, 2048),
    gpu_problem: tuple[int, int] = (4096, 4096),
    iterations: int = 128,
    repeats: int = 2,
    python_executable: str = sys.executable,
    remote_host: str | None = None,
    remote_dir: str = "/home/rjorge/tmp/lmx_scaling_repo",
) -> dict[str, object]:
    repo_root = Path(__file__).resolve().parents[1]
    out_dir.mkdir(parents=True, exist_ok=True)
    records = run_local_cpu_scaling(
        repo_root=repo_root,
        out_dir=out_dir,
        device_counts=cpu_counts,
        ny=cpu_problem[0],
        nz=cpu_problem[1],
        iterations=iterations,
        repeats=repeats,
        python_executable=python_executable,
    )
    if remote_host is not None:
        records.extend(
            run_remote_gpu_scaling(
                repo_root=repo_root,
                out_dir=out_dir,
                remote_host=remote_host,
                remote_dir=remote_dir,
                device_counts=gpu_counts,
                ny=gpu_problem[0],
                nz=gpu_problem[1],
                iterations=iterations,
                repeats=repeats,
            )
        )

    plots = write_strong_scaling_plots(records, out_dir, case_title="LMX strong scaling")
    summary = {
        "records": records,
        "plots": [path.name for path in plots],
    }
    (out_dir / "strong_scaling_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the LMX strong-scaling demo.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/examples/strong_scaling"))
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--remote-host", type=str, default=None)
    parser.add_argument("--remote-dir", type=str, default="/home/rjorge/tmp/lmx_scaling_repo")
    parser.add_argument("--iterations", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--cpu-counts", type=str, default="1,4,8")
    parser.add_argument("--gpu-counts", type=str, default="1,2")
    parser.add_argument("--cpu-ny", type=int, default=2048)
    parser.add_argument("--cpu-nz", type=int, default=2048)
    parser.add_argument("--gpu-ny", type=int, default=4096)
    parser.add_argument("--gpu-nz", type=int, default=4096)
    args = parser.parse_args(argv)

    run_strong_scaling_demo(
        out_dir=args.output,
        cpu_counts=tuple(int(value) for value in args.cpu_counts.split(",") if value),
        gpu_counts=tuple(int(value) for value in args.gpu_counts.split(",") if value),
        cpu_problem=(args.cpu_ny, args.cpu_nz),
        gpu_problem=(args.gpu_ny, args.gpu_nz),
        python_executable=args.python,
        remote_host=args.remote_host,
        remote_dir=args.remote_dir,
        iterations=args.iterations,
        repeats=args.repeats,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

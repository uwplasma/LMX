from __future__ import annotations

# ruff: noqa: E402 -- repository-root bootstrap must precede project imports.

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from lmx.plotting import write_strong_scaling_plots
from lmx.scaling import (
    summarize_strong_scaling_records,
    write_strong_scaling_summary_table,
)


def _resource_environment(
    path: Path | None, *, backend: str, num_devices: int, required: bool
) -> dict[str, object]:
    """Validate a 60-second admission record before an expensive worker starts."""

    if path is None:
        if required:
            raise ValueError(f"Sustained {backend} scaling requires environment evidence")
        return {"resource_environment_verified": False}
    payload = json.loads(path.read_text())
    rung = payload.get("rungs", {}).get(str(num_devices), {})
    verified = bool(
        payload.get("backend") == backend
        and float(payload.get("sample_seconds", 0.0)) >= 60.0
        and rung.get("num_devices") == num_devices
        and rung.get("verified") is True
    )
    if backend == "cpu":
        affinity = rung.get("affinity_cpus", [])
        verified &= bool(
            len(set(affinity)) == len(affinity) == 2 * num_devices
            and rung.get("allocated_cpu_count") == len(affinity)
        )
    else:
        devices, identities = rung.get("visible_devices", []), rung.get("gpu_identities", [])
        physical = {(item.get("uuid"), item.get("pci_bus_id")) for item in identities}
        verified &= bool(
            len(devices) == len(identities) == len(physical) == num_devices
            and rung.get("foreign_compute_process_count") == 0
            and float(rung.get("max_gpu_utilization_percent", 100.0)) <= 5.0
        )
    if not verified:
        raise ValueError(f"Invalid {backend} environment evidence for {num_devices} device(s)")
    return {
        "resource_environment_verified": True,
        "resource_environment_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "resource_environment": rung,
    }


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
    restart_path: Path | None = None,
    matched_input: Path | None = None,
    evaluator: Path | None = None,
    source_commit: str | None = None,
    env: dict[str, str] | None = None,
    timeout_seconds: float | None = None,
    minimum_warm_seconds: float = 0.0,
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
        "--minimum-warm-seconds",
        str(minimum_warm_seconds),
        "--output",
        str(output_path),
        *(["--source-commit", source_commit] if source_commit is not None else []),
        *(["--profile-dir", str(profile_dir)] if profile_dir is not None else []),
        *(["--restart", str(restart_path)] if restart_path is not None else []),
        *(["--matched-input", str(matched_input)] if matched_input is not None else []),
        *(["--evaluator", str(evaluator)] if evaluator is not None else []),
    ]
    worker_env = os.environ.copy()
    if env is not None:
        worker_env.update(env)
    existing_pythonpath = worker_env.get("PYTHONPATH")
    worker_env["PYTHONPATH"] = (
        str(repo_root)
        if not existing_pythonpath
        else f"{repo_root}:{existing_pythonpath}"
    )
    subprocess.run(
        command, check=True, cwd=repo_root, env=worker_env, timeout=timeout_seconds
    )
    return json.loads(output_path.read_text())


def _forced_cpu_environment(count: int) -> dict[str, str]:
    """Preserve safe XLA options while selecting a bounded CPU device mesh."""

    env = os.environ.copy()
    flags = [flag for flag in shlex.split(env.get("XLA_FLAGS", ""))
        if not flag.startswith(("--xla_force_host_platform_device_count=",
            "--xla_cpu_multi_thread_eigen=", "intra_op_parallelism_threads="))]
    for flag in ("--xla_cpu_multi_thread_eigen=false", "intra_op_parallelism_threads=1"):
        if flag not in flags:
            flags.append(flag)
    env.update(JAX_PLATFORMS="cpu", XLA_FLAGS=" ".join(
        (f"--xla_force_host_platform_device_count={count}", *flags)))
    for name, value in (("JAX_ENABLE_X64", "true"),
        ("XLA_PYTHON_CLIENT_PREALLOCATE", "false"), ("OMP_NUM_THREADS", "1"),
        ("OPENBLAS_NUM_THREADS", "1"), ("MKL_NUM_THREADS", "1"),
        ("NUMEXPR_NUM_THREADS", "1")):
        env[name] = value
    return env


def _validate_matched_b2_topologies(records: list[dict[str, object]]) -> None:
    """Fail closed when fixed-grid device topologies change B2 numerics."""

    if not records:
        raise RuntimeError("Matched B2 topology gate produced no records")
    reference = records[0]
    identity = ("source_fingerprint", "input_sha256", "evaluator_sha256",
        "restart_schema", "coupling_acceleration", "coupling_history_depth")
    for record in records:
        if not record.get("validation_passed", False):
            raise RuntimeError("Matched B2 topology failed its worker validation")
        if any(record.get(name) != reference.get(name) for name in identity):
            raise RuntimeError("Matched B2 topology records do not share one contract")
        for name in ("velocity_l2", "potential_l2", "current_l2"):
            if not np.isclose(
                record[name], reference[name], rtol=2.0e-8, atol=2.0e-9
            ):
                raise RuntimeError(f"Matched B2 topology changed {name}")
        for name in ("pressure_observable", "courant_mean", "courant_max"):
            if not np.allclose(
                record["observables"][name], reference["observables"][name],
                rtol=2.0e-8, atol=2.0e-9,
            ):
                raise RuntimeError(f"Matched B2 topology changed {name}")
        if record.get("schema6_active"):
            if not record.get("anderson_validation_passed", False):
                raise RuntimeError("Matched B2 topology failed schema-6 validation")
            for name in ("anderson_gram", "anderson_weights"):
                if not np.allclose(
                    record[name], reference[name], rtol=2.0e-8, atol=2.0e-9
                ):
                    raise RuntimeError(f"Matched B2 topology changed {name}")


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
    restart_path: Path | None = None,
    matched_input: Path | None = None,
    evaluator: Path | None = None,
    source_commit: str | None = None,
    environment_evidence: Path | None = None,
    timeout_seconds: float | None = None,
    minimum_warm_seconds: float = 0.0,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for count in device_counts:
        environment = _resource_environment(
            environment_evidence, backend="cpu", num_devices=count,
            required=minimum_warm_seconds >= 120.0,
        )
        env = _forced_cpu_environment(count)
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
            profile_dir=profile_dir / f"cpu_{count}"
            if profile_dir is not None
            else None,
            restart_path=restart_path,
            matched_input=matched_input,
            evaluator=evaluator,
            source_commit=source_commit,
            timeout_seconds=timeout_seconds,
            minimum_warm_seconds=minimum_warm_seconds,
        )
        record.update(environment)
        if environment["resource_environment_verified"] and record.get(
            "cpu_affinity") != environment["resource_environment"]["affinity_cpus"]:
            raise RuntimeError("Scaling worker escaped its verified CPU affinity")
        output_path.write_text(json.dumps(record, indent=2))
        records.append(record)
        if benchmark_kind == "matched_b2_smoke" and not record["validation_passed"]:
            raise RuntimeError(f"Matched B2 CPU gate failed; evidence: {output_path}")
    if benchmark_kind == "matched_b2_smoke" and records:
        _validate_matched_b2_topologies(records)
    return records


def _sync_repo_to_remote(*, repo_root: Path, remote_host: str, remote_dir: str) -> None:
    with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as handle:
        archive_path = Path(handle.name)
    try:
        with tarfile.open(archive_path, "w") as archive:
            archive.add(repo_root / "lmx", arcname="lmx")
            archive.add(
                repo_root / "scripts" / "run_strong_scaling_worker.py",
                arcname="scripts/run_strong_scaling_worker.py",
            )
            archive.add(
                repo_root / "scripts" / "run_freemhd_parity_suite.py",
                arcname="scripts/run_freemhd_parity_suite.py",
            )
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
        raise ValueError(
            f"Requested {count} GPU devices, but remote host exposes {len(indices)}."
        )
    chosen = indices[-count:]
    return ",".join(chosen)


def _remote_worker_failure(
    record: dict[str, object], error: subprocess.CalledProcessError
) -> tuple[str, str]:
    """Summarize a failed remote worker without discarding its JSON evidence."""

    failure = record.get("failure")
    if isinstance(failure, dict):
        phase = str(failure.get("phase", "worker"))
        message = str(failure.get("message", "remote worker failed"))
    else:
        phase = "validation" if record.get("validation_passed") is False else "worker"
        message = f"remote worker exited with status {error.returncode}"
    return phase, message


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
    restart_path: Path | None = None,
    matched_input: Path | None = None,
    evaluator: Path | None = None,
    source_commit: str | None = None,
    environment_evidence: Path | None = None,
    timeout_seconds: float = 1800.0,
    minimum_warm_seconds: float = 0.0,
) -> list[dict[str, object]]:
    _sync_repo_to_remote(
        repo_root=repo_root, remote_host=remote_host, remote_dir=remote_dir
    )
    remote_restart = None
    if restart_path is not None:
        remote_restart = f"{remote_dir}/initial_restart.npz"
        subprocess.run(
            ["scp", str(restart_path), f"{remote_host}:{remote_restart}"], check=True
        )
    remote_matched = remote_evaluator = None
    if matched_input is not None and evaluator is not None:
        remote_matched, remote_evaluator = (
            f"{remote_dir}/matched_b2_input.json",
            f"{remote_dir}/matched_b2_evaluator.json",
        )
        subprocess.run(
            ["scp", str(matched_input), f"{remote_host}:{remote_matched}"], check=True
        )
        subprocess.run(
            ["scp", str(evaluator), f"{remote_host}:{remote_evaluator}"], check=True
        )
    local_records: list[dict[str, object]] = []
    for count in device_counts:
        visible_devices = _default_visible_devices(remote_host, count)
        environment = _resource_environment(
            environment_evidence, backend="gpu", num_devices=count,
            required=minimum_warm_seconds >= 120.0,
        )
        if environment["resource_environment_verified"] and environment[
            "resource_environment"]["visible_devices"] != visible_devices.split(","):
            raise RuntimeError("GPU admission evidence does not match selected devices")
        remote_json = f"{remote_dir}/artifacts/strong_scaling/gpu_{count}.json"
        profile_arg = ""
        if profile_dir is not None:
            remote_profile = (
                Path(remote_dir)
                / "artifacts"
                / "strong_scaling"
                / f"profile_gpu_{count}"
            )
            profile_arg = f" --profile-dir {shlex.quote(str(remote_profile))}"
        remote_command = (
            f"cd {shlex.quote(remote_dir)} && "
            f"PYTHONPATH={shlex.quote(remote_dir)} CUDA_VISIBLE_DEVICES={shlex.quote(visible_devices)} "
            "JAX_PLATFORMS=cuda JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false "
            f"{shlex.quote(python_executable)} scripts/run_strong_scaling_worker.py "
            f"--benchmark-kind {shlex.quote(benchmark_kind)} --platform GPU --num-devices {count} "
            f"{'' if nx is None else f'--nx {nx} '}--ny {ny} --nz {nz} "
            f"--iterations {iterations} --repeats {repeats} --output {shlex.quote(remote_json)}"
            f" --minimum-warm-seconds {minimum_warm_seconds}"
            f"{'' if source_commit is None else f' --source-commit {shlex.quote(source_commit)}'}"
            f"{profile_arg}"
            f"{'' if remote_restart is None else f' --restart {shlex.quote(remote_restart)}'}"
            f"{'' if remote_matched is None else f' --matched-input {shlex.quote(remote_matched)} --evaluator {shlex.quote(remote_evaluator)}'}"
        )
        worker_error = None
        try:
            subprocess.run(["ssh", remote_host, remote_command], check=True,
                timeout=timeout_seconds)
        except subprocess.CalledProcessError as error:
            worker_error = error
        local_output = out_dir / f"gpu_{count}.json"
        try:
            subprocess.run(
                ["scp", f"{remote_host}:{remote_json}", str(local_output)], check=True
            )
        except subprocess.CalledProcessError as evidence_error:
            if worker_error is None:
                raise
            raise RuntimeError(
                f"Remote GPU worker failed for {count} device(s), and its JSON "
                f"evidence could not be retrieved: {evidence_error}"
            ) from worker_error
        record = json.loads(local_output.read_text())
        if worker_error is not None:
            phase, message = _remote_worker_failure(record, worker_error)
            raise RuntimeError(
                f"Remote GPU worker failed for {count} device(s) during {phase}: "
                f"{message}; evidence: {local_output}"
            ) from worker_error
        record.update(environment)
        local_output.write_text(json.dumps(record, indent=2))
        local_records.append(record)
        if benchmark_kind == "matched_b2_smoke" and not record["validation_passed"]:
            raise RuntimeError(f"Matched B2 GPU gate failed; evidence: {local_output}")
    if benchmark_kind == "matched_b2_smoke":
        _validate_matched_b2_topologies(local_records)
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
    cpu_restart_path: Path | None = None,
    gpu_restart_path: Path | None = None,
    timeout_seconds: float | None = None,
    minimum_warm_seconds: float = 0.0,
    cpu_environment_evidence: Path | None = None,
    gpu_environment_evidence: Path | None = None,
) -> dict[str, object]:
    repo_root = Path(__file__).resolve().parents[1]
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    out_dir.mkdir(parents=True, exist_ok=True)
    matched_input = gpu_matched_input = evaluator = None
    if benchmark_kind == "matched_b2_smoke":
        if cpu_problem == (2048, 64, 64):
            cpu_problem = (8, 7, 7)
        if gpu_problem == (6144, 96, 96):
            gpu_problem = (8, 7, 7)
        if repeats < 4:
            raise ValueError("matched_b2_smoke requires one cold and three warm runs")
        if any(cpu_problem[0] % count for count in cpu_counts) or (
            remote_host is not None
            and any(gpu_problem[0] % count for count in gpu_counts)
        ):
            raise ValueError("Matched B2 axial size must divide every device count")
        from scripts.run_freemhd_parity_suite import (
            materialize_matched_b2_evaluator,
            materialize_matched_b2_lmx_input,
        )
        shape_label = "x".join(map(str, cpu_problem))
        matched_input, evaluator = (
            out_dir / f"matched_b2_cpu_{shape_label}_{cpu_iterations}steps.json",
            out_dir / "matched_b2_evaluator.json",
        )
        materialize_matched_b2_lmx_input(
            matched_input,
            solver_shape=cpu_problem,
            executed_steps=cpu_iterations,
        )
        materialize_matched_b2_evaluator(evaluator)
        gpu_label = "x".join(map(str, gpu_problem))
        gpu_matched_input = (
            out_dir / f"matched_b2_gpu_{gpu_label}_{gpu_iterations}steps.json"
        )
        if remote_host is not None:
            materialize_matched_b2_lmx_input(
                gpu_matched_input,
                solver_shape=gpu_problem,
                executed_steps=gpu_iterations,
            )
    records = run_local_cpu_scaling(
        repo_root=repo_root,
        out_dir=out_dir,
        device_counts=cpu_counts,
        benchmark_kind=benchmark_kind,
        nx=None if matched_input is not None else cpu_problem[0],
        ny=cpu_problem[1],
        nz=cpu_problem[2],
        iterations=cpu_iterations,
        repeats=repeats,
        python_executable=python_executable,
        profile_dir=profile_dir,
        restart_path=cpu_restart_path,
        matched_input=matched_input,
        evaluator=evaluator,
        source_commit=source_commit,
        environment_evidence=cpu_environment_evidence,
        timeout_seconds=timeout_seconds,
        minimum_warm_seconds=minimum_warm_seconds,
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
                nx=None if matched_input is not None else gpu_problem[0],
                ny=gpu_problem[1],
                nz=gpu_problem[2],
                iterations=gpu_iterations,
                repeats=repeats,
                python_executable=python_executable,
                profile_dir=profile_dir,
                restart_path=gpu_restart_path,
                matched_input=gpu_matched_input,
                evaluator=evaluator,
                source_commit=source_commit,
                environment_evidence=gpu_environment_evidence,
                timeout_seconds=(1800.0 if timeout_seconds is None else timeout_seconds),
                minimum_warm_seconds=minimum_warm_seconds,
            )
        )

    diagnostics = summarize_strong_scaling_records(records)
    plots = write_strong_scaling_plots(records, out_dir, case_title=(
        "LMX sustained fixed-work scaling" if diagnostics["sustained_claim_eligible"]
        else "LMX scaling smoke / calibration"))
    table_path = write_strong_scaling_summary_table(
        records, out_dir / "strong_scaling_table.csv"
    )
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
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/examples/strong_scaling")
    )
    parser.add_argument(
        "--benchmark-kind",
        choices=("extruded3d", "extruded_solve", "matched_b2_smoke"),
        default="extruded3d",
    )
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--remote-host", type=str, default=None)
    parser.add_argument(
        "--remote-dir", type=str, default="/home/rjorge/tmp/lmx_scaling_repo"
    )
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
    parser.add_argument(
        "--worker-timeout",
        type=float,
        default=None,
        help="Hard wall-time ceiling in seconds for each isolated scaling worker.",
    )
    parser.add_argument(
        "--minimum-warm-seconds",
        type=float,
        default=0.0,
        help="Require every warm fixed-work trajectory to meet this duration.",
    )
    parser.add_argument("--cpu-environment-evidence", type=Path)
    parser.add_argument("--gpu-environment-evidence", type=Path)
    parser.add_argument(
        "--cpu-restart",
        type=Path,
        help="Validated restart matching the CPU production-solve grid.",
    )
    parser.add_argument(
        "--gpu-restart",
        type=Path,
        help="Validated restart matching the GPU production-solve grid.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Collect a JAX trace for the first repeat of each worker.",
    )
    args = parser.parse_args(argv)
    if args.benchmark_kind == "matched_b2_smoke" and args.repeats < 4:
        parser.error("matched_b2_smoke requires --repeats 4 or greater")
    if args.benchmark_kind == "matched_b2_smoke" and args.profile:
        parser.error("profile matched_b2_smoke in a separate untimed run")

    shared_iterations = args.iterations
    cpu_iterations = (
        args.cpu_iterations if args.cpu_iterations is not None else shared_iterations
    )
    gpu_iterations = (
        args.gpu_iterations if args.gpu_iterations is not None else shared_iterations
    )
    if cpu_iterations is None:
        cpu_iterations = 2 if args.benchmark_kind == "matched_b2_smoke" else 1024
    if gpu_iterations is None:
        gpu_iterations = 2 if args.benchmark_kind == "matched_b2_smoke" else 2048

    cpu_problem = (args.cpu_nx, args.cpu_ny, args.cpu_nz)
    gpu_problem = (args.gpu_nx, args.gpu_ny, args.gpu_nz)
    if args.benchmark_kind == "matched_b2_smoke":
        cpu_problem = tuple(
            default if value == old else value
            for value, old, default in zip(cpu_problem, (2048, 64, 64), (8, 7, 7))
        )
        gpu_problem = tuple(
            default if value == old else value
            for value, old, default in zip(gpu_problem, (6144, 96, 96), (8, 7, 7))
        )
    if args.benchmark_kind == "extruded_solve":
        if args.cpu_restart is None:
            parser.error("--cpu-restart is required for --benchmark-kind extruded_solve")
        if args.remote_host is not None and args.gpu_restart is None:
            parser.error(
                "--gpu-restart is required for remote extruded_solve scaling"
            )
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
        cpu_restart_path=args.cpu_restart,
        gpu_restart_path=args.gpu_restart,
        timeout_seconds=args.worker_timeout,
        minimum_warm_seconds=args.minimum_warm_seconds,
        cpu_environment_evidence=args.cpu_environment_evidence,
        gpu_environment_evidence=args.gpu_environment_evidence,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

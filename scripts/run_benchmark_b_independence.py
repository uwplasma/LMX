#!/usr/bin/env python3
"""Run checkpointed ALEX B1/B2 solver and thin-wall independence gates."""

# ruff: noqa: E402 -- repository-root bootstrap must precede project imports.

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax
import numpy as np

import lmx
from lmx.benchmarks import (
    benchmark_b_pressure_observable,
    build_benchmark_b_problem,
    load_benchmark_b_spec,
)
from lmx.fringing import solve_extruded_inductionless
from lmx.io import load_extruded_restart_bundle, write_extruded_restart_npz


if ROOT not in Path(lmx.__file__).resolve().parents:
    raise RuntimeError(
        f"Benchmark B runner imported LMX outside its source tree: {lmx.__file__}"
    )

CASE_IDS = ("B1-fringing-pipe", "B2-fringing-square")
VARIANTS = ("baseline", "tight_tolerance", "extended_iterations", "thin_wall")


def _source_fingerprint(root: Path = ROOT) -> str:
    digest = hashlib.sha256()
    paths = (
        sorted((root / "lmx").glob("*.py"))
        + sorted((root / "benchmarks" / "specs").glob("alex-b*.toml"))
        + [root / "scripts" / "run_benchmark_b_independence.py"]
    )
    for path in paths:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _variant_problem(case_id: str, mesh_level: str, variant: str):
    wall = "confirmation" if variant == "thin_wall" else "nominal"
    problem = build_benchmark_b_problem(
        case_id, mesh_level=mesh_level, wall_realization=wall
    )
    spec = load_benchmark_b_spec(case_id)
    tolerance = float(problem.case.solver.coupling_tolerance)
    coupling_iterations = int(problem.case.solver.coupling_iterations)
    potential_iterations = int(problem.case.time_stepper.potential_iterations)
    max_steps = int(problem.case.time_stepper.max_steps)
    if variant == "tight_tolerance":
        tolerance *= float(spec["solver"]["tolerance_independence_factor"])
        potential_iterations *= 2
    elif variant == "extended_iterations":
        factor = float(spec["solver"]["iteration_independence_factor"])
        coupling_iterations = int(round(coupling_iterations * factor))
        max_steps = int(round(max_steps * factor))
    elif variant not in {"baseline", "thin_wall"}:
        raise ValueError(f"Unsupported independence variant {variant!r}")
    case = replace(
        problem.case,
        solver=replace(
            problem.case.solver,
            coupling_tolerance=tolerance,
            coupling_iterations=coupling_iterations,
        ),
        time_stepper=replace(
            problem.case.time_stepper,
            potential_iterations=potential_iterations,
            max_steps=max_steps,
        ),
    )
    return replace(problem, case=case)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_record(
    case_id: str,
    mesh_level: str,
    variant: str,
    *,
    restart_path: Path,
    initial_bundle=None,
    initialization: str | None = None,
    initialization_sha256: str | None = None,
) -> tuple[dict[str, Any], Any]:
    problem = _variant_problem(case_id, mesh_level, variant)
    started = time.perf_counter()
    solution = solve_extruded_inductionless(problem, initial_bundle=initial_bundle)
    jax.block_until_ready(solution.bundle.u)
    elapsed = time.perf_counter() - started
    restart_path.parent.mkdir(parents=True, exist_ok=True)
    write_extruded_restart_npz(solution, problem.case, restart_path)
    observable = np.asarray(benchmark_b_pressure_observable(solution, case_id))
    validation = solution.validation
    return {
        "case_id": case_id,
        "mesh_level": mesh_level,
        "variant": variant,
        "source_fingerprint": _source_fingerprint(),
        "runtime_seconds": elapsed,
        "controls": {
            "wall_realization": "confirmation" if variant == "thin_wall" else "nominal",
            "coupling_tolerance": problem.case.solver.coupling_tolerance,
            "coupling_iterations": problem.case.solver.coupling_iterations,
            "potential_iterations": problem.case.time_stepper.potential_iterations,
            "max_steps": problem.case.time_stepper.max_steps,
            "initialization": (
                initialization
                if initialization is not None
                else (
                    "baseline_restart"
                    if initial_bundle is not None
                    else "frozen_initial_state"
                )
            ),
            "initialization_sha256": initialization_sha256,
        },
        "restart": {
            "path": str(restart_path),
            "sha256": _file_sha256(restart_path),
        },
        "x_over_L": np.asarray(solution.bundle.x).tolist(),
        "primary_observable": observable.tolist(),
        "iteration_residual_history": np.asarray(
            solution.bundle.iteration_residual_history
        ).tolist(),
        "iteration_component_residual_history": np.asarray(
            solution.bundle.iteration_component_residual_history
        ).tolist(),
        "iteration_pressure_residual_history": np.asarray(
            solution.bundle.iteration_pressure_residual_history
        ).tolist(),
        "iteration_electric_linear_history": np.asarray(
            solution.bundle.iteration_electric_linear_history
        ).tolist(),
        "iteration_potential_residual_history": np.asarray(
            solution.bundle.iteration_potential_residual_history
        ).tolist(),
        "diagnostics": {
            "max_residual": validation.max_residual,
            "max_divergence_residual": validation.max_divergence_residual,
            "max_charge_balance_residual": validation.max_charge_balance_residual,
            "volumetric_flow_rate_span": validation.volumetric_flow_rate_span,
            "max_wall_current_leakage": validation.max_wall_current_leakage,
            "net_boundary_current_residual": validation.net_boundary_current_residual,
        },
    }, solution


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _comparison(case_id: str, records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    missing = sorted(set(VARIANTS) - set(records))
    if "baseline" not in records:
        return {"case_id": case_id, "complete": False, "missing_variants": missing}
    spec = load_benchmark_b_spec(case_id)
    baseline = np.asarray(records["baseline"]["primary_observable"], dtype=float)
    uncertainty = float(spec["reference"]["combined_uncertainty_absolute"])
    base_diagnostics = records["baseline"]["diagnostics"]
    steady_max = float(spec["solver"]["steady_residual_max"])

    def steady_limit(record: dict[str, Any]) -> float:
        """Honor a variant's tighter requested tolerance in its steady gate."""

        requested = record.get("controls", {}).get("coupling_tolerance", steady_max)
        return min(steady_max, float(requested))

    gates = {
        "steady_residual": float(base_diagnostics["max_residual"])
        <= steady_limit(records["baseline"]),
        "mass_balance": max(
            float(base_diagnostics["volumetric_flow_rate_span"]),
            float(base_diagnostics.get("max_divergence_residual", 0.0)),
        )
        <= float(spec["solver"]["mass_balance_max"]),
        "current_balance": float(base_diagnostics["max_charge_balance_residual"])
        <= float(spec["solver"]["current_balance_max"]),
        "boundary_current": float(base_diagnostics["net_boundary_current_residual"])
        <= float(spec["solver"]["current_balance_max"]),
    }
    for variant, record in records.items():
        if variant == "baseline":
            continue
        diagnostics = record["diagnostics"]
        prefix = variant.replace("_", "-")
        gates[f"{prefix}_steady_residual"] = float(
            diagnostics["max_residual"]
        ) <= steady_limit(record)
        gates[f"{prefix}_mass_balance"] = max(
            float(diagnostics["volumetric_flow_rate_span"]),
            float(diagnostics.get("max_divergence_residual", 0.0)),
        ) <= float(spec["solver"]["mass_balance_max"])
        gates[f"{prefix}_current_balance"] = float(
            diagnostics["max_charge_balance_residual"]
        ) <= float(spec["solver"]["current_balance_max"])
        gates[f"{prefix}_boundary_current"] = float(
            diagnostics["net_boundary_current_residual"]
        ) <= float(spec["solver"]["current_balance_max"])
    result: dict[str, Any] = {
        "case_id": case_id,
        "complete": not missing,
        "missing_variants": missing,
        "gates": gates,
    }
    if "tight_tolerance" in records:
        tight = np.asarray(
            records["tight_tolerance"]["primary_observable"], dtype=float
        )
        tolerance_delta = float(np.max(np.abs(tight - baseline)) / uncertainty)
        result["tolerance_delta_uncertainty_fraction"] = tolerance_delta
        gates["tolerance_independence"] = tolerance_delta <= float(
            spec["solver"]["tolerance_independence_uncertainty_fraction_max"]
        )
    if "extended_iterations" in records:
        extended = np.asarray(
            records["extended_iterations"]["primary_observable"], dtype=float
        )
        iteration_delta = float(np.max(np.abs(extended - baseline)) / uncertainty)
        result["iteration_delta_uncertainty_fraction"] = iteration_delta
        gates["iteration_independence"] = iteration_delta <= float(
            spec["solver"]["iteration_independence_uncertainty_fraction_max"]
        )
    if "thin_wall" in records:
        thin_wall = np.asarray(records["thin_wall"]["primary_observable"], dtype=float)
        wall_relative = float(
            np.linalg.norm(thin_wall - baseline)
            / max(np.linalg.norm(baseline), uncertainty * np.sqrt(baseline.size))
        )
        result["thin_wall_relative_difference"] = wall_relative
        gates["thin_wall_independence"] = wall_relative <= float(
            spec["wall"]["thickness_independence_relative_max"]
        )
    result["pass"] = bool(result["complete"]) and all(gates.values())
    return result


def _parse_variant_restarts(values: list[str]) -> dict[str, Path]:
    restarts: dict[str, Path] = {}
    for value in values:
        variant, separator, raw_path = value.partition("=")
        if not separator or variant not in VARIANTS or not raw_path:
            raise ValueError(
                "--variant-restart VARIANT=PATH must use one of "
                f"{', '.join(VARIANTS)} followed by '=PATH'"
            )
        restarts[variant] = Path(raw_path)
    return restarts


def _parse_gpu_devices(value: str) -> tuple[str, ...]:
    devices = tuple(item.strip() for item in value.split(",") if item.strip())
    if not devices or len(set(devices)) != len(devices):
        raise ValueError("--gpu-devices requires a comma-separated list of unique IDs")
    return devices


def _gpu_child_command(args, case_id: str, variant: str) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--output",
        str(args.output),
        "--cases",
        case_id,
        "--mesh-level",
        args.mesh_level,
        "--variants",
        variant,
        "--worker",
    ]
    if args.resume:
        command.append("--resume")
    if variant == "baseline" and args.initial_restart is not None:
        command.extend(("--initial-restart", str(args.initial_restart)))
    for restart in args.variant_restart:
        if restart.startswith(f"{variant}="):
            command.extend(("--variant-restart", restart))
    return command


def _run_gpu_wave(args, tasks: list[tuple[str, str]], devices: tuple[str, ...]) -> None:
    """Run independent case variants concurrently, one process per GPU."""

    cache_dir = (
        Path(tempfile.gettempdir()) / "lmx-jax-cache" / _source_fingerprint()[:16]
    )
    for offset in range(0, len(tasks), len(devices)):
        processes = []
        for device, (case_id, variant) in zip(devices, tasks[offset:]):
            environment = {**os.environ, "CUDA_VISIBLE_DEVICES": device}
            environment.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
            # Workers are separate processes; a shared persistent cache avoids
            # recompiling unchanged solver kernels in later restart waves.
            environment.setdefault("JAX_COMPILATION_CACHE_DIR", str(cache_dir))
            process = subprocess.Popen(
                _gpu_child_command(args, case_id, variant),
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            print(f"[GPU {device}] started {case_id}/{variant}", flush=True)
            processes.append((device, case_id, variant, time.perf_counter(), process))
        with ThreadPoolExecutor(max_workers=len(processes)) as pool:
            pending = {
                pool.submit(process.communicate): (
                    device,
                    case_id,
                    variant,
                    started,
                    process,
                )
                for device, case_id, variant, started, process in processes
            }
            for future in as_completed(pending):
                device, case_id, variant, started, process = pending[future]
                stdout, stderr = future.result()
                # A single-variant child normally returns 2 because the complete
                # comparison is unavailable; solver/runtime failures return 1.
                if process.returncode not in {0, 2}:
                    detail = stderr.strip() or stdout.strip()
                    raise RuntimeError(
                        f"GPU worker failed for {case_id}/{variant}: {detail}"
                    )
                elapsed = time.perf_counter() - started
                print(
                    f"[GPU {device}] finished {case_id}/{variant} in {elapsed:.1f}s",
                    flush=True,
                )


def _wave_physics_passes(args, tasks: list[tuple[str, str]]) -> bool:
    """Reject dependent work when completed prerequisite states miss a gate."""

    records: dict[str, dict[str, dict[str, Any]]] = {}
    for case_id, variant in tasks:
        path = args.output / "runs" / f"{case_id}-{args.mesh_level}-{variant}.json"
        records.setdefault(case_id, {})[variant] = json.loads(path.read_text())
    comparisons = [
        _comparison(case_id, case_records) for case_id, case_records in records.items()
    ]
    passed = all(all(item["gates"].values()) for item in comparisons)
    if not passed:
        print(
            json.dumps(
                {"wave": "prerequisites", "cases": comparisons},
                indent=2,
                sort_keys=True,
            )
        )
    return passed


def _run_gpu_campaign(args) -> int:
    devices = _parse_gpu_devices(args.gpu_devices)
    independent = [
        (case_id, variant)
        for case_id in args.cases
        for variant in args.variants
        if variant in {"baseline", "thin_wall"}
    ]
    dependent = [
        (case_id, variant)
        for case_id in args.cases
        for variant in args.variants
        if variant in {"tight_tolerance", "extended_iterations"}
    ]
    _run_gpu_wave(args, independent, devices)
    if independent and not _wave_physics_passes(args, independent):
        return 2
    _run_gpu_wave(args, dependent, devices)
    summary_args = [
        "--output",
        str(args.output),
        "--cases",
        *args.cases,
        "--mesh-level",
        args.mesh_level,
        "--variants",
        *args.variants,
        "--resume",
    ]
    return main(summary_args)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/validation/benchmark_b_independence"),
    )
    parser.add_argument("--cases", nargs="+", choices=CASE_IDS, default=CASE_IDS)
    parser.add_argument(
        "--mesh-level", choices=("coarse", "medium", "fine"), default="coarse"
    )
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--initial-restart",
        type=Path,
        help="Explicit restart used only to initialize a newly run baseline variant.",
    )
    parser.add_argument(
        "--variant-restart",
        action="append",
        default=[],
        metavar="VARIANT=PATH",
        help="Explicit restart for a newly run variant; repeat for multiple variants.",
    )
    parser.add_argument(
        "--gpu-devices",
        metavar="ID[,ID...]",
        help="Run independent variants concurrently, one subprocess per CUDA GPU.",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.gpu_devices is not None and not args.dry_run:
        return _run_gpu_campaign(args)
    variant_restarts = _parse_variant_restarts(args.variant_restart)

    fingerprint = _source_fingerprint()
    records_by_case: dict[str, dict[str, dict[str, Any]]] = {}
    for case_id in args.cases:
        records_by_case[case_id] = {}
        baseline_bundle = None
        baseline_restart_sha256 = None
        for variant in args.variants:
            path = args.output / "runs" / f"{case_id}-{args.mesh_level}-{variant}.json"
            restart_path = path.with_suffix(".npz")
            if args.resume and path.is_file():
                record = json.loads(path.read_text())
                if record.get("source_fingerprint") != fingerprint:
                    raise ValueError(f"Checkpoint fingerprint mismatch: {path}")
                if restart_path.is_file() and variant == "baseline":
                    baseline_bundle = load_extruded_restart_bundle(restart_path).bundle
                    baseline_restart_sha256 = _file_sha256(restart_path)
            elif args.dry_run:
                record = {
                    "case_id": case_id,
                    "mesh_level": args.mesh_level,
                    "variant": variant,
                    "source_fingerprint": fingerprint,
                    "dry_run": True,
                }
            else:
                initialization = None
                initialization_sha256 = None
                explicit_restart = variant_restarts.get(variant)
                if explicit_restart is not None:
                    initial_bundle = load_extruded_restart_bundle(
                        explicit_restart
                    ).bundle
                    initialization = f"provided_restart:{explicit_restart}"
                    initialization_sha256 = _file_sha256(explicit_restart)
                elif variant == "baseline" and args.initial_restart is not None:
                    initial_bundle = load_extruded_restart_bundle(
                        args.initial_restart
                    ).bundle
                    initialization = f"provided_restart:{args.initial_restart}"
                    initialization_sha256 = _file_sha256(args.initial_restart)
                elif variant in {"tight_tolerance", "extended_iterations"}:
                    if baseline_bundle is None:
                        baseline_restart = (
                            args.output
                            / "runs"
                            / f"{case_id}-{args.mesh_level}-baseline.npz"
                        )
                        if baseline_restart.is_file():
                            baseline_bundle = load_extruded_restart_bundle(
                                baseline_restart
                            ).bundle
                            baseline_restart_sha256 = _file_sha256(baseline_restart)
                    initial_bundle = baseline_bundle
                    initialization_sha256 = baseline_restart_sha256
                else:
                    initial_bundle = None
                record, solution = _run_record(
                    case_id,
                    args.mesh_level,
                    variant,
                    restart_path=restart_path,
                    initial_bundle=initial_bundle,
                    initialization=initialization,
                    initialization_sha256=initialization_sha256,
                )
                if variant == "baseline":
                    baseline_bundle = solution.bundle
                    baseline_restart_sha256 = record["restart"]["sha256"]
                _atomic_json(path, record)
            records_by_case[case_id][variant] = record

    if args.worker:
        return 2

    comparisons = [
        _comparison(case_id, records)
        if not any(record.get("dry_run") for record in records.values())
        else {"case_id": case_id, "complete": False, "dry_run": True}
        for case_id, records in records_by_case.items()
    ]
    summary = {
        "schema_version": 1,
        "source_fingerprint": fingerprint,
        "mesh_level": args.mesh_level,
        "cases": comparisons,
        "pass": bool(comparisons)
        and all(item.get("pass") is True for item in comparisons),
    }
    _atomic_json(args.output / "benchmark-b-independence.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["pass"] or args.dry_run else 2


if __name__ == "__main__":
    raise SystemExit(main())

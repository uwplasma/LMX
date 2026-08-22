#!/usr/bin/env python3
"""Run checkpointed ALEX B1/B2 solver and thin-wall independence gates."""

# ruff: noqa: E402 -- repository-root bootstrap must precede project imports.

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import jax
import numpy as np

import lmx
from lmx._fringing_duct import _cross_section_mesh
from lmx.fringing import solve_extruded_inductionless
from lmx.io import (
    load_extruded_restart_bundle,
    write_extruded_bundle_restart_npz,
    write_extruded_restart_npz,
)
from lmx.validation import (
    benchmark_b_pressure_observable,
    build_benchmark_b_problem,
    load_benchmark_b_reference,
    load_benchmark_b_spec,
)
from validation.freemhd import validate_matched_b_record

if ROOT not in Path(lmx.__file__).resolve().parents:
    raise RuntimeError(f"Benchmark B runner imported LMX outside its source tree: {lmx.__file__}")

CASE_IDS = ("B1-fringing-pipe", "B2-fringing-square")
MESH_LEVELS = ("coarse", "medium", "fine")
VARIANTS = ("baseline", "tight_tolerance", "extended_iterations", "thin_wall")


def _summarize_pressure_linear_history(
    history: object, *, expected_steps: int | None = None
) -> dict[str, float | int | bool | None]:
    """Summarize the pressure solves recorded by a Benchmark B run."""

    rows = np.asarray(history, dtype=float).reshape((-1, 5))
    valid = rows[np.all(np.isfinite(rows), axis=1) & (rows[:, 4] >= 0.0)]
    count = len(valid)
    return {
        "max_pressure_linear_residual": (float(np.max(np.abs(valid[:, 0]))) if count else None),
        "max_pressure_linear_relative_residual": (float(np.max(np.abs(valid[:, 1]))) if count else None),
        "pressure_linear_iterations_max": int(np.max(valid[:, 2])) if count else None,
        "pressure_linear_iterations_mean": float(np.mean(valid[:, 2])) if count else None,
        "pressure_linear_solve_count": count,
        "pressure_solves_converged": bool(np.all(valid[:, 3] == 1.0)) if count else None,
        "pressure_linear_diagnostics_complete": count
        == (len(rows) if expected_steps is None else expected_steps),
    }


def _source_fingerprint(root: Path = ROOT) -> str:
    digest = hashlib.sha256()
    paths = (
        sorted((root / "src/lmx").glob("*.py"))
        + sorted((root / "src/lmx/data/benchmarks/specs").glob("alex-b*.toml"))
        + [root / "scripts" / "run_benchmark_b_independence.py"]
    )
    for path in paths:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _evaluate_acceptance(
    case_id: str,
    mesh_campaigns: dict[str, dict[str, Any]],
    matched_freemhd: dict[str, Any] | None = None,
    *,
    matched_freemhd_artifact_root: Path | None = None,
) -> dict[str, Any]:
    """Evaluate the frozen ALEX literature, mesh, and FreeMHD gates."""

    spec = load_benchmark_b_spec(case_id)
    levels = tuple(level["name"] for level in spec["mesh"]["levels"])
    missing = [level for level in levels if level not in mesh_campaigns]
    if missing:
        return {
            "case_id": case_id,
            "complete": False,
            "missing_mesh_levels": missing,
            "pass": False,
        }
    reference = load_benchmark_b_reference(case_id)
    reference_x = np.asarray(reference["x_over_L"], dtype=float)
    reference_p = np.asarray(reference["pressure_observable"], dtype=float)
    reference_u = np.asarray(reference["pressure_uncertainty"], dtype=float)
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    literature: dict[str, dict[str, float]] = {}
    independence: dict[str, bool] = {}
    fingerprints = set()
    for level in levels:
        campaign = mesh_campaigns[level]
        record = campaign.get("baseline", {})
        comparison = campaign.get("independence", {})
        fingerprint = str(campaign.get("source_fingerprint", ""))
        if not fingerprint or record.get("source_fingerprint") != fingerprint:
            raise ValueError(f"Benchmark B {level} source provenance does not match")
        fingerprints.add(fingerprint)
        if record.get("case_id") != case_id or record.get("mesh_level") != level:
            raise ValueError(f"Benchmark B {level} baseline metadata do not match")
        x = np.asarray(record.get("x_over_L"), dtype=float)
        observable = np.asarray(record.get("primary_observable"), dtype=float)
        if (
            x.ndim != 1
            or observable.shape != x.shape
            or x.size < 2
            or not np.all(np.isfinite(x))
            or not np.all(np.isfinite(observable))
            or not np.all(np.diff(x) > 0.0)
            or x[0] < reference_x[0]
            or x[-1] > reference_x[-1]
        ):
            raise ValueError(f"Benchmark B {level} baseline curve is invalid")
        expected = np.interp(x, reference_x, reference_p)
        uncertainty = np.interp(x, reference_x, reference_u)
        weighted = (observable - expected) / uncertainty
        reference_integral = float(np.trapezoid(expected, x))
        literature[level] = {
            "weighted_rms": float(np.sqrt(np.mean(weighted**2))),
            "weighted_linf": float(np.max(np.abs(weighted))),
            "integrated_pressure_relative_error": float(
                abs(np.trapezoid(observable, x) - reference_integral)
                / max(abs(reference_integral), float(np.trapezoid(uncertainty, x)))
            ),
        }
        curves[level] = (x, observable)
        independence[level] = bool(
            comparison.get("case_id") == case_id
            and comparison.get("complete") is True
            and comparison.get("pass") is True
        )
    if len(fingerprints) != 1:
        raise ValueError("Benchmark B mesh campaigns use different source fingerprints")
    uncertainty_floor = float(spec["reference"]["combined_uncertainty_absolute"])

    def relative_change(coarser: str, finer: str) -> float:
        coarse_x, coarse_y = curves[coarser]
        fine_x, fine_y = curves[finer]
        keep = (fine_x >= coarse_x[0]) & (fine_x <= coarse_x[-1])
        fine_y = fine_y[keep]
        delta = np.interp(fine_x[keep], coarse_x, coarse_y) - fine_y
        return float(
            np.linalg.norm(delta) / max(np.linalg.norm(fine_y), uncertainty_floor * np.sqrt(fine_y.size))
        )

    coarse_medium = relative_change(levels[0], levels[1])
    medium_fine = relative_change(levels[1], levels[2])
    finest = literature[levels[-1]]
    freemhd_validation = (
        validate_matched_b_record(
            matched_freemhd,
            expected_case_id=case_id,
            artifact_root=matched_freemhd_artifact_root,
        )
        if matched_freemhd is not None
        else None
    )
    exact_freemhd = bool(freemhd_validation and freemhd_validation["acceptance_pass"])
    acceptance = spec["acceptance"]
    gates = {
        "all_mesh_independence": all(independence.values()),
        "weighted_rms": finest["weighted_rms"] <= float(acceptance["weighted_rms_max"]),
        "weighted_linf": finest["weighted_linf"] <= float(acceptance["weighted_linf_max"]),
        "integrated_pressure": finest["integrated_pressure_relative_error"]
        <= float(acceptance["integrated_pressure_relative_error_max"]),
        "finest_mesh_change": medium_fine <= float(acceptance["finest_mesh_change_relative_max"]),
        "monotonic_or_asymptotic_refinement": (
            literature[levels[2]]["weighted_rms"]
            <= literature[levels[1]]["weighted_rms"]
            <= literature[levels[0]]["weighted_rms"]
        )
        or medium_fine <= coarse_medium,
        "matched_freemhd": exact_freemhd,
    }
    return {
        "case_id": case_id,
        "complete": bool(freemhd_validation and freemhd_validation["schema_complete"]),
        "missing_mesh_levels": [],
        "literature": literature,
        "independence": independence,
        "mesh_change_relative": {
            "coarse_to_medium": coarse_medium,
            "medium_to_fine": medium_fine,
        },
        "freemhd": matched_freemhd,
        "freemhd_validation": freemhd_validation,
        "gates": gates,
        "pass": all(gates.values()),
    }


def _variant_problem(case_id: str, mesh_level: str, variant: str, *, num_devices: int | None = None):
    wall = "confirmation" if variant == "thin_wall" else "nominal"
    problem = build_benchmark_b_problem(
        case_id,
        mesh_level=mesh_level,
        wall_realization=wall,
        num_devices=num_devices,
    )
    spec = load_benchmark_b_spec(case_id)
    tolerance = float(problem.case.solver.coupling_tolerance)
    coupling_iterations = int(problem.case.solver.coupling_iterations)
    potential_iterations = int(problem.case.time_stepper.potential_iterations)
    max_steps = int(problem.case.time_stepper.max_steps)
    if variant == "tight_tolerance":
        tolerance_factor = float(spec["solver"]["tolerance_independence_factor"])
        iteration_factor = float(spec["solver"]["iteration_independence_factor"])
        tolerance *= tolerance_factor
        # Budget both independent perturbations; early convergence preserves
        # the normal runtime while slow refined meshes retain bounded headroom.
        coupling_iterations = int(round(coupling_iterations * iteration_factor / tolerance_factor))
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


def _effective_iteration_limits(problem) -> dict[str, int]:
    """Expose the production caps that implement the frozen ALEX spec."""

    requested = int(problem.case.time_stepper.potential_iterations)
    b1 = problem.case.name.startswith("alex_b1-fringing-pipe_")
    return {
        "electric_iterations": max(requested, 4000 if b1 else 600),
        "projection_iterations": max(requested, 4000),
        "momentum_iterations": max(requested, 400),
    }


def _spatial_placement(field, expected: int | None) -> dict[str, Any]:
    """Record and enforce actual JAX shard placement for a solved field."""

    shards = tuple(field.addressable_shards)
    actual = len(shards)
    requested = expected or 1
    if actual != requested:
        raise RuntimeError(f"Requested {requested} spatial devices, but the solution has {actual} shards")
    return {
        "spatial_devices": requested,
        "actual_spatial_shards": actual,
        "spatial_device_ids": [str(shard.device) for shard in shards],
    }


def _interpolate_axis(values, old, new, axis: int):
    """Linearly interpolate one tensor axis in physical coordinates."""

    moved = np.moveaxis(np.asarray(values), axis, -1)
    flat = moved.reshape((-1, moved.shape[-1]))
    interpolated = np.stack([np.interp(new, old, row) for row in flat])
    shape = (*moved.shape[:-1], len(new))
    return np.moveaxis(interpolated.reshape(shape), -1, axis)


def _prolong_b2_restart(bundle, problem):
    """Trilinearly prolong a B2 restart for initialization on a finer mesh."""

    if not problem.case.name.startswith("alex_b2-fringing-square_"):
        raise ValueError("Restart prolongation currently supports only ALEX B2")
    mesh = _cross_section_mesh(problem.case)
    target = tuple(np.asarray(axis) for axis in (mesh.x_centers, mesh.y_centers, mesh.z_centers))
    source = tuple(np.asarray(axis) for axis in (bundle.x, bundle.y, bundle.z))
    expected_shape = tuple(len(axis) for axis in source)
    if any(np.asarray(getattr(bundle, name)).shape != expected_shape for name in ("u", "v", "w", "p", "phi")):
        raise ValueError("Restart coordinates and field shapes do not match")

    def interpolate(values):
        for axis, (old, new) in enumerate(zip(source, target)):
            values = _interpolate_axis(values, old, new, axis)
        return values

    fields = {name: interpolate(getattr(bundle, name)) for name in ("u", "v", "w", "p", "phi")}
    prolonged = replace(
        bundle,
        x=target[0],
        y=target[1],
        z=target[2],
        field_scale=np.asarray(problem.profile.field_scale),
        rho_phi_plus=None,
        rho_phi_inlet=None,
        **fields,
    )
    return prolonged, {
        "method": "trilinear_physical_coordinates",
        "compact_flux": "reinitialized_on_target_mesh",
        "source_shape": list(expected_shape),
        "target_shape": [len(axis) for axis in target],
    }


def _progress_writer(
    *,
    problem,
    case_id: str,
    variant: str,
    progress_path: Path,
    partial_restart_path: Path,
    started: float,
):
    """Return a callback that atomically records progress and restart state."""

    fingerprint = _source_fingerprint()
    checkpoint_sha256 = None
    checkpoint_step = None

    def write(progress) -> None:
        nonlocal checkpoint_sha256, checkpoint_step
        if progress.checkpoint is not None:
            temporary = partial_restart_path.with_suffix(".tmp.npz")
            write_extruded_bundle_restart_npz(progress.checkpoint, problem.case, temporary)
            temporary.replace(partial_restart_path)
            checkpoint_sha256 = _file_sha256(partial_restart_path)
            checkpoint_step = progress.step
        _atomic_json(
            progress_path,
            {
                "case_id": case_id,
                "variant": variant,
                "source_fingerprint": fingerprint,
                "step": progress.step,
                "total_steps": progress.total_steps,
                "residual": progress.residual,
                "component_residuals": list(progress.component_residuals),
                "pressure_residual": progress.pressure_residual,
                "potential_residual": progress.potential_residual,
                "elapsed_seconds": time.perf_counter() - started,
                "checkpoint": (
                    {
                        "path": str(partial_restart_path),
                        "sha256": checkpoint_sha256,
                        "step": checkpoint_step,
                    }
                    if checkpoint_sha256 is not None
                    else None
                ),
            },
        )

    return write


def _load_partial_restart(
    partial_restart_path: Path,
    progress_path: Path,
    fingerprint: str,
):
    """Load a partial restart only when its progress provenance matches."""

    if not progress_path.is_file():
        raise ValueError(f"Partial restart has no progress metadata: {partial_restart_path}")
    progress = json.loads(progress_path.read_text())
    if progress.get("source_fingerprint") != fingerprint:
        raise ValueError(f"Checkpoint fingerprint mismatch: {partial_restart_path}")
    checkpoint = progress.get("checkpoint") or {}
    if checkpoint.get("sha256") != _file_sha256(partial_restart_path):
        raise ValueError(f"Checkpoint checksum mismatch: {partial_restart_path}")
    return load_extruded_restart_bundle(partial_restart_path).bundle


def _load_restart_observable(restart_path: Path, case_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Read the acceptance curve from the exact checksummed restart."""

    bundle = load_extruded_restart_bundle(restart_path).bundle
    observable = benchmark_b_pressure_observable(SimpleNamespace(bundle=bundle), case_id)
    return np.asarray(bundle.x), np.asarray(observable)


def _run_record(
    case_id: str,
    mesh_level: str,
    variant: str,
    *,
    restart_path: Path,
    checkpoint_interval: int,
    initial_bundle=None,
    initialization: str | None = None,
    initialization_sha256: str | None = None,
    num_devices: int | None = None,
    prolong_restart: bool = False,
) -> tuple[dict[str, Any], Any]:
    problem = _variant_problem(case_id, mesh_level, variant, num_devices=num_devices)
    prolongation = None
    mesh = _cross_section_mesh(problem.case)
    target_shape = (mesh.nx, mesh.ny, mesh.nz)
    if initial_bundle is not None and initial_bundle.u.shape != target_shape:
        if not prolong_restart:
            raise ValueError("Restart shape differs from the target mesh; pass --prolong-restart")
        initial_bundle, prolongation = _prolong_b2_restart(initial_bundle, problem)
    started = time.perf_counter()
    progress_path = restart_path.with_suffix(".progress.json")
    partial_restart_path = restart_path.with_suffix(".partial.npz")
    solution = solve_extruded_inductionless(
        problem,
        initial_bundle=initial_bundle,
        progress_callback=_progress_writer(
            problem=problem,
            case_id=case_id,
            variant=variant,
            progress_path=progress_path,
            partial_restart_path=partial_restart_path,
            started=started,
        ),
        checkpoint_interval=checkpoint_interval,
        num_devices=num_devices,
    )
    jax.block_until_ready(solution.bundle.u)
    placement = _spatial_placement(solution.bundle.u, num_devices)
    elapsed = time.perf_counter() - started
    restart_path.parent.mkdir(parents=True, exist_ok=True)
    write_extruded_restart_npz(solution, problem.case, restart_path)
    observable_x, observable = _load_restart_observable(restart_path, case_id)
    validation = solution.validation
    courant_history = np.asarray(solution.bundle.iteration_courant_history, dtype=float)
    measured_courant = courant_history[:, 2][courant_history[:, 2] >= 0.0]
    completed_steps, final_streak, stop_reason = solution.bundle.stopping_state
    pressure_linear_history = np.asarray(
        solution.bundle.iteration_pressure_linear_history, dtype=float
    ).reshape((-1, 5))
    pressure_linear_diagnostics = _summarize_pressure_linear_history(
        pressure_linear_history, expected_steps=int(completed_steps)
    )
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
            **_effective_iteration_limits(problem),
            "max_steps": problem.case.time_stepper.max_steps,
            "initialization": (
                initialization
                if initialization is not None
                else ("baseline_restart" if initial_bundle is not None else "frozen_initial_state")
            ),
            "initialization_sha256": initialization_sha256,
            "restart_prolongation": prolongation,
            **placement,
            "b1_compatible_steady": case_id == "B1-fringing-pipe",
            "b1_retained_modal_blocks": case_id == "B1-fringing-pipe",
        },
        "restart": {
            "path": str(restart_path),
            "sha256": _file_sha256(restart_path),
        },
        "progress": {
            "path": str(progress_path),
            "checkpoint_interval": checkpoint_interval,
            "partial_restart_path": str(partial_restart_path),
        },
        "x_over_L": observable_x.tolist(),
        "primary_observable": observable.tolist(),
        "iteration_residual_history": np.asarray(solution.bundle.iteration_residual_history).tolist(),
        "iteration_component_residual_history": np.asarray(
            solution.bundle.iteration_component_residual_history
        ).tolist(),
        "iteration_pressure_residual_history": np.asarray(
            solution.bundle.iteration_pressure_residual_history
        ).tolist(),
        "iteration_pressure_linear_history": pressure_linear_history.tolist(),
        "iteration_electric_linear_history": np.asarray(
            solution.bundle.iteration_electric_linear_history
        ).tolist(),
        "iteration_potential_residual_history": np.asarray(
            solution.bundle.iteration_potential_residual_history
        ).tolist(),
        "iteration_courant_history": courant_history.tolist(),
        "diagnostics": {
            "max_residual": validation.max_residual,
            "max_divergence_residual": validation.max_divergence_residual,
            "max_charge_balance_residual": validation.max_charge_balance_residual,
            "volumetric_flow_rate_span": validation.volumetric_flow_rate_span,
            "max_wall_current_leakage": validation.max_wall_current_leakage,
            "net_boundary_current_residual": validation.net_boundary_current_residual,
            "max_courant": float(np.max(measured_courant)) if measured_courant.size else None,
            "completed_steps": completed_steps,
            "final_steady_streak": final_streak,
            "stop_reason": stop_reason,
            **pressure_linear_diagnostics,
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
    steady_steps = int(spec["solver"]["steady_steps_min"])

    def steady_limit(record: dict[str, Any]) -> float:
        """Honor a variant's tighter requested tolerance in its steady gate."""

        requested = record.get("controls", {}).get("coupling_tolerance", steady_max)
        return min(steady_max, float(requested))

    gates = {
        "steady_residual": float(base_diagnostics["max_residual"]) <= steady_limit(records["baseline"]),
        "mass_balance": max(
            float(base_diagnostics["volumetric_flow_rate_span"]),
            float(base_diagnostics.get("max_divergence_residual", 0.0)),
        )
        <= float(spec["solver"]["mass_balance_max"]),
        "current_balance": float(base_diagnostics["max_charge_balance_residual"])
        <= float(spec["solver"]["current_balance_max"]),
        "boundary_current": float(base_diagnostics["net_boundary_current_residual"])
        <= float(spec["solver"]["current_balance_max"]),
        "sustained_stopping": (
            base_diagnostics.get("stop_reason") == "converged"
            and int(base_diagnostics.get("final_steady_streak", -1)) >= steady_steps
        ),
        "pressure_linear_diagnostics": base_diagnostics.get("pressure_linear_diagnostics_complete") is True,
        "pressure_solve_convergence": base_diagnostics.get("pressure_solves_converged") is True,
    }
    for variant, record in records.items():
        if variant == "baseline":
            continue
        diagnostics = record["diagnostics"]
        prefix = variant.replace("_", "-")
        gates[f"{prefix}_steady_residual"] = float(diagnostics["max_residual"]) <= steady_limit(record)
        gates[f"{prefix}_mass_balance"] = max(
            float(diagnostics["volumetric_flow_rate_span"]),
            float(diagnostics.get("max_divergence_residual", 0.0)),
        ) <= float(spec["solver"]["mass_balance_max"])
        gates[f"{prefix}_current_balance"] = float(diagnostics["max_charge_balance_residual"]) <= float(
            spec["solver"]["current_balance_max"]
        )
        gates[f"{prefix}_boundary_current"] = float(diagnostics["net_boundary_current_residual"]) <= float(
            spec["solver"]["current_balance_max"]
        )
        gates[f"{prefix}_sustained_stopping"] = (
            diagnostics.get("stop_reason") == "converged"
            and int(diagnostics.get("final_steady_streak", -1)) >= steady_steps
        )
        gates[f"{prefix}_pressure_linear_diagnostics"] = (
            diagnostics.get("pressure_linear_diagnostics_complete") is True
        )
        gates[f"{prefix}_pressure_solve_convergence"] = diagnostics.get("pressure_solves_converged") is True
    result: dict[str, Any] = {
        "case_id": case_id,
        "complete": not missing,
        "missing_variants": missing,
        "gates": gates,
    }
    if "tight_tolerance" in records:
        tight = np.asarray(records["tight_tolerance"]["primary_observable"], dtype=float)
        tolerance_delta = float(np.max(np.abs(tight - baseline)) / uncertainty)
        result["tolerance_delta_uncertainty_fraction"] = tolerance_delta
        gates["tolerance_independence"] = tolerance_delta <= float(
            spec["solver"]["tolerance_independence_uncertainty_fraction_max"]
        )
    if "extended_iterations" in records:
        extended = np.asarray(records["extended_iterations"]["primary_observable"], dtype=float)
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
                f"--variant-restart VARIANT=PATH must use one of {', '.join(VARIANTS)} followed by '=PATH'"
            )
        restarts[variant] = Path(raw_path)
    return restarts


def _parse_gpu_devices(value: str) -> tuple[str, ...]:
    devices = tuple(item.strip() for item in value.split(",") if item.strip())
    if not devices or len(set(devices)) != len(devices):
        raise ValueError("--gpu-devices requires a comma-separated list of unique IDs")
    return devices


def _parse_acceptance_paths(values: list[str], choices: tuple[str, ...], option: str) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or name not in choices or not raw_path or name in paths:
            raise ValueError(f"{option} NAME=PATH requires each of {', '.join(choices)} once")
        paths[name] = Path(raw_path)
    return paths


def _freeze_acceptance(args) -> int:
    """Combine three completed mesh campaigns without rerunning a solver."""

    mesh_paths = _parse_acceptance_paths(args.acceptance_mesh, MESH_LEVELS, "--acceptance-mesh")
    if set(mesh_paths) != set(MESH_LEVELS):
        raise ValueError("--acceptance-mesh requires coarse, medium, and fine campaigns")
    freemhd_paths = _parse_acceptance_paths(args.freemhd_record, CASE_IDS, "--freemhd-record")
    fingerprint = _source_fingerprint()
    results = []
    for case_id in args.cases:
        mesh_campaigns = {}
        for level, directory in mesh_paths.items():
            summary = json.loads((directory / "benchmark-b-independence.json").read_text())
            if summary.get("mesh_level") != level:
                raise ValueError(f"Benchmark B {level} campaign metadata do not match")
            if summary.get("source_fingerprint") != fingerprint:
                raise ValueError(f"Benchmark B {level} campaign fingerprint mismatch")
            comparison = next(
                (item for item in summary.get("cases", []) if item.get("case_id") == case_id),
                None,
            )
            if comparison is None:
                raise ValueError(f"Benchmark B {level} campaign lacks {case_id}")
            baseline = json.loads((directory / "runs" / f"{case_id}-{level}-baseline.json").read_text())
            mesh_campaigns[level] = {
                "source_fingerprint": summary.get("source_fingerprint"),
                "baseline": baseline,
                "independence": comparison,
            }
        record_path = freemhd_paths.get(case_id)
        freemhd = json.loads(record_path.read_text()) if record_path is not None else None
        results.append(
            _evaluate_acceptance(
                case_id,
                mesh_campaigns,
                freemhd,
                matched_freemhd_artifact_root=record_path.resolve().parent
                if record_path is not None
                else None,
            )
        )
    payload = {
        "schema_version": 1,
        "source_fingerprint": fingerprint,
        "cases": results,
        "pass": bool(results) and all(result["pass"] for result in results),
    }
    _atomic_json(args.output / "benchmark-b-acceptance.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["pass"] else 2


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
        "--checkpoint-interval",
        str(args.checkpoint_interval),
    ]
    if args.resume:
        command.append("--resume")
    if getattr(args, "prolong_restart", False):
        command.append("--prolong-restart")
    if variant == "baseline" and args.initial_restart is not None:
        command.extend(("--initial-restart", str(args.initial_restart)))
    for restart in args.variant_restart:
        if restart.startswith(f"{variant}="):
            command.extend(("--variant-restart", restart))
    return command


def _run_gpu_wave(args, tasks: list[tuple[str, str]], devices: tuple[str, ...]) -> None:
    """Run independent case variants concurrently, one process per GPU."""

    cache_dir = Path(tempfile.gettempdir()) / "lmx-jax-cache" / _source_fingerprint()[:16]
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
                    raise RuntimeError(f"GPU worker failed for {case_id}/{variant}: {detail}")
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
    comparisons = [_comparison(case_id, case_records) for case_id, case_records in records.items()]
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
        "--checkpoint-interval",
        str(args.checkpoint_interval),
    ]
    return main(summary_args)


def _write_b2_evidence_plot(
    reference_csv: Path,
    transverse_record: Path,
    consistent_record: Path,
    field_record: Path,
    output: Path,
) -> Path:
    """Plot existing B2 field and pressure evidence without running a solver."""

    import matplotlib.pyplot as plt

    reference = np.genfromtxt(reference_csv, delimiter=",", names=True)

    def curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        x = np.asarray(payload["x_over_L"], dtype=float)
        pressure = np.asarray(payload["primary_observable"], dtype=float)
        if x.ndim != 1 or pressure.shape != x.shape or x.size < 2:
            raise ValueError(f"Invalid B2 pressure record: {path}")
        return x, pressure

    transverse_x, transverse_p = curve(transverse_record)
    consistent_x, consistent_p = curve(consistent_record)
    with np.load(field_record) as field:
        x = np.asarray(field["x"], dtype=float)
        y = np.asarray(field["y"], dtype=float)
        bx = np.asarray(field["bx"], dtype=float)
        by = np.asarray(field["by"], dtype=float)
    if bx.shape != by.shape or bx.shape[:2] != (x.size, y.size):
        raise ValueError("B2 field coordinates and arrays do not match")
    section = bx.shape[2] // 2
    bx, by = bx[:, :, section], by[:, :, section]
    center = int(np.argmin(np.abs(y)))
    b0 = float(np.max(np.abs(by[:, center])))
    if not np.isfinite(b0) or b0 <= 0.0:
        raise ValueError("B2 field normalization must be positive")
    x_over_l = x - 15.0 if float(np.min(x)) >= 0.0 else x
    magnitude = np.hypot(bx, by) / b0

    fig, (field_ax, pressure_ax) = plt.subplots(1, 2, figsize=(12, 4.6))
    image = field_ax.pcolormesh(x_over_l, y, magnitude.T, shading="auto", cmap="Blues", vmin=0.0)
    field_ax.streamplot(
        x_over_l,
        y,
        (bx / b0).T,
        (by / b0).T,
        color="#17324d",
        density=1.15,
        linewidth=0.7,
        arrowsize=0.7,
    )
    fig.colorbar(image, ax=field_ax, label=r"$|\mathbf{B}|/B_0$")
    field_ax.set(title="Maxwell-consistent fringe field", xlabel=r"$x/L$", ylabel=r"$y/L$")

    pressure_ax.errorbar(
        reference["x_over_L"],
        reference["pressure_observable"],
        yerr=reference["pressure_uncertainty"],
        fmt="o",
        ms=4,
        capsize=2,
        color="black",
        label="ALEX experiment",
    )
    pressure_ax.plot(
        transverse_x,
        transverse_p,
        lw=2.0,
        color="#2563eb",
        label="fine, transverse-only diagnostic",
    )
    pressure_ax.plot(
        consistent_x,
        consistent_p,
        lw=2.0,
        color="#d97706",
        label="coarse, Maxwell-consistent diagnostic",
    )
    pressure_ax.axhline(0.0, color="0.75", lw=0.8)
    pressure_ax.set(
        title="ALEX pressure response",
        xlabel=r"$x/L$",
        ylabel="normalized pressure observable",
    )
    pressure_ax.legend(frameon=False, fontsize=8)
    fig.text(
        0.5,
        0.985,
        "B2 / ALEX — ACCEPTANCE OPEN",
        ha="center",
        va="top",
        color="#92400e",
        weight="bold",
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "#fef3c7",
            "edgecolor": "#f59e0b",
        },
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    save_options = {"pil_kwargs": {"quality": 82, "method": 6}} if output.suffix.lower() == ".webp" else {}
    fig.savefig(output, dpi=120, bbox_inches="tight", **save_options)
    plt.close(fig)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/validation/benchmark_b_independence"),
    )
    parser.add_argument("--cases", nargs="+", choices=CASE_IDS, default=CASE_IDS)
    parser.add_argument("--mesh-level", choices=MESH_LEVELS, default="coarse")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=8,
        help="Write an atomic partial restart every N outer iterations (default: 8).",
    )
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
        "--prolong-restart",
        action="store_true",
        help="Trilinearly initialize a finer B2 mesh from an explicit restart.",
    )
    parser.add_argument(
        "--gpu-devices",
        metavar="ID[,ID...]",
        help="Run independent variants concurrently, one subprocess per CUDA GPU.",
    )
    parser.add_argument(
        "--spatial-devices",
        type=int,
        help="Shard each B2 solve axially across this many visible JAX devices.",
    )
    parser.add_argument(
        "--acceptance-mesh",
        action="append",
        default=[],
        metavar="LEVEL=DIR",
        help="Freeze acceptance from completed coarse/medium/fine campaign directories.",
    )
    parser.add_argument(
        "--freemhd-record",
        action="append",
        default=[],
        metavar="CASE=PATH",
        help="Exact-case record; artifact paths resolve relative to its parent directory.",
    )
    parser.add_argument(
        "--plot-evidence",
        type=Path,
        metavar="OUTPUT",
        help="Write the B2/ALEX evidence panel from existing records and exit.",
    )
    parser.add_argument("--plot-transverse-record", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--plot-consistent-record", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--plot-field-record", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--plot-reference-csv",
        type=Path,
        default=ROOT / "src/lmx/data/benchmarks/references/alex-b2-square.csv",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.plot_evidence is not None:
        records = (
            args.plot_transverse_record,
            args.plot_consistent_record,
            args.plot_field_record,
        )
        if any(path is None for path in records):
            parser.error("--plot-evidence requires all three --plot-*-record inputs")
        _write_b2_evidence_plot(
            args.plot_reference_csv,
            args.plot_transverse_record,
            args.plot_consistent_record,
            args.plot_field_record,
            args.plot_evidence,
        )
        print(args.plot_evidence)
        return 0
    if args.checkpoint_interval <= 0:
        parser.error("--checkpoint-interval must be positive")
    if args.spatial_devices is not None and args.spatial_devices < 1:
        parser.error("--spatial-devices must be positive")
    if args.spatial_devices is not None and args.gpu_devices is not None:
        parser.error("--spatial-devices and --gpu-devices are separate execution modes")
    if (
        args.spatial_devices
        and args.spatial_devices > 1
        and any(case_id != "B2-fringing-square" for case_id in args.cases)
    ):
        parser.error("multi-device spatial sharding currently supports only ALEX B2")
    if args.acceptance_mesh:
        if args.gpu_devices is not None or args.dry_run or args.worker:
            parser.error("acceptance assembly cannot run solver, GPU, or dry-run modes")
        return _freeze_acceptance(args)
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
            partial_restart_path = path.with_suffix(".partial.npz")
            progress_path = path.with_suffix(".progress.json")
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
                    initial_bundle = load_extruded_restart_bundle(explicit_restart).bundle
                    initialization = f"provided_restart:{explicit_restart}"
                    initialization_sha256 = _file_sha256(explicit_restart)
                elif variant == "baseline" and args.initial_restart is not None:
                    initial_bundle = load_extruded_restart_bundle(args.initial_restart).bundle
                    initialization = f"provided_restart:{args.initial_restart}"
                    initialization_sha256 = _file_sha256(args.initial_restart)
                elif args.resume and partial_restart_path.is_file():
                    initial_bundle = _load_partial_restart(partial_restart_path, progress_path, fingerprint)
                    initialization = f"partial_restart:{partial_restart_path}"
                    initialization_sha256 = _file_sha256(partial_restart_path)
                elif variant in {"tight_tolerance", "extended_iterations"}:
                    if baseline_bundle is None:
                        baseline_restart = args.output / "runs" / f"{case_id}-{args.mesh_level}-baseline.npz"
                        if baseline_restart.is_file():
                            baseline_bundle = load_extruded_restart_bundle(baseline_restart).bundle
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
                    checkpoint_interval=args.checkpoint_interval,
                    initial_bundle=initial_bundle,
                    initialization=initialization,
                    initialization_sha256=initialization_sha256,
                    num_devices=args.spatial_devices,
                    prolong_restart=args.prolong_restart,
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
        "pass": bool(comparisons) and all(item.get("pass") is True for item in comparisons),
    }
    _atomic_json(args.output / "benchmark-b-independence.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["pass"] or args.dry_run else 2


if __name__ == "__main__":
    raise SystemExit(main())

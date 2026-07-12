from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp

from .cases import make_hartmann_case, make_hunt_case, make_shercliff_case
from .io import write_paraview
from .physics import build_material_fields, magnetic_field_components
from .plotting import write_case_overview_plots, write_transient_movies
from .reference_data import default_closed_channel_reference_root
from .solvers import solve_steady
import lmx.solvers as solvers
from .validation import (
    closed_channel_validation,
    extract_centerline,
    extract_midplane_profile,
    hartmann_validation,
    validation_summary,
    write_metrics_json,
    write_profile_csv,
)


def _portable_path(path: str | Path, *, relative_to: str | Path | None = None) -> str:
    candidate = Path(path)
    base = Path(relative_to) if relative_to is not None else Path.cwd()
    try:
        return str(candidate.relative_to(base))
    except ValueError:
        try:
            return str(candidate.resolve().relative_to(base.resolve()))
        except ValueError:
            return candidate.name if candidate.name else str(candidate)


def _default_reference_root() -> Path | None:
    root = default_closed_channel_reference_root()
    return root if root.exists() else None


def _build_case(case_kind: str, ha: float, ny: int, nz: int):
    if case_kind == "hartmann":
        return make_hartmann_case(ha=ha, ny=ny, nz=nz)
    if case_kind == "shercliff":
        return make_shercliff_case(ha=ha, ny=ny, nz=nz)
    if case_kind == "hunt":
        return make_hunt_case(ha=ha, ny=ny, nz=nz)
    raise ValueError(f"Unsupported case kind {case_kind!r}")


def solve_case_snapshots(
    case,
    *,
    frame_count: int = 12,
) -> list[dict[str, object]]:
    mesh = solvers._build_mesh(case)
    materials = build_material_fields(case, mesh)
    target_mean_velocity = solvers._target_mean_velocity(case)
    reference_mean_velocity = solvers._reference_mean_velocity(case)
    potential_solver = solvers._resolve_potential_solver(case.time_stepper.potential_solver, materials.fluid_mask)
    interpolate_direct_fluid_walls = not bool(materials.fluid_mask.all())
    (
        initial_u,
        initial_phi,
        initial_jy,
        initial_jz,
        initial_lorentz,
        start_time,
    ) = solvers._initial_solver_state(
        case=case,
        mesh=mesh,
        fluid_mask=materials.fluid_mask,
        interpolate_direct_fluid_walls=interpolate_direct_fluid_walls,
        initial_state=None,
    )
    dt = case.time_stepper.dt
    steps = solvers._bounded_time_step_count(
        start_time=start_time,
        dt=dt,
        t_final=case.time_stepper.t_final,
        max_steps=case.time_stepper.max_steps,
    )
    stride = max(1, steps // max(frame_count, 1))

    frames: list[dict[str, object]] = []
    initial_mean_velocity = float(jnp.mean(jnp.where(materials.fluid_mask, initial_u, 0.0)))
    frames.append(
        {
            "time": float(start_time),
            "case": case,
            "u": initial_u,
            "phi": initial_phi,
            "jy": initial_jy,
            "jz": initial_jz,
            "lorentz_x": initial_lorentz,
            "fluid_mask": materials.fluid_mask,
            "residual": 0.0,
            "potential_residual": 0.0,
            "potential_iterations": 0.0,
            "face_current_max": 0.0,
            "emf_max": 0.0,
            "face_lorentz_max": 0.0,
            "mean_velocity": initial_mean_velocity,
            "applied_forcing": float(case.forcing),
            "pressure_proxy": float(case.forcing),
            "mesh": mesh,
        }
    )
    u = initial_u
    for step_index in range(steps):
        step_time = float(start_time + (step_index + 1) * dt)
        if case.solver.kind != "fully_developed_inductionless":
            raise NotImplementedError("solve_case_snapshots only supports fully_developed_inductionless cases")
        linear_solver = (
            "solvax_pcg"
            if case.solver.linear_solver == "auto"
            else case.solver.linear_solver
        )
        (
            u,
            phi,
            jy,
            jz,
            lorentz,
            residual,
            potential_residual,
            potential_iteration_count,
            _linear_residual,
            _linear_iteration_count,
            face_current_max,
            emf_max,
            face_lorentz_max,
            mean_velocity,
            applied_forcing,
            _potential_initial_residual,
            _linear_initial_residual,
        ) = solvers._fully_developed_case_step(
            case=case,
            mesh=mesh,
            materials=materials,
            u_previous=u,
            step_time=step_time,
            potential_solver=potential_solver,
            target_mean_velocity=target_mean_velocity,
            linear_solver=linear_solver,
            preconditioner=case.solver.preconditioner,
            coupling_iterations=case.solver.coupling_iterations,
            coupling_tolerance=case.solver.coupling_tolerance,
        )
        pressure_proxy = applied_forcing
        should_store = (step_index % stride == 0) or (step_index == steps - 1)
        if should_store:
            frames.append(
                {
                    "time": step_time,
                    "case": case,
                    "u": u,
                    "phi": phi,
                    "jy": jy,
                    "jz": jz,
                    "lorentz_x": lorentz,
                    "fluid_mask": materials.fluid_mask,
                    "residual": float(residual),
                    "potential_residual": float(potential_residual),
                    "potential_iterations": float(potential_iteration_count),
                    "face_current_max": float(face_current_max),
                    "emf_max": float(emf_max),
                    "face_lorentz_max": float(face_lorentz_max),
                    "mean_velocity": float(mean_velocity),
                    "applied_forcing": float(applied_forcing),
                    "pressure_proxy": float(pressure_proxy),
                    "fluid_mask": materials.fluid_mask,
                    "mesh": mesh,
                }
            )
    return frames


def run_case_example(
    *,
    case_kind: str,
    ha: float,
    ny: int,
    nz: int,
    out_dir: str | Path,
    reference_root: str | Path | None = None,
) -> dict[str, object]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    case = _build_case(case_kind, ha, ny, nz)
    solution = solve_steady(case)

    write_paraview(solution, out_dir)
    write_profile_csv(out_dir / f"{case.name}_centerline.csv", extract_centerline(solution))
    write_profile_csv(out_dir / f"{case.name}_midplane_y.csv", extract_midplane_profile(solution, axis="y", fluid_only=True))
    write_profile_csv(out_dir / f"{case.name}_midplane_z.csv", extract_midplane_profile(solution, axis="z", fluid_only=True))

    metrics = validation_summary(solution, case.name, ha=ha)
    reference_root = Path(reference_root) if reference_root else _default_reference_root()
    reference_payload: dict[str, object] = {"available": False}

    y_reference_coordinate = None
    y_reference_values = None
    z_reference_coordinate = None
    z_reference_values = None
    reference_label = "Reference"

    if case_kind == "hartmann":
        comparison = hartmann_validation(solution, ha)
        y_reference_coordinate = comparison.coordinate
        y_reference_values = comparison.reference
        reference_payload = {
            "available": True,
            "kind": "hartmann_analytic",
            "y_l2_error": comparison.l2_error,
            "y_linf_error": comparison.linf_error,
        }
    elif reference_root is not None and reference_root.exists():
        try:
            comparison = closed_channel_validation(solution, case_kind, int(ha), reference_root=reference_root)
        except FileNotFoundError:
            comparison = None
        if comparison is not None:
            y_reference_coordinate = comparison.y_profile.coordinate
            y_reference_values = comparison.y_profile.reference
            z_reference_coordinate = comparison.z_profile.coordinate
            z_reference_values = comparison.z_profile.reference
            reference_label = "Analytical"
            reference_payload = {
                "available": True,
                "kind": "closed_channel_analytical",
                "path": comparison.reference_path,
                "y_l2_error": comparison.y_profile.l2_error,
                "z_l2_error": comparison.z_profile.l2_error,
            }

    plot_paths = write_case_overview_plots(
        solution,
        out_dir,
        case_title=f"{case_kind.capitalize()} case (Ha={int(ha)})",
        y_reference_coordinate=y_reference_coordinate,
        y_reference_values=y_reference_values,
        z_reference_coordinate=z_reference_coordinate,
        z_reference_values=z_reference_values,
        reference_label=reference_label,
    )

    report = {
        "case": case.name,
        "ha": ha,
        "output_dir": _portable_path(out_dir),
        "plots": [_portable_path(path) for path in plot_paths],
        "reference": reference_payload,
        "metrics": metrics,
    }
    write_metrics_json(report, out_dir / "example_report.json")
    return report


def run_theory_meeting_demo(
    *,
    out_dir: str | Path,
    hartmann_ha: float = 20.0,
    shercliff_ha: float = 20.0,
    hunt_ha: float = 20.0,
    resolution: int = 32,
    movie_case: str = "shercliff",
    movie_resolution: int = 24,
    movie_dt: float = 5e-6,
    movie_t_final: float = 6e-5,
    movie_frames: int = 8,
    reference_root: str | Path | None = None,
) -> dict[str, object]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reference_root = Path(reference_root) if reference_root else _default_reference_root()

    hartmann_dir = out_dir / "hartmann"
    shercliff_dir = out_dir / "shercliff"
    hunt_dir = out_dir / "hunt"

    hartmann_report = run_case_example(
        case_kind="hartmann",
        ha=hartmann_ha,
        ny=resolution,
        nz=resolution,
        out_dir=hartmann_dir,
        reference_root=reference_root,
    )
    shercliff_report = run_case_example(
        case_kind="shercliff",
        ha=shercliff_ha,
        ny=resolution,
        nz=resolution,
        out_dir=shercliff_dir,
        reference_root=reference_root,
    )

    hunt_case = make_hunt_case(ha=hunt_ha, ny=resolution, nz=resolution)
    hunt_solution = solve_steady(hunt_case)
    write_paraview(hunt_solution, hunt_dir)
    write_profile_csv(hunt_dir / f"{hunt_case.name}_centerline.csv", extract_centerline(hunt_solution))
    write_profile_csv(hunt_dir / f"{hunt_case.name}_midplane_y.csv", extract_midplane_profile(hunt_solution, axis="y", fluid_only=True))
    write_profile_csv(hunt_dir / f"{hunt_case.name}_midplane_z.csv", extract_midplane_profile(hunt_solution, axis="z", fluid_only=True))
    hunt_metrics = validation_summary(hunt_solution, hunt_case.name, ha=hunt_ha)

    y_reference_coordinate = None
    y_reference_values = None
    z_reference_coordinate = None
    z_reference_values = None
    hunt_reference: dict[str, object] = {"available": False}
    if reference_root is not None and reference_root.exists():
        try:
            comparison = closed_channel_validation(hunt_solution, "hunt", int(hunt_ha), reference_root=reference_root)
        except FileNotFoundError:
            comparison = None
        if comparison is not None:
            y_reference_coordinate = comparison.y_profile.coordinate
            y_reference_values = comparison.y_profile.reference
            z_reference_coordinate = comparison.z_profile.coordinate
            z_reference_values = comparison.z_profile.reference
            hunt_reference = {
                "available": True,
                "kind": "closed_channel_analytical",
                "path": comparison.reference_path,
                "y_l2_error": comparison.y_profile.l2_error,
                "z_l2_error": comparison.z_profile.l2_error,
            }

    hunt_plot_paths = write_case_overview_plots(
        hunt_solution,
        hunt_dir,
        case_title=f"Hunt case (Ha={int(hunt_ha)})",
        y_reference_coordinate=y_reference_coordinate,
        y_reference_values=y_reference_values,
        z_reference_coordinate=z_reference_coordinate,
        z_reference_values=z_reference_values,
        reference_label="Analytical",
    )
    movie_case_ha = {
        "hartmann": hartmann_ha,
        "shercliff": shercliff_ha,
        "hunt": hunt_ha,
    }.get(movie_case)
    if movie_case_ha is None:
        raise ValueError(f"Unsupported movie_case {movie_case!r}")
    movie_case_spec = _build_case(movie_case, movie_case_ha, movie_resolution, movie_resolution)
    movie_case_spec = replace(
        movie_case_spec,
        time_stepper=replace(
            movie_case_spec.time_stepper,
            dt=movie_dt,
            t_final=movie_t_final,
            max_steps=solvers._bounded_time_step_count(
                start_time=0.0,
                dt=movie_dt,
                t_final=movie_t_final,
                max_steps=movie_case_spec.time_stepper.max_steps,
            ),
        ),
    )
    movie_frames_payload = solve_case_snapshots(movie_case_spec, frame_count=movie_frames)
    movie_dir = {"hartmann": hartmann_dir, "shercliff": shercliff_dir, "hunt": hunt_dir}[movie_case]
    movie_field_mode = "bulk_deviation" if movie_case == "hunt" else "raw"
    movie_paths = write_transient_movies(
        movie_frames_payload,
        movie_dir,
        case_title=f"{movie_case.capitalize()} startup (Ha={int(movie_case_ha)})",
        field_mode=movie_field_mode,
        output_stem=f"{movie_case}_startup",
    )

    hunt_report = {
        "case": hunt_case.name,
        "ha": hunt_ha,
        "output_dir": _portable_path(hunt_dir),
        "plots": [_portable_path(path) for path in hunt_plot_paths],
        "reference": hunt_reference,
        "metrics": hunt_metrics,
    }
    write_metrics_json(hunt_report, hunt_dir / "example_report.json")

    report = {
        "output_dir": _portable_path(out_dir),
        "movie_case": movie_case,
        "movie_mode": movie_field_mode,
        "movie_outputs": [_portable_path(path) for path in movie_paths],
        "hartmann": hartmann_report,
        "shercliff": shercliff_report,
        "hunt": hunt_report,
    }
    write_metrics_json(report, out_dir / "meeting_demo_report.json")
    return report


def run_case_example_cli(
    *,
    case_kind: str,
    ha: float,
    ny: int,
    nz: int,
    out_dir: str | Path,
    reference_root: str | Path | None = None,
) -> int:
    report = run_case_example(
        case_kind=case_kind,
        ha=ha,
        ny=ny,
        nz=nz,
        out_dir=out_dir,
        reference_root=reference_root,
    )
    print(json.dumps(report, indent=2))
    return 0

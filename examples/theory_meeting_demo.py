#!/usr/bin/env python3
"""A teachable, verbose LMX workflow for theory-meeting demos.

This example intentionally defines the workflow functions in this file instead
of hiding the details behind a one-line helper. It shows how a user can set up
cases, solver controls, diagnostics, NPZ output, and Matplotlib visualizations.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from lmx.cases import make_hartmann_case, make_hunt_case, make_shercliff_case
from lmx.example_runner import solve_case_snapshots
from lmx.io import write_paraview
from lmx.physics import build_material_fields
from lmx.solvers import solve_steady
from lmx.validation import (
    closed_channel_validation,
    extract_centerline,
    extract_midplane_profile,
    hartmann_validation,
    validation_summary,
    write_metrics_json,
    write_profile_csv,
)
from plot_npz_results import plot_movie_npz, plot_solution_npz


def print_banner(title: str) -> None:
    width = 88
    print("\n" + "=" * width)
    print(f"{title:^{width}}")
    print("=" * width)


def print_block(title: str, payload: dict[str, object]) -> None:
    print(f"\n--- {title} ---")
    for key, value in payload.items():
        print(f"{key:>28s} : {value}")


def make_demo_case(case_kind: str, *, ha: float, resolution: int, dt: float | None = None, t_final: float | None = None):
    """Create one case and optionally override its time controls."""
    if case_kind == "hartmann":
        case = make_hartmann_case(ha=ha, ny=resolution, nz=resolution)
    elif case_kind == "shercliff":
        case = make_shercliff_case(ha=ha, ny=resolution, nz=resolution)
    elif case_kind == "hunt":
        case = make_hunt_case(ha=ha, ny=resolution, nz=resolution)
    else:
        raise ValueError(f"Unsupported case kind {case_kind!r}")

    if dt is not None or t_final is not None:
        dt = float(dt if dt is not None else case.time_stepper.dt)
        t_final = float(t_final if t_final is not None else case.time_stepper.t_final)
        case = replace(
            case,
            time_stepper=replace(
                case.time_stepper,
                dt=dt,
                t_final=t_final,
                max_steps=max(1, int(round(t_final / dt))),
            ),
        )
    return case


def print_case_setup(case, case_kind: str, ha: float, resolution: int) -> None:
    ts = case.time_stepper
    print_block(
        f"{case_kind} setup",
        {
            "case name": case.name,
            "Hartmann number": ha,
            "cross-section cells": f"{resolution} x {resolution}",
            "regions": ", ".join(region.name for region in case.regions),
            "forcing": case.forcing,
            "initial velocity": case.initial_velocity,
            "dt": ts.dt,
            "t_final": ts.t_final,
            "max_steps": ts.max_steps,
            "outer_iterations": ts.outer_iterations,
            "potential_solver": ts.potential_solver,
            "potential_iterations": ts.potential_iterations,
            "potential_tolerance": ts.potential_tolerance,
            "current_reconstruction": ts.current_reconstruction,
        },
    )


def print_solution_log(solution, *, max_rows: int = 14) -> None:
    diag = solution.diagnostics
    time = np.asarray(diag.time_history)
    if time.size == 0:
        return
    stride = max(1, time.size // max_rows)
    selected = list(range(0, time.size, stride))
    if selected[-1] != time.size - 1:
        selected.append(time.size - 1)

    print("\n--- Solver progress log ---")
    print("Time        | max|u|      | mean(u)     | max|J|      | max|JxB|    | U-res       | phi-res     | phi-it")
    print("-" * 111)
    for idx in selected:
        print(
            f"{time[idx]:10.3e} | "
            f"{float(diag.u_max_history[idx]):10.3e} | "
            f"{float(diag.mean_velocity_history[idx]):10.3e} | "
            f"{float(diag.current_max_history[idx]):10.3e} | "
            f"{float(diag.lorentz_max_history[idx]):10.3e} | "
            f"{float(diag.residual_history[idx]):10.3e} | "
            f"{float(diag.potential_residual_history[idx]):10.3e} | "
            f"{float(diag.potential_iterations_history[idx]):6.0f}"
        )


def save_solution_npz(solution, case, path: Path, *, case_kind: str, ha: float) -> Path:
    materials = build_material_fields(case, solution.mesh)
    metadata = {
        "case": solution.case_name,
        "case_kind": case_kind,
        "ha": ha,
        "time": float(solution.state.time),
        "description": "LMX steady solution dump written by examples/theory_meeting_demo.py",
    }
    diag = solution.diagnostics
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        metadata_json=json.dumps(metadata),
        y_centers=np.asarray(solution.mesh.y_centers),
        z_centers=np.asarray(solution.mesh.z_centers),
        y_faces=np.asarray(solution.mesh.y_faces),
        z_faces=np.asarray(solution.mesh.z_faces),
        u=np.asarray(solution.state.u),
        phi=np.asarray(solution.state.phi),
        jy=np.asarray(solution.state.jy),
        jz=np.asarray(solution.state.jz),
        lorentz_x=np.asarray(solution.state.lorentz_x),
        conductivity=np.asarray(materials.conductivity),
        density=np.asarray(materials.density),
        viscosity=np.asarray(materials.viscosity),
        fluid_mask=np.asarray(materials.fluid_mask),
        time_history=np.asarray(diag.time_history),
        residual_history=np.asarray(diag.residual_history),
        potential_residual_history=np.asarray(diag.potential_residual_history),
        potential_iterations_history=np.asarray(diag.potential_iterations_history),
        u_max_history=np.asarray(diag.u_max_history),
        mean_velocity_history=np.asarray(diag.mean_velocity_history),
        current_max_history=np.asarray(diag.current_max_history),
        face_current_max_history=np.asarray(diag.face_current_max_history),
        emf_max_history=np.asarray(diag.emf_max_history),
        lorentz_max_history=np.asarray(diag.lorentz_max_history),
        applied_forcing_history=np.asarray(diag.applied_forcing_history),
        pressure_proxy_history=np.asarray(diag.pressure_proxy_history),
    )
    print(f"NPZ solution dump       : {path}")
    return path


def save_snapshot_npz(frames: list[dict[str, object]], path: Path, *, case_kind: str, ha: float, title: str) -> Path:
    if not frames:
        raise ValueError("Cannot save an empty movie frame list")
    mesh = frames[0]["mesh"]
    metadata = {
        "case": f"{case_kind}_startup",
        "case_kind": case_kind,
        "ha": ha,
        "title": title,
        "description": "LMX transient snapshot dump for Matplotlib movie generation",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        metadata_json=json.dumps(metadata),
        y_centers=np.asarray(mesh.y_centers),
        z_centers=np.asarray(mesh.z_centers),
        y_faces=np.asarray(mesh.y_faces),
        z_faces=np.asarray(mesh.z_faces),
        time=np.asarray([float(frame["time"]) for frame in frames]),
        u_stack=np.asarray([frame["u"] for frame in frames]),
        phi_stack=np.asarray([frame["phi"] for frame in frames]),
        jy_stack=np.asarray([frame["jy"] for frame in frames]),
        jz_stack=np.asarray([frame["jz"] for frame in frames]),
        lorentz_x_stack=np.asarray([frame["lorentz_x"] for frame in frames]),
        residual_history=np.asarray([float(frame["residual"]) for frame in frames]),
        potential_residual_history=np.asarray([float(frame["potential_residual"]) for frame in frames]),
        potential_iterations_history=np.asarray([float(frame["potential_iterations"]) for frame in frames]),
        face_current_max_history=np.asarray([float(frame["face_current_max"]) for frame in frames]),
        emf_max_history=np.asarray([float(frame["emf_max"]) for frame in frames]),
        mean_velocity_history=np.asarray([float(frame["mean_velocity"]) for frame in frames]),
        applied_forcing_history=np.asarray([float(frame["applied_forcing"]) for frame in frames]),
        pressure_proxy_history=np.asarray([float(frame["pressure_proxy"]) for frame in frames]),
    )
    print(f"NPZ snapshot dump       : {path}")
    return path


def run_steady_case(case_kind: str, *, ha: float, resolution: int, out_dir: Path, reference_root: Path | None) -> dict[str, object]:
    case = make_demo_case(case_kind, ha=ha, resolution=resolution)
    print_case_setup(case, case_kind, ha, resolution)
    solution = solve_steady(case)
    print_solution_log(solution)

    case_dir = out_dir / case_kind
    case_dir.mkdir(parents=True, exist_ok=True)
    write_paraview(solution, case_dir)
    write_profile_csv(case_dir / f"{case.name}_centerline.csv", extract_centerline(solution))
    write_profile_csv(case_dir / f"{case.name}_midplane_y.csv", extract_midplane_profile(solution, axis="y", fluid_only=True))
    write_profile_csv(case_dir / f"{case.name}_midplane_z.csv", extract_midplane_profile(solution, axis="z", fluid_only=True))

    metrics = validation_summary(solution, case.name, ha=ha)
    reference: dict[str, object] = {"available": False}
    if case_kind == "hartmann":
        comparison = hartmann_validation(solution, ha)
        reference = {"available": True, "kind": "hartmann_analytic", "l2_error": comparison.l2_error, "linf_error": comparison.linf_error}
    elif reference_root and reference_root.exists():
        try:
            comparison = closed_channel_validation(solution, case_kind, int(ha), reference_root=reference_root)
            reference = {
                "available": True,
                "kind": "closed_channel_analytical",
                "path": comparison.reference_path,
                "y_l2_error": comparison.y_profile.l2_error,
                "z_l2_error": comparison.z_profile.l2_error,
            }
        except FileNotFoundError:
            reference = {"available": False}

    npz_path = save_solution_npz(solution, case, case_dir / f"{case.name}_results.npz", case_kind=case_kind, ha=ha)
    plot_paths = plot_solution_npz(npz_path, case_dir / "plots", title=f"{case_kind.capitalize()} case (Ha={int(ha)})")
    report = {
        "case": case.name,
        "case_kind": case_kind,
        "ha": ha,
        "output_dir": str(case_dir.resolve()),
        "npz": str(npz_path.resolve()),
        "plots": [str(path.resolve()) for path in plot_paths],
        "reference": reference,
        "metrics": metrics,
    }
    write_metrics_json(report, case_dir / "example_report.json")
    return report


def run_movie_case(case_kind: str, *, ha: float, resolution: int, dt: float, t_final: float, frames: int, out_dir: Path) -> dict[str, object]:
    case = make_demo_case(case_kind, ha=ha, resolution=resolution, dt=dt, t_final=t_final)
    print_case_setup(case, f"{case_kind} movie", ha, resolution)
    snapshots = solve_case_snapshots(case, frame_count=frames)
    for frame in snapshots:
        print(
            f"movie frame | t={float(frame['time']):.3e} | "
            f"max|u|={float(np.max(np.abs(frame['u']))):.3e} | "
            f"max|JxB|={float(frame['lorentz_max'] if 'lorentz_max' in frame else np.max(np.abs(frame['lorentz_x']))):.3e}"
        )
    case_dir = out_dir / case_kind
    snapshot_npz = save_snapshot_npz(
        snapshots,
        case_dir / f"{case_kind}_startup_snapshots.npz",
        case_kind=case_kind,
        ha=ha,
        title=f"{case_kind.capitalize()} startup (Ha={int(ha)})",
    )
    movie_paths = plot_movie_npz(snapshot_npz, case_dir / "movie", stem=f"{case_kind}_startup")
    return {
        "case_kind": case_kind,
        "ha": ha,
        "npz": str(snapshot_npz.resolve()),
        "movies": [str(path.resolve()) for path in movie_paths],
    }


def default_reference_root() -> Path | None:
    root = Path("./external/FreeMHDPaperAllFigures/FreeMHDPaperAllFigures/ClosedChannel")
    return root if root.exists() else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a verbose, teachable LMX theory-meeting demo.")
    parser.add_argument("--output", type=Path, default=Path("./artifacts/examples/theory_meeting_demo"))
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--movie-case", choices=["hartmann", "shercliff", "hunt"], default="shercliff")
    parser.add_argument("--movie-resolution", type=int, default=24)
    parser.add_argument("--movie-dt", type=float, default=1e-3)
    parser.add_argument("--movie-t-final", type=float, default=1e-1)
    parser.add_argument("--movie-frames", type=int, default=8)
    parser.add_argument("--hartmann-ha", type=float, default=20.0)
    parser.add_argument("--shercliff-ha", type=float, default=20.0)
    parser.add_argument("--hunt-ha", type=float, default=20.0)
    parser.add_argument("--reference-root", type=Path, default=None)
    args = parser.parse_args(argv)

    reference_root = args.reference_root or default_reference_root()
    args.output.mkdir(parents=True, exist_ok=True)

    print_banner("LMX verbose theory-meeting demo")
    print_block(
        "Input dictionary",
        {
            "output": args.output.resolve(),
            "resolution": args.resolution,
            "movie_case": args.movie_case,
            "movie_resolution": args.movie_resolution,
            "movie_dt": args.movie_dt,
            "movie_t_final": args.movie_t_final,
            "movie_frames": args.movie_frames,
            "reference_root": reference_root,
        },
    )

    reports = {
        "hartmann": run_steady_case("hartmann", ha=args.hartmann_ha, resolution=args.resolution, out_dir=args.output, reference_root=reference_root),
        "shercliff": run_steady_case("shercliff", ha=args.shercliff_ha, resolution=args.resolution, out_dir=args.output, reference_root=reference_root),
        "hunt": run_steady_case("hunt", ha=args.hunt_ha, resolution=args.resolution, out_dir=args.output, reference_root=reference_root),
    }
    movie_ha = {"hartmann": args.hartmann_ha, "shercliff": args.shercliff_ha, "hunt": args.hunt_ha}[args.movie_case]
    movie_report = run_movie_case(
        args.movie_case,
        ha=movie_ha,
        resolution=args.movie_resolution,
        dt=args.movie_dt,
        t_final=args.movie_t_final,
        frames=args.movie_frames,
        out_dir=args.output,
    )

    report = {
        "output_dir": str(args.output.resolve()),
        "movie_case": args.movie_case,
        "steady_cases": reports,
        "movie": movie_report,
    }
    report_path = args.output / "meeting_demo_report.json"
    write_metrics_json(report, report_path)
    print_block(
        "Written outputs",
        {
            "report": report_path.resolve(),
            "hartmann_npz": reports["hartmann"]["npz"],
            "shercliff_npz": reports["shercliff"]["npz"],
            "hunt_npz": reports["hunt"]["npz"],
            "movie_npz": movie_report["npz"],
            "movie_count": len(movie_report["movies"]),
        },
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

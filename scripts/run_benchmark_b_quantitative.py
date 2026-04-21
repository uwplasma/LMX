from __future__ import annotations

import argparse
import csv
import json
from dataclasses import is_dataclass, replace
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.pipe_reference_comparison_demo import _extract_pipe_profile
from lmx import enable_compilation_cache
from lmx.fringing import (
    build_layered_duct_extruded_problem,
    build_pipe_ogrid_extruded_problem,
    build_square_duct_extruded_problem,
    solve_extruded_inductionless,
)
from lmx.reference_data import default_fringing_pipe_reference_root, load_fringing_pipe_profile

JAX_CACHE_DIR = Path("artifacts/jax_cache")


def _set_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.figsize": (12.8, 7.4),
            "axes.grid": True,
            "grid.alpha": 0.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
        }
    )


def _pipe_reference_root(reference_dir: Path | None) -> Path:
    if reference_dir is not None:
        return reference_dir
    return default_fringing_pipe_reference_root()


def _replace_fields(obj, **changes):
    if is_dataclass(obj):
        return replace(obj, **changes)
    for name, value in changes.items():
        setattr(obj, name, value)
    return obj


def _pipe_profile_errors(bundle, reference_dir: Path | None) -> dict[str, float]:
    root = _pipe_reference_root(reference_dir)
    reference_profiles = {name: load_fringing_pipe_profile(name, root) for name in ("center", "negative", "positive")}
    reference_velocity_scale = max(
        max(np.max(np.abs(np.asarray(profile.velocity, dtype=float))), 1.0e-12)
        for profile in reference_profiles.values()
    )
    lmx_velocity_profiles = {
        name: _extract_pipe_profile(bundle, x_offset_fraction=profile.x_offset_fraction, field_name="u")
        for name, profile in reference_profiles.items()
    }
    lmx_potential_profiles = {
        name: _extract_pipe_profile(bundle, x_offset_fraction=profile.x_offset_fraction, field_name="phi")
        for name, profile in reference_profiles.items()
    }
    lmx_velocity_scale = max(max(np.max(np.abs(profile)), 1.0e-12) for _, profile in lmx_velocity_profiles.values())
    errors: dict[str, float] = {}
    for name in ("center", "negative", "positive"):
        reference = reference_profiles[name]
        reference_velocity = np.asarray(reference.velocity, dtype=float)
        use_velocity = float(np.max(np.abs(reference_velocity))) > 1.0e-8
        if use_velocity:
            lmx_coord, lmx_profile_raw = lmx_velocity_profiles[name]
            ref_profile = reference_velocity / reference_velocity_scale
            lmx_profile = lmx_profile_raw / lmx_velocity_scale
            metric_prefix = "velocity"
        else:
            lmx_coord, lmx_profile_raw = lmx_potential_profiles[name]
            potential_values = np.loadtxt(reference.path, delimiter=",", skiprows=1, usecols=13)
            ref_scale = max(float(np.max(np.abs(potential_values))), 1.0e-12)
            ref_profile = potential_values / ref_scale
            lmx_scale = max(float(np.max(np.abs(lmx_profile_raw))), 1.0e-12)
            lmx_profile = lmx_profile_raw / lmx_scale
            metric_prefix = "potential"
        interpolated = np.interp(np.asarray(reference.coordinate, dtype=float), lmx_coord, lmx_profile)
        errors[f"{name}_{metric_prefix}_l2_error"] = float(np.sqrt(np.mean((interpolated - ref_profile) ** 2)))
        errors[f"{name}_{metric_prefix}_linf_error"] = float(np.max(np.abs(interpolated - ref_profile)))
    return errors


def _build_problem(
    geometry_kind: str,
    *,
    ha_peak: float,
    ny: int,
    nz: int,
    nx_stations: int,
    max_steps: int,
    coupling_iterations: int,
    potential_iterations: int,
):
    if geometry_kind == "rect_duct":
        problem = build_square_duct_extruded_problem(ha_peak=ha_peak, ny=ny, nz=nz, nx_stations=nx_stations)
        return _replace_fields(
            problem,
            case=_replace_fields(
                problem.case,
                solver=_replace_fields(problem.case.solver, coupling_iterations=coupling_iterations),
                time_stepper=_replace_fields(problem.case.time_stepper, max_steps=max_steps, potential_iterations=potential_iterations),
            ),
        )
    if geometry_kind == "layered_duct":
        problem = build_layered_duct_extruded_problem(
            ha_peak=ha_peak,
            ny=ny,
            nz=nz,
            nx_stations=nx_stations,
            wall_cells=max(1, min(3, ny // 8)),
            insulator_cells=max(1, min(2, ny // 10)),
        )
        return _replace_fields(
            problem,
            case=_replace_fields(
                problem.case,
                solver=_replace_fields(problem.case.solver, coupling_iterations=coupling_iterations),
                time_stepper=_replace_fields(problem.case.time_stepper, max_steps=max_steps, potential_iterations=potential_iterations),
            ),
        )
    if geometry_kind == "pipe_ogrid":
        problem = build_pipe_ogrid_extruded_problem(ha_peak=ha_peak, nr=ny, ntheta=nz, nx_stations=nx_stations)
        return _replace_fields(
            problem,
            case=_replace_fields(
                problem.case,
                solver=_replace_fields(problem.case.solver, coupling_iterations=coupling_iterations),
                time_stepper=_replace_fields(problem.case.time_stepper, max_steps=max_steps, potential_iterations=potential_iterations),
            ),
        )
    raise ValueError(f"Unsupported geometry {geometry_kind!r}")


def _row_for_solution(geometry_kind: str, solution, *, ha_peak: float, ny: int, nz: int, nx_stations: int) -> dict[str, float | str]:
    validation = solution.validation
    row: dict[str, float | str] = {
        "geometry_kind": geometry_kind,
        "ha_peak": float(ha_peak),
        "cross_section_y": float(ny),
        "cross_section_z": float(nz),
        "nx_stations": float(nx_stations),
        "station_count": float(validation.station_count),
        "max_residual": float(validation.max_residual),
        "max_charge_balance_residual": float(validation.max_charge_balance_residual),
        "volumetric_flow_rate_span": float(validation.volumetric_flow_rate_span),
        "axial_current_span": float(validation.axial_current_span),
        "axial_current_mirror_residual": float(getattr(validation, "axial_current_mirror_residual", 0.0)),
        "peak_velocity_span": float(validation.peak_velocity_span),
        "pressure_span_range": float(validation.pressure_span_range),
        "pressure_span_mirror_residual": float(getattr(validation, "pressure_span_mirror_residual", 0.0)),
        "center_axial_current": float(getattr(validation, "center_axial_current", 0.0)),
        "center_pressure_span": float(getattr(validation, "center_pressure_span", 0.0)),
        "max_wall_current_leakage": float(validation.max_wall_current_leakage),
        "net_boundary_current_residual": float(validation.net_boundary_current_residual),
        "field_mean_velocity_correlation": float(validation.field_mean_velocity_correlation),
    }
    return row


def _write_csv(rows: list[dict[str, float | str]], path: Path) -> Path:
    fieldnames: set[str] = set()
    for row in rows:
        fieldnames.update(row.keys())
    ordered = [
        "geometry_kind",
        "ha_peak",
        "cross_section_y",
        "cross_section_z",
        "nx_stations",
        "max_charge_balance_residual",
        "volumetric_flow_rate_span",
        "axial_current_span",
        "axial_current_mirror_residual",
        "pressure_span_range",
        "pressure_span_mirror_residual",
        "center_axial_current",
        "center_pressure_span",
        "field_mean_velocity_correlation",
        "center_velocity_l2_error",
        "negative_potential_l2_error",
        "positive_potential_l2_error",
    ]
    remaining = sorted(name for name in fieldnames if name not in ordered)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*ordered, *remaining], extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _write_markdown(rows: list[dict[str, float | str]], path: Path) -> Path:
    lines = [
        "# Benchmark B Quantitative Summary",
        "",
        "| Geometry | Ha | Resolution | Charge balance | Flow span | Axial span | Axial mirror | Pressure range | Pressure mirror | Correlation | Center L2 | Neg L2 | Pos L2 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        resolution = f"{int(float(row['cross_section_y']))}×{int(float(row['cross_section_z']))}×{int(float(row['nx_stations']))}"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["geometry_kind"]),
                    f"{float(row['ha_peak']):.0f}",
                    resolution,
                    f"{float(row['max_charge_balance_residual']):.3e}",
                    f"{float(row['volumetric_flow_rate_span']):.3e}",
                    f"{float(row['axial_current_span']):.3e}",
                    f"{float(row['axial_current_mirror_residual']):.3e}",
                    f"{float(row['pressure_span_range']):.3e}",
                    f"{float(row['pressure_span_mirror_residual']):.3e}",
                    f"{float(row['field_mean_velocity_correlation']):.3f}",
                    "-" if "center_velocity_l2_error" not in row else f"{float(row['center_velocity_l2_error']):.3f}",
                    "-" if "negative_potential_l2_error" not in row else f"{float(row['negative_potential_l2_error']):.3f}",
                    "-" if "positive_potential_l2_error" not in row else f"{float(row['positive_potential_l2_error']):.3f}",
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n")
    return path


def _write_plot(rows: list[dict[str, float | str]], path: Path) -> list[Path]:
    _set_style()
    fig, axes = plt.subplots(2, 2, constrained_layout=True)
    colors = {"rect_duct": "#0f766e", "layered_duct": "#b45309", "pipe_ogrid": "#1d4ed8"}
    labels = {"rect_duct": "Rectangular duct", "layered_duct": "Layered duct", "pipe_ogrid": "Mapped pipe"}
    geometries = [str(row["geometry_kind"]) for row in rows]
    x = np.arange(len(rows))

    axes[0, 0].bar(x, [float(row["max_charge_balance_residual"]) for row in rows], color=[colors[g] for g in geometries])
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Charge-balance residual")
    axes[0, 1].bar(x, [float(row["volumetric_flow_rate_span"]) for row in rows], color=[colors[g] for g in geometries])
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("Throughput span")
    axes[1, 0].bar(x, [float(row["pressure_span_range"]) for row in rows], color=[colors[g] for g in geometries])
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("Pressure-span range")
    axes[1, 1].bar(x, [float(row["axial_current_span"]) for row in rows], color=[colors[g] for g in geometries])
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("Axial-current span")
    for ax in axes.ravel():
        ax.set_xticks(x)
        ax.set_xticklabels([labels[g] for g in geometries], rotation=12, ha="right")
    png = path
    pdf = path.with_suffix(".pdf")
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a quantitative Benchmark B summary over duct and pipe fringing cases.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/validation/benchmark_b_quantitative"))
    parser.add_argument("--ha-peak", type=float, default=20.0)
    parser.add_argument("--duct-ny", type=int, default=20)
    parser.add_argument("--duct-nz", type=int, default=20)
    parser.add_argument("--pipe-nr", type=int, default=20)
    parser.add_argument("--pipe-ntheta", type=int, default=80)
    parser.add_argument("--nx-stations", type=int, default=21)
    parser.add_argument("--max-steps", type=int, default=24)
    parser.add_argument("--coupling-iterations", type=int, default=12)
    parser.add_argument("--potential-iterations", type=int, default=80)
    parser.add_argument("--reference-dir", type=Path, default=None)
    parser.add_argument("--include-pipe-reference", action="store_true")
    parser.add_argument(
        "--geometries",
        nargs="+",
        choices=("rect_duct", "layered_duct", "pipe_ogrid"),
        default=("rect_duct", "layered_duct", "pipe_ogrid"),
    )
    args = parser.parse_args(argv)
    enable_compilation_cache(JAX_CACHE_DIR)

    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | str]] = []
    geometry_map = {
        "rect_duct": (args.duct_ny, args.duct_nz),
        "layered_duct": (args.duct_ny, args.duct_nz),
        "pipe_ogrid": (args.pipe_nr, args.pipe_ntheta),
    }
    for geometry_kind in args.geometries:
        ny, nz = geometry_map[geometry_kind]
        problem = _build_problem(
            geometry_kind,
            ha_peak=args.ha_peak,
            ny=ny,
            nz=nz,
            nx_stations=args.nx_stations,
            max_steps=args.max_steps,
            coupling_iterations=args.coupling_iterations,
            potential_iterations=args.potential_iterations,
        )
        solution = solve_extruded_inductionless(problem)
        row = _row_for_solution(geometry_kind, solution, ha_peak=args.ha_peak, ny=ny, nz=nz, nx_stations=args.nx_stations)
        if geometry_kind == "pipe_ogrid" and args.include_pipe_reference:
            row.update(_pipe_profile_errors(solution.bundle, args.reference_dir))
        rows.append(row)

    rows.sort(key=lambda row: str(row["geometry_kind"]))
    json_path = args.output / "benchmark_b_quantitative_summary.json"
    csv_path = args.output / "benchmark_b_quantitative_summary.csv"
    md_path = args.output / "benchmark_b_quantitative_summary.md"
    plot_path = args.output / "benchmark_b_quantitative_summary.png"
    json_path.write_text(json.dumps(rows, indent=2) + "\n")
    _write_csv(rows, csv_path)
    _write_markdown(rows, md_path)
    plots = _write_plot(rows, plot_path)
    payload = {
        "rows": rows,
        "artifacts": {
            "json": str(json_path),
            "csv": str(csv_path),
            "markdown": str(md_path),
            "plots": [str(path) for path in plots],
        },
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lmx.cases import make_hartmann_case, make_hunt_case, make_shercliff_case
from lmx.fringing import (
    build_layered_duct_extruded_problem,
    build_pipe_ogrid_extruded_problem,
    build_square_duct_extruded_problem,
    solve_extruded_inductionless,
)
from lmx.solvers import solve_steady
from lmx.validation import (
    closed_channel_validation,
    combined_profile_error,
    duct_layer_resolution_metrics,
    hartmann_acceptance,
    validation_summary,
)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting is optional in the script.
    plt = None


def _cases(ha: float, resolution: int):
    return [
        make_hartmann_case(ha=ha, ny=resolution, nz=resolution),
        make_shercliff_case(ha=ha, ny=resolution, nz=resolution),
        make_hunt_case(ha=ha, ny=resolution, nz=resolution, wall_cells=2),
    ]


def _bounded_case(case, *, max_steps: int, potential_iterations: int, coupling_iterations: int):
    return replace(
        case,
        time_stepper=replace(
            case.time_stepper,
            max_steps=min(case.time_stepper.max_steps, max_steps),
            potential_iterations=min(case.time_stepper.potential_iterations, potential_iterations),
        ),
        solver=replace(case.solver, coupling_iterations=min(case.solver.coupling_iterations, coupling_iterations)),
    )


def _parse_int_list(text: str | None, *, default: list[int]) -> list[int]:
    if text is None or not str(text).strip():
        return default
    return [int(item.strip()) for item in str(text).split(",") if item.strip()]


def _summary_key(name: str, resolution: int, *, multi_resolution: bool) -> str:
    return f"{name}_n{resolution}" if multi_resolution else name


def _write_summary_csv(payload: dict[str, dict[str, float | str]], path: Path) -> Path:
    rows = []
    fieldnames: set[str] = {"key"}
    for key, value in payload.items():
        row = {"key": key, **value}
        rows.append(row)
        fieldnames.update(row.keys())
    ordered = ["key", "case", "geometry_kind", "solver_kind", "ha", "resolution"]
    remaining = sorted(name for name in fieldnames if name not in ordered)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*ordered, *remaining], extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _write_summary_plot(payload: dict[str, dict[str, float | str]], path: Path) -> Path | None:
    if plt is None:
        return None
    fringing_rows = [
        row
        for row in payload.values()
        if str(row.get("geometry_kind", "")).startswith("fringing_") or str(row.get("solver_kind", "")) == "extruded_inductionless"
    ]
    if not fringing_rows:
        return None
    grouped: dict[tuple[str, float], list[dict[str, float | str]]] = {}
    for row in fringing_rows:
        grouped.setdefault((str(row["geometry_kind"]), float(row.get("ha", 0.0))), []).append(row)
    plt.style.use("default")
    plt.rcParams.update({"figure.dpi": 160, "savefig.dpi": 300, "font.family": "STIXGeneral", "mathtext.fontset": "stix"})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.4), constrained_layout=True)
    palette = ["#0f766e", "#1d4ed8", "#b45309", "#7c3aed", "#dc2626"]
    line_styles = {10.0: "-", 20.0: "--", 30.0: ":", 40.0: "-."}
    for color, ((geometry_kind, ha_value), rows) in zip(palette, sorted(grouped.items()), strict=False):
        rows = sorted(rows, key=lambda row: (float(row.get("ha", 0.0)), float(row.get("resolution", 0.0))))
        resolutions = [float(row.get("resolution", 0.0)) for row in rows]
        charge = [max(float(row.get("max_charge_balance_residual", 0.0)), 1.0e-16) for row in rows]
        axial_current_span = [max(float(row.get("axial_current_span", 0.0)), 1.0e-16) for row in rows]
        peak_velocity_span = [max(float(row.get("peak_velocity_span", 0.0)), 1.0e-16) for row in rows]
        pressure_span_range = [max(float(row.get("pressure_span_range", 0.0)), 1.0e-16) for row in rows]
        label = f"{geometry_kind}, Ha={int(ha_value)}"
        style = line_styles.get(ha_value, "-")
        axes[0, 0].semilogy(resolutions, charge, marker="o", color=color, linestyle=style, label=label)
        axes[0, 1].semilogy(resolutions, axial_current_span, marker="o", color=color, linestyle=style, label=label)
        axes[1, 0].semilogy(resolutions, peak_velocity_span, marker="o", color=color, linestyle=style, label=label)
        axes[1, 1].semilogy(resolutions, pressure_span_range, marker="o", color=color, linestyle=style, label=label)
    axes[0, 0].set_title("Fringing charge-balance residual")
    axes[0, 0].set_xlabel("Cross-section resolution")
    axes[0, 0].set_ylabel(r"$\max |\nabla \cdot J|$")
    axes[0, 1].set_title("Axial-current span")
    axes[0, 1].set_xlabel("Cross-section resolution")
    axes[0, 1].set_ylabel(r"$\Delta \int J_x\,dA$")
    axes[1, 0].set_title("Peak-velocity span")
    axes[1, 0].set_xlabel("Cross-section resolution")
    axes[1, 0].set_ylabel(r"$\Delta u_{max}$")
    axes[1, 1].set_title("Pressure-span range")
    axes[1, 1].set_xlabel("Cross-section resolution")
    axes[1, 1].set_ylabel(r"$\Delta (\max p - \min p)$")
    axes[0, 0].legend(frameon=False, ncols=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    pdf_path = path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the heavier manual LMX solver-family validation lane.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/manual_validation/solver_family_summary.json"))
    parser.add_argument("--ha-values", type=str, default="10,20")
    parser.add_argument("--resolution", type=int, default=24)
    parser.add_argument("--resolutions", type=str, default="")
    parser.add_argument("--reference-root", type=Path, default=None)
    parser.add_argument("--hartmann-l2-threshold", type=float, default=0.05)
    parser.add_argument("--hartmann-linf-threshold", type=float, default=0.1)
    parser.add_argument("--include-fringing", action="store_true")
    parser.add_argument("--fringing-geometries", type=str, default="rect_duct,layered_duct,pipe_ogrid")
    parser.add_argument("--fringing-nx", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--potential-iterations", type=int, default=80)
    parser.add_argument("--coupling-iterations", type=int, default=8)
    parser.add_argument("--max-charge-balance", type=float, default=8.0e-1)
    parser.add_argument("--max-interface-current", type=float, default=2.5e-1)
    parser.add_argument("--max-fringing-wall-current-leakage", type=float, default=1.0e-1)
    parser.add_argument("--max-fringing-boundary-current", type=float, default=1.0e-5)
    parser.add_argument("--write-csv", action="store_true")
    parser.add_argument("--write-plot", action="store_true")
    parser.add_argument("--fail-on-threshold", action="store_true")
    args = parser.parse_args(argv)

    ha_values = [float(item) for item in args.ha_values.split(",") if item]
    resolutions = _parse_int_list(args.resolutions, default=[args.resolution])
    multi_resolution = len(resolutions) > 1
    summary: dict[str, dict[str, float | str]] = {}
    failures: list[str] = []

    for resolution in resolutions:
        for ha in ha_values:
            for case in _cases(ha, resolution):
                case = _bounded_case(
                    case,
                    max_steps=args.max_steps,
                    potential_iterations=args.potential_iterations,
                    coupling_iterations=args.coupling_iterations,
                )
                solution = solve_steady(case)
                metrics = validation_summary(solution, case.name, ha=ha)
                metrics.update(duct_layer_resolution_metrics(case, solution.mesh))
                metrics["ha"] = float(ha)
                metrics["resolution"] = float(resolution)
                metrics["case"] = case.name
                metrics["geometry_kind"] = case.geometry.kind
                metrics["solver_kind"] = case.solver.kind
                if case.name.startswith("hartmann"):
                    acceptance = hartmann_acceptance(
                        solution,
                        ha,
                        l2_threshold=args.hartmann_l2_threshold,
                        linf_threshold=args.hartmann_linf_threshold,
                    )
                    metrics["accepted"] = float(acceptance.passed)
                    metrics["acceptance_l2_error"] = acceptance.l2_error
                    metrics["acceptance_linf_error"] = acceptance.linf_error
                elif args.reference_root is not None:
                    comparison = closed_channel_validation(solution, case.name.split("_", 1)[0], int(ha), reference_root=args.reference_root)
                    metrics["combined_l2_error"] = combined_profile_error(
                        comparison.y_profile.l2_error,
                        comparison.z_profile.l2_error,
                    )
                    metrics["combined_linf_error"] = combined_profile_error(
                        comparison.y_profile.linf_error,
                        comparison.z_profile.linf_error,
                    )
                conservation_pass = (
                    float(metrics.get("charge_balance_residual", 0.0)) <= args.max_charge_balance
                    and float(metrics.get("interface_current_residual", 0.0)) <= args.max_interface_current
                )
                metrics["conservation_pass"] = float(conservation_pass)
                metrics["charge_balance_threshold"] = float(args.max_charge_balance)
                metrics["interface_current_threshold"] = float(args.max_interface_current)
                if not conservation_pass:
                    failures.append(_summary_key(case.name, resolution, multi_resolution=multi_resolution))
                summary[_summary_key(case.name, resolution, multi_resolution=multi_resolution)] = metrics

    if args.include_fringing:
        geometry_kinds = [item.strip() for item in args.fringing_geometries.split(",") if item.strip()]
        for resolution in resolutions:
            cross_section_resolution = max(6, resolution // 2)
            for ha in ha_values:
                for geometry_kind in geometry_kinds:
                    if geometry_kind == "rect_duct":
                        problem = build_square_duct_extruded_problem(
                            ha_peak=ha,
                            ny=cross_section_resolution,
                            nz=cross_section_resolution,
                            nx_stations=args.fringing_nx,
                        )
                    elif geometry_kind == "layered_duct":
                        problem = build_layered_duct_extruded_problem(
                            ha_peak=ha,
                            ny=cross_section_resolution,
                            nz=cross_section_resolution,
                            wall_cells=1,
                            insulator_cells=1,
                            nx_stations=args.fringing_nx,
                        )
                    elif geometry_kind == "pipe_ogrid":
                        problem = build_pipe_ogrid_extruded_problem(
                            ha_peak=ha,
                            nr=cross_section_resolution,
                            ntheta=max(12, resolution),
                            nx_stations=args.fringing_nx,
                        )
                    else:
                        raise ValueError(f"Unsupported fringing geometry {geometry_kind!r}")
                    problem = replace(
                        problem,
                        case=_bounded_case(
                            problem.case,
                            max_steps=args.max_steps,
                            potential_iterations=args.potential_iterations,
                            coupling_iterations=args.coupling_iterations,
                        ),
                    )
                    solution = solve_extruded_inductionless(problem)
                    key = _summary_key(f"fringing_{geometry_kind}_ha{int(ha)}", resolution, multi_resolution=multi_resolution)
                    summary[key] = {
                        "case": problem.case.name,
                        "geometry_kind": geometry_kind,
                        "solver_kind": problem.case.solver.kind,
                        "ha": float(ha),
                        "resolution": float(resolution),
                        "station_count": float(solution.validation.station_count),
                        "max_residual": solution.validation.max_residual,
                        "max_charge_balance_residual": solution.validation.max_charge_balance_residual,
                        "mean_velocity_span": solution.validation.mean_velocity_span,
                        "volumetric_flow_rate_span": solution.validation.volumetric_flow_rate_span,
                        "axial_current_span": solution.validation.axial_current_span,
                        "peak_velocity_span": solution.validation.peak_velocity_span,
                        "pressure_span_range": solution.validation.pressure_span_range,
                        "max_wall_current_leakage": solution.validation.max_wall_current_leakage,
                        "net_boundary_current_residual": solution.validation.net_boundary_current_residual,
                        "field_mean_velocity_correlation": solution.validation.field_mean_velocity_correlation,
                        "conservation_pass": float(
                            solution.validation.max_charge_balance_residual <= args.max_charge_balance
                            and solution.validation.max_wall_current_leakage <= args.max_fringing_wall_current_leakage
                            and solution.validation.net_boundary_current_residual <= args.max_fringing_boundary_current
                        ),
                        "charge_balance_threshold": float(args.max_charge_balance),
                        "wall_current_leakage_threshold": float(args.max_fringing_wall_current_leakage),
                        "boundary_current_threshold": float(args.max_fringing_boundary_current),
                    }
                    if not bool(summary[key]["conservation_pass"]):
                        failures.append(key)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    if args.write_csv:
        _write_summary_csv(summary, args.output.with_suffix(".csv"))
    if args.write_plot:
        _write_summary_plot(summary, args.output.with_name(f"{args.output.stem}_fringing.png"))
    print(args.output.read_text())
    if failures and args.fail_on_threshold:
        print(f"Conservation thresholds failed for: {', '.join(failures)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

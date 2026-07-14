from __future__ import annotations

import argparse
import json
import sys
from dataclasses import is_dataclass, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lmx.cases import make_hartmann_case, make_hunt_case, make_shercliff_case
from lmx.solvers import _bounded_time_step_count, solve_steady
from lmx.validation import (
    closed_channel_validation,
    combined_profile_error,
    duct_layer_resolution_metrics,
    estimate_observed_order,
    hartmann_acceptance,
    hartmann_validation,
    processed_slice_validation,
    validation_summary,
)


def _parse_csv_numbers(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_csv_floats(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def _hunt_wall_cells(resolution: int) -> int:
    return max(2, round(resolution * 8 / 72))


def _replace_like(obj, **changes):
    if is_dataclass(obj):
        return replace(obj, **changes)
    if hasattr(obj, "__dict__"):
        return obj.__class__(**{**vars(obj), **changes})
    raise TypeError(f"Unsupported object replacement for {type(obj)!r}")


def _build_case(case_kind: str, ha: float, resolution: int, output_dir: Path):
    if case_kind == "hartmann":
        return make_hartmann_case(ha=ha, ny=resolution, nz=resolution, output_dir=str(output_dir))
    if case_kind == "shercliff":
        return make_shercliff_case(ha=ha, ny=resolution, nz=resolution, output_dir=str(output_dir))
    if case_kind == "hunt":
        return make_hunt_case(
            ha=ha,
            ny=resolution,
            nz=resolution,
            wall_cells=_hunt_wall_cells(resolution),
            output_dir=str(output_dir),
        )
    raise ValueError(case_kind)


def _mesh_spacing(case) -> float:
    return max(case.geometry.width / max(case.geometry.ny, 1), case.geometry.height / max(case.geometry.nz, 1))


def _collect_metrics(
    solution,
    case_kind: str,
    ha: float,
    *,
    reference_root: Path | None,
    x_slice: str,
    hartmann_l2_threshold: float,
    hartmann_linf_threshold: float,
) -> dict[str, float | str]:
    case_name = f"{case_kind}_ha{int(ha)}"
    metrics = validation_summary(solution, case_name, ha=ha)
    if case_kind == "hartmann":
        comparison = hartmann_validation(solution, ha)
        acceptance = hartmann_acceptance(
            solution,
            ha,
            l2_threshold=hartmann_l2_threshold,
            linf_threshold=hartmann_linf_threshold,
        )
        metrics["l2_error"] = comparison.l2_error
        metrics["linf_error"] = comparison.linf_error
        metrics["accepted"] = float(acceptance.passed)
        metrics["acceptance_l2_threshold"] = acceptance.l2_threshold
        metrics["acceptance_linf_threshold"] = acceptance.linf_threshold
    elif reference_root is not None:
        comparison = closed_channel_validation(solution, case_kind, int(ha), reference_root=reference_root)
        metrics["y_l2_error"] = comparison.y_profile.l2_error
        metrics["y_linf_error"] = comparison.y_profile.linf_error
        metrics["z_l2_error"] = comparison.z_profile.l2_error
        metrics["z_linf_error"] = comparison.z_profile.linf_error
        metrics["combined_l2_error"] = combined_profile_error(
            comparison.y_profile.l2_error,
            comparison.z_profile.l2_error,
        )
        metrics["combined_linf_error"] = combined_profile_error(
            comparison.y_profile.linf_error,
            comparison.z_profile.linf_error,
        )
        try:
            slice_report = processed_slice_validation(
                solution,
                case_kind,
                int(ha),
                x_slice=x_slice,
                reference_root=reference_root,
            )
        except FileNotFoundError:
            slice_report = None
        if slice_report is not None:
            metrics["slice_y_l2_error"] = slice_report.y_profile.l2_error
            metrics["slice_y_linf_error"] = slice_report.y_profile.linf_error
            metrics["slice_z_l2_error"] = slice_report.z_profile.l2_error
            metrics["slice_z_linf_error"] = slice_report.z_profile.linf_error
            metrics["slice_combined_l2_error"] = combined_profile_error(
                slice_report.y_profile.l2_error,
                slice_report.z_profile.l2_error,
            )
            metrics["slice_combined_linf_error"] = combined_profile_error(
                slice_report.y_profile.linf_error,
                slice_report.z_profile.linf_error,
            )
    return metrics


def _observed_orders(
    levels: list[dict[str, float | str]], *, scale_key: str = "mesh_spacing"
) -> dict[str, list[dict[str, float]]]:
    orders: dict[str, list[dict[str, float]]] = {}
    if len(levels) < 2:
        return orders
    numeric_keys = {
        key
        for level in levels
        for key, value in level.items()
        if key.endswith("_error") and isinstance(value, (int, float))
    }
    for key in sorted(numeric_keys):
        entries: list[dict[str, float]] = []
        for coarse, fine in zip(levels[:-1], levels[1:]):
            if key not in coarse or key not in fine:
                continue
            order = estimate_observed_order(
                float(coarse[key]),
                float(fine[key]),
                float(coarse[scale_key]),
                float(fine[scale_key]),
            )
            if order is None:
                continue
            label = "dt" if scale_key == "dt" else "resolution"
            entries.append({
                f"coarse_{label}": float(coarse[label]),
                f"fine_{label}": float(fine[label]),
                "order": order,
            })
        if entries:
            orders[key] = entries
    return orders


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run native LMX convergence studies.")
    parser.add_argument("--mode", choices=("mesh", "time"), default="mesh")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cases", type=str, default="hartmann,shercliff,hunt")
    parser.add_argument("--ha", type=float, default=20.0)
    parser.add_argument("--resolutions", type=str, default="16,32,48")
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--dts", type=str, default="0.002,0.001,0.0005")
    parser.add_argument("--t-final", type=float)
    parser.add_argument("--reference-root", type=Path, default=None)
    parser.add_argument("--x-slice", type=str, default="1m")
    parser.add_argument("--hartmann-l2-threshold", type=float, default=0.05)
    parser.add_argument("--hartmann-linf-threshold", type=float, default=0.1)
    args = parser.parse_args(argv)

    output = args.output or Path(f"artifacts/{'time_' if args.mode == 'time' else ''}convergence")
    output.mkdir(parents=True, exist_ok=True)
    cases = [case.strip() for case in args.cases.split(",") if case.strip()]
    payload: dict[str, object] = {"mode": args.mode, "ha": args.ha, "cases": {}}

    for case_kind in cases:
        levels: list[dict[str, float | str]] = []
        scales = (
            _parse_csv_numbers(args.resolutions)
            if args.mode == "mesh"
            else _parse_csv_floats(args.dts)
        )
        for scale in scales:
            resolution = int(scale) if args.mode == "mesh" else args.resolution
            case_dir = output / case_kind / (
                f"n{resolution}" if args.mode == "mesh" else f"dt{scale:g}"
            )
            case = _build_case(case_kind, args.ha, resolution, case_dir)
            if args.mode == "time":
                t_final = (
                    case.time_stepper.t_final
                    if args.t_final is None
                    else float(args.t_final)
                )
                max_steps = _bounded_time_step_count(
                    start_time=0.0,
                    dt=float(scale),
                    t_final=float(t_final),
                    max_steps=case.time_stepper.max_steps,
                )
                case = _replace_like(
                    case,
                    time_stepper=_replace_like(
                        case.time_stepper,
                        dt=float(scale),
                        t_final=float(t_final),
                        max_steps=max_steps,
                    ),
                )
            solution = solve_steady(case)
            metrics = _collect_metrics(
                solution,
                case_kind,
                args.ha,
                reference_root=args.reference_root,
                x_slice=args.x_slice,
                hartmann_l2_threshold=args.hartmann_l2_threshold,
                hartmann_linf_threshold=args.hartmann_linf_threshold,
            )
            common = {
                "case": case.name,
                "dt": case.time_stepper.dt,
                "max_steps": float(case.time_stepper.max_steps),
                **duct_layer_resolution_metrics(case, solution.mesh),
                **metrics,
            }
            level = (
                {
                    **common,
                    "resolution": float(resolution),
                    "mesh_spacing": _mesh_spacing(case),
                }
                if args.mode == "mesh"
                else {**common, "dt": float(scale), "t_final": float(t_final)}
            )
            levels.append(level)
        scale_name = "resolutions" if args.mode == "mesh" else "dts"
        payload["cases"][case_kind] = {
            scale_name: scales,
            "levels": levels,
            "observed_orders": _observed_orders(
                levels, scale_key="mesh_spacing" if args.mode == "mesh" else "dt"
            ),
        }

    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2))
    print(summary_path.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

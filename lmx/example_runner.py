from __future__ import annotations

import json
from pathlib import Path

from .cases import make_hartmann_case, make_hunt_case, make_shercliff_case
from .io import write_paraview
from .plotting import write_case_overview_plots
from .reference_data import default_closed_channel_reference_root
from .solvers import solve_steady
from .validation import (
    closed_channel_validation,
    extract_centerline,
    extract_midplane_profile,
    hartmann_validation,
    validation_summary,
    write_metrics_json,
    write_profile_csv,
)


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
        comparison = closed_channel_validation(solution, case_kind, int(ha), reference_root=reference_root)
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
        "output_dir": str(out_dir.resolve()),
        "plots": [str(path.resolve()) for path in plot_paths],
        "reference": reference_payload,
        "metrics": metrics,
    }
    write_metrics_json(report, out_dir / "example_report.json")
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

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import jax.numpy as jnp

from lmx.cases import make_hartmann_case, make_hunt_case
from lmx.plotting import write_case_overview_plots, write_geometry_preview_plots
from lmx.solvers import _build_mesh, solve_steady
from lmx.specs import MagneticFieldSpec
from lmx.validation import validation_summary


def _analytic_cross_section_field(width: float, height: float, base_bz: float):
    def field(y: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
        y_hat = 2.0 * y / width
        z_hat = 2.0 * z / height
        bx = jnp.zeros_like(y)
        by = 0.08 * base_bz * jnp.sin(jnp.pi * z_hat)
        bz = base_bz * (
            1.0 + 0.15 * jnp.cos(jnp.pi * y_hat) * jnp.cos(0.5 * jnp.pi * z_hat)
        )
        return jnp.stack([bx, by, bz], axis=-1)

    return field


def build_variable_field_rectangular_case():
    case = make_hartmann_case(ha=18.0, width=2.4, height=1.6, ny=40, nz=32)
    return replace(
        case,
        name="variable_field_rectangular",
        geometry=replace(case.geometry, width=2.4, height=1.6, ny=40, nz=32),
        magnetic_field=MagneticFieldSpec(
            kind="analytic",
            fn=_analytic_cross_section_field(2.4, 1.6, base_bz=12.0),
            ramp_start=0.0,
            ramp_duration=0.05,
        ),
        notes=(
            "Custom rectangular duct with an analytic cross-sectional magnetic field. "
            "Built directly from Python to show variable-field workflows."
        ),
    )


def build_variable_geometry_layered_case():
    case = make_hunt_case(
        ha=16.0,
        width=2.6,
        height=1.4,
        ny=28,
        nz=24,
        wall_cells=4,
        wall_thickness=0.12,
        insulator_cells=3,
        insulator_thickness=0.08,
    )
    return replace(
        case,
        name="variable_geometry_layered",
        magnetic_field=MagneticFieldSpec(
            kind="analytic",
            fn=_analytic_cross_section_field(2.6, 1.4, base_bz=9.0),
            ramp_start=0.0,
            ramp_duration=0.05,
        ),
        notes=(
            "Custom layered duct with explicit side and Hartmann wall regions and "
            "an analytic variable magnetic field."
        ),
    )


def run_variable_field_geometry_demo(*, out_dir: Path) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rect_case = build_variable_field_rectangular_case()
    layered_case = build_variable_geometry_layered_case()

    rect_preview = write_geometry_preview_plots(
        _build_mesh(rect_case),
        out_dir / "rectangular_preview",
        case_title="Variable-field rectangular duct",
    )
    layered_preview = write_geometry_preview_plots(
        _build_mesh(layered_case),
        out_dir / "layered_preview",
        case_title="Variable-field layered duct",
    )

    solution = solve_steady(rect_case)
    plots = write_case_overview_plots(
        solution,
        out_dir / "rectangular_run",
        case_title="Variable-field rectangular duct",
    )
    metrics = validation_summary(
        solution, rect_case.name, ha=rect_case.geometry.target_ha
    )
    summary = {
        "rectangular_case": {
            "name": rect_case.name,
            "geometry_kind": rect_case.geometry.kind,
            "magnetic_field_kind": rect_case.magnetic_field.kind,
            "preview": [path.name for path in rect_preview],
            "run_plots": [path.name for path in plots],
            "metrics": metrics,
        },
        "layered_case": {
            "name": layered_case.name,
            "geometry_kind": layered_case.geometry.kind,
            "magnetic_field_kind": layered_case.magnetic_field.kind,
            "preview": [path.name for path in layered_preview],
        },
    }
    (out_dir / "variable_field_geometry_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the LMX variable magnetic-field and geometry demo."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/examples/variable_field_geometry"),
    )
    args = parser.parse_args(argv)
    run_variable_field_geometry_demo(out_dir=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

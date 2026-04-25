from __future__ import annotations

import json
from pathlib import Path

from lmx import enable_compilation_cache
from lmx.showcase import solve_closed_channel_benchmark, write_closed_channel_validation_ladder_figure
from lmx.solvers import _build_mesh
from lmx.validation import duct_layer_resolution_gate


OUTPUT_DIR = Path("artifacts/examples/straight_duct_validation_ladder")
JAX_CACHE_DIR = Path("artifacts/jax_cache")
HA_VALUES = (20.0, 100.0)
WIDTH = 0.2
HEIGHT = 0.2
SHERCLIFF_NY = 45
SHERCLIFF_NZ = 45
HUNT_NY = 49
HUNT_NZ = 49
WALL_CELLS = 2
WALL_THICKNESS = 0.001
COUPLING_ITERATIONS = 16
POTENTIAL_ITERATIONS = 200
MAX_STEPS = 96
VELOCITY_UPDATE_LIMIT = 5.0e-2
POTENTIAL_TOLERANCE = 1.0e-8

FLUID_CONDUCTIVITY = 1.0
CONDUCTING_WALL_CONDUCTIVITY = 5.0
INSULATING_WALL_CONDUCTIVITY = 1.0e-12
DENSITY = 1.0
VISCOSITY = 1.0


def _layer_resolution_summary(case: object) -> dict[str, object]:
    try:
        return dict(duct_layer_resolution_gate(case, _build_mesh(case)))  # type: ignore[arg-type]
    except Exception:
        return {
            "layer_resolution_supported": False,
            "layer_resolution_pass": False,
        }


def run_straight_duct_validation_ladder(
    *,
    out_dir: Path = OUTPUT_DIR,
) -> dict[str, object]:
    enable_compilation_cache(JAX_CACHE_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    shercliff_records: list[dict[str, object]] = []
    hunt_records: list[dict[str, object]] = []
    for ha in HA_VALUES:
        shercliff_case, _, shercliff_comparison = solve_closed_channel_benchmark(
            "shercliff",
            ha=ha,
            width=WIDTH,
            height=HEIGHT,
            ny=SHERCLIFF_NY,
            nz=SHERCLIFF_NZ,
            fluid_conductivity=FLUID_CONDUCTIVITY,
            density=DENSITY,
            viscosity=VISCOSITY,
            coupling_iterations=COUPLING_ITERATIONS,
            potential_iterations=POTENTIAL_ITERATIONS,
            max_steps=MAX_STEPS,
            velocity_update_limit=VELOCITY_UPDATE_LIMIT,
            current_reconstruction="face_averaged",
            potential_tolerance=POTENTIAL_TOLERANCE,
            initial_profile="zero",
        )
        shercliff_records.append(
            {
                "ha": ha,
                "y_profile": shercliff_comparison.y_profile,
                "z_profile": shercliff_comparison.z_profile,
                "reference_path": getattr(shercliff_comparison, "reference_path", ""),
                "layer_resolution": _layer_resolution_summary(shercliff_case),
            }
        )

        hunt_case, _, hunt_comparison = solve_closed_channel_benchmark(
            "hunt",
            ha=ha,
            width=WIDTH,
            height=HEIGHT,
            ny=HUNT_NY,
            nz=HUNT_NZ,
            wall_cells=WALL_CELLS,
            wall_thickness=WALL_THICKNESS,
            fluid_conductivity=FLUID_CONDUCTIVITY,
            density=DENSITY,
            viscosity=VISCOSITY,
            conducting_wall_conductivity=CONDUCTING_WALL_CONDUCTIVITY,
            insulating_wall_conductivity=INSULATING_WALL_CONDUCTIVITY,
            coupling_iterations=COUPLING_ITERATIONS,
            potential_iterations=POTENTIAL_ITERATIONS,
            max_steps=MAX_STEPS,
            velocity_update_limit=VELOCITY_UPDATE_LIMIT,
            current_reconstruction="face_averaged",
            potential_tolerance=POTENTIAL_TOLERANCE,
            initial_profile="zero",
        )
        hunt_records.append(
            {
                "ha": ha,
                "y_profile": hunt_comparison.y_profile,
                "z_profile": hunt_comparison.z_profile,
                "reference_path": getattr(hunt_comparison, "reference_path", ""),
                "layer_resolution": _layer_resolution_summary(hunt_case),
            }
        )

    outputs = write_closed_channel_validation_ladder_figure(
        out_dir,
        shercliff_records=shercliff_records,
        hunt_records=hunt_records,
    )

    def _serialize(records: list[dict[str, object]]) -> list[dict[str, object]]:
        payload: list[dict[str, object]] = []
        for record in records:
            y_comp = record["y_profile"]
            z_comp = record["z_profile"]
            payload.append(
                {
                    "ha": record["ha"],
                    "reference_path": record["reference_path"],
                    "y_l2_error": y_comp.l2_error,
                    "y_linf_error": y_comp.linf_error,
                    "z_l2_error": z_comp.l2_error,
                    "z_linf_error": z_comp.linf_error,
                    "layer_resolution": record["layer_resolution"],
                }
            )
        return payload

    summary = {
        "case": "straight_duct_validation_ladder",
        "ha_values": list(HA_VALUES),
        "shercliff": _serialize(shercliff_records),
        "hunt": _serialize(hunt_records),
        "outputs": [path.name for path in outputs],
        "comparison_method": "normalized analytical comparison with no-slip wall reconstruction at the duct boundaries",
        "initial_profile": "zero",
        "mesh": {
            "shercliff": {"ny": SHERCLIFF_NY, "nz": SHERCLIFF_NZ},
            "hunt": {"ny": HUNT_NY, "nz": HUNT_NZ},
        },
        "hunt_wall_model": {
            "wall_thickness": WALL_THICKNESS,
            "conducting_wall_conductivity": CONDUCTING_WALL_CONDUCTIVITY,
            "fluid_conductivity": FLUID_CONDUCTIVITY,
            "conductance_ratio": CONDUCTING_WALL_CONDUCTIVITY * WALL_THICKNESS / (FLUID_CONDUCTIVITY * (0.5 * HEIGHT)),
        },
    }
    (out_dir / "straight_duct_validation_ladder_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    run_straight_duct_validation_ladder()

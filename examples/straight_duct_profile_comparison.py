from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

from lmx import enable_compilation_cache, solve_closed_channel_benchmark, write_closed_channel_profile_comparison_figure
from lmx.cases import make_hartmann_case
from lmx.solvers import solve_steady
from lmx.validation import hartmann_validation


OUTPUT_DIR = Path("artifacts/examples/straight_duct_profile_comparison")
JAX_CACHE_DIR = Path("artifacts/jax_cache")
HA = 20.0
WIDTH = 0.2
HEIGHT = 0.2
NY = 37
NZ = 37
WALL_CELLS = 6
WALL_THICKNESS = 0.02
COUPLING_ITERATIONS = 16
POTENTIAL_ITERATIONS = 240
MAX_STEPS = 96
VELOCITY_UPDATE_LIMIT = 5.0e-2
POTENTIAL_TOLERANCE = 1.0e-8

FLUID_CONDUCTIVITY = 1.0
CONDUCTING_WALL_CONDUCTIVITY = 0.25
INSULATING_WALL_CONDUCTIVITY = 1.0e-12
DENSITY = 1.0
VISCOSITY = 1.0


def run_straight_duct_profile_comparison(
    *,
    out_dir: Path = OUTPUT_DIR,
) -> dict[str, object]:
    enable_compilation_cache(JAX_CACHE_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    hartmann_case = make_hartmann_case(
        ha=HA,
        width=WIDTH,
        height=HEIGHT,
        ny=NY,
        nz=NZ,
        conductivity=FLUID_CONDUCTIVITY,
        density=DENSITY,
        viscosity=VISCOSITY,
    )
    hartmann_case = replace(
        hartmann_case,
        time_stepper=replace(
            hartmann_case.time_stepper,
            potential_iterations=POTENTIAL_ITERATIONS,
            max_steps=MAX_STEPS,
            potential_tolerance=POTENTIAL_TOLERANCE,
            steady_potential_tolerance=POTENTIAL_TOLERANCE,
            velocity_update_limit=VELOCITY_UPDATE_LIMIT,
            current_reconstruction="face_averaged",
        ),
        solver=replace(hartmann_case.solver, coupling_iterations=COUPLING_ITERATIONS),
    )
    hartmann_solution = solve_steady(hartmann_case)
    hartmann_comparison = hartmann_validation(hartmann_solution, ha=HA)

    _, shercliff_solution, shercliff_comparison = solve_closed_channel_benchmark(
        "shercliff",
        ha=HA,
        width=WIDTH,
        height=HEIGHT,
        ny=NY,
        nz=NZ,
        fluid_conductivity=FLUID_CONDUCTIVITY,
        density=DENSITY,
        viscosity=VISCOSITY,
        coupling_iterations=COUPLING_ITERATIONS,
        potential_iterations=POTENTIAL_ITERATIONS,
        max_steps=MAX_STEPS,
        velocity_update_limit=VELOCITY_UPDATE_LIMIT,
        current_reconstruction="face_averaged",
        potential_tolerance=POTENTIAL_TOLERANCE,
    )
    _, hunt_solution, hunt_comparison = solve_closed_channel_benchmark(
        "hunt",
        ha=HA,
        width=WIDTH,
        height=HEIGHT,
        ny=NY,
        nz=NZ,
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
    )

    comparison_outputs = write_closed_channel_profile_comparison_figure(
        out_dir,
        hartmann_solution=hartmann_solution,
        shercliff_solution=shercliff_solution,
        hunt_solution=hunt_solution,
        ha=HA,
    )

    summary = {
        "case": "straight_duct_profile_comparison",
        "ha": HA,
        "hartmann": {
            "l2_error": hartmann_comparison.l2_error,
            "linf_error": hartmann_comparison.linf_error,
        },
        "shercliff": {
            "y_l2_error": shercliff_comparison.y_profile.l2_error,
            "y_linf_error": shercliff_comparison.y_profile.linf_error,
            "z_l2_error": shercliff_comparison.z_profile.l2_error,
            "z_linf_error": shercliff_comparison.z_profile.linf_error,
        },
        "hunt": {
            "y_l2_error": hunt_comparison.y_profile.l2_error,
            "y_linf_error": hunt_comparison.y_profile.linf_error,
            "z_l2_error": hunt_comparison.z_profile.l2_error,
            "z_linf_error": hunt_comparison.z_profile.linf_error,
        },
        "outputs": [path.name for path in comparison_outputs],
        "comparison_method": "normalized analytical comparison with no-slip wall reconstruction at the duct boundaries",
    }
    (out_dir / "straight_duct_profile_comparison_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    run_straight_duct_profile_comparison()

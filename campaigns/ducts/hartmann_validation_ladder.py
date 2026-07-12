from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from lmx import enable_compilation_cache
from lmx.showcase import write_hartmann_validation_ladder_figure
from lmx.cases import make_hartmann_case
from lmx.solvers import solve_steady
from lmx.validation import hartmann_validation


OUTPUT_DIR = Path("artifacts/examples/hartmann_validation_ladder")
JAX_CACHE_DIR = Path("artifacts/jax_cache")
HA_VALUES = (20.0, 100.0)
WIDTH = 0.2
HEIGHT = 0.2
NY = 25
NZ = 25
COUPLING_ITERATIONS = 16
POTENTIAL_ITERATIONS = 240
MAX_STEPS = 240
VELOCITY_UPDATE_LIMIT = 1.0e-4
POTENTIAL_TOLERANCE = 1.0e-8

FLUID_CONDUCTIVITY = 1.0
DENSITY = 1.0
VISCOSITY = 1.0
L2_TARGET = 1.2e-2


def run_hartmann_validation_ladder(
    *,
    out_dir: Path = OUTPUT_DIR,
) -> dict[str, object]:
    enable_compilation_cache(JAX_CACHE_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    hartmann_records: list[dict[str, object]] = []
    for ha in HA_VALUES:
        case = make_hartmann_case(
            ha=ha,
            width=WIDTH,
            height=HEIGHT,
            ny=NY,
            nz=NZ,
            conductivity=FLUID_CONDUCTIVITY,
            density=DENSITY,
            viscosity=VISCOSITY,
        )
        case = replace(
            case,
            solver=replace(case.solver, coupling_iterations=COUPLING_ITERATIONS),
            time_stepper=replace(
                case.time_stepper,
                potential_iterations=POTENTIAL_ITERATIONS,
                max_steps=MAX_STEPS,
                potential_tolerance=POTENTIAL_TOLERANCE,
                steady_potential_tolerance=POTENTIAL_TOLERANCE,
                velocity_update_limit=VELOCITY_UPDATE_LIMIT,
                current_reconstruction="face_averaged",
            ),
        )
        solution = solve_steady(case)
        comparison = hartmann_validation(solution, ha=ha)
        hartmann_records.append({"ha": ha, "comparison": comparison})

    outputs = write_hartmann_validation_ladder_figure(
        out_dir,
        hartmann_records=hartmann_records,
    )

    summary = {
        "case": "hartmann_validation_ladder",
        "ha_values": list(HA_VALUES),
        "release_l2_target": L2_TARGET,
        "reference": "Hartmann analytical profile",
        "hartmann": [
            {
                "ha": record["ha"],
                "l2_error": record["comparison"].l2_error,
                "linf_error": record["comparison"].linf_error,
                "passes_release_target": bool(
                    record["comparison"].l2_error <= L2_TARGET
                ),
            }
            for record in hartmann_records
        ],
        "outputs": [path.name for path in outputs],
        "comparison_method": "normalized analytical comparison against the Hartmann centerline profile",
    }
    (out_dir / "hartmann_validation_ladder_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary


if __name__ == "__main__":
    run_hartmann_validation_ladder()

from __future__ import annotations

import json
from pathlib import Path

from lmx import (
    build_q2d_wall_bounded_forced_case,
    q2d_turbulence_observables,
    solve_q2d_wall_bounded_forced,
    validate_q2d_wall_bounded_forced_solution,
    write_q2d_wall_bounded_forced_plots,
)


OUTPUT_DIR = Path("artifacts/examples/q2d_wall_bounded_validation")
NX = 96
NY = 96
VISCOSITY = 0.01
HARTMANN_FRICTION = 2.0
FORCING_AMPLITUDE = 1.0
DT = 5.0e-4
T_FINAL = 0.2


def run_q2d_wall_bounded_validation() -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    case = build_q2d_wall_bounded_forced_case(
        nx=NX,
        ny=NY,
        viscosity=VISCOSITY,
        hartmann_friction=HARTMANN_FRICTION,
        forcing_amplitude=FORCING_AMPLITUDE,
        dt=DT,
        t_final=T_FINAL,
    )
    solution = solve_q2d_wall_bounded_forced(case)
    plots = write_q2d_wall_bounded_forced_plots(case, solution, OUTPUT_DIR)
    validation = validate_q2d_wall_bounded_forced_solution(case, solution)
    turbulence_observables = q2d_turbulence_observables(
        solution.field,
        lx=case.lx,
        ly=case.ly,
        viscosity=case.viscosity,
        hartmann_friction=case.hartmann_friction,
    )
    summary = {
        "case": "q2d_wall_bounded_validation",
        "plots": [path.name for path in plots],
        "validation": validation,
        "turbulence_observables": turbulence_observables,
    }
    (OUTPUT_DIR / "q2d_wall_bounded_validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    run_q2d_wall_bounded_validation()

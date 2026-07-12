from __future__ import annotations

import json
from pathlib import Path

from lmx.q2d import (
    build_q2d_forced_case,
    solve_q2d_forced,
    validate_q2d_forced_energy_budget,
    validate_q2d_forced_solution,
    write_q2d_forced_plots,
)


OUTPUT_DIR = Path("artifacts/examples/q2d_forced_validation")
NX = 96
NY = 96
VISCOSITY = 0.01
HARTMANN_FRICTION = 2.0
FORCING_AMPLITUDE = 1.0
DT = 5.0e-4
T_FINAL = 0.2


def run_q2d_forced_validation() -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    case = build_q2d_forced_case(
        nx=NX,
        ny=NY,
        viscosity=VISCOSITY,
        hartmann_friction=HARTMANN_FRICTION,
        forcing_amplitude=FORCING_AMPLITUDE,
        dt=DT,
        t_final=T_FINAL,
    )
    solution = solve_q2d_forced(case)
    plots = write_q2d_forced_plots(case, solution, OUTPUT_DIR)
    validation = validate_q2d_forced_solution(case, solution)
    energy_budget = validate_q2d_forced_energy_budget(case, solution)
    summary = {
        "case": "q2d_forced_validation",
        "plots": [path.name for path in plots],
        "validation": validation,
        "energy_budget": energy_budget,
    }
    (OUTPUT_DIR / "q2d_forced_validation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary


if __name__ == "__main__":
    run_q2d_forced_validation()

from __future__ import annotations

import json
from pathlib import Path

from lmx import (
    build_q2d_decay_case,
    solve_q2d_decay,
    validate_q2d_decay_energy_budget,
    validate_q2d_decay_solution,
    write_q2d_decay_plots,
)


OUTPUT_DIR = Path("artifacts/examples/q2d_decay_validation")
NX = 96
NY = 96
VISCOSITY = 0.01
HARTMANN_FRICTION = 2.0
DT = 5.0e-4
T_FINAL = 0.08


def run_q2d_decay_validation() -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    case = build_q2d_decay_case(
        nx=NX,
        ny=NY,
        viscosity=VISCOSITY,
        hartmann_friction=HARTMANN_FRICTION,
        dt=DT,
        t_final=T_FINAL,
    )
    solution = solve_q2d_decay(case)
    plots = write_q2d_decay_plots(case, solution, OUTPUT_DIR)
    validation = validate_q2d_decay_solution(case, solution)
    energy_budget = validate_q2d_decay_energy_budget(case, solution)
    summary = {
        "case": "q2d_decay_validation",
        "plots": [path.name for path in plots],
        "validation": validation,
        "energy_budget": energy_budget,
    }
    (OUTPUT_DIR / "q2d_decay_validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    run_q2d_decay_validation()

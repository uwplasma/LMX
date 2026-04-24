from __future__ import annotations

import json
from pathlib import Path

from lmx import (
    build_q2d_turbulence_decay_case,
    solve_q2d_turbulence_decay,
    validate_q2d_turbulence_decay_observables,
    write_q2d_turbulence_decay_movie,
)


OUTPUT_DIR = Path("artifacts/examples/q2d_turbulence_decay")
NX = 96
NY = 96
VISCOSITY = 0.006
HARTMANN_FRICTION = 0.35
DT = 5.0e-4
T_FINAL = 0.18
FRAME_COUNT = 24
FPS = 10


def run_q2d_turbulence_decay_demo() -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    case = build_q2d_turbulence_decay_case(
        nx=NX,
        ny=NY,
        viscosity=VISCOSITY,
        hartmann_friction=HARTMANN_FRICTION,
        dt=DT,
        t_final=T_FINAL,
        frame_count=FRAME_COUNT,
    )
    solution = solve_q2d_turbulence_decay(case)
    media = write_q2d_turbulence_decay_movie(solution, OUTPUT_DIR, fps=FPS)
    validation = validate_q2d_turbulence_decay_observables(case, solution)
    summary = {
        "case": "q2d_turbulence_decay",
        "media": [path.name for path in media],
        "validation": validation,
        "notes": (
            "Deterministic multi-mode Hartmann-friction decay movie. This is a "
            "spectral/energy observable gate, not a nonlinear turbulent parity "
            "claim until compared with a published turbulent Q2D reference."
        ),
    }
    (OUTPUT_DIR / "q2d_turbulence_decay_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    run_q2d_turbulence_decay_demo()

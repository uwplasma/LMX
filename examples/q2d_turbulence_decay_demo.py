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
VISCOSITY = 8.0e-4
HARTMANN_FRICTION = 0.08
AMPLITUDE = 6.0
FORCING_AMPLITUDE = 0.08
FORCING_WAVENUMBER = 4
DT = 2.0e-3
T_FINAL = 3.0
FRAME_COUNT = 72
FPS = 14


def run_q2d_turbulence_decay_demo() -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    case = build_q2d_turbulence_decay_case(
        nx=NX,
        ny=NY,
        viscosity=VISCOSITY,
        hartmann_friction=HARTMANN_FRICTION,
        amplitude=AMPLITUDE,
        forcing_amplitude=FORCING_AMPLITUDE,
        forcing_wavenumber=FORCING_WAVENUMBER,
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
            "Deterministic nonlinear periodic Q2D vorticity movie with "
            "Hartmann-friction damping and weak large-scale forcing. This is "
            "a bounded SM82-style physics gate; it is not an external turbulent "
            "parity claim until matched to a published turbulent reference."
        ),
    }
    (OUTPUT_DIR / "q2d_turbulence_decay_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    run_q2d_turbulence_decay_demo()

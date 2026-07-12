from __future__ import annotations

import json
from pathlib import Path

from lmx.external_validation import (
    compare_scalar_reference_observables,
    load_scalar_reference_observables,
    write_q2d_turbulence_reference_template,
    write_scalar_reference_comparison_plots,
    write_scalar_reference_comparison_table,
)
from lmx.q2d import (
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
EXTERNAL_REFERENCE_FILENAME = "q2d_turbulence_reference_observables.csv"
EXTERNAL_REFERENCE_TEMPLATE_FILENAME = (
    "q2d_turbulence_reference_observables_template.csv"
)


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
    external_reference_comparison = _write_external_reference_artifacts(
        validation, OUTPUT_DIR
    )
    summary = {
        "case": "q2d_turbulence_decay",
        "media": [path.name for path in media],
        "validation": validation,
        "external_reference_comparison": external_reference_comparison,
        "notes": (
            "Deterministic nonlinear periodic Q2D vorticity movie with "
            "Hartmann-friction damping and weak large-scale forcing. This is "
            "a bounded SM82-style physics gate; it is not an external turbulent "
            "parity claim until matched to a published turbulent reference."
        ),
    }
    (OUTPUT_DIR / "q2d_turbulence_decay_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary


def _write_external_reference_artifacts(
    validation: dict[str, object], out_dir: Path
) -> dict[str, object]:
    reference_path = out_dir / EXTERNAL_REFERENCE_FILENAME
    if not reference_path.exists():
        template_path = write_q2d_turbulence_reference_template(
            out_dir / EXTERNAL_REFERENCE_TEMPLATE_FILENAME
        )
        return {
            "status": "external_reference_csv_missing",
            "validation_pass": False,
            "reference_path": reference_path.name,
            "template_path": template_path.name,
            "note": (
                "Fill the template with matched Sommeria-Moreau-style turbulent "
                "observables to turn this nonlinear movie gate into an external "
                "turbulence-parity comparison."
            ),
        }

    lmx_observables = {
        key: float(value)
        for key, value in validation.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    reference_observables = load_scalar_reference_observables(
        reference_path,
        context="Q2D turbulence reference CSV",
    )
    comparison = compare_scalar_reference_observables(
        lmx_observables, reference_observables
    )
    table_path = write_scalar_reference_comparison_table(
        comparison,
        out_dir / "q2d_turbulence_reference_comparison.csv",
    )
    plot_paths = write_scalar_reference_comparison_plots(
        comparison,
        out_dir,
        output_stem="q2d_turbulence_reference_comparison",
        title="Q2D turbulence external-reference observables",
        no_data_label="No compared Q2D turbulence observables",
    )
    return {
        "status": "external_reference_compared",
        "validation_pass": bool(comparison["validation_pass"]),
        "reference_path": reference_path.name,
        "comparison_table": table_path.name,
        "plots": [path.name for path in plot_paths],
        "comparison": comparison,
    }


if __name__ == "__main__":
    run_q2d_turbulence_decay_demo()

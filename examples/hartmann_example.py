"""Solve and validate an insulating Hartmann duct.

Edit the inputs below, then run ``python examples/hartmann_example.py``.
Outputs are written beneath the ignored ``artifacts/`` directory.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from lmx import make_hartmann_case, solve_steady
from lmx.io import write_solution_outputs
from lmx.plotting import write_case_overview_plots
from lmx.validation import hartmann_validation, validation_summary

# Inputs: geometry, material properties, numerics, and output controls.
OUTPUT_DIR = Path("artifacts/examples/hartmann")
HARTMANN_NUMBER = 20.0
WIDTH = 2.0
HEIGHT = 2.0
NY = 32
NZ = 32
CONDUCTIVITY = 1.0
DENSITY = 1.0
VISCOSITY = 1.0
FORCING = 1.0
TIME_STEP = 0.001
FINAL_TIME = 1.0
MAX_STEPS = 48
POTENTIAL_ITERATIONS = 160
COUPLING_ITERATIONS = 12
STEADY_TOLERANCE = 1.0e-8
WRITE_PARAVIEW = True
WRITE_CSV = True
WRITE_NPZ = True
WRITE_PLOTS = True


# Set up the case. ``replace`` exposes solver controls without private APIs.
case = make_hartmann_case(
    ha=HARTMANN_NUMBER,
    width=WIDTH,
    height=HEIGHT,
    ny=NY,
    nz=NZ,
    conductivity=CONDUCTIVITY,
    density=DENSITY,
    viscosity=VISCOSITY,
    output_dir=str(OUTPUT_DIR),
)
case = replace(
    case,
    forcing=FORCING,
    time_stepper=replace(
        case.time_stepper,
        dt=TIME_STEP,
        t_final=FINAL_TIME,
        max_steps=MAX_STEPS,
        potential_iterations=POTENTIAL_ITERATIONS,
        steady_tolerance=STEADY_TOLERANCE,
    ),
    solver=replace(case.solver, coupling_iterations=COUPLING_ITERATIONS),
    output=replace(
        case.output,
        write_paraview=WRITE_PARAVIEW,
        write_csv_profiles=WRITE_CSV,
        write_npz=WRITE_NPZ,
        write_plots=WRITE_PLOTS,
    ),
)

# Run, validate against the analytical profile, and save reusable fields.
solution = solve_steady(case)
comparison = hartmann_validation(solution, HARTMANN_NUMBER)
generated = write_solution_outputs(solution, case, OUTPUT_DIR)
plots = (
    write_case_overview_plots(
        solution,
        OUTPUT_DIR,
        case_title=f"Hartmann duct (Ha={HARTMANN_NUMBER:g})",
        y_reference_coordinate=comparison.coordinate,
        y_reference_values=comparison.reference,
        reference_label="Analytical",
    )
    if WRITE_PLOTS
    else []
)
summary = {
    "case": case.name,
    "validation": validation_summary(solution, case.name, HARTMANN_NUMBER),
    "analytical_profile": {
        "l2_error": comparison.l2_error,
        "linf_error": comparison.linf_error,
    },
    "generated_files": {
        **{kind: [path.name for path in paths] for kind, paths in generated.items()},
        "plots": [path.name for path in plots],
    },
}
summary_path = OUTPUT_DIR / "hartmann_summary.json"
summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))

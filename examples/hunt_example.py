"""Solve a Hunt duct with conducting Hartmann and insulating side walls.

Edit the inputs below, then run ``python examples/hunt_example.py``. Set
``REFERENCE_ROOT`` to a ClosedChannel data directory for an analytical overlay.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from lmx import make_hunt_case, solve_steady
from lmx.io import write_solution_outputs
from lmx.plotting import write_case_overview_plots
from lmx.validation import closed_channel_validation, validation_summary

# Inputs: geometry, wall model, material properties, numerics, and outputs.
OUTPUT_DIR = Path("artifacts/examples/hunt")
REFERENCE_ROOT: Path | None = None
HARTMANN_NUMBER = 20.0
WIDTH = 2.0
HEIGHT = 2.0
NY = 32
NZ = 32
WALL_CELLS = 3
WALL_THICKNESS = 0.1
WALL_CONDUCTANCE_RATIO = 0.05
INSULATOR_CONDUCTIVITY_RATIO = 1.0e-12
FLUID_CONDUCTIVITY = 1.0
DENSITY = 1.0
VISCOSITY = 1.0
FORCING = 1.0
TIME_STEP = 0.002
FINAL_TIME = 1.0
MAX_STEPS = 48
POTENTIAL_ITERATIONS = 160
COUPLING_ITERATIONS = 12
STEADY_TOLERANCE = 1.0e-8
WRITE_PARAVIEW = True
WRITE_CSV = True
WRITE_NPZ = True
WRITE_PLOTS = True


# Set up the layered wall regions and solver controls with public dataclasses.
case = make_hunt_case(
    ha=HARTMANN_NUMBER,
    width=WIDTH,
    height=HEIGHT,
    ny=NY,
    nz=NZ,
    wall_cells=WALL_CELLS,
    wall_thickness=WALL_THICKNESS,
    insulator_cells=WALL_CELLS,
    insulator_thickness=WALL_THICKNESS,
    fluid_conductivity=FLUID_CONDUCTIVITY,
    wall_conductance_ratio=WALL_CONDUCTANCE_RATIO,
    insulator_conductivity_ratio=INSULATOR_CONDUCTIVITY_RATIO,
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

# Run the solve and optionally load an independent analytical comparison.
solution = solve_steady(case)
comparison = (
    closed_channel_validation(solution, "hunt", int(HARTMANN_NUMBER), reference_root=REFERENCE_ROOT)
    if REFERENCE_ROOT is not None
    else None
)
generated = write_solution_outputs(solution, case, OUTPUT_DIR)
plots = (
    write_case_overview_plots(
        solution,
        OUTPUT_DIR,
        case_title=f"Hunt duct (Ha={HARTMANN_NUMBER:g})",
        y_reference_coordinate=(None if comparison is None else comparison.y_profile.coordinate),
        y_reference_values=(None if comparison is None else comparison.y_profile.reference),
        z_reference_coordinate=(None if comparison is None else comparison.z_profile.coordinate),
        z_reference_values=(None if comparison is None else comparison.z_profile.reference),
        reference_label="Analytical",
    )
    if WRITE_PLOTS
    else []
)
summary = {
    "case": case.name,
    "wall_model": "conducting Hartmann walls; insulating side walls",
    "validation": validation_summary(solution, case.name, HARTMANN_NUMBER),
    "analytical_profile": None
    if comparison is None
    else {
        "reference": comparison.reference_path,
        "y_l2_error": comparison.y_profile.l2_error,
        "z_l2_error": comparison.z_profile.l2_error,
    },
    "generated_files": {
        **{kind: [path.name for path in paths] for kind, paths in generated.items()},
        "plots": [path.name for path in plots],
    },
}
summary_path = OUTPUT_DIR / "hunt_summary.json"
summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))

"""Solve an extruded rectangular duct with a custom analytic magnetic field.

Edit the inputs below, then run ``python examples/variable_field_extruded_demo.py``.
Outputs are written beneath the ignored ``artifacts/`` directory.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from lmx.cases import make_shercliff_case
from lmx.field_models import make_divergence_free_cross_section_field
from lmx.fringing import (
    ExtrudedInductionlessProblem,
    smooth_fringing_profile,
    solve_extruded_inductionless,
    validate_variable_field_extruded_solution,
)
from lmx.plotting import write_extruded_overview_plots
from lmx.specs import CaseSpec, MagneticFieldSpec

# Inputs: geometry, field variation, axial profile, solver effort, and outputs.
OUTPUT_DIR = Path("artifacts/examples/variable_field_extruded")
WIDTH, HEIGHT, LENGTH = 2.4, 1.6, 6.0
NY = NZ = 16  # Increase together to refine the rectangular cross-section.
NX_STATIONS = 9
BASE_BZ = 12.0
PERTURBATION = 0.12  # Set to zero for a cross-sectionally uniform field.
ENTRY_CENTER, EXIT_CENTER = 1.5, 4.5
TRANSITION_WIDTH = 0.35
MAX_STEPS = 80
POTENTIAL_ITERATIONS = 80
COUPLING_ITERATIONS = 8
STEADY_TOLERANCE = 1.0e-6


# Define the imposed vector field explicitly; this callable is evaluated on the mesh.
field_function = make_divergence_free_cross_section_field(
    width=WIDTH,
    height=HEIGHT,
    base_bz=BASE_BZ,
    perturbation=PERTURBATION,
)

# Compose a public CaseSpec so geometry, numerics, and field remain easy to inspect.
case: CaseSpec = make_shercliff_case(
    ha=1.0, width=WIDTH, height=HEIGHT, ny=NY, nz=NZ, output_dir=str(OUTPUT_DIR)
)
case = replace(
    case,
    name=f"variable_field_duct_bz{int(BASE_BZ)}",
    geometry=replace(case.geometry, length=LENGTH, nx=NX_STATIONS),
    magnetic_field=MagneticFieldSpec(kind="analytic", fn=field_function),
    time_stepper=replace(
        case.time_stepper,
        max_steps=MAX_STEPS,
        potential_iterations=POTENTIAL_ITERATIONS,
        steady_tolerance=STEADY_TOLERANCE,
    ),
    solver=replace(
        case.solver,
        kind="extruded_inductionless",
        coupling_iterations=COUPLING_ITERATIONS,
        coupling_tolerance=1.0e-7,
    ),
    notes="Extruded rectangular duct with an analytic divergence-free imposed field.",
)

# Run: add the editable axial envelope, then form and solve the extruded problem.
profile = smooth_fringing_profile(
    length=LENGTH,
    nx=NX_STATIONS,
    entry_center=ENTRY_CENTER,
    exit_center=EXIT_CENTER,
    transition_width=TRANSITION_WIDTH,
    axis="z",
)
problem = ExtrudedInductionlessProblem(case=case, profile=profile)
solution = solve_extruded_inductionless(problem)

# Save flow, field-scale, and conservation diagnostics from the solved state.
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
flow_plots = write_extruded_overview_plots(solution, OUTPUT_DIR, case_title="Variable-field extruded duct")
validation = validate_variable_field_extruded_solution(solution)
summary = {
    "case": case.name,
    "geometry_kind": solution.bundle.geometry_kind,
    "flow_plots": [path.name for path in flow_plots],
    "validation": validation,
}
summary_path = OUTPUT_DIR / "variable_field_extruded_summary.json"
summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))

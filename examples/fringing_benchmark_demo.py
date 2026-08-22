"""Run an internal rectangular-duct fringing-field diagnostic.

Edit the inputs below, then run ``python examples/fringing_benchmark_demo.py``.
This research-stage workflow checks response and conservation trends; it is not
an ALEX/FreeMHD comparison or a mesh-converged validation result.
The portable default takes about seven seconds and writes PNG/PDF plots plus JSON.
"""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from lmx.cases import make_shercliff_case
from lmx.fringing import (
    ExtrudedInductionlessProblem,
    smooth_fringing_profile,
    solve_extruded_inductionless,
)
from lmx.io import write_extruded_overview_plots

# Inputs: geometry, materials, field envelope, numerics, and output location.
OUTPUT_DIR = Path("artifacts/examples/fringing_benchmark")
HARTMANN_NUMBER = 20.0
WIDTH, HEIGHT, LENGTH = 2.0, 2.0, 6.0
NY = NZ = 8  # Increase together for a cross-section mesh study.
NX_STATIONS = 7  # Increase independently to refine the axial field transition.
CONDUCTIVITY, DENSITY, VISCOSITY = 1.0, 1.0, 1.0
ENTRY_CENTER, EXIT_CENTER = 1.5, 4.5
TRANSITION_WIDTH = 0.35
FIELD_AXIS = "z"
PEAK_FIELD_SCALE = 1.0
FORCING = 1.0
INITIAL_VELOCITY = 0.0
TIME_STEP = 0.001
MAX_STEPS = 4  # Keep the default portable; raise this for steady studies.
POTENTIAL_ITERATIONS = 60
COUPLING_ITERATIONS = 6
STEADY_TOLERANCE = 1.0e-6
COUPLING_TOLERANCE = 1.0e-7


# Compose the rectangular case explicitly from public, editable dataclasses.
case = make_shercliff_case(
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
    name=f"rectangular_fringe_ha{int(HARTMANN_NUMBER)}",
    geometry=replace(case.geometry, length=LENGTH, nx=NX_STATIONS),
    time_stepper=replace(
        case.time_stepper,
        dt=TIME_STEP,
        max_steps=MAX_STEPS,
        potential_iterations=POTENTIAL_ITERATIONS,
        steady_tolerance=STEADY_TOLERANCE,
    ),
    solver=replace(
        case.solver,
        kind="extruded_inductionless",
        coupling_iterations=COUPLING_ITERATIONS,
        coupling_tolerance=COUPLING_TOLERANCE,
    ),
    forcing=FORCING,
    initial_velocity=INITIAL_VELOCITY,
    notes="Internal rectangular-duct fringing-field conservation diagnostic.",
)
profile = smooth_fringing_profile(
    length=LENGTH,
    nx=NX_STATIONS,
    entry_center=ENTRY_CENTER,
    exit_center=EXIT_CENTER,
    transition_width=TRANSITION_WIDTH,
    peak_scale=PEAK_FIELD_SCALE,
    axis=FIELD_AXIS,
)
problem = ExtrudedInductionlessProblem(case=case, profile=profile)

# Run the bounded diagnostic, then save reusable plots and a compact JSON record.
solution = solve_extruded_inductionless(problem)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
plots = write_extruded_overview_plots(
    solution,
    OUTPUT_DIR,
    case_title=f"Rectangular fringing diagnostic (Ha={HARTMANN_NUMBER:g})",
)
bundle = solution.bundle
summary = {
    "case": case.name,
    "status": "research-stage internal diagnostic",
    "geometry_kind": bundle.geometry_kind,
    "shape": list(np.asarray(bundle.u).shape),
    "field_scale": np.asarray(bundle.field_scale).tolist(),
    "mean_velocity": np.asarray(bundle.mean_velocity).tolist(),
    "axial_current": np.asarray(bundle.axial_current).tolist(),
    "validation": asdict(solution.validation),
    "plots": [path.name for path in plots],
}
summary_path = OUTPUT_DIR / "fringing_benchmark_summary.json"
summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))

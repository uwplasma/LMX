"""Research-stage check that an extruded restart is bit-for-bit reproducible.

Edit the inputs below, then run this portable roughly five-second example.
It writes two restart bundles, a comparison PNG, and a JSON exactness record.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lmx.cases import make_hunt_case
from lmx.fringing import (
    ExtrudedInductionlessProblem,
    smooth_fringing_profile,
    solve_extruded_inductionless,
)
from lmx.io import (
    load_extruded_restart_bundle,
    validate_extruded_restart_bundle,
    write_extruded_restart_npz,
)


# Inputs: geometry, materials, field envelope, numerics, and output location.
OUTPUT_DIR = Path("artifacts/examples/extruded_restart_demo")
WIDTH, HEIGHT, LENGTH = 2.0, 2.0, 6.0
NY = NZ = 6
NX_STATIONS = 5
WALL_CELLS = INSULATOR_CELLS = 1
WALL_THICKNESS = INSULATOR_THICKNESS = 0.1
FLUID_CONDUCTIVITY, DENSITY, VISCOSITY = 1.0, 1.0, 1.0
WALL_CONDUCTANCE_RATIO = 0.05
INSULATOR_CONDUCTIVITY_RATIO = 1.0e-12
HARTMANN_NUMBER = 10.0
ENTRY_CENTER, EXIT_CENTER, TRANSITION_WIDTH = 1.5, 4.5, 0.35
FIELD_AXIS = "z"
FORCING = INITIAL_VELOCITY = 1.0
TIME_STEP = 0.002
POTENTIAL_ITERATIONS, COUPLING_ITERATIONS = 80, 8
STEADY_TOLERANCE, COUPLING_TOLERANCE = 1.0e-6, 1.0e-7
SPLIT_STEPS = RESUME_STEPS = 3


def _with_steps(
    problem: ExtrudedInductionlessProblem, steps: int
) -> ExtrudedInductionlessProblem:
    """Return ``problem`` with a bounded number of coupling iterations."""

    controls = replace(problem.case.time_stepper, max_steps=steps)
    return replace(problem, case=replace(problem.case, time_stepper=controls))


def _difference(direct: object, resumed: object, name: str) -> np.ndarray:
    """Return the absolute difference between one direct and resumed field."""

    direct_value = np.asarray(getattr(direct, name))
    resumed_value = np.asarray(getattr(resumed, name))
    return np.abs(direct_value - resumed_value)


# Compose the public layered-duct case so every physical choice remains visible.
case = make_hunt_case(
    ha=HARTMANN_NUMBER, width=WIDTH, height=HEIGHT,
    ny=NY, nz=NZ,
    wall_cells=WALL_CELLS, wall_thickness=WALL_THICKNESS,
    insulator_cells=INSULATOR_CELLS, insulator_thickness=INSULATOR_THICKNESS,
    fluid_conductivity=FLUID_CONDUCTIVITY, density=DENSITY, viscosity=VISCOSITY,
    wall_conductance_ratio=WALL_CONDUCTANCE_RATIO,
    insulator_conductivity_ratio=INSULATOR_CONDUCTIVITY_RATIO,
    output_dir=str(OUTPUT_DIR),
)
case = replace(
    case,
    geometry=replace(case.geometry, length=LENGTH, nx=NX_STATIONS),
    time_stepper=replace(
        case.time_stepper,
        dt=TIME_STEP,
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
)
profile = smooth_fringing_profile(
    length=LENGTH, nx=NX_STATIONS,
    entry_center=ENTRY_CENTER, exit_center=EXIT_CENTER,
    transition_width=TRANSITION_WIDTH, axis=FIELD_AXIS,
)
problem = ExtrudedInductionlessProblem(case=case, profile=profile)

# Run the first segment, validate its checkpoint against the inputs, and resume.
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
split_problem = _with_steps(problem, SPLIT_STEPS)
split_solution = solve_extruded_inductionless(split_problem)
restart_path = write_extruded_restart_npz(
    split_solution, split_problem.case, OUTPUT_DIR / "split_restart.npz"
)
restart = load_extruded_restart_bundle(restart_path)
resume_problem = _with_steps(problem, RESUME_STEPS)
validate_extruded_restart_bundle(restart, case=resume_problem.case)
resumed_solution = solve_extruded_inductionless(
    resume_problem, initial_bundle=restart.bundle
)
resumed_restart_path = write_extruded_restart_npz(
    resumed_solution, resume_problem.case, OUTPUT_DIR / "resumed_restart.npz"
)

# Run the unsplit reference and require bit-for-bit equality of restartable state.
direct_problem = _with_steps(problem, SPLIT_STEPS + RESUME_STEPS)
direct_solution = solve_extruded_inductionless(direct_problem)
direct = direct_solution.bundle
resumed = resumed_solution.bundle
state_differences = {
    name: float(np.max(_difference(direct, resumed, name)))
    for name in ("u", "v", "w", "p", "phi")
}
mean_difference = _difference(direct, resumed, "mean_velocity")
charge_difference = _difference(direct, resumed, "charge_balance_residual")
assert max(state_differences.values()) == 0.0
assert np.max(mean_difference) == np.max(charge_difference) == 0.0

# Plot the coincident histories and their machine-exact differences.
figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
axes[0].plot(profile.x, direct.mean_velocity, label="Direct")
axes[0].plot(profile.x, resumed.mean_velocity, "--", label="Restarted")
axes[0].set(xlabel="x", ylabel="Mean velocity", title="Restart equivalence")
axes[1].semilogy(profile.x, np.maximum(mean_difference, 1.0e-16), label="Velocity")
axes[1].semilogy(profile.x, np.maximum(charge_difference, 1.0e-16), "--", label="Charge")
axes[1].set(xlabel="x", ylabel="Absolute difference", title="State difference")
figure.legend(loc="outside lower center", ncols=4)
plot_path = OUTPUT_DIR / "extruded_restart_demo.png"
figure.savefig(plot_path, dpi=180)
plt.close(figure)

# Save the compact, machine-readable reproducibility record.
summary = {
    "case": case.name, "geometry_kind": case.geometry.kind,
    "split_steps": SPLIT_STEPS, "resume_steps": RESUME_STEPS,
    "restart_input": str(restart_path),
    "restart_output": str(resumed_restart_path),
    "max_mean_velocity_difference": float(np.max(mean_difference)),
    "max_charge_balance_difference": float(np.max(charge_difference)),
    "state_differences": state_differences,
    "max_state_difference": max(state_differences.values()),
    "plots": [plot_path.name],
}
summary_path = OUTPUT_DIR / "extruded_restart_summary.json"
summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))

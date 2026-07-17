"""Compare a mapped-pipe fringing solution with optional FreeMHD profiles.
Edit the inputs below, then run ``python examples/pipe_reference_comparison_demo.py``.
This is a research-stage mismatch diagnostic, not ALEX-B1 acceptance evidence."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from lmx.fringing import (
    ExtrudedInductionlessProblem,
    smooth_fringing_profile,
    solve_extruded_inductionless,
)
from lmx.specs import (
    BoundaryCondition,
    CaseSpec,
    GeometrySpec,
    MagneticFieldSpec,
    OutputSpec,
    RegionSpec,
    SolverConfig,
    TimeStepperConfig,
)
from lmx.units import magnetic_field_from_hartmann


# Inputs: external data, pipe/field geometry, materials, numerics, and outputs.
OUTPUT_DIR = Path("artifacts/examples/pipe_reference_comparison")
REFERENCE_DIR = Path("external/FreeMHDPaperAllFigures/FreeMHDPaperAllFigures/FringingBPipe")
HARTMANN_NUMBER, RADIUS = 20.0, 1.0
NR, NTHETA, LENGTH, NX_STATIONS = 4, 16, 6.0, 4
CONDUCTIVITY, DENSITY, VISCOSITY = 1.0, 1.0, 1.0
ENTRY_CENTER, EXIT_CENTER, TRANSITION_WIDTH = 1.5, 4.5, 0.35
FIELD_AXIS, FORCING, REFERENCE_PRESSURE_GRADIENT = "z", 1.0, -1.0
TIME_STEP, FINAL_TIME, MAX_STEPS, PROFILE_SAMPLES = 0.001, 1.0, 4, 121
POTENTIAL_ITERATIONS, COUPLING_ITERATIONS = 16, 4
STEADY_TOLERANCE, COUPLING_TOLERANCE = 1.0e-6, 1.0e-7
SOLVER_MODE, LINEAR_SOLVER, PRECONDITIONER, TIME_SCHEME = "steady", "auto", "jacobi", "implicit_euler"


def _load_reference(kind: str) -> dict[str, object]:
    """Load one named FreeMHD line sample without adding a package API."""

    stem = {"center": "CenterLine", "negative": "NegXLine", "positive": "PosXLine"}[kind]
    matches = sorted(REFERENCE_DIR.glob(f"*_{stem}_*.csv"))
    if not matches:
        raise FileNotFoundError(f"No {stem} reference CSV found under {REFERENCE_DIR}")
    with matches[0].open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    coordinate_key = "Points:2" if "Points:2" in rows[0] else "Points2"
    velocity_key = "U:2" if "U:2" in rows[0] else "U2"
    x_key = "Points:0" if "Points:0" in rows[0] else "Points0"
    potential_key = "potE"
    columns = {name: np.asarray([float(row[name]) for row in rows]) for name in (coordinate_key, velocity_key, x_key, potential_key)}
    coordinate = columns[coordinate_key]
    scale = max(float(np.max(np.abs(coordinate))), 1.0e-12)
    return {
        "path": matches[0], "coordinate": coordinate / scale,
        "velocity": columns[velocity_key], "potential": columns[potential_key],
        "x_fraction": float(np.mean(columns[x_key]) / scale),
    }


def _pipe_cut(bundle, field_name: str, x_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate a transverse Cartesian cut from one cylindrical station."""

    station = int(np.argmax(np.abs(np.asarray(bundle.field_scale))))
    radial, theta = np.asarray(bundle.y), np.asarray(bundle.z)
    field = np.asarray(getattr(bundle, field_name)[station])
    theta = np.append(theta, theta[0] + 2.0 * np.pi)
    field = np.concatenate((field, field[:, :1]), axis=1)
    offset = x_fraction * float(np.max(radial))
    limit = max(np.sqrt(max(float(np.max(radial)) ** 2 - offset**2, 0.0)), 1.0e-12)
    transverse = np.linspace(-limit, limit, PROFILE_SAMPLES)
    points = np.column_stack((np.hypot(offset, transverse), np.mod(np.arctan2(transverse, offset), 2.0 * np.pi)))
    values = RegularGridInterpolator((radial, theta), field, bounds_error=False)(points)
    return transverse / limit, values


# Build the public case explicitly so every geometry and solver choice is editable.
field_strength = magnetic_field_from_hartmann(
    hartmann=HARTMANN_NUMBER, length_scale=RADIUS, conductivity=CONDUCTIVITY,
    density=DENSITY, kinematic_viscosity=VISCOSITY,
)
case = CaseSpec(
    name=f"pipe_fringing_ha{int(HARTMANN_NUMBER)}",
    geometry=GeometrySpec(
        kind="pipe_ogrid", width=2 * RADIUS, height=2 * RADIUS, radius=RADIUS,
        length=LENGTH, nx=NX_STATIONS, nr=NR, ntheta=NTHETA,
    ),
    regions=(RegionSpec("fluid", "fluid", CONDUCTIVITY, DENSITY, VISCOSITY),),
    magnetic_field=MagneticFieldSpec(kind="constant", value=(0.0, 0.0, field_strength)),
    boundary_conditions=(BoundaryCondition("wall", "no_slip"), BoundaryCondition("electric", "insulating")),
    time_stepper=TimeStepperConfig(
        dt=TIME_STEP, t_final=FINAL_TIME, max_steps=MAX_STEPS,
        potential_iterations=POTENTIAL_ITERATIONS, steady_tolerance=STEADY_TOLERANCE,
    ),
    solver=SolverConfig(
        kind="extruded_inductionless", mode=SOLVER_MODE, linear_solver=LINEAR_SOLVER,
        preconditioner=PRECONDITIONER, time_scheme=TIME_SCHEME, coupling_iterations=COUPLING_ITERATIONS,
        coupling_tolerance=COUPLING_TOLERANCE,
    ),
    output=OutputSpec(directory=str(OUTPUT_DIR)), forcing=FORCING,
    reference_pressure_gradient=REFERENCE_PRESSURE_GRADIENT, reference_phi_cell=(max(1, NR // 4), max(1, NTHETA // 8)),
    notes="Mapped-pipe FreeMHD-profile mismatch diagnostic.",
)
profile = smooth_fringing_profile(
    length=LENGTH, nx=NX_STATIONS, entry_center=ENTRY_CENTER, exit_center=EXIT_CENTER,
    transition_width=TRANSITION_WIDTH, axis=FIELD_AXIS,
)
problem = ExtrudedInductionlessProblem(case=case, profile=profile)

# Run, compare normalized profiles, and write a compact visible diagnostic.
solution = solve_extruded_inductionless(problem)
references = {name: _load_reference(name) for name in ("center", "negative", "positive")}
comparisons, metrics = {}, {}
for name, reference in references.items():
    field_name = "u" if name == "center" else "phi"
    coordinate, values = _pipe_cut(solution.bundle, field_name, reference["x_fraction"])
    reference_values = reference["velocity" if name == "center" else "potential"]
    reference_values = reference_values / max(float(np.max(np.abs(reference_values))), 1.0e-12)
    values = values / max(float(np.max(np.abs(values))), 1.0e-12)
    difference = np.interp(reference["coordinate"], coordinate, values) - reference_values
    comparisons[name] = (coordinate, values, reference["coordinate"], reference_values)
    metrics[name] = {"normalized_l2_error": float(np.sqrt(np.mean(difference**2))), "normalized_linf_error": float(np.max(np.abs(difference)))}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
figure, axes = plt.subplots(2, 2, figsize=(12.5, 7.5), constrained_layout=True)
for axis, name in zip(axes.ravel()[:3], comparisons, strict=True):
    x_lmx, y_lmx, x_ref, y_ref = comparisons[name]
    axis.plot(x_ref, y_ref, label="FreeMHD reference")
    axis.plot(x_lmx, y_lmx, "--", label="LMX")
    axis.set(title=f"{name.title()} profile | L2={metrics[name]['normalized_l2_error']:.3f}", xlabel="Normalized transverse coordinate", ylabel="Normalized value")
    axis.legend(frameon=False)
bundle = solution.bundle
axes[1, 1].plot(bundle.x, bundle.field_scale, label="Field scale")
axes[1, 1].plot(bundle.x, bundle.mean_velocity, label="Mean velocity")
axes[1, 1].set(title="Axial response", xlabel="x")
axes[1, 1].legend(frameon=False)
plot_paths = [OUTPUT_DIR / f"pipe_reference_comparison.{suffix}" for suffix in ("png", "pdf")]
for path in plot_paths:
    figure.savefig(path, bbox_inches="tight")
plt.close(figure)
summary = {
    "case": case.name, "geometry_kind": "pipe_ogrid",
    "status": "research-stage FreeMHD-profile mismatch diagnostic",
    "normalization": {"center": "independent_peak_axial_velocity", "negative": "independent_peak_electric_potential", "positive": "independent_peak_electric_potential"},
    "profiles": metrics, "validation": asdict(solution.validation),
    "plots": [path.name for path in plot_paths],
}
summary_path = OUTPUT_DIR / "pipe_reference_comparison_summary.json"
summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))

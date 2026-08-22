"""Compare editable LMX Shercliff and Hunt ducts with processed FreeMHD data.

This external-data workflow runs both cases from top to bottom and writes a JSON
record plus normalized midplane plots.  Point ``REFERENCE_ROOT`` at the
``ClosedChannel`` reference directory, edit the inputs below, then run
``python examples/freemhd_closed_channel_observable_parity.py``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from lmx.freemhd import load_benchmark_a_spec
from lmx.io import write_freemhd_observable_parity_plots
from lmx.mesh import generate_layered_duct_mesh, generate_rect_duct_mesh
from lmx.solvers import fully_developed_power_balance, solve_steady
from lmx.specs import (
    BoundaryCondition,
    CaseSpec,
    GeometrySpec,
    MagneticFieldSpec,
    MHDState,
    RegionSpec,
    SolverConfig,
    TimeStepperConfig,
)
from lmx.validation import (
    compare_profiles_with_shared_scale,
    default_closed_channel_reference_root,
    duct_layer_resolution_gate,
    extract_midplane_scalar_profile,
    extract_processed_profile,
    load_closed_channel_analytical,
    load_processed_slice,
    processed_slice_area_mean,
    validation_summary,
)

# Inputs: paths, geometry, materials, field, mesh, and solver controls.  The two
# path variables also accept environment overrides for scheduled validation.
OUTPUT_DIR = Path(
    os.environ.get(
        "LMX_FREEMHD_OBSERVABLE_OUTPUT",
        "artifacts/examples/freemhd_closed_channel_observable_parity",
    )
)
REFERENCE_ROOT = Path(os.environ.get("LMX_FREEMHD_PROCESSED_ROOT", default_closed_channel_reference_root()))
CASES = ("shercliff", "hunt")
HARTMANN_NUMBER = 20
X_SLICE = "1m"
WIDTH = 0.2
HEIGHT = 0.2
WALL_THICKNESS = 0.001
WALL_CELLS = 2
MAGNETIC_FIELD = (0.0, 0.2, 0.0)
FLUID_CONDUCTIVITY = 1.0e6
DENSITY = 1.0e3
VISCOSITY = 1.0e-3
CONDUCTING_WALL_CONDUCTIVITY = 5.0e6
INSULATING_WALL_CONDUCTIVITY = 1.0e-6
FLOW_RATE_TARGET_MEAN_VELOCITY: float | None = None  # None uses each reference.
CASE_SETTINGS = {
    "shercliff": {
        "ny": 49,
        "nz": 37,
        "dt": 0.001,
        "final_time": 1.5,
        "max_steps": 64,
        "outer_iterations": 2,
        "potential_iterations": 640,
        "relaxation": 0.1,
        "velocity_update_limit": 0.1,
    },
    "hunt": {
        "ny": 49,
        "nz": 37,
        "dt": 0.002,
        "final_time": 1.0,
        "max_steps": 64,
        "outer_iterations": 6,
        "potential_iterations": 1280,
        "relaxation": 0.08,
        "velocity_update_limit": 0.1,
    },
}
POTENTIAL_TOLERANCE = 1.0e-9
COUPLING_ITERATIONS = 16
OBSERVABLE_L2_TARGET = 1.0e-2


def _build_problem(case_kind: str, target_velocity: float):
    """Build the editable public case, mesh, and analytical initial state."""

    settings = CASE_SETTINGS[case_kind]
    ny, nz = int(settings["ny"]), int(settings["nz"])
    regions = [RegionSpec("fluid", "fluid", FLUID_CONDUCTIVITY, DENSITY, VISCOSITY)]
    boundaries = [BoundaryCondition("walls", "no_slip")]

    if case_kind == "shercliff":
        geometry = GeometrySpec(
            kind="rect_duct",
            width=WIDTH,
            height=HEIGHT,
            ny=ny,
            nz=nz,
            target_ha=HARTMANN_NUMBER,
        )
        boundaries.append(BoundaryCondition("electric", "insulating"))
        mesh = generate_rect_duct_mesh(
            width=WIDTH,
            height=HEIGHT,
            ny=ny,
            nz=nz,
            target_ha=HARTMANN_NUMBER,
            magnetic_axis="y",
        )
    elif case_kind == "hunt":
        wall_cells = (WALL_CELLS,) * 4
        wall_thickness = (WALL_THICKNESS,) * 4
        geometry = GeometrySpec(
            kind="layered_duct",
            width=WIDTH,
            height=HEIGHT,
            ny=ny,
            nz=nz,
            wall_thickness=wall_thickness,
            wall_cells=wall_cells,
            target_ha=HARTMANN_NUMBER,
        )
        regions.extend(
            (
                RegionSpec(
                    "conducting_wall",
                    "solid",
                    CONDUCTING_WALL_CONDUCTIVITY,
                    DENSITY,
                    VISCOSITY,
                    WALL_THICKNESS,
                ),
                RegionSpec(
                    "insulating_wall",
                    "solid",
                    INSULATING_WALL_CONDUCTIVITY,
                    DENSITY,
                    VISCOSITY,
                    WALL_THICKNESS,
                ),
            )
        )
        boundaries.extend(
            (
                BoundaryCondition(
                    "conducting_hartmann_walls",
                    "conducting_wall",
                    region="conducting_wall",
                    side="left_right",
                ),
                BoundaryCondition(
                    "insulating_side_walls",
                    "insulating",
                    region="insulating_wall",
                    side="top_bottom",
                ),
            )
        )
        mesh = generate_layered_duct_mesh(
            width=WIDTH,
            height=HEIGHT,
            ny=ny,
            nz=nz,
            wall_thickness=wall_thickness,
            wall_cells=wall_cells,
            target_ha=HARTMANN_NUMBER,
            magnetic_axis="y",
        )
    else:
        raise ValueError(f"Unsupported case kind {case_kind!r}")

    boundaries.append(
        BoundaryCondition(
            "inlet",
            "inlet_flow_rate",
            value=target_velocity * WIDTH * HEIGHT,
            axis="x",
        )
    )
    case = CaseSpec(
        name=f"{case_kind}_ha{HARTMANN_NUMBER}",
        geometry=geometry,
        regions=tuple(regions),
        magnetic_field=MagneticFieldSpec(kind="constant", value=MAGNETIC_FIELD),
        boundary_conditions=tuple(boundaries),
        time_stepper=TimeStepperConfig(
            dt=float(settings["dt"]),
            t_final=float(settings["final_time"]),
            max_steps=int(settings["max_steps"]),
            outer_iterations=int(settings["outer_iterations"]),
            potential_iterations=int(settings["potential_iterations"]),
            potential_tolerance=POTENTIAL_TOLERANCE,
            steady_tolerance=POTENTIAL_TOLERANCE,
            steady_potential_tolerance=POTENTIAL_TOLERANCE,
            relaxation=float(settings["relaxation"]),
            current_reconstruction="face_averaged",
            velocity_update_limit=float(settings["velocity_update_limit"]),
        ),
        solver=SolverConfig(
            coupling_iterations=COUPLING_ITERATIONS,
            coupling_tolerance=POTENTIAL_TOLERANCE,
        ),
        forcing=0.0,
        initial_velocity=target_velocity,
        reference_phi_cell=(mesh.ny // 2, mesh.nz // 2),
    )

    analytical = load_closed_channel_analytical(case_kind, HARTMANN_NUMBER, REFERENCE_ROOT)
    y_profile = np.interp(mesh.y_centers, analytical.coordinate, analytical.midplane_y)
    z_profile = np.interp(mesh.z_centers, analytical.coordinate, analytical.midplane_z)
    velocity = np.outer(y_profile, z_profile)
    velocity /= max(float(np.max(np.abs(velocity))), 1.0e-12)
    if mesh.fluid_mask is not None:
        velocity = np.where(np.asarray(mesh.fluid_mask), velocity, 0.0)
    zeros = jnp.zeros_like(velocity)
    initial_state = MHDState(
        u=jnp.asarray(velocity),
        phi=zeros,
        jy=zeros,
        jz=zeros,
        lorentz_x=zeros,
        time=0.0,
        residual=0.0,
    )
    return case, mesh, initial_state


def _profile_metrics(
    simulated: dict[str, jnp.ndarray],
    reference: dict[str, jnp.ndarray],
    *,
    coordinate_scale: float,
    value_scale: float,
    boundary_values: tuple[float, float] | None = None,
    remove_offset: bool = False,
) -> dict[str, object]:
    """Compare one cut with a shared physical scale and no peak fitting."""

    sim_values, ref_values = simulated["value"], reference["value"]
    sim_offset = float(jnp.mean(sim_values)) if remove_offset else 0.0
    ref_offset = float(jnp.mean(ref_values)) if remove_offset else 0.0
    comparison = compare_profiles_with_shared_scale(
        simulated["coordinate"],
        sim_values,
        reference["coordinate"],
        ref_values,
        coordinate_scale=coordinate_scale,
        value_scale=value_scale,
        simulated_offset=sim_offset,
        reference_offset=ref_offset,
        simulated_boundary_values=boundary_values,
    )
    ref_peak = max(float(jnp.max(jnp.abs(ref_values))), 1.0e-12)
    sim_peak = max(float(jnp.max(jnp.abs(sim_values))), 1.0e-12)
    return {
        "coordinate": comparison.coordinate.tolist(),
        "reference": comparison.reference.tolist(),
        "simulated": comparison.simulated.tolist(),
        "l2_error": float(comparison.l2_error),
        "linf_error": float(comparison.linf_error),
        "reference_peak_abs": ref_peak,
        "simulated_peak_abs": sim_peak,
        "peak_ratio": sim_peak / ref_peak,
        "coordinate_scale": coordinate_scale,
        "value_scale": value_scale,
        "simulated_offset": sim_offset,
        "reference_offset": ref_offset,
        "per_profile_peak_fitting": False,
    }


def _continuum_velocity_audit(
    case_kind: str,
    observables: dict[str, object],
    *,
    length_scale: float,
    velocity_scale: float,
) -> dict[str, object]:
    """Disclose analytical endpoint effects without changing accepted profiles."""

    analytical = load_closed_channel_analytical(case_kind, HARTMANN_NUMBER, REFERENCE_ROOT)
    coordinate = jnp.asarray(analytical.coordinate) / length_scale
    axes = {}
    for axis, raw_values in (
        ("y", analytical.midplane_y),
        ("z", analytical.midplane_z),
    ):
        values = jnp.asarray(raw_values) / velocity_scale
        no_slip = values.at[0].set(0.0).at[-1].set(0.0)
        cut = observables["velocity"][axis]

        def error(candidate: object, target: jnp.ndarray) -> dict[str, float]:
            """Interpolate one saved cut and return dimensionless errors."""

            interpolated = jnp.interp(
                coordinate,
                jnp.asarray(cut["coordinate"]),
                jnp.asarray(candidate),
            )
            difference = interpolated - target
            return {
                "l2_error": float(jnp.sqrt(jnp.mean(difference**2))),
                "linf_error": float(jnp.max(jnp.abs(difference))),
            }

        axes[axis] = {
            "analytical_endpoint_values": [float(values[0]), float(values[-1])],
            "lmx_raw_analytical": error(cut["simulated"], values),
            "processed_freemhd_raw_analytical": error(cut["reference"], values),
            "lmx_no_slip_endpoint_corrected_analytical": error(cut["simulated"], no_slip),
            "processed_freemhd_no_slip_endpoint_corrected_analytical": error(cut["reference"], no_slip),
        }
    return {"reference_path": analytical.path, "axes": axes}


def _side_jet_comparison(cut: dict[str, object]) -> dict[str, object]:
    """Compare the symmetric Hunt jet locations and amplitudes."""

    coordinate = np.asarray(cut["coordinate"], dtype=float)

    def metrics(values: object) -> dict[str, float]:
        """Locate both jets and their peak-to-center amplitude ratio."""

        values = np.asarray(values, dtype=float)
        cut_off = 0.02 * max(float(np.max(np.abs(coordinate))), 1.0e-20)
        sides = (
            np.flatnonzero(coordinate <= -cut_off),
            np.flatnonzero(coordinate >= cut_off),
        )
        peaks = [int(side[np.argmax(values[side])]) for side in sides]
        center = float(np.interp(0.0, coordinate, values))
        peak = float(max(values[peaks[0]], values[peaks[1]]))
        return {
            "negative_location": float(coordinate[peaks[0]]),
            "positive_location": float(coordinate[peaks[1]]),
            "negative_value": float(values[peaks[0]]),
            "positive_value": float(values[peaks[1]]),
            "center_value": center,
            "peak_value": peak,
            "peak_to_center_ratio": peak / max(abs(center), 1.0e-20),
        }

    simulated, reference = metrics(cut["simulated"]), metrics(cut["reference"])
    location_errors = [
        abs(simulated[f"{side}_location"] - reference[f"{side}_location"])
        for side in ("negative", "positive")
    ]
    return {
        "simulated": simulated,
        "reference": reference,
        "negative_location_error": location_errors[0],
        "positive_location_error": location_errors[1],
        "normalized_location_error": max(location_errors)
        / max(
            abs(reference["negative_location"]),
            abs(reference["positive_location"]),
            1.0e-20,
        ),
        "peak_value_relative_error": abs(simulated["peak_value"] - reference["peak_value"])
        / max(abs(reference["peak_value"]), 1.0e-20),
        "peak_to_center_ratio_error": abs(
            simulated["peak_to_center_ratio"] - reference["peak_to_center_ratio"]
        )
        / max(abs(reference["peak_to_center_ratio"]), 1.0e-20),
    }


def _build_record(case_kind: str, case: CaseSpec, solution, spec) -> dict[str, object]:
    """Extract independently normalized profiles and conservation evidence."""

    geometry, drive = spec["geometry"], spec["drive"]
    reference = load_processed_slice(
        case_kind,
        HARTMANN_NUMBER,
        x_slice=X_SLICE,
        reference_root=REFERENCE_ROOT,
    )
    field_strength = max(map(abs, MAGNETIC_FIELD))
    length_scale = float(geometry["length_scale"])
    velocity_scale = float(drive["reference_mean_velocity"])
    scales = {
        "velocity": velocity_scale,
        "potential": velocity_scale * field_strength * length_scale,
        "current": FLUID_CONDUCTIVITY * velocity_scale * field_strength,
        "lorentz": FLUID_CONDUCTIVITY * velocity_scale * field_strength**2,
    }
    fields = {
        "velocity": (solution.state.u, "U", 0, (0.0, 0.0), False),
        "potential": (solution.state.phi, "potE", None, None, True),
        "current": (solution.state.jy, "J", 1, None, False),
        "lorentz": (solution.state.lorentz_x, "JxB", 0, None, False),
    }
    observables = {}
    for name, (
        sim_field,
        ref_name,
        component,
        boundaries,
        remove_offset,
    ) in fields.items():
        cuts = {}
        for axis in ("y", "z"):
            cuts[axis] = _profile_metrics(
                extract_midplane_scalar_profile(solution, sim_field, axis=axis, fluid_only=True),
                extract_processed_profile(
                    reference,
                    axis=axis,
                    field_name=ref_name,
                    component=component,
                ),
                coordinate_scale=length_scale,
                value_scale=scales[name],
                boundary_values=boundaries,
                remove_offset=remove_offset,
            )
        cuts["peak_ratio"] = sum(cut["peak_ratio"] for cut in cuts.values()) / 2
        observables[name] = cuts

    current_vector_audit = {}
    for name, sim_field, component in (
        ("jy", solution.state.jy, 1),
        ("jz", solution.state.jz, 2),
    ):
        current_vector_audit[name] = {
            axis: _profile_metrics(
                extract_midplane_scalar_profile(solution, sim_field, axis=axis, fluid_only=True),
                extract_processed_profile(reference, axis=axis, field_name="J", component=component),
                coordinate_scale=length_scale,
                value_scale=scales["current"],
            )
            for axis in ("y", "z")
        }

    diagnostics = validation_summary(solution, case.name, HARTMANN_NUMBER)
    current_scale = scales["current"]
    acceptance = float(spec["acceptance"]["profile_l2_max"]) * float(
        spec["acceptance"]["balance_error_fraction_of_observable_tolerance"]
    )
    flow_rate = float(solution.diagnostics.volumetric_flow_rate_history[-1])
    mean_velocity = flow_rate / (WIDTH * HEIGHT)
    reference_mean = float(drive["reference_mean_velocity"])
    processed_mean = processed_slice_area_mean(reference)
    pressure_reference = float(drive["reference_pressure_gradient"])
    pressure_applied = float(solution.diagnostics.applied_forcing_history[-1])
    power_balance = fully_developed_power_balance(case, solution)
    power_balance["acceptance_target"] = acceptance
    settings = CASE_SETTINGS[case_kind]
    return {
        "case_kind": case_kind,
        "ha": HARTMANN_NUMBER,
        "x_slice": X_SLICE,
        "initial_profile": "analytic",
        "drive_mode": "flow_rate",
        "target_mean_velocity": float(case.initial_velocity),
        "target_mean_velocity_source": (
            "matched_benchmark_spec" if FLOW_RATE_TARGET_MEAN_VELOCITY is None else "configured"
        ),
        "settings": {
            **settings,
            "current_reconstruction": "face_averaged",
            "effective_ny": int(solution.mesh.ny),
            "effective_nz": int(solution.mesh.nz),
        },
        "benchmark_spec": {
            "id": spec["id"],
            "path": spec["path"],
            "sha256": spec["sha256"],
        },
        "normalization": spec["normalization"],
        "observable_scales": scales,
        "layer_resolution": duct_layer_resolution_gate(case, solution.mesh),
        "solver_diagnostics": diagnostics,
        "current_balance": {
            "div_current_max_normalized": float(diagnostics["div_current_max"])
            / (current_scale / length_scale),
            "charge_balance_normalized": float(diagnostics["charge_balance_residual"])
            / (current_scale / length_scale),
            "interface_current_residual_normalized": float(diagnostics["interface_current_residual"])
            / current_scale,
            "acceptance_target": acceptance,
        },
        "power_balance": power_balance,
        "continuum_velocity_audit": _continuum_velocity_audit(
            case_kind,
            observables,
            length_scale=length_scale,
            velocity_scale=velocity_scale,
        ),
        "steady_steps_used": int(solution.diagnostics.residual_history.size),
        "applied_pressure_gradient": pressure_applied,
        "reference_path": reference.path,
        "observables": observables,
        "current_vector_audit": current_vector_audit,
        "integral_observables": {
            "simulated_flow_rate": flow_rate,
            "simulated_mean_velocity": mean_velocity,
            "reference_mean_velocity": reference_mean,
            "processed_reference_mean_velocity": processed_mean,
            "spec_to_processed_mean_velocity_relative_error": abs(reference_mean - processed_mean)
            / max(abs(processed_mean), 1.0e-20),
            "mean_velocity_relative_error": abs(mean_velocity - reference_mean)
            / max(abs(reference_mean), 1.0e-20),
            "applied_pressure_gradient": pressure_applied,
            "reference_pressure_gradient": pressure_reference,
            "pressure_gradient_relative_error": abs(pressure_applied - pressure_reference)
            / max(abs(pressure_reference), 1.0e-20),
        },
        "hunt_side_jet": (
            _side_jet_comparison(observables["velocity"]["z"]) if case_kind == "hunt" else None
        ),
    }


def _observable_gate(records: list[dict[str, object]]):
    """Rank normalized cuts and report missing research-grade evidence."""

    required = ("velocity", "potential", "current", "lorentz")
    missing, ranked = [], []
    for record in records:
        observables = record.get("observables", {})
        for name in required:
            observable = observables.get(name) if isinstance(observables, dict) else None
            peak = max(
                (
                    float(cut.get("reference_peak_abs", 1.0))
                    for axis in ("y", "z")
                    if isinstance(observable, dict) and isinstance(cut := observable.get(axis), dict)
                ),
                default=1.0,
            )
            for axis in ("y", "z"):
                cut = observable.get(axis) if isinstance(observable, dict) else None
                if not isinstance(cut, dict):
                    missing.append(
                        {
                            "case_kind": record["case_kind"],
                            "observable": name,
                            "axis": axis,
                        }
                    )
                    continue
                peak_fraction = float(cut.get("reference_peak_abs", peak)) / max(peak, 1.0e-20)
                l2_error = float(cut["l2_error"])
                status = (
                    "low_signal"
                    if peak_fraction < 1.0e-3
                    else "pass"
                    if l2_error <= OBSERVABLE_L2_TARGET
                    else "offender"
                )
                ranked.append(
                    {
                        "case_kind": record["case_kind"],
                        "drive_mode": record["drive_mode"],
                        "observable": name,
                        "axis": axis,
                        "l2_error": l2_error,
                        "linf_error": float(cut["linf_error"]),
                        "peak_ratio": float(cut.get("peak_ratio", 1.0)),
                        "reference_peak_abs": float(cut.get("reference_peak_abs", 1.0)),
                        "reference_peak_fraction": peak_fraction,
                        "l2_target": OBSERVABLE_L2_TARGET,
                        "target_ratio": l2_error / OBSERVABLE_L2_TARGET,
                        "status": status,
                    }
                )
    order = {"offender": 2, "pass": 1, "low_signal": 0}
    ranked.sort(
        key=lambda item: (
            order[item["status"]],
            item["target_ratio"],
            item["linf_error"],
        ),
        reverse=True,
    )
    counts = {status: sum(item["status"] == status for item in ranked) for status in order}
    return {
        "case_count": len(records),
        "cases": sorted(str(record["case_kind"]) for record in records),
        "l2_target": OBSERVABLE_L2_TARGET,
        "required_observables": list(required),
        "required_axes": ["y", "z"],
        "observable_pass_count": counts["pass"],
        "observable_offender_count": counts["offender"],
        "low_signal_count": counts["low_signal"],
        "missing_observable_count": len(missing),
        "missing_observables": missing,
        "top_observable_offenders": ranked[:8],
        "research_grade_validation_pass": counts["offender"] == 0 and not missing,
    }, ranked[:8]


# Run both editable cases explicitly, then extract, plot, and save the evidence.
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
records = []
for case_kind in CASES:
    benchmark_spec = load_benchmark_a_spec(case_kind)
    target_velocity = (
        float(FLOW_RATE_TARGET_MEAN_VELOCITY)
        if FLOW_RATE_TARGET_MEAN_VELOCITY is not None
        else float(benchmark_spec["drive"]["reference_mean_velocity"])
    )
    case, mesh, initial_state = _build_problem(case_kind, target_velocity)
    solution = solve_steady(case, mesh=mesh, initial_state=initial_state)
    records.append(_build_record(case_kind, case, solution, benchmark_spec))

plots = write_freemhd_observable_parity_plots(
    records,
    OUTPUT_DIR,
    case_title=(f"LMX vs FreeMHD normalized midplane observables (Ha={HARTMANN_NUMBER})"),
)
observable_gate, offenders = _observable_gate(records)
summary = {
    "case": "freemhd_closed_channel_observable_parity",
    "ha": HARTMANN_NUMBER,
    "x_slice": X_SLICE,
    "initial_profile": "analytic",
    "drive_mode": "flow_rate",
    "target_mean_velocity": FLOW_RATE_TARGET_MEAN_VELOCITY,
    "target_mean_velocity_by_case": {
        str(record["case_kind"]): record["target_mean_velocity"] for record in records
    },
    "target_mean_velocity_source": (
        "matched_benchmark_spec" if FLOW_RATE_TARGET_MEAN_VELOCITY is None else "configured"
    ),
    "settings": CASE_SETTINGS,
    "geometry": {
        "width": WIDTH,
        "height": HEIGHT,
        "wall_thickness": WALL_THICKNESS,
        "wall_cells": WALL_CELLS,
    },
    "material": {
        "fluid_conductivity": FLUID_CONDUCTIVITY,
        "density": DENSITY,
        "viscosity": VISCOSITY,
        "conducting_wall_conductivity": CONDUCTING_WALL_CONDUCTIVITY,
        "insulating_wall_conductivity": INSULATING_WALL_CONDUCTIVITY,
    },
    "records": records,
    "observable_gate": observable_gate,
    "top_observable_offenders": offenders,
    "plots": [path.name for path in plots],
}
summary_path = OUTPUT_DIR / "freemhd_closed_channel_observable_parity_summary.json"
summary_path.write_text(json.dumps(summary, indent=2) + "\n")
print(f"Wrote {summary_path}")

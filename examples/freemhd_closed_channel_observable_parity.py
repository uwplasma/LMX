from __future__ import annotations

import json
import os
from pathlib import Path

import jax.numpy as jnp

from lmx.freemhd import (
    compare_side_jet_profiles,
    load_benchmark_a_spec,
    summarize_observable_gate,
    summarize_observable_offenders,
)
from lmx.reference_data import (
    default_closed_channel_reference_root,
    extract_processed_profile,
    load_closed_channel_analytical,
    load_processed_slice,
    processed_slice_area_mean,
)
from lmx.showcase import solve_closed_channel_benchmark
from lmx.plotting import write_freemhd_observable_parity_plots
from lmx.solvers import fully_developed_power_balance
from lmx.validation import (
    compare_profiles_with_shared_scale,
    duct_layer_resolution_gate,
    extract_midplane_scalar_profile,
    validation_summary,
)


OUTPUT_DIR = Path("artifacts/examples/freemhd_closed_channel_observable_parity")
REFERENCE_ROOT = Path(
    os.environ.get(
        "LMX_FREEMHD_PROCESSED_ROOT", default_closed_channel_reference_root()
    )
)
HA = 20
X_SLICE = "1m"
WIDTH = 0.2
HEIGHT = 0.2
WALL_THICKNESS = 0.001
WALL_CELLS = 2
FLUID_CONDUCTIVITY = 1.0e6
DENSITY = 1.0e3
VISCOSITY = 1.0e-3
CONDUCTING_WALL_CONDUCTIVITY = 5.0e6
INSULATING_WALL_CONDUCTIVITY = 1.0e-6
CASE_SETTINGS = {
    "shercliff": {
        "ny": 49,
        "nz": 37,
        "max_steps": 64,
        "potential_iterations": 640,
        "current_reconstruction": "face_averaged",
        "velocity_update_limit": 0.1,
    },
    "hunt": {
        "ny": 49,
        "nz": 37,
        "max_steps": 64,
        "potential_iterations": 1280,
        "current_reconstruction": "face_averaged",
        "velocity_update_limit": 0.1,
    },
}
INITIAL_PROFILE = "analytic"
DRIVE_MODE = "flow_rate"
FLOW_RATE_TARGET_MEAN_VELOCITY: float | None = None


def _max_abs(value: jnp.ndarray) -> float:
    return float(jnp.max(jnp.abs(value))) if value.size else 0.0


def _dimensionless_profile_error(
    candidate_coordinate: list[float] | jnp.ndarray,
    candidate_values: list[float] | jnp.ndarray,
    reference_coordinate: jnp.ndarray,
    reference_values: jnp.ndarray,
) -> dict[str, float]:
    interpolated = jnp.interp(
        jnp.asarray(reference_coordinate, dtype=float),
        jnp.asarray(candidate_coordinate, dtype=float),
        jnp.asarray(candidate_values, dtype=float),
    )
    difference = interpolated - jnp.asarray(reference_values, dtype=float)
    return {
        "l2_error": float(jnp.sqrt(jnp.mean(difference**2))),
        "linf_error": float(jnp.max(jnp.abs(difference))),
    }


def _continuum_velocity_audit(
    case_kind: str,
    observables: dict[str, object],
    *,
    ha: int,
    length_scale: float,
    velocity_scale: float,
    reference_root: Path,
) -> dict[str, object]:
    analytical = load_closed_channel_analytical(case_kind, ha, reference_root)
    coordinate = analytical.coordinate / length_scale
    payload: dict[str, object] = {"reference_path": analytical.path, "axes": {}}
    for axis, values in (
        ("y", analytical.midplane_y / velocity_scale),
        ("z", analytical.midplane_z / velocity_scale),
    ):
        cut = observables["velocity"][axis]
        no_slip_values = jnp.asarray(values).at[0].set(0.0).at[-1].set(0.0)
        payload["axes"][axis] = {
            "analytical_endpoint_values": [float(values[0]), float(values[-1])],
            "lmx_raw_analytical": _dimensionless_profile_error(
                cut["coordinate"], cut["simulated"], coordinate, values
            ),
            "processed_freemhd_raw_analytical": _dimensionless_profile_error(
                cut["coordinate"], cut["reference"], coordinate, values
            ),
            "lmx_no_slip_endpoint_corrected_analytical": _dimensionless_profile_error(
                cut["coordinate"], cut["simulated"], coordinate, no_slip_values
            ),
            "processed_freemhd_no_slip_endpoint_corrected_analytical": _dimensionless_profile_error(
                cut["coordinate"], cut["reference"], coordinate, no_slip_values
            ),
        }
    return payload


def _profile_metrics(
    *,
    simulated_coordinate: jnp.ndarray,
    simulated_values: jnp.ndarray,
    reference_coordinate: jnp.ndarray,
    reference_values: jnp.ndarray,
    coordinate_scale: float,
    value_scale: float,
    simulated_boundary_values: tuple[float, float] | None = None,
    remove_offset: bool = False,
) -> dict[str, object]:
    simulated_offset = float(jnp.mean(simulated_values)) if remove_offset else 0.0
    reference_offset = float(jnp.mean(reference_values)) if remove_offset else 0.0
    comparison = compare_profiles_with_shared_scale(
        simulated_coordinate,
        simulated_values,
        reference_coordinate,
        reference_values,
        coordinate_scale=coordinate_scale,
        value_scale=value_scale,
        simulated_offset=simulated_offset,
        reference_offset=reference_offset,
        simulated_boundary_values=simulated_boundary_values,
    )
    ref_scale = max(_max_abs(reference_values), 1.0e-12)
    sim_scale = max(_max_abs(simulated_values), 1.0e-12)
    return {
        "coordinate": comparison.coordinate.tolist(),
        "reference": comparison.reference.tolist(),
        "simulated": comparison.simulated.tolist(),
        "l2_error": float(comparison.l2_error),
        "linf_error": float(comparison.linf_error),
        "reference_peak_abs": float(ref_scale),
        "simulated_peak_abs": float(sim_scale),
        "peak_ratio": float(sim_scale / ref_scale),
        "coordinate_scale": float(coordinate_scale),
        "value_scale": float(value_scale),
        "simulated_offset": simulated_offset,
        "reference_offset": reference_offset,
        "per_profile_peak_fitting": False,
    }


def _observable_record(
    case_kind: str,
    *,
    drive_mode: str = DRIVE_MODE,
    initial_profile: str = INITIAL_PROFILE,
    flow_rate_target_mean_velocity: float | None = FLOW_RATE_TARGET_MEAN_VELOCITY,
    case_settings: dict[str, dict[str, object]] | None = None,
    linear_solver: str = "auto",
) -> dict[str, object]:
    if case_settings is None:
        case_settings = CASE_SETTINGS
    settings = case_settings[case_kind]
    spec = load_benchmark_a_spec(case_kind)
    geometry = spec["geometry"]
    fluid = spec["fluid"]
    field = spec["magnetic_field"]
    wall = spec["wall"]
    drive = spec["drive"]
    ha = int(round(float(field["hartmann_number"])))
    width = float(geometry["width"])
    height = float(geometry["height"])
    reference = load_processed_slice(
        case_kind, ha, x_slice=X_SLICE, reference_root=REFERENCE_ROOT
    )
    target_mean_velocity = (
        float(flow_rate_target_mean_velocity)
        if flow_rate_target_mean_velocity is not None
        else float(drive["reference_mean_velocity"])
    )
    case, solution, _comparison = solve_closed_channel_benchmark(
        case_kind,
        ha=ha,
        width=width,
        height=height,
        ny=settings["ny"],
        nz=settings["nz"],
        wall_cells=int(geometry["wall_cells"]),
        wall_thickness=float(geometry["wall_thickness"]),
        fluid_conductivity=float(fluid["conductivity"]),
        density=float(fluid["density"]),
        viscosity=float(fluid["kinematic_viscosity"]),
        conducting_wall_conductivity=float(wall["conducting_wall_conductivity"]),
        insulating_wall_conductivity=float(wall["insulating_wall_conductivity"]),
        max_steps=settings["max_steps"],
        coupling_iterations=settings.get("coupling_iterations", 16),
        potential_iterations=settings.get("potential_iterations", 160),
        potential_tolerance=settings.get("potential_tolerance", 1.0e-9),
        drive_mode=drive_mode,
        target_mean_velocity=target_mean_velocity,
        initial_profile=initial_profile,
        current_reconstruction=settings["current_reconstruction"],
        velocity_update_limit=settings["velocity_update_limit"],
        linear_solver=linear_solver,
        reference_root=REFERENCE_ROOT,
    )
    magnetic_field = max(abs(float(value)) for value in field["vector"])
    length_scale = float(geometry["length_scale"])
    velocity_scale = float(drive["reference_mean_velocity"])
    observable_scales = {
        "velocity": velocity_scale,
        "potential": velocity_scale * magnetic_field * length_scale,
        "current": float(fluid["conductivity"]) * velocity_scale * magnetic_field,
        "lorentz": float(fluid["conductivity"]) * velocity_scale * magnetic_field**2,
    }
    observable_specs = {
        "velocity": {
            "y": (solution.state.u, "U", 0, (0.0, 0.0), False),
            "z": (solution.state.u, "U", 0, (0.0, 0.0), False),
        },
        "potential": {
            "y": (solution.state.phi, "potE", None, None, True),
            "z": (solution.state.phi, "potE", None, None, True),
        },
        "current": {
            "y": (solution.state.jy, "J", 1, None, False),
            "z": (solution.state.jz, "J", 2, None, False),
        },
        "lorentz": {
            "y": (solution.state.lorentz_x, "JxB", 0, None, False),
            "z": (solution.state.lorentz_x, "JxB", 0, None, False),
        },
    }

    observables: dict[str, object] = {}
    for observable_name, axis_specs in observable_specs.items():
        axis_payload: dict[str, object] = {}
        peak_ratios: list[float] = []
        for axis, (
            sim_field,
            ref_name,
            ref_component,
            boundary_values,
            remove_offset,
        ) in axis_specs.items():
            simulated = extract_midplane_scalar_profile(
                solution, sim_field, axis=axis, fluid_only=True
            )
            reference_profile = extract_processed_profile(
                reference, axis=axis, field_name=ref_name, component=ref_component
            )
            metrics = _profile_metrics(
                simulated_coordinate=simulated["coordinate"],
                simulated_values=simulated["value"],
                reference_coordinate=reference_profile["coordinate"],
                reference_values=reference_profile["value"],
                coordinate_scale=length_scale,
                value_scale=observable_scales[observable_name],
                simulated_boundary_values=boundary_values,
                remove_offset=remove_offset,
            )
            axis_payload[axis] = metrics
            peak_ratios.append(float(metrics["peak_ratio"]))
        axis_payload["peak_ratio"] = float(sum(peak_ratios) / len(peak_ratios))
        observables[observable_name] = axis_payload

    current_vector_audit: dict[str, object] = {}
    for component_name, sim_field, component in (
        ("jy", solution.state.jy, 1),
        ("jz", solution.state.jz, 2),
    ):
        component_payload = {}
        for axis in ("y", "z"):
            simulated = extract_midplane_scalar_profile(
                solution, sim_field, axis=axis, fluid_only=True
            )
            reference_profile = extract_processed_profile(
                reference, axis=axis, field_name="J", component=component
            )
            component_payload[axis] = _profile_metrics(
                simulated_coordinate=simulated["coordinate"],
                simulated_values=simulated["value"],
                reference_coordinate=reference_profile["coordinate"],
                reference_values=reference_profile["value"],
                coordinate_scale=length_scale,
                value_scale=observable_scales["current"],
            )
        current_vector_audit[component_name] = component_payload

    final_flow_rate = float(solution.diagnostics.volumetric_flow_rate_history[-1])
    area = width * height
    simulated_mean_velocity = final_flow_rate / max(area, 1.0e-20)
    reference_mean_velocity = float(drive["reference_mean_velocity"])
    processed_reference_mean_velocity = processed_slice_area_mean(reference)
    pressure_reference = float(drive["reference_pressure_gradient"])
    applied_pressure_gradient = float(solution.diagnostics.applied_forcing_history[-1])
    integral_observables = {
        "simulated_flow_rate": final_flow_rate,
        "simulated_mean_velocity": simulated_mean_velocity,
        "reference_mean_velocity": reference_mean_velocity,
        "processed_reference_mean_velocity": processed_reference_mean_velocity,
        "spec_to_processed_mean_velocity_relative_error": abs(
            reference_mean_velocity - processed_reference_mean_velocity
        )
        / max(abs(processed_reference_mean_velocity), 1.0e-20),
        "mean_velocity_relative_error": abs(
            simulated_mean_velocity - reference_mean_velocity
        )
        / max(abs(reference_mean_velocity), 1.0e-20),
        "applied_pressure_gradient": applied_pressure_gradient,
        "reference_pressure_gradient": pressure_reference,
        "pressure_gradient_relative_error": (
            None
            if pressure_reference is None
            else abs(applied_pressure_gradient - float(pressure_reference))
            / max(abs(float(pressure_reference)), 1.0e-20)
        ),
    }

    hunt_side_jet = None
    if case_kind == "hunt":
        velocity_z = observables["velocity"]["z"]
        hunt_side_jet = compare_side_jet_profiles(
            velocity_z["coordinate"],
            velocity_z["simulated"],
            velocity_z["coordinate"],
            velocity_z["reference"],
        )

    solver_diagnostics = validation_summary(solution, case.name, ha)
    current_scale = observable_scales["current"]
    divergence_scale = current_scale / length_scale
    current_balance = {
        "div_current_max_normalized": float(solver_diagnostics["div_current_max"])
        / divergence_scale,
        "charge_balance_normalized": float(
            solver_diagnostics["charge_balance_residual"]
        )
        / divergence_scale,
        "interface_current_residual_normalized": float(
            solver_diagnostics["interface_current_residual"]
        )
        / current_scale,
        "acceptance_target": float(spec["acceptance"]["profile_l2_max"])
        * float(spec["acceptance"]["balance_error_fraction_of_observable_tolerance"]),
    }
    power_balance = fully_developed_power_balance(case, solution)
    power_balance["acceptance_target"] = current_balance["acceptance_target"]
    continuum_velocity_audit = _continuum_velocity_audit(
        case_kind,
        observables,
        ha=ha,
        length_scale=length_scale,
        velocity_scale=velocity_scale,
        reference_root=REFERENCE_ROOT,
    )

    return {
        "case_kind": case_kind,
        "ha": ha,
        "x_slice": X_SLICE,
        "initial_profile": initial_profile,
        "drive_mode": drive_mode,
        "target_mean_velocity": target_mean_velocity
        if drive_mode == "flow_rate"
        else None,
        "target_mean_velocity_source": (
            "matched_benchmark_spec"
            if drive_mode == "flow_rate" and flow_rate_target_mean_velocity is None
            else "configured"
        ),
        "settings": {
            **settings,
            "effective_ny": int(solution.mesh.ny),
            "effective_nz": int(solution.mesh.nz),
        },
        "benchmark_spec": {
            "id": spec["id"],
            "path": spec["path"],
            "sha256": spec["sha256"],
        },
        "normalization": spec["normalization"],
        "observable_scales": observable_scales,
        "layer_resolution": duct_layer_resolution_gate(case, solution.mesh),
        "solver_diagnostics": solver_diagnostics,
        "current_balance": current_balance,
        "power_balance": power_balance,
        "continuum_velocity_audit": continuum_velocity_audit,
        "steady_steps_used": int(solution.diagnostics.residual_history.size),
        "applied_pressure_gradient": float(
            solution.diagnostics.applied_forcing_history[-1]
        ),
        "reference_path": reference.path,
        "observables": observables,
        "current_vector_audit": current_vector_audit,
        "integral_observables": integral_observables,
        "hunt_side_jet": hunt_side_jet,
    }


def run_freemhd_closed_channel_observable_parity() -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = [_observable_record("shercliff"), _observable_record("hunt")]
    plots = write_freemhd_observable_parity_plots(
        records,
        OUTPUT_DIR,
        case_title=f"LMX vs FreeMHD normalized midplane observables (Ha={HA})",
    )
    summary = {
        "case": "freemhd_closed_channel_observable_parity",
        "ha": HA,
        "x_slice": X_SLICE,
        "initial_profile": INITIAL_PROFILE,
        "drive_mode": DRIVE_MODE,
        "target_mean_velocity": FLOW_RATE_TARGET_MEAN_VELOCITY
        if DRIVE_MODE == "flow_rate"
        else None,
        "target_mean_velocity_by_case": {
            str(record["case_kind"]): record.get("target_mean_velocity")
            for record in records
            if DRIVE_MODE == "flow_rate"
        },
        "target_mean_velocity_source": "matched_benchmark_spec"
        if DRIVE_MODE == "flow_rate" and FLOW_RATE_TARGET_MEAN_VELOCITY is None
        else "configured",
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
        "observable_gate": summarize_observable_gate(records, l2_target=1.0e-2),
        "top_observable_offenders": summarize_observable_offenders(
            records, l2_target=1.0e-2, top_n=8
        ),
        "plots": [path.name for path in plots],
    }
    (OUTPUT_DIR / "freemhd_closed_channel_observable_parity_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary


if __name__ == "__main__":
    run_freemhd_closed_channel_observable_parity()

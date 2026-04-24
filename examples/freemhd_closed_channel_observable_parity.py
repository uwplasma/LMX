from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp

from lmx import write_freemhd_observable_parity_plots
from lmx.freemhd import summarize_observable_offenders
from lmx.reference_data import extract_processed_profile, load_processed_slice
from lmx.showcase import solve_closed_channel_benchmark
from lmx.validation import compare_normalized_profiles, extract_midplane_scalar_profile


OUTPUT_DIR = Path("artifacts/examples/freemhd_closed_channel_observable_parity")
REFERENCE_ROOT = Path("/Users/rogerio/local/tests/freemhd_test_cases/FreeMHDPaperAllFigures/ClosedChannel")
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
        "current_reconstruction": "face_averaged",
        "velocity_update_limit": 0.1,
    },
    "hunt": {
        "ny": 49,
        "nz": 37,
        "max_steps": 64,
        "current_reconstruction": "face_averaged",
        "velocity_update_limit": 0.1,
    },
}
INITIAL_PROFILE = "analytic"
DRIVE_MODE = "pressure_gradient"
FLOW_RATE_TARGET_MEAN_VELOCITY = 0.9725


def _max_abs(value: jnp.ndarray) -> float:
    return float(jnp.max(jnp.abs(value))) if value.size else 0.0


def _profile_metrics(
    *,
    simulated_coordinate: jnp.ndarray,
    simulated_values: jnp.ndarray,
    reference_coordinate: jnp.ndarray,
    reference_values: jnp.ndarray,
    simulated_boundary_values: tuple[float, float] | None = None,
    remove_offset: bool = False,
) -> dict[str, object]:
    if remove_offset:
        simulated_values = simulated_values - jnp.mean(simulated_values)
        reference_values = reference_values - jnp.mean(reference_values)
    comparison = compare_normalized_profiles(
        simulated_coordinate,
        simulated_values,
        reference_coordinate,
        reference_values,
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
    }


def _observable_record(case_kind: str) -> dict[str, object]:
    settings = CASE_SETTINGS[case_kind]
    _, solution, _ = solve_closed_channel_benchmark(
        case_kind,
        ha=HA,
        width=WIDTH,
        height=HEIGHT,
        ny=settings["ny"],
        nz=settings["nz"],
        wall_cells=WALL_CELLS,
        wall_thickness=WALL_THICKNESS,
        fluid_conductivity=FLUID_CONDUCTIVITY,
        density=DENSITY,
        viscosity=VISCOSITY,
        conducting_wall_conductivity=CONDUCTING_WALL_CONDUCTIVITY,
        insulating_wall_conductivity=INSULATING_WALL_CONDUCTIVITY,
        max_steps=settings["max_steps"],
        drive_mode=DRIVE_MODE,
        target_mean_velocity=FLOW_RATE_TARGET_MEAN_VELOCITY,
        initial_profile=INITIAL_PROFILE,
        current_reconstruction=settings["current_reconstruction"],
        velocity_update_limit=settings["velocity_update_limit"],
    )
    reference = load_processed_slice(case_kind, HA, x_slice=X_SLICE, reference_root=REFERENCE_ROOT)
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
        for axis, (sim_field, ref_name, ref_component, boundary_values, remove_offset) in axis_specs.items():
            simulated = extract_midplane_scalar_profile(solution, sim_field, axis=axis, fluid_only=True)
            reference_profile = extract_processed_profile(reference, axis=axis, field_name=ref_name, component=ref_component)
            metrics = _profile_metrics(
                simulated_coordinate=simulated["coordinate"],
                simulated_values=simulated["value"],
                reference_coordinate=reference_profile["coordinate"],
                reference_values=reference_profile["value"],
                simulated_boundary_values=boundary_values,
                remove_offset=remove_offset,
            )
            axis_payload[axis] = metrics
            peak_ratios.append(float(metrics["peak_ratio"]))
        axis_payload["peak_ratio"] = float(sum(peak_ratios) / len(peak_ratios))
        observables[observable_name] = axis_payload

    return {
        "case_kind": case_kind,
        "ha": HA,
        "x_slice": X_SLICE,
        "initial_profile": INITIAL_PROFILE,
        "drive_mode": DRIVE_MODE,
        "target_mean_velocity": FLOW_RATE_TARGET_MEAN_VELOCITY if DRIVE_MODE == "flow_rate" else None,
        "settings": settings,
        "applied_pressure_gradient": float(solution.diagnostics.applied_forcing_history[-1]),
        "reference_path": reference.path,
        "observables": observables,
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
        "target_mean_velocity": FLOW_RATE_TARGET_MEAN_VELOCITY if DRIVE_MODE == "flow_rate" else None,
        "settings": CASE_SETTINGS,
        "geometry": {"width": WIDTH, "height": HEIGHT, "wall_thickness": WALL_THICKNESS, "wall_cells": WALL_CELLS},
        "material": {
            "fluid_conductivity": FLUID_CONDUCTIVITY,
            "density": DENSITY,
            "viscosity": VISCOSITY,
            "conducting_wall_conductivity": CONDUCTING_WALL_CONDUCTIVITY,
            "insulating_wall_conductivity": INSULATING_WALL_CONDUCTIVITY,
        },
        "records": records,
        "top_observable_offenders": summarize_observable_offenders(records, l2_target=1.0e-2, top_n=8),
        "plots": [path.name for path in plots],
    }
    (OUTPUT_DIR / "freemhd_closed_channel_observable_parity_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    run_freemhd_closed_channel_observable_parity()

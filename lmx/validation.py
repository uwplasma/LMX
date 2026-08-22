"""Analytical, conservation, convergence, and parity validation."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from .core import Solution
from .mesh import StructuredMesh
from .reference_data import (
    ClosedChannelAnalyticalReference,
    load_closed_channel_analytical,
)
from .specs import CaseSpec


@dataclass(frozen=True)
class AnalyticComparison:
    coordinate: jnp.ndarray
    simulated: jnp.ndarray
    reference: jnp.ndarray
    l2_error: float
    linf_error: float


@dataclass(frozen=True)
class ProfileSymmetry:
    axis: str
    mean_abs_error: float
    max_abs_error: float


@dataclass(frozen=True)
class ClosedChannelValidation:
    case_kind: str
    ha: int
    y_profile: AnalyticComparison
    z_profile: AnalyticComparison
    reference_pressure_drop: float | None
    reference_path: str


@dataclass(frozen=True)
class AcceptanceReport:
    case_name: str
    l2_error: float
    linf_error: float
    l2_threshold: float
    linf_threshold: float
    passed_l2: bool
    passed_linf: bool
    passed: bool


_SUMMARY_HISTORIES = (
    ("potential_residual", "potential_residual_history", 0.0),
    ("potential_iterations_used", "potential_iterations_history", 0.0),
    ("mean_velocity", "mean_velocity_history", 0.0),
    ("applied_forcing", "applied_forcing_history", 0.0),
    ("pressure_proxy", "pressure_proxy_history", 0.0),
    ("current_scaled_pressure_proxy", "current_scaled_pressure_proxy_history", 0.0),
    ("linear_residual", "linear_residual_history", 0.0),
    ("linear_iterations_used", "linear_iterations_history", 0.0),
    ("volumetric_flow_rate", "volumetric_flow_rate_history", 0.0),
    ("mean_current_magnitude", "mean_current_magnitude_history", 0.0),
    ("lorentz_power", "lorentz_power_history", 0.0),
    ("div_current_max", "div_current_max_history", 0.0),
    ("charge_balance_residual", "charge_balance_residual_history", 0.0),
    ("gauge_residual", "gauge_residual_history", 0.0),
    ("interface_current_residual", "interface_current_residual_history", 0.0),
    ("raw_update_max", "raw_update_max_history", 0.0),
    ("limiter_scale", "limiter_scale_history", 1.0),
    ("limited_fraction", "limited_fraction_history", 0.0),
)


def combined_profile_error(*errors: float) -> float:
    if not errors:
        return 0.0
    values = jnp.asarray(errors, dtype=float)
    return float(jnp.sqrt(jnp.mean(values**2)))


def _dominant_magnetic_axis(case: CaseSpec) -> str | None:
    if case.magnetic_field.kind != "constant" or case.magnetic_field.value is None:
        return None
    bx, by, bz = case.magnetic_field.value
    magnitudes = {"x": abs(bx), "y": abs(by), "z": abs(bz)}
    axis = max(magnitudes, key=magnitudes.get)
    return axis if magnitudes[axis] > 0.0 else None


def _fluid_axis_profile(mesh: StructuredMesh, axis: str) -> tuple[jnp.ndarray, jnp.ndarray]:
    fluid_mask = mesh.fluid_mask
    if axis == "y":
        coordinates = mesh.y_centers
        widths = mesh.dy
        if fluid_mask is None:
            return coordinates, widths
        mid_z = int(jnp.argmax(jnp.sum(fluid_mask, axis=0)))
        mask = fluid_mask[:, mid_z]
        return coordinates[mask], widths[mask]
    coordinates = mesh.z_centers
    widths = mesh.dz
    if fluid_mask is None:
        return coordinates, widths
    mid_y = int(jnp.argmax(jnp.sum(fluid_mask, axis=1)))
    mask = fluid_mask[mid_y, :]
    return coordinates[mask], widths[mask]


def _cells_across_layer(widths: jnp.ndarray, layer_thickness: float) -> float:
    cumulative = jnp.cumsum(widths)
    full_cells = jnp.sum(cumulative < layer_thickness)
    covered_before = jnp.where(full_cells > 0, cumulative[full_cells - 1], 0.0)
    remaining = jnp.maximum(layer_thickness - covered_before, 0.0)
    next_width = jnp.where(full_cells < widths.size, widths[full_cells], widths[-1])
    partial_cell = jnp.minimum(remaining / jnp.maximum(next_width, 1e-12), 1.0)
    return float(full_cells + partial_cell)


def duct_layer_resolution_metrics(case: CaseSpec, mesh: StructuredMesh) -> dict[str, float]:
    axis = _dominant_magnetic_axis(case)
    ha = case.geometry.target_ha
    if axis not in {"y", "z"} or ha is None or ha <= 0.0:
        return {}

    hartmann_axis = axis
    side_axis = "z" if axis == "y" else "y"
    hartmann_coordinates, hartmann_widths = _fluid_axis_profile(mesh, hartmann_axis)
    side_coordinates, side_widths = _fluid_axis_profile(mesh, side_axis)
    if hartmann_coordinates.size == 0 or side_coordinates.size == 0:
        return {}

    hartmann_half_spacing = 0.5 * float(
        hartmann_coordinates[-1] - hartmann_coordinates[0] + 0.5 * (hartmann_widths[0] + hartmann_widths[-1])
    )
    side_half_spacing = 0.5 * float(
        side_coordinates[-1] - side_coordinates[0] + 0.5 * (side_widths[0] + side_widths[-1])
    )
    hartmann_layer_thickness = hartmann_half_spacing / float(ha)
    side_layer_thickness = side_half_spacing / float(jnp.sqrt(ha))

    return {
        "hartmann_layer_thickness": hartmann_layer_thickness,
        "side_layer_thickness": side_layer_thickness,
        "hartmann_layer_cells": _cells_across_layer(hartmann_widths, hartmann_layer_thickness),
        "side_layer_cells": _cells_across_layer(side_widths, side_layer_thickness),
        "min_hartmann_spacing": float(jnp.min(hartmann_widths)),
        "min_side_spacing": float(jnp.min(side_widths)),
    }


def duct_layer_resolution_gate(
    case: CaseSpec,
    mesh: StructuredMesh,
    *,
    min_hartmann_cells: float = 8.0,
    min_side_cells: float = 6.0,
    cell_tolerance: float = 1.0e-9,
) -> dict[str, float | bool]:
    """Return benchmark-readiness metrics for Hartmann and side layers."""

    metrics = duct_layer_resolution_metrics(case, mesh)
    if not metrics:
        return {
            "layer_resolution_supported": False,
            "layer_resolution_pass": False,
            "min_required_hartmann_layer_cells": float(min_hartmann_cells),
            "min_required_side_layer_cells": float(min_side_cells),
            "hartmann_layer_cell_ratio": 0.0,
            "side_layer_cell_ratio": 0.0,
            "minimum_mesh_refinement_factor": 0.0,
        }
    # Cell counts come from partial-cell geometric integration, so exact
    # threshold cases can land at 7.99999999999999 rather than 8.0.
    hartmann_pass = metrics["hartmann_layer_cells"] + cell_tolerance >= min_hartmann_cells
    side_pass = metrics["side_layer_cells"] + cell_tolerance >= min_side_cells
    hartmann_ratio = metrics["hartmann_layer_cells"] / max(float(min_hartmann_cells), 1.0e-20)
    side_ratio = metrics["side_layer_cells"] / max(float(min_side_cells), 1.0e-20)
    limiting_ratio = min(hartmann_ratio, side_ratio)
    return {
        **metrics,
        "layer_resolution_supported": True,
        "min_required_hartmann_layer_cells": float(min_hartmann_cells),
        "min_required_side_layer_cells": float(min_side_cells),
        "hartmann_layer_cell_deficit": max(float(min_hartmann_cells) - metrics["hartmann_layer_cells"], 0.0),
        "side_layer_cell_deficit": max(float(min_side_cells) - metrics["side_layer_cells"], 0.0),
        "hartmann_layer_cell_ratio": float(hartmann_ratio),
        "side_layer_cell_ratio": float(side_ratio),
        "minimum_mesh_refinement_factor": 1.0 / max(float(limiting_ratio), 1.0e-20),
        "hartmann_layer_resolution_pass": bool(hartmann_pass),
        "side_layer_resolution_pass": bool(side_pass),
        "layer_resolution_pass": bool(hartmann_pass and side_pass),
    }


def hartmann_analytic_profile(y: jnp.ndarray, ha: float) -> jnp.ndarray:
    denom = jnp.cosh(ha) - 1.0
    denom = jnp.where(jnp.abs(denom) < 1e-12, 1.0, denom)
    return 1.0 - (jnp.cosh(ha * y) - 1.0) / denom


def _exact_coordinate_index(
    coordinate: jnp.ndarray, *, target: float = 0.0, tolerance: float = 1.0e-12
) -> int | None:
    coordinate = jnp.asarray(coordinate, dtype=float)
    matches = np.where(np.abs(np.asarray(coordinate, dtype=float) - float(target)) <= tolerance)[0]
    if matches.size == 0:
        return None
    return int(matches[0])


def _midplane_indices(coordinate: jnp.ndarray) -> tuple[int, int, float]:
    if coordinate.size == 1:
        return 0, 0, 0.0
    center = _exact_coordinate_index(coordinate)
    if center is not None:
        return center, center, 0.0
    upper = max(1, min(int(jnp.searchsorted(coordinate, 0.0)), coordinate.size - 1))
    lower = upper - 1
    span = float(coordinate[upper] - coordinate[lower])
    weight = 0.5 if abs(span) <= 1.0e-12 else -float(coordinate[lower]) / span
    return lower, upper, min(max(weight, 0.0), 1.0)


def _midplane_values(
    field: jnp.ndarray, coordinate: jnp.ndarray, fixed_axis: int
) -> tuple[jnp.ndarray, tuple[int, int]]:
    lower, upper, weight = _midplane_indices(coordinate)
    lower_values = jnp.take(field, lower, axis=fixed_axis)
    if lower == upper:
        return lower_values, (lower, upper)
    upper_values = jnp.take(field, upper, axis=fixed_axis)
    return (1.0 - weight) * lower_values + weight * upper_values, (lower, upper)


def _midplane_field(
    solution: Solution,
    field: jnp.ndarray,
    axis: str,
    fluid_only: bool,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    if axis == "y":
        coordinate, fixed_coordinate, fixed_axis = (
            solution.mesh.y_centers,
            solution.mesh.z_centers,
            1,
        )
    elif axis == "z":
        coordinate, fixed_coordinate, fixed_axis = (
            solution.mesh.z_centers,
            solution.mesh.y_centers,
            0,
        )
    else:
        raise ValueError(f"Unsupported axis {axis}")
    values, (lower, upper) = _midplane_values(field, fixed_coordinate, fixed_axis)
    if not fluid_only or solution.mesh.fluid_mask is None:
        return coordinate, values
    mask = jnp.take(solution.mesh.fluid_mask, lower, axis=fixed_axis)
    if lower != upper:
        mask &= jnp.take(solution.mesh.fluid_mask, upper, axis=fixed_axis)
    return coordinate[mask], values[mask]


def extract_centerline(solution: Solution) -> dict[str, jnp.ndarray]:
    return extract_midplane_profile(solution, axis="y")


def extract_midplane_profile(
    solution: Solution, axis: str = "y", fluid_only: bool = False
) -> dict[str, jnp.ndarray]:
    phi = getattr(solution.state, "phi", jnp.zeros_like(solution.state.u))
    coordinate, velocity = _midplane_field(solution, solution.state.u, axis, fluid_only)
    _, potential = _midplane_field(solution, phi, axis, fluid_only)
    return {axis: coordinate, "u": velocity, "phi": potential}


def extract_midplane_scalar_profile(
    solution: Solution,
    field: jnp.ndarray,
    *,
    axis: str = "y",
    fluid_only: bool = False,
) -> dict[str, jnp.ndarray]:
    coordinate, values = _midplane_field(solution, field, axis, fluid_only)
    return {"coordinate": coordinate, "value": values}


def compare_profile_to_reference(
    coordinate: jnp.ndarray,
    simulated: jnp.ndarray,
    reference: jnp.ndarray,
) -> AnalyticComparison:
    diff = simulated - reference
    l2 = float(jnp.sqrt(jnp.mean(diff**2)))
    linf = float(jnp.max(jnp.abs(diff)))
    return AnalyticComparison(
        coordinate=coordinate,
        simulated=simulated,
        reference=reference,
        l2_error=l2,
        linf_error=linf,
    )


def _normalization_scale(coordinate: jnp.ndarray, *, infer_boundary_extent: bool) -> jnp.ndarray:
    coordinate = jnp.asarray(coordinate, dtype=float)
    abs_max = jnp.max(jnp.abs(coordinate))
    abs_max = jnp.where(abs_max > 0.0, abs_max, 1.0)
    if not infer_boundary_extent or coordinate.size <= 1:
        return abs_max
    lower_step = coordinate[1] - coordinate[0]
    upper_step = coordinate[-1] - coordinate[-2]
    lower_bound = coordinate[0] - 0.5 * lower_step
    upper_bound = coordinate[-1] + 0.5 * upper_step
    inferred_extent = jnp.max(jnp.abs(jnp.asarray([lower_bound, upper_bound], dtype=coordinate.dtype)))
    return jnp.maximum(inferred_extent, abs_max)


def compare_normalized_profiles(
    simulated_coordinate: jnp.ndarray,
    simulated: jnp.ndarray,
    reference_coordinate: jnp.ndarray,
    reference: jnp.ndarray,
    *,
    simulated_boundary_values: tuple[float, float] | None = None,
) -> AnalyticComparison:
    """Compare profile shapes after normalizing their coordinates and peaks."""

    sim_coord = simulated_coordinate / _normalization_scale(simulated_coordinate, infer_boundary_extent=True)
    ref_coord = reference_coordinate / _normalization_scale(reference_coordinate, infer_boundary_extent=False)
    sim_scale = jnp.maximum(jnp.max(jnp.abs(simulated)), 1.0e-12)
    ref_scale = jnp.maximum(jnp.max(jnp.abs(reference)), 1.0e-12)
    normalized_simulated = simulated / sim_scale
    normalized_reference = reference / ref_scale
    sim_coord, normalized_simulated = _sorted_profile(sim_coord, normalized_simulated)
    if simulated_boundary_values is not None:
        lower_value, upper_value = simulated_boundary_values
        sim_coord, normalized_simulated = _extend_profile_with_boundary_values(
            sim_coord,
            normalized_simulated,
            lower_value=lower_value / float(sim_scale),
            upper_value=upper_value / float(sim_scale),
        )
    ref_coord, normalized_reference = _sorted_profile(ref_coord, normalized_reference)
    return compare_profile_to_reference(
        ref_coord,
        jnp.interp(ref_coord, sim_coord, normalized_simulated),
        normalized_reference,
    )


def _sorted_profile(coordinate: jnp.ndarray, values: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    order = jnp.argsort(coordinate)
    return coordinate[order], values[order]


def _extend_profile_with_boundary_values(
    coordinate: jnp.ndarray,
    values: jnp.ndarray,
    *,
    lower_coordinate: float = -1.0,
    upper_coordinate: float = 1.0,
    lower_value: float = 0.0,
    upper_value: float = 0.0,
    tolerance: float = 1.0e-12,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    coordinate = jnp.asarray(coordinate, dtype=float)
    values = jnp.asarray(values, dtype=float)
    lower_present = coordinate.size > 0 and abs(float(coordinate[0]) - lower_coordinate) <= tolerance
    upper_present = coordinate.size > 0 and abs(float(coordinate[-1]) - upper_coordinate) <= tolerance
    if lower_present:
        values = values.at[0].set(lower_value)
    else:
        coordinate = jnp.concatenate([jnp.asarray([lower_coordinate], dtype=coordinate.dtype), coordinate])
        values = jnp.concatenate([jnp.asarray([lower_value], dtype=values.dtype), values])
    if upper_present:
        values = values.at[-1].set(upper_value)
    else:
        coordinate = jnp.concatenate([coordinate, jnp.asarray([upper_coordinate], dtype=coordinate.dtype)])
        values = jnp.concatenate([values, jnp.asarray([upper_value], dtype=values.dtype)])
    return coordinate, values


def compare_profiles_with_shared_scale(
    simulated_coordinate: jnp.ndarray,
    simulated: jnp.ndarray,
    reference_coordinate: jnp.ndarray,
    reference: jnp.ndarray,
    *,
    coordinate_scale: float,
    value_scale: float,
    simulated_offset: float = 0.0,
    reference_offset: float = 0.0,
    simulated_boundary_values: tuple[float, float] | None = None,
) -> AnalyticComparison:
    """Compare profiles using declared common coordinate and observable scales.

    This function never fits each profile to its own peak. Gauge-like offsets
    must be supplied explicitly, keeping cross-code normalization auditable.
    """

    if coordinate_scale <= 0.0:
        raise ValueError("coordinate_scale must be positive")
    if value_scale <= 0.0:
        raise ValueError("value_scale must be positive")
    sim_coord = jnp.asarray(simulated_coordinate, dtype=float) / float(coordinate_scale)
    ref_coord = jnp.asarray(reference_coordinate, dtype=float) / float(coordinate_scale)
    scaled_simulated = (jnp.asarray(simulated, dtype=float) - float(simulated_offset)) / float(value_scale)
    scaled_reference = (jnp.asarray(reference, dtype=float) - float(reference_offset)) / float(value_scale)
    sim_coord, scaled_simulated = _sorted_profile(sim_coord, scaled_simulated)
    if simulated_boundary_values is not None:
        lower_value, upper_value = simulated_boundary_values
        sim_coord, scaled_simulated = _extend_profile_with_boundary_values(
            sim_coord,
            scaled_simulated,
            lower_value=(float(lower_value) - float(simulated_offset)) / float(value_scale),
            upper_value=(float(upper_value) - float(simulated_offset)) / float(value_scale),
        )
    ref_coord, scaled_reference = _sorted_profile(ref_coord, scaled_reference)
    interpolated_simulated = jnp.interp(ref_coord, sim_coord, scaled_simulated)
    return compare_profile_to_reference(ref_coord, interpolated_simulated, scaled_reference)


def symmetry_metrics(profile: jnp.ndarray, axis: str) -> ProfileSymmetry:
    mirrored = jnp.flip(profile)
    diff = profile - mirrored
    return ProfileSymmetry(
        axis=axis,
        mean_abs_error=float(jnp.mean(jnp.abs(diff))),
        max_abs_error=float(jnp.max(jnp.abs(diff))),
    )


def profile_sign_changes(profile: jnp.ndarray, *, tolerance: float = 1e-12) -> int:
    signs = jnp.where(jnp.abs(profile) <= tolerance, 0.0, jnp.sign(profile))
    left = signs[:-1]
    right = signs[1:]
    transitions = (left * right) < 0.0
    return int(jnp.sum(transitions))


def negative_fraction(profile: jnp.ndarray, *, tolerance: float = 1e-12) -> float:
    return float(jnp.mean((profile < -tolerance).astype(float)))


def duct_profile_metrics(solution: Solution) -> dict[str, float]:
    y_profile = extract_midplane_profile(solution, axis="y")["u"]
    z_profile = extract_midplane_profile(solution, axis="z")["u"]
    y_sym = symmetry_metrics(y_profile, axis="y")
    z_sym = symmetry_metrics(z_profile, axis="z")
    return {
        "symmetry_y_mean_abs_error": y_sym.mean_abs_error,
        "symmetry_y_max_abs_error": y_sym.max_abs_error,
        "symmetry_z_mean_abs_error": z_sym.mean_abs_error,
        "symmetry_z_max_abs_error": z_sym.max_abs_error,
        "centerline_y_sign_changes": float(profile_sign_changes(y_profile)),
        "centerline_z_sign_changes": float(profile_sign_changes(z_profile)),
        "centerline_y_negative_fraction": negative_fraction(y_profile),
        "centerline_z_negative_fraction": negative_fraction(z_profile),
        "u_max": float(jnp.max(solution.state.u)),
        "u_mean": float(jnp.mean(solution.state.u)),
    }


def validation_summary(solution: Solution, case_name: str, ha: float | None = None) -> dict[str, float | str]:
    payload: dict[str, float | str] = {
        "case": case_name,
        "time": solution.state.time,
        "residual": solution.state.residual,
    }
    for name, attribute, default in _SUMMARY_HISTORIES:
        history = getattr(solution.diagnostics, attribute)
        payload[name] = float(history[-1]) if history.size else default
    payload.update(duct_profile_metrics(solution))
    if case_name.startswith("hartmann") and ha is not None:
        comparison = hartmann_validation(solution, ha)
        payload["l2_error"] = comparison.l2_error
        payload["linf_error"] = comparison.linf_error
    return payload


def hartmann_validation(solution: Solution, ha: float) -> AnalyticComparison:
    profile = extract_centerline(solution)
    half_width = 0.5 * float(solution.mesh.y_faces[-1] - solution.mesh.y_faces[0])
    scale_y = half_width if half_width > 0.0 else float(jnp.max(jnp.abs(profile["y"])))
    coordinate = profile["y"] / max(scale_y, 1.0e-12)
    u = profile["u"]
    scale = jnp.max(jnp.abs(u))
    scale = jnp.where(scale > 0.0, scale, 1.0)
    normalized = u / scale
    reference = hartmann_analytic_profile(coordinate, ha)
    return compare_profile_to_reference(coordinate, normalized, reference)


def hartmann_acceptance(
    solution: Solution,
    ha: float,
    *,
    l2_threshold: float,
    linf_threshold: float,
) -> AcceptanceReport:
    comparison = hartmann_validation(solution, ha)
    passed_l2 = comparison.l2_error <= l2_threshold
    passed_linf = comparison.linf_error <= linf_threshold
    return AcceptanceReport(
        case_name=f"hartmann_ha{int(ha)}",
        l2_error=comparison.l2_error,
        linf_error=comparison.linf_error,
        l2_threshold=float(l2_threshold),
        linf_threshold=float(linf_threshold),
        passed_l2=passed_l2,
        passed_linf=passed_linf,
        passed=passed_l2 and passed_linf,
    )


def closed_channel_validation(
    solution: Solution,
    case_kind: str,
    ha: int,
    reference_root: str | Path | None = None,
) -> ClosedChannelValidation:
    reference: ClosedChannelAnalyticalReference = load_closed_channel_analytical(
        case_kind, ha, reference_root
    )
    y_profile = extract_midplane_profile(solution, axis="y", fluid_only=True)
    z_profile = extract_midplane_profile(solution, axis="z", fluid_only=True)
    y_comparison = compare_normalized_profiles(
        y_profile["y"],
        y_profile["u"],
        reference.coordinate,
        reference.midplane_y,
        simulated_boundary_values=(0.0, 0.0),
    )
    z_comparison = compare_normalized_profiles(
        z_profile["z"],
        z_profile["u"],
        reference.coordinate,
        reference.midplane_z,
        simulated_boundary_values=(0.0, 0.0),
    )
    return ClosedChannelValidation(
        case_kind=case_kind,
        ha=ha,
        y_profile=y_comparison,
        z_profile=z_comparison,
        reference_pressure_drop=reference.pressure_drop,
        reference_path=reference.path,
    )


def write_profile_csv(path: str | Path, data: dict[str, jnp.ndarray]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(data.keys())
    rows = zip(*(jnp.asarray(data[key]).tolist() for key in keys))
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(keys)
        writer.writerows(rows)
    return path


def _write_json(payload: object, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return path


def _comparison_payload(comparison: AnalyticComparison, axis_name: str = "coordinate") -> dict[str, object]:
    return {
        axis_name: jnp.asarray(comparison.coordinate).tolist(),
        "simulated": jnp.asarray(comparison.simulated).tolist(),
        "reference": jnp.asarray(comparison.reference).tolist(),
        "l2_error": comparison.l2_error,
        "linf_error": comparison.linf_error,
    }


def _profile_validation_payload(
    report: ClosedChannelValidation,
) -> dict[str, object]:
    payload = {name: value for name, value in vars(report).items() if name not in {"y_profile", "z_profile"}}
    for name in ("y_profile", "z_profile"):
        payload[name] = _comparison_payload(getattr(report, name))
    return payload


def write_analytic_comparison(
    comparison: AnalyticComparison, path: str | Path, axis_name: str = "coordinate"
) -> Path:
    return _write_json(_comparison_payload(comparison, axis_name), path)


def write_closed_channel_validation(report: ClosedChannelValidation, path: str | Path) -> Path:
    return _write_json(_profile_validation_payload(report), path)


def write_acceptance_report(report: AcceptanceReport, path: str | Path) -> Path:
    return _write_json(vars(report), path)


def write_metrics_json(metrics: dict[str, float | str], path: str | Path) -> Path:
    return _write_json(metrics, path)

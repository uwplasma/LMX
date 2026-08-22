"""Analytical, conservation, convergence, and parity validation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import re
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .cases import make_hartmann_case
from .mesh import StructuredMesh, generate_rect_duct_mesh_from_faces
from .specs import CaseSpec, Solution


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


@dataclass(frozen=True)
class ClosedChannelAnalyticalReference:
    case_kind: str
    ha: int
    coordinate: jnp.ndarray
    midplane_z: jnp.ndarray
    midplane_y: jnp.ndarray
    pressure_drop: float | None
    path: str


@dataclass(frozen=True)
class ProcessedSliceReference:
    case_kind: str
    ha: int
    columns: dict[str, jnp.ndarray]
    path: str


def default_closed_channel_reference_root(reference_root: str | Path | None = None) -> Path:
    if reference_root is not None:
        return Path(reference_root)
    repo_root = Path(__file__).resolve().parents[1]
    preferred = repo_root / "external" / "reference_data" / "ClosedChannel"
    if preferred.exists():
        return preferred
    return repo_root / "external" / "FreeMHDPaperAllFigures" / "FreeMHDPaperAllFigures" / "ClosedChannel"


def _match_single(patterns: list[str], reference_root: Path) -> Path:
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(sorted(reference_root.glob(pattern)))
    if not matches:
        raise FileNotFoundError(f"No reference files matched {patterns} under {reference_root}")
    return matches[0]


def analytical_reference_path(case_kind: str, ha: int, reference_root: str | Path | None = None) -> Path:
    root = default_closed_channel_reference_root(reference_root) / "AnalyticalSolutions"
    stem = case_kind.capitalize()
    patterns = [f"{stem}_Analytical_Ha{ha}_*.txt"]
    return _match_single(patterns, root)


def processed_slice_reference_path(
    case_kind: str,
    ha: int,
    x_slice: str = "1m",
    reference_root: str | Path | None = None,
) -> Path:
    root = default_closed_channel_reference_root(reference_root)
    patterns = [f"{case_kind}_*Ha{ha}_*XSlice{x_slice}_*.csv", f"{case_kind}_Ha{ha}_*XSlice{x_slice}_*.csv"]
    return _match_single(patterns, root)


def load_closed_channel_analytical(
    case_kind: str, ha: int, reference_root: str | Path | None = None
) -> ClosedChannelAnalyticalReference:
    """Load a bundled or user-supplied analytical closed-channel profile."""

    path = analytical_reference_path(case_kind, ha, reference_root)
    rows = path.read_text().strip().splitlines()
    _, *body = rows
    coordinate = []
    midplane_z = []
    midplane_y = []
    for row in body:
        radius, u1, u2 = row.split()
        coordinate.append(float(radius))
        midplane_z.append(float(u1))
        midplane_y.append(float(u2))
    match = re.search(r"PresDrop([0-9.]+)", path.name)
    pressure_drop = None if match is None else float(match.group(1).rstrip("."))
    return ClosedChannelAnalyticalReference(
        case_kind=case_kind,
        ha=ha,
        coordinate=jnp.asarray(coordinate),
        midplane_z=jnp.asarray(midplane_z),
        midplane_y=jnp.asarray(midplane_y),
        pressure_drop=pressure_drop,
        path=str(path),
    )


def load_shercliff_analytical(
    ha: int, reference_root: str | Path | None = None
) -> ClosedChannelAnalyticalReference:
    """Load an analytical Shercliff profile at the requested Hartmann number."""

    return load_closed_channel_analytical("shercliff", ha, reference_root)


def load_hunt_analytical(
    ha: int, reference_root: str | Path | None = None
) -> ClosedChannelAnalyticalReference:
    """Load an analytical Hunt profile at the requested Hartmann number."""

    return load_closed_channel_analytical("hunt", ha, reference_root)


def load_processed_slice(
    case_kind: str,
    ha: int,
    x_slice: str = "1m",
    reference_root: str | Path | None = None,
) -> ProcessedSliceReference:
    """Load a processed FreeMHD slice and its named field columns."""

    path = processed_slice_reference_path(case_kind, ha, x_slice=x_slice, reference_root=reference_root)
    with path.open() as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        columns: dict[str, list[float]] = {name: [] for name in fieldnames}
        for row in reader:
            for name in fieldnames:
                columns[name].append(float(row[name]))
    return ProcessedSliceReference(
        case_kind=case_kind,
        ha=ha,
        columns={name: jnp.asarray(values) for name, values in columns.items()},
        path=str(path),
    )


def _processed_field_column(
    reference: ProcessedSliceReference, field_name: str, component: int | None
) -> jnp.ndarray:
    column_name = field_name if component is None else f"{field_name}:{component}"
    try:
        return reference.columns[column_name]
    except KeyError as exc:
        available = ", ".join(sorted(reference.columns))
        raise KeyError(
            f"Processed slice {reference.path} has no column {column_name!r}; available columns: {available}"
        ) from exc


def _fill_missing_structured_values(grid: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    if not np.isnan(grid).any():
        return grid
    filled = np.array(grid, copy=True)
    for row_index in range(filled.shape[0]):
        row = filled[row_index, :]
        valid = np.isfinite(row)
        if valid.sum() >= 2:
            filled[row_index, :] = np.interp(z, z[valid], row[valid])
        elif valid.sum() == 1:
            filled[row_index, :] = row[valid][0]
    for column_index in range(filled.shape[1]):
        column = filled[:, column_index]
        valid = np.isfinite(column)
        if valid.sum() >= 2:
            filled[:, column_index] = np.interp(y, y[valid], column[valid])
        elif valid.sum() == 1:
            filled[:, column_index] = column[valid][0]
    if np.isnan(filled).any():
        fallback = float(np.nanmean(filled)) if np.isfinite(filled).any() else 0.0
        filled = np.where(np.isfinite(filled), filled, fallback)
    return filled


def processed_slice_field_grid(
    reference: ProcessedSliceReference,
    *,
    field_name: str,
    component: int | None = None,
) -> dict[str, jnp.ndarray]:
    """Return a structured ``(y, z)`` grid from a processed FreeMHD slice.

    ParaView/OpenFOAM slice exports can contain duplicated points at block
    interfaces and may not be ordered as a tensor grid. This helper averages
    duplicate point values before assembling a structured grid for quadrature,
    plotting, and reference-derived flow-rate targets.
    """

    points_y = np.asarray(reference.columns["Points:1"], dtype=float)
    points_z = np.asarray(reference.columns["Points:2"], dtype=float)
    values = np.asarray(_processed_field_column(reference, field_name, component), dtype=float)
    y = np.unique(points_y)
    z = np.unique(points_z)
    y_index = {float(value): index for index, value in enumerate(y)}
    z_index = {float(value): index for index, value in enumerate(z)}
    accumulator = np.zeros((y.size, z.size), dtype=float)
    counts = np.zeros((y.size, z.size), dtype=float)
    for point_y, point_z, value in zip(points_y, points_z, values, strict=True):
        iy = y_index[float(point_y)]
        iz = z_index[float(point_z)]
        accumulator[iy, iz] += float(value)
        counts[iy, iz] += 1.0
    grid = np.divide(accumulator, counts, out=np.full_like(accumulator, np.nan), where=counts > 0.0)
    grid = _fill_missing_structured_values(grid, y, z)
    return {"y": jnp.asarray(y), "z": jnp.asarray(z), "value": jnp.asarray(grid)}


def processed_slice_area_mean(
    reference: ProcessedSliceReference,
    *,
    field_name: str = "U",
    component: int | None = 0,
) -> float:
    """Compute an area-weighted mean over a processed ``y-z`` slice."""

    grid = processed_slice_field_grid(reference, field_name=field_name, component=component)
    y = np.asarray(grid["y"], dtype=float)
    z = np.asarray(grid["z"], dtype=float)
    values = np.asarray(grid["value"], dtype=float)
    if y.size < 2 or z.size < 2:
        return float(np.mean(values)) if values.size else 0.0
    area = (float(y[-1]) - float(y[0])) * (float(z[-1]) - float(z[0]))
    integral_z = np.trapezoid(values, z, axis=1)
    integral = float(np.trapezoid(integral_z, y))
    return integral / area


def processed_slice_point_mesh(
    reference: ProcessedSliceReference,
    *,
    length: float = 1.0,
    nx: int = 1,
) -> StructuredMesh:
    """Build a rectangular mesh whose cross-section faces match slice points."""

    y_faces = jnp.asarray(np.unique(np.asarray(reference.columns["Points:1"], dtype=float)))
    z_faces = jnp.asarray(np.unique(np.asarray(reference.columns["Points:2"], dtype=float)))
    return generate_rect_duct_mesh_from_faces(y_faces=y_faces, z_faces=z_faces, length=length, nx=nx)


def _unique_plane_profile(profile_coord: jnp.ndarray, values: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    coord_np = np.asarray(profile_coord, dtype=float)
    values_np = np.asarray(values, dtype=float)
    unique_coord = np.unique(coord_np)
    unique_values = np.asarray([np.mean(values_np[np.isclose(coord_np, coord)]) for coord in unique_coord])
    order = np.argsort(unique_coord)
    return jnp.asarray(unique_coord[order]), jnp.asarray(unique_values[order])


def _interpolated_centerline_profile(
    profile_coord: jnp.ndarray,
    cross_coord: jnp.ndarray,
    values: jnp.ndarray,
    *,
    tolerance: float = 1.0e-12,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    profile_np = np.asarray(profile_coord, dtype=float)
    cross_np = np.asarray(cross_coord, dtype=float)
    values_np = np.asarray(values, dtype=float)
    unique_cross = np.unique(cross_np)
    if unique_cross.size == 0:
        return jnp.asarray([]), jnp.asarray([])

    exact = unique_cross[np.isclose(unique_cross, 0.0, atol=tolerance, rtol=0.0)]
    if exact.size:
        target = float(exact[np.argmin(np.abs(exact))])
        mask = np.isclose(cross_np, target, atol=tolerance, rtol=0.0)
        return _unique_plane_profile(jnp.asarray(profile_np[mask]), jnp.asarray(values_np[mask]))

    lower_candidates = unique_cross[unique_cross < 0.0]
    upper_candidates = unique_cross[unique_cross > 0.0]
    if lower_candidates.size == 0 or upper_candidates.size == 0:
        target = float(unique_cross[np.argmin(np.abs(unique_cross))])
        mask = np.isclose(cross_np, target, atol=tolerance, rtol=0.0)
        return _unique_plane_profile(jnp.asarray(profile_np[mask]), jnp.asarray(values_np[mask]))

    lower = float(lower_candidates[np.argmax(lower_candidates)])
    upper = float(upper_candidates[np.argmin(upper_candidates)])
    lower_mask = np.isclose(cross_np, lower, atol=tolerance, rtol=0.0)
    upper_mask = np.isclose(cross_np, upper, atol=tolerance, rtol=0.0)
    lower_coord, lower_values = _unique_plane_profile(
        jnp.asarray(profile_np[lower_mask]), jnp.asarray(values_np[lower_mask])
    )
    upper_coord, upper_values = _unique_plane_profile(
        jnp.asarray(profile_np[upper_mask]), jnp.asarray(values_np[upper_mask])
    )
    coord = np.union1d(np.asarray(lower_coord, dtype=float), np.asarray(upper_coord, dtype=float))
    lower_interp = np.interp(
        coord, np.asarray(lower_coord, dtype=float), np.asarray(lower_values, dtype=float)
    )
    upper_interp = np.interp(
        coord, np.asarray(upper_coord, dtype=float), np.asarray(upper_values, dtype=float)
    )
    weight = (0.0 - lower) / (upper - lower)
    return jnp.asarray(coord), jnp.asarray((1.0 - weight) * lower_interp + weight * upper_interp)


def extract_processed_midplane_profile(
    reference: ProcessedSliceReference, axis: str = "y"
) -> dict[str, jnp.ndarray]:
    points_y = reference.columns["Points:1"]
    points_z = reference.columns["Points:2"]
    u_x = reference.columns["U:0"]
    pot_e = reference.columns.get("potE", jnp.zeros_like(u_x))

    if axis == "y":
        coordinate, u_values = _interpolated_centerline_profile(points_y, points_z, u_x)
        _, phi_values = _interpolated_centerline_profile(points_y, points_z, pot_e)
        return {
            "y": coordinate,
            "u": u_values,
            "phi": phi_values,
        }
    if axis == "z":
        coordinate, u_values = _interpolated_centerline_profile(points_z, points_y, u_x)
        _, phi_values = _interpolated_centerline_profile(points_z, points_y, pot_e)
        return {
            "z": coordinate,
            "u": u_values,
            "phi": phi_values,
        }
    raise ValueError(f"Unsupported axis {axis}")


def extract_processed_profile(
    reference: ProcessedSliceReference,
    *,
    axis: str,
    field_name: str,
    component: int | None = None,
) -> dict[str, jnp.ndarray]:
    points_y = reference.columns["Points:1"]
    points_z = reference.columns["Points:2"]
    if component is None:
        values = reference.columns[field_name]
    else:
        values = reference.columns[f"{field_name}:{component}"]

    if axis == "y":
        coordinate, profile_values = _interpolated_centerline_profile(points_y, points_z, values)
        return {
            "coordinate": coordinate,
            "value": profile_values,
        }
    if axis == "z":
        coordinate, profile_values = _interpolated_centerline_profile(points_z, points_y, values)
        return {
            "coordinate": coordinate,
            "value": profile_values,
        }
    raise ValueError(f"Unsupported axis {axis}")


BENCHMARK_B_SPEC_FILES = {
    "B1-fringing-pipe": "alex-b1-pipe.toml",
    "B2-fringing-square": "alex-b2-square.toml",
}
_MATCHED_SHARED_SECTIONS = """equations nondimensional_groups geometry magnetic_field wall
boundary_drive observable normalization""".split()
_MATCHED_ROLE_SECTIONS = ("mesh_coordinates", "stopping_rules")
_MATCHED_CONTRACT_SECTIONS = (*_MATCHED_SHARED_SECTIONS, *_MATCHED_ROLE_SECTIONS)
_MATCHED_PRODUCTION_ROLES = {
    "B1-fringing-pipe": "b1-production",
    "B2-fringing-square": "b2-production",
}
_FREEMHD_DISCRETIZATION_REFERENCE = {
    "repository_commit": "14b54a3e8e1a05b6ee4c98331995abaaae96e7a5",
    "openfoam_release": "v2206",
    "momentum_source": "MHD_Solvers/solvers/epotMultiRegionInterFoam/fluid/mhdUEqn.H",
    "momentum_source_sha256": "ce88d93bf0fd575809e373497335dcad17bd1c31b792449bec00820fc9e1fcc6",
    "electric_source": "MHD_Solvers/solvers/epotMultiRegionInterFoam/fluid/ePotEqn.H",
    "electric_source_sha256": "605058509958d14d2c44d10d0b671573027f712ec59fa56d9aa6611e8f38f0e1",
    "limiter_source": "OpenFOAM-v2206/src/finiteVolume/interpolation/surfaceInterpolation/limitedSchemes/limitedLinear/limitedLinear.H",
    "limiter_source_sha256": "f30f319041e1546703cc8ee20250d1f89c9187927b264b308f336fe0dae2b06e",
    "scheme_macro_source": "OpenFOAM-v2206/src/finiteVolume/interpolation/surfaceInterpolation/limitedSchemes/LimitedScheme/LimitedScheme.H",
    "scheme_macro_source_sha256": "c5cae1690ceebb94ee813601c0e71180b63c8f2e50b4624bc7a74b9654df580a",
    "limiter_registration_source": "OpenFOAM-v2206/src/finiteVolume/interpolation/surfaceInterpolation/limitedSchemes/limitedLinear/limitedLinear.C",
    "limiter_registration_source_sha256": "298d9603b9bfd2c02deb50e8382be8ac147f1ebcccecd69233c77909fd4a0094",
    "nvd_source": "OpenFOAM-v2206/src/finiteVolume/interpolation/surfaceInterpolation/limitedSchemes/LimitedScheme/NVDTVD.H",
    "nvd_source_sha256": "ee85d9b26c257ccdd3ab6d2fa0403daed9df30d63f483c35bea1586ce6d4fd07",
    "vector_transform_source": "OpenFOAM-v2206/src/finiteVolume/interpolation/surfaceInterpolation/limitedSchemes/LimitedScheme/LimitFuncs.C",
    "vector_transform_source_sha256": "f99e4011a0a4ac957c992493b9391796dd12191c16293ba07c127168640c53b3",
}


def _repository_root(root: str | Path | None = None) -> Path:
    """Return an override root or the benchmark data shipped with LMX."""

    return Path(root) if root is not None else Path(__file__).with_name("data")


def canonical_matched_b_contract(spec: dict[str, Any], role: str) -> dict[str, Any]:
    """Compose one role without allowing execution settings to override physics."""

    matched = spec.get("matched_contract")
    shared = matched.get("shared") if isinstance(matched, dict) else None
    roles = matched.get("roles") if isinstance(matched, dict) else None
    execution = roles.get(role) if isinstance(roles, dict) else None
    if not isinstance(shared, dict) or set(shared) != set(_MATCHED_SHARED_SECTIONS):
        raise ValueError("Benchmark B matched shared contract is incomplete")
    if not isinstance(execution, dict) or set(execution) != set(_MATCHED_ROLE_SECTIONS):
        raise ValueError(f"Benchmark B matched role {role!r} is unavailable or incomplete")
    return {**deepcopy(shared), **deepcopy(execution)}


def _validate_benchmark_b_spec(spec: dict[str, Any], root: Path) -> None:
    case_id = str(spec.get("id"))
    if case_id not in BENCHMARK_B_SPEC_FILES:
        raise ValueError(f"Unsupported Benchmark B id {case_id!r}")
    if spec.get("schema_version") != 1 or spec.get("status") != "frozen":
        raise ValueError("Benchmark B specification must use schema 1 and status=frozen")
    if spec.get("tolerances_frozen_before_production") is not True:
        raise ValueError("Benchmark B tolerances must be frozen before production")

    expected = {
        "B1-fringing-pipe": (6600.0, 10700.0, 0.027, "pipe_ogrid"),
        "B2-fringing-square": (2900.0, 540.0, 0.07, "square_duct"),
    }[case_id]
    actual = (
        float(spec["physics"]["hartmann_number"]),
        float(spec["physics"]["interaction_parameter"]),
        float(spec["wall"]["wall_conductance_ratio"]),
        str(spec["geometry"]["kind"]),
    )
    if actual != expected:
        raise ValueError(f"Benchmark B frozen parameters differ: {actual!r}")

    matched = spec.get("matched_contract")
    roles = matched.get("roles") if isinstance(matched, dict) else None
    expected_role = _MATCHED_PRODUCTION_ROLES[case_id]
    if not isinstance(matched, dict) or set(matched) != {"shared", "roles"}:
        raise ValueError("Benchmark B matched formulation contract is incomplete")
    expected_roles = {expected_role, "harness-smoke"} if case_id == "B2-fringing-square" else {expected_role}
    if not isinstance(roles, dict) or set(roles) != expected_roles:
        raise ValueError("Benchmark B matched production role differs")
    if case_id == "B2-fringing-square":
        encoded_smoke = json.dumps(roles["harness-smoke"], sort_keys=True, separators=(",", ":")).encode()
        if (
            hashlib.sha256(encoded_smoke).hexdigest()
            != "3ef7c6f58900629221bc83c90fe3afef8a656efddd1d455cd60a71e8b38ac4d5"
        ):
            raise ValueError("Benchmark B matched smoke role differs")
        smoke = spec.get("harness_smoke_execution")
        expected_smoke = {
            "output_schema_version": 1,
            "executed_steps": 2,
            "dt_absolute_tolerance": 1.0e-18,
            "courant_max": 0.4,
            "mass_balance_max": 1.0e-3,
            "current_balance_max": 1.0e-3,
            "interface_current_balance_max": 1.0e-3,
            "interface_current_activity_min": 1.0e-12,
            "restart_absolute_tolerance": 1.0e-12,
            "cross_code_courant_relative_tolerance": 0.05,
            "cross_code_courant_absolute_tolerance": 1.0e-8,
            "cross_code_pressure_rms_max": 0.16,
            "cross_code_pressure_linf_max": 0.32,
            "pressure_tolerance_basis": "second-order smoke-grid truncation scale: h=0.4, RMS<=h^2 and Linf<=2h^2",
            "outer_current_evidence": "zero normal current from the independently observed solid zeroGradient boundary; no reconstructed solid flux",
        }
        if smoke != expected_smoke:
            raise ValueError("Benchmark B harness smoke execution contract differs")
    elif "harness_smoke_execution" in spec:
        raise ValueError("Benchmark B1 cannot define the B2 harness smoke execution contract")
    contract = canonical_matched_b_contract(spec, expected_role)
    semantics = (
        contract["equations"].get("inertia"),
        contract["equations"].get("advection_discretization"),
        contract["equations"].get("advection_assembly"),
        contract["equations"].get("advection_vector_limiter"),
        contract["equations"].get("gradient_discretization"),
        float(contract["nondimensional_groups"].get("reynolds_number", 0.0)),
        *(
            contract["boundary_drive"].get(name)
            for name in (
                "flow_constraint_scope",
                "pressure_outlet",
                "electric_axial_ends",
            )
        ),
    )
    if semantics != (
        "conservative div(rhoPhi,U)",
        "Gauss limitedLinear 1.0",
        "implicit fvm::div with frozen rhoPhi and limiter weights",
        "single magSqr(U) limiter applied to all components",
        "cellLimited leastSquares 1.0",
        expected[0] ** 2 / expected[1],
        "inlet face only",
        "fixed gauge",
        "zero normal current",
    ):
        raise ValueError("Benchmark B matched formulation semantics differ")
    if case_id == "B2-fringing-square" and int(contract["stopping_rules"].get("steady_steps_min", 0)) != 3:
        raise ValueError("Benchmark B matched stopping contract differs")

    if spec.get("free_mhd_discretization_reference") != _FREEMHD_DISCRETIZATION_REFERENCE:
        raise ValueError("Benchmark B FreeMHD discretization reference differs")

    sources = spec.get("sources")
    if not isinstance(sources, list) or {source.get("id") for source in sources} != {
        "smolentsev-vv",
        "alex-results-1987",
    }:
        raise ValueError("Benchmark B sources must identify both review and primary ALEX evidence")
    for source in sources:
        if not source.get("pages") or not source.get("figures") or len(source.get("sha256", "")) != 64:
            raise ValueError("Every Benchmark B source needs pages, figures, and SHA-256")

    field = spec["field"]
    if (
        field.get("representation") != "tabulated monotone interpolation"
        or field.get("no_extrapolation") is not True
        or "exactly divergence free" not in field.get("divergence_model", "")
        or float(field.get("divergence_acceptance", math.inf)) > 1.0e-8
    ):
        raise ValueError("Benchmark B field reconstruction contract is incomplete")

    levels = spec["mesh"].get("levels")
    if not isinstance(levels, list) or [level.get("name") for level in levels] != [
        "coarse",
        "medium",
        "fine",
    ]:
        raise ValueError("Benchmark B requires coarse, medium, and fine mesh levels")
    for key in (
        "axial_stations_min",
        "hartmann_layer_cells_min",
        "side_layer_cells_min",
    ):
        values = [int(level[key]) for level in levels]
        if values != sorted(values) or len(set(values)) != 3:
            raise ValueError(f"Benchmark B mesh requirement {key} must increase by level")
    resolution_key = "radial_cells_min" if case_id == "B1-fringing-pipe" else "cross_section_cells_min"
    resolution = [int(level[resolution_key]) for level in levels]
    if resolution != sorted(resolution) or len(set(resolution)) != 3:
        raise ValueError(f"Benchmark B mesh requirement {resolution_key} must increase by level")
    wall = spec["wall"]
    if (
        not str(wall.get("numerical_realization", "")).startswith("explicit volumetric")
        or float(wall.get("nominal_thickness_over_L", 0.0)) <= 0.0
        or float(wall.get("confirmation_thickness_over_L", math.inf))
        >= float(wall.get("nominal_thickness_over_L", 0.0))
        or float(wall.get("thickness_independence_relative_max", math.inf)) > 0.02
    ):
        raise ValueError("Benchmark B thin-wall numerical realization is incomplete")

    acceptance = spec["acceptance"]
    if (
        float(acceptance.get("weighted_rms_max", math.inf)) != 1.0
        or float(acceptance.get("weighted_linf_max", math.inf)) != 2.0
        or acceptance.get("matched_freemhd_case_required") is not True
        or acceptance.get("primary_before_secondary") is not True
    ):
        raise ValueError("Benchmark B uncertainty-aware acceptance contract is incomplete")
    solver = spec["solver"]
    reference_uncertainty = float(spec["reference"]["combined_uncertainty_absolute"])
    steady_uncertainty_fraction = float(solver.get("steady_residual_uncertainty_fraction_max", math.inf))
    expected_elliptic_controls = {
        "B1-fringing-pipe": (4000, 1.0e-12, 4000, 1.0e-12),
        "B2-fringing-square": (600, 1.0e-12, 4000, 1.0e-12),
    }[case_id]
    actual_elliptic_controls = (
        int(solver.get("electric_iterations_min", 0)),
        float(solver.get("electric_tolerance_max", math.inf)),
        int(solver.get("projection_iterations_min", 0)),
        float(solver.get("projection_tolerance_max", math.inf)),
    )
    acceleration = solver.get("coupling_acceleration")
    if (
        acceleration not in {"aitken", "anderson", "none"}
        or (acceleration == "anderson" and int(solver.get("coupling_history_depth", 0)) < 1)
        or float(solver.get("coupling_regularization", -1.0)) < 0.0
        or not 0.0 <= float(solver.get("coupling_damping", math.inf)) <= 1.0
        or (
            acceleration == "aitken"
            and (
                float(solver.get("coupling_min_relaxation", 0.0)) <= 0.0
                or float(solver.get("coupling_max_relaxation", 0.0))
                < float(solver.get("coupling_min_relaxation", 0.0))
            )
        )
        or steady_uncertainty_fraction > 0.05
        or (case_id == "B2-fringing-square" and int(solver.get("steady_steps_min", 0)) != 3)
        or float(solver.get("steady_residual_max", math.inf))
        > steady_uncertainty_fraction * reference_uncertainty
        or actual_elliptic_controls != expected_elliptic_controls
        or float(solver.get("tolerance_independence_factor", math.inf)) != 0.5
        or float(solver.get("tolerance_independence_uncertainty_fraction_max", math.inf)) > 0.25
        or float(solver.get("iteration_independence_factor", 0.0)) != 2.0
        or float(solver.get("iteration_independence_uncertainty_fraction_max", math.inf)) > 0.25
        or solver.get("time_or_iteration_independence_required") is not True
    ):
        raise ValueError("Benchmark B tolerance and iteration independence contract is incomplete")
    rights = spec["data_rights"]
    if "extracted numerical facts" not in rights.get("redistribution", ""):
        raise ValueError("Benchmark B reference-data redistribution policy is missing")

    reference = spec["reference"]
    data_path = root / str(reference["data_path"])
    if not data_path.is_file() or hashlib.sha256(data_path.read_bytes()).hexdigest() != reference.get(
        "data_sha256"
    ):
        raise ValueError("Benchmark B reference data are missing or fail SHA-256")


def load_benchmark_b_spec(case_id: str, root: str | Path | None = None) -> dict[str, Any]:
    """Load and validate one frozen ALEX Benchmark B specification."""

    repository = _repository_root(root)
    try:
        filename = BENCHMARK_B_SPEC_FILES[case_id]
    except KeyError as exc:
        raise ValueError(f"Unsupported Benchmark B id {case_id!r}") from exc
    try:
        import tomllib
    except ImportError:  # pragma: no cover - Python 3.10
        import tomli as tomllib
    with (repository / "benchmarks" / "specs" / filename).open("rb") as handle:
        spec = tomllib.load(handle)
    _validate_benchmark_b_spec(spec, repository)
    return spec


def load_benchmark_b_reference(case_id: str, root: str | Path | None = None) -> dict[str, tuple[float, ...]]:
    """Load checksummed field and pressure anchors for Benchmark B."""

    repository = _repository_root(root)
    spec = load_benchmark_b_spec(case_id, repository)
    path = repository / spec["reference"]["data_path"]
    columns = (
        "x_over_L",
        "b_over_B0",
        "b_uncertainty",
        "pressure_observable",
        "pressure_uncertainty",
    )
    values = {name: [] for name in columns}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != columns:
            raise ValueError("Benchmark B reference columns do not match the frozen schema")
        for row in reader:
            for name in columns:
                value = float(row[name])
                if not math.isfinite(value):
                    raise ValueError("Benchmark B reference values must be finite")
                values[name].append(value)
    if len(values["x_over_L"]) < 10 or any(
        right <= left for left, right in zip(values["x_over_L"], values["x_over_L"][1:])
    ):
        raise ValueError("Benchmark B reference coordinates must be strictly increasing")
    geometry = spec["geometry"]
    if values["x_over_L"][0] != float(geometry["x_over_L_min"]) or values["x_over_L"][-1] != float(
        geometry["x_over_L_max"]
    ):
        raise ValueError("Benchmark B reference data do not span the frozen domain")
    if any(value <= 0.0 for value in values["b_uncertainty"] + values["pressure_uncertainty"]):
        raise ValueError("Benchmark B reference uncertainties must be positive")
    if any(value < 0.0 or value > 1.05 for value in values["b_over_B0"]):
        raise ValueError("Benchmark B normalized magnetic field is outside its physical range")
    return {name: tuple(column) for name, column in values.items()}


def build_benchmark_b_field_profile(
    case_id: str,
    *,
    axial_stations: int,
    root: str | Path | None = None,
):
    """Reconstruct the frozen ALEX field on cell-centred axial stations.

    The computational coordinate starts at the upstream end of the published
    domain, while the returned profile retains the literature coordinate
    ``x/L`` (whose zero is the magnet pole face).  Linear interpolation is
    shape preserving for the frozen monotone anchors and never extrapolates.
    """

    if axial_stations < 2:
        raise ValueError("axial_stations must be at least 2")
    spec = load_benchmark_b_spec(case_id, root)
    reference = load_benchmark_b_reference(case_id, root)
    x_min = float(spec["geometry"]["x_over_L_min"])
    x_max = float(spec["geometry"]["x_over_L_max"])
    dx = (x_max - x_min) / axial_stations
    x_over_l = jnp.linspace(
        x_min + 0.5 * dx,
        x_max - 0.5 * dx,
        axial_stations,
    )
    anchors_x = jnp.asarray(reference["x_over_L"], dtype=float)
    anchors_b = jnp.asarray(reference["b_over_B0"], dtype=float)
    if float(x_over_l[0]) < float(anchors_x[0]) or float(x_over_l[-1]) > float(anchors_x[-1]):
        raise ValueError("ALEX field reconstruction cannot extrapolate")
    field_scale = jnp.interp(x_over_l, anchors_x, anchors_b)
    if bool(jnp.any(jnp.diff(field_scale) > 1.0e-12)):
        raise ValueError("ALEX field reconstruction must remain monotone")

    from .specs import FringingProfile

    return FringingProfile(x=x_over_l, field_scale=field_scale, axis="y")


def build_benchmark_b_problem(
    case_id: str,
    *,
    mesh_level: str,
    root: str | Path | None = None,
    wall_realization: str = "nominal",
    num_devices: int | None = None,
):
    """Build one immutable nondimensional ALEX B1/B2 production problem.

    A sharded mesh rounds the frozen axial minimum upward to the nearest
    multiple of ``num_devices``; cross-section resolution and physics remain
    unchanged.
    """

    from .specs import (
        BoundaryCondition,
        CaseSpec,
        ExtrudedInductionlessProblem,
        GeometrySpec,
        MagneticFieldSpec,
        RegionSpec,
        SolverConfig,
        TimeStepperConfig,
    )

    spec = load_benchmark_b_spec(case_id, root)
    levels = {str(level["name"]): level for level in spec["mesh"]["levels"]}
    if mesh_level not in levels:
        raise ValueError("mesh_level must be 'coarse', 'medium', or 'fine'")
    if wall_realization not in {"nominal", "confirmation"}:
        raise ValueError("wall_realization must be 'nominal' or 'confirmation'")
    level = levels[mesh_level]
    wall = spec["wall"]
    thickness = float(wall[f"{wall_realization}_thickness_over_L"])
    conductance = float(wall["wall_conductance_ratio"])
    wall_conductivity = conductance / thickness
    ha = float(spec["physics"]["hartmann_number"])
    interaction = float(spec["physics"]["interaction_parameter"])
    reynolds = ha**2 / interaction
    viscosity = 1.0 / reynolds
    peak_field = math.sqrt(interaction)
    if num_devices is not None and num_devices < 1:
        raise ValueError("num_devices must be positive")
    nx_min = int(level["axial_stations_min"])
    nx = math.ceil(nx_min / num_devices) * num_devices if num_devices is not None else nx_min
    wall_cells = int(level["side_layer_cells_min"])
    x_min = float(spec["geometry"]["x_over_L_min"])
    length = float(spec["geometry"]["x_over_L_max"]) - x_min

    if case_id == "B1-fringing-pipe":
        geometry = GeometrySpec(
            kind="pipe_ogrid",
            width=2.0,
            height=2.0,
            radius=1.0,
            length=length,
            axial_origin=x_min,
            nx=nx,
            nr=int(level["radial_cells_min"]),
            ntheta=int(level["azimuthal_cells_min"]),
            wall_thickness=(thickness,) * 4,
            wall_cells=(wall_cells,) * 4,
            target_ha=ha,
            hartmann_layer_cells=int(level["hartmann_layer_cells_min"]),
        )
    else:
        cross_cells = int(level["cross_section_cells_min"])
        geometry = GeometrySpec(
            kind="layered_duct",
            width=2.0,
            height=2.0,
            length=length,
            axial_origin=x_min,
            nx=nx,
            ny=cross_cells,
            nz=cross_cells,
            wall_thickness=(thickness,) * 4,
            wall_cells=(wall_cells,) * 4,
            target_ha=ha,
            hartmann_layer_cells=int(level["hartmann_layer_cells_min"]),
        )

    case = CaseSpec(
        name=f"alex_{case_id.lower()}_{mesh_level}_{wall_realization}",
        geometry=geometry,
        regions=(
            RegionSpec("fluid", "fluid", 1.0, 1.0, viscosity),
            RegionSpec("conducting_wall", "solid", wall_conductivity, 1.0, viscosity, thickness),
        ),
        magnetic_field=MagneticFieldSpec(kind="constant", value=(0.0, peak_field, 0.0)),
        boundary_conditions=(
            BoundaryCondition("walls", "no_slip"),
            BoundaryCondition(
                "inlet",
                "inlet_flow_rate",
                value=float(spec["drive"]["nondimensional_flow_rate"]),
                axis="x",
            ),
            BoundaryCondition("outlet", "outlet_pressure", value=0.0, axis="x"),
            BoundaryCondition(
                "uniform_conducting_wall",
                "conducting_wall",
                region="conducting_wall",
                side="left,right,bottom,top",
            ),
        ),
        time_stepper=TimeStepperConfig(
            dt=0.01,
            t_final=10.0,
            max_steps=1000,
            potential_iterations=400,
            steady_tolerance=float(spec["solver"]["steady_residual_max"]),
        ),
        solver=SolverConfig(
            kind="extruded_inductionless",
            coupling_iterations=64,
            coupling_tolerance=float(spec["solver"]["steady_residual_max"]),
            coupling_acceleration=str(spec["solver"]["coupling_acceleration"]),
            coupling_history_depth=int(spec["solver"]["coupling_history_depth"]),
            coupling_regularization=float(spec["solver"]["coupling_regularization"]),
            coupling_damping=float(spec["solver"]["coupling_damping"]),
            coupling_min_relaxation=float(spec["solver"].get("coupling_min_relaxation", 0.05)),
            coupling_max_relaxation=float(spec["solver"].get("coupling_max_relaxation", 100.0)),
        ),
        forcing=0.0,
        initial_velocity=1.0,
        reference_pressure_gradient=-1.0,
        notes=(
            f"Frozen {case_id} nondimensionalization: Re=Ha^2/N={reynolds:.12g}; "
            "reported axial pressure loss is -dp/dx and transverse pressure is high-z minus low-z."
        ),
    )
    profile = build_benchmark_b_field_profile(case_id, axial_stations=nx, root=root)
    return ExtrudedInductionlessProblem(case=case, profile=profile)


def benchmark_b_pressure_observable(solution, case_id: str) -> jnp.ndarray:
    """Return the frozen primary pressure observable in ALEX normalization."""

    spec = load_benchmark_b_spec(case_id)
    interaction = float(spec["physics"]["interaction_parameter"])
    if case_id == "B1-fringing-pipe":
        gradient = jnp.asarray(solution.bundle.axial_pressure_loss_gradient)
        if gradient.size == 0:
            raise ValueError("B1 requires the direct axial pressure-loss gradient")
        x = jnp.asarray(getattr(solution.bundle, "x", jnp.zeros((0,))))
        plateau_start = float(spec["reference"]["downstream_plateau_x_over_L_min"])
        if x.shape == gradient.shape and bool(jnp.any(x >= plateau_start)):
            downstream = jnp.nanmean(jnp.where(x >= plateau_start, gradient, jnp.nan))
        else:
            downstream = jnp.mean(gradient[-max(3, gradient.size // 10) :])
        return gradient / interaction - downstream / interaction
    difference = jnp.asarray(solution.bundle.transverse_pressure_difference)
    if difference.size == 0:
        raise ValueError("B2 requires the direct transverse pressure difference")
    x = jnp.asarray(getattr(solution.bundle, "x", jnp.zeros((0,))))
    if x.shape == difference.shape:
        upstream = float(spec["reference"]["baseline_x_over_L_upstream_max"])
        downstream = float(spec["reference"]["baseline_x_over_L_downstream_min"])
        baseline_mask = (x <= upstream) | (x >= downstream)
        if bool(jnp.any(baseline_mask)):
            baseline = jnp.nanmean(jnp.where(baseline_mask, difference, jnp.nan))
            difference = difference - baseline
    return difference / interaction


def benchmark_solver(
    repeats: int = 3, ha: float = 20.0, ny: int = 48, nz: int = 48
) -> dict[str, float | str]:
    from .cases import solve_steady

    case = make_hartmann_case(ha=ha, ny=ny, nz=nz)
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        solve_steady(case)
        timings.append(time.perf_counter() - start)
    cold = timings[0]
    warm = min(timings[1:] or timings)
    return {
        "case": case.name,
        "ha": ha,
        "ny": float(ny),
        "nz": float(nz),
        "repeats": float(repeats),
        "cold_seconds": cold,
        "warm_seconds": warm,
        "mean_seconds": sum(timings) / len(timings),
        "backend": jax.default_backend(),
        "device_kind": jax.devices()[0].device_kind,
        "jax_version": jax.__version__,
        "python_version": platform.python_version(),
    }


def write_benchmark_report(report: dict[str, float | str], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    return path

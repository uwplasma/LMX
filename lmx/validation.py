from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from math import log
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from .core import Solution
from .mesh import StructuredMesh
from .operators import center_coordinates
from .reference_data import (
    ClosedChannelAnalyticalReference,
    ProcessedSliceReference,
    extract_processed_midplane_profile,
    load_closed_channel_analytical,
    load_processed_slice,
)
from .specs import CaseSpec
from .solvers import solve_transient


@dataclass(frozen=True)
class ValidationReport:
    case_name: str
    metrics: dict[str, float]
    artifacts: dict[str, str]


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
class ProcessedSliceValidation:
    case_kind: str
    ha: int
    x_slice: str
    y_profile: AnalyticComparison
    z_profile: AnalyticComparison
    reference_path: str


@dataclass(frozen=True)
class ReferenceProfileValidation:
    sample_time: float
    y_profile: AnalyticComparison
    z_profile: AnalyticComparison
    y_path: str
    z_path: str


@dataclass(frozen=True)
class ReferenceCaseInspection:
    case_dir: str
    control_dicts: tuple[str, ...]
    fv_schemes: tuple[str, ...]
    fv_solutions: tuple[str, ...]
    region_properties: tuple[str, ...]
    block_mesh_dicts: tuple[str, ...]
    boundary_field_dirs: tuple[str, ...]
    latest_time_dirs: tuple[str, ...]
    region_zero_dirs: tuple[str, ...]
    zero_field_files: tuple[str, ...]
    processor_layout_dirs: tuple[str, ...]
    parallel_time_dirs: tuple[str, ...]


@dataclass(frozen=True)
class FieldMinMaxRecord:
    time: float
    field: str
    min_value: float
    max_value: float
    min_location: tuple[float, float, float] | None = None
    max_location: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class ReferenceLineSample:
    path: str
    distance: jnp.ndarray
    pot_e: jnp.ndarray
    u_x: jnp.ndarray
    u_y: jnp.ndarray
    u_z: jnp.ndarray


@dataclass(frozen=True)
class SamplingGeometry:
    x_position: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float


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
    if axis == "z":
        coordinates = mesh.z_centers
        widths = mesh.dz
        if fluid_mask is None:
            return coordinates, widths
        mid_y = int(jnp.argmax(jnp.sum(fluid_mask, axis=1)))
        mask = fluid_mask[mid_y, :]
        return coordinates[mask], widths[mask]
    raise ValueError(f"Unsupported axis {axis}")


def _cells_across_layer(widths: jnp.ndarray, layer_thickness: float) -> float:
    if widths.size == 0 or layer_thickness <= 0.0:
        return 0.0
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

    hartmann_half_spacing = 0.5 * float(hartmann_coordinates[-1] - hartmann_coordinates[0] + 0.5 * (hartmann_widths[0] + hartmann_widths[-1]))
    side_half_spacing = 0.5 * float(side_coordinates[-1] - side_coordinates[0] + 0.5 * (side_widths[0] + side_widths[-1]))
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


def hartmann_analytic_profile(y: jnp.ndarray, ha: float) -> jnp.ndarray:
    denom = jnp.cosh(ha) - 1.0
    denom = jnp.where(jnp.abs(denom) < 1e-12, 1.0, denom)
    return 1.0 - (jnp.cosh(ha * y) - 1.0) / denom


def _exact_coordinate_index(coordinate: jnp.ndarray, *, target: float = 0.0, tolerance: float = 1.0e-12) -> int | None:
    coordinate = jnp.asarray(coordinate, dtype=float)
    if coordinate.size == 0:
        return None
    matches = np.where(np.abs(np.asarray(coordinate, dtype=float) - float(target)) <= tolerance)[0]
    if matches.size == 0:
        return None
    return int(matches[0])


def extract_centerline(solution: Solution) -> dict[str, jnp.ndarray]:
    z_coords = solution.mesh.z_centers
    phi_state = getattr(solution.state, "phi", jnp.zeros_like(solution.state.u))
    if z_coords.size == 1:
        u_profile = solution.state.u[:, 0]
        phi_profile = phi_state[:, 0]
    else:
        center_index = _exact_coordinate_index(z_coords)
        if center_index is not None:
            u_profile = solution.state.u[:, center_index]
            phi_profile = phi_state[:, center_index]
        else:
            right = int(jnp.searchsorted(z_coords, 0.0))
            right = max(1, min(right, z_coords.size - 1))
            left = right - 1
            z_left = float(z_coords[left])
            z_right = float(z_coords[right])
            if abs(z_right - z_left) <= 1.0e-12:
                weight = 0.5
            else:
                weight = float((0.0 - z_left) / (z_right - z_left))
            weight = min(max(weight, 0.0), 1.0)
            u_profile = (1.0 - weight) * solution.state.u[:, left] + weight * solution.state.u[:, right]
            phi_profile = (1.0 - weight) * phi_state[:, left] + weight * phi_state[:, right]
    return {
        "y": solution.mesh.y_centers,
        "u": u_profile,
        "phi": phi_profile,
    }


def _profile_axis_mask(solution: Solution, axis: str, fixed_index: int) -> jnp.ndarray:
    fluid_mask = solution.mesh.fluid_mask
    if fluid_mask is None:
        if axis == "y":
            return jnp.ones(solution.state.u.shape[0], dtype=bool)
        if axis == "z":
            return jnp.ones(solution.state.u.shape[1], dtype=bool)
        raise ValueError(f"Unsupported axis {axis}")
    if axis == "y":
        return fluid_mask[:, fixed_index]
    if axis == "z":
        return fluid_mask[fixed_index, :]
    raise ValueError(f"Unsupported axis {axis}")


def extract_midplane_profile(solution: Solution, axis: str = "y", fluid_only: bool = False) -> dict[str, jnp.ndarray]:
    if axis == "y":
        profile = extract_centerline(solution)
        if not fluid_only:
            return profile
        z_coords = solution.mesh.z_centers
        if z_coords.size == 1:
            mask = _profile_axis_mask(solution, axis="y", fixed_index=0)
        else:
            center_index = _exact_coordinate_index(z_coords)
            if center_index is not None:
                mask = _profile_axis_mask(solution, axis="y", fixed_index=center_index)
            else:
                right = int(jnp.searchsorted(z_coords, 0.0))
                right = max(1, min(right, z_coords.size - 1))
                left = right - 1
                mask = _profile_axis_mask(solution, axis="y", fixed_index=left) & _profile_axis_mask(
                    solution,
                    axis="y",
                    fixed_index=right,
                )
        return {
            "y": profile["y"][mask],
            "u": profile["u"][mask],
            "phi": profile["phi"][mask],
        }
    if axis == "z":
        y_coords = solution.mesh.y_centers
        phi_state = getattr(solution.state, "phi", jnp.zeros_like(solution.state.u))
        if y_coords.size == 1:
            u_profile = solution.state.u[0, :]
            phi_profile = phi_state[0, :]
            mask = _profile_axis_mask(solution, axis="z", fixed_index=0)
        else:
            center_index = _exact_coordinate_index(y_coords)
            if center_index is not None:
                u_profile = solution.state.u[center_index, :]
                phi_profile = phi_state[center_index, :]
                mask = _profile_axis_mask(solution, axis="z", fixed_index=center_index)
            else:
                upper = int(jnp.searchsorted(y_coords, 0.0))
                upper = max(1, min(upper, y_coords.size - 1))
                lower = upper - 1
                y_lower = float(y_coords[lower])
                y_upper = float(y_coords[upper])
                if abs(y_upper - y_lower) <= 1.0e-12:
                    weight = 0.5
                else:
                    weight = float((0.0 - y_lower) / (y_upper - y_lower))
                weight = min(max(weight, 0.0), 1.0)
                u_profile = (1.0 - weight) * solution.state.u[lower, :] + weight * solution.state.u[upper, :]
                phi_profile = (1.0 - weight) * phi_state[lower, :] + weight * phi_state[upper, :]
                mask = _profile_axis_mask(solution, axis="z", fixed_index=lower) & _profile_axis_mask(
                    solution,
                    axis="z",
                    fixed_index=upper,
                )
        profile = {
            "z": solution.mesh.z_centers,
            "u": u_profile,
            "phi": phi_profile,
        }
        if not fluid_only:
            return profile
        return {
            "z": profile["z"][mask],
            "u": profile["u"][mask],
            "phi": profile["phi"][mask],
        }
    raise ValueError(f"Unsupported axis {axis}")


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


def compare_normalized_profiles(
    simulated_coordinate: jnp.ndarray,
    simulated: jnp.ndarray,
    reference_coordinate: jnp.ndarray,
    reference: jnp.ndarray,
    *,
    simulated_boundary_values: tuple[float, float] | None = None,
) -> AnalyticComparison:
    sim_scale_coord = _normalization_scale(simulated_coordinate, infer_boundary_extent=True)
    ref_scale_coord = _normalization_scale(reference_coordinate, infer_boundary_extent=False)
    sim_coord = simulated_coordinate / sim_scale_coord
    ref_coord = reference_coordinate / ref_scale_coord
    sim_scale = jnp.max(jnp.abs(simulated))
    ref_scale = jnp.max(jnp.abs(reference))
    sim_scale = jnp.where(sim_scale > 0.0, sim_scale, 1.0)
    ref_scale = jnp.where(ref_scale > 0.0, ref_scale, 1.0)
    normalized_simulated = simulated / sim_scale
    normalized_reference = reference / ref_scale
    sim_coord, normalized_simulated = _sorted_profile(sim_coord, normalized_simulated)
    if simulated_boundary_values is not None:
        normalized_simulated = jnp.asarray(normalized_simulated, dtype=float)
        lower_value, upper_value = simulated_boundary_values
        normalized_simulated = normalized_simulated.astype(float)
        sim_coord, normalized_simulated = _extend_profile_with_boundary_values(
            sim_coord,
            normalized_simulated,
            lower_value=lower_value / float(sim_scale),
            upper_value=upper_value / float(sim_scale),
        )
    ref_coord, normalized_reference = _sorted_profile(ref_coord, normalized_reference)
    interpolated_simulated = jnp.interp(ref_coord, sim_coord, normalized_simulated)
    return compare_profile_to_reference(ref_coord, interpolated_simulated, normalized_reference)


def symmetry_metrics(profile: jnp.ndarray, axis: str) -> ProfileSymmetry:
    mirrored = jnp.flip(profile)
    diff = profile - mirrored
    return ProfileSymmetry(
        axis=axis,
        mean_abs_error=float(jnp.mean(jnp.abs(diff))),
        max_abs_error=float(jnp.max(jnp.abs(diff))),
    )


def profile_sign_changes(profile: jnp.ndarray, *, tolerance: float = 1e-12) -> int:
    if profile.size <= 1:
        return 0
    signs = jnp.where(jnp.abs(profile) <= tolerance, 0.0, jnp.sign(profile))
    left = signs[:-1]
    right = signs[1:]
    transitions = (left * right) < 0.0
    return int(jnp.sum(transitions))


def negative_fraction(profile: jnp.ndarray, *, tolerance: float = 1e-12) -> float:
    if profile.size == 0:
        return 0.0
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
        "potential_residual": float(solution.diagnostics.potential_residual_history[-1])
        if solution.diagnostics.potential_residual_history.size
        else 0.0,
        "potential_iterations_used": float(solution.diagnostics.potential_iterations_history[-1])
        if solution.diagnostics.potential_iterations_history.size
        else 0.0,
        "mean_velocity": float(solution.diagnostics.mean_velocity_history[-1])
        if solution.diagnostics.mean_velocity_history.size
        else 0.0,
        "applied_forcing": float(solution.diagnostics.applied_forcing_history[-1])
        if solution.diagnostics.applied_forcing_history.size
        else 0.0,
        "pressure_proxy": float(solution.diagnostics.pressure_proxy_history[-1])
        if solution.diagnostics.pressure_proxy_history.size
        else 0.0,
        "current_scaled_pressure_proxy": float(solution.diagnostics.current_scaled_pressure_proxy_history[-1])
        if solution.diagnostics.current_scaled_pressure_proxy_history.size
        else 0.0,
        "linear_residual": float(solution.diagnostics.linear_residual_history[-1])
        if solution.diagnostics.linear_residual_history.size
        else 0.0,
        "linear_iterations_used": float(solution.diagnostics.linear_iterations_history[-1])
        if solution.diagnostics.linear_iterations_history.size
        else 0.0,
        "volumetric_flow_rate": float(solution.diagnostics.volumetric_flow_rate_history[-1])
        if solution.diagnostics.volumetric_flow_rate_history.size
        else 0.0,
        "mean_current_magnitude": float(solution.diagnostics.mean_current_magnitude_history[-1])
        if solution.diagnostics.mean_current_magnitude_history.size
        else 0.0,
        "lorentz_power": float(solution.diagnostics.lorentz_power_history[-1])
        if solution.diagnostics.lorentz_power_history.size
        else 0.0,
        "div_current_max": float(solution.diagnostics.div_current_max_history[-1])
        if solution.diagnostics.div_current_max_history.size
        else 0.0,
        "charge_balance_residual": float(solution.diagnostics.charge_balance_residual_history[-1])
        if solution.diagnostics.charge_balance_residual_history.size
        else 0.0,
        "gauge_residual": float(solution.diagnostics.gauge_residual_history[-1])
        if solution.diagnostics.gauge_residual_history.size
        else 0.0,
        "interface_current_residual": float(solution.diagnostics.interface_current_residual_history[-1])
        if solution.diagnostics.interface_current_residual_history.size
        else 0.0,
        "raw_update_max": float(solution.diagnostics.raw_update_max_history[-1])
        if solution.diagnostics.raw_update_max_history.size
        else 0.0,
        "limiter_scale": float(solution.diagnostics.limiter_scale_history[-1])
        if solution.diagnostics.limiter_scale_history.size
        else 1.0,
        "limited_fraction": float(solution.diagnostics.limited_fraction_history[-1])
        if solution.diagnostics.limited_fraction_history.size
        else 0.0,
    }
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


def estimate_observed_order(
    coarse_error: float,
    fine_error: float,
    coarse_spacing: float,
    fine_spacing: float,
) -> float | None:
    if coarse_error <= 0.0 or fine_error <= 0.0:
        return None
    if coarse_spacing <= 0.0 or fine_spacing <= 0.0:
        return None
    spacing_ratio = coarse_spacing / fine_spacing
    if spacing_ratio <= 1.0:
        return None
    return log(coarse_error / fine_error) / log(spacing_ratio)


def closed_channel_validation(
    solution: Solution,
    case_kind: str,
    ha: int,
    reference_root: str | Path | None = None,
) -> ClosedChannelValidation:
    reference: ClosedChannelAnalyticalReference = load_closed_channel_analytical(case_kind, ha, reference_root)
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


def processed_slice_validation(
    solution: Solution,
    case_kind: str,
    ha: int,
    x_slice: str = "1m",
    reference_root: str | Path | None = None,
) -> ProcessedSliceValidation:
    reference: ProcessedSliceReference = load_processed_slice(
        case_kind,
        ha,
        x_slice=x_slice,
        reference_root=reference_root,
    )
    y_profile = extract_midplane_profile(solution, axis="y", fluid_only=True)
    z_profile = extract_midplane_profile(solution, axis="z", fluid_only=True)
    reference_y = extract_processed_midplane_profile(reference, axis="y")
    reference_z = extract_processed_midplane_profile(reference, axis="z")
    y_comparison = compare_normalized_profiles(
        y_profile["y"],
        y_profile["u"],
        reference_y["y"],
        reference_y["u"],
    )
    z_comparison = compare_normalized_profiles(
        z_profile["z"],
        z_profile["u"],
        reference_z["z"],
        reference_z["u"],
    )
    return ProcessedSliceValidation(
        case_kind=case_kind,
        ha=ha,
        x_slice=x_slice,
        y_profile=y_comparison,
        z_profile=z_comparison,
        reference_path=reference.path,
    )


def reference_profile_validation(
    solution: Solution,
    reference_run_dir: str | Path,
) -> ReferenceProfileValidation:
    sampled_profiles = latest_reference_sampled_profiles(reference_run_dir)
    if sampled_profiles is None:
        raise FileNotFoundError(f"No paired sampled reference profiles found under {reference_run_dir}")
    y_sample, z_sample = sampled_profiles
    y_profile = extract_midplane_profile(solution, axis="y", fluid_only=True)
    z_profile = extract_midplane_profile(solution, axis="z", fluid_only=True)
    y_comparison = compare_normalized_profiles(
        y_profile["y"],
        y_profile["u"],
        normalize_sample_distance(y_sample.distance),
        y_sample.u_x,
    )
    z_comparison = compare_normalized_profiles(
        z_profile["z"],
        z_profile["u"],
        normalize_sample_distance(z_sample.distance),
        z_sample.u_x,
    )
    return ReferenceProfileValidation(
        sample_time=infer_sample_time_from_path(y_sample.path),
        y_profile=y_comparison,
        z_profile=z_comparison,
        y_path=y_sample.path,
        z_path=z_sample.path,
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


def compare_with_reference_outputs(case_spec: CaseSpec, reference_run_dir: str | Path) -> ValidationReport:
    run_dir = Path(reference_run_dir)
    inspection = inspect_reference_case(run_dir)
    expected_region_count = float(len(case_spec.regions))
    expected_solid_count = float(sum(1 for region in case_spec.regions if region.kind == "solid"))
    minmax_files = tuple(sorted(str(path.relative_to(run_dir)) for path in run_dir.glob("postProcessing/**/fieldMinMax.dat")))
    sampled_profiles = latest_reference_sampled_profiles(run_dir)
    y_sample_path = sampled_profiles[0].path if sampled_profiles is not None else ""
    z_sample_path = sampled_profiles[1].path if sampled_profiles is not None else ""
    metrics = {
        "run_dir_exists": float(run_dir.exists()),
        "has_system": float((run_dir / "system").exists()),
        "has_constant": float((run_dir / "constant").exists()),
        "has_zero_dir": float((run_dir / "0").exists()),
        "control_dict_count": float(len(inspection.control_dicts)),
        "region_properties_count": float(len(inspection.region_properties)),
        "latest_time_count": float(len(inspection.latest_time_dirs)),
        "region_zero_dir_count": float(len(inspection.region_zero_dirs)),
        "zero_field_file_count": float(len(inspection.zero_field_files)),
        "processor_layout_count": float(len(inspection.processor_layout_dirs)),
        "parallel_time_count": float(len(inspection.parallel_time_dirs)),
        "has_potE_zero_field": float(any(path.endswith("/potE") for path in inspection.zero_field_files)),
        "has_velocity_zero_field": float(any(path.endswith("/U") for path in inspection.zero_field_files)),
        "expected_region_count": expected_region_count,
        "expected_solid_region_count": expected_solid_count,
        "field_minmax_file_count": float(len(minmax_files)),
        "sampled_profile_pair_available": float(sampled_profiles is not None),
    }
    latest_u_record = latest_field_minmax_record(run_dir, field="mag(U)")
    lmx_solution: Solution | None = None
    if latest_u_record is not None:
        lmx_solution = solve_transient(case_spec)
        lmx_u_max = float(jnp.max(jnp.abs(lmx_solution.state.u)))
        metrics["reference_latest_time"] = latest_u_record.time
        metrics["reference_u_max_latest"] = latest_u_record.max_value
        metrics["lmx_u_max"] = lmx_u_max
        metrics["u_max_abs_diff"] = abs(lmx_u_max - latest_u_record.max_value)
    if sampled_profiles is not None:
        if lmx_solution is None:
            lmx_solution = solve_transient(case_spec)
        sample_validation = reference_profile_validation(lmx_solution, run_dir)
        metrics["reference_sample_time"] = sample_validation.sample_time
        metrics["reference_sample_y_l2_error"] = sample_validation.y_profile.l2_error
        metrics["reference_sample_y_linf_error"] = sample_validation.y_profile.linf_error
        metrics["reference_sample_z_l2_error"] = sample_validation.z_profile.l2_error
        metrics["reference_sample_z_linf_error"] = sample_validation.z_profile.linf_error
    artifacts = {
        "reference_run_dir": str(run_dir),
        "control_dicts": json.dumps(inspection.control_dicts),
        "region_properties": json.dumps(inspection.region_properties),
        "block_mesh_dicts": json.dumps(inspection.block_mesh_dicts),
        "boundary_field_dirs": json.dumps(inspection.boundary_field_dirs),
        "latest_time_dirs": json.dumps(inspection.latest_time_dirs),
        "region_zero_dirs": json.dumps(inspection.region_zero_dirs),
        "zero_field_files": json.dumps(inspection.zero_field_files),
        "processor_layout_dirs": json.dumps(inspection.processor_layout_dirs),
        "parallel_time_dirs": json.dumps(inspection.parallel_time_dirs),
        "field_minmax_files": json.dumps(minmax_files),
        "sampled_profile_y_path": y_sample_path,
        "sampled_profile_z_path": z_sample_path,
    }
    return ValidationReport(case_name=case_spec.name, metrics=metrics, artifacts=artifacts)


def write_validation_report(report: ValidationReport, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "case_name": report.case_name,
        "metrics": report.metrics,
        "artifacts": report.artifacts,
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def write_analytic_comparison(comparison: AnalyticComparison, path: str | Path, axis_name: str = "coordinate") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        axis_name: jnp.asarray(comparison.coordinate).tolist(),
        "simulated": jnp.asarray(comparison.simulated).tolist(),
        "reference": jnp.asarray(comparison.reference).tolist(),
        "l2_error": comparison.l2_error,
        "linf_error": comparison.linf_error,
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def write_closed_channel_validation(report: ClosedChannelValidation, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "case_kind": report.case_kind,
        "ha": report.ha,
        "reference_pressure_drop": report.reference_pressure_drop,
        "reference_path": report.reference_path,
        "y_profile": {
            "coordinate": jnp.asarray(report.y_profile.coordinate).tolist(),
            "simulated": jnp.asarray(report.y_profile.simulated).tolist(),
            "reference": jnp.asarray(report.y_profile.reference).tolist(),
            "l2_error": report.y_profile.l2_error,
            "linf_error": report.y_profile.linf_error,
        },
        "z_profile": {
            "coordinate": jnp.asarray(report.z_profile.coordinate).tolist(),
            "simulated": jnp.asarray(report.z_profile.simulated).tolist(),
            "reference": jnp.asarray(report.z_profile.reference).tolist(),
            "l2_error": report.z_profile.l2_error,
            "linf_error": report.z_profile.linf_error,
        },
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def write_processed_slice_validation(report: ProcessedSliceValidation, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "case_kind": report.case_kind,
        "ha": report.ha,
        "x_slice": report.x_slice,
        "reference_path": report.reference_path,
        "y_profile": {
            "coordinate": jnp.asarray(report.y_profile.coordinate).tolist(),
            "simulated": jnp.asarray(report.y_profile.simulated).tolist(),
            "reference": jnp.asarray(report.y_profile.reference).tolist(),
            "l2_error": report.y_profile.l2_error,
            "linf_error": report.y_profile.linf_error,
        },
        "z_profile": {
            "coordinate": jnp.asarray(report.z_profile.coordinate).tolist(),
            "simulated": jnp.asarray(report.z_profile.simulated).tolist(),
            "reference": jnp.asarray(report.z_profile.reference).tolist(),
            "l2_error": report.z_profile.l2_error,
            "linf_error": report.z_profile.linf_error,
        },
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def write_acceptance_report(report: AcceptanceReport, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "case_name": report.case_name,
        "l2_error": report.l2_error,
        "linf_error": report.linf_error,
        "l2_threshold": report.l2_threshold,
        "linf_threshold": report.linf_threshold,
        "passed_l2": report.passed_l2,
        "passed_linf": report.passed_linf,
        "passed": report.passed,
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def write_metrics_json(metrics: dict[str, float | str], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2))
    return path


def read_field_minmax(path: str | Path) -> tuple[FieldMinMaxRecord, ...]:
    records: list[FieldMinMaxRecord] = []
    for raw_line in Path(path).read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(
            r"^(?P<time>\S+)\s+"
            r"(?P<field>\S+)\s+"
            r"(?P<min>\S+)\s+"
            r"(?P<min_loc>\([^)]+\))\s+\S+\s+"
            r"(?P<max>\S+)\s+"
            r"(?P<max_loc>\([^)]+\))\s+\S+\s*$",
            line,
        )
        if match is None:
            continue
        min_location = parse_location_tuple(match.group("min_loc"))
        max_location = parse_location_tuple(match.group("max_loc"))
        records.append(
            FieldMinMaxRecord(
                time=float(match.group("time")),
                field=match.group("field"),
                min_value=float(match.group("min")),
                max_value=float(match.group("max")),
                min_location=min_location,
                max_location=max_location,
            )
        )
    return tuple(records)


def latest_field_minmax_record(run_dir: str | Path, field: str = "mag(U)") -> FieldMinMaxRecord | None:
    root = Path(run_dir)
    latest: FieldMinMaxRecord | None = None
    for path in root.glob("postProcessing/**/fieldMinMax.dat"):
        for record in read_field_minmax(path):
            if record.field != field:
                continue
            if latest is None or record.time > latest.time:
                latest = record
    return latest


def parse_location_tuple(text: str) -> tuple[float, float, float] | None:
    match = re.match(r"^\(\s*(\S+)\s+(\S+)\s+(\S+)\s*\)$", text.strip())
    if match is None:
        return None
    return (float(match.group(1)), float(match.group(2)), float(match.group(3)))


def infer_mesh_bounds(run_dir: str | Path, region: str = "liquid") -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None:
    root = Path(run_dir)
    candidates = [
        root / "constant" / region / "polyMesh" / "points",
        root / "constant" / "polyMesh" / "points",
    ]
    points_path = next((path for path in candidates if path.exists()), None)
    if points_path is None:
        return None

    mins = [float("inf"), float("inf"), float("inf")]
    maxs = [float("-inf"), float("-inf"), float("-inf")]
    found = False
    pattern = re.compile(r"^\(\s*(\S+)\s+(\S+)\s+(\S+)\s*\)$")
    with points_path.open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            match = pattern.match(line)
            if match is None:
                continue
            coords = [float(match.group(1)), float(match.group(2)), float(match.group(3))]
            for idx, value in enumerate(coords):
                mins[idx] = min(mins[idx], value)
                maxs[idx] = max(maxs[idx], value)
            found = True
    if not found:
        return None
    return ((mins[0], maxs[0]), (mins[1], maxs[1]), (mins[2], maxs[2]))


def infer_mesh_axis_coordinates(run_dir: str | Path, region: str = "liquid", axis: str = "x") -> tuple[float, ...] | None:
    axis_index = {"x": 0, "y": 1, "z": 2}.get(axis.lower())
    if axis_index is None:
        raise ValueError(f"Unsupported axis {axis}")
    root = Path(run_dir)
    candidates = [
        root / "constant" / region / "polyMesh" / "points",
        root / "constant" / "polyMesh" / "points",
    ]
    points_path = next((path for path in candidates if path.exists()), None)
    if points_path is None:
        return None

    values: set[float] = set()
    pattern = re.compile(r"^\(\s*(\S+)\s+(\S+)\s+(\S+)\s*\)$")
    with points_path.open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            match = pattern.match(line)
            if match is None:
                continue
            values.add(float(match.group(axis_index + 1)))
    if not values:
        return None
    return tuple(sorted(values))


def interior_sample_coordinate(coordinates: tuple[float, ...], preferred: float, tolerance: float = 1e-12) -> float:
    if len(coordinates) == 1:
        return coordinates[0]
    lower = coordinates[0]
    upper = coordinates[-1]
    if preferred <= lower + tolerance:
        if len(coordinates) >= 3:
            return coordinates[1]
        return 0.5 * (lower + upper)
    if preferred >= upper - tolerance:
        if len(coordinates) >= 3:
            return coordinates[-2]
        return 0.5 * (lower + upper)
    return preferred


def infer_region_conductivity(run_dir: str | Path, region: str) -> float | None:
    root = Path(run_dir)
    candidates = [
        root / "constant" / region / "thermophysicalProperties",
        root / "constant" / region / "thermophysicalProperties.liquidMetal",
    ]
    for path in candidates:
        if not path.exists():
            continue
        text = path.read_text()
        match = re.search(r"\belcond\b(?:\s+\[[^\]]+\])?\s*([0-9eE+.\-]+)\s*;", text)
        if match is not None:
            return float(match.group(1))
    return None


def has_conducting_wall_region(run_dir: str | Path) -> bool:
    liquid_sigma = infer_region_conductivity(run_dir, "liquid")
    wall_sigma = infer_region_conductivity(run_dir, "solidWalls")
    if liquid_sigma is None or wall_sigma is None:
        return False
    return wall_sigma > liquid_sigma * 1e-2


def infer_sampling_geometry(run_dir: str | Path, field: str = "mag(U)") -> SamplingGeometry:
    latest = latest_field_minmax_record(run_dir, field=field)
    mesh_bounds = infer_mesh_bounds(run_dir, region="liquid")
    x_coordinates = infer_mesh_axis_coordinates(run_dir, region="liquid", axis="x")
    if mesh_bounds is not None and not has_conducting_wall_region(run_dir):
        (x_min, x_max), (y_min, y_max), (z_min, z_max) = mesh_bounds
        return SamplingGeometry(
            x_position=0.5 * (x_min + x_max),
            y_min=y_min,
            y_max=y_max,
            z_min=z_min,
            z_max=z_max,
        )
    if latest is None or latest.min_location is None or latest.max_location is None:
        raise ValueError(f"Unable to infer sampling geometry from {run_dir}")
    x_position = latest.max_location[0]
    if x_coordinates is not None:
        x_position = interior_sample_coordinate(x_coordinates, x_position)
    y_extent = max(abs(latest.min_location[1]), abs(latest.max_location[1]))
    z_extent = max(abs(latest.min_location[2]), abs(latest.max_location[2]))
    return SamplingGeometry(
        x_position=x_position,
        y_min=-y_extent,
        y_max=y_extent,
        z_min=-z_extent,
        z_max=z_extent,
    )


def read_reference_xy_sample(path: str | Path) -> ReferenceLineSample:
    distance = []
    pot_e = []
    u_x = []
    u_y = []
    u_z = []
    for raw_line in Path(path).read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        distance.append(float(parts[0]))
        pot_e.append(float(parts[1]))
        u_x.append(float(parts[2]))
        u_y.append(float(parts[3]))
        u_z.append(float(parts[4]))
    return ReferenceLineSample(
        path=str(Path(path)),
        distance=jnp.asarray(distance, dtype=float),
        pot_e=jnp.asarray(pot_e, dtype=float),
        u_x=jnp.asarray(u_x, dtype=float),
        u_y=jnp.asarray(u_y, dtype=float),
        u_z=jnp.asarray(u_z, dtype=float),
    )


def read_reference_csv_sample(path: str | Path) -> ReferenceLineSample:
    distance: list[float] = []
    pot_e: list[float] = []
    u_x: list[float] = []
    u_y: list[float] = []
    u_z: list[float] = []
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row is None:
                continue
            keys = tuple(row.keys())
            if not keys:
                continue
            coordinate_key = keys[0]
            try:
                distance.append(float(row[coordinate_key]))
                pot_e.append(float(row.get("p", 0.0)))
                u_x.append(float(row.get("U_0", 0.0)))
                u_y.append(float(row.get("U_1", 0.0)))
                u_z.append(float(row.get("U_2", 0.0)))
            except (TypeError, ValueError):
                continue
    return ReferenceLineSample(
        path=str(Path(path)),
        distance=jnp.asarray(distance, dtype=float),
        pot_e=jnp.asarray(pot_e, dtype=float),
        u_x=jnp.asarray(u_x, dtype=float),
        u_y=jnp.asarray(u_y, dtype=float),
        u_z=jnp.asarray(u_z, dtype=float),
    )


def infer_sample_time_from_path(path: str | Path) -> float:
    sample_path = Path(path)
    for parent in sample_path.parents:
        try:
            return float(parent.name)
        except ValueError:
            continue
    raise ValueError(f"Unable to infer sample time from path {sample_path}")


def normalize_sample_distance(distance: jnp.ndarray) -> jnp.ndarray:
    max_distance = jnp.max(distance)
    max_distance = jnp.where(max_distance > 0.0, max_distance, 1.0)
    return 2.0 * distance / max_distance - 1.0


def latest_reference_sampled_profiles(run_dir: str | Path) -> tuple[ReferenceLineSample, ReferenceLineSample] | None:
    root = Path(run_dir)
    candidates = sorted(root.glob("postProcessing/*/liquid/*/centerlineY_potE_U.xy"))
    latest_y_path: Path | None = None
    latest_key: tuple[float, int] | None = None
    for path in candidates:
        try:
            sample_time = infer_sample_time_from_path(path)
        except ValueError:
            continue
        ordering_key = (sample_time, int(path.stat().st_mtime_ns))
        if latest_key is None or ordering_key > latest_key:
            latest_key = ordering_key
            latest_y_path = path
    if latest_y_path is None:
        csv_candidates = sorted(root.glob("postProcessing/outputLines/liquid/*/lineTransverse_p_U.csv"))
        latest_csv_path: Path | None = None
        latest_csv_key: tuple[float, int] | None = None
        for path in csv_candidates:
            try:
                sample_time = infer_sample_time_from_path(path)
            except ValueError:
                continue
            ordering_key = (sample_time, int(path.stat().st_mtime_ns))
            if latest_csv_key is None or ordering_key > latest_csv_key:
                latest_csv_key = ordering_key
                latest_csv_path = path
        if latest_csv_path is None:
            return None
        z_path = latest_csv_path.with_name("lineVertical_p_U.csv")
        if not z_path.exists():
            return None
        return read_reference_csv_sample(latest_csv_path), read_reference_csv_sample(z_path)
    z_path = latest_y_path.with_name("centerlineZ_potE_U.xy")
    if not z_path.exists():
        return None
    return read_reference_xy_sample(latest_y_path), read_reference_xy_sample(z_path)


def inspect_reference_case(case_dir: str | Path) -> ReferenceCaseInspection:
    root = Path(case_dir)
    if not root.exists():
        return ReferenceCaseInspection(
            case_dir=str(root),
            control_dicts=(),
            fv_schemes=(),
            fv_solutions=(),
            region_properties=(),
            block_mesh_dicts=(),
            boundary_field_dirs=(),
            latest_time_dirs=(),
            region_zero_dirs=(),
            zero_field_files=(),
            processor_layout_dirs=(),
            parallel_time_dirs=(),
        )

    def _relative_matches(pattern: str) -> tuple[str, ...]:
        return tuple(sorted(str(path.relative_to(root)) for path in root.glob(pattern)))

    def _numeric_time_dirs() -> tuple[str, ...]:
        matches: list[str] = []
        for path in root.iterdir():
            if not path.is_dir():
                continue
            if path.name in {"0", "constant", "system", "processor0"}:
                continue
            try:
                float(path.name)
            except ValueError:
                continue
            matches.append(str(path.relative_to(root)))
        return tuple(sorted(matches, key=float))

    def _region_zero_dirs() -> tuple[str, ...]:
        zero_root = root / "0"
        if not zero_root.is_dir():
            return ()
        return tuple(sorted(str(path.relative_to(root)) for path in zero_root.iterdir() if path.is_dir()))

    def _zero_field_files() -> tuple[str, ...]:
        zero_root = root / "0"
        if not zero_root.is_dir():
            return ()
        matches: list[str] = []
        for region_dir in zero_root.iterdir():
            if not region_dir.is_dir():
                continue
            for field_path in region_dir.iterdir():
                if field_path.is_file():
                    matches.append(str(field_path.relative_to(root)))
        return tuple(sorted(matches))

    def _processor_layout_dirs() -> tuple[str, ...]:
        return tuple(sorted(str(path.relative_to(root)) for path in root.iterdir() if path.is_dir() and path.name.startswith("processors")))

    def _parallel_time_dirs() -> tuple[str, ...]:
        matches: list[str] = []
        for processor_root in root.iterdir():
            if not processor_root.is_dir() or not processor_root.name.startswith("processors"):
                continue
            for path in processor_root.iterdir():
                if not path.is_dir():
                    continue
                try:
                    float(path.name)
                except ValueError:
                    continue
                matches.append(str(path.relative_to(root)))
        return tuple(sorted(matches, key=lambda value: float(Path(value).name)))

    return ReferenceCaseInspection(
        case_dir=str(root),
        control_dicts=_relative_matches("**/system/controlDict"),
        fv_schemes=_relative_matches("**/system/fvSchemes"),
        fv_solutions=_relative_matches("**/system/fvSolution"),
        region_properties=_relative_matches("**/constant/regionProperties"),
        block_mesh_dicts=_relative_matches("**/system/blockMeshDict"),
        boundary_field_dirs=_relative_matches("0")
        + _relative_matches("**/0"),
        latest_time_dirs=_numeric_time_dirs(),
        region_zero_dirs=_region_zero_dirs(),
        zero_field_files=_zero_field_files(),
        processor_layout_dirs=_processor_layout_dirs(),
        parallel_time_dirs=_parallel_time_dirs(),
    )

"""Materials, magnetic fields, nondimensional groups, and wall physics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import jax.numpy as jnp
import numpy as np

from .mesh import StructuredMesh, center_coordinates, sample_tabulated_cross_section_field
from .specs import BoundaryCondition, CaseSpec, MagneticFieldSpec, RegionSpec

MU0 = 4.0e-7 * math.pi


def _require_positive(name: str, value: float) -> None:
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")


def dynamic_to_kinematic_viscosity(dynamic_viscosity: float, density: float) -> float:
    """Return kinematic viscosity ``nu = mu / rho`` in ``m^2/s``."""

    _require_positive("density", density)
    if dynamic_viscosity < 0.0:
        raise ValueError("dynamic_viscosity must be non-negative")
    return float(dynamic_viscosity) / float(density)


def kinematic_to_dynamic_viscosity(kinematic_viscosity: float, density: float) -> float:
    """Return dynamic viscosity ``mu = rho nu`` in ``Pa s``."""

    _require_positive("density", density)
    if kinematic_viscosity < 0.0:
        raise ValueError("kinematic_viscosity must be non-negative")
    return float(kinematic_viscosity) * float(density)


def hartmann_number(
    *,
    magnetic_field: float,
    length_scale: float,
    conductivity: float,
    density: float,
    kinematic_viscosity: float,
) -> float:
    """Return ``Ha = B a sqrt(sigma / (rho nu))``."""

    for name, value in (
        ("length_scale", length_scale),
        ("conductivity", conductivity),
        ("density", density),
        ("kinematic_viscosity", kinematic_viscosity),
    ):
        _require_positive(name, value)
    return (
        abs(float(magnetic_field))
        * float(length_scale)
        * math.sqrt(float(conductivity) / (float(density) * float(kinematic_viscosity)))
    )


def magnetic_field_from_hartmann(
    *,
    hartmann: float,
    length_scale: float,
    conductivity: float,
    density: float,
    kinematic_viscosity: float,
) -> float:
    """Return ``B`` from a target Hartmann number using kinematic viscosity."""

    for name, value in (
        ("length_scale", length_scale),
        ("conductivity", conductivity),
        ("density", density),
        ("kinematic_viscosity", kinematic_viscosity),
    ):
        _require_positive(name, value)
    return float(hartmann) / (
        float(length_scale) * math.sqrt(float(conductivity) / (float(density) * float(kinematic_viscosity)))
    )


def reynolds_number(*, velocity: float, length_scale: float, kinematic_viscosity: float) -> float:
    """Return ``Re = U a / nu``."""

    _require_positive("length_scale", length_scale)
    _require_positive("kinematic_viscosity", kinematic_viscosity)
    return abs(float(velocity)) * float(length_scale) / float(kinematic_viscosity)


def interaction_parameter(
    *,
    magnetic_field: float,
    length_scale: float,
    conductivity: float,
    density: float,
    velocity: float,
) -> float:
    """Return ``N = sigma B^2 a / (rho U)``."""

    for name, value in (
        ("length_scale", length_scale),
        ("conductivity", conductivity),
        ("density", density),
        ("velocity", abs(velocity)),
    ):
        _require_positive(name, value)
    return (
        float(conductivity)
        * float(magnetic_field) ** 2
        * float(length_scale)
        / (float(density) * abs(float(velocity)))
    )


def magnetic_reynolds_number(
    *,
    velocity: float,
    length_scale: float,
    conductivity: float,
    magnetic_permeability: float = MU0,
) -> float:
    """Return ``Rm = mu0 sigma U a``."""

    for name, value in (
        ("length_scale", length_scale),
        ("conductivity", conductivity),
        ("magnetic_permeability", magnetic_permeability),
    ):
        _require_positive(name, value)
    return float(magnetic_permeability) * float(conductivity) * abs(float(velocity)) * float(length_scale)


def wall_conductance_ratio(
    *,
    wall_conductivity: float,
    wall_thickness: float,
    fluid_conductivity: float,
    length_scale: float,
) -> float:
    """Return thin-wall tangential conductance ratio ``c``."""

    for name, value in (
        ("wall_thickness", wall_thickness),
        ("fluid_conductivity", fluid_conductivity),
        ("length_scale", length_scale),
    ):
        _require_positive(name, value)
    if wall_conductivity < 0.0:
        raise ValueError("wall_conductivity must be non-negative")
    return (
        float(wall_conductivity) * float(wall_thickness) / (float(fluid_conductivity) * float(length_scale))
    )


def normal_leakage_ratio(
    *,
    coating_conductivity: float,
    coating_thickness: float,
    fluid_conductivity: float,
    length_scale: float,
) -> float:
    """Return normal shunt ratio ``g_perp``."""

    for name, value in (
        ("coating_thickness", coating_thickness),
        ("fluid_conductivity", fluid_conductivity),
        ("length_scale", length_scale),
    ):
        _require_positive(name, value)
    if coating_conductivity < 0.0:
        raise ValueError("coating_conductivity must be non-negative")
    return (
        float(coating_conductivity)
        * float(length_scale)
        / (float(fluid_conductivity) * float(coating_thickness))
    )


@dataclass(frozen=True)
class WallLayer:
    """One solid layer in a fluid-facing electrical wall stack."""

    name: str
    conductivity: float
    thickness: float
    cells: int = 1


def _validate_layers(layers: Sequence[WallLayer]) -> None:
    if not layers:
        raise ValueError("at least one wall layer is required")
    for layer in layers:
        if layer.thickness <= 0.0:
            raise ValueError(f"wall layer {layer.name!r} has non-positive thickness")
        if layer.conductivity < 0.0:
            raise ValueError(f"wall layer {layer.name!r} has negative conductivity")
        if layer.cells < 0:
            raise ValueError(f"wall layer {layer.name!r} has negative cells")


def tangential_stack_conductance_ratio(
    layers: Sequence[WallLayer], *, fluid_conductivity: float, length_scale: float
) -> float:
    """Return the thin-wall tangential ratio for layers in parallel."""

    _validate_layers(layers)
    _require_positive("fluid_conductivity", fluid_conductivity)
    _require_positive("length_scale", length_scale)
    surface_conductance = sum(float(layer.conductivity) * float(layer.thickness) for layer in layers)
    return surface_conductance / (float(fluid_conductivity) * float(length_scale))


def normal_stack_leakage_ratio(
    layers: Sequence[WallLayer], *, fluid_conductivity: float, length_scale: float
) -> float:
    """Return the normal leakage ratio for layers in series."""

    _validate_layers(layers)
    _require_positive("fluid_conductivity", fluid_conductivity)
    _require_positive("length_scale", length_scale)
    if any(layer.conductivity <= 0.0 for layer in layers):
        return 0.0
    resistance = sum(float(layer.thickness) / float(layer.conductivity) for layer in layers)
    return float(length_scale) / (float(fluid_conductivity) * resistance)


def effective_pinhole_conductance_ratio(
    *, intact_conductance_ratio: float, metal_conductance_ratio: float, pinhole_fraction: float
) -> float:
    """Return the area-weighted conductance ratio for a pinholed coating."""

    if not 0.0 <= pinhole_fraction <= 1.0:
        raise ValueError("pinhole_fraction must be between 0 and 1")
    if intact_conductance_ratio < 0.0 or metal_conductance_ratio < 0.0:
        raise ValueError("conductance ratios must be non-negative")
    return (1.0 - float(pinhole_fraction)) * float(intact_conductance_ratio) + float(
        pinhole_fraction
    ) * float(metal_conductance_ratio)


def equivalent_single_layer(layers: Sequence[WallLayer], *, name: str = "equivalent_wall") -> WallLayer:
    """Return one layer with the same tangential surface conductance."""

    _validate_layers(layers)
    total_thickness = sum(float(layer.thickness) for layer in layers)
    surface_conductance = sum(float(layer.conductivity) * float(layer.thickness) for layer in layers)
    return WallLayer(
        name=name,
        conductivity=surface_conductance / total_thickness,
        thickness=total_thickness,
        cells=sum(max(int(layer.cells), 0) for layer in layers),
    )


def nested_wall_layer_resolution_summary(
    layers: Sequence[WallLayer], *, minimum_cells_per_layer: int = 3
) -> dict[str, object]:
    """Return mesh-resolution metrics for a wall-layer stack."""

    _validate_layers(layers)
    if minimum_cells_per_layer < 1:
        raise ValueError("minimum_cells_per_layer must be positive")
    rows = [
        {
            "name": layer.name,
            "thickness": float(layer.thickness),
            "conductivity": float(layer.conductivity),
            "cells": int(layer.cells),
            "cell_width": float(layer.thickness) / max(int(layer.cells), 1),
            "cell_count_pass": int(layer.cells) >= minimum_cells_per_layer,
        }
        for layer in layers
    ]
    minimum = min(int(layer.cells) for layer in layers)
    return {
        "layer_count": len(layers),
        "total_thickness": sum(float(layer.thickness) for layer in layers),
        "total_cells": sum(int(layer.cells) for layer in layers),
        "minimum_cells_per_layer": minimum,
        "minimum_required_cells_per_layer": int(minimum_cells_per_layer),
        "resolution_pass": minimum >= minimum_cells_per_layer,
        "layers": rows,
    }


def wall_layer_from_conductance_ratio(
    *,
    name: str,
    conductance_ratio: float,
    thickness: float,
    fluid_conductivity: float,
    length_scale: float,
    cells: int = 1,
) -> WallLayer:
    """Construct a wall layer from a target thin-wall conductance ratio."""

    _require_positive("thickness", thickness)
    if conductance_ratio < 0.0:
        raise ValueError("conductance_ratio must be non-negative")
    layer = WallLayer(
        name=name,
        conductivity=(
            float(conductance_ratio) * float(fluid_conductivity) * float(length_scale) / float(thickness)
        ),
        thickness=thickness,
        cells=cells,
    )
    recovered = wall_conductance_ratio(
        wall_conductivity=layer.conductivity,
        wall_thickness=layer.thickness,
        fluid_conductivity=fluid_conductivity,
        length_scale=length_scale,
    )
    if not math.isclose(recovered, conductance_ratio, rel_tol=1e-12, abs_tol=1e-15):
        raise RuntimeError("failed to construct requested wall conductance ratio")
    return layer


@dataclass(frozen=True)
class MaterialFields:
    conductivity: jnp.ndarray
    density: jnp.ndarray
    viscosity: jnp.ndarray
    fluid_mask: jnp.ndarray


def magnetic_ramp_scale(spec: MagneticFieldSpec, time: float | jnp.ndarray | None = None) -> jnp.ndarray:
    if time is None or spec.ramp_duration <= 0.0:
        return jnp.asarray(1.0)
    time_value = jnp.asarray(time, dtype=float)
    # Startup ramps use the same clipped affine law as the benchmark archive:
    # scale = max(min((t - t_start)/(duration + 1e-6), 1), 0)
    return jnp.clip((time_value - spec.ramp_start) / (spec.ramp_duration + 1.0e-6), 0.0, 1.0)


def magnetic_field_components(
    spec: MagneticFieldSpec,
    mesh: StructuredMesh,
    time: float | jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    yc, zc = center_coordinates(mesh)
    if spec.kind == "constant":
        bx, by, bz = spec.value or (0.0, 0.0, 0.0)
        shape = yc.shape
        field = (jnp.full(shape, bx), jnp.full(shape, by), jnp.full(shape, bz))
    elif spec.kind == "analytic":
        if spec.fn is None:
            raise ValueError("Analytic magnetic field requires fn")
        field = spec.fn(yc, zc)
        field = field[..., 0], field[..., 1], field[..., 2]
    elif spec.kind == "tabulated":
        if spec.table_path is None:
            raise ValueError("Tabulated magnetic field requires table_path")
        sampled = sample_tabulated_cross_section_field(
            spec.table_path,
            y=np.asarray(yc, dtype=float),
            z=np.asarray(zc, dtype=float),
        )
        field = (
            jnp.asarray(sampled[..., 0], dtype=float),
            jnp.asarray(sampled[..., 1], dtype=float),
            jnp.asarray(sampled[..., 2], dtype=float),
        )
    else:
        raise ValueError(f"Unsupported magnetic-field kind {spec.kind!r}")
    scale = magnetic_ramp_scale(spec, time)
    return field[0] * scale, field[1] * scale, field[2] * scale


def region_lookup(regions: tuple[RegionSpec, ...]) -> dict[str, RegionSpec]:
    return {region.name: region for region in regions}


def _boundary_sides(boundary: BoundaryCondition) -> tuple[str, ...]:
    if boundary.side is None:
        return ()
    side = boundary.side.lower()
    if side == "left_right":
        return ("left", "right")
    if side == "top_bottom":
        return ("bottom", "top")
    return tuple(part.strip() for part in side.split(",") if part.strip())


def _layered_side_distance_masks(
    mesh: StructuredMesh,
    case: CaseSpec,
) -> dict[str, tuple[jnp.ndarray, jnp.ndarray]]:
    yc, zc = center_coordinates(mesh)
    half_width = 0.5 * case.geometry.width
    half_height = 0.5 * case.geometry.height
    inf = jnp.asarray(jnp.inf, dtype=yc.dtype)
    left_distance = jnp.where(yc < -half_width, -half_width - yc, inf)
    right_distance = jnp.where(yc > half_width, yc - half_width, inf)
    bottom_distance = jnp.where(zc < -half_height, -half_height - zc, inf)
    top_distance = jnp.where(zc > half_height, zc - half_height, inf)
    return {
        "left": (jnp.isfinite(left_distance), left_distance),
        "right": (jnp.isfinite(right_distance), right_distance),
        "bottom": (jnp.isfinite(bottom_distance), bottom_distance),
        "top": (jnp.isfinite(top_distance), top_distance),
    }


def build_material_fields(case: CaseSpec, mesh: StructuredMesh) -> MaterialFields:
    region_map = region_lookup(case.regions)
    fluid = next(region for region in case.regions if region.kind == "fluid")
    solid_candidates = [region for region in case.regions if region.kind == "solid"]
    solid = solid_candidates[0] if solid_candidates else fluid
    fluid_mask = mesh.fluid_mask if mesh.fluid_mask is not None else jnp.ones(mesh.yz_shape, dtype=bool)

    conductivity = jnp.full(mesh.yz_shape, fluid.conductivity, dtype=float)
    density = jnp.full(mesh.yz_shape, fluid.density or 1.0, dtype=float)
    viscosity = jnp.full(mesh.yz_shape, fluid.viscosity or 1.0, dtype=float)

    if solid_candidates:
        side_assignments: list[tuple[str, RegionSpec]] = []
        for boundary in case.boundary_conditions:
            if boundary.region is None:
                continue
            region = region_map.get(boundary.region)
            if region is None or region.kind != "solid":
                continue
            for side in _boundary_sides(boundary):
                side_assignments.append((side, region))

        if case.geometry.kind == "layered_duct" and side_assignments:
            distance_by_side = _layered_side_distance_masks(mesh, case)
            best_distance = jnp.full(mesh.yz_shape, jnp.inf, dtype=float)
            assigned = jnp.zeros(mesh.yz_shape, dtype=bool)
            for side, region in side_assignments:
                mask, distance = distance_by_side.get(
                    side,
                    (jnp.zeros(mesh.yz_shape, dtype=bool), jnp.full(mesh.yz_shape, jnp.inf, dtype=float)),
                )
                better = (~fluid_mask) & mask & (distance < best_distance)
                conductivity = jnp.where(better, region.conductivity, conductivity)
                density = jnp.where(better, region.density or fluid.density or 1.0, density)
                viscosity = jnp.where(better, region.viscosity or fluid.viscosity or 1.0, viscosity)
                best_distance = jnp.where(better, distance, best_distance)
                assigned = assigned | better

            fallback = (~fluid_mask) & (~assigned)
            conductivity = jnp.where(fallback, solid.conductivity, conductivity)
            density = jnp.where(fallback, solid.density or fluid.density or 1.0, density)
            viscosity = jnp.where(fallback, solid.viscosity or fluid.viscosity or 1.0, viscosity)

            # Exact insulating boundaries own wall intersections. Nearest-side
            # assignment is ambiguous at corners and otherwise leaves a thin
            # conducting bridge around an ideal insulating wall.
            for side, region in side_assignments:
                if region.conductivity != 0.0:
                    continue
                mask, _ = distance_by_side.get(
                    side,
                    (
                        jnp.zeros(mesh.yz_shape, dtype=bool),
                        jnp.full(mesh.yz_shape, jnp.inf, dtype=float),
                    ),
                )
                conductivity = jnp.where((~fluid_mask) & mask, 0.0, conductivity)
        else:
            conductivity = jnp.where(fluid_mask, conductivity, solid.conductivity)
            density = jnp.where(fluid_mask, density, solid.density or fluid.density or 1.0)
            viscosity = jnp.where(fluid_mask, viscosity, solid.viscosity or fluid.viscosity or 1.0)

    if mesh.sigma is not None:
        explicit_sigma = jnp.asarray(mesh.sigma, dtype=float)
        if explicit_sigma.shape != mesh.yz_shape:
            raise ValueError("mesh.sigma must have the same shape as the mesh cross-section")
        conductivity = explicit_sigma

    return MaterialFields(
        conductivity=conductivity,
        density=density,
        viscosity=viscosity,
        fluid_mask=fluid_mask,
    )

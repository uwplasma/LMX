from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from .field_models import sample_tabulated_cross_section_field
from .mesh import StructuredMesh
from .operators import center_coordinates
from .specs import BoundaryCondition, CaseSpec, MagneticFieldSpec, RegionSpec


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

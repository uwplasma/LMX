from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from .mesh import StructuredMesh
from .operators import center_coordinates
from .specs import CaseSpec, MagneticFieldSpec, RegionSpec


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
    # Mirror the recovered FreeMHD controlDict startup ramp law:
    # scale = max(min((t - BtStartTime)/(BtDuration + 1e-6), 1), 0)
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
    else:
        raise NotImplementedError("Tabulated magnetic fields are planned but not yet implemented.")
    scale = magnetic_ramp_scale(spec, time)
    return field[0] * scale, field[1] * scale, field[2] * scale


def region_lookup(regions: tuple[RegionSpec, ...]) -> dict[str, RegionSpec]:
    return {region.name: region for region in regions}


def build_material_fields(case: CaseSpec, mesh: StructuredMesh) -> MaterialFields:
    region_map = region_lookup(case.regions)
    fluid = next(region for region in case.regions if region.kind == "fluid")
    solid_candidates = [region for region in case.regions if region.kind == "solid"]
    solid = solid_candidates[0] if solid_candidates else fluid
    fluid_mask = mesh.fluid_mask if mesh.fluid_mask is not None else jnp.ones(mesh.yz_shape, dtype=bool)

    conductivity = jnp.where(fluid_mask, fluid.conductivity, solid.conductivity)
    density = jnp.where(fluid_mask, fluid.density or 1.0, solid.density or fluid.density or 1.0)
    viscosity = jnp.where(fluid_mask, fluid.viscosity or 1.0, solid.viscosity or fluid.viscosity or 1.0)

    return MaterialFields(
        conductivity=conductivity,
        density=density,
        viscosity=viscosity,
        fluid_mask=fluid_mask,
    )

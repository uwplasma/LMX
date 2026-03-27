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


def magnetic_field_components(spec: MagneticFieldSpec, mesh: StructuredMesh) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    yc, zc = center_coordinates(mesh)
    if spec.kind == "constant":
        bx, by, bz = spec.value or (0.0, 0.0, 0.0)
        shape = yc.shape
        return (jnp.full(shape, bx), jnp.full(shape, by), jnp.full(shape, bz))
    if spec.kind == "analytic":
        if spec.fn is None:
            raise ValueError("Analytic magnetic field requires fn")
        field = spec.fn(yc, zc)
        return field[..., 0], field[..., 1], field[..., 2]
    raise NotImplementedError("Tabulated magnetic fields are planned but not yet implemented.")


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

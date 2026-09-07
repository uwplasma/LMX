"""Fully developed duct design: throughput, pumping power and their derivatives.

At fixed field, materials and geometry the fully developed inductionless problem
is *linear* in the drive: doubling the force density doubles the velocity
everywhere. The volumetric flow rate is therefore ``Q = G f`` for a single
response ``G`` that one solve measures, and the drive that delivers a requested
throughput is ``f = Q_target / G`` exactly.

That matters for design. Searching for a drive with an optimizer would spend many
solves rediscovering a scalar that one solve determines, and would return an
answer good only to its own tolerance. Eliminating it leaves the optimizer to
work on the inputs that genuinely change the flow: wall conductance, aspect ratio
and field strength. The derivative is exact too, ``df/dQ = 1/G``, which the tests
check against automatic differentiation of the solve itself.

The pressure drop follows the drive. For a duct driven by a uniform pressure
gradient the wall-to-wall drop over a length ``L`` is ``f L``, so the hydraulic
pumping power is ``f L Q``. This is the isothermal hydraulic power of the
fully developed segment; it is not a blanket pumping-power budget, which would
also carry entry and exit losses, manifolds and thermal effects.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .cases import solve_fully_developed_fields
from .physics import build_material_fields
from .solvers import _build_mesh
from .specs import CaseSpec

__all__ = [
    "DuctResponse",
    "drive_for_flow_rate",
    "fluid_cell_areas",
    "hydraulic_power",
    "linear_flow_response",
    "pressure_drop",
    "volumetric_flow_rate",
]


def fluid_cell_areas(case: CaseSpec) -> jnp.ndarray:
    """Return the cross-section area of every cell, zero outside the fluid."""
    with jax.ensure_compile_time_eval():
        mesh = _build_mesh(case)
        materials = build_material_fields(case, mesh)
        areas = jnp.asarray(mesh.dy)[:, None] * jnp.asarray(mesh.dz)[None, :]
        return jnp.where(materials.fluid_mask, areas, 0.0)


def volumetric_flow_rate(case: CaseSpec, velocity: jnp.ndarray) -> jnp.ndarray:
    """Integrate an axial velocity over the fluid cross-section."""
    areas = fluid_cell_areas(case).astype(velocity.dtype)
    if velocity.shape != areas.shape:
        raise ValueError(f"velocity shape {velocity.shape} does not match the mesh {areas.shape}")
    return jnp.sum(areas * velocity)


@dataclass(frozen=True)
class DuctResponse:
    """The linear throughput response of one duct at one field strength."""

    flow_per_unit_drive: jnp.ndarray
    magnetic_field_scale: jnp.ndarray

    def drive_for(self, target_flow_rate: float | jnp.ndarray) -> jnp.ndarray:
        """Return the drive that delivers ``target_flow_rate`` exactly."""
        return jnp.asarray(target_flow_rate) / self.flow_per_unit_drive


def linear_flow_response(case: CaseSpec, *, magnetic_field_scale: float | jnp.ndarray = 1.0) -> DuctResponse:
    """Measure ``G = Q(f = 1)``, the flow rate a unit drive produces.

    One solve determines the whole drive-to-flow relation because the problem is
    linear in the drive.
    """
    velocity, *_ = solve_fully_developed_fields(case, forcing=1.0, magnetic_field_scale=magnetic_field_scale)
    response = volumetric_flow_rate(case, velocity)
    return DuctResponse(response, jnp.asarray(magnetic_field_scale))


def drive_for_flow_rate(
    case: CaseSpec,
    target_flow_rate: float | jnp.ndarray,
    *,
    magnetic_field_scale: float | jnp.ndarray = 1.0,
) -> jnp.ndarray:
    """Return the force density whose fully developed flow rate is ``target_flow_rate``."""
    return linear_flow_response(case, magnetic_field_scale=magnetic_field_scale).drive_for(target_flow_rate)


def pressure_drop(drive: float | jnp.ndarray, length: float) -> jnp.ndarray:
    """Return the pressure drop of a uniform pressure-gradient drive over ``length``."""
    if length <= 0.0:
        raise ValueError("length must be positive")
    return jnp.asarray(drive) * length


def hydraulic_power(drive: float | jnp.ndarray, flow_rate: float | jnp.ndarray, length: float) -> jnp.ndarray:
    """Return the isothermal hydraulic power of a fully developed segment.

    This is the pressure drop times the throughput. It excludes entry and exit
    losses, manifolds and every thermal effect, so it is a segment quantity and
    not a blanket pumping budget.
    """
    return pressure_drop(drive, length) * jnp.asarray(flow_rate)


def fixed_flow_hydraulic_power(
    case: CaseSpec,
    target_flow_rate: float,
    length: float,
    *,
    magnetic_field_scale: float | jnp.ndarray = 1.0,
) -> jnp.ndarray:
    """Return the hydraulic power needed to hold a throughput at a given field.

    The drive is eliminated analytically, so this is a function of the design
    inputs alone and is differentiable through the solve.
    """
    response = linear_flow_response(case, magnetic_field_scale=magnetic_field_scale)
    drive = response.drive_for(target_flow_rate)
    return hydraulic_power(drive, target_flow_rate, length)

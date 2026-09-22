"""Fully developed duct design: throughput, pumping power and their derivatives.

At fixed field, materials and geometry, flow is linear in force density:
``Q = G f``. One unit-drive solve gives ``G``, eliminating the drive from
fixed-flow optimization as ``f = Q_target / G`` with ``df/dQ = 1/G``.
For a uniform pressure-gradient drive over length ``L``, pressure drop is
``f L`` and hydraulic power is ``f L Q``. These are isothermal segment
quantities, excluding entry/exit losses, manifolds and thermal effects.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .cases import solve_fully_developed_fields
from .core3d import ChannelProblem
from .solvers import _build_mesh
from .specs import CaseSpec

__all__ = [
    "DuctResponse",
    "channel_cross_section_weights",
    "drive_for_flow_rate",
    "fluid_cell_areas",
    "hydraulic_power",
    "linear_flow_response",
    "pressure_drop",
    "volumetric_flow_rate",
]


def fluid_cell_areas(case: CaseSpec) -> jnp.ndarray:
    """Return mesh-only integration weights, zero outside the fluid."""
    with jax.ensure_compile_time_eval():
        mesh = _build_mesh(case)
        areas = jnp.asarray(mesh.dy)[:, None] * jnp.asarray(mesh.dz)[None, :]
        return areas if mesh.fluid_mask is None else jnp.where(mesh.fluid_mask, areas, 0.0)


def channel_cross_section_weights(problem: ChannelProblem) -> jnp.ndarray:
    """Return the cross-section integration weights of a :class:`ChannelProblem`.

    Axis 0 is the flow axis of every channel this package builds (see
    :func:`lmx.core3d.duct_problem`), so a cell's weight is its transverse
    ``(y, z)`` area alone, independent of the axial spacing -- unlike
    :func:`fluid_cell_areas`, a channel carries no fluid mask, so every
    transverse cell counts.
    """
    _, dy, dz = problem.grid.widths
    return jnp.asarray(dy)[:, None] * jnp.asarray(dz)[None, :]


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

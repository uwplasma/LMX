"""Boundary conditions applied by padding ghost cells before differencing.

Every stencil in :mod:`lmx.ops` reads a padded array, so a boundary condition is
expressed once, here, as the ghost value that reproduces the wall condition. The
alternative -- special-casing edges inside each stencil -- is what makes wall
treatment hard to audit at high Hartmann number, where the wall layers carry the
physics.

A cell-centred value sits half a cell from the wall, so a prescribed wall value
``g`` needs ``ghost = 2*g - first`` for the face average to return ``g``, while a
prescribed wall-normal derivative ``q`` needs ``ghost = first -/+ q*dx``.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from .grid import Grid

__all__ = ["DIRICHLET", "NEUMANN", "PERIODIC", "BoundaryCondition", "pad"]

PERIODIC = "periodic"
DIRICHLET = "dirichlet"
NEUMANN = "neumann"
_KINDS = (PERIODIC, DIRICHLET, NEUMANN)


@dataclass(frozen=True)
class BoundaryCondition:
    """Condition on both ends of one axis.

    ``lower`` and ``upper`` are the prescribed wall value for :data:`DIRICHLET`
    and the prescribed outward-normal derivative for :data:`NEUMANN`; they are
    ignored for :data:`PERIODIC`.
    """

    kind: str
    lower: float = 0.0
    upper: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in _KINDS:
            raise ValueError(f"unknown boundary kind {self.kind!r}; expected one of {_KINDS}")
        if self.kind == PERIODIC and (self.lower, self.upper) != (0.0, 0.0):
            raise ValueError("periodic boundaries do not take prescribed values")

    @property
    def is_periodic(self) -> bool:
        return self.kind == PERIODIC


def pad(
    data: jnp.ndarray,
    axis: int,
    condition: BoundaryCondition,
    *,
    grid: Grid | None = None,
) -> jnp.ndarray:
    """Return ``data`` with one ghost entry added at each end of ``axis``.

    ``grid`` supplies the wall cell widths that a :data:`NEUMANN` condition needs
    and may be omitted for the other kinds or for a homogeneous gradient.
    """
    if data.ndim != 3:
        raise ValueError("padding expects a three-dimensional cell field")
    if axis not in (0, 1, 2):
        raise ValueError(f"axis index {axis} is out of range")
    if condition.is_periodic:
        return jnp.concatenate((_slice(data, axis, -1), data, _slice(data, axis, 0)), axis=axis)
    first, last = _slice(data, axis, 0), _slice(data, axis, -1)
    if condition.kind == DIRICHLET:
        lower = 2.0 * condition.lower - first
        upper = 2.0 * condition.upper - last
    else:
        widths = _wall_widths(axis, condition, grid)
        lower = first - condition.lower * widths[0]
        upper = last + condition.upper * widths[1]
    return jnp.concatenate((lower, data, upper), axis=axis)


def _wall_widths(axis: int, condition: BoundaryCondition, grid: Grid | None) -> tuple[float, float]:
    """Return the two wall cell widths a Neumann ghost needs."""
    if grid is None:
        if (condition.lower, condition.upper) != (0.0, 0.0):
            raise ValueError("a nonzero Neumann condition requires the grid for its wall spacing")
        return (0.0, 0.0)
    widths = np.asarray(grid.widths[axis])
    return (float(widths[0]), float(widths[-1]))


def _slice(data: jnp.ndarray, axis: int, index: int) -> jnp.ndarray:
    """Return one entry along ``axis``, keeping the axis so concatenation works."""
    return jnp.take(data, jnp.asarray([index % data.shape[axis]]), axis=axis)

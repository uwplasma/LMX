"""Boundary conditions applied by padding ghost cells before differencing.

Every stencil in :mod:`lmhdx.ops` reads a padded array, so a boundary condition is
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

import jax
import jax.numpy as jnp
import numpy as np

from .grid import Grid

__all__ = ["DIRICHLET", "NEUMANN", "PERIODIC", "BoundaryCondition", "pad"]

PERIODIC = "periodic"
DIRICHLET = "dirichlet"
NEUMANN = "neumann"
_KINDS = (PERIODIC, DIRICHLET, NEUMANN)


@dataclass(frozen=True, eq=False)
class BoundaryCondition:
    """Condition on both ends of one axis.

    ``lower`` and ``upper`` are the prescribed wall value for :data:`DIRICHLET`
    and the prescribed outward-normal derivative for :data:`NEUMANN`; they are
    ignored for :data:`PERIODIC`. Either may be an array over the two other axes,
    in their order: an inlet profile, for instance.

    ``upper_kind`` gives the upper end a different kind from ``kind``, which then
    names the lower end only. That is an inflow-outflow axis (plan D26): the
    velocity prescribed at the inlet and free at the outlet, the pressure the
    other way round. Its boundary faces are unknowns, so they carry the half cell
    they own in every inner product (:func:`lmhdx.ops.face_inner_product`).
    """

    kind: str
    lower: float | np.ndarray = 0.0
    upper: float | np.ndarray = 0.0
    upper_kind: str | None = None

    def __post_init__(self) -> None:
        for kind in self.kinds:
            if kind not in _KINDS:
                raise ValueError(f"unknown boundary kind {kind!r}; expected one of {_KINDS}")
        if PERIODIC in self.kinds and self.is_mixed:
            raise ValueError("a periodic axis cannot mix kinds")
        for name in ("lower", "upper"):
            value = getattr(self, name)
            if np.ndim(value):
                frozen = np.array(value, dtype=np.float64)
                if frozen.ndim != 2 or not np.all(np.isfinite(frozen)):
                    raise ValueError(f"array boundary data must be finite and two dimensional, got {name}")
                frozen.setflags(write=False)
                object.__setattr__(self, name, frozen)
            else:
                object.__setattr__(self, name, float(value))
        if self.kind == PERIODIC and not self.is_homogeneous:
            raise ValueError("periodic boundaries do not take prescribed values")

    @property
    def kinds(self) -> tuple[str, str]:
        """The kinds of the lower and the upper end."""
        return (self.kind, self.kind if self.upper_kind is None else self.upper_kind)

    @property
    def is_periodic(self) -> bool:
        return self.kind == PERIODIC

    @property
    def is_mixed(self) -> bool:
        """Whether the two ends differ in kind: an inflow-outflow axis."""
        return self.kinds[0] != self.kinds[1]

    @property
    def is_homogeneous(self) -> bool:
        """Whether both ends prescribe zero."""
        return not (np.any(self.lower) or np.any(self.upper))

    def homogeneous(self) -> "BoundaryCondition":
        """The same kinds with zero data, which is what a factorization represents."""
        return BoundaryCondition(self.kind, upper_kind=self.upper_kind)

    def _key(self) -> tuple:
        values = tuple(
            np.asarray(v).tobytes() + bytes(str(np.shape(v)), "ascii") for v in (self.lower, self.upper)
        )
        return (*self.kinds, *values)

    def __hash__(self) -> int:
        return hash(self._key())

    def __eq__(self, other: object) -> bool:
        return self._key() == other._key() if isinstance(other, BoundaryCondition) else NotImplemented


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
    kinds = condition.kinds
    lower_value, upper_value = (_end(value, axis) for value in (condition.lower, condition.upper))
    widths = _wall_widths(axis, condition, grid) if NEUMANN in kinds else (0.0, 0.0)
    lower = 2.0 * lower_value - first if kinds[0] == DIRICHLET else first - lower_value * widths[0]
    upper = 2.0 * upper_value - last if kinds[1] == DIRICHLET else last + upper_value * widths[1]
    return jnp.concatenate((lower, data, upper), axis=axis)


def _end(value: float | np.ndarray, axis: int) -> float | np.ndarray:
    """Boundary data shaped to broadcast against one end slab of ``axis``."""
    return np.expand_dims(value, axis) if np.ndim(value) else value


def _wall_widths(axis: int, condition: BoundaryCondition, grid: Grid | None) -> tuple[float, float]:
    """Return the two wall cell widths a Neumann ghost needs."""
    if grid is None:
        if not condition.is_homogeneous:
            raise ValueError("a nonzero Neumann condition requires the grid for its wall spacing")
        return (0.0, 0.0)
    widths = np.asarray(grid.widths[axis])
    return (float(widths[0]), float(widths[-1]))


def _slice(data: jnp.ndarray, axis: int, index: int) -> jnp.ndarray:
    """Return one entry along ``axis``, keeping the axis so concatenation works."""
    start = index % data.shape[axis]
    return jax.lax.slice_in_dim(data, start, start + 1, axis=axis)

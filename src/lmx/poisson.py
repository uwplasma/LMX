"""Direct Poisson solves by fast diagonalization on a tensor-product grid.

The finite-volume Laplacian of :mod:`lmx.ops` separates: on a tensor-product
grid it is the Kronecker sum of three one-dimensional operators. Each of those is
symmetric once the cell widths are folded in, so diagonalizing them on the host
turns a Poisson solve into three tensor contractions and one elementwise divide.
The cost is a handful of matrix multiplies rather than an iteration whose count
depends on the Hartmann number, the result is exact to round-off rather than to a
tolerance, and the solve is a linear map, so it differentiates for free.

The one-dimensional operators are read out of :mod:`lmx.ops` itself, by applying
the assembled Laplacian to unit vectors on a grid that is one cell wide across
the other two axes. The factorization therefore cannot drift away from the
stencil the rest of the code uses.

A pure Neumann or fully periodic problem determines its solution only up to a
constant. The compatible component is removed from the right-hand side and the
returned field has zero volume-weighted mean.

The factorization represents the homogeneous operator. Inhomogeneous boundary
data is affine, not linear, so it belongs in the right-hand side; passing a
condition that carries a value is refused rather than silently linearized.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from .bc import BoundaryCondition
from .grid import CENTER, Field, Grid, uniform_faces
from .ops import laplacian

__all__ = [
    "FastDiagonalHelmholtz",
    "FastDiagonalPoisson",
    "assemble_axis_laplacian",
    "assemble_staggered_axis_operator",
    "fast_diagonal_helmholtz",
    "fast_diagonal_poisson",
    "free_slice",
]

_SINGULAR_TOLERANCE = 1.0e-9


def _symmetry_tolerance(operator: np.ndarray) -> float:
    """Return the asymmetry a correct assembly may still show.

    The one-dimensional operators are read out through the production stencil,
    so they are assembled at whatever precision the session runs in. A fixed
    absolute bound would reject a perfectly good float32 assembly, so the
    tolerance follows the dtype.
    """
    scale = max(float(np.max(np.abs(operator))), 1.0)
    return max(1.0e-10, 200.0 * float(np.finfo(operator.dtype).eps)) * scale


def assemble_axis_laplacian(grid: Grid, axis: int, condition: BoundaryCondition) -> np.ndarray:
    """Return the dense one-dimensional Laplacian :mod:`lmx.ops` applies along ``axis``.

    The other two axes are collapsed to a single cell with a homogeneous
    Neumann condition, which contributes nothing, so the result is exactly the
    stencil the three-dimensional operator uses along ``axis``.
    """
    count = grid.shape[axis]
    faces = [uniform_faces(1, 0.0, 1.0)] * 3
    faces[axis] = np.asarray(grid.faces[axis])
    line = Grid(*faces)
    conditions = tuple(
        condition if position == axis else BoundaryCondition("neumann") for position in range(3)
    )
    columns = []
    for index in range(count):
        unit = np.zeros(line.shape)
        unit[(0,) * axis + (index,) + (0,) * (2 - axis)] = 1.0
        field = Field(jnp.asarray(unit), (CENTER,) * 3, line)
        columns.append(np.asarray(laplacian(field, conditions).data).reshape(count))
    return np.stack(columns, axis=1)


@dataclass(frozen=True)
class FastDiagonalPoisson:
    """A factorized Laplacian that solves ``laplacian(u) = rhs`` in three contractions."""

    grid: Grid
    conditions: tuple[BoundaryCondition, BoundaryCondition, BoundaryCondition]
    vectors: tuple[np.ndarray, np.ndarray, np.ndarray]
    values: tuple[np.ndarray, np.ndarray, np.ndarray]
    scales: tuple[np.ndarray, np.ndarray, np.ndarray]
    singular: bool

    def solve(self, rhs: Field) -> Field:
        """Return the field whose Laplacian is ``rhs``."""
        if rhs.grid != self.grid:
            raise ValueError("right-hand side must share the factorized grid")
        if rhs.offset != (CENTER, CENTER, CENTER):
            raise ValueError("right-hand side must be cell centred")
        dtype = rhs.dtype
        volumes = jnp.asarray(self.grid.cell_volumes(), dtype=dtype)
        data = rhs.data
        if self.singular:
            mean = jnp.sum(volumes * data) / jnp.sum(volumes)
            data = data - mean
        scaled = self._apply_scales(data, dtype, inverse=False)
        transformed = self._contract(scaled, dtype, transpose=True)
        denominator = self._eigenvalue_sum(dtype)
        solution = transformed / denominator
        if self.singular:
            solution = solution.at[0, 0, 0].set(0.0)
        restored = self._contract(solution, dtype, transpose=False)
        result = self._apply_scales(restored, dtype, inverse=True)
        if self.singular:
            mean = jnp.sum(volumes * result) / jnp.sum(volumes)
            result = result - mean
        return Field(result, (CENTER, CENTER, CENTER), self.grid)

    def residual_norm(self, solution: Field, rhs: Field) -> jnp.ndarray:
        """Return the maximum absolute residual of a candidate solution."""
        applied = laplacian(solution, self.conditions)
        difference = applied.data - rhs.data
        if self.singular:
            volumes = jnp.asarray(self.grid.cell_volumes(), dtype=rhs.dtype)
            difference = difference - jnp.sum(volumes * difference) / jnp.sum(volumes)
        return jnp.max(jnp.abs(difference))

    def _eigenvalue_sum(self, dtype) -> jnp.ndarray:
        total = self.values[0][:, None, None] + self.values[1][None, :, None] + self.values[2][None, None, :]
        if self.singular:
            total = total.copy()
            total[0, 0, 0] = 1.0
        return jnp.asarray(total, dtype=dtype)

    def _apply_scales(self, data: jnp.ndarray, dtype, *, inverse: bool) -> jnp.ndarray:
        for axis, scale in enumerate(self.scales):
            factor = 1.0 / scale if inverse else scale
            shape = [-1 if position == axis else 1 for position in range(3)]
            data = data * jnp.asarray(factor.reshape(shape), dtype=dtype)
        return data

    def _contract(self, data: jnp.ndarray, dtype, *, transpose: bool) -> jnp.ndarray:
        for axis, vectors in enumerate(self.vectors):
            matrix = jnp.asarray(vectors.T if transpose else vectors, dtype=dtype)
            data = jnp.moveaxis(jnp.tensordot(matrix, data, axes=([1], [axis])), 0, axis)
        return data


def fast_diagonal_poisson(
    grid: Grid, conditions: tuple[BoundaryCondition, BoundaryCondition, BoundaryCondition]
) -> FastDiagonalPoisson:
    """Factorize the separable Laplacian for ``grid`` under ``conditions``."""
    if len(conditions) != 3:
        raise ValueError("a factorization needs one boundary condition per axis")
    for axis, condition in enumerate(conditions):
        if (condition.lower, condition.upper) != (0.0, 0.0):
            raise ValueError(
                f"axis {axis} carries inhomogeneous boundary data; factorize the homogeneous "
                "operator and move the boundary contribution into the right-hand side"
            )
    vectors, values, scales = [], [], []
    for axis, condition in enumerate(conditions):
        operator = assemble_axis_laplacian(grid, axis, condition)
        widths = np.asarray(grid.widths[axis])
        root = np.sqrt(widths)
        symmetric = root[:, None] * operator / root[None, :]
        asymmetry = np.max(np.abs(symmetric - symmetric.T))
        if asymmetry > _symmetry_tolerance(symmetric):
            raise ValueError(
                f"axis {axis} operator is not symmetric under the cell widths "
                f"(asymmetry {asymmetry:.3e}); fast diagonalization does not apply"
            )
        symmetric = 0.5 * (symmetric + symmetric.T)
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
        vectors.append(eigenvectors)
        values.append(eigenvalues)
        scales.append(root)
    singular = _is_singular(values)
    if singular:
        # Order the constant mode first so a single entry carries the nullspace.
        vectors, values = _promote_null_mode(vectors, values)
    return FastDiagonalPoisson(
        grid, tuple(conditions), tuple(vectors), tuple(values), tuple(scales), singular
    )


def _is_singular(values: list[np.ndarray]) -> bool:
    """Return whether the Kronecker sum has a zero eigenvalue."""
    magnitude = max(float(np.max(np.abs(value))) for value in values)
    smallest = sum(float(np.min(np.abs(value))) for value in values)
    return smallest <= _SINGULAR_TOLERANCE * max(magnitude, 1.0)


def _promote_null_mode(
    vectors: list[np.ndarray], values: list[np.ndarray]
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Move each axis's smallest eigenvalue to index zero."""
    ordered_vectors, ordered_values = [], []
    for vector, value in zip(vectors, values, strict=True):
        order = np.argsort(np.abs(value), kind="stable")
        ordered_vectors.append(vector[:, order])
        ordered_values.append(value[order])
    return ordered_vectors, ordered_values


def free_slice(grid: Grid, axis: int, offset_value: float, condition: BoundaryCondition) -> slice:
    """Return the entries of a staggered axis that a solve may change.

    A cell-centred axis is free everywhere. A face-centred axis with a wall has
    its two boundary faces prescribed, and a periodic one carries a duplicate of
    its first face at the end. Solving on anything else would either invent a
    value for a prescribed face or treat one face as two unknowns.
    """
    if offset_value == CENTER:
        return slice(None)
    if condition.is_periodic:
        return slice(0, grid.shape[axis])
    return slice(1, grid.shape[axis])


def assemble_staggered_axis_operator(
    grid: Grid, axis: int, offset: tuple[float, float, float], condition: BoundaryCondition
) -> np.ndarray:
    """Return the dense one-dimensional operator :mod:`lmx.ops` applies along ``axis``.

    As with :func:`assemble_axis_laplacian`, the other axes are collapsed to a
    single cell under a homogeneous Neumann condition so they contribute
    nothing, and the operator is read out of the production stencil rather than
    written a second time.
    """
    from .ops import staggered_laplacian

    faces = [uniform_faces(1, 0.0, 1.0)] * 3
    faces[axis] = np.asarray(grid.faces[axis])
    line = Grid(*faces)
    line_offset = tuple(offset[axis] if position == axis else CENTER for position in range(3))
    conditions = tuple(
        condition if position == axis else BoundaryCondition("neumann") for position in range(3)
    )
    selection = free_slice(line, axis, offset[axis], condition)
    count = len(range(*selection.indices(line.offset_shape(line_offset)[axis])))
    columns = []
    for index in range(count):
        unit = np.zeros(line.offset_shape(line_offset))
        position = [0, 0, 0]
        position[axis] = range(*selection.indices(unit.shape[axis]))[index]
        unit[tuple(position)] = 1.0
        field = Field(jnp.asarray(unit), line_offset, line)
        applied = np.asarray(staggered_laplacian(field, conditions).data).reshape(-1)
        columns.append(applied[selection])
    return np.stack(columns, axis=1)


@dataclass(frozen=True)
class FastDiagonalHelmholtz:
    """A factorized ``shift * I - coefficient * laplacian`` for one staggered position.

    This is what an implicit viscous solve needs. The operator separates exactly
    as the pressure Laplacian does, so the same host-side eigendecomposition
    turns the solve into three contractions and a divide, and the step is no
    longer bounded by the mesh.
    """

    grid: Grid
    offset: tuple[float, float, float]
    conditions: tuple[BoundaryCondition, BoundaryCondition, BoundaryCondition]
    shift: float
    coefficient: float
    vectors: tuple[np.ndarray, np.ndarray, np.ndarray]
    values: tuple[np.ndarray, np.ndarray, np.ndarray]
    scales: tuple[np.ndarray, np.ndarray, np.ndarray]
    slices: tuple[slice, slice, slice]

    def solve(self, rhs: Field) -> Field:
        """Return the field this operator maps to ``rhs``.

        Prescribed entries are returned as zero: they are boundary data the
        caller owns, not unknowns this solve may set.
        """
        if rhs.grid != self.grid or rhs.offset != self.offset:
            raise ValueError("right-hand side must match the factorized position")
        dtype = rhs.dtype
        interior = rhs.data[self.slices]
        scaled = self._scale(interior, dtype, inverse=False)
        transformed = self._contract(scaled, dtype, transpose=True)
        total = self.values[0][:, None, None] + self.values[1][None, :, None] + self.values[2][None, None, :]
        denominator = jnp.asarray(self.shift - self.coefficient * total, dtype=dtype)
        restored = self._contract(transformed / denominator, dtype, transpose=False)
        solution = self._scale(restored, dtype, inverse=True)
        return rhs.replace_data(jnp.zeros_like(rhs.data).at[self.slices].set(solution))

    def _scale(self, data: jnp.ndarray, dtype, *, inverse: bool) -> jnp.ndarray:
        for axis, scale in enumerate(self.scales):
            factor = 1.0 / scale if inverse else scale
            shape = [-1 if position == axis else 1 for position in range(3)]
            data = data * jnp.asarray(factor.reshape(shape), dtype=dtype)
        return data

    def _contract(self, data: jnp.ndarray, dtype, *, transpose: bool) -> jnp.ndarray:
        for axis, vectors in enumerate(self.vectors):
            matrix = jnp.asarray(vectors.T if transpose else vectors, dtype=dtype)
            data = jnp.moveaxis(jnp.tensordot(matrix, data, axes=([1], [axis])), 0, axis)
        return data


def fast_diagonal_helmholtz(
    grid: Grid,
    offset: tuple[float, float, float],
    conditions: tuple[BoundaryCondition, BoundaryCondition, BoundaryCondition],
    *,
    shift: float = 1.0,
    coefficient: float = 1.0,
) -> FastDiagonalHelmholtz:
    """Factorize ``shift * I - coefficient * laplacian`` at one staggered position."""
    if len(conditions) != 3:
        raise ValueError("a factorization needs one boundary condition per axis")
    for axis, condition in enumerate(conditions):
        if (condition.lower, condition.upper) != (0.0, 0.0):
            raise ValueError(
                f"axis {axis} carries inhomogeneous boundary data; factorize the homogeneous "
                "operator and move the boundary contribution into the right-hand side"
            )
    vectors, values, scales, slices = [], [], [], []
    for axis, condition in enumerate(conditions):
        operator = assemble_staggered_axis_operator(grid, axis, offset, condition)
        selection = free_slice(grid, axis, offset[axis], condition)
        weights = _axis_weights(grid, axis, offset[axis], condition)[selection]
        root = np.sqrt(weights)
        symmetric = root[:, None] * operator / root[None, :]
        asymmetry = np.max(np.abs(symmetric - symmetric.T))
        if asymmetry > _symmetry_tolerance(symmetric):
            raise ValueError(
                f"axis {axis} operator is not symmetric under its cell weights "
                f"(asymmetry {asymmetry:.3e}); fast diagonalization does not apply"
            )
        eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (symmetric + symmetric.T))
        vectors.append(eigenvectors)
        values.append(eigenvalues)
        scales.append(root)
        slices.append(selection)
    return FastDiagonalHelmholtz(
        grid,
        tuple(offset),
        tuple(conditions),
        float(shift),
        float(coefficient),
        tuple(vectors),
        tuple(values),
        tuple(scales),
        tuple(slices),
    )


def _axis_weights(grid: Grid, axis: int, offset_value: float, condition: BoundaryCondition) -> np.ndarray:
    """Return the measure that makes the one-dimensional operator symmetric.

    A cell-centred value owns its cell; a face-centred one owns the dual cell
    spanning the two half-cells on either side of the face.
    """
    widths = np.asarray(grid.widths[axis])
    if offset_value == CENTER:
        return widths
    if condition.is_periodic:
        wrap = 0.5 * (widths[0] + widths[-1])
        return np.concatenate(([wrap], 0.5 * (widths[:-1] + widths[1:]), [wrap]))
    return np.concatenate(([widths[0]], 0.5 * (widths[:-1] + widths[1:]), [widths[-1]]))

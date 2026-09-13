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

``precision="mixed"`` runs the contractions of a float64 solve in float32 and
recovers float64 accuracy by defect correction: the residual against the
assembled operator is formed in float64 and solved again in float32,
``refinements`` times (:func:`solvax.iterative_refinement`). The orthogonal
transforms keep their relative accuracy mode by mode, so the shifted viscous
operator reaches the float64 floor after one correction; the singular
Laplacian on a layer-resolving mesh contracts more slowly and takes two, which
are the factory defaults. The contraction assumes true float32
matmuls; on Ampere GPUs pin ``jax_default_matmul_precision`` to ``"float32"``,
because TensorFloat-32 stalls the correction near 1e-4. Float32 states are
solved in float32 as before.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np
import solvax

from .bc import NEUMANN, PERIODIC, BoundaryCondition
from .grid import CENTER, POLAR, Field, Grid, uniform_faces
from .ops import laplacian, staggered_laplacian

__all__ = [
    "FastDiagonalPolarPoisson",
    "assemble_radial_laplacian",
    "azimuthal_eigenvalues",
    "fast_diagonal_polar_poisson",
    "FastDiagonalHelmholtz",
    "FastDiagonalPoisson",
    "assemble_axis_laplacian",
    "assemble_staggered_axis_operator",
    "fast_diagonal_helmholtz",
    "fast_diagonal_poisson",
    "free_slice",
]

_SINGULAR_TOLERANCE = 1.0e-9
_PRECISIONS = ("state", "mixed")


def _check_precision(precision: str, refinements: int) -> None:
    if precision not in _PRECISIONS:
        raise ValueError(f"precision must be one of {list(_PRECISIONS)}, got {precision!r}")
    if int(refinements) != refinements or refinements < 1:
        raise ValueError("refinements must be a positive integer")


def _single(array: np.ndarray) -> np.ndarray:
    """Cast a host array to float32 once, at factorization, rather than per solve."""
    return np.ascontiguousarray(array, dtype=np.float32)


def _refined(factorization, direct, matvec, data: jnp.ndarray, volumes: np.ndarray | None) -> jnp.ndarray:
    """Return ``direct(data)`` in the precision the factorization asks for.

    A float64 right-hand side under ``precision="mixed"`` is solved in float32
    and corrected against ``matvec`` in float64; anything else is solved
    directly in its own precision. ``volumes`` marks a singular operator: the
    incompatible mean is removed from the right-hand side, from every residual
    and from the result in float64, where the float32 solve would lose it.
    """
    if factorization.precision != "mixed" or data.dtype != jnp.float64:
        return direct(data)
    operator = matvec
    if volumes is not None:
        data = _volume_mean_removed(data, volumes)

        def matvec(values):
            return _volume_mean_removed(operator(values), volumes)

    def single(values):
        return direct(values, single=True)

    solution, _ = solvax.iterative_refinement(
        matvec,
        data,
        solvax.as_low_precision(single, jnp.float32),
        iterations=int(factorization.refinements),
        residual_dtype=data.dtype,
    )
    return solution if volumes is None else _volume_mean_removed(solution, volumes)


def _volume_mean_removed(values: jnp.ndarray, volumes: np.ndarray) -> jnp.ndarray:
    weights = jnp.asarray(volumes, dtype=values.dtype)
    return values - jnp.sum(weights * values) / jnp.sum(weights)


def _single_bases(vectors, scales, denominator: np.ndarray) -> dict:
    """Return the float32 copies a mixed-precision Cartesian solve contracts with."""
    shapes = [tuple(-1 if position == axis else 1 for position in range(3)) for axis in range(3)]
    return {
        "vectors": tuple(_single(vector) for vector in vectors),
        "transposed": tuple(_single(vector.T) for vector in vectors),
        "scales": tuple(_single(scale.reshape(shape)) for scale, shape in zip(scales, shapes, strict=True)),
        "inverse": tuple(
            _single((1.0 / scale).reshape(shape)) for scale, shape in zip(scales, shapes, strict=True)
        ),
        "denominator": _single(denominator),
    }


def _scaled(data: jnp.ndarray, scales, low: dict | None, *, inverse: bool) -> jnp.ndarray:
    for axis, scale in enumerate(scales):
        if low is None:
            factor = 1.0 / scale if inverse else scale
            shape = [-1 if position == axis else 1 for position in range(3)]
            data = data * jnp.asarray(factor.reshape(shape), dtype=data.dtype)
        else:
            data = data * jnp.asarray(low["inverse" if inverse else "scales"][axis])
    return data


def _contracted(data: jnp.ndarray, vectors, low: dict | None, *, transpose: bool) -> jnp.ndarray:
    for axis, vector in enumerate(vectors):
        if low is None:
            matrix = jnp.asarray(vector.T if transpose else vector, dtype=data.dtype)
        else:
            matrix = jnp.asarray(low["transposed" if transpose else "vectors"][axis])
        data = jnp.moveaxis(jnp.tensordot(matrix, data, axes=([1], [axis])), 0, axis)
    return data


def _relative_asymmetry(operator: np.ndarray) -> float:
    """Return the asymmetry of an operator against its own diagonal scale.

    A wall-resolving mesh can span four orders of magnitude in cell width, and
    the entries of the operator span eight. Measured against the largest entry,
    the round-off of the small rows looks like a defect; measured against
    ``sqrt(|d_i d_j|)``, the natural scale of the entry itself, it does not.
    A stencil that is genuinely not symmetric is wrong by an order one fraction
    of its own entries, so this separates the two cleanly.
    """
    diagonal = np.sqrt(np.abs(np.diag(operator)))
    scale = np.outer(diagonal, diagonal)
    floor = max(float(np.max(scale)), 1.0) * float(np.finfo(operator.dtype).eps)
    return float(np.max(np.abs(operator - operator.T) / np.maximum(scale, floor)))


def _symmetry_tolerance(operator: np.ndarray) -> float:
    """Return the relative asymmetry a correct assembly may still show.

    The one-dimensional operators are read out through the production stencil,
    so they are assembled at whatever precision the session runs in. A bound
    tied to float64 would reject a perfectly good float32 assembly, so the
    tolerance follows the dtype.
    """
    return max(1.0e-8, 1.0e11 * float(np.finfo(operator.dtype).eps))


def _require_separable(grid: Grid) -> None:
    if grid.is_polar:
        raise ValueError(
            "fast diagonalization assumes the Laplacian separates into a sum of one-dimensional "
            "operators, which the 1/r^2 azimuthal term of a polar grid does not"
        )


def assemble_axis_laplacian(grid: Grid, axis: int, condition: BoundaryCondition) -> np.ndarray:
    """Return the dense one-dimensional Laplacian :mod:`lmx.ops` applies along ``axis``.

    The other two axes are collapsed to a single cell with a homogeneous
    Neumann condition, which contributes nothing, so the result is exactly the
    stencil the three-dimensional operator uses along ``axis``.
    """
    _require_separable(grid)
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


def assemble_radial_laplacian(grid: Grid, condition: BoundaryCondition) -> np.ndarray:
    """Return the dense radial Laplacian the polar flux form applies.

    The azimuth is collapsed to a single periodic cell, which contributes
    nothing because its two faces carry the same value, and the axial direction
    to a single cell with a homogeneous Neumann condition. The metric factors of
    the azimuth and the axis cancel between the face areas and the cell volume,
    so what is left is exactly ``(1/r) d/dr (r d/dr)`` as the production stencil
    discretizes it, including the zero-area face on the axis.
    """
    count = grid.shape[0]
    line = Grid(
        np.asarray(grid.x_faces),
        uniform_faces(1, 0.0, 2.0 * np.pi),
        uniform_faces(1, 0.0, 1.0),
        geometry=POLAR,
    )
    conditions = (condition, BoundaryCondition(PERIODIC), BoundaryCondition(NEUMANN))
    columns = []
    for index in range(count):
        unit = np.zeros(line.shape)
        unit[index, 0, 0] = 1.0
        field = Field(jnp.asarray(unit), (CENTER,) * 3, line)
        columns.append(np.asarray(laplacian(field, conditions).data).reshape(count))
    return np.stack(columns, axis=1)


def azimuthal_eigenvalues(grid: Grid) -> np.ndarray:
    """Return the eigenvalue of the azimuthal second difference for every mode.

    The discrete Fourier basis diagonalizes a uniform periodic second
    difference, so mode ``m`` contributes ``-4 sin^2(pi m / N) / dtheta^2``
    divided by ``r^2``. That last division is what stops the polar Laplacian
    separating into a sum of one-dimensional operators -- and what makes it
    separate again once the azimuth is transformed, one radial operator per mode.
    """
    widths = np.asarray(grid.widths[1])
    if not np.allclose(widths, widths[0]):
        raise ValueError("the azimuthal transform needs a uniform azimuth")
    count = grid.shape[1]
    modes = np.arange(count)
    return -4.0 * np.sin(np.pi * modes / count) ** 2 / float(widths[0]) ** 2


@dataclass(frozen=True)
class FastDiagonalPolarPoisson:
    """A factorized polar Laplacian: one radial eigendecomposition per azimuthal mode."""

    grid: Grid
    conditions: tuple[BoundaryCondition, BoundaryCondition, BoundaryCondition]
    radial_vectors: np.ndarray
    radial_values: np.ndarray
    radial_scale: np.ndarray
    axial_vectors: np.ndarray
    axial_values: np.ndarray
    axial_scale: np.ndarray
    singular: bool
    shift: float = 0.0
    coefficient: float = -1.0
    precision: str = "state"
    refinements: int = 2
    _low: dict | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _check_precision(self.precision, self.refinements)
        if self.precision == "mixed":
            total = self.radial_values[:, :, None] + self.axial_values[None, None, :]
            denominator = np.moveaxis(self.shift - self.coefficient * total, 0, 1)
            if self.singular:
                denominator[0, 0, 0] = 1.0
            low = {"radial": _single(self.radial_vectors), "axial": _single(self.axial_vectors)}
            object.__setattr__(self, "_low", low | {"denominator": _single(denominator)})

    def solve(self, rhs: Field) -> Field:
        """Return the field this operator maps to ``rhs``.

        The operator is ``shift*I - coefficient*laplacian``; the Poisson factory
        passes ``(0, -1)``, which leaves the Laplacian itself.
        """
        if rhs.grid != self.grid:
            raise ValueError("right-hand side must share the factorized grid")
        if rhs.offset != (CENTER, CENTER, CENTER):
            raise ValueError("the polar factorization solves for a cell-centred field")

        def operator(values):
            return (
                self.shift * values
                - self.coefficient * laplacian(rhs.replace_data(values), self.conditions).data
            )

        volumes = self.grid.cell_volumes() if self.singular else None
        return rhs.replace_data(_refined(self, self._direct, operator, rhs.data, volumes))

    def _direct(self, data: jnp.ndarray, single: bool = False) -> jnp.ndarray:
        dtype = data.dtype
        low = self._low if single else None
        radial_vectors = jnp.asarray(self.radial_vectors if low is None else low["radial"])
        axial_vectors = jnp.asarray(self.axial_vectors if low is None else low["axial"])
        volumes = jnp.asarray(self.grid.cell_volumes(), dtype=dtype)
        if self.singular:
            data = data - jnp.sum(volumes * data) / jnp.sum(volumes)
        data = data * jnp.asarray(self.radial_scale[:, None, None], dtype=dtype)
        data = data * jnp.asarray(self.axial_scale[None, None, :], dtype=dtype)
        transformed = jnp.fft.fft(data, axis=1)
        transformed = jnp.einsum("mji,jmz->imz", radial_vectors, transformed)
        transformed = jnp.tensordot(axial_vectors.T, transformed, axes=([1], [2]))
        transformed = jnp.moveaxis(transformed, 0, 2)
        if low is None:
            total = (
                jnp.asarray(self.radial_values)[:, :, None] + jnp.asarray(self.axial_values)[None, None, :]
            )
            denominator = jnp.moveaxis(self.shift - self.coefficient * total, 0, 1)
            if self.singular:
                denominator = denominator.at[0, 0, 0].set(1.0)
        else:
            denominator = jnp.asarray(low["denominator"])
        if self.singular:
            transformed = transformed.at[0, 0, 0].set(0.0)
        transformed = transformed / denominator
        restored = jnp.tensordot(axial_vectors, transformed, axes=([1], [2]))
        restored = jnp.moveaxis(restored, 0, 2)
        restored = jnp.einsum("mji,imz->jmz", radial_vectors, restored)
        solution = jnp.real(jnp.fft.ifft(restored, axis=1))
        solution = solution / jnp.asarray(self.radial_scale[:, None, None], dtype=dtype)
        solution = solution / jnp.asarray(self.axial_scale[None, None, :], dtype=dtype)
        if self.singular:
            solution = solution - jnp.sum(volumes * solution) / jnp.sum(volumes)
        return solution


def fast_diagonal_polar_poisson(
    grid: Grid,
    conditions: tuple[BoundaryCondition, BoundaryCondition, BoundaryCondition],
    *,
    shift: float = 0.0,
    coefficient: float = -1.0,
    precision: str = "state",
    refinements: int = 2,
) -> FastDiagonalPolarPoisson:
    """Factorize ``shift*I - coefficient*laplacian``, one eigendecomposition per azimuthal mode.

    The default leaves the Laplacian itself. A positive shift with a positive
    coefficient is the damped operator that preconditions a pipe at large
    Hartmann number. ``precision="mixed"`` solves float64 right-hand sides in
    float32 with ``refinements`` float64 corrections (module docstring).
    """
    if not grid.is_polar:
        raise ValueError("this factorization is for a polar grid; use fast_diagonal_poisson")
    if len(conditions) != 3:
        raise ValueError("a factorization needs one boundary condition per axis")
    if not conditions[1].is_periodic:
        raise ValueError("the azimuth of a polar grid is periodic by construction")
    radial = assemble_radial_laplacian(grid, conditions[0])
    radial_weights = np.asarray(grid.centers[0]) * np.asarray(grid.widths[0])
    radial_root = np.sqrt(radial_weights)
    symmetric = radial_root[:, None] * radial / radial_root[None, :]
    asymmetry = _relative_asymmetry(symmetric)
    if asymmetry > _symmetry_tolerance(symmetric):
        raise ValueError(
            f"the radial operator is not symmetric under the annular volumes "
            f"(relative asymmetry {asymmetry:.3e}); fast diagonalization does not apply"
        )
    symmetric = 0.5 * (symmetric + symmetric.T)
    azimuthal = azimuthal_eigenvalues(grid)
    inverse_square = 1.0 / np.asarray(grid.centers[0]) ** 2
    vectors, values = [], []
    for eigenvalue in azimuthal:
        operator = symmetric + np.diag(eigenvalue * inverse_square)
        mode_values, mode_vectors = np.linalg.eigh(operator)
        # Descending, so the zero eigenvalue of a singular mode is first, as the
        # Cartesian factorization also arranges it.
        vectors.append(mode_vectors[:, ::-1])
        values.append(mode_values[::-1])
    axial = assemble_axis_laplacian(
        Grid(*(uniform_faces(1, 0.0, 1.0),) * 2, np.asarray(grid.z_faces)), 2, conditions[2]
    )
    axial_weights = np.asarray(grid.widths[2])
    axial_root = np.sqrt(axial_weights)
    axial_symmetric = axial_root[:, None] * axial / axial_root[None, :]
    axial_values, axial_vectors = np.linalg.eigh(0.5 * (axial_symmetric + axial_symmetric.T))
    axial_values, axial_vectors = axial_values[::-1], axial_vectors[:, ::-1]
    singular = shift == 0.0 and conditions[0].kind == NEUMANN and conditions[2].is_periodic
    return FastDiagonalPolarPoisson(
        grid,
        tuple(conditions),
        np.stack(vectors),
        np.stack(values),
        radial_root,
        axial_vectors,
        axial_values,
        axial_root,
        singular,
        float(shift),
        float(coefficient),
        precision,
        refinements,
    )


@dataclass(frozen=True)
class FastDiagonalPoisson:
    """A factorized Laplacian that solves ``laplacian(u) = rhs`` in three contractions."""

    grid: Grid
    conditions: tuple[BoundaryCondition, BoundaryCondition, BoundaryCondition]
    vectors: tuple[np.ndarray, np.ndarray, np.ndarray]
    values: tuple[np.ndarray, np.ndarray, np.ndarray]
    scales: tuple[np.ndarray, np.ndarray, np.ndarray]
    singular: bool
    precision: str = "state"
    refinements: int = 2
    _low: dict | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _check_precision(self.precision, self.refinements)
        if self.precision == "mixed":
            object.__setattr__(
                self, "_low", _single_bases(self.vectors, self.scales, self._eigenvalue_total())
            )

    def solve(self, rhs: Field) -> Field:
        """Return the field whose Laplacian is ``rhs``."""
        if rhs.grid != self.grid:
            raise ValueError("right-hand side must share the factorized grid")
        if rhs.offset != (CENTER, CENTER, CENTER):
            raise ValueError("right-hand side must be cell centred")

        def operator(values):
            return laplacian(rhs.replace_data(values), self.conditions).data

        volumes = self.grid.cell_volumes() if self.singular else None
        result = _refined(self, self._direct, operator, rhs.data, volumes)
        return Field(result, (CENTER, CENTER, CENTER), self.grid)

    def _direct(self, data: jnp.ndarray, single: bool = False) -> jnp.ndarray:
        dtype = data.dtype
        low = self._low if single else None
        volumes = jnp.asarray(self.grid.cell_volumes(), dtype=dtype)
        if self.singular:
            mean = jnp.sum(volumes * data) / jnp.sum(volumes)
            data = data - mean
        scaled = _scaled(data, self.scales, low, inverse=False)
        transformed = _contracted(scaled, self.vectors, low, transpose=True)
        denominator = self._eigenvalue_sum(dtype) if low is None else jnp.asarray(low["denominator"])
        solution = transformed / denominator
        if self.singular:
            solution = solution.at[0, 0, 0].set(0.0)
        restored = _contracted(solution, self.vectors, low, transpose=False)
        result = _scaled(restored, self.scales, low, inverse=True)
        if self.singular:
            mean = jnp.sum(volumes * result) / jnp.sum(volumes)
            result = result - mean
        return result

    def residual_norm(self, solution: Field, rhs: Field) -> jnp.ndarray:
        """Return the maximum absolute residual of a candidate solution."""
        applied = laplacian(solution, self.conditions)
        difference = applied.data - rhs.data
        if self.singular:
            volumes = jnp.asarray(self.grid.cell_volumes(), dtype=rhs.dtype)
            difference = difference - jnp.sum(volumes * difference) / jnp.sum(volumes)
        return jnp.max(jnp.abs(difference))

    def _eigenvalue_total(self) -> np.ndarray:
        total = self.values[0][:, None, None] + self.values[1][None, :, None] + self.values[2][None, None, :]
        if self.singular:
            total[0, 0, 0] = 1.0
        return total

    def _eigenvalue_sum(self, dtype) -> jnp.ndarray:
        return jnp.asarray(self._eigenvalue_total(), dtype=dtype)


def fast_diagonal_poisson(
    grid: Grid,
    conditions: tuple[BoundaryCondition, BoundaryCondition, BoundaryCondition],
    *,
    precision: str = "state",
    refinements: int = 2,
) -> FastDiagonalPoisson:
    """Factorize the separable Laplacian for ``grid`` under ``conditions``.

    ``precision="mixed"`` solves float64 right-hand sides in float32 with
    ``refinements`` float64 corrections (module docstring).
    """
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
        asymmetry = _relative_asymmetry(symmetric)
        if asymmetry > _symmetry_tolerance(symmetric):
            raise ValueError(
                f"axis {axis} operator is not symmetric under the cell widths "
                f"(relative asymmetry {asymmetry:.3e}); fast diagonalization does not apply"
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
        grid,
        tuple(conditions),
        tuple(vectors),
        tuple(values),
        tuple(scales),
        singular,
        precision,
        refinements,
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
    precision: str = "state"
    refinements: int = 1
    _low: dict | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _check_precision(self.precision, self.refinements)
        if self.precision == "mixed":
            object.__setattr__(self, "_low", _single_bases(self.vectors, self.scales, self._denominator()))

    def solve(self, rhs: Field) -> Field:
        """Return the field this operator maps to ``rhs``.

        Prescribed entries are returned as zero: they are boundary data the
        caller owns, not unknowns this solve may set.
        """
        if rhs.grid != self.grid or rhs.offset != self.offset:
            raise ValueError("right-hand side must match the factorized position")

        def operator(values):
            embedded = rhs.replace_data(jnp.zeros(rhs.shape, dtype=values.dtype).at[self.slices].set(values))
            laplacian_ = staggered_laplacian(embedded, self.conditions).data
            return (self.shift * embedded.data - self.coefficient * laplacian_)[self.slices]

        solution = _refined(self, self._direct, operator, rhs.data[self.slices], None)
        return rhs.replace_data(jnp.zeros_like(rhs.data).at[self.slices].set(solution))

    def _denominator(self) -> np.ndarray:
        total = self.values[0][:, None, None] + self.values[1][None, :, None] + self.values[2][None, None, :]
        return self.shift - self.coefficient * total

    def _direct(self, interior: jnp.ndarray, single: bool = False) -> jnp.ndarray:
        dtype = interior.dtype
        low = self._low if single else None
        scaled = _scaled(interior, self.scales, low, inverse=False)
        transformed = _contracted(scaled, self.vectors, low, transpose=True)
        denominator = (
            jnp.asarray(self._denominator(), dtype=dtype) if low is None else jnp.asarray(low["denominator"])
        )
        restored = _contracted(transformed / denominator, self.vectors, low, transpose=False)
        return _scaled(restored, self.scales, low, inverse=True)


def fast_diagonal_helmholtz(
    grid: Grid,
    offset: tuple[float, float, float],
    conditions: tuple[BoundaryCondition, BoundaryCondition, BoundaryCondition],
    *,
    shift: float = 1.0,
    coefficient: float = 1.0,
    precision: str = "state",
    refinements: int = 1,
) -> FastDiagonalHelmholtz:
    """Factorize ``shift * I - coefficient * laplacian`` at one staggered position.

    ``precision="mixed"`` solves float64 right-hand sides in float32 with
    ``refinements`` float64 corrections (module docstring).
    """
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
        asymmetry = _relative_asymmetry(symmetric)
        if asymmetry > _symmetry_tolerance(symmetric):
            raise ValueError(
                f"axis {axis} operator is not symmetric under its cell weights "
                f"(relative asymmetry {asymmetry:.3e}); fast diagonalization does not apply"
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
        precision,
        refinements,
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

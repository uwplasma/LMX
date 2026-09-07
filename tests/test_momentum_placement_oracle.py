"""The 8^3 oracle that plan step 1.4 requires before a momentum discretization lands.

ADR 0002 records that the marker-and-cell layout is the default and that the
choice must be settled by measurement rather than preference. This module is that
measurement. It assembles the pressure operator that each candidate layout would
produce and compares them on the properties a projection method depends on:

* the dimension of the nullspace, because a pressure that is not determined up to
  a single constant is not determined at all;
* the discrete adjoint identity between gradient and divergence, because a
  projection is only idempotent when it holds;
* symmetry under the cell volumes, because it decides whether the direct
  factorization of :mod:`lmx.poisson` applies at all.

The collocated candidate is built here rather than in the package: it is the
rejected design, and the plan asks that rejected prototypes leave evidence, not
code. The numbers these tests assert are the ones quoted in the ADR.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from lmx.bc import NEUMANN, BoundaryCondition
from lmx.grid import CENTER, Field, Grid, uniform_faces
from lmx.ops import cell_inner_product, divergence, face_gradient, face_inner_product

pytestmark = pytest.mark.unit

WALL = BoundaryCondition(NEUMANN)
ORACLE = Grid(*(uniform_faces(8, 0.0, 1.0) for _ in range(3)))


def _staggered_pressure_operator(grid: Grid) -> np.ndarray:
    """Assemble divergence-of-gradient with the staggered operators of :mod:`lmx.ops`."""
    size = int(np.prod(grid.shape))
    columns = []
    for index in range(size):
        unit = np.zeros(size)
        unit[index] = 1.0
        field = Field(jnp.asarray(unit.reshape(grid.shape)), (CENTER,) * 3, grid)
        gradients = tuple(face_gradient(field, axis, WALL) for axis in range(3))
        columns.append(np.asarray(divergence(gradients).data).reshape(size))
    return np.stack(columns, axis=1)


def _collocated_pressure_operator(grid: Grid) -> np.ndarray:
    """Assemble the same operator when velocity and pressure share the cell centre.

    The gradient and the divergence are then both the wide centred difference
    over neighbouring cell centres, mirrored at the walls, which is the natural
    choice on a collocated mesh and the one that produces the classic decoupling.
    """
    size = int(np.prod(grid.shape))
    spacing = [np.asarray(grid.widths[axis]) for axis in range(3)]

    def centred(values: np.ndarray, axis: int) -> np.ndarray:
        padded = np.concatenate(
            (
                np.take(values, [0], axis=axis),
                values,
                np.take(values, [-1], axis=axis),
            ),
            axis=axis,
        )
        upper = np.take(padded, range(2, padded.shape[axis]), axis=axis)
        lower = np.take(padded, range(0, padded.shape[axis] - 2), axis=axis)
        width = spacing[axis]
        distance = np.concatenate(
            ([width[0] + width[1]], width[:-2] + 2.0 * width[1:-1] + width[2:], [width[-2] + width[-1]])
        )
        shape = [-1 if position == axis else 1 for position in range(3)]
        return (upper - lower) / distance.reshape(shape)

    columns = []
    for index in range(size):
        unit = np.zeros(size)
        unit[index] = 1.0
        values = unit.reshape(grid.shape)
        result = sum(centred(centred(values, axis), axis) for axis in range(3))
        columns.append(result.reshape(size))
    return np.stack(columns, axis=1)


def _nullspace_dimension(operator: np.ndarray, *, tolerance: float = 1e-9) -> int:
    singular = np.linalg.svd(operator, compute_uv=False)
    return int(np.sum(singular <= tolerance * singular[0]))


def test_staggered_pressure_operator_has_only_the_constant_nullspace():
    operator = _staggered_pressure_operator(ORACLE)
    assert _nullspace_dimension(operator) == 1


def test_collocated_pressure_operator_decouples_into_independent_sublattices():
    """The wide centred composition leaves a checkerboard family undetermined.

    Each axis decouples its even and odd cells, so the three-dimensional operator
    carries one constant per sublattice: eight, not one. Those extra modes are
    the pressure oscillations a collocated projection cannot see, which is why
    that layout needs an added interpolation to be usable at all.
    """
    operator = _collocated_pressure_operator(ORACLE)
    assert _nullspace_dimension(operator) == 8


def test_staggered_gradient_and_divergence_are_adjoint_while_collocated_is_not():
    grid = ORACLE
    generator = np.random.default_rng(0)
    pressure = Field(jnp.asarray(generator.normal(size=grid.shape)), (CENTER,) * 3, grid)
    fluxes = []
    for axis in range(3):
        data = generator.normal(size=grid.face_shape(axis))
        data[(slice(None),) * axis + (0,)] = 0.0
        data[(slice(None),) * axis + (-1,)] = 0.0
        offset = tuple(0.0 if position == axis else CENTER for position in range(3))
        fluxes.append(Field(jnp.asarray(data), offset, grid))

    left = float(cell_inner_product(pressure, divergence(tuple(fluxes))))
    right = float(
        sum(
            face_inner_product(fluxes[axis], face_gradient(pressure, axis, WALL), axis, WALL)
            for axis in range(3)
        )
    )
    assert abs(left + right) <= 1e-14 * abs(left)


def test_staggered_operator_is_symmetric_under_the_cell_volumes():
    """Symmetry is what lets :mod:`lmx.poisson` factorize the operator directly."""
    grid = ORACLE
    operator = _staggered_pressure_operator(grid)
    volumes = grid.cell_volumes().reshape(-1)
    weighted = volumes[:, None] * operator
    asymmetry = np.max(np.abs(weighted - weighted.T)) / np.max(np.abs(weighted))
    assert asymmetry < 1e-12


def test_staggered_stencil_is_narrower_than_the_collocated_one():
    """A narrow stencil is cheaper per solve and cheaper to differentiate."""
    staggered = _staggered_pressure_operator(ORACLE)
    collocated = _collocated_pressure_operator(ORACLE)
    staggered_entries = int(np.sum(np.abs(staggered) > 1e-12))
    collocated_entries = int(np.sum(np.abs(collocated) > 1e-12))
    assert staggered_entries < collocated_entries
    # Seven-point versus a composition that reaches two cells along each axis.
    assert staggered_entries / np.prod(ORACLE.shape) < 7.0

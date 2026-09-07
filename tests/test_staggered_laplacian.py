"""The Laplacian of a staggered field, which a marker-and-cell velocity needs.

A velocity component is cell-centred along two axes and face-centred along the
third, so its diffusion term mixes both stencils. These tests check each in
isolation and then together, and record the deliberate treatment of the two
boundary faces on a face-centred axis.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lmx.bc import DIRICHLET, NEUMANN, PERIODIC, BoundaryCondition
from lmx.grid import CENTER, FACE, Field, Grid, geometric_faces, uniform_faces
from lmx.ops import staggered_laplacian

pytestmark = pytest.mark.unit

WALL = BoundaryCondition(NEUMANN)
WRAPPED = BoundaryCondition(PERIODIC)
UNIFORM = Grid(uniform_faces(8, 0.0, 2.0), uniform_faces(6, -1.0, 1.0), uniform_faces(4, -0.5, 0.5))
STRETCHED = Grid(geometric_faces(8, 0.0, 2.0, 1.2), uniform_faces(6, -1.0, 1.0), uniform_faces(4, -0.5, 0.5))


def _positions(grid: Grid, offset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coordinates = []
    for axis in range(3):
        values = grid.faces[axis] if offset[axis] == FACE else grid.centers[axis]
        coordinates.append(np.asarray(values))
    x, y, z = coordinates
    return x[:, None, None], y[None, :, None], z[None, None, :]


def _field(grid: Grid, offset, function) -> Field:
    x, y, z = _positions(grid, offset)
    values = np.broadcast_to(function(x, y, z), grid.offset_shape(offset))
    return Field(jnp.asarray(values, dtype=jnp.float64), offset, grid)


AXIAL = (FACE, CENTER, CENTER)


def test_linear_field_has_zero_laplacian_everywhere_it_is_defined():
    grid = STRETCHED
    field = _field(grid, AXIAL, lambda x, y, z: 2.0 * x - 3.0 * y + z)
    conditions = (WALL, BoundaryCondition(DIRICHLET), WALL)
    result = np.asarray(staggered_laplacian(field, conditions).data)
    # Cell-centred axes carry a wall condition, so only the interior is defined.
    assert np.max(np.abs(result[1:-1, 1:-1, 1:-1])) < 1e-12


def test_quadratic_along_the_face_axis_is_exact_on_interior_faces():
    grid = UNIFORM
    field = _field(grid, AXIAL, lambda x, y, z: x**2 + 0.0 * (y + z))
    result = np.asarray(staggered_laplacian(field, (WALL, WALL, WALL)).data)
    assert np.max(np.abs(result[1:-1] - 2.0)) < 1e-12


def test_boundary_faces_of_a_walled_face_axis_are_returned_as_zero():
    """Their value is prescribed, not evolved, so no one-sided stencil is invented."""
    grid = UNIFORM
    field = _field(grid, AXIAL, lambda x, y, z: x**2 + 0.0 * (y + z))
    result = np.asarray(staggered_laplacian(field, (WALL, WALL, WALL)).data)
    assert np.all(result[0] == 0.0)
    assert np.all(result[-1] == 0.0)


def test_periodic_face_axis_wraps_and_matches_the_analytic_second_derivative():
    grid = Grid(uniform_faces(32, 0.0, 2.0 * np.pi), uniform_faces(3, 0.0, 1.0), uniform_faces(3, 0.0, 1.0))
    field = _field(grid, AXIAL, lambda x, y, z: np.sin(x) + 0.0 * (y + z))
    result = np.asarray(staggered_laplacian(field, (WRAPPED, WALL, WALL)).data)
    expected = -np.asarray(np.sin(grid.faces[0]))[:, None, None]
    # Second-order accuracy on 32 cells over a full period.
    assert np.max(np.abs(result - expected)) < 5e-3
    assert np.allclose(result[0], result[-1], atol=1e-14)


def test_quadratic_along_a_cell_centred_axis_is_exact_in_the_interior():
    grid = UNIFORM
    field = _field(grid, AXIAL, lambda x, y, z: y**2 + 0.0 * (x + z))
    faces = np.asarray(grid.faces[1])
    condition = BoundaryCondition(DIRICHLET, lower=float(faces[0]) ** 2, upper=float(faces[-1]) ** 2)
    result = np.asarray(staggered_laplacian(field, (WALL, condition, WALL)).data)
    assert np.max(np.abs(result[1:-1, 1:-1, :] - 2.0)) < 1e-12


def test_all_three_axes_add_up():
    grid = UNIFORM
    field = _field(grid, AXIAL, lambda x, y, z: x**2 + y**2 + z**2)
    conditions = []
    for axis in range(3):
        faces = np.asarray(grid.faces[axis])
        conditions.append(
            BoundaryCondition(DIRICHLET, lower=float(faces[0]) ** 2, upper=float(faces[-1]) ** 2)
        )
    result = np.asarray(staggered_laplacian(field, tuple(conditions)).data)
    assert np.max(np.abs(result[1:-1, 1:-1, 1:-1] - 6.0)) < 1e-12


@pytest.mark.parametrize("offset", [(FACE, CENTER, CENTER), (CENTER, FACE, CENTER), (CENTER, CENTER, FACE)])
def test_every_velocity_position_is_supported(offset):
    grid = UNIFORM
    field = _field(grid, offset, lambda x, y, z: x + y + z)
    result = staggered_laplacian(field, (WALL, WALL, WALL))
    assert result.offset == offset
    assert result.shape == grid.offset_shape(offset)


def test_the_interior_operator_is_symmetric():
    """An implicit viscous solve needs symmetry; this checks it on the free faces."""
    grid = UNIFORM
    conditions = (WALL, BoundaryCondition(DIRICHLET), BoundaryCondition(DIRICHLET))
    shape = grid.offset_shape(AXIAL)
    interior = np.zeros(shape, dtype=bool)
    interior[1:-1] = True
    indices = np.argwhere(interior)
    operator = np.zeros((len(indices), len(indices)))
    for column, position in enumerate(indices):
        unit = np.zeros(shape)
        unit[tuple(position)] = 1.0
        applied = np.asarray(staggered_laplacian(Field(jnp.asarray(unit), AXIAL, grid), conditions).data)
        operator[:, column] = applied[interior]
    asymmetry = np.max(np.abs(operator - operator.T)) / np.max(np.abs(operator))
    assert asymmetry < 1e-12


def test_the_staggered_laplacian_differentiates_and_jits():
    grid = UNIFORM
    conditions = (WALL, WALL, WALL)

    def energy(values):
        field = Field(values, AXIAL, grid)
        return jnp.sum(staggered_laplacian(field, conditions).data ** 2)

    values = jax.random.normal(jax.random.PRNGKey(0), grid.offset_shape(AXIAL), dtype=jnp.float64)
    gradient = jax.jit(jax.grad(energy))(values)
    step = 1.0e-6
    direction = jax.random.normal(jax.random.PRNGKey(1), grid.offset_shape(AXIAL), dtype=jnp.float64)
    difference = (energy(values + step * direction) - energy(values - step * direction)) / (2.0 * step)
    assert abs(float(jnp.sum(gradient * direction)) - float(difference)) < 1e-6 * abs(float(difference))


def test_the_staggered_laplacian_validates_its_inputs():
    grid = UNIFORM
    field = _field(grid, AXIAL, lambda x, y, z: x + 0.0 * (y + z))
    with pytest.raises(ValueError, match="one boundary condition per axis"):
        staggered_laplacian(field, (WALL, WALL))
    with pytest.raises(ValueError, match="does not match its offset"):
        staggered_laplacian(Field(jnp.zeros((2, 2, 2)), AXIAL, grid), (WALL, WALL, WALL))

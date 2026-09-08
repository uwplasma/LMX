"""Boundary padding and the compatible gradient, divergence and Laplacian."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lmx.bc import DIRICHLET, NEUMANN, PERIODIC, BoundaryCondition, pad
from lmx.grid import CENTER, FACE, POLAR, Field, Grid, geometric_faces, tanh_faces, uniform_faces
from lmx.ops import (
    cell_inner_product,
    divergence,
    face_distances,
    face_gradient,
    face_inner_product,
    face_interpolate,
    laplacian,
    staggered_laplacian,
)

pytestmark = pytest.mark.unit

WALL = BoundaryCondition(NEUMANN)
WRAP = BoundaryCondition(PERIODIC)
STRETCHED = Grid(
    geometric_faces(6, 0.0, 3.0, 1.25),
    tanh_faces(8, -1.0, 1.0, 1.6),
    uniform_faces(5, -0.5, 0.5),
)


def _cells(grid: Grid, function) -> Field:
    x, y, z = (np.asarray(values) for values in grid.centers)
    values = function(x[:, None, None], y[None, :, None], z[None, None, :])
    data = jnp.broadcast_to(jnp.asarray(values, dtype=jnp.float64), grid.shape)
    return Field(data, (CENTER,) * 3, grid)


def _faces(grid: Grid, axis: int, function) -> Field:
    coordinates = [np.asarray(values) for values in grid.centers]
    coordinates[axis] = np.asarray(grid.faces[axis])
    x, y, z = coordinates
    values = function(x[:, None, None], y[None, :, None], z[None, None, :])
    data = jnp.broadcast_to(jnp.asarray(values, dtype=jnp.float64), grid.face_shape(axis))
    offset = tuple(FACE if position == axis else CENTER for position in range(3))
    return Field(data, offset, grid)


def test_constant_field_has_zero_gradient_on_every_interior_face():
    field = _cells(STRETCHED, lambda x, y, z: jnp.full(jnp.broadcast_shapes(x.shape, y.shape, z.shape), 2.5))
    for axis in range(3):
        gradient = np.asarray(face_gradient(field, axis, WALL).data)
        interior = gradient[(slice(None),) * axis + (slice(1, -1),)]
        assert np.max(np.abs(interior)) < 1e-15
        assert np.max(np.abs(gradient)) < 1e-15  # a homogeneous Neumann wall is flat too


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_linear_field_gradient_is_exact_including_the_walls(axis):
    slope, grid = 1.7, STRETCHED
    field = _cells(grid, lambda x, y, z: slope * (x, y, z)[axis])
    faces = np.asarray(grid.faces[axis])
    condition = BoundaryCondition(DIRICHLET, lower=slope * faces[0], upper=slope * faces[-1])
    gradient = np.asarray(face_gradient(field, axis, condition).data)
    assert np.max(np.abs(gradient - slope)) < 1e-13


def test_divergence_of_a_linear_flux_matches_its_analytic_value():
    grid = STRETCHED
    faces = (
        _faces(grid, 0, lambda x, y, z: 2.0 * x),
        _faces(grid, 1, lambda x, y, z: -3.0 * y),
        _faces(grid, 2, lambda x, y, z: 0.5 * z),
    )
    result = np.asarray(divergence(faces).data)
    assert np.max(np.abs(result - (2.0 - 3.0 + 0.5))) < 1e-12


UNIFORM = Grid(uniform_faces(6, 0.0, 3.0), uniform_faces(8, -1.0, 1.0), uniform_faces(5, -0.5, 0.5))


def _quadratic_conditions(grid, axis):
    faces = np.asarray(grid.faces[axis])
    return tuple(
        BoundaryCondition(DIRICHLET, lower=float(faces[0]) ** 2, upper=float(faces[-1]) ** 2)
        if position == axis
        else WALL
        for position in range(3)
    )


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_laplacian_of_a_quadratic_is_exact_away_from_the_walls(axis):
    # One axis at a time keeps the wall value constant along the wall, which a
    # scalar Dirichlet condition can represent exactly.
    grid = UNIFORM
    field = _cells(grid, lambda x, y, z: (x, y, z)[axis] ** 2)
    result = np.asarray(laplacian(field, _quadratic_conditions(grid, axis)).data)
    interior = result[(slice(None),) * axis + (slice(1, -1),)]
    assert np.max(np.abs(interior - 2.0)) < 1e-12


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_two_point_wall_flux_leaves_its_known_defect_in_the_wall_cell(axis):
    """Pin the accuracy of the symmetric two-point Dirichlet flux at the wall.

    The wall face carries ``(p_0 - g) / (dx_0 / 2)``, a difference centred a
    quarter cell inside the wall, so for a quadratic the wall cell's Laplacian
    returns 1.5 rather than 2. Keeping that stencil is what makes the operator
    symmetric and conservative; a three-point wall stencil would raise the wall
    order at the cost of that symmetry and must change this test deliberately.
    """
    grid = UNIFORM
    field = _cells(grid, lambda x, y, z: (x, y, z)[axis] ** 2)
    result = np.asarray(laplacian(field, _quadratic_conditions(grid, axis)).data)
    for wall in (0, -1):
        assert np.max(np.abs(result[(slice(None),) * axis + (wall,)] - 1.5)) < 1e-12


@pytest.mark.parametrize("axis", [0, 1])
def test_stretched_truncation_error_is_bounded_by_the_mesh_nonuniformity(axis):
    """A stretched two-point gradient is centred between cell centres, not on the face.

    The resulting Laplacian truncation error is first order in the spacing
    change, which is expected: solution order on stretched meshes is a
    manufactured-solution question and is verified where that study lives.
    """
    grid = STRETCHED
    field = _cells(grid, lambda x, y, z: (x, y, z)[axis] ** 2)
    result = np.asarray(laplacian(field, _quadratic_conditions(grid, axis)).data)
    interior = result[(slice(None),) * axis + (slice(1, -1),)]
    widths = np.asarray(grid.widths[axis])
    bound = np.max(np.abs(np.diff(widths))) / np.min(widths)
    assert np.max(np.abs(interior - 2.0)) <= 2.0 * bound


def test_gradient_and_divergence_are_exact_discrete_adjoints():
    grid, key = STRETCHED, jax.random.PRNGKey(0)
    pressure_key, flux_key = jax.random.split(key)
    pressure = Field(jax.random.normal(pressure_key, grid.shape, dtype=jnp.float64), (CENTER,) * 3, grid)
    faces, keys = [], jax.random.split(flux_key, 3)
    for axis in range(3):
        data = jax.random.normal(keys[axis], grid.face_shape(axis), dtype=jnp.float64)
        # A wall-impermeable flux removes the boundary term from the identity.
        data = data.at[(slice(None),) * axis + (0,)].set(0.0)
        data = data.at[(slice(None),) * axis + (-1,)].set(0.0)
        offset = tuple(FACE if position == axis else CENTER for position in range(3))
        faces.append(Field(data, offset, grid))

    left = cell_inner_product(pressure, divergence(tuple(faces)))
    right = sum(
        face_inner_product(faces[axis], face_gradient(pressure, axis, WALL), axis, WALL) for axis in range(3)
    )
    assert float(jnp.abs(left + right)) <= 1e-14 * float(jnp.abs(left))


def test_periodic_gradient_wraps_and_its_divergence_sums_to_zero():
    grid = Grid(uniform_faces(8, 0.0, 2.0 * np.pi), uniform_faces(2, 0.0, 1.0), uniform_faces(2, 0.0, 1.0))
    periodic = BoundaryCondition(PERIODIC)
    field = _cells(grid, lambda x, y, z: jnp.sin(x) + 0.0 * (y + z))
    gradient = face_gradient(field, 0, periodic)
    values = np.asarray(gradient.data)
    assert np.allclose(values[0], values[-1], atol=1e-15)
    faces = (gradient, _faces(grid, 1, lambda x, y, z: 0.0 * x), _faces(grid, 2, lambda x, y, z: 0.0 * x))
    volumes = grid.cell_volumes()
    assert abs(float(np.sum(volumes * np.asarray(divergence(faces).data)))) < 1e-13


def test_face_interpolation_is_exact_for_a_linear_field():
    grid, axis = STRETCHED, 1
    field = _cells(grid, lambda x, y, z: 3.0 * y - 1.0)
    faces = np.asarray(grid.faces[axis])
    condition = BoundaryCondition(DIRICHLET, lower=3.0 * faces[0] - 1.0, upper=3.0 * faces[-1] - 1.0)
    interpolated = np.asarray(face_interpolate(field, axis, condition).data)
    expected = 3.0 * faces - 1.0
    assert np.max(np.abs(interpolated - expected.reshape(1, -1, 1))) < 1e-13


def test_neumann_padding_reproduces_a_prescribed_wall_gradient():
    grid, axis, flux = STRETCHED, 0, 0.75
    field = _cells(grid, lambda x, y, z: 0.0 * x)
    condition = BoundaryCondition(NEUMANN, lower=flux, upper=flux)
    gradient = np.asarray(face_gradient(field, axis, condition).data)
    assert np.allclose(gradient[0], flux, atol=1e-14)
    assert np.allclose(gradient[-1], flux, atol=1e-14)


def test_dirichlet_padding_places_the_prescribed_value_on_the_wall():
    grid, axis = STRETCHED, 2
    field = _cells(grid, lambda x, y, z: 1.0 + 0.0 * z)
    condition = BoundaryCondition(DIRICHLET, lower=-2.0, upper=4.0)
    padded = np.asarray(pad(field.data, axis, condition, grid=grid))
    assert np.allclose(0.5 * (padded[:, :, 0] + padded[:, :, 1]), -2.0, atol=1e-15)
    assert np.allclose(0.5 * (padded[:, :, -1] + padded[:, :, -2]), 4.0, atol=1e-15)


def test_periodic_padding_wraps_both_ends():
    data = jnp.arange(24.0).reshape(2, 3, 4)
    padded = np.asarray(pad(data, 2, BoundaryCondition(PERIODIC)))
    assert np.array_equal(padded[:, :, 0], np.asarray(data)[:, :, -1])
    assert np.array_equal(padded[:, :, -1], np.asarray(data)[:, :, 0])


def test_face_distances_span_the_axis_and_wrap_when_periodic():
    grid, axis = STRETCHED, 0
    widths = np.asarray(grid.widths[axis])
    wall = face_distances(grid, axis, WALL)
    assert wall.size == grid.shape[axis] + 1
    assert wall[0] == pytest.approx(widths[0]) and wall[-1] == pytest.approx(widths[-1])
    wrapped = face_distances(grid, axis, BoundaryCondition(PERIODIC))
    assert wrapped[0] == wrapped[-1] == pytest.approx(0.5 * (widths[0] + widths[-1]))


def test_operators_preserve_single_precision():
    grid = STRETCHED
    field = Field(jnp.ones(grid.shape, dtype=jnp.float32), (CENTER,) * 3, grid)
    gradient = face_gradient(field, 0, WALL)
    assert gradient.dtype == jnp.float32
    assert face_interpolate(field, 0, WALL).dtype == jnp.float32


def test_operators_compose_under_jit_and_differentiation():
    grid = STRETCHED
    conditions = (WALL, WALL, WALL)

    def energy(values):
        field = Field(values, (CENTER,) * 3, grid)
        return jnp.sum(np.asarray(grid.cell_volumes()) * laplacian(field, conditions).data ** 2)

    values = jax.random.normal(jax.random.PRNGKey(1), grid.shape, dtype=jnp.float64)
    gradient = jax.jit(jax.grad(energy))(values)
    assert gradient.shape == grid.shape and np.all(np.isfinite(np.asarray(gradient)))


def test_boundary_conditions_validate_their_inputs():
    with pytest.raises(ValueError, match="unknown boundary kind"):
        BoundaryCondition("sponge")
    with pytest.raises(ValueError, match="do not take prescribed values"):
        BoundaryCondition(PERIODIC, lower=1.0)
    with pytest.raises(ValueError, match="requires the grid"):
        pad(jnp.zeros((2, 2, 2)), 0, BoundaryCondition(NEUMANN, lower=1.0))
    with pytest.raises(ValueError, match="three-dimensional"):
        pad(jnp.zeros((2, 2)), 0, WALL)
    with pytest.raises(ValueError, match="out of range"):
        pad(jnp.zeros((2, 2, 2)), 3, WALL)


def test_operators_reject_mismatched_positions_and_grids():
    grid = STRETCHED
    cell = Field(jnp.zeros(grid.shape), (CENTER,) * 3, grid)
    face = Field(jnp.zeros(grid.face_shape(0)), (FACE, CENTER, CENTER), grid)
    with pytest.raises(ValueError, match="expected a cell-centred field"):
        face_gradient(face, 0, WALL)
    with pytest.raises(ValueError, match="expected a field on faces"):
        divergence((cell, cell, cell))
    with pytest.raises(ValueError, match="does not match grid"):
        face_gradient(Field(jnp.zeros((2, 2, 2)), (CENTER,) * 3, grid), 0, WALL)
    with pytest.raises(ValueError, match="one boundary condition per axis"):
        laplacian(cell, (WALL, WALL))
    other = Grid(uniform_faces(6, 0.0, 3.0), *grid.faces[1:])
    with pytest.raises(ValueError, match="share one grid"):
        cell_inner_product(cell, Field(jnp.zeros(other.shape), (CENTER,) * 3, other))
    with pytest.raises(ValueError, match="share one grid"):
        divergence((face, Field(jnp.zeros(other.face_shape(1)), (CENTER, FACE, CENTER), other), face))


def test_homogeneous_neumann_padding_needs_no_grid():
    data = jnp.arange(8.0).reshape(2, 2, 2)
    padded = np.asarray(pad(data, 1, WALL))
    assert np.array_equal(padded[:, 0], np.asarray(data)[:, 0])
    assert np.array_equal(padded[:, -1], np.asarray(data)[:, -1])


def test_face_inner_product_rejects_mismatched_grids_and_shapes():
    grid = STRETCHED
    other = Grid(uniform_faces(6, 0.0, 3.0), *grid.faces[1:])
    face = Field(jnp.zeros(grid.face_shape(0)), (FACE, CENTER, CENTER), grid)
    with pytest.raises(ValueError, match="share one grid"):
        face_inner_product(
            face, Field(jnp.zeros(other.face_shape(0)), (FACE, CENTER, CENTER), other), 0, WALL
        )
    with pytest.raises(ValueError, match="does not match grid"):
        face_inner_product(face, Field(jnp.zeros((2, 2, 2)), (FACE, CENTER, CENTER), grid), 0, WALL)


def _polar(radial: int, azimuthal: int) -> Grid:
    return Grid(
        uniform_faces(radial, 0.0, 1.0),
        uniform_faces(azimuthal, 0.0, 2.0 * np.pi),
        uniform_faces(1, 0.0, 1.0),
        geometry=POLAR,
    )


def _polar_cells(grid: Grid, function) -> Field:
    radius, angle, axial = (np.asarray(values) for values in grid.centers)
    r, theta, z = np.meshgrid(radius, angle, axial, indexing="ij")
    return Field(jnp.asarray(function(r, theta)), (CENTER, CENTER, CENTER), grid)


def test_the_flux_form_laplacian_is_exact_on_a_paraboloid():
    """`1 - r^2` has Laplacian `-4` everywhere, and the flux form reproduces it exactly.

    Everywhere except the wall cell: the two-point wall flux is first order, and
    a cell whose volume is also first order therefore carries an order-one error
    in the Laplacian. That is the same closure the Cartesian operator uses, and
    it is why the Poisson *solution* stays second order while this pointwise
    reading of the operator does not.
    """
    grid = _polar(16, 32)
    field = _polar_cells(grid, lambda r, theta: 1.0 - r**2)
    values = np.asarray(laplacian(field, (BoundaryCondition(DIRICHLET), WRAP, WRAP)).data)
    assert np.max(np.abs(values[:-1] + 4.0)) < 1e-12
    assert abs(values[-1, 0, 0] + 4.0) > 0.1


def test_the_polar_laplacian_is_second_order_away_from_the_axis():
    """`r cos(theta)` is harmonic; the radial and azimuthal terms cancel at order `1/r`."""
    errors = []
    for count in (16, 32, 64):
        grid = _polar(count, 2 * count)
        field = _polar_cells(grid, lambda r, theta: r * np.cos(theta))
        values = np.asarray(laplacian(field, (BoundaryCondition(DIRICHLET), WRAP, WRAP)).data)
        radius = np.asarray(grid.centers[0])
        inside = (radius > 0.2) & (radius < 0.95)
        errors.append(float(np.max(np.abs(values[inside]))))
    orders = [np.log2(errors[index] / errors[index + 1]) for index in range(2)]
    assert min(orders) > 1.8, orders
    # Against the axis the two terms are each of order 1/r, so their cancellation
    # loses an order. A pipe resolves its wall layers, not its centre.
    grid = _polar(32, 64)
    field = _polar_cells(grid, lambda r, theta: r * np.cos(theta))
    values = np.asarray(laplacian(field, (BoundaryCondition(DIRICHLET), WRAP, WRAP)).data)
    assert np.max(np.abs(values[0])) > 2.0 * errors[1]


def test_the_separable_stencils_refuse_a_polar_grid():
    """A wrong answer on a metric a stencil does not carry is worse than no answer."""
    grid = _polar(4, 8)
    field = _polar_cells(grid, lambda r, theta: r)
    with pytest.raises(ValueError, match="use lmx.ops.laplacian"):
        staggered_laplacian(field, (BoundaryCondition(DIRICHLET), WRAP, WRAP))
    from lmx.poisson import assemble_axis_laplacian

    with pytest.raises(ValueError, match="fast diagonalization assumes"):
        assemble_axis_laplacian(grid, 0, BoundaryCondition(DIRICHLET))

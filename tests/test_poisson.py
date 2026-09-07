"""Direct Poisson solves against dense factorizations and the assembled operator."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lmx.bc import DIRICHLET, NEUMANN, PERIODIC, BoundaryCondition
from lmx.grid import CENTER, Field, Grid, geometric_faces, tanh_faces, uniform_faces
from lmx.ops import laplacian
from lmx.poisson import assemble_axis_laplacian, fast_diagonal_poisson

pytestmark = pytest.mark.unit

WALL = BoundaryCondition(NEUMANN)
FIXED = BoundaryCondition(DIRICHLET)
WRAPPED = BoundaryCondition(PERIODIC)
STRETCHED = Grid(
    geometric_faces(6, 0.0, 3.0, 1.25),
    tanh_faces(8, -1.0, 1.0, 1.6),
    uniform_faces(5, -0.5, 0.5),
)
SMALL = Grid(*(uniform_faces(4, 0.0, 1.0) for _ in range(3)))


def _random_cells(grid: Grid, seed: int = 0) -> Field:
    data = jax.random.normal(jax.random.PRNGKey(seed), grid.shape, dtype=jnp.float64)
    return Field(data, (CENTER,) * 3, grid)


def _dense_operator(grid: Grid, conditions) -> np.ndarray:
    """Assemble the full Laplacian by applying it to unit vectors."""
    size = int(np.prod(grid.shape))
    columns = []
    for index in range(size):
        unit = np.zeros(size)
        unit[index] = 1.0
        field = Field(jnp.asarray(unit.reshape(grid.shape)), (CENTER,) * 3, grid)
        columns.append(np.asarray(laplacian(field, conditions).data).reshape(size))
    return np.stack(columns, axis=1)


@pytest.mark.parametrize(
    "conditions",
    [
        (FIXED, FIXED, FIXED),
        (FIXED, WALL, FIXED),
        (WRAPPED, FIXED, FIXED),
        (WRAPPED, WRAPPED, FIXED),
    ],
    ids=["dirichlet", "mixed-neumann", "one-periodic", "two-periodic"],
)
def test_direct_solve_inverts_the_assembled_operator(conditions):
    grid = STRETCHED
    factorization = fast_diagonal_poisson(grid, conditions)
    assert not factorization.singular
    exact = _random_cells(grid, seed=3)
    rhs = laplacian(exact, conditions)
    solution = factorization.solve(rhs)
    assert float(jnp.max(jnp.abs(solution.data - exact.data))) < 1e-11
    assert float(factorization.residual_norm(solution, rhs)) < 1e-11


def test_direct_solve_matches_a_dense_factorization_on_a_small_grid():
    grid, conditions = SMALL, (FIXED, FIXED, WALL)
    dense = _dense_operator(grid, conditions)
    rhs = _random_cells(grid, seed=5)
    expected = np.linalg.solve(dense, np.asarray(rhs.data).reshape(-1)).reshape(grid.shape)
    solution = fast_diagonal_poisson(grid, conditions).solve(rhs)
    assert np.max(np.abs(np.asarray(solution.data) - expected)) < 1e-12


def test_axis_operator_reproduces_the_three_dimensional_stencil():
    grid, axis = STRETCHED, 1
    operator = assemble_axis_laplacian(grid, axis, FIXED)
    assert operator.shape == (grid.shape[axis],) * 2
    profile = jax.random.normal(jax.random.PRNGKey(7), (grid.shape[axis],), dtype=jnp.float64)
    field = Field(jnp.broadcast_to(profile[None, :, None], grid.shape), (CENTER,) * 3, grid)
    conditions = tuple(FIXED if position == axis else WALL for position in range(3))
    expected = np.asarray(laplacian(field, conditions).data)[0, :, 0]
    assert np.max(np.abs(operator @ np.asarray(profile) - expected)) < 1e-12


@pytest.mark.parametrize(
    "conditions",
    [(WALL, WALL, WALL), (WRAPPED, WRAPPED, WRAPPED), (WALL, WRAPPED, WALL)],
    ids=["all-neumann", "all-periodic", "neumann-periodic"],
)
def test_singular_problems_solve_up_to_a_constant(conditions):
    grid = STRETCHED
    factorization = fast_diagonal_poisson(grid, conditions)
    assert factorization.singular
    volumes = np.asarray(grid.cell_volumes())
    exact = _random_cells(grid, seed=11)
    centred = exact.replace_data(
        exact.data - jnp.sum(jnp.asarray(volumes) * exact.data) / jnp.sum(jnp.asarray(volumes))
    )
    rhs = laplacian(centred, conditions)
    solution = factorization.solve(rhs)
    assert abs(float(jnp.sum(jnp.asarray(volumes) * solution.data))) < 1e-12
    assert float(jnp.max(jnp.abs(solution.data - centred.data))) < 1e-11
    assert float(factorization.residual_norm(solution, rhs)) < 1e-11


def test_singular_solve_discards_the_incompatible_component():
    grid, conditions = SMALL, (WALL, WALL, WALL)
    factorization = fast_diagonal_poisson(grid, conditions)
    rhs = _random_cells(grid, seed=13)
    solution = factorization.solve(rhs)
    # The constant part of the right-hand side cannot be represented; the rest is.
    assert float(factorization.residual_norm(solution, rhs)) < 1e-11


def test_solution_is_differentiable_and_jits():
    grid, conditions = SMALL, (FIXED, FIXED, FIXED)
    factorization = fast_diagonal_poisson(grid, conditions)

    def energy(values):
        solution = factorization.solve(Field(values, (CENTER,) * 3, grid))
        return jnp.sum(solution.data**2)

    values = jax.random.normal(jax.random.PRNGKey(17), grid.shape, dtype=jnp.float64)
    gradient = jax.jit(jax.grad(energy))(values)
    step, direction = 1.0e-6, jax.random.normal(jax.random.PRNGKey(19), grid.shape, dtype=jnp.float64)
    difference = (energy(values + step * direction) - energy(values - step * direction)) / (2.0 * step)
    assert abs(float(jnp.sum(gradient * direction)) - float(difference)) < 1e-6 * abs(float(difference))


def test_single_precision_solves_at_single_precision_accuracy():
    grid, conditions = SMALL, (FIXED, FIXED, FIXED)
    factorization = fast_diagonal_poisson(grid, conditions)
    exact = Field(
        jnp.asarray(np.random.default_rng(0).normal(size=grid.shape), dtype=jnp.float32), (CENTER,) * 3, grid
    )
    rhs = laplacian(exact, conditions)
    solution = factorization.solve(rhs)
    assert solution.dtype == jnp.float32
    assert float(jnp.max(jnp.abs(solution.data - exact.data))) < 1e-3


def test_factorization_refuses_inhomogeneous_boundary_data():
    with pytest.raises(ValueError, match="inhomogeneous boundary data"):
        fast_diagonal_poisson(SMALL, (BoundaryCondition(DIRICHLET, lower=1.0), FIXED, FIXED))


def test_factorization_validates_its_inputs():
    with pytest.raises(ValueError, match="one boundary condition per axis"):
        fast_diagonal_poisson(SMALL, (FIXED, FIXED))
    factorization = fast_diagonal_poisson(SMALL, (FIXED, FIXED, FIXED))
    other = Grid(uniform_faces(5, 0.0, 1.0), *SMALL.faces[1:])
    with pytest.raises(ValueError, match="share the factorized grid"):
        factorization.solve(_random_cells(other))
    with pytest.raises(ValueError, match="cell centred"):
        factorization.solve(Field(jnp.zeros(SMALL.face_shape(0)), (0.0, CENTER, CENTER), SMALL))


def test_factorization_refuses_an_operator_that_is_not_symmetric(monkeypatch):
    """A wall stencil that broke symmetry would invalidate the factorization."""
    import lmx.poisson as poisson

    def asymmetric(grid, axis, condition):
        operator = assemble_axis_laplacian(grid, axis, condition)
        operator[0, 1] += 1.0
        return operator

    monkeypatch.setattr(poisson, "assemble_axis_laplacian", asymmetric)
    with pytest.raises(ValueError, match="not symmetric under the cell widths"):
        poisson.fast_diagonal_poisson(SMALL, (FIXED, FIXED, FIXED))


@pytest.mark.parametrize(
    ("offset", "conditions", "name"),
    [
        ((0.0, CENTER, CENTER), (WRAPPED, WALL, WRAPPED), "periodic-face"),
        ((CENTER, 0.0, CENTER), (WRAPPED, WALL, WRAPPED), "walled-face"),
        ((CENTER, CENTER, CENTER), (WRAPPED, WALL, WRAPPED), "cell-centred"),
    ],
    ids=["periodic-face", "walled-face", "cell-centred"],
)
def test_the_helmholtz_factorization_inverts_its_operator(offset, conditions, name):
    """A staggered implicit viscous solve, checked against the stencil it factorizes."""
    from lmx.ops import staggered_laplacian
    from lmx.poisson import fast_diagonal_helmholtz, free_slice

    del name
    grid = Grid(uniform_faces(4, 0.0, 1.0), geometric_faces(8, -1.0, 1.0, 1.15), uniform_faces(4, -1.0, 1.0))
    coefficient = 0.05
    factorization = fast_diagonal_helmholtz(grid, offset, conditions, shift=1.0, coefficient=coefficient)
    generator = np.random.default_rng(0)
    rhs = Field(jnp.asarray(generator.normal(size=grid.offset_shape(offset))), offset, grid)
    solution = factorization.solve(rhs)
    applied = solution.data - coefficient * staggered_laplacian(solution, conditions).data
    free = tuple(free_slice(grid, axis, offset[axis], conditions[axis]) for axis in range(3))
    residual = np.max(np.abs(np.asarray(applied)[free] - np.asarray(rhs.data)[free]))
    assert residual < 1e-12


def test_the_helmholtz_solve_leaves_prescribed_entries_at_zero():
    """Wall faces are boundary data the caller owns, not unknowns to set."""
    from lmx.poisson import fast_diagonal_helmholtz

    grid = Grid(uniform_faces(4, 0.0, 1.0), uniform_faces(6, -1.0, 1.0), uniform_faces(4, -1.0, 1.0))
    offset, conditions = (CENTER, 0.0, CENTER), (WRAPPED, WALL, WRAPPED)
    factorization = fast_diagonal_helmholtz(grid, offset, conditions, shift=1.0, coefficient=0.1)
    rhs = Field(jnp.ones(grid.offset_shape(offset)), offset, grid)
    solution = np.asarray(factorization.solve(rhs).data)
    assert np.all(solution[:, 0, :] == 0.0)
    assert np.all(solution[:, -1, :] == 0.0)
    assert np.any(solution[:, 1:-1, :] != 0.0)


def test_the_helmholtz_factorization_validates_its_inputs():
    from lmx.poisson import fast_diagonal_helmholtz

    grid = Grid(*(uniform_faces(4, 0.0, 1.0) for _ in range(3)))
    with pytest.raises(ValueError, match="one boundary condition per axis"):
        fast_diagonal_helmholtz(grid, (CENTER,) * 3, (WALL, WALL))
    with pytest.raises(ValueError, match="inhomogeneous boundary data"):
        fast_diagonal_helmholtz(grid, (CENTER,) * 3, (BoundaryCondition(DIRICHLET, lower=1.0), WALL, WALL))
    factorization = fast_diagonal_helmholtz(grid, (CENTER,) * 3, (WALL, WALL, WALL))
    with pytest.raises(ValueError, match="match the factorized position"):
        factorization.solve(Field(jnp.zeros(grid.face_shape(0)), (0.0, CENTER, CENTER), grid))


def test_the_helmholtz_factorization_refuses_an_asymmetric_operator(monkeypatch):
    """The same tripwire as the scalar case, for the staggered assembly."""
    import lmx.poisson as poisson

    grid = Grid(*(uniform_faces(4, 0.0, 1.0) for _ in range(3)))

    original = poisson.assemble_staggered_axis_operator

    def asymmetric(grid_, axis, offset, condition):
        operator = original(grid_, axis, offset, condition)
        operator[0, 1] += 5.0
        return operator

    monkeypatch.setattr(poisson, "assemble_staggered_axis_operator", asymmetric)
    with pytest.raises(ValueError, match="not symmetric under its cell weights"):
        poisson.fast_diagonal_helmholtz(grid, (CENTER,) * 3, (WALL, WALL, WALL))

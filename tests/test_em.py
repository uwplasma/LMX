"""Charge-conservative face currents, Ohm's law and the face Lorentz force."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lmx.bc import DIRICHLET, NEUMANN, BoundaryCondition
from lmx.em import (
    cell_average,
    charge_residual,
    face_conductivity,
    face_current,
    face_electromotive_force,
    lorentz_force,
    thin_wall_flux,
    wall_insulated,
)
from lmx.grid import CENTER, FACE, Field, Grid, geometric_faces, tanh_faces, uniform_faces

pytestmark = pytest.mark.unit

WALL = BoundaryCondition(NEUMANN)
WALLS = (WALL, WALL, WALL)
STRETCHED = Grid(
    geometric_faces(6, 0.0, 3.0, 1.25),
    tanh_faces(8, -1.0, 1.0, 1.6),
    uniform_faces(5, -0.5, 0.5),
)
UNIFORM = Grid(uniform_faces(4, 0.0, 2.0), uniform_faces(5, -1.0, 1.0), uniform_faces(3, -0.5, 0.5))


def _cells(grid: Grid, function) -> Field:
    x, y, z = (np.asarray(values) for values in grid.centers)
    values = function(x[:, None, None], y[None, :, None], z[None, None, :])
    return Field(jnp.broadcast_to(jnp.asarray(values, dtype=jnp.float64), grid.shape), (CENTER,) * 3, grid)


def _faces(grid: Grid, axis: int, function) -> Field:
    coordinates = [np.asarray(values) for values in grid.centers]
    coordinates[axis] = np.asarray(grid.faces[axis])
    x, y, z = coordinates
    values = function(x[:, None, None], y[None, :, None], z[None, None, :])
    offset = tuple(FACE if position == axis else CENTER for position in range(3))
    return Field(
        jnp.broadcast_to(jnp.asarray(values, dtype=jnp.float64), grid.face_shape(axis)), offset, grid
    )


def _uniform_velocity(grid: Grid, components) -> tuple[Field, Field, Field]:
    return tuple(
        _faces(grid, axis, lambda x, y, z, value=components[axis]: value * jnp.ones_like(x * y * z))
        for axis in range(3)
    )


def _uniform_field(grid: Grid, components) -> tuple[Field, Field, Field]:
    return tuple(
        _cells(grid, lambda x, y, z, value=components[axis]: value * jnp.ones_like(x * y * z))
        for axis in range(3)
    )


def test_cell_average_of_a_linear_face_field_returns_the_cell_value():
    grid, axis = STRETCHED, 0
    faces = _faces(grid, axis, lambda x, y, z: 2.0 * x + 1.0)
    averaged = np.asarray(cell_average(faces, axis).data)
    expected = 2.0 * np.asarray(grid.centers[axis]) + 1.0
    assert np.max(np.abs(averaged - expected[:, None, None])) < 1e-13


def test_uniform_conductivity_survives_the_face_average():
    sigma = _cells(STRETCHED, lambda x, y, z: 3.5 * jnp.ones_like(x * y * z))
    for axis in range(3):
        faces = face_conductivity(sigma, axis, WALL)
        assert faces.shape == STRETCHED.face_shape(axis)
        assert np.max(np.abs(np.asarray(faces.data) - 3.5)) < 1e-13


def test_face_conductivity_across_a_jump_is_the_harmonic_mean():
    grid, axis = UNIFORM, 1
    low, high = 1.0e-3, 5.0
    sigma = _cells(grid, lambda x, y, z: jnp.where(y < 0.0, low, high) * jnp.ones_like(x * z))
    faces = np.asarray(face_conductivity(sigma, axis, WALL).data)
    centers = np.asarray(grid.centers[axis])
    interface = int(np.searchsorted(centers, 0.0))
    harmonic = 2.0 * low * high / (low + high)
    assert faces[:, interface, :] == pytest.approx(harmonic, rel=1e-12)
    # Away from the interface each side keeps its own value.
    assert faces[:, 1, :] == pytest.approx(low, rel=1e-12)
    assert faces[:, -2, :] == pytest.approx(high, rel=1e-12)
    # The harmonic mean is dominated by the poor conductor, unlike the arithmetic one.
    assert harmonic < 0.5 * (low + high)


def test_electromotive_force_matches_the_analytic_cross_product():
    grid = STRETCHED
    velocity_values, field_values = (0.3, -1.1, 2.0), (0.7, 0.2, -0.5)
    velocity = _uniform_velocity(grid, velocity_values)
    magnetic_field = _uniform_field(grid, field_values)
    expected = np.cross(np.asarray(velocity_values), np.asarray(field_values))
    for axis in range(3):
        emf = face_electromotive_force(velocity, magnetic_field, axis, WALLS)
        assert emf.shape == grid.face_shape(axis)
        assert np.max(np.abs(np.asarray(emf.data) - expected[axis])) < 1e-13


def test_ohms_law_reduces_to_each_of_its_two_terms():
    grid, axis, sigma_value = STRETCHED, 2, 2.5
    sigma = _cells(grid, lambda x, y, z: sigma_value * jnp.ones_like(x * y * z))
    conductivity = face_conductivity(sigma, axis, WALL)
    zero_emf = conductivity.replace_data(jnp.zeros_like(conductivity.data))
    slope = 1.7
    faces = np.asarray(grid.faces[axis])
    potential = _cells(grid, lambda x, y, z: slope * z)
    condition = BoundaryCondition(DIRICHLET, lower=slope * faces[0], upper=slope * faces[-1])

    conduction = face_current(potential, conductivity, zero_emf, axis, condition)
    assert np.max(np.abs(np.asarray(conduction.data) + sigma_value * slope)) < 1e-12

    flat = _cells(grid, lambda x, y, z: jnp.zeros_like(x * y * z))
    emf_value = 0.9
    emf = conductivity.replace_data(emf_value * jnp.ones_like(conductivity.data))
    motional = face_current(flat, conductivity, emf, axis, WALL)
    assert np.max(np.abs(np.asarray(motional.data) - sigma_value * emf_value)) < 1e-12


def test_uniform_current_in_a_uniform_field_gives_the_analytic_lorentz_force():
    grid = UNIFORM
    current_values, field_values = (0.5, -2.0, 1.25), (0.3, 0.8, -1.4)
    currents = _uniform_velocity(grid, current_values)
    magnetic_field = _uniform_field(grid, field_values)
    expected = np.cross(np.asarray(current_values), np.asarray(field_values))
    force = lorentz_force(currents, magnetic_field, WALLS)
    for axis in range(3):
        assert force[axis].shape == grid.shape
        assert np.max(np.abs(np.asarray(force[axis].data) - expected[axis])) < 1e-12


def test_lorentz_force_is_orthogonal_to_a_uniform_field():
    grid = STRETCHED
    field_values = (0.0, 1.0, 0.0)
    magnetic_field = _uniform_field(grid, field_values)
    currents = tuple(
        _faces(grid, axis, lambda x, y, z, axis=axis: jnp.sin(x + 2.0 * y - z + axis)) for axis in range(3)
    )
    force = lorentz_force(currents, magnetic_field, WALLS)
    projection = sum(np.asarray(force[axis].data) * field_values[axis] for axis in range(3))
    assert np.max(np.abs(projection)) < 1e-13


def test_a_divergence_free_current_leaves_no_charge_residual():
    grid = STRETCHED
    currents = (
        _faces(grid, 0, lambda x, y, z: 2.0 * x),
        _faces(grid, 1, lambda x, y, z: -3.0 * y),
        _faces(grid, 2, lambda x, y, z: 1.0 * z),
    )
    residual = np.asarray(charge_residual(currents).data)
    assert np.max(np.abs(residual)) < 1e-12


def test_insulated_conduction_current_conserves_charge_when_the_potential_solves_its_equation():
    """The potential that solves the discrete equation makes the face fluxes close."""
    from lmx.bc import PERIODIC
    from lmx.ops import laplacian
    from lmx.poisson import fast_diagonal_poisson

    grid = STRETCHED
    conditions = (BoundaryCondition(PERIODIC), WALL, WALL)
    sigma = _cells(grid, lambda x, y, z: jnp.ones_like(x * y * z))
    velocity = _uniform_velocity(grid, (0.0, 0.0, 1.3))
    magnetic_field = _uniform_field(grid, (0.0, 0.7, 0.0))
    conductivities = [face_conductivity(sigma, axis, conditions[axis]) for axis in range(3)]
    emfs = [face_electromotive_force(velocity, magnetic_field, axis, conditions) for axis in range(3)]

    # With uniform conductivity the potential equation is the plain Laplacian.
    source = charge_residual(
        tuple(
            conductivities[axis].replace_data(conductivities[axis].data * emfs[axis].data)
            for axis in range(3)
        )
    )
    potential = fast_diagonal_poisson(grid, conditions).solve(source)
    assert float(jnp.max(jnp.abs(laplacian(potential, conditions).data - source.data))) < 1e-11

    currents = tuple(
        face_current(potential, conductivities[axis], emfs[axis], axis, conditions[axis]) for axis in range(3)
    )
    residual = np.asarray(charge_residual(currents).data)
    scale = float(np.max(np.abs(np.asarray(source.data)))) or 1.0
    assert np.max(np.abs(residual)) < 1e-12 * scale


def test_coupling_is_differentiable_and_jits():
    grid = UNIFORM
    sigma = _cells(grid, lambda x, y, z: 1.0 + 0.1 * jnp.sin(x))
    magnetic_field = _uniform_field(grid, (0.0, 1.0, 0.0))

    def work(values):
        potential = Field(values, (CENTER,) * 3, grid)
        velocity = _uniform_velocity(grid, (1.0, 0.0, 0.0))
        currents, forces = [], None
        for axis in range(3):
            conductivity = face_conductivity(sigma, axis, WALL)
            emf = face_electromotive_force(velocity, magnetic_field, axis, WALLS)
            currents.append(face_current(potential, conductivity, emf, axis, WALL))
        forces = lorentz_force(tuple(currents), magnetic_field, WALLS)
        return jnp.sum(forces[0].data ** 2 + forces[2].data ** 2)

    values = jax.random.normal(jax.random.PRNGKey(3), grid.shape, dtype=jnp.float64)
    gradient = jax.jit(jax.grad(work))(values)
    step = 1.0e-6
    direction = jax.random.normal(jax.random.PRNGKey(5), grid.shape, dtype=jnp.float64)
    difference = (work(values + step * direction) - work(values - step * direction)) / (2.0 * step)
    assert abs(float(jnp.sum(gradient * direction)) - float(difference)) < 1e-6 * abs(float(difference))


def test_coupling_preserves_single_precision():
    grid = UNIFORM
    sigma = Field(jnp.ones(grid.shape, dtype=jnp.float32), (CENTER,) * 3, grid)
    conductivity = face_conductivity(sigma, 0, WALL)
    assert conductivity.dtype == jnp.float32


def test_electric_helpers_validate_their_inputs():
    grid = UNIFORM
    cell = Field(jnp.zeros(grid.shape), (CENTER,) * 3, grid)
    face = Field(jnp.zeros(grid.face_shape(0)), (FACE, CENTER, CENTER), grid)
    with pytest.raises(ValueError, match="expected a field on faces"):
        cell_average(cell, 0)
    with pytest.raises(ValueError, match="conductivity must be cell centred"):
        face_conductivity(face, 0, WALL)
    with pytest.raises(ValueError, match="does not match grid"):
        face_conductivity(Field(jnp.zeros((2, 2, 2)), (CENTER,) * 3, grid), 0, WALL)
    with pytest.raises(ValueError, match="conductivity must live on the same faces"):
        face_current(cell, cell, face, 0, WALL)
    with pytest.raises(ValueError, match="electromotive force must live on the same faces"):
        face_current(cell, face, cell, 0, WALL)


def test_an_insulating_wall_carries_no_current():
    """`J.n = 0` is the wall condition, so the closure has to zero the wall faces."""
    grid = UNIFORM
    potential = _cells(grid, lambda x, y, z: x + y)
    conductivity = face_conductivity(_cells(grid, lambda x, y, z: 1.0 + 0.0 * x), 0, WALL)
    electromotive = _faces(grid, 0, lambda x, y, z: 1.0 + y)
    current = wall_insulated(face_current(potential, conductivity, electromotive, 0, WALL), 0, WALL)
    assert float(jnp.max(jnp.abs(current.data[0]))) == 0.0
    assert float(jnp.max(jnp.abs(current.data[-1]))) == 0.0
    assert float(jnp.max(jnp.abs(current.data[1:-1]))) > 0.0


def test_only_an_insulating_wall_is_closed_off():
    """A periodic axis has no wall, and a prescribed potential is a conducting one."""
    grid = UNIFORM
    flux = _faces(grid, 1, lambda x, y, z: 1.0 + 0.0 * x)
    assert wall_insulated(flux, 1, BoundaryCondition("periodic")) is flux
    assert wall_insulated(flux, 1, BoundaryCondition(DIRICHLET)) is flux
    insulated = wall_insulated(flux, 1, WALL)
    assert float(jnp.max(jnp.abs(insulated.data[:, 0]))) == 0.0
    assert float(jnp.max(jnp.abs(insulated.data[:, 1:-1]))) == 1.0


def test_a_thin_wall_conducts_only_what_the_tangential_potential_drives():
    """The wall current is the surface Laplacian of the potential on the wall layer."""
    grid = UNIFORM
    conditions = (WALL, WALL, WALL)
    uniform = _cells(grid, lambda x, y, z: 3.0 + 0.0 * x)
    assert float(jnp.max(jnp.abs(thin_wall_flux(uniform, 1, WALL, 0.05, conditions).data))) < 1e-14

    curved = _cells(grid, lambda x, y, z: z**2)
    doubled = thin_wall_flux(curved, 1, WALL, 0.10, conditions)
    driven = thin_wall_flux(curved, 1, WALL, 0.05, conditions)
    assert float(jnp.max(jnp.abs(driven.data[:, 0]))) > 0.0
    # The stored value points along the axis, so the two walls carry opposite signs.
    assert np.allclose(np.asarray(driven.data[:, 0]), -np.asarray(driven.data[:, -1]))
    assert float(jnp.max(jnp.abs(driven.data[:, 1:-1]))) == 0.0
    # A wall twice as conductive carries twice the current.
    assert np.allclose(np.asarray(doubled.data), 2.0 * np.asarray(driven.data))


def test_no_wall_conducts_without_a_conductance_or_without_a_wall():
    grid = UNIFORM
    conditions = (WALL, WALL, WALL)
    curved = _cells(grid, lambda x, y, z: z**2)
    assert float(jnp.max(jnp.abs(thin_wall_flux(curved, 1, WALL, 0.0, conditions).data))) == 0.0
    periodic = BoundaryCondition("periodic")
    assert float(jnp.max(jnp.abs(thin_wall_flux(curved, 1, periodic, 0.05, conditions).data))) == 0.0

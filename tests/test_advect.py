"""Conservative momentum transport: conservation, order on a stretched mesh, boundedness."""

import jax.numpy as jnp
import numpy as np
import pytest

from lmx.advect import advective_step_limit, momentum_advection
from lmx.bc import DIRICHLET, PERIODIC, BoundaryCondition
from lmx.core3d import ChannelProblem, step, velocity_offset, zero_velocity
from lmx.grid import Field, Grid, uniform_faces

pytestmark = pytest.mark.unit

PERIODIC_AXIS = BoundaryCondition(PERIODIC)
NO_SLIP = BoundaryCondition(DIRICHLET)
BOX = (PERIODIC_AXIS,) * 3


def _periodic_faces(count: int, amplitude: float = 0.0) -> np.ndarray:
    """Faces on ``[0, 2 pi]``; a nonzero amplitude stretches them smoothly and periodically."""
    fraction = np.linspace(0.0, 1.0, count + 1)
    return 2.0 * np.pi * (fraction + amplitude * np.sin(2.0 * np.pi * fraction) / (2.0 * np.pi))


def _coordinates(grid: Grid, component: int) -> tuple[np.ndarray, ...]:
    axes = [grid.faces[axis] if axis == component else grid.centers[axis] for axis in range(3)]
    return np.meshgrid(*axes, indexing="ij")


def _taylor_green(grid: Grid, component: int) -> tuple[Field, np.ndarray]:
    """A divergence-free field and the exact ``div(u u)`` of that field.

    With ``u = (sin x cos y cos z, -cos x sin y cos z, 0)`` the divergence
    vanishes and the flux divergence reduces to ``(sin x cos x cos^2 z,
    sin y cos y cos^2 z, 0)``, derived by hand rather than from any operator here.
    """
    x, y, z = _coordinates(grid, component)
    values = (
        np.sin(x) * np.cos(y) * np.cos(z),
        -np.cos(x) * np.sin(y) * np.cos(z),
        np.zeros_like(x),
    )
    exact = (
        np.sin(x) * np.cos(x) * np.cos(z) ** 2,
        np.sin(y) * np.cos(y) * np.cos(z) ** 2,
        np.zeros_like(x),
    )
    field = Field(jnp.asarray(values[component]), velocity_offset(component), grid)
    return field, exact[component]


def _uniform(grid: Grid, values: tuple[float, float, float]) -> tuple[Field, Field, Field]:
    return tuple(
        Field(
            jnp.full(grid.offset_shape(velocity_offset(component)), value), velocity_offset(component), grid
        )
        for component, value in enumerate(values)
    )


def _error(count: int, amplitude: float) -> float:
    faces = _periodic_faces(count, amplitude)
    grid = Grid(faces, faces, faces)
    fields, exact = zip(*(_taylor_green(grid, component) for component in range(3)))
    transported = momentum_advection(fields, BOX, limited=False)
    return max(
        float(np.sqrt(np.mean((np.asarray(transported[component].data) - exact[component]) ** 2)))
        for component in range(2)
    )


@pytest.mark.parametrize("limited", [False, True])
def test_a_uniform_flow_is_not_transported(limited):
    """Galilean invariance: a constant velocity has no flux divergence, on any mesh."""
    faces = _periodic_faces(8, 0.4)
    grid = Grid(faces, faces, faces)
    transported = momentum_advection(_uniform(grid, (1.3, -0.7, 0.2)), BOX, limited=limited)
    for field in transported:
        assert float(jnp.max(jnp.abs(field.data))) < 1e-13


@pytest.mark.parametrize("limited", [False, True])
def test_the_flux_form_conserves_momentum(limited):
    """Telescoping fluxes: the transported momentum of a periodic box sums to zero."""
    faces = _periodic_faces(10, 0.3)
    grid = Grid(faces, faces, faces)
    fields = tuple(_taylor_green(grid, component)[0] for component in range(3))
    transported = momentum_advection(fields, BOX, limited=limited)
    for component, field in enumerate(transported):
        interior = np.asarray(field.data)[(slice(None),) * component + (slice(None, -1),)]
        scale = float(np.max(np.abs(interior))) + 1.0
        assert abs(float(np.sum(interior))) < 1e-11 * scale * interior.size


@pytest.mark.parametrize("amplitude", [0.0, 0.5])
def test_transport_is_second_order_including_on_a_stretched_mesh(amplitude):
    coarse, medium, fine = (_error(count, amplitude) for count in (16, 32, 64))
    orders = (np.log2(coarse / medium), np.log2(medium / fine))
    assert min(orders) > 1.9, orders


def _carried_in_y(grid: Grid, profile: np.ndarray) -> tuple[Field, Field, Field]:
    """Transport ``profile(y)`` as ``u_x`` on a uniform ``u_y = 1``.

    The state is divergence free, so this is linear scalar advection along ``y``
    and the limiter has to satisfy the usual bound on it.
    """
    return (
        Field(jnp.asarray(profile), velocity_offset(0), grid),
        Field(jnp.ones(grid.offset_shape(velocity_offset(1))), velocity_offset(1), grid),
        Field(jnp.zeros(grid.offset_shape(velocity_offset(2))), velocity_offset(2), grid),
    )


def test_the_limiter_reduces_to_the_central_flux_on_a_uniform_gradient():
    """Where successive gradients agree the van Leer weight is one, so nothing is clipped."""
    faces = _periodic_faces(16, 0.0)
    grid = Grid(faces, faces, faces)
    _, y, _ = _coordinates(grid, 0)
    fields = _carried_in_y(grid, 0.3 * y)
    central = np.asarray(momentum_advection(fields, BOX, limited=False)[0].data)
    limited = np.asarray(momentum_advection(fields, BOX, limited=True)[0].data)
    # The ramp is linear everywhere except across the periodic wrap; compare away from it.
    interior = (slice(None), slice(2, -2), slice(None))
    assert np.max(np.abs(central[interior] - limited[interior])) < 1e-14


def test_the_limiter_does_not_overshoot_a_step():
    """A discontinuity stays inside its own bounds; the central flux does not."""
    faces = _periodic_faces(32, 0.0)
    grid = Grid(faces, faces, faces)
    _, y, _ = _coordinates(grid, 0)
    profile = np.where((y > 2.0) & (y < 4.0), 1.0, 0.0)
    fields = _carried_in_y(grid, profile)
    dt = 0.5 * float(np.min(grid.widths[1]))
    stepped = {
        name: profile - dt * np.asarray(momentum_advection(fields, BOX, limited=name)[0].data)
        for name in (False, True)
    }
    assert np.max(stepped[True]) <= 1.0 + 1e-12
    assert np.min(stepped[True]) >= -1e-12
    assert np.max(stepped[False]) > 1.0 + 1e-3


def test_the_step_limit_reports_the_convective_bound():
    faces = _periodic_faces(8, 0.0)
    grid = Grid(faces, faces, faces)
    limit = advective_step_limit(_uniform(grid, (2.0, 1.0, 0.0)))
    widths = [float(np.min(grid.widths[axis])) for axis in range(3)]
    assert float(limit) == pytest.approx(1.0 / (2.0 / widths[0] + 1.0 / widths[1]))


def test_advection_needs_one_condition_per_axis():
    faces = _periodic_faces(4, 0.0)
    grid = Grid(faces, faces, faces)
    with pytest.raises(ValueError, match="one boundary condition per axis"):
        momentum_advection(_uniform(grid, (1.0, 0.0, 0.0)), (PERIODIC_AXIS, PERIODIC_AXIS))


def _channel(advection: str) -> ChannelProblem:
    grid = Grid(uniform_faces(4, 0.0, 1.0), uniform_faces(8, -1.0, 1.0), uniform_faces(8, -1.0, 1.0))
    return ChannelProblem(
        grid=grid,
        conditions=(PERIODIC_AXIS, NO_SLIP, NO_SLIP),
        forcing=(1.0, 0.0, 0.0),
        dt=2.0e-3,
        advection=advection,
    )


def test_the_channel_rejects_an_unknown_advection_choice():
    with pytest.raises(ValueError, match="advection must be one of"):
        _channel("upwind")


@pytest.mark.parametrize("advection", ["central", "limited"])
def test_a_step_with_transport_stays_divergence_free(advection):
    """Transport enters the predictor, so the projection still has to clean up after it."""
    from lmx.ops import divergence

    problem = _channel(advection)
    velocity = zero_velocity(problem)
    for _ in range(4):
        velocity, _, _ = step(velocity, problem)
    assert float(jnp.max(jnp.abs(divergence(velocity).data))) < 1e-11
    assert np.all(np.isfinite(np.asarray(velocity[0].data)))


def test_transport_off_reproduces_the_stokes_step():
    problem = _channel("off")
    stokes, _, _ = step(zero_velocity(problem), problem)
    transported, _, _ = step(zero_velocity(problem), _channel("central"))
    # From rest the first step has nothing to transport, so the two agree exactly.
    for left, right in zip(stokes, transported, strict=True):
        assert float(jnp.max(jnp.abs(left.data - right.data))) == 0.0

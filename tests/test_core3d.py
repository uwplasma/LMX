"""The projection step: incompressibility, the damping treatment, and duct flow.

The strongest check here is the last one. A periodic duct driven by a constant
body force reaches the fully developed state, which LMX already solves by an
entirely different route in :func:`lmx.solve_fully_developed_fields`: a
two-dimensional cross-section solve on `lmx.mesh`, with its own operators and its
own linear algebra. Agreement between the two exercises the new staggered core,
its electric coupling and its projection at once, against code that shares none
of them.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lmx.bc import NEUMANN, PERIODIC, BoundaryCondition
from lmx.core3d import (
    ChannelProblem,
    enforce_face_constraints,
    project,
    step,
    velocity_condition,
    velocity_offset,
    zero_velocity,
)
from lmx.grid import CENTER, Field, Grid, uniform_faces
from lmx.ops import divergence

pytestmark = pytest.mark.unit

PERIODIC_X = BoundaryCondition(PERIODIC)
WALL = BoundaryCondition(NEUMANN)


def _duct(ny: int = 12, nz: int = 12, nx: int = 4) -> Grid:
    return Grid(uniform_faces(nx, 0.0, 1.0), uniform_faces(ny, -1.0, 1.0), uniform_faces(nz, -1.0, 1.0))


def _problem(grid: Grid, **overrides) -> ChannelProblem:
    settings = dict(
        grid=grid,
        conditions=(PERIODIC_X, WALL, WALL),
        density=1.0,
        viscosity=1.0,
        conductivity=1.0,
        magnetic_field=(0.0, 0.0, 0.0),
        forcing=(1.0, 0.0, 0.0),
        dt=2.0e-3,
    )
    settings.update(overrides)
    return ChannelProblem(**settings)


def _advance(problem: ChannelProblem, steps: int) -> tuple[Field, Field, Field]:
    factorization = problem.factorization()
    velocity = zero_velocity(problem)
    for _ in range(steps):
        velocity, _, _ = step(velocity, problem, factorization)
    return velocity


def test_velocity_positions_and_conditions_follow_the_layout():
    assert velocity_offset(0) == (0.0, CENTER, CENTER)
    assert velocity_offset(2) == (CENTER, CENTER, 0.0)
    conditions = (PERIODIC_X, WALL, WALL)
    assert velocity_condition(conditions, 0).is_periodic
    assert velocity_condition(conditions, 1).kind == "dirichlet"


def test_face_constraints_close_the_boundary_flux():
    grid = _duct(ny=6, nz=6)
    problem = _problem(grid)
    key = jax.random.PRNGKey(0)
    velocity = tuple(
        Field(
            jax.random.normal(fold, grid.offset_shape(velocity_offset(component)), dtype=jnp.float64),
            velocity_offset(component),
            grid,
        )
        for component, fold in enumerate(jax.random.split(key, 3))
    )
    constrained = enforce_face_constraints(velocity, problem)
    # The periodic axis must agree across its duplicated face, and the walls
    # must be impermeable; together these make the net boundary flux vanish.
    assert np.allclose(np.asarray(constrained[0].data)[0], np.asarray(constrained[0].data)[-1], atol=0.0)
    assert np.all(np.asarray(constrained[1].data)[:, 0, :] == 0.0)
    assert np.all(np.asarray(constrained[2].data)[:, :, -1] == 0.0)
    volumes = grid.cell_volumes()
    net = float(np.sum(volumes * np.asarray(divergence(constrained).data)))
    assert abs(net) < 1e-12


def test_projection_removes_the_divergence_and_is_idempotent():
    grid = _duct(ny=8, nz=8)
    problem = _problem(grid)
    key = jax.random.PRNGKey(1)
    velocity = tuple(
        Field(
            jax.random.normal(fold, grid.offset_shape(velocity_offset(component)), dtype=jnp.float64),
            velocity_offset(component),
            grid,
        )
        for component, fold in enumerate(jax.random.split(key, 3))
    )
    projected, _ = project(velocity, problem)
    scale = float(np.max(np.abs(np.asarray(divergence(enforce_face_constraints(velocity, problem)).data))))
    assert np.max(np.abs(np.asarray(divergence(projected).data))) < 1e-11 * scale

    twice, _ = project(projected, problem)
    for first, second in zip(projected, twice, strict=True):
        assert np.max(np.abs(np.asarray(first.data) - np.asarray(second.data))) < 1e-11


def test_the_step_keeps_the_velocity_divergence_free():
    problem = _problem(_duct(ny=8, nz=8))
    residual = np.asarray(divergence(_advance(problem, 20)).data)
    assert np.max(np.abs(residual)) < 1e-11


def test_a_driven_duct_develops_a_symmetric_forward_profile():
    problem = _problem(_duct(ny=10, nz=10))
    axial = np.asarray(_advance(problem, 40)[0].data)
    assert np.all(axial >= -1e-12)
    assert np.max(np.abs(axial[:, ::-1, :] - axial)) < 1e-11
    assert np.max(np.abs(axial[:, :, ::-1] - axial)) < 1e-11
    # The profile peaks in the middle and vanishes towards the walls.
    middle = axial[0, axial.shape[1] // 2, axial.shape[2] // 2]
    assert middle > axial[0, 0, axial.shape[2] // 2]


def test_damping_rates_follow_the_field_direction():
    problem = _problem(_duct(ny=4, nz=4), magnetic_field=(0.0, 3.0, 0.0), conductivity=2.0, density=4.0)
    rates = problem.damping_rates
    # A field along y damps the two transverse components and leaves its own free.
    assert rates[1] == pytest.approx(0.0)
    assert rates[0] == pytest.approx(2.0 * 9.0 / 4.0)
    assert rates[2] == pytest.approx(2.0 * 9.0 / 4.0)


def test_implicit_damping_stays_stable_past_the_explicit_limit():
    """The damping rate here is far beyond what an explicit step could carry."""
    problem = _problem(_duct(ny=8, nz=8), magnetic_field=(0.0, 30.0, 0.0), dt=2.0e-3)
    assert problem.dt * problem.damping_rates[0] > 1.0, "the test needs a step past the explicit limit"
    axial = np.asarray(_advance(problem, 40)[0].data)
    assert np.all(np.isfinite(axial))
    assert np.max(np.abs(axial)) < 10.0


def test_a_transverse_field_slows_the_flow_it_is_driven_through():
    """Magnetic drag: the same drive reaches a smaller throughput at higher Ha."""
    grid = _duct(ny=10, nz=10)
    throughputs = []
    for strength in (0.0, 3.0, 10.0):
        problem = _problem(grid, magnetic_field=(0.0, strength, 0.0), dt=2.0e-3)
        throughputs.append(float(np.mean(np.asarray(_advance(problem, 60)[0].data))))
    assert throughputs[0] > throughputs[1] > throughputs[2] > 0.0


def test_the_diffusive_step_limit_is_reported_and_binding():
    """Diffusion is explicit, so the limit is real; the field imposes none."""
    problem = _problem(_duct(ny=12, nz=12, nx=3), viscosity=1.0)
    limit = problem.diffusive_step_limit
    assert limit == pytest.approx(1.0 / (2.0 * (9.0 + 36.0 + 36.0)), rel=1e-12)
    # Halving the viscosity doubles the allowed step; the magnetic field, being
    # treated implicitly, does not enter at all.
    assert _problem(_duct(ny=12, nz=12, nx=3), viscosity=0.5).diffusive_step_limit == pytest.approx(
        2.0 * limit, rel=1e-12
    )
    strong = _problem(_duct(ny=12, nz=12, nx=3), magnetic_field=(0.0, 50.0, 0.0))
    assert strong.diffusive_step_limit == pytest.approx(limit, rel=1e-12)


def _channel(cells: int) -> Grid:
    """A plane channel: periodic along x and z, walls at y = +/- 1."""
    return Grid(uniform_faces(2, 0.0, 1.0), uniform_faces(cells, -1.0, 1.0), uniform_faces(2, 0.0, 1.0))


def _poiseuille_error(cells: int, *, drive: float = 1.0, viscosity: float = 1.0) -> float:
    """Return the relative L2 error against the analytic plane-channel profile."""
    grid = _channel(cells)
    base = ChannelProblem(
        grid=grid,
        conditions=(PERIODIC_X, WALL, PERIODIC_X),
        viscosity=viscosity,
        forcing=(drive, 0.0, 0.0),
    )
    problem = ChannelProblem(
        grid=grid,
        conditions=(PERIODIC_X, WALL, PERIODIC_X),
        viscosity=viscosity,
        forcing=(drive, 0.0, 0.0),
        dt=0.4 * base.diffusive_step_limit,
    )
    steps = int(round(4.0 / problem.dt))
    computed = np.asarray(_advance(problem, steps)[0].data)[0, :, 0]
    y = np.asarray(grid.centers[1])
    exact = drive / (2.0 * viscosity) * (1.0 - y**2)
    return float(np.linalg.norm(computed - exact) / np.linalg.norm(exact))


def test_plane_channel_matches_the_analytic_parabola():
    """Steady Stokes flow between plates has the exact profile f(1-y^2)/(2 nu).

    Sixteen cells across the channel give a relative L2 error near 5e-3, which
    is what a second-order scheme gives on a mesh this coarse; the rate itself
    is checked by the refinement test below.
    """
    assert _poiseuille_error(16) < 8.0e-3


def test_the_plane_channel_converges_at_second_order():
    """Refinement must remove the error at the rate the discretization claims."""
    coarse, fine = _poiseuille_error(8), _poiseuille_error(16)
    assert fine < coarse
    order = np.log2(coarse / fine)
    assert 1.7 < order < 2.3, (coarse, fine, order)


def test_a_transverse_field_reduces_the_channel_throughput():
    """The same drive moves less fluid once the field is switched on."""
    grid = _channel(16)
    throughputs = []
    for strength in (0.0, 4.0):
        base = ChannelProblem(
            grid=grid,
            conditions=(PERIODIC_X, WALL, PERIODIC_X),
            magnetic_field=(0.0, strength, 0.0),
            forcing=(1.0, 0.0, 0.0),
        )
        problem = ChannelProblem(
            grid=grid,
            conditions=(PERIODIC_X, WALL, PERIODIC_X),
            magnetic_field=(0.0, strength, 0.0),
            forcing=(1.0, 0.0, 0.0),
            dt=0.4 * base.diffusive_step_limit,
        )
        steps = int(round(4.0 / problem.dt))
        throughputs.append(float(np.mean(np.asarray(_advance(problem, steps)[0].data))))
    assert throughputs[0] > throughputs[1] > 0.0


def test_channel_problem_validates_its_inputs():
    grid = _duct(ny=4, nz=4)
    with pytest.raises(ValueError, match="one boundary condition per axis"):
        ChannelProblem(grid=grid, conditions=(WALL, WALL))
    with pytest.raises(ValueError, match="viscosity must be positive"):
        ChannelProblem(grid=grid, conditions=(WALL,) * 3, viscosity=0.0)
    with pytest.raises(ValueError, match="conductivity must not be negative"):
        ChannelProblem(grid=grid, conditions=(WALL,) * 3, conductivity=-1.0)


def test_the_step_differentiates_and_jits():
    grid = _duct(ny=6, nz=6, nx=2)
    base = _problem(grid, dt=5.0e-3)
    factorization = base.factorization()

    def kinetic(drive):
        problem = _problem(grid, forcing=(drive, 0.0, 0.0), dt=5.0e-3)
        velocity = zero_velocity(problem)
        for _ in range(3):
            velocity, _, _ = step(velocity, problem, factorization)
        return jnp.sum(velocity[0].data ** 2)

    gradient = float(jax.jit(jax.grad(kinetic))(1.0))
    size = 1.0e-5
    difference = (kinetic(1.0 + size) - kinetic(1.0 - size)) / (2.0 * size)
    assert gradient == pytest.approx(float(difference), rel=1e-6)

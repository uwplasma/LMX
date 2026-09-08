"""The projection step: incompressibility, the damping treatment, and duct flow.

The strongest check here is the last one. A periodic duct driven by a constant
body force reaches the fully developed state, which LMX already solves by an
entirely different route in :func:`lmx.solve_fully_developed_fields`: a
two-dimensional cross-section solve on `lmx.mesh`, with its own operators and its
own linear algebra. Agreement between the two exercises the new staggered core,
its electric coupling and its projection at once, against code that shares none
of them.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lmx.bc import NEUMANN, PERIODIC, BoundaryCondition
from lmx.cases import make_hartmann_case, solve_fully_developed_fields
from lmx.core3d import (
    ChannelProblem,
    duct_problem,
    enforce_face_constraints,
    project,
    step,
    velocity_condition,
    velocity_offset,
    zero_velocity,
)
from lmx.grid import CENTER, Field, Grid, uniform_faces, wall_resolving_faces
from lmx.ops import divergence
from lmx.timeloop import advance
from validation.shercliff import flow_rate

# Physics validation rather than unit checks: the channel cases integrate to a
# steady state, which the tier system runs in the regression lane.
pytestmark = pytest.mark.numerical

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
    grid = _channel(12)
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
        steps = int(round(2.0 / problem.dt))
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


def _implicit_channel_error(cells: int, *, step_multiple: float) -> tuple[float, float]:
    """Integrate the plane channel with implicit viscosity at a chosen step size.

    ``step_multiple`` is the step as a multiple of the explicit stability limit,
    so a value above one is a step no explicit scheme could take.
    """
    grid = _channel(cells)
    conditions = (PERIODIC_X, WALL, PERIODIC_X)
    base = ChannelProblem(grid=grid, conditions=conditions, forcing=(1.0, 0.0, 0.0))
    problem = ChannelProblem(
        grid=grid,
        conditions=conditions,
        forcing=(1.0, 0.0, 0.0),
        dt=step_multiple * base.diffusive_step_limit,
    )
    factorization, viscous = problem.factorization(), problem.viscous_factorizations()
    velocity = zero_velocity(problem)
    for _ in range(int(round(4.0 / problem.dt))):
        velocity, _, _ = step(velocity, problem, factorization, viscous)
    computed = np.asarray(velocity[0].data)[0, :, 0]
    y = np.asarray(grid.centers[1])
    exact = 0.5 * (1.0 - y**2)
    return float(np.linalg.norm(computed - exact) / np.linalg.norm(exact)), problem.dt


def test_implicit_viscosity_is_stable_far_past_the_explicit_limit():
    """Twenty times the explicit limit: an explicit step would diverge here."""
    error, used = _implicit_channel_error(16, step_multiple=20.0)
    reference = ChannelProblem(
        grid=_channel(16), conditions=(PERIODIC_X, WALL, PERIODIC_X), forcing=(1.0, 0.0, 0.0)
    )
    assert used > 10.0 * reference.diffusive_step_limit
    assert np.isfinite(error) and error < 1.0e-2


def test_implicit_and_explicit_viscosity_reach_the_same_steady_state():
    """The treatment changes the path, not the state the path ends at."""
    explicit = _poiseuille_error(16)
    implicit, _ = _implicit_channel_error(16, step_multiple=0.4)
    assert implicit == pytest.approx(explicit, rel=2.0e-2)


def test_implicit_viscosity_keeps_second_order_convergence():
    coarse, _ = _implicit_channel_error(8, step_multiple=5.0)
    fine, _ = _implicit_channel_error(16, step_multiple=5.0)
    order = np.log2(coarse / fine)
    assert 1.7 < order < 2.3, (coarse, fine, order)


# --- Reconciliation with the production solver and an independent reference ---
#
# `lmx.solve_fully_developed_fields` solves the same duct on a two-dimensional
# cross-section mesh with its own operators. Its conventions had to be matched
# before the two could be compared at all: `GeometrySpec.width` and `height` are
# the *full* transverse extents, so a `width=height=2` case is the grid
# `[-1, 1]^2` used here; `CaseSpec.forcing` is the axial pressure gradient
# `-dp/dx` and enters the momentum equation divided by the density, exactly as
# `ChannelProblem.forcing` does; and both read `viscosity` as the kinematic one.
#
# `validation.shercliff` closes the loop from outside the package: a Chebyshev
# collocation solve of the governing system, converged to eight digits, sharing
# no operator with either route.


def _duct_mean_velocity(cells: int, hartmann: float, *, resolve_layers: bool = False) -> float:
    """Return the volume-averaged axial velocity of a steady insulating duct."""
    if resolve_layers:
        transverse = wall_resolving_faces(
            cells, -1.0, 1.0, layer_thickness=1.0 / hartmann, cells_in_layer=6, max_ratio=1.35
        )
        spanwise = wall_resolving_faces(
            cells, -1.0, 1.0, layer_thickness=1.0 / np.sqrt(hartmann), cells_in_layer=6, max_ratio=1.35
        )
    else:
        transverse = spanwise = uniform_faces(cells, -1.0, 1.0)
    grid = Grid(uniform_faces(1, 0.0, 1.0), transverse, spanwise)
    problem = ChannelProblem(
        grid=grid,
        conditions=(PERIODIC_X, WALL, WALL),
        conductivity=1.0 if hartmann else 0.0,
        magnetic_field=(0.0, hartmann, 0.0),
        forcing=(1.0, 0.0, 0.0),
        dt=0.02,
    )
    velocity = advance(
        problem,
        400,
        factorization=problem.factorization(),
        viscous=problem.viscous_factorizations(),
    ).velocity
    volumes = np.asarray(grid.cell_volumes())[0]
    return float((np.asarray(velocity[0].data)[0] * volumes).sum() / volumes.sum())


def _production_duct(cells: int, hartmann: float) -> np.ndarray:
    """Solve the same duct through the production fully developed route."""
    case = make_hartmann_case(ha=hartmann, width=2.0, height=2.0, ny=cells, nz=cells)
    uniform = dataclasses.replace(case.geometry, target_ha=None)
    return np.asarray(solve_fully_developed_fields(dataclasses.replace(case, geometry=uniform))[0])


def test_the_hydrodynamic_duct_reconciles_with_the_production_solver():
    """Without a field the two routes agree to well inside their shared truncation error."""
    cells = 32
    production = _production_duct(cells, 0.0)
    grid = Grid(uniform_faces(1, 0.0, 1.0), uniform_faces(cells, -1.0, 1.0), uniform_faces(cells, -1.0, 1.0))
    problem = ChannelProblem(
        grid=grid, conditions=(PERIODIC_X, WALL, WALL), conductivity=0.0, forcing=(1.0, 0.0, 0.0), dt=0.02
    )
    velocity = advance(
        problem, 400, factorization=problem.factorization(), viscous=problem.viscous_factorizations()
    ).velocity
    new = np.asarray(velocity[0].data)[0]
    assert np.linalg.norm(new - production) / np.linalg.norm(production) < 5e-3
    # Both sit within their own discretisation error of the independent reference.
    exact = flow_rate(0.0, 40)
    for mean in (float(new.mean()), float(production.mean())):
        assert abs(mean - exact) / exact < 5e-3


def test_the_duct_converges_to_the_spectral_reference_with_a_field():
    """Second order in the mean velocity, against a reference that shares no code."""
    exact = flow_rate(5.0, 40)
    coarse = abs(_duct_mean_velocity(16, 5.0) - exact) / exact
    fine = abs(_duct_mean_velocity(32, 5.0) - exact) / exact
    assert fine < 0.02
    assert np.log2(coarse / fine) > 1.8


def test_a_wall_resolving_mesh_reaches_the_reference_flow_rate_at_hartmann_20():
    """The layers carry the physics, so the mesh has to resolve them rather than be fine."""
    exact = flow_rate(20.0, 40)
    resolved = _duct_mean_velocity(32, 20.0, resolve_layers=True)
    assert abs(resolved - exact) / exact < 0.02
    # The same cell count spread uniformly cannot resolve the a/Ha layer.
    assert abs(_duct_mean_velocity(32, 20.0) - exact) / exact > 0.1


def test_the_duct_helper_resolves_the_layers_it_names():
    """The mesh follows the physics: a/Ha against the field, a/sqrt(Ha) across it."""
    problem = duct_problem(hartmann=100.0, cells=40)
    transverse, spanwise = (np.diff(problem.grid.faces[axis]) for axis in (1, 2))
    assert transverse.min() < 1.0 / 100.0
    assert spanwise.min() < 1.0 / np.sqrt(100.0)
    # The gentlest stretching that spans the duct, not the finest one available.
    assert transverse.max() / transverse.min() < spanwise.max() / spanwise.min() * 1.0e3
    assert problem.wall_conductance == (0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="hartmann must not be negative"):
        duct_problem(hartmann=-1.0)


def test_the_public_solve_reaches_the_new_core():
    """`lmx.solve` dispatches a ChannelProblem to the steady Newton-Krylov path."""
    import lmx

    problem = lmx.duct_problem(hartmann=5.0, cells=16)
    solution = lmx.solve(problem)
    assert float(np.mean(np.asarray(solution.velocity[0].data))) > 0.0
    assert float(solution.residual_norm) < 1e-8

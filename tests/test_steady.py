"""The steady solve: it is the fixed point of the step, and its gradient is exact."""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lmx.bc import NEUMANN, PERIODIC, BoundaryCondition
from lmx.core3d import ChannelProblem, duct_problem, step, zero_velocity
from lmx.grid import Grid, uniform_faces, wall_resolving_faces
from lmx.steady import solve_steady_state, steady_residual
from validation.shercliff import flow_rate

pytestmark = pytest.mark.numerical

PERIODIC_X = BoundaryCondition(PERIODIC)
WALL = BoundaryCondition(NEUMANN)

# Mean velocity of a unit-forced insulating square duct, from validation.shercliff at the
# resolution of 64 points per direction, or 96 for Ha 1000, whose layers are thinner. Held as
# constants so a solver test costs a solve and not a dense spectral factorization;
# `test_the_reference_values_are_what_the_spectral_solve_returns` regenerates them.
REFERENCE_FLOW_RATE = {
    0.0: 0.140577010,
    5.0: 0.098192820,
    20.0: 0.038321780,
    100.0: 0.009054397,
    300.0: 0.003160369,
    1000.0: 0.000977900,
}


def _duct(
    cells: int,
    hartmann: float,
    *,
    ratio: float | None = None,
    dt: float = 1.0,
    conductance: float = 0.0,
    conductivity: float = 1.0,
) -> ChannelProblem:
    """A square duct; ``ratio`` clusters cells into the two wall layers.

    ``conductance`` is the wall conductance ratio of the two walls normal to the
    field, which is Hunt's configuration. Only ``sigma B^2`` is physical, so the
    field is scaled to keep the Hartmann number fixed when the conductivity moves.
    """
    if ratio is None:
        transverse = spanwise = uniform_faces(cells, -1.0, 1.0)
    else:
        transverse = wall_resolving_faces(
            cells, -1.0, 1.0, layer_thickness=1.0 / hartmann, cells_in_layer=6, max_ratio=ratio
        )
        spanwise = wall_resolving_faces(
            cells, -1.0, 1.0, layer_thickness=1.0 / np.sqrt(hartmann), cells_in_layer=6, max_ratio=ratio
        )
    return ChannelProblem(
        grid=Grid(uniform_faces(1, 0.0, 1.0), transverse, spanwise),
        conditions=(PERIODIC_X, WALL, WALL),
        conductivity=conductivity if hartmann else 0.0,
        magnetic_field=(0.0, hartmann / np.sqrt(conductivity), 0.0),
        forcing=(1.0, 0.0, 0.0),
        dt=dt,
        wall_conductance=(0.0, conductance, 0.0),
    )


HUNT_FLOW_RATE = {0.027: 0.027577890, 0.100: 0.017148080}


def _mean(problem: ChannelProblem, velocity) -> float:
    volumes = np.asarray(problem.grid.cell_volumes())[0]
    return float((np.asarray(velocity[0].data)[0] * volumes).sum() / volumes.sum())


def test_the_residual_vanishes_exactly_at_a_fixed_point_of_the_step():
    """The two have to agree on what steady means, or the solve answers a different question."""
    problem = _duct(8, 5.0, dt=0.02)
    factorization = problem.factorization()
    viscous = problem.viscous_factorizations()
    velocity = zero_velocity(problem)
    for _ in range(300):
        velocity, _, _ = step(velocity, problem, factorization, viscous)
    residual = steady_residual(velocity, problem, factorization)
    size = max(float(jnp.max(jnp.abs(velocity[0].data))), 1.0)
    assert max(float(jnp.max(jnp.abs(field.data))) for field in residual) < 1e-9 * size


def test_the_steady_solve_reproduces_the_marched_state_far_faster(monkeypatch):
    """The steady solve reaches the marched state with far fewer projections than marching.

    A claim about the method, not about the machine, so the work is counted rather
    than timed: both paths are built from the same projection, and a clock also
    measures host load and whether the solve's loops were already compiled.
    ``jax.debug.callback`` fires once per execution, inside compiled loops too, so
    the counts are applications and not traces.
    """
    import lmx.core3d
    import lmx.steady

    projections = [0]
    project = lmx.core3d.project

    def bump():
        projections[0] += 1

    def counted_project(*args, **kwargs):
        jax.debug.callback(bump)
        return project(*args, **kwargs)

    monkeypatch.setattr(lmx.core3d, "project", counted_project)
    monkeypatch.setattr(lmx.steady, "project", counted_project)

    problem = _duct(12, 5.0, dt=0.02)
    factorization = problem.factorization()
    viscous = problem.viscous_factorizations()
    velocity = zero_velocity(problem)
    for _ in range(300):
        velocity, _, _ = step(velocity, problem, factorization, viscous)
    jax.effects_barrier()
    marched = projections[0]
    assert marched == 300

    projections[0] = 0
    solution = solve_steady_state(problem, pseudo_step=100.0)
    jax.effects_barrier()
    assert abs(_mean(problem, solution.velocity) - _mean(problem, velocity)) < 1e-8
    # This insulating Stokes duct is one CG solve, whose operator and preconditioner both
    # project through the counted name, so the count is the work. Newton-Krylov projects
    # the linearised residual in the tangent, where no callback fires, but each such
    # application is paired with a counted one: a Krylov iteration with its
    # preconditioner, a restart cycle's true residual with its Newton step's residual.
    # Doubling the count therefore bounds the work of either route.
    newton = projections[0]
    # A count of zero means the solve no longer projects through the patched names, and
    # would pass the comparison below without measuring anything.
    assert newton > 0, "the steady solve applied no counted projection; the counter no longer sees it"
    assert 2 * newton < marched, (
        f"steady solve: {newton} projections (bound {2 * newton}); marching: {marched}"
    )


@pytest.mark.parametrize(
    ("hartmann", "cells", "ratio", "bound"),
    [
        (0.0, 16, None, 0.02),
        (5.0, 32, None, 0.02),
        (20.0, 32, 1.35, 0.02),
        (100.0, 48, 1.35, 0.01),
        (300.0, 48, 1.45, 0.02),
    ],
)
def test_the_steady_duct_matches_the_spectral_reference(hartmann, cells, ratio, bound):
    problem = _duct(cells, hartmann, ratio=ratio)
    solution = solve_steady_state(problem, pseudo_step=1.0e3)
    exact = REFERENCE_FLOW_RATE[hartmann]
    assert abs(_mean(problem, solution.velocity) - exact) / exact < bound


@pytest.mark.slow
def test_the_steady_duct_reaches_hartmann_1000():
    """The blanket-scale end of the range, where the Hartmann layer is a thousandth wide."""
    problem = _duct(64, 1000.0, ratio=1.5)
    solution = solve_steady_state(problem, pseudo_step=1.0e3, tolerance=1.0e-7, linear_restart=600)
    exact = REFERENCE_FLOW_RATE[1000.0]
    assert abs(_mean(problem, solution.velocity) - exact) / exact < 0.02


@pytest.mark.slow
def test_the_reference_values_are_what_the_spectral_solve_returns():
    """The constants above are cached results, so something has to regenerate them."""
    for hartmann, cached in REFERENCE_FLOW_RATE.items():
        assert flow_rate(hartmann, 96 if hartmann > 300.0 else 64) == pytest.approx(cached, rel=2e-4)


@pytest.mark.parametrize(
    ("cells", "hartmann", "conductance"), [(4, 0.0, 0.0), (10, 5.0, 0.0), (6, 5.0, 0.027)]
)
def test_the_adjoint_matches_finite_differences(cells, hartmann, conductance):
    """One linearised solve at the root, not a tape of the iteration."""
    problem = _duct(cells, hartmann, conductance=conductance)

    def throughput(drive, scale):
        solution = solve_steady_state(
            problem, pseudo_step=100.0, forcing=(drive, 0.0, 0.0), field_scale=scale
        )
        return jnp.mean(solution.velocity[0].data)

    value, gradient = jax.value_and_grad(throughput, argnums=(0, 1))(1.0, 1.0)
    compiled_value, compiled_gradient = jax.jit(jax.value_and_grad(throughput, argnums=(0, 1)))(1.0, 1.0)
    np.testing.assert_allclose(compiled_value, value, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(compiled_gradient, gradient, rtol=1e-8, atol=1e-12)
    # The compiled objective is held to the eager one, then reused for the differences:
    # an eager call traces and compiles the steady solve every time, which was most of
    # this test's cost.
    compiled = jax.jit(throughput)
    assert float(compiled(1.0, 1.0)) == pytest.approx(float(value), rel=1e-10)
    size = 1.0e-5
    for index, argument in enumerate(((1.0, 1.0), (1.0, 1.0))):
        raised = list(argument)
        lowered = list(argument)
        raised[index] += size
        lowered[index] -= size
        difference = (compiled(*raised) - compiled(*lowered)) / (2.0 * size)
        assert float(gradient[index]) == pytest.approx(float(difference), rel=1e-6)
    # The Stokes limit is linear in the drive, so the derivative is the value itself.
    assert float(gradient[0]) == pytest.approx(float(value), rel=1e-12)


@pytest.mark.parametrize(
    ("hartmann", "cells"), [(100.0, 24), pytest.param(300.0, 48, marks=pytest.mark.slow)]
)
def test_the_adjoint_matches_finite_differences_where_the_layers_are_thin(hartmann, cells):
    """Stretched layer meshes need the preconditioned transpose; without it Ha 300 gave NaN."""
    problem = duct_problem(hartmann=hartmann, cells=cells)

    def throughput(drive, scale):
        solution = solve_steady_state(
            problem, pseudo_step=1.0e3, forcing=(drive, 0.0, 0.0), field_scale=scale
        )
        return jnp.mean(solution.velocity[0].data)

    value, gradient = jax.value_and_grad(throughput, argnums=(0, 1))(1.0, 1.0)
    assert np.isfinite(value) and np.isfinite(gradient).all()
    size = 1.0e-4
    for index in range(2):
        raised = [1.0, 1.0]
        lowered = [1.0, 1.0]
        raised[index] += size
        lowered[index] -= size
        difference = (throughput(*raised) - throughput(*lowered)) / (2.0 * size)
        assert float(gradient[index]) == pytest.approx(float(difference), rel=1e-6)
    # A stronger field brakes the flow.
    assert float(gradient[1]) < 0.0


def _random_velocity(problem: ChannelProblem, seed: int):
    """A random velocity with no constraint imposed: wall faces and periodic copies are free."""
    from lmx.core3d import velocity_offset
    from lmx.grid import Field

    keys = jax.random.split(jax.random.PRNGKey(seed), 3)
    return tuple(
        Field(
            jax.random.normal(
                keys[axis], problem.grid.offset_shape(velocity_offset(axis)), dtype=jnp.float64
            ),
            velocity_offset(axis),
            problem.grid,
        )
        for axis in range(3)
    )


def _face_volume_inner(problem: ChannelProblem, left, right) -> float:
    from lmx.core3d import velocity_condition
    from lmx.ops import face_inner_product

    return sum(
        float(face_inner_product(a, b, axis, velocity_condition(problem.conditions, axis)))
        for axis, (a, b) in enumerate(zip(left, right, strict=True))
    )


def _stokes_operator_samples(problem: ChannelProblem):
    """Return <v, A u>, <u, A v> and <u, A u> for two random divergence-free velocities."""
    from lmx.core3d import project

    factorization = problem.factorization()
    rest = steady_residual(zero_velocity(problem), problem, factorization)

    def operator(velocity):
        value = steady_residual(velocity, problem, factorization)
        return tuple(
            field.replace_data(field.data - base.data) for field, base in zip(value, rest, strict=True)
        )

    u, v = (project(_random_velocity(problem, seed), problem, factorization)[0] for seed in (1, 2))
    inner = functools.partial(_face_volume_inner, problem)
    return inner(v, operator(u)), inner(u, operator(v)), inner(u, operator(u))


def _extruded(problem: ChannelProblem, axial_cells: int, hartmann: float, cells: int) -> ChannelProblem:
    """The same duct cross-section repeated over several periodic axial cells."""
    transverse = wall_resolving_faces(
        cells, -1.0, 1.0, layer_thickness=1.0 / hartmann, cells_in_layer=6, max_ratio=None
    )
    spanwise = wall_resolving_faces(
        cells, -1.0, 1.0, layer_thickness=1.0 / np.sqrt(hartmann), cells_in_layer=6, max_ratio=None
    )
    return dataclasses.replace(problem, grid=Grid(uniform_faces(axial_cells, 0.0, 1.0), transverse, spanwise))


def test_the_conjugate_gradient_preconditioner_is_symmetric_positive_definite():
    """CG needs the preconditioner symmetric positive definite in the same inner product as the operator.

    Several axial cells, because the periodic wrap face is where a one-cell
    duct hides an error: the viscous solve returns the duplicate face as zero,
    and completing it by averaging instead of copying made the axial viscous
    inverse asymmetric by 2.7e-2 on a 24-cube Ha 20 duct, and CG took 320
    iterations instead of 119.
    """
    from lmx.core3d import project
    from lmx.steady import _orthogonal_projection, _preconditioner, _projection_solves

    problem = _extruded(duct_problem(hartmann=100.0, cells=24), 4, 100.0, 24)
    factorization = problem.factorization()
    precond = _preconditioner(problem, factorization, _projection_solves(problem, 1.0e3), 1.0e3)
    inner = functools.partial(_face_volume_inner, problem)

    u, v = _random_velocity(problem, 1), _random_velocity(problem, 2)
    orthogonal = functools.partial(_orthogonal_projection, problem=problem, factorization=factorization)
    assert abs(inner(v, orthogonal(u)) - inner(orthogonal(v), u)) <= 1e-12 * abs(inner(v, orthogonal(u)))
    # The oblique projection the time step uses is not self-adjoint, which is why the adjoint needs the other.
    oblique = project(u, problem, factorization)[0], project(v, problem, factorization)[0]
    assert abs(inner(v, oblique[0]) - inner(oblique[1], u)) > 1e-3 * abs(inner(v, oblique[0]))

    up, vp = orthogonal(_random_velocity(problem, 3)), orthogonal(_random_velocity(problem, 4))
    forward, backward = inner(vp, precond(up)), inner(up, precond(vp))
    # The eigenbases of a stretched axis carry about 1e-9; the wrap-face defect was 2.7e-2.
    assert abs(forward - backward) <= 1e-8 * max(abs(forward), abs(backward))
    assert inner(up, precond(up)) > 0.0 and inner(vp, precond(vp)) > 0.0


def _conjugate_gradient_iterations(problem: ChannelProblem, solves) -> int:
    """CG iterations of the insulating steady solve from rest, with the given projection-step solves."""
    from lmx.steady import _orthogonal_projection, _preconditioner, _stokes_limit_root

    factorization = problem.factorization()
    precond = _preconditioner(problem, factorization, solves(problem, 1.0e3), 1.0e3)
    start = _orthogonal_projection(zero_velocity(problem), problem, factorization)
    controls = dict(forcing=None, field_scale=1.0, tolerance=1.0e-9, max_iterations=12000)
    _, (iterations, _, converged) = _stokes_limit_root(problem, start, factorization, precond, **controls)
    assert bool(converged)
    return int(iterations)


@pytest.mark.parametrize(
    ("hartmann", "cells", "bound"),
    [(300.0, 48, 70), pytest.param(1000.0, 64, 110, marks=pytest.mark.slow)],
)
def test_field_lines_cut_the_iterations_where_the_layers_are_thin(hartmann, cells, bound):
    """Field lines take CG from 403 to 44 iterations at Ha 300 and from 714 to 68 at Ha 1000.

    Round-off in the secondary components seeds part of the count, hence ~60 % headroom; the
    1.10b parity gate is 300 at Ha 1000.
    """
    from lmx.steady import _isotropic_viscous, _projection_solves

    problem = duct_problem(hartmann=hartmann, cells=cells)
    lines = _conjugate_gradient_iterations(problem, _projection_solves)
    assert lines <= bound, f"field lines took {lines} iterations"
    damped = _conjugate_gradient_iterations(problem, _isotropic_viscous)
    assert 6 * lines <= damped, f"field lines {lines}, damped {damped}"


def test_field_lines_invert_the_fully_developed_operator_on_a_uniform_mesh():
    """On a uniform mesh the induction form is the discrete operator: 3 iterations at Ha 300, 218 damped."""
    from lmx.steady import _projection_solves

    assert _conjugate_gradient_iterations(_duct(24, 300.0), _projection_solves) <= 5


def test_the_conjugate_gradient_route_agrees_with_newton_krylov():
    """A unidirectional duct flow does not advect itself, so both routes answer the same question.

    Without advection the steady state is one CG solve; with the central
    advection switched on it goes through Newton-GMRES, whose restart cycle is
    what memory allows in three dimensions. The advective flux needs more than
    one axial cell, so both ducts are extruded.
    """
    symmetric = _extruded(duct_problem(hartmann=100.0, cells=24), 4, 100.0, 24)
    advective = _extruded(duct_problem(hartmann=100.0, cells=24, advection="central"), 4, 100.0, 24)
    cg = solve_steady_state(symmetric, pseudo_step=1.0e3).velocity
    newton = solve_steady_state(advective, pseudo_step=1.0e3).velocity
    size = max(float(jnp.max(jnp.abs(field.data))) for field in newton)
    difference = max(float(jnp.max(jnp.abs(a.data - b.data))) for a, b in zip(cg, newton, strict=True))
    assert difference <= 1e-8 * size


def test_a_periodic_extrusion_reproduces_the_duct_by_conjugate_gradients():
    """Six axial cells carry the one-cell solution; round-off excites the axial modes CG must still resolve."""
    duct = duct_problem(hartmann=20.0, cells=24)
    flat = solve_steady_state(duct, pseudo_step=1.0e3).velocity[0].data
    deep = solve_steady_state(_extruded(duct, 6, 20.0, 24), pseudo_step=1.0e3).velocity[0].data
    assert float(jnp.max(jnp.abs(deep - flat[:1]))) <= 1e-8 * float(jnp.max(jnp.abs(flat)))


def test_mixed_precision_reaches_the_same_steady_duct():
    """Float32 fast solves with a float64 correction keep CG's operator linear enough."""
    duct = duct_problem(hartmann=20.0, cells=24)
    state = _mean(duct, solve_steady_state(duct, pseudo_step=1.0e3).velocity)
    mixed_duct = dataclasses.replace(duct, precision="mixed")
    mixed = _mean(mixed_duct, solve_steady_state(mixed_duct, pseudo_step=1.0e3).velocity)
    assert mixed == pytest.approx(state, rel=1e-9)


@pytest.mark.parametrize("conductance", [0.0, 0.05])
def test_the_stokes_operator_is_symmetric_on_a_layer_mesh(conductance):
    """The electromotive and force interpolations are adjoint, so conjugate gradients apply.

    The Stokes-limit residual is affine in the velocity. Its linear part was
    symmetric in the face-volume inner product on a uniform mesh and 1e-2 away
    from it on this one, because the force was interpolated with a stencil that
    was not the transpose of the electromotive one. A thin conducting wall is a
    sheet of potential joined to the fluid by the half-cell flux and solved
    directly, so it keeps the operator symmetric as well.
    """
    problem = duct_problem(hartmann=100.0, cells=32, wall_conductance=conductance)
    forward, backward, energy = _stokes_operator_samples(problem)
    asymmetry = abs(forward - backward) / max(abs(forward), abs(backward))
    assert asymmetry <= 1e-12
    # Viscosity and Joule dissipation both remove energy.
    assert energy < 0.0


def test_a_solve_that_does_not_converge_raises():
    """A plausible field and a gradient taken away from a root are worse than an error."""
    problem = _duct(16, 20.0)
    with pytest.raises(RuntimeError, match="did not converge"):
        solve_steady_state(problem, pseudo_step=1.0e3, max_steps=1, linear_restart=1, linear_max_restarts=1)


@pytest.mark.parametrize("drive", [1.0, np.nan, np.inf])
def test_a_rejected_root_cannot_produce_a_finite_objective_or_gradient(drive):
    problem = _duct(4, 0.0)

    def objective(drive):
        solution = solve_steady_state(problem, forcing=(drive, 0.0, 0.0), max_steps=0)
        return jnp.mean(solution.velocity[0].data)

    with pytest.raises(RuntimeError, match="did not converge"):
        objective(drive)
    value, gradient = jax.value_and_grad(objective)(drive)
    assert not np.isfinite(value) and not np.isfinite(gradient)
    value, gradient = jax.jit(jax.value_and_grad(objective))(drive)
    assert not np.isfinite(value) and not np.isfinite(gradient)


def test_a_failed_tangent_solve_is_rejected_eagerly_and_under_jit():
    from lmx.steady import _krylov, _tangent_solve

    def solve(rhs):
        return _krylov(jnp.zeros_like, rhs)

    with pytest.raises(RuntimeError, match="linear solve did not converge"):
        solve(jnp.ones(2))
    assert not np.isfinite(jax.jit(solve)(jnp.ones(2))).all()
    value, gradient = jax.value_and_grad(lambda rhs: jnp.sum(_tangent_solve(jnp.zeros_like, rhs)))(
        jnp.ones(2)
    )
    assert not np.isfinite(value) and not np.isfinite(gradient).all()


@pytest.mark.parametrize("name", ["pseudo_step", "tolerance", "linear_tolerance"])
@pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf])
def test_the_solver_controls_are_validated(name, value):
    with pytest.raises(ValueError, match=f"{name} must be positive and finite"):
        solve_steady_state(_duct(4, 0.0), **{name: value})


@pytest.mark.parametrize("conductance", [0.027, 0.100])
def test_a_hunt_duct_matches_the_spectral_reference(conductance):
    """Conducting Hartmann walls carry the current the layer would otherwise have to."""
    problem = _duct(32, 20.0, ratio=1.35, conductance=conductance)
    rate = _mean(problem, solve_steady_state(problem, pseudo_step=1.0e3).velocity)
    assert abs(rate - HUNT_FLOW_RATE[conductance]) / HUNT_FLOW_RATE[conductance] < 0.02
    # A conducting wall short-circuits the Hartmann layer, so it always slows the flow.
    assert rate < REFERENCE_FLOW_RATE[20.0]


def test_a_wall_of_no_conductance_is_the_insulating_duct():
    """The conducting closure has to collapse onto the one it generalises."""
    rates = [
        _mean(problem, solve_steady_state(problem, pseudo_step=100.0).velocity)
        for problem in (_duct(16, 5.0), _duct(16, 5.0, conductance=0.0))
    ]
    assert rates[0] == rates[1]


@pytest.mark.parametrize("conductance", [0.0, 0.027])
def test_the_answer_follows_the_hartmann_number_and_not_the_conductivity(conductance):
    """Only `sigma B^2` is physical, so the potential has to be scaled by sigma, and a sheet's conduction with it."""
    rates = [
        _mean(problem, solve_steady_state(problem, pseudo_step=100.0).velocity)
        for problem in (
            _duct(16, 5.0, conductance=conductance),
            _duct(16, 5.0, conductance=conductance, conductivity=4.0),
        )
    ]
    assert rates[0] == pytest.approx(rates[1], rel=1e-12)


@pytest.mark.slow
def test_the_hunt_reference_values_are_what_the_spectral_solve_returns():
    for conductance, cached in HUNT_FLOW_RATE.items():
        assert flow_rate(20.0, 64, hartmann_wall=conductance) == pytest.approx(cached, rel=2e-4)


def test_the_steady_state_closes_its_mechanical_power_balance():
    """What the drive and the field put in is what viscosity takes out, to round-off.

    The pressure does no work on a discretely divergence-free velocity, so this
    is the residual of the steady solve projected onto the velocity itself: an
    independent reading of the same claim, in energy rather than in momentum.
    """
    from lmx.timeloop import energy_budget

    for conductance in (0.0, 0.027):
        problem = _duct(32, 20.0, ratio=1.35, conductance=conductance)
        budget = energy_budget(solve_steady_state(problem, pseudo_step=1.0e3).velocity, problem)
        assert abs(float(budget.defect / budget.scale)) < 1e-10
        assert float(budget.drive) > 0.0
        assert float(budget.viscous) > 0.0

"""The steady solve: it is the fixed point of the step, and its gradient is exact."""

import time

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


def test_the_steady_solve_reproduces_the_marched_state_far_faster():
    problem = _duct(12, 5.0, dt=0.02)
    factorization = problem.factorization()
    viscous = problem.viscous_factorizations()
    velocity = zero_velocity(problem)
    started = time.perf_counter()
    for _ in range(300):
        velocity, _, _ = step(velocity, problem, factorization, viscous)
    marched = time.perf_counter() - started

    started = time.perf_counter()
    solution = solve_steady_state(problem, pseudo_step=100.0)
    newton = time.perf_counter() - started
    assert abs(_mean(problem, solution.velocity) - _mean(problem, velocity)) < 1e-8
    # A claim about the method, not about the machine: Newton has to beat 300 steps.
    assert newton < marched


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
    assert float(jax.jit(throughput)(1.0, 1.0)) == pytest.approx(float(value), rel=1e-10)
    size = 1.0e-5
    for index, argument in enumerate(((1.0, 1.0), (1.0, 1.0))):
        raised = list(argument)
        lowered = list(argument)
        raised[index] += size
        lowered[index] -= size
        difference = (throughput(*raised) - throughput(*lowered)) / (2.0 * size)
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


def _stokes_operator_samples(problem: ChannelProblem):
    """Return <v, A u>, <u, A v> and <u, A u> for two random divergence-free velocities."""
    from lmx.core3d import project, velocity_condition, velocity_offset
    from lmx.grid import Field
    from lmx.ops import face_inner_product

    factorization = problem.factorization()
    rest = steady_residual(zero_velocity(problem), problem, factorization)

    def operator(velocity):
        value = steady_residual(velocity, problem, factorization)
        return tuple(
            field.replace_data(field.data - base.data) for field, base in zip(value, rest, strict=True)
        )

    def sample(seed):
        keys = jax.random.split(jax.random.PRNGKey(seed), 3)
        raw = tuple(
            Field(
                jax.random.normal(
                    keys[axis], problem.grid.offset_shape(velocity_offset(axis)), dtype=jnp.float64
                ),
                velocity_offset(axis),
                problem.grid,
            )
            for axis in range(3)
        )
        return project(raw, problem, factorization)[0]

    def inner(left, right):
        return sum(
            float(face_inner_product(a, b, axis, velocity_condition(problem.conditions, axis)))
            for axis, (a, b) in enumerate(zip(left, right, strict=True))
        )

    u, v = sample(1), sample(2)
    return inner(v, operator(u)), inner(u, operator(v)), inner(u, operator(u))


@pytest.mark.parametrize("conductance", [0.0, 0.05])
def test_the_stokes_operator_is_symmetric_on_a_layer_mesh(conductance):
    """The electromotive and force interpolations are adjoint, so conjugate gradients apply.

    The Stokes-limit residual is affine in the velocity. Its linear part was
    symmetric in the face-volume inner product on a uniform mesh and 1e-2 away
    from it on this one, because the force was interpolated with a stencil that
    was not the transpose of the electromotive one. The conducting wall is not
    gated here: its closure is first order and is replaced in plan step 1.3b.
    """
    problem = duct_problem(hartmann=100.0, cells=32, wall_conductance=conductance)
    forward, backward, energy = _stokes_operator_samples(problem)
    asymmetry = abs(forward - backward) / max(abs(forward), abs(backward))
    if not conductance:
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
    assert rates[0] == pytest.approx(rates[1], rel=1e-14)


def test_the_answer_follows_the_hartmann_number_and_not_the_conductivity():
    """Only `sigma B^2` is physical, so the potential has to be scaled by sigma."""
    rates = [
        _mean(problem, solve_steady_state(problem, pseudo_step=100.0).velocity)
        for problem in (_duct(16, 5.0), _duct(16, 5.0, conductivity=4.0))
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

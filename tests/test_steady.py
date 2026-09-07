"""The steady solve: it is the fixed point of the step, and its gradient is exact."""

import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lmx.bc import NEUMANN, PERIODIC, BoundaryCondition
from lmx.core3d import ChannelProblem, step, zero_velocity
from lmx.grid import Grid, uniform_faces, wall_resolving_faces
from lmx.steady import solve_steady_state, steady_residual
from validation.shercliff import flow_rate

pytestmark = pytest.mark.numerical

PERIODIC_X = BoundaryCondition(PERIODIC)
WALL = BoundaryCondition(NEUMANN)


def _duct(cells: int, hartmann: float, *, resolve_layers: bool = False, dt: float = 1.0) -> ChannelProblem:
    if resolve_layers:
        transverse = wall_resolving_faces(
            cells, -1.0, 1.0, layer_thickness=1.0 / hartmann, cells_in_layer=6, max_ratio=1.35
        )
        spanwise = wall_resolving_faces(
            cells, -1.0, 1.0, layer_thickness=1.0 / np.sqrt(hartmann), cells_in_layer=6, max_ratio=1.35
        )
    else:
        transverse = spanwise = uniform_faces(cells, -1.0, 1.0)
    return ChannelProblem(
        grid=Grid(uniform_faces(1, 0.0, 1.0), transverse, spanwise),
        conditions=(PERIODIC_X, WALL, WALL),
        conductivity=1.0 if hartmann else 0.0,
        magnetic_field=(0.0, hartmann, 0.0),
        forcing=(1.0, 0.0, 0.0),
        dt=dt,
    )


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
    ("hartmann", "cells", "resolve", "bound"),
    [(0.0, 16, False, 0.02), (5.0, 32, False, 0.02), (20.0, 32, True, 0.02), (100.0, 48, True, 0.01)],
)
def test_the_steady_duct_matches_the_spectral_reference(hartmann, cells, resolve, bound):
    problem = _duct(cells, hartmann, resolve_layers=resolve)
    solution = solve_steady_state(problem, pseudo_step=1.0e3)
    exact = flow_rate(hartmann, 48)
    assert abs(_mean(problem, solution.velocity) - exact) / exact < bound


def test_the_adjoint_matches_finite_differences():
    """One linearised solve at the root, not a tape of the iteration."""
    problem = _duct(10, 5.0)

    def throughput(drive, scale):
        solution = solve_steady_state(
            problem, pseudo_step=100.0, forcing=(drive, 0.0, 0.0), field_scale=scale
        )
        return jnp.mean(solution.velocity[0].data)

    value, gradient = jax.value_and_grad(throughput, argnums=(0, 1))(1.0, 1.0)
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


def test_a_solve_that_does_not_converge_raises():
    """A plausible field and a gradient taken away from a root are worse than an error."""
    problem = _duct(16, 20.0)
    with pytest.raises(RuntimeError, match="did not converge"):
        solve_steady_state(problem, pseudo_step=1.0e3, max_steps=1, linear_restart=1, linear_max_restarts=1)


def test_the_pseudo_step_is_validated():
    with pytest.raises(ValueError, match="pseudo_step must be positive"):
        solve_steady_state(_duct(6, 0.0), pseudo_step=0.0)

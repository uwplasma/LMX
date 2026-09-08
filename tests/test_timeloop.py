"""The compiled trajectory: equivalence with the host loop, and no host synchronisation."""

import inspect
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lmx import core3d, timeloop
from lmx.bc import NEUMANN, PERIODIC, BoundaryCondition
from lmx.core3d import ChannelProblem, step, zero_velocity
from lmx.grid import Grid, uniform_faces
from lmx.ops import divergence
from lmx.timeloop import advance, energy_budget, kinetic_energy, trajectory_diagnostics

pytestmark = pytest.mark.unit

PERIODIC_AXIS = BoundaryCondition(PERIODIC)
WALL = BoundaryCondition(NEUMANN)


def _problem(cells: int = 8, **overrides) -> ChannelProblem:
    grid = Grid(uniform_faces(3, 0.0, 1.0), uniform_faces(cells, -1.0, 1.0), uniform_faces(cells, -1.0, 1.0))
    settings = dict(
        grid=grid,
        conditions=(PERIODIC_AXIS, WALL, WALL),
        forcing=(1.0, 0.0, 0.0),
        dt=2.0e-3,
    )
    settings.update(overrides)
    return ChannelProblem(**settings)


def test_the_compiled_trajectory_reproduces_the_host_loop():
    problem = _problem()
    factorization = problem.factorization()
    steps = 12

    velocity = zero_velocity(problem)
    for _ in range(steps):
        velocity, _, _ = step(velocity, problem, factorization)

    run = advance(problem, steps, factorization=factorization)
    for scanned, looped in zip(run.velocity, velocity, strict=True):
        assert np.max(np.abs(np.asarray(scanned.data) - np.asarray(looped.data))) < 1e-13


def test_the_trajectory_reports_a_history_without_synchronising():
    problem = _problem()
    run = advance(problem, 10)
    assert run.steps == 10
    assert run.divergence_residual.shape == (10,)
    assert run.kinetic_energy.shape == (10,)
    # A driven flow starting from rest gains energy monotonically at first.
    energy = np.asarray(run.kinetic_energy)
    assert np.all(np.diff(energy) > 0.0)
    assert float(np.max(np.asarray(run.divergence_residual))) < 1e-11


def test_the_end_state_carries_the_final_pressure_and_potential():
    problem = _problem()
    run = advance(problem, 6)
    assert run.pressure.shape == problem.grid.shape
    assert run.potential.shape == problem.grid.shape
    assert np.all(np.isfinite(np.asarray(run.pressure.data)))
    assert float(np.max(np.abs(np.asarray(divergence(run.velocity).data)))) < 1e-11


def test_the_time_loop_holds_no_host_synchronisation():
    """A single `float(...)` inside the loop would serialise the device queue."""
    source = inspect.getsource(timeloop) + inspect.getsource(core3d)
    for pattern in (r"\bfloat\(\s*[a-z_]+\.data", r"\bbool\(", r"device_get", r"\.item\(\)"):
        assert not re.search(pattern, source), pattern


def test_the_trajectory_compiles_once_and_differentiates():
    problem = _problem(cells=6)
    factorization = problem.factorization()

    def kinetic(drive):
        driven = ChannelProblem(
            grid=problem.grid,
            conditions=problem.conditions,
            forcing=(drive, 0.0, 0.0),
            dt=problem.dt,
        )
        return advance(driven, 8, factorization=factorization).kinetic_energy[-1]

    value, gradient = jax.jit(jax.value_and_grad(kinetic))(1.0)
    assert float(value) > 0.0
    size = 1.0e-5
    difference = (kinetic(1.0 + size) - kinetic(1.0 - size)) / (2.0 * size)
    assert float(gradient) == pytest.approx(float(difference), rel=1e-6)


def test_checkpointing_does_not_change_the_answer():
    """Rematerialisation is a memory trade, not a numerical one."""
    problem = _problem(cells=6)
    factorization = problem.factorization()
    plain = advance(problem, 8, factorization=factorization, checkpoint=False)
    saved = advance(problem, 8, factorization=factorization, checkpoint=True)
    for first, second in zip(plain.velocity, saved.velocity, strict=True):
        assert np.max(np.abs(np.asarray(first.data) - np.asarray(second.data))) < 1e-14


def test_a_longer_trajectory_costs_no_extra_live_state():
    """Scan length is static, so the traced graph does not grow with the run."""
    problem = _problem(cells=6)
    factorization = problem.factorization()
    short = jax.make_jaxpr(lambda: advance(problem, 4, factorization=factorization).velocity[0].data)()
    long = jax.make_jaxpr(lambda: advance(problem, 64, factorization=factorization).velocity[0].data)()
    # The scan body is compiled once; only its trip count differs.
    assert abs(len(str(long)) - len(str(short))) < 0.05 * len(str(short))


def test_diagnostics_match_a_direct_evaluation():
    problem = _problem(cells=6)
    velocity = zero_velocity(problem)
    velocity, _, _ = step(velocity, problem)
    residual, energy = trajectory_diagnostics(velocity, problem)
    assert float(residual) == pytest.approx(float(jnp.max(jnp.abs(divergence(velocity).data))), rel=1e-12)
    assert float(energy) >= 0.0


def test_advance_validates_its_length():
    with pytest.raises(ValueError, match="steps must be positive"):
        advance(_problem(cells=4), 0)


def _uniform_duct(cells: int = 12, hartmann: float = 10.0, conductance: float = 0.0, **overrides):
    """A duct on a uniform mesh, where the interpolations carry no stretching error."""
    grid = Grid(uniform_faces(1, 0.0, 1.0), uniform_faces(cells, -1.0, 1.0), uniform_faces(cells, -1.0, 1.0))
    settings = dict(
        grid=grid,
        conditions=(PERIODIC_AXIS, WALL, WALL),
        conductivity=1.0,
        magnetic_field=(0.0, hartmann, 0.0),
        forcing=(1.0, 0.0, 0.0),
        dt=2.0e-3,
        wall_conductance=(0.0, conductance, 0.0),
    )
    settings.update(overrides)
    return ChannelProblem(**settings)


def test_the_lorentz_force_does_exactly_minus_the_joule_dissipation():
    """A discrete identity, not an approximation: it holds at any state, to round-off."""
    problem = _uniform_duct()
    run = advance(problem, 20)
    budget = energy_budget(run.velocity, problem)
    assert float(jnp.abs(budget.ohmic_defect / budget.scale)) < 1e-12
    assert float(budget.joule) > 0.0
    assert float(budget.lorentz) < 0.0


def test_a_conducting_wall_takes_power_out_through_the_boundary():
    """Leaving the boundary work out is a large error, not a small one."""
    budgets = {}
    for cells in (12, 24):
        problem = _uniform_duct(cells, conductance=0.05)
        budgets[cells] = energy_budget(advance(problem, 20).velocity, problem)
    budget = budgets[12]
    assert float(budget.wall) > 0.0
    # Dropping the boundary work is a percent-level error, an order above the closure itself.
    without = float(jnp.abs((budget.joule + budget.lorentz) / budget.scale))
    assert without > 20.0 * abs(float(budget.ohmic_defect / budget.scale))
    # The wall potential is the adjacent cell value, so the closure is first order.
    assert abs(float(budgets[24].ohmic_defect / budgets[24].scale)) < 0.75 * abs(
        float(budget.ohmic_defect / budget.scale)
    )


def test_the_budget_is_the_rate_of_change_of_kinetic_energy():
    """`defect` is dE/dt, so a step of half the size halves the error against it."""
    errors = []
    for step_size in (4.0e-3, 2.0e-3):
        problem = _uniform_duct(dt=step_size)
        start = advance(problem, 10).velocity
        budget = energy_budget(start, problem)
        after, _, _ = step(start, problem)
        rate = (kinetic_energy(after, problem) - kinetic_energy(start, problem)) / step_size
        errors.append(abs(float((rate - budget.defect) / budget.scale)))
    assert errors[1] < 0.6 * errors[0], errors


def test_a_run_can_be_taken_in_chunks():
    """Restart is the same physics: the state is the whole of it."""
    problem = _problem()
    factorization = problem.factorization()
    whole = advance(problem, 12, factorization=factorization)
    first = advance(problem, 6, factorization=factorization)
    second = advance(problem, 6, first.velocity, factorization=factorization)
    for chunked, complete in zip(second.velocity, whole.velocity, strict=True):
        assert np.max(np.abs(np.asarray(chunked.data) - np.asarray(complete.data))) < 1e-14

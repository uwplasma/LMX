import jax.numpy as jnp
import pytest

from lmx.cases import make_hartmann_case, make_hunt_case, make_shercliff_case
from lmx.solvers import solve_steady


pytestmark = pytest.mark.physics


def test_hartmann_solver_runs():
    case = make_hartmann_case(ha=10.0, ny=24, nz=24)
    solution = solve_steady(case)
    assert solution.state.u.shape == (24, 24)
    assert float(jnp.max(solution.state.u)) > 0.0
    assert jnp.isfinite(solution.state.phi).all()


def test_hunt_solver_keeps_solid_velocity_zero():
    case = make_hunt_case(ha=10.0, ny=20, nz=20, wall_cells=3)
    solution = solve_steady(case)
    assert solution.mesh.fluid_mask is not None
    assert jnp.allclose(solution.state.u[~solution.mesh.fluid_mask], 0.0)


def test_hunt_case_uses_ha_aware_coupling_controls():
    ha20 = make_hunt_case(ha=20.0, ny=16, nz=16, wall_cells=2)
    ha100 = make_hunt_case(ha=100.0, ny=16, nz=16, wall_cells=2)
    ha1000 = make_hunt_case(ha=1000.0, ny=16, nz=16, wall_cells=2)

    assert ha20.time_stepper.outer_iterations == 6
    assert ha20.time_stepper.velocity_update_limit == pytest.approx(2e-3)
    assert ha100.time_stepper.outer_iterations == 4
    assert ha100.time_stepper.velocity_update_limit == pytest.approx(1e-3)
    assert ha1000.time_stepper.outer_iterations == 3
    assert ha1000.time_stepper.velocity_update_limit == pytest.approx(1e-3)


def test_shercliff_solution_stays_finite_and_zero_at_walls():
    case = make_shercliff_case(ha=10.0, ny=24, nz=24)
    solution = solve_steady(case)
    assert jnp.isfinite(solution.state.u).all()
    assert jnp.allclose(solution.state.u[0, :], 0.0)
    assert jnp.allclose(solution.state.u[-1, :], 0.0)

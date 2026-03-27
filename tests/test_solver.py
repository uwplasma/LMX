import jax.numpy as jnp

from lmx.cases import make_hartmann_case, make_hunt_case, make_shercliff_case
from lmx.solvers import solve_steady


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


def test_shercliff_solution_stays_finite_and_zero_at_walls():
    case = make_shercliff_case(ha=10.0, ny=24, nz=24)
    solution = solve_steady(case)
    assert jnp.isfinite(solution.state.u).all()
    assert jnp.allclose(solution.state.u[0, :], 0.0)
    assert jnp.allclose(solution.state.u[-1, :], 0.0)

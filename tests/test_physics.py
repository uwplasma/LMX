from dataclasses import replace

import jax.numpy as jnp
import pytest

from lmx.cases import make_hartmann_case, make_hunt_case, make_shercliff_case
from lmx.solvers import solve_steady


pytestmark = pytest.mark.physics


def test_hartmann_profile_is_wall_bounded_and_center_peaked():
    case = make_hartmann_case(ha=2.0, ny=12, nz=12)
    solution = solve_steady(case)
    centerline = solution.state.u[:, solution.state.u.shape[1] // 2]
    left_half = centerline[: centerline.shape[0] // 2 + 1]
    assert jnp.allclose(centerline[0], 0.0)
    assert jnp.allclose(centerline[-1], 0.0)
    center_slice = centerline[centerline.shape[0] // 2 - 1 : centerline.shape[0] // 2 + 1]
    assert jnp.allclose(jnp.max(center_slice), jnp.max(centerline), atol=1e-6)
    assert jnp.all(jnp.diff(left_half) >= -5e-6)


def test_shercliff_profile_remains_symmetric_on_small_case():
    case = make_shercliff_case(ha=2.0, ny=12, nz=12)
    solution = solve_steady(case)
    centerline_y = solution.state.u[:, solution.state.u.shape[1] // 2]
    centerline_z = solution.state.u[solution.state.u.shape[0] // 2, :]
    assert jnp.allclose(centerline_y, jnp.flip(centerline_y), atol=3e-3)
    assert jnp.allclose(centerline_z, jnp.flip(centerline_z), atol=3e-3)


def test_shercliff_ha20_default_case_stays_bounded():
    case = make_shercliff_case(ha=20.0, ny=16, nz=16)
    solution = solve_steady(case)
    assert solution.state.residual < 1e-4
    assert float(jnp.max(solution.state.u)) < 0.1
    assert float(jnp.min(solution.state.u)) >= 0.0


def test_hunt_case_can_be_stabilized_with_small_pseudostep():
    case = make_hunt_case(ha=20.0, ny=16, nz=16, wall_cells=2)
    case = replace(case, time_stepper=replace(case.time_stepper, dt=1e-4, relaxation=0.05, max_steps=400, potential_iterations=500))
    solution = solve_steady(case)
    fluid_u = solution.state.u[solution.mesh.fluid_mask]
    assert solution.state.residual < 1e-4
    assert float(jnp.min(fluid_u)) >= 0.0
    assert float(jnp.max(fluid_u)) < 0.01


def test_hunt_default_case_now_stays_bounded():
    case = make_hunt_case(ha=20.0, ny=16, nz=16, wall_cells=2)
    solution = solve_steady(case)
    fluid_u = solution.state.u[solution.mesh.fluid_mask]
    assert solution.state.residual <= 1.1e-3
    assert float(jnp.max(fluid_u)) < 0.01
    assert float(jnp.min(fluid_u)) > -1e-3

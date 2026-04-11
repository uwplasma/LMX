import jax
import jax.numpy as jnp
import pytest

from lmx.autodiff import (
    build_hartmann_autodiff_problem,
    hartmann_mean_velocity,
    hartmann_mean_velocity_finite_difference_gradients,
    hartmann_mean_velocity_gradients,
    hartmann_profile_loss,
    solve_differentiable_hartmann,
)


pytestmark = pytest.mark.unit


def test_differentiable_hartmann_solution_returns_finite_fields():
    problem = build_hartmann_autodiff_problem(ny=12, nz=12, macro_iterations=3, potential_iterations=12, velocity_iterations=16)
    u, phi = solve_differentiable_hartmann(problem, forcing=1.0, hartmann_number=5.0)

    assert u.shape == (12, 12)
    assert phi.shape == (12, 12)
    assert jnp.isfinite(u).all()
    assert jnp.isfinite(phi).all()


def test_hartmann_mean_velocity_is_differentiable():
    problem = build_hartmann_autodiff_problem(ny=12, nz=12, macro_iterations=3, potential_iterations=12, velocity_iterations=16)
    value, gradient = jax.value_and_grad(lambda ha: hartmann_mean_velocity(problem, forcing=1.0, hartmann_number=ha))(5.0)

    assert jnp.isfinite(value)
    assert jnp.isfinite(gradient)


def test_profile_loss_gradient_step_reduces_objective():
    problem = build_hartmann_autodiff_problem(ny=12, nz=12, macro_iterations=3, potential_iterations=12, velocity_iterations=16)
    target_u, _ = solve_differentiable_hartmann(problem, forcing=1.0, hartmann_number=9.0)
    target_profile = target_u[:, target_u.shape[1] // 2]

    objective = lambda ha: hartmann_profile_loss(problem, forcing=1.0, hartmann_number=ha, target_profile=target_profile)
    loss0, grad0 = jax.value_and_grad(objective)(4.0)
    loss1 = objective(jnp.clip(4.0 - 2.0 * grad0, 0.5, 30.0))

    assert jnp.isfinite(loss0)
    assert jnp.isfinite(grad0)
    assert float(loss1) <= float(loss0)


def test_mean_velocity_gradients_match_finite_difference():
    problem = build_hartmann_autodiff_problem(ny=12, nz=12, macro_iterations=3, potential_iterations=12, velocity_iterations=16)
    autodiff = hartmann_mean_velocity_gradients(problem, forcing=1.1, hartmann_number=5.0)
    finite_diff = hartmann_mean_velocity_finite_difference_gradients(problem, forcing=1.1, hartmann_number=5.0)

    assert jnp.isfinite(autodiff["d_mean_velocity_d_forcing"])
    assert jnp.isfinite(autodiff["d_mean_velocity_d_ha"])
    assert float(jnp.abs(autodiff["d_mean_velocity_d_forcing"] - finite_diff["d_mean_velocity_d_forcing"])) < 5.0e-2
    assert float(jnp.abs(autodiff["d_mean_velocity_d_ha"] - finite_diff["d_mean_velocity_d_ha"])) < 5.0e-2

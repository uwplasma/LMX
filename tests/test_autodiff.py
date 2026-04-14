import jax
import jax.numpy as jnp
import pytest

from lmx.autodiff import (
    build_extruded_response_targets,
    build_fringing_autodiff_problem,
    build_hartmann_autodiff_problem,
    extruded_rect_response_history,
    extruded_rect_response_loss_gradients,
    fringing_history_loss_gradients,
    fringing_mean_velocity_history,
    fringing_response_history,
    fringing_response_loss_gradients,
    hartmann_mean_velocity,
    hartmann_mean_velocity_finite_difference_gradients,
    hartmann_mean_velocity_gradients,
    hartmann_profile_loss,
    hartmann_profile_loss_gradients,
    run_extruded_rect_inverse_design,
    run_fringing_response_inverse_design,
    run_extruded_target_inverse_design,
    run_fringing_history_inverse_design,
    run_hartmann_profile_inverse_design,
    solve_differentiable_hartmann,
)
from lmx.fringing import build_square_duct_extruded_problem, solve_extruded_inductionless


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


def test_profile_loss_gradients_are_finite():
    problem = build_hartmann_autodiff_problem(ny=12, nz=12, macro_iterations=3, potential_iterations=12, velocity_iterations=16)
    target_u, _ = solve_differentiable_hartmann(problem, forcing=1.0, hartmann_number=8.0)
    target_profile = target_u[:, target_u.shape[1] // 2]

    gradients = hartmann_profile_loss_gradients(
        problem,
        forcing=0.7,
        hartmann_number=4.5,
        target_profile=target_profile,
    )

    assert jnp.isfinite(gradients["loss"])
    assert jnp.isfinite(gradients["d_loss_d_forcing"])
    assert jnp.isfinite(gradients["d_loss_d_ha"])


def test_profile_inverse_design_reduces_loss():
    problem = build_hartmann_autodiff_problem(ny=12, nz=12, macro_iterations=3, potential_iterations=12, velocity_iterations=16)
    target_u, _ = solve_differentiable_hartmann(problem, forcing=1.0, hartmann_number=10.0)
    target_profile = target_u[:, target_u.shape[1] // 2]

    result = run_hartmann_profile_inverse_design(
        problem,
        target_profile=target_profile,
        forcing_init=0.4,
        hartmann_init=4.0,
        learning_rate_forcing=10.0,
        learning_rate_ha=2.0,
        steps=8,
    )

    assert len(result["history"]) == 8
    assert result["history"][-1]["loss"] <= result["history"][0]["loss"]
    assert jnp.isfinite(result["recovered_profile"]).all()


def test_fringing_mean_velocity_history_returns_finite_trace():
    problem = build_fringing_autodiff_problem(nx_stations=9, ny=10, nz=10, macro_iterations=2, potential_iterations=10, velocity_iterations=12)
    payload = fringing_mean_velocity_history(
        problem,
        forcing=1.0,
        peak_hartmann_number=12.0,
        entry_center=1.0,
        exit_center=4.0,
        transition_width=0.4,
    )

    assert payload["x"].shape == (9,)
    assert payload["field_scale"].shape == (9,)
    assert payload["mean_velocity"].shape == (9,)
    assert jnp.isfinite(payload["mean_velocity"]).all()


def test_fringing_history_loss_gradients_are_finite():
    problem = build_fringing_autodiff_problem(nx_stations=9, ny=10, nz=10, macro_iterations=2, potential_iterations=10, velocity_iterations=12)
    target = fringing_mean_velocity_history(
        problem,
        forcing=1.0,
        peak_hartmann_number=14.0,
        entry_center=1.2,
        exit_center=4.1,
        transition_width=0.35,
    )
    gradients = fringing_history_loss_gradients(
        problem,
        forcing=1.0,
        peak_hartmann_number=10.0,
        entry_center=0.8,
        exit_center=4.8,
        transition_width=0.6,
        target_mean_velocity=target["mean_velocity"],
    )

    assert jnp.isfinite(gradients["loss"])
    assert jnp.isfinite(gradients["d_peak_hartmann_number"])
    assert jnp.isfinite(gradients["d_entry_center"])
    assert jnp.isfinite(gradients["d_exit_center"])
    assert jnp.isfinite(gradients["d_transition_width"])


def test_fringing_response_history_returns_finite_current_proxy():
    problem = build_fringing_autodiff_problem(nx_stations=9, ny=10, nz=10, macro_iterations=2, potential_iterations=10, velocity_iterations=12)
    payload = fringing_response_history(
        problem,
        forcing=1.0,
        peak_hartmann_number=12.0,
        entry_center=1.0,
        exit_center=4.0,
        transition_width=0.4,
    )
    assert payload["current_proxy"].shape == (9,)
    assert jnp.isfinite(payload["current_proxy"]).all()


def test_fringing_response_loss_gradients_are_finite():
    problem = build_fringing_autodiff_problem(nx_stations=9, ny=10, nz=10, macro_iterations=2, potential_iterations=10, velocity_iterations=12)
    target = fringing_response_history(
        problem,
        forcing=1.0,
        peak_hartmann_number=14.0,
        entry_center=1.2,
        exit_center=4.1,
        transition_width=0.35,
    )
    gradients = fringing_response_loss_gradients(
        problem,
        forcing=1.0,
        peak_hartmann_number=10.0,
        entry_center=0.8,
        exit_center=4.8,
        transition_width=0.6,
        target_mean_velocity=target["mean_velocity"],
        target_current_proxy=target["current_proxy"],
        current_weight=0.5,
    )
    assert jnp.isfinite(gradients["loss"])
    assert jnp.isfinite(gradients["d_peak_hartmann_number"])
    assert jnp.isfinite(gradients["d_entry_center"])
    assert jnp.isfinite(gradients["d_exit_center"])
    assert jnp.isfinite(gradients["d_transition_width"])


def test_fringing_inverse_design_reduces_loss():
    problem = build_fringing_autodiff_problem(nx_stations=9, ny=10, nz=10, macro_iterations=2, potential_iterations=10, velocity_iterations=12)
    target = fringing_mean_velocity_history(
        problem,
        forcing=1.0,
        peak_hartmann_number=14.0,
        entry_center=1.2,
        exit_center=4.1,
        transition_width=0.35,
    )
    result = run_fringing_history_inverse_design(
        problem,
        target_mean_velocity=target["mean_velocity"],
        forcing=1.0,
        peak_hartmann_init=8.0,
        entry_center_init=0.7,
        exit_center_init=5.0,
        transition_width_init=0.7,
        steps=6,
    )

    assert len(result["history"]) == 6
    assert result["history"][-1]["loss"] <= result["history"][0]["loss"]
    assert jnp.isfinite(result["recovered_mean_velocity"]).all()


def test_fringing_response_inverse_design_reduces_loss():
    problem = build_fringing_autodiff_problem(nx_stations=9, ny=10, nz=10, macro_iterations=2, potential_iterations=10, velocity_iterations=12)
    target = fringing_response_history(
        problem,
        forcing=1.0,
        peak_hartmann_number=14.0,
        entry_center=1.2,
        exit_center=4.1,
        transition_width=0.35,
    )
    result = run_fringing_response_inverse_design(
        problem,
        target_mean_velocity=target["mean_velocity"],
        target_current_proxy=target["current_proxy"],
        forcing=1.0,
        peak_hartmann_init=8.0,
        entry_center_init=0.7,
        exit_center_init=5.0,
        transition_width_init=0.7,
        current_weight=0.5,
        steps=6,
    )
    assert len(result["history"]) == 6
    assert result["history"][-1]["loss"] <= result["history"][0]["loss"]
    assert jnp.isfinite(result["recovered_current_proxy"]).all()


def test_build_extruded_response_targets_returns_finite_histories():
    problem = build_square_duct_extruded_problem(ha_peak=6.0, nx_stations=4, ny=4, nz=4)
    solution = solve_extruded_inductionless(problem)

    targets = build_extruded_response_targets(solution)

    assert targets["x"].shape == (4,)
    assert targets["mean_velocity"].shape == (4,)
    assert targets["current_proxy"].shape == (4,)
    assert targets["charge_balance_residual"].shape == (4,)
    assert targets["wall_current_leakage"].shape == (4,)
    assert targets["axial_current"].shape == (4,)
    assert jnp.isfinite(targets["mean_velocity"]).all()


def test_extruded_rect_response_history_returns_finite_trace():
    problem = build_fringing_autodiff_problem(
        nx_stations=5,
        ny=6,
        nz=6,
        macro_iterations=2,
        potential_iterations=8,
        velocity_iterations=10,
    )
    payload = extruded_rect_response_history(
        problem,
        forcing=1.0,
        peak_hartmann_number=8.0,
        entry_center=1.0,
        exit_center=4.0,
        transition_width=0.5,
    )
    assert payload["mean_velocity"].shape == (5,)
    assert payload["current_proxy"].shape == (5,)
    assert payload["charge_balance_residual"].shape == (5,)
    assert jnp.isfinite(payload["mean_velocity"]).all()


def test_extruded_rect_response_loss_gradients_are_finite():
    problem = build_fringing_autodiff_problem(
        nx_stations=5,
        ny=6,
        nz=6,
        macro_iterations=2,
        potential_iterations=8,
        velocity_iterations=10,
    )
    target = extruded_rect_response_history(
        problem,
        forcing=1.0,
        peak_hartmann_number=9.0,
        entry_center=1.1,
        exit_center=4.1,
        transition_width=0.45,
    )
    gradients = extruded_rect_response_loss_gradients(
        problem,
        forcing=1.0,
        peak_hartmann_number=7.0,
        entry_center=0.8,
        exit_center=4.8,
        transition_width=0.7,
        target_mean_velocity=target["mean_velocity"],
        target_current_proxy=target["current_proxy"],
        target_charge_balance=target["charge_balance_residual"],
        current_weight=0.5,
        charge_balance_weight=0.1,
    )
    assert jnp.isfinite(gradients["loss"])
    assert jnp.isfinite(gradients["d_peak_hartmann_number"])


def test_extruded_rect_inverse_design_reduces_loss():
    problem = build_fringing_autodiff_problem(
        nx_stations=5,
        ny=6,
        nz=6,
        macro_iterations=2,
        potential_iterations=8,
        velocity_iterations=10,
    )
    target = extruded_rect_response_history(
        problem,
        forcing=1.0,
        peak_hartmann_number=9.0,
        entry_center=1.2,
        exit_center=4.0,
        transition_width=0.4,
    )
    result = run_extruded_rect_inverse_design(
        problem,
        target_mean_velocity=target["mean_velocity"],
        target_current_proxy=target["current_proxy"],
        target_charge_balance=target["charge_balance_residual"],
        forcing=1.0,
        peak_hartmann_init=6.0,
        entry_center_init=0.8,
        exit_center_init=4.8,
        transition_width_init=0.7,
        current_weight=0.5,
        steps=4,
    )
    assert result["model"] == "direct_extruded_rect"
    assert result["history"][-1]["loss"] <= result["history"][0]["loss"]


def test_extruded_target_inverse_design_returns_finite_payload():
    problem = build_square_duct_extruded_problem(ha_peak=6.0, nx_stations=4, ny=4, nz=4)
    solution = solve_extruded_inductionless(problem)

    result = run_extruded_target_inverse_design(solution, ny=6, nz=6, steps=4)

    assert result["geometry_kind"] == "rect_duct"
    assert "target" in result
    assert "recovered" in result
    assert len(result["recovered"]["history"]) == 4
    assert result["recovered"]["model"] == "direct_extruded_rect"
    assert jnp.isfinite(result["target"]["mean_velocity"]).all()

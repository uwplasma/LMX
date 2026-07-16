import jax
import jax.numpy as jnp
import importlib.util
import pytest

import lmx.autodiff as autodiff
from lmx.autodiff import (
    build_extruded_response_targets,
    build_fringing_autodiff_problem,
    build_hartmann_autodiff_problem,
    extruded_rect_projection_history,
    extruded_rect_projection_iteration_history,
    extruded_rect_projection_field_loss_gradients,
    extruded_rect_projection_loss_gradients,
    extruded_rect_projection_trajectory_loss_gradients,
    fringing_history_loss_gradients,
    fringing_mean_velocity_history,
    fringing_response_history,
    fringing_response_loss_gradients,
    hartmann_mean_velocity,
    hartmann_mean_velocity_finite_difference_gradients,
    hartmann_mean_velocity_gradients,
    hartmann_profile_loss,
    hartmann_profile_loss_gradients,
    run_extruded_rect_projection_field_inverse_design,
    run_extruded_rect_projection_inverse_design,
    run_extruded_rect_projection_trajectory_inverse_design,
    run_fringing_response_inverse_design,
    run_extruded_target_inverse_design,
    run_fringing_history_inverse_design,
    run_hartmann_profile_inverse_design,
    solve_differentiable_hartmann,
    wham_mirror_pressure_drop_sensitivity,
)
from lmx.fringing import (
    build_pipe_ogrid_extruded_problem,
    build_square_duct_extruded_problem,
    solve_extruded_inductionless,
)


pytestmark = pytest.mark.unit


def _skip_without_magpylib_jax() -> None:
    if importlib.util.find_spec("magpylib_jax") is None:
        pytest.skip(
            "magpylib_jax is optional and is not installed in the bounded CI environment"
        )


@pytest.fixture(scope="module")
def hartmann_problem():
    return build_hartmann_autodiff_problem(
        ny=12, nz=12, macro_iterations=3, potential_iterations=12, velocity_iterations=16
    )


@pytest.fixture(scope="module")
def fringing_problem():
    return build_fringing_autodiff_problem(
        nx_stations=9, ny=10, nz=10, macro_iterations=2,
        potential_iterations=10, velocity_iterations=12,
    )


@pytest.fixture(scope="module")
def fringing_mean_target(fringing_problem):
    return fringing_mean_velocity_history(
        fringing_problem, forcing=1.0, peak_hartmann_number=14.0,
        entry_center=1.2, exit_center=4.1, transition_width=0.35,
    )


@pytest.fixture(scope="module")
def projection_problem():
    return build_fringing_autodiff_problem(
        nx_stations=5, ny=6, nz=6, macro_iterations=2,
        potential_iterations=8, velocity_iterations=10,
    )


@pytest.fixture(scope="module")
def projection_target(projection_problem):
    return extruded_rect_projection_history(
        projection_problem, forcing=1.0, peak_hartmann_number=9.0,
        entry_center=1.2, exit_center=4.0, transition_width=0.4,
    )


@pytest.fixture(scope="module")
def trajectory_problem():
    return build_fringing_autodiff_problem(
        nx_stations=5, ny=6, nz=6, macro_iterations=3,
        potential_iterations=8, velocity_iterations=10,
    )


@pytest.fixture(scope="module")
def trajectory_target(trajectory_problem):
    indices = jnp.asarray([1, 3])
    return indices, extruded_rect_projection_iteration_history(
        trajectory_problem, forcing=1.0, peak_hartmann_number=9.0,
        entry_center=1.2, exit_center=4.0, transition_width=0.4,
        station_indices=indices,
    )


def test_shared_fringing_descent_enforces_parameter_bounds():
    problem = build_fringing_autodiff_problem(nx_stations=3, ny=4, nz=4)

    def gradients(_, **kwargs):
        del kwargs
        return {
            "loss": jnp.asarray(1.0),
            "d_peak_hartmann_number": jnp.asarray(100.0),
            "d_entry_center": jnp.asarray(100.0),
            "d_exit_center": jnp.asarray(-100.0),
            "d_transition_width": jnp.asarray(100.0),
        }

    parameters, history = autodiff._run_fringing_parameter_descent(
        problem,
        gradient_function=gradients,
        gradient_kwargs={},
        initial=(5.0, 1.0, 2.0, 0.5),
        learning_rates=(1.0, 1.0, 1.0, 1.0),
        steps=1,
    )
    assert [float(value) for value in parameters] == pytest.approx([0.5, 0.0, 6.0, 0.05])
    assert history[0]["loss"] == 1.0


def test_differentiable_hartmann_solution_returns_finite_fields(hartmann_problem):
    u, phi = solve_differentiable_hartmann(hartmann_problem, forcing=1.0, hartmann_number=5.0)

    assert u.shape == (12, 12)
    assert phi.shape == (12, 12)
    assert jnp.isfinite(u).all()
    assert jnp.isfinite(phi).all()


def test_hartmann_mean_velocity_is_differentiable(hartmann_problem):
    value, gradient = jax.value_and_grad(
        lambda ha: hartmann_mean_velocity(hartmann_problem, forcing=1.0, hartmann_number=ha)
    )(5.0)

    assert jnp.isfinite(value)
    assert jnp.isfinite(gradient)


def test_profile_loss_gradient_step_reduces_objective(hartmann_problem):
    target_u, _ = solve_differentiable_hartmann(
        hartmann_problem, forcing=1.0, hartmann_number=9.0
    )
    target_profile = target_u[:, target_u.shape[1] // 2]

    objective = lambda ha: hartmann_profile_loss(
        hartmann_problem, forcing=1.0, hartmann_number=ha, target_profile=target_profile
    )
    loss0, grad0 = jax.value_and_grad(objective)(4.0)
    loss1 = objective(jnp.clip(4.0 - 2.0 * grad0, 0.5, 30.0))

    assert jnp.isfinite(loss0)
    assert jnp.isfinite(grad0)
    assert float(loss1) <= float(loss0)


def test_mean_velocity_gradients_match_finite_difference(hartmann_problem):
    autodiff = hartmann_mean_velocity_gradients(
        hartmann_problem, forcing=1.1, hartmann_number=5.0
    )
    finite_diff = hartmann_mean_velocity_finite_difference_gradients(
        hartmann_problem, forcing=1.1, hartmann_number=5.0
    )

    assert jnp.isfinite(autodiff["d_mean_velocity_d_forcing"])
    assert jnp.isfinite(autodiff["d_mean_velocity_d_ha"])
    assert (
        float(
            jnp.abs(
                autodiff["d_mean_velocity_d_forcing"]
                - finite_diff["d_mean_velocity_d_forcing"]
            )
        )
        < 5.0e-2
    )
    assert (
        float(
            jnp.abs(
                autodiff["d_mean_velocity_d_ha"] - finite_diff["d_mean_velocity_d_ha"]
            )
        )
        < 5.0e-2
    )


def test_profile_loss_gradients_are_finite(hartmann_problem):
    target_u, _ = solve_differentiable_hartmann(
        hartmann_problem, forcing=1.0, hartmann_number=8.0
    )
    target_profile = target_u[:, target_u.shape[1] // 2]

    gradients = hartmann_profile_loss_gradients(
        hartmann_problem,
        forcing=0.7,
        hartmann_number=4.5,
        target_profile=target_profile,
    )

    assert jnp.isfinite(gradients["loss"])
    assert jnp.isfinite(gradients["d_loss_d_forcing"])
    assert jnp.isfinite(gradients["d_loss_d_ha"])


def test_profile_inverse_design_reduces_loss(hartmann_problem):
    target_u, _ = solve_differentiable_hartmann(
        hartmann_problem, forcing=1.0, hartmann_number=10.0
    )
    target_profile = target_u[:, target_u.shape[1] // 2]

    result = run_hartmann_profile_inverse_design(
        hartmann_problem,
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


def test_fringing_mean_velocity_history_returns_finite_trace(fringing_mean_target):
    payload = fringing_mean_target

    assert payload["x"].shape == (9,)
    assert payload["field_scale"].shape == (9,)
    assert payload["mean_velocity"].shape == (9,)
    assert jnp.isfinite(payload["mean_velocity"]).all()


def test_fringing_history_loss_gradients_are_finite(fringing_problem, fringing_mean_target):
    gradients = fringing_history_loss_gradients(
        fringing_problem,
        forcing=1.0,
        peak_hartmann_number=10.0,
        entry_center=0.8,
        exit_center=4.8,
        transition_width=0.6,
        target_mean_velocity=fringing_mean_target["mean_velocity"],
    )

    assert jnp.isfinite(gradients["loss"])
    assert jnp.isfinite(gradients["d_peak_hartmann_number"])
    assert jnp.isfinite(gradients["d_entry_center"])
    assert jnp.isfinite(gradients["d_exit_center"])
    assert jnp.isfinite(gradients["d_transition_width"])


def test_fringing_response_history_returns_finite_current_proxy(fringing_problem):
    payload = fringing_response_history(
        fringing_problem,
        forcing=1.0,
        peak_hartmann_number=12.0,
        entry_center=1.0,
        exit_center=4.0,
        transition_width=0.4,
    )
    assert payload["current_proxy"].shape == (9,)
    assert jnp.isfinite(payload["current_proxy"]).all()


def test_fringing_response_loss_gradients_are_finite(fringing_problem):
    target = fringing_response_history(
        fringing_problem,
        forcing=1.0,
        peak_hartmann_number=14.0,
        entry_center=1.2,
        exit_center=4.1,
        transition_width=0.35,
    )
    gradients = fringing_response_loss_gradients(
        fringing_problem,
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


def test_fringing_inverse_design_reduces_loss(fringing_problem, fringing_mean_target):
    result = run_fringing_history_inverse_design(
        fringing_problem,
        target_mean_velocity=fringing_mean_target["mean_velocity"],
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
    problem = build_fringing_autodiff_problem(
        nx_stations=7,
        ny=8,
        nz=8,
        macro_iterations=2,
        potential_iterations=8,
        velocity_iterations=10,
    )
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
        steps=4,
    )
    assert len(result["history"]) == 4
    assert result["history"][-1]["loss"] <= result["history"][0]["loss"]


def test_wham_mirror_pressure_drop_sensitivity_matches_finite_difference():
    _skip_without_magpylib_jax()
    problem = build_fringing_autodiff_problem(
        nx_stations=9,
        length=1.2,
        ny=8,
        nz=8,
        macro_iterations=2,
        potential_iterations=8,
        velocity_iterations=10,
    )
    sensitivity = wham_mirror_pressure_drop_sensitivity(
        problem,
        forcing=1.0,
        peak_hartmann_number=10.0,
        coil_separation=1.5,
        radial_loops=4,
        axial_loops=2,
    )
    plus = wham_mirror_pressure_drop_sensitivity(
        problem,
        forcing=1.0,
        peak_hartmann_number=10.0,
        coil_separation=1.5 + 1.0e-2,
        radial_loops=4,
        axial_loops=2,
    )
    minus = wham_mirror_pressure_drop_sensitivity(
        problem,
        forcing=1.0,
        peak_hartmann_number=10.0,
        coil_separation=1.5 - 1.0e-2,
        radial_loops=4,
        axial_loops=2,
    )
    finite_difference = (
        plus["pressure_drop_proxy"] - minus["pressure_drop_proxy"]
    ) / 2.0e-2
    assert jnp.isfinite(sensitivity["pressure_drop_proxy"])
    assert jnp.isfinite(sensitivity["d_pressure_drop_d_separation"])
    assert (
        float(jnp.abs(sensitivity["d_pressure_drop_d_separation"] - finite_difference))
        < 5.0e-2
    )


def test_build_extruded_response_targets_returns_finite_histories():
    problem = build_square_duct_extruded_problem(ha_peak=6.0, nx_stations=4, ny=4, nz=4)
    solution = solve_extruded_inductionless(problem)

    targets = build_extruded_response_targets(solution)

    assert targets["x"].shape == (4,)
    assert targets["mean_velocity"].shape == (4,)
    assert targets["current_proxy"].shape == (4,)
    assert targets["charge_balance_residual"].shape == (4,)
    assert targets["boundary_current_residual"].shape == (4,)
    assert targets["wall_current_leakage"].shape == (4,)
    assert targets["axial_current"].shape == (4,)
    assert targets["pressure_span"].shape == (4,)
    assert targets["transverse_kinetic_energy"].shape == (4,)
    assert targets["u_field"].shape == (4, 4, 4)
    assert targets["phi_field"].shape == (4, 4, 4)
    assert targets["jy_field"].shape == (4, 4, 4)
    assert targets["pressure_field"].shape == (4, 4, 4)
    assert jnp.isfinite(targets["mean_velocity"]).all()


def test_extruded_rect_projection_history_returns_finite_trace(projection_target):
    payload = projection_target
    assert payload["pressure_span"].shape == (5,)
    assert payload["transverse_kinetic_energy"].shape == (5,)
    assert payload["boundary_current_residual"].shape == (5,)
    assert payload["u_field"].shape == (5, 6, 6)
    assert payload["phi_field"].shape == (5, 6, 6)
    assert payload["jy_field"].shape == (5, 6, 6)
    assert payload["pressure_field"].shape == (5, 6, 6)
    assert jnp.isfinite(payload["pressure_span"]).all()


def test_extruded_rect_projection_loss_gradients_are_finite(
    projection_problem, projection_target
):
    gradients = extruded_rect_projection_loss_gradients(
        projection_problem,
        forcing=1.0,
        peak_hartmann_number=7.0,
        entry_center=0.8,
        exit_center=4.8,
        transition_width=0.7,
        target_mean_velocity=projection_target["mean_velocity"],
        target_current_proxy=projection_target["current_proxy"],
        target_charge_balance=projection_target["charge_balance_residual"],
        target_boundary_current=projection_target["boundary_current_residual"],
        target_pressure_span=projection_target["pressure_span"],
        current_weight=0.5,
        pressure_span_weight=0.2,
    )
    assert jnp.isfinite(gradients["loss"])
    assert jnp.isfinite(gradients["d_peak_hartmann_number"])


def test_extruded_rect_projection_field_loss_gradients_are_finite(
    projection_problem, projection_target
):
    gradients = extruded_rect_projection_field_loss_gradients(
        projection_problem,
        forcing=1.0,
        peak_hartmann_number=7.0,
        entry_center=0.8,
        exit_center=4.8,
        transition_width=0.7,
        target_u_field=projection_target["u_field"][jnp.asarray([1, 3])],
        target_phi_field=projection_target["phi_field"][jnp.asarray([1, 3])],
        target_jy_field=projection_target["jy_field"][jnp.asarray([1, 3])],
        target_pressure_field=projection_target["pressure_field"][jnp.asarray([1, 3])],
        station_indices=jnp.asarray([1, 3]),
    )
    assert jnp.isfinite(gradients["loss"])
    assert jnp.isfinite(gradients["d_peak_hartmann_number"])


def test_extruded_target_inverse_design_returns_finite_payload():
    problem = build_square_duct_extruded_problem(ha_peak=6.0, nx_stations=4, ny=4, nz=4)
    solution = solve_extruded_inductionless(problem)

    result = run_extruded_target_inverse_design(solution, ny=6, nz=6, steps=1)

    assert result["geometry_kind"] == "rect_duct"
    assert "target" in result
    assert "recovered" in result
    assert len(result["recovered"]["history"]) == 1
    assert result["recovered"]["model"] == "direct_extruded_projection"
    assert jnp.isfinite(result["target"]["mean_velocity"]).all()


def test_extruded_target_inverse_design_uses_surrogate_for_nonrect_geometry():
    problem = build_pipe_ogrid_extruded_problem(
        ha_peak=6.0, nx_stations=4, nr=4, ntheta=8
    )
    solution = solve_extruded_inductionless(problem)

    result = run_extruded_target_inverse_design(solution, ny=4, nz=4, steps=1)

    assert result["geometry_kind"] == "pipe_ogrid"
    assert result["recovered"]["model"] == "fringing_response_surrogate"


def test_extruded_rect_projection_inverse_design_reduces_loss(
    projection_problem, projection_target
):
    result = run_extruded_rect_projection_inverse_design(
        projection_problem,
        target_mean_velocity=projection_target["mean_velocity"],
        target_current_proxy=projection_target["current_proxy"],
        target_charge_balance=projection_target["charge_balance_residual"],
        target_boundary_current=projection_target["boundary_current_residual"],
        target_pressure_span=projection_target["pressure_span"],
        forcing=1.0,
        peak_hartmann_init=6.0,
        entry_center_init=0.8,
        exit_center_init=4.8,
        transition_width_init=0.7,
        current_weight=0.5,
        pressure_span_weight=0.2,
        steps=4,
    )
    assert result["model"] == "direct_extruded_projection"
    assert result["history"][-1]["loss"] <= result["history"][0]["loss"]


def test_extruded_rect_projection_field_inverse_design_reduces_loss(
    projection_problem, projection_target
):
    result = run_extruded_rect_projection_field_inverse_design(
        projection_problem,
        target_u_field=projection_target["u_field"][jnp.asarray([1, 3])],
        target_phi_field=projection_target["phi_field"][jnp.asarray([1, 3])],
        target_jy_field=projection_target["jy_field"][jnp.asarray([1, 3])],
        target_pressure_field=projection_target["pressure_field"][jnp.asarray([1, 3])],
        station_indices=jnp.asarray([1, 3]),
        forcing=1.0,
        peak_hartmann_init=6.0,
        entry_center_init=0.8,
        exit_center_init=4.8,
        transition_width_init=0.7,
        steps=4,
    )
    assert result["model"] == "direct_extruded_projection_fields"
    assert result["history"][-1]["loss"] <= result["history"][0]["loss"]


def test_extruded_rect_projection_iteration_history_and_gradients_are_finite(
    trajectory_problem, trajectory_target
):
    station_indices, target = trajectory_target
    assert target["u_field"].shape[0] == 3
    assert target["u_field"].shape[1] == 2
    terminal = extruded_rect_projection_history(
        trajectory_problem, forcing=1.0, peak_hartmann_number=9.0,
        entry_center=1.2, exit_center=4.0, transition_width=0.4,
    )
    for name in (
        "u_field", "phi_field", "jy_field", "pressure_field",
        "charge_balance_residual", "boundary_current_residual",
    ):
        assert jnp.allclose(
            target[name][-1], terminal[name][station_indices], rtol=0.0, atol=1.0e-11
        )
    gradients = extruded_rect_projection_trajectory_loss_gradients(
        trajectory_problem,
        forcing=1.0,
        peak_hartmann_number=7.0,
        entry_center=0.8,
        exit_center=4.8,
        transition_width=0.7,
        target_u_history=target["u_field"],
        target_phi_history=target["phi_field"],
        target_jy_history=target["jy_field"],
        target_pressure_history=target["pressure_field"],
        target_charge_balance_history=target["charge_balance_residual"],
        target_boundary_current_history=target["boundary_current_residual"],
        station_indices=station_indices,
    )
    assert jnp.isfinite(gradients["loss"])
    assert jnp.isfinite(gradients["d_peak_hartmann_number"])


def test_extruded_rect_projection_trajectory_inverse_design_reduces_loss(
    trajectory_problem, trajectory_target
):
    station_indices, target = trajectory_target
    result = run_extruded_rect_projection_trajectory_inverse_design(
        trajectory_problem,
        target_u_history=target["u_field"],
        target_phi_history=target["phi_field"],
        target_jy_history=target["jy_field"],
        target_pressure_history=target["pressure_field"],
        target_charge_balance_history=target["charge_balance_residual"],
        target_boundary_current_history=target["boundary_current_residual"],
        station_indices=station_indices,
        forcing=1.0,
        peak_hartmann_init=6.0,
        entry_center_init=0.8,
        exit_center_init=4.8,
        transition_width_init=0.7,
        steps=4,
    )
    assert result["model"] == "direct_extruded_projection_trajectory"
    assert result["history"][-1]["loss"] <= result["history"][0]["loss"]

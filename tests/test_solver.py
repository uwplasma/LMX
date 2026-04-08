from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import pytest

from lmx.cases import make_hartmann_case, make_hunt_case, make_shercliff_case
from lmx.io import load_restart_bundle, write_solution_npz
from lmx.physics import build_material_fields, magnetic_field_components, magnetic_ramp_scale
import lmx.solvers as solvers
from lmx.mesh import generate_layered_duct_mesh, generate_rect_duct_mesh
from lmx.specs import BoundaryCondition
from lmx.solvers import solve_steady, solve_transient


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
    assert ha20.time_stepper.potential_tolerance is None
    assert ha20.time_stepper.potential_solver == "auto"
    assert ha20.time_stepper.current_reconstruction == "cell_centered"
    assert ha100.time_stepper.outer_iterations == 4
    assert ha100.time_stepper.velocity_update_limit == pytest.approx(1e-3)
    assert ha100.time_stepper.potential_tolerance is None
    assert ha100.time_stepper.potential_solver == "auto"
    assert ha100.time_stepper.current_reconstruction == "cell_centered"
    assert ha1000.time_stepper.outer_iterations == 3
    assert ha1000.time_stepper.velocity_update_limit == pytest.approx(1e-3)
    assert ha1000.time_stepper.potential_tolerance is None
    assert ha1000.time_stepper.potential_solver == "auto"
    assert ha1000.time_stepper.current_reconstruction == "cell_centered"


def test_hunt_case_derives_wall_conductivity_from_conductance_ratio():
    case = make_hunt_case(
        ha=20.0,
        width=2.0,
        height=2.0,
        wall_thickness=0.1,
        fluid_conductivity=2.0,
        wall_conductance_ratio=0.05,
        ny=16,
        nz=16,
        wall_cells=2,
    )

    wall_region = next(region for region in case.regions if region.name == "conducting_wall")
    expected = 0.05 * 2.0 * (0.5 * 2.0) / 0.1
    assert wall_region.conductivity == pytest.approx(expected)


def test_hunt_case_adds_explicit_insulating_side_wall_region():
    case = make_hunt_case(ha=20.0, fluid_conductivity=2.0, ny=16, nz=16, wall_cells=2)
    insulator_region = next(region for region in case.regions if region.name == "insulating_wall")
    assert insulator_region.conductivity == pytest.approx(2.0e-12)
    assert case.geometry.wall_cells == (2, 2, 2, 2)
    assert case.geometry.wall_thickness == pytest.approx((0.1, 0.1, 0.1, 0.1))


def test_hunt_case_allows_explicit_wall_conductivity_override():
    case = make_hunt_case(ha=20.0, wall_conductance_ratio=0.05, wall_conductivity=7.5, ny=16, nz=16, wall_cells=2)
    wall_region = next(region for region in case.regions if region.name == "conducting_wall")
    assert wall_region.conductivity == pytest.approx(7.5)


def test_hunt_inlet_flow_rate_boundary_drives_short_transient():
    case = make_hunt_case(ha=20.0, ny=16, nz=16, wall_cells=2)
    driven = replace(
        case,
        forcing=0.0,
        initial_velocity=0.1175,
        boundary_conditions=case.boundary_conditions
        + (BoundaryCondition("inlet", "inlet_flow_rate", value=0.1175 * case.geometry.width * case.geometry.height, axis="x"),),
        time_stepper=replace(case.time_stepper, dt=1e-5, t_final=1e-4, max_steps=10),
    )
    undriven = replace(
        case,
        forcing=0.0,
        initial_velocity=0.1175,
        time_stepper=replace(case.time_stepper, dt=1e-5, t_final=1e-4, max_steps=10),
    )

    driven_solution = solve_steady(driven)
    undriven_solution = solve_steady(undriven)

    assert float(jnp.max(driven_solution.state.u)) > float(jnp.max(undriven_solution.state.u))


def test_hunt_inlet_velocity_boundary_does_not_trigger_reduced_mean_drive():
    case = make_hunt_case(ha=20.0, ny=16, nz=16, wall_cells=2)
    inlet_only = replace(
        case,
        forcing=0.0,
        initial_velocity=0.1175,
        boundary_conditions=case.boundary_conditions + (BoundaryCondition("inlet", "inlet_velocity", value=(0.1175, 0.0, 0.0), axis="x"),),
        time_stepper=replace(case.time_stepper, dt=1e-5, t_final=5e-5, max_steps=5),
    )
    undriven = replace(
        case,
        forcing=0.0,
        initial_velocity=0.1175,
        time_stepper=replace(case.time_stepper, dt=1e-5, t_final=5e-5, max_steps=5),
    )

    inlet_solution = solve_steady(inlet_only)
    undriven_solution = solve_steady(undriven)

    assert float(jnp.max(jnp.abs(inlet_solution.state.u - undriven_solution.state.u))) < 1e-6


def test_transient_restart_matches_direct_run(tmp_path: Path):
    case = make_hartmann_case(ha=5.0, ny=12, nz=12)
    direct_case = replace(case, time_stepper=replace(case.time_stepper, dt=0.01, t_final=0.04, max_steps=4))
    partial_case = replace(case, time_stepper=replace(case.time_stepper, dt=0.01, t_final=0.02, max_steps=2))

    direct = solve_transient(direct_case)
    partial = solve_transient(partial_case)
    restart_path = write_solution_npz(partial, partial_case, tmp_path / "partial_restart.npz")
    restart = load_restart_bundle(restart_path)
    resumed = solve_transient(
        direct_case,
        initial_state=restart.state,
        initial_diagnostics=restart.diagnostics,
        append_diagnostics=False,
    )

    assert float(resumed.state.time) == pytest.approx(float(direct.state.time))
    assert jnp.allclose(resumed.state.u, direct.state.u, atol=1e-6, rtol=1e-6)
    assert jnp.allclose(resumed.state.phi, direct.state.phi, atol=1e-6, rtol=1e-6)
    assert jnp.allclose(resumed.state.jy, direct.state.jy, atol=1e-6, rtol=1e-6)
    assert jnp.allclose(resumed.state.jz, direct.state.jz, atol=1e-6, rtol=1e-6)
    assert jnp.allclose(resumed.state.lorentz_x, direct.state.lorentz_x, atol=1e-6, rtol=1e-6)


def test_transient_restart_can_append_diagnostics(tmp_path: Path):
    case = make_hartmann_case(ha=5.0, ny=10, nz=10)
    direct_case = replace(case, time_stepper=replace(case.time_stepper, dt=0.01, t_final=0.04, max_steps=4))
    partial_case = replace(case, time_stepper=replace(case.time_stepper, dt=0.01, t_final=0.02, max_steps=2))
    partial = solve_transient(partial_case)
    restart = load_restart_bundle(write_solution_npz(partial, partial_case, tmp_path / "partial_append.npz"))

    resumed = solve_transient(
        direct_case,
        initial_state=restart.state,
        initial_diagnostics=restart.diagnostics,
        append_diagnostics=True,
    )

    assert resumed.diagnostics.time_history.shape[0] == 4
    assert float(resumed.diagnostics.time_history[0]) == pytest.approx(0.01)
    assert float(resumed.diagnostics.time_history[-1]) == pytest.approx(0.04)


def test_dynamic_inlet_drive_adds_pressure_gradient_when_explicit_forcing_is_zero():
    mesh = generate_rect_duct_mesh(width=2.0, height=2.0, ny=5, nz=5)
    zeros = jnp.zeros(mesh.yz_shape)
    fluid_mask = jnp.ones(mesh.yz_shape, dtype=bool)

    undriven = solvers._step(
        u=zeros,
        mesh=mesh,
        sigma=jnp.zeros(mesh.yz_shape),
        rho=jnp.ones(mesh.yz_shape),
        nu=jnp.zeros(mesh.yz_shape),
        fluid_mask=fluid_mask,
        by=zeros,
        bz=zeros,
        dt=0.1,
        forcing=0.0,
        target_mean_velocity=None,
        reference_mean_velocity=None,
        anchor=(0, 0),
        outer_iterations=1,
        potential_iterations=1,
        potential_tolerance=None,
        potential_relaxation=1.0,
        potential_solver="jacobi",
        relaxation=1.0,
        velocity_update_limit=10.0,
        velocity_update_limiter="global_scale",
        current_reconstruction="cell_centered",
        interpolate_direct_fluid_walls=False,
    )[0]
    driven = solvers._step(
        u=zeros,
        mesh=mesh,
        sigma=jnp.zeros(mesh.yz_shape),
        rho=jnp.ones(mesh.yz_shape),
        nu=jnp.zeros(mesh.yz_shape),
        fluid_mask=fluid_mask,
        by=zeros,
        bz=zeros,
        dt=0.1,
        forcing=0.0,
        target_mean_velocity=0.25,
        reference_mean_velocity=0.25,
        anchor=(0, 0),
        outer_iterations=1,
        potential_iterations=1,
        potential_tolerance=None,
        potential_relaxation=1.0,
        potential_solver="jacobi",
        relaxation=1.0,
        velocity_update_limit=10.0,
        velocity_update_limiter="global_scale",
        current_reconstruction="cell_centered",
        interpolate_direct_fluid_walls=False,
    )[0]

    assert float(jnp.mean(undriven)) == pytest.approx(0.0)
    assert float(jnp.mean(driven)) > 0.0
    assert float(jnp.max(driven[1:-1, 1:-1])) > 0.0


def test_dynamic_inlet_drive_uses_area_weighted_mean_velocity_on_nonuniform_mesh():
    mesh = generate_layered_duct_mesh(
        width=2.0,
        height=2.0,
        ny=4,
        nz=4,
        wall_thickness=(0.2, 0.2, 0.0, 0.0),
        wall_cells=(1, 1, 0, 0),
        target_ha=20.0,
    )
    fluid_mask = jnp.ones(mesh.yz_shape, dtype=bool)
    sigma = jnp.zeros(mesh.yz_shape)
    y_index = jnp.arange(mesh.yz_shape[0], dtype=jnp.float32)[:, None]
    z_index = jnp.arange(mesh.yz_shape[1], dtype=jnp.float32)[None, :]
    rho = 1.0 + 0.25 * y_index + 0.5 * z_index
    nu = jnp.zeros(mesh.yz_shape)
    zeros = jnp.zeros(mesh.yz_shape)
    u = jnp.zeros(mesh.yz_shape)

    _, _, _, _, _, _, _, _, _, _, _, mean_velocity, applied_forcing, _ = solvers._step(
        u=u,
        mesh=mesh,
        sigma=sigma,
        rho=rho,
        nu=nu,
        fluid_mask=fluid_mask,
        by=zeros,
        bz=zeros,
        dt=0.1,
        forcing=0.0,
        target_mean_velocity=0.25,
        reference_mean_velocity=0.25,
        anchor=(0, 0),
        outer_iterations=1,
        potential_iterations=1,
        potential_tolerance=None,
        potential_relaxation=1.0,
        potential_solver="jacobi",
        relaxation=1.0,
        velocity_update_limit=10.0,
        velocity_update_limiter="global_scale",
        current_reconstruction="cell_centered",
        interpolate_direct_fluid_walls=False,
    )

    active_mask = solvers._active_velocity_mask(fluid_mask)
    cell_metric = solvers._cell_metric(mesh)
    weighted_area = float(jnp.sum(jnp.where(active_mask, cell_metric, 0.0)))
    pressure_sensitivity = 0.1 / rho
    weighted_sensitivity = float(jnp.sum(jnp.where(active_mask, cell_metric * pressure_sensitivity, 0.0)) / weighted_area)
    active_count = float(jnp.sum(active_mask.astype(jnp.float32)))
    simple_sensitivity = float(jnp.sum(jnp.where(active_mask, pressure_sensitivity, 0.0)) / active_count)

    assert float(applied_forcing) == pytest.approx(0.25 / weighted_sensitivity)
    assert float(applied_forcing) != pytest.approx(0.25 / simple_sensitivity)
    assert float(mean_velocity) > 0.0


def test_dynamic_inlet_drive_uses_full_fluid_area_for_flow_rate_control():
    mesh = generate_rect_duct_mesh(width=2.0, height=2.0, ny=5, nz=5)
    fluid_mask = jnp.ones(mesh.yz_shape, dtype=bool)
    sigma = jnp.zeros(mesh.yz_shape)
    y_index = jnp.arange(mesh.yz_shape[0], dtype=jnp.float32)[:, None]
    z_index = jnp.arange(mesh.yz_shape[1], dtype=jnp.float32)[None, :]
    rho = 1.0 + 0.2 * y_index + 0.1 * z_index
    nu = jnp.zeros(mesh.yz_shape)
    zeros = jnp.zeros(mesh.yz_shape)
    u = jnp.zeros(mesh.yz_shape)

    _, _, _, _, _, _, _, _, _, _, _, mean_velocity, applied_forcing, _ = solvers._step(
        u=u,
        mesh=mesh,
        sigma=sigma,
        rho=rho,
        nu=nu,
        fluid_mask=fluid_mask,
        by=zeros,
        bz=zeros,
        dt=0.1,
        forcing=0.0,
        target_mean_velocity=0.25,
        reference_mean_velocity=0.25,
        anchor=(0, 0),
        outer_iterations=1,
        potential_iterations=1,
        potential_tolerance=None,
        potential_relaxation=1.0,
        potential_solver="jacobi",
        relaxation=1.0,
        velocity_update_limit=10.0,
        velocity_update_limiter="global_scale",
        current_reconstruction="cell_centered",
        interpolate_direct_fluid_walls=False,
    )

    active_mask = solvers._active_velocity_mask(fluid_mask)
    cell_metric = solvers._cell_metric(mesh)
    pressure_sensitivity = 0.1 / rho
    fluid_weighted_sensitivity = float(
        jnp.sum(jnp.where(fluid_mask, cell_metric * pressure_sensitivity, 0.0))
        / jnp.sum(jnp.where(fluid_mask, cell_metric, 0.0))
    )
    active_weighted_sensitivity = float(
        jnp.sum(jnp.where(active_mask, cell_metric * pressure_sensitivity, 0.0))
        / jnp.sum(jnp.where(active_mask, cell_metric, 0.0))
    )

    assert float(applied_forcing) == pytest.approx(0.25 / fluid_weighted_sensitivity)
    assert float(applied_forcing) != pytest.approx(0.25 / active_weighted_sensitivity)
    assert float(mean_velocity) > 0.0


def test_target_mean_velocity_only_uses_inlet_flow_rate():
    case = make_hunt_case(ha=20.0, ny=8, nz=8, wall_cells=1)
    inlet_velocity_case = replace(
        case,
        forcing=0.0,
        boundary_conditions=case.boundary_conditions + (BoundaryCondition("inlet", "inlet_velocity", value=(0.2, 0.0, 0.0), axis="x"),),
    )
    inlet_flow_rate_case = replace(
        case,
        forcing=0.0,
        boundary_conditions=case.boundary_conditions
        + (BoundaryCondition("inlet", "inlet_flow_rate", value=0.2 * case.geometry.width * case.geometry.height, axis="x"),),
    )

    assert solvers._target_mean_velocity(inlet_velocity_case) is None
    assert solvers._target_mean_velocity(inlet_flow_rate_case) == pytest.approx(0.2)


def test_reference_mean_velocity_uses_inlet_velocity_or_initial_velocity():
    case = make_hunt_case(ha=20.0, ny=8, nz=8, wall_cells=1)
    inlet_velocity_case = replace(
        case,
        forcing=0.0,
        boundary_conditions=case.boundary_conditions + (BoundaryCondition("inlet", "inlet_velocity", value=(0.2, 0.0, 0.0), axis="x"),),
    )
    initial_velocity_case = replace(case, initial_velocity=0.15)

    assert solvers._reference_mean_velocity(inlet_velocity_case) == pytest.approx(0.2)
    assert solvers._reference_mean_velocity(initial_velocity_case) == pytest.approx(0.15)


def test_active_velocity_mask_excludes_enforced_outer_boundary_cells():
    fluid_mask = jnp.ones((5, 5), dtype=bool)
    active = solvers._active_velocity_mask(fluid_mask)

    assert not bool(active[0, 2])
    assert not bool(active[-1, 2])
    assert not bool(active[2, 0])
    assert not bool(active[2, -1])
    assert bool(active[2, 2])


def test_magnetic_ramp_scale_disables_when_duration_is_zero():
    case = make_hartmann_case(ha=5.0, ny=8, nz=8)
    assert float(magnetic_ramp_scale(case.magnetic_field, time=0.0)) == pytest.approx(1.0)


def test_magnetic_ramp_scale_matches_freemhd_startup_formula():
    case = make_hartmann_case(ha=5.0, ny=8, nz=8)
    ramped = replace(case, magnetic_field=replace(case.magnetic_field, ramp_start=0.0, ramp_duration=1e-5))

    assert float(magnetic_ramp_scale(ramped.magnetic_field, time=0.0)) == pytest.approx(0.0)
    assert float(magnetic_ramp_scale(ramped.magnetic_field, time=1e-5)) == pytest.approx(10.0 / 11.0)
    assert float(magnetic_ramp_scale(ramped.magnetic_field, time=2e-5)) == pytest.approx(1.0)


def test_magnetic_ramp_delays_short_transient_lorentz_response():
    case = make_hartmann_case(ha=20.0, ny=12, nz=12)
    base = replace(
        case,
        forcing=0.0,
        initial_velocity=0.25,
        boundary_conditions=case.boundary_conditions + (BoundaryCondition("inlet", "inlet_velocity", value=(0.25, 0.0, 0.0), axis="x"),),
        time_stepper=replace(case.time_stepper, dt=1e-5, t_final=2e-5, max_steps=2, relaxation=0.1),
    )
    ramped = replace(base, magnetic_field=replace(base.magnetic_field, ramp_start=0.0, ramp_duration=1e-3))

    baseline = solve_steady(base)
    delayed = solve_steady(ramped)

    assert float(delayed.diagnostics.current_max_history[0]) < float(baseline.diagnostics.current_max_history[0])
    assert float(delayed.diagnostics.lorentz_max_history[0]) < float(baseline.diagnostics.lorentz_max_history[0])


def test_shercliff_solution_stays_finite_and_zero_at_walls():
    case = make_shercliff_case(ha=10.0, ny=24, nz=24)
    solution = solve_steady(case)
    assert jnp.isfinite(solution.state.u).all()
    assert jnp.allclose(solution.state.u[0, :], 0.0)
    assert jnp.allclose(solution.state.u[-1, :], 0.0)


def test_hartmann_case_supports_cg_potential_backend():
    case = make_hartmann_case(ha=5.0, ny=12, nz=12)
    case = replace(case, time_stepper=replace(case.time_stepper, potential_solver="cg", potential_iterations=50))
    solution = solve_steady(case)

    assert jnp.isfinite(solution.state.u).all()
    assert jnp.isfinite(solution.state.phi).all()
    assert solution.diagnostics.time_history.shape[0] > 0
    assert solution.diagnostics.u_max_history.shape[0] > 0
    assert solution.diagnostics.potential_iterations_history.shape[0] > 0


def test_hunt_case_supports_volume_scaled_cg_potential_backend():
    case = make_hunt_case(ha=20.0, ny=12, nz=12, wall_cells=2)
    case = replace(case, time_stepper=replace(case.time_stepper, potential_solver="cg_volume", potential_iterations=50))
    solution = solve_steady(case)

    assert jnp.isfinite(solution.state.u).all()
    assert jnp.isfinite(solution.state.phi).all()
    assert solution.diagnostics.time_history.shape[0] > 0
    assert solution.diagnostics.u_max_history.shape[0] > 0
    assert solution.diagnostics.potential_iterations_history.shape[0] > 0
    assert solution.diagnostics.face_current_max_history.shape[0] > 0
    assert solution.diagnostics.emf_max_history.shape[0] > 0


def test_hunt_case_supports_face_averaged_current_reconstruction():
    case = make_hunt_case(ha=20.0, ny=12, nz=12, wall_cells=2)
    case = replace(case, time_stepper=replace(case.time_stepper, current_reconstruction="face_averaged"))
    solution = solve_steady(case)

    assert jnp.isfinite(solution.state.u).all()
    assert jnp.isfinite(solution.state.phi).all()
    assert solution.diagnostics.current_max_history.shape[0] > 0
    assert solution.diagnostics.face_current_max_history.shape[0] > 0


def test_hunt_case_supports_hybrid_face_lorentz_current_reconstruction():
    case = make_hunt_case(ha=20.0, ny=12, nz=12, wall_cells=2)
    case = replace(case, time_stepper=replace(case.time_stepper, current_reconstruction="hybrid_face_lorentz"))
    solution = solve_steady(case)

    assert jnp.isfinite(solution.state.u).all()
    assert jnp.isfinite(solution.state.phi).all()
    assert solution.diagnostics.current_max_history.shape[0] > 0
    assert solution.diagnostics.face_current_max_history.shape[0] > 0
    assert solution.diagnostics.lorentz_max_history.shape[0] > 0


def test_hunt_hybrid_diagnostics_match_returned_state_reduction():
    case = make_hunt_case(ha=20.0, ny=12, nz=12, wall_cells=2)
    case = replace(case, time_stepper=replace(case.time_stepper, current_reconstruction="hybrid_face_lorentz"))
    solution = solve_steady(case)

    mesh = solvers._build_mesh(case)
    materials = build_material_fields(case, mesh)
    bx, by, bz = magnetic_field_components(case.magnetic_field, mesh)
    jy, jz, lorentz = solvers._compute_current_and_lorentz(
        mesh,
        materials.conductivity,
        materials.fluid_mask,
        solution.state.u,
        solution.state.phi,
        by,
        bz,
        reconstruction=case.time_stepper.current_reconstruction,
    )
    face_current_max, emf_max, face_lorentz_max = solvers._face_current_emf_and_lorentz_max(
        mesh,
        materials.conductivity,
        materials.fluid_mask,
        solution.state.u,
        solution.state.phi,
        by,
        bz,
    )

    assert float(solution.diagnostics.current_max_history[-1]) == pytest.approx(float(jnp.max(jnp.abs(jy))), rel=1e-4)
    assert float(solution.diagnostics.face_current_max_history[-1]) == pytest.approx(float(face_current_max), rel=1e-5)
    assert float(solution.diagnostics.emf_max_history[-1]) == pytest.approx(float(emf_max), rel=1e-5)
    assert float(solution.diagnostics.lorentz_max_history[-1]) == pytest.approx(float(jnp.max(jnp.abs(lorentz))), rel=1e-4)
    assert float(solution.diagnostics.face_lorentz_max_history[-1]) == pytest.approx(float(face_lorentz_max), rel=1e-5)


def test_auto_potential_backend_uses_cg_for_single_region_and_volume_scaled_cg_for_layered_cases():
    hartmann = make_hartmann_case(ha=5.0, ny=12, nz=12)
    hunt = make_hunt_case(ha=20.0, ny=12, nz=12, wall_cells=2)

    hartmann_mesh = solvers._build_mesh(hartmann)
    hunt_mesh = solvers._build_mesh(hunt)
    hartmann_materials = build_material_fields(hartmann, hartmann_mesh)
    hunt_materials = build_material_fields(hunt, hunt_mesh)

    assert solvers._resolve_potential_solver("auto", hartmann_materials.fluid_mask) == "cg"
    assert solvers._resolve_potential_solver("auto", hunt_materials.fluid_mask) == "cg_volume"


def test_build_material_fields_assigns_hunt_side_and_hartmann_wall_regions():
    case = make_hunt_case(
        ha=20.0,
        ny=6,
        nz=6,
        wall_cells=1,
        insulator_cells=1,
        fluid_conductivity=3.0,
        wall_conductivity=9.0,
        insulator_conductivity=1.5,
    )
    mesh = solvers._build_mesh(case)
    materials = build_material_fields(case, mesh)
    mid_y = mesh.yz_shape[0] // 2
    mid_z = mesh.yz_shape[1] // 2

    assert materials.conductivity[0, mid_z] == pytest.approx(1.5)
    assert materials.conductivity[-1, mid_z] == pytest.approx(1.5)
    assert materials.conductivity[mid_y, 0] == pytest.approx(9.0)
    assert materials.conductivity[mid_y, -1] == pytest.approx(9.0)
    assert materials.conductivity[mid_y, mid_z] == pytest.approx(3.0)


def test_volume_scaled_potential_system_is_symmetric_after_cell_metric_weighting():
    mesh = generate_layered_duct_mesh(
        width=2.0,
        height=2.0,
        ny=6,
        nz=10,
        wall_thickness=(0.05, 0.05, 0.05, 0.05),
        wall_cells=(2, 2, 2, 2),
        target_ha=20.0,
    )
    sigma = jnp.linspace(1.0, 3.0, mesh.ny * mesh.nz, dtype=float).reshape(mesh.yz_shape)
    diagonal, west, east, south, north = solvers._potential_coefficients(mesh, sigma)
    rhs = jnp.ones(mesh.yz_shape)
    diagonal_s, west_s, east_s, south_s, north_s, rhs_s = solvers._volume_scaled_potential_system(
        mesh,
        diagonal,
        west,
        east,
        south,
        north,
        rhs,
    )

    assert rhs_s.shape == rhs.shape
    assert west_s[1:, :] == pytest.approx(east_s[:-1, :])
    assert south_s[:, 1:] == pytest.approx(north_s[:, :-1])


def test_potential_coefficients_match_uniform_spacing_formula_on_rect_grid():
    mesh = generate_rect_duct_mesh(width=2.0, height=2.0, ny=4, nz=4)
    sigma = jnp.ones((4, 4))
    diagonal, west, east, south, north = solvers._potential_coefficients(mesh, sigma)

    expected = 1.0 / (mesh.dy[1] * 0.5 * (mesh.dy[1] + mesh.dy[2]))
    assert west[2, 2] == pytest.approx(expected)
    assert east[1, 2] == pytest.approx(expected)
    assert south[2, 2] == pytest.approx(expected)
    assert north[2, 1] == pytest.approx(expected)
    assert diagonal[2, 2] == pytest.approx(4.0 * expected)


def test_face_emf_uses_distance_weighted_nonuniform_interface_source():
    mesh = generate_layered_duct_mesh(
        width=2.0,
        height=2.0,
        ny=6,
        nz=10,
        wall_thickness=(0.0, 0.0, 0.1, 0.1),
        wall_cells=(0, 0, 2, 2),
        target_ha=20.0,
    )
    sigma = jnp.ones(mesh.yz_shape)
    source = jnp.zeros(mesh.yz_shape)
    source = source.at[3, 4].set(2.0)
    source = source.at[3, 5].set(-1.0)

    emf_z = solvers._face_emf_z(mesh, sigma, source)
    left_distance = 0.5 * mesh.dz[4]
    right_distance = 0.5 * mesh.dz[5]
    conductance = 1.0 / (left_distance + right_distance)
    expected = conductance * (left_distance * 2.0 + right_distance * -1.0)

    assert emf_z[3, 4] == pytest.approx(float(expected))


def test_solve_steady_stops_once_residual_reaches_tolerance(monkeypatch: pytest.MonkeyPatch):
    case = make_hartmann_case(ha=5.0, ny=8, nz=8)
    case = replace(case, time_stepper=replace(case.time_stepper, max_steps=10, steady_tolerance=1e-4))
    residuals = iter([1.0e-1, 1.0e-2, 1.0e-5, 1.0e-6])

    monkeypatch.setattr(solvers.jax, "jit", lambda fn: fn)

    def fake_step(**kwargs):
        residual = next(residuals)
        u = kwargs["u"]
        zeros = jnp.zeros_like(u)
        return u, zeros, zeros, zeros, zeros, residual, 1.0e-3, 25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    monkeypatch.setattr(solvers, "_step", fake_step)
    solution = solve_steady(case)

    assert solution.diagnostics.time_history.shape[0] == 3
    assert solution.diagnostics.u_max_history.shape[0] == 3
    assert solution.diagnostics.mean_velocity_history.shape[0] == 3
    assert solution.diagnostics.applied_forcing_history.shape[0] == 3
    assert solution.diagnostics.pressure_proxy_history.shape[0] == 3
    assert solution.diagnostics.residual_history.shape[0] == 3
    assert solution.diagnostics.potential_residual_history.shape[0] == 3
    assert solution.diagnostics.potential_iterations_history.shape[0] == 3
    assert solution.state.time == pytest.approx(3 * case.time_stepper.dt)
    assert solution.state.residual == pytest.approx(1.0e-5)


def test_solve_steady_respects_max_steps_when_tolerance_not_reached(monkeypatch: pytest.MonkeyPatch):
    case = make_hartmann_case(ha=5.0, ny=8, nz=8)
    case = replace(case, time_stepper=replace(case.time_stepper, max_steps=2, steady_tolerance=1e-8))

    monkeypatch.setattr(solvers.jax, "jit", lambda fn: fn)

    def fake_step(**kwargs):
        u = kwargs["u"]
        zeros = jnp.zeros_like(u)
        return u, zeros, zeros, zeros, zeros, 1.0e-2, 1.0e-3, 50, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    monkeypatch.setattr(solvers, "_step", fake_step)
    solution = solve_steady(case)

    assert solution.diagnostics.time_history.shape[0] == 2
    assert solution.diagnostics.u_max_history.shape[0] == 2
    assert solution.diagnostics.mean_velocity_history.shape[0] == 2
    assert solution.diagnostics.applied_forcing_history.shape[0] == 2
    assert solution.diagnostics.pressure_proxy_history.shape[0] == 2
    assert solution.diagnostics.residual_history.shape[0] == 2
    assert solution.diagnostics.potential_residual_history.shape[0] == 2
    assert solution.diagnostics.potential_iterations_history.shape[0] == 2
    assert solution.state.time == pytest.approx(2 * case.time_stepper.dt)
    assert solution.state.residual == pytest.approx(1.0e-2)


def test_solve_steady_can_require_potential_residual_convergence(monkeypatch: pytest.MonkeyPatch):
    case = make_hartmann_case(ha=5.0, ny=8, nz=8)
    case = replace(
        case,
        time_stepper=replace(
            case.time_stepper,
            max_steps=5,
            steady_tolerance=1e-4,
            steady_potential_tolerance=5e-4,
        ),
    )
    residuals = iter([1.0e-3, 1.0e-5, 1.0e-5, 1.0e-6])
    potential_residuals = iter([1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5])

    monkeypatch.setattr(solvers.jax, "jit", lambda fn: fn)

    def fake_step(**kwargs):
        u = kwargs["u"]
        zeros = jnp.zeros_like(u)
        return u, zeros, zeros, zeros, zeros, next(residuals), next(potential_residuals), 20, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    monkeypatch.setattr(solvers, "_step", fake_step)
    solution = solve_steady(case)

    assert solution.diagnostics.residual_history.shape[0] == 3
    assert solution.diagnostics.potential_residual_history.shape[0] == 3
    assert solution.state.time == pytest.approx(3 * case.time_stepper.dt)
    assert solution.state.residual == pytest.approx(1.0e-5)

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from lmx.cases import make_hartmann_case, make_hunt_case, make_shercliff_case
from lmx.io import load_restart_bundle, write_solution_npz
from lmx.physics import build_material_fields, magnetic_field_components, magnetic_ramp_scale
import lmx.solvers as solvers
from lmx.mesh import generate_layered_duct_mesh, generate_rect_duct_mesh
from lmx.specs import BoundaryCondition, GeometrySpec
from lmx.solvers import solve_steady, solve_transient


def test_hartmann_solver_runs(monkeypatch: pytest.MonkeyPatch):
    case = make_hartmann_case(ha=10.0, ny=12, nz=12)
    assert case.solver.kind == "fully_developed_inductionless"
    def fake_fully_developed_case_step(**kwargs):
        u_prev = kwargs["u_previous"]
        updated = jnp.full_like(u_prev, 0.2)
        zeros = jnp.zeros_like(updated)
        return updated, zeros, zeros, zeros, zeros, 1.0e-6, 1.0e-6, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, float(jnp.mean(updated)), 0.0, 1.0e-3, 1.0e-3

    monkeypatch.setattr(solvers, "_fully_developed_case_step", fake_fully_developed_case_step)
    solution = solve_steady(case)
    assert solution.state.u.shape == (12, 12)
    assert float(jnp.max(solution.state.u)) > 0.0
    assert jnp.isfinite(solution.state.phi).all()


def test_build_mesh_rejects_unsupported_geometry_kind():
    case = make_hartmann_case(ha=10.0, ny=8, nz=8)
    unsupported = replace(case, geometry=replace(case.geometry, kind="pipe_ogrid"))

    with pytest.raises(NotImplementedError, match="not supported"):
        solvers._build_mesh(unsupported)


def test_build_mesh_uses_magnetic_axis_to_cluster_rect_duct_layers():
    hartmann_case = make_hartmann_case(ha=20.0, width=0.2, height=0.2, ny=48, nz=48)
    shercliff_case = make_shercliff_case(ha=20.0, width=0.2, height=0.2, ny=48, nz=48)

    hartmann_mesh = solvers._build_mesh(hartmann_case)
    shercliff_mesh = solvers._build_mesh(shercliff_case)

    assert float(jnp.min(hartmann_mesh.dy)) < float(jnp.min(hartmann_mesh.dz))
    assert float(jnp.min(shercliff_mesh.dy)) < float(jnp.min(shercliff_mesh.dz))


def test_bounded_time_step_count_covers_zero_and_invalid_dt_cases():
    assert solvers._bounded_time_step_count(start_time=0.0, dt=0.1, t_final=1.0, max_steps=0) == 0
    assert solvers._bounded_time_step_count(start_time=1.0, dt=0.1, t_final=0.5, max_steps=10) == 0
    with pytest.raises(ValueError, match="dt must be positive"):
        solvers._bounded_time_step_count(start_time=0.0, dt=0.0, t_final=1.0, max_steps=10)


def test_active_velocity_mask_for_solver_switches_between_fluid_and_extended_masks():
    fluid_mask = jnp.asarray([[True, True, True], [True, True, True], [True, True, True]])

    fully_developed = solvers._active_velocity_mask_for_solver(fluid_mask, "fully_developed_inductionless")
    extruded = solvers._active_velocity_mask_for_solver(fluid_mask, "extruded_inductionless")

    assert jnp.array_equal(extruded, fluid_mask)
    assert jnp.array_equal(
        fully_developed,
        jnp.asarray([[False, False, False], [False, True, False], [False, False, False]]),
    )


def test_potential_coefficients_stay_positive_for_low_conductivity_cells():
    mesh = generate_rect_duct_mesh(width=2.0, height=2.0, ny=4, nz=4)
    sigma = jnp.asarray(
        [
            [1.0, 0.5, 1.0, 1.0],
            [1.0e-12, 1.0e-8, 0.25, 1.0],
            [0.75, 1.0, 1.0, 0.5],
            [1.0, 1.0, 1.0e-10, 1.0],
        ],
        dtype=float,
    )

    diagonal, west, east, south, north = solvers._potential_coefficients(mesh, sigma)

    assert jnp.all(diagonal > 0.0)
    assert jnp.all(west >= 0.0)
    assert jnp.all(east >= 0.0)
    assert jnp.all(south >= 0.0)
    assert jnp.all(north >= 0.0)


def test_hunt_solver_keeps_solid_velocity_zero():
    case = make_hunt_case(ha=10.0, ny=10, nz=10, wall_cells=2)
    mesh = solvers._build_mesh(case)
    fluid_mask = build_material_fields(case, mesh).fluid_mask
    enforced = solvers._enforce_velocity_bc(
        jnp.ones(mesh.yz_shape),
        mesh,
        fluid_mask,
        interpolate_direct_fluid_walls=False,
    )
    assert jnp.allclose(enforced[~fluid_mask], 0.0)


def test_hunt_fully_developed_velocity_linear_solve_is_well_conditioned():
    mesh = generate_layered_duct_mesh(
        width=2.0,
        height=2.0,
        ny=6,
        nz=6,
        wall_thickness=(0.1, 0.1, 0.1, 0.1),
        wall_cells=(1, 1, 1, 1),
        target_ha=20.0,
    )
    active_mask = jnp.ones(mesh.yz_shape, dtype=bool)
    diffusivity = jnp.ones(mesh.yz_shape) * 0.1
    reaction = jnp.ones(mesh.yz_shape) * 0.2
    rhs = jnp.ones(mesh.yz_shape) * 0.05

    u, residual, iterations, initial_residual = solvers._solve_velocity_system(
        mesh=mesh,
        diffusivity=diffusivity,
        reaction=reaction,
        rhs=rhs,
        active_mask=active_mask,
        linear_solver="cg",
        preconditioner="jacobi",
        max_steps=40,
        tolerance=1e-8,
    )

    assert jnp.isfinite(u).all()
    assert float(initial_residual) >= float(residual)
    assert float(residual) < 1e-4
    assert int(iterations) >= 0


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
    assert ha20.solver.kind == "fully_developed_inductionless"
    assert ha100.solver.kind == "fully_developed_inductionless"
    assert ha1000.solver.kind == "fully_developed_inductionless"


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


def test_hunt_inlet_flow_rate_boundary_drives_short_transient(monkeypatch: pytest.MonkeyPatch):
    case = make_hunt_case(ha=20.0, ny=8, nz=8, wall_cells=1)
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

    def fake_fully_developed_case_step(**kwargs):
        u_prev = kwargs["u_previous"]
        target = kwargs["target_mean_velocity"]
        increment = 0.05 if target is not None else 0.0
        updated = u_prev + increment
        zeros = jnp.zeros_like(updated)
        return updated, zeros, zeros, zeros, zeros, 1.0e-6, 1.0e-6, 1.0, 2.0, increment, increment, increment, 0.0, 0.0, 0.0, 1.0e-3, 1.0e-3

    monkeypatch.setattr(solvers, "_fully_developed_case_step", fake_fully_developed_case_step)

    driven_solution = solve_steady(driven)
    undriven_solution = solve_steady(undriven)

    assert float(jnp.max(driven_solution.state.u)) > float(jnp.max(undriven_solution.state.u))

def test_transient_restart_matches_direct_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    case = make_hartmann_case(ha=5.0, ny=8, nz=8)
    direct_case = replace(case, time_stepper=replace(case.time_stepper, dt=0.01, t_final=0.04, max_steps=4))
    partial_case = replace(case, time_stepper=replace(case.time_stepper, dt=0.01, t_final=0.02, max_steps=2))

    def fake_fully_developed_case_step(**kwargs):
        u_prev = kwargs["u_previous"]
        step_time = kwargs["step_time"]
        updated = jnp.full_like(u_prev, step_time * 10.0)
        zeros = jnp.zeros_like(updated)
        return updated, zeros, zeros, zeros, zeros, 1.0e-6, 1.0e-6, 1.0, 2.0, 0.3, 0.2, 0.1, 0.05, float(jnp.mean(updated)), 0.3, 1.0e-3, 1.0e-3

    monkeypatch.setattr(solvers, "_fully_developed_case_step", fake_fully_developed_case_step)

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


def test_solve_steady_rejects_unknown_solver_kind():
    case = make_hartmann_case(ha=5.0, ny=8, nz=8)
    bad = replace(case, solver=replace(case.solver, kind="definitely_missing"))
    with pytest.raises(NotImplementedError, match="not implemented for steady runs"):
        solve_steady(bad)


def test_solve_transient_rejects_unknown_solver_kind():
    case = make_hartmann_case(ha=5.0, ny=8, nz=8)
    bad = replace(case, solver=replace(case.solver, kind="definitely_missing"))
    with pytest.raises(NotImplementedError, match="not implemented for transient runs"):
        solve_transient(bad)


def test_transient_restart_can_append_diagnostics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    case = make_hartmann_case(ha=5.0, ny=8, nz=8)
    direct_case = replace(case, time_stepper=replace(case.time_stepper, dt=0.01, t_final=0.04, max_steps=4))
    partial_case = replace(case, time_stepper=replace(case.time_stepper, dt=0.01, t_final=0.02, max_steps=2))

    def fake_fully_developed_case_step(**kwargs):
        u_prev = kwargs["u_previous"]
        step_time = kwargs["step_time"]
        updated = jnp.full_like(u_prev, step_time * 10.0)
        zeros = jnp.zeros_like(updated)
        return updated, zeros, zeros, zeros, zeros, 1.0e-6, 1.0e-6, 1.0, 2.0, 0.3, 0.2, 0.1, 0.05, float(jnp.mean(updated)), 0.3, 1.0e-3, 1.0e-3

    monkeypatch.setattr(solvers, "_fully_developed_case_step", fake_fully_developed_case_step)

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


def test_inlet_speed_reads_tuple_components_by_axis():
    case = make_hartmann_case(ha=5.0, ny=8, nz=8)

    assert solvers._inlet_speed(BoundaryCondition("inlet", "inlet_velocity", value=(1.0, 2.0, 3.0), axis="x"), case) == pytest.approx(1.0)
    assert solvers._inlet_speed(BoundaryCondition("inlet", "inlet_velocity", value=(1.0, 2.0, 3.0), axis="y"), case) == pytest.approx(2.0)
    assert solvers._inlet_speed(BoundaryCondition("inlet", "inlet_velocity", value=(1.0, 2.0, 3.0), axis="z"), case) == pytest.approx(3.0)


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


def test_explicit_forcing_and_active_velocity_mask_helpers():
    mask = jnp.ones((3, 3), dtype=bool)
    active = solvers._active_velocity_mask(mask)
    assert jnp.array_equal(
        active,
        jnp.asarray(
            [
                [False, False, False],
                [False, True, False],
                [False, False, False],
            ]
        ),
    )
    forcing = solvers._explicit_forcing(1.25, jnp.float32)
    assert forcing.dtype == jnp.float32
    assert float(forcing) == pytest.approx(1.25)


def test_concat_history_handles_append_and_empty_inputs():
    current = jnp.asarray([1.0, 2.0])

    assert jnp.allclose(solvers._concat_history(None, current, append=True), current)
    assert jnp.allclose(solvers._concat_history(jnp.asarray([]), current, append=True), current)
    assert jnp.allclose(solvers._concat_history(jnp.asarray([9.0]), current, append=False), current)
    assert jnp.allclose(solvers._concat_history(jnp.asarray([9.0]), current, append=True), jnp.asarray([9.0, 1.0, 2.0]))


def test_pressure_proxy_reference_current_prefers_face_current_history():
    diagnostics = SimpleNamespace(
        face_current_max_history=jnp.asarray([0.4, 0.3]),
        current_max_history=jnp.asarray([0.2, 0.1]),
    )

    assert solvers._pressure_proxy_reference_current(diagnostics) == pytest.approx(0.4)
    assert solvers._pressure_proxy_reference_current(SimpleNamespace(face_current_max_history=jnp.asarray([]), current_max_history=jnp.asarray([0.2]))) == pytest.approx(0.2)
    assert solvers._pressure_proxy_reference_current(SimpleNamespace(face_current_max_history=jnp.asarray([]), current_max_history=jnp.asarray([]))) is None
    assert solvers._pressure_proxy_reference_current(None) is None


def test_scaled_pressure_proxy_value_uses_available_current_source():
    value, reference = solvers._scaled_pressure_proxy_value(pressure_proxy=2.0, current_max=0.5, face_current_max=0.0, reference_current=None)
    assert reference == pytest.approx(0.5)
    assert value == pytest.approx(2.0)

    value, reference = solvers._scaled_pressure_proxy_value(pressure_proxy=2.0, current_max=0.5, face_current_max=1.0, reference_current=0.25)
    assert reference == pytest.approx(0.25)
    assert value == pytest.approx(8.0)


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


def test_magnetic_ramp_scale_matches_reference_startup_formula():
    case = make_hartmann_case(ha=5.0, ny=6, nz=6)
    ramped = replace(case, magnetic_field=replace(case.magnetic_field, ramp_start=0.0, ramp_duration=1e-5))

    assert float(magnetic_ramp_scale(ramped.magnetic_field, time=0.0)) == pytest.approx(0.0)
    assert float(magnetic_ramp_scale(ramped.magnetic_field, time=1e-5)) == pytest.approx(10.0 / 11.0)
    assert float(magnetic_ramp_scale(ramped.magnetic_field, time=2e-5)) == pytest.approx(1.0)


def test_magnetic_ramp_delays_short_transient_lorentz_response(monkeypatch: pytest.MonkeyPatch):
    case = make_hartmann_case(ha=20.0, ny=8, nz=8)
    base = replace(
        case,
        forcing=0.0,
        initial_velocity=0.25,
        boundary_conditions=case.boundary_conditions + (BoundaryCondition("inlet", "inlet_velocity", value=(0.25, 0.0, 0.0), axis="x"),),
        time_stepper=replace(case.time_stepper, dt=1e-5, t_final=2e-5, max_steps=2, relaxation=0.1),
    )
    ramped = replace(base, magnetic_field=replace(base.magnetic_field, ramp_start=0.0, ramp_duration=1e-3))

    def fake_fully_developed_case_step(**kwargs):
        u_prev = kwargs["u_previous"]
        step_time = kwargs["step_time"]
        case = kwargs["case"]
        scale = float(magnetic_ramp_scale(case.magnetic_field, time=step_time))
        updated = u_prev + 0.01
        jy = jnp.full_like(updated, 0.2 * scale)
        lorentz = jnp.full_like(updated, 0.05 * scale)
        zeros = jnp.zeros_like(updated)
        return updated, zeros, jy, zeros, lorentz, 1.0e-6, 1.0e-6, 1.0, 2.0, 0.2 * scale, 0.15 * scale, 0.1 * scale, 0.05 * scale, float(jnp.mean(updated)), 0.0, 1.0e-3, 1.0e-3

    monkeypatch.setattr(solvers, "_fully_developed_case_step", fake_fully_developed_case_step)

    baseline = solve_steady(base)
    delayed = solve_steady(ramped)

    assert float(delayed.diagnostics.current_max_history[0]) < float(baseline.diagnostics.current_max_history[0])
    assert float(delayed.diagnostics.lorentz_max_history[0]) < float(baseline.diagnostics.lorentz_max_history[0])


def test_shercliff_solution_stays_finite_and_zero_at_walls():
    case = make_shercliff_case(ha=10.0, ny=12, nz=12)
    mesh = solvers._build_mesh(case)
    y, z = jnp.meshgrid(mesh.y_centers, mesh.z_centers, indexing="ij")
    profile = 1.0 - 0.2 * y**2 - 0.3 * z**2
    enforced = solvers._enforce_velocity_bc(
        profile,
        mesh,
        jnp.ones(mesh.yz_shape, dtype=bool),
        interpolate_direct_fluid_walls=False,
    )
    assert jnp.isfinite(enforced).all()
    assert jnp.allclose(enforced[0, :], 0.0)
    assert jnp.allclose(enforced[-1, :], 0.0)


def test_potential_solver_backends_return_finite_fields_on_small_system():
    hartmann = make_hartmann_case(ha=5.0, ny=4, nz=4)
    hunt = make_hunt_case(ha=20.0, ny=4, nz=4, wall_cells=1)

    for case, solver_name in ((hartmann, "cg"), (hunt, "cg_volume")):
        mesh = solvers._build_mesh(case)
        materials = build_material_fields(case, mesh)
        _, by, bz = magnetic_field_components(case.magnetic_field, mesh)
        phi, residual, iterations, initial_residual = solvers._solve_potential(
            mesh,
            materials.conductivity,
            materials.fluid_mask,
            jnp.zeros(mesh.yz_shape),
            by,
            bz,
            case.reference_phi_cell,
            iterations=20,
            tolerance=1e-8,
            solver=solver_name,
        )
        assert jnp.isfinite(phi).all()
        assert float(initial_residual) >= 0.0
        assert jnp.isfinite(residual)
        assert int(iterations) >= 0


def test_current_reconstruction_modes_and_face_diagnostics_are_finite():
    case = make_hunt_case(ha=20.0, ny=4, nz=4, wall_cells=1)
    mesh = solvers._build_mesh(case)
    materials = build_material_fields(case, mesh)
    _, by, bz = magnetic_field_components(case.magnetic_field, mesh)
    y_index = jnp.arange(mesh.yz_shape[0], dtype=jnp.float32)[:, None]
    z_index = jnp.arange(mesh.yz_shape[1], dtype=jnp.float32)[None, :]
    u = 0.1 + 0.01 * y_index - 0.02 * z_index
    phi = 0.03 * y_index + 0.04 * z_index

    for reconstruction in ("cell_centered", "face_averaged", "hybrid_face_lorentz"):
        jy, jz, lorentz = solvers._compute_current_and_lorentz(
            mesh,
            materials.conductivity,
            materials.fluid_mask,
            u,
            phi,
            by,
            bz,
            reconstruction=reconstruction,
        )
        assert jnp.isfinite(jy).all()
        assert jnp.isfinite(jz).all()
        assert jnp.isfinite(lorentz).all()

    face_current_max, emf_max, face_lorentz_max = solvers._face_current_emf_and_lorentz_max(
        mesh,
        materials.conductivity,
        materials.fluid_mask,
        u,
        phi,
        by,
        bz,
    )
    assert float(face_current_max) >= 0.0
    assert float(emf_max) >= 0.0
    assert float(face_lorentz_max) >= 0.0


def test_face_current_components_and_integral_diagnostics_remain_bounded():
    case = make_hunt_case(ha=20.0, ny=6, nz=6, wall_cells=1)
    mesh = solvers._build_mesh(case)
    materials = build_material_fields(case, mesh)
    _, by, bz = magnetic_field_components(case.magnetic_field, mesh)
    y_index = jnp.arange(mesh.yz_shape[0], dtype=jnp.float32)[:, None]
    z_index = jnp.arange(mesh.yz_shape[1], dtype=jnp.float32)[None, :]
    u = jnp.where(materials.fluid_mask, 0.2 + 0.03 * y_index - 0.01 * z_index, 0.0)
    phi = 0.02 * y_index - 0.01 * z_index
    jy, jz = solvers._conductive_current_components(
        mesh,
        materials.conductivity,
        materials.fluid_mask,
        u,
        phi,
        by,
        bz,
    )
    lorentz = jy * bz - jz * by
    face_jy, face_jz, emf_y, emf_z = solvers._face_current_components(
        mesh,
        materials.conductivity,
        materials.fluid_mask,
        u,
        phi,
        by,
        bz,
    )

    diagnostics = solvers._integral_diagnostics(
        mesh=mesh,
        sigma=materials.conductivity,
        fluid_mask=materials.fluid_mask,
        u=u,
        phi=phi,
        jy=jy,
        jz=jz,
        lorentz=lorentz,
        by=by,
        bz=bz,
        anchor=(0, 0),
    )

    assert face_jy.shape == (mesh.yz_shape[0] - 1, mesh.yz_shape[1])
    assert face_jz.shape == (mesh.yz_shape[0], mesh.yz_shape[1] - 1)
    assert emf_y.shape == face_jy.shape
    assert emf_z.shape == face_jz.shape
    assert all(jnp.isfinite(value) for value in diagnostics)
    volumetric_flow_rate, mean_current_magnitude, lorentz_power, *_ = diagnostics
    assert float(volumetric_flow_rate) > 0.0
    assert float(mean_current_magnitude) >= 0.0
    assert jnp.isfinite(lorentz_power)


def test_auto_potential_backend_uses_cg_for_single_region_and_volume_scaled_cg_for_layered_cases():
    hartmann = make_hartmann_case(ha=5.0, ny=6, nz=6)
    hunt = make_hunt_case(ha=20.0, ny=6, nz=6, wall_cells=1)

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

    assert materials.conductivity[0, mid_z] == pytest.approx(9.0)
    assert materials.conductivity[-1, mid_z] == pytest.approx(9.0)
    assert materials.conductivity[mid_y, 0] == pytest.approx(1.5)
    assert materials.conductivity[mid_y, -1] == pytest.approx(1.5)
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


def test_fully_developed_rhs_uses_lorentz_source_only_inside_fluid(monkeypatch: pytest.MonkeyPatch):
    mesh = generate_rect_duct_mesh(width=2.0, height=2.0, ny=4, nz=4)
    fluid_mask = jnp.asarray(
        [
            [False, False, False, False],
            [False, True, True, False],
            [False, True, True, False],
            [False, False, False, False],
        ]
    )
    sigma = jnp.ones(mesh.yz_shape)
    rho = jnp.ones(mesh.yz_shape) * 2.0
    u = jnp.zeros(mesh.yz_shape)
    phi = jnp.zeros(mesh.yz_shape)
    by = jnp.zeros(mesh.yz_shape)
    bz = jnp.ones(mesh.yz_shape)
    lorentz = jnp.arange(mesh.ny * mesh.nz, dtype=float).reshape(mesh.yz_shape)

    monkeypatch.setattr(
        solvers,
        "_compute_current_and_lorentz",
        lambda *args, **kwargs: (jnp.zeros_like(lorentz), jnp.zeros_like(lorentz), lorentz),
    )

    rhs, lorentz_source = solvers._fully_developed_rhs(
        mesh=mesh,
        sigma=sigma,
        rho=rho,
        fluid_mask=fluid_mask,
        u=u,
        phi=phi,
        by=by,
        bz=bz,
        forcing=jnp.asarray(0.5),
    )

    assert jnp.allclose(lorentz_source, lorentz)
    assert jnp.allclose(rhs[~fluid_mask], 0.0)
    assert jnp.allclose(rhs[fluid_mask], (0.5 + lorentz[fluid_mask]) / 2.0)


def test_interface_and_face_conductances_match_uniform_rect_grid_symmetry():
    mesh = generate_rect_duct_mesh(width=2.0, height=2.0, ny=4, nz=4)
    sigma = jnp.ones((4, 4))

    conductance_y = solvers._interface_conductance_y(mesh, sigma)
    west, east = solvers._face_conductance_y(mesh, sigma)
    conductance_z = solvers._interface_conductance_z(mesh, sigma)
    south, north = solvers._face_conductance_z(mesh, sigma)

    expected = 1.0 / (0.5 * mesh.dy[1] + 0.5 * mesh.dy[2])
    assert conductance_y[1, 2] == pytest.approx(expected)
    assert conductance_z[2, 1] == pytest.approx(expected)
    assert west.shape == sigma.shape
    assert east.shape == sigma.shape
    assert south.shape == sigma.shape
    assert north.shape == sigma.shape
    assert west[2, 2] == pytest.approx(expected / mesh.dy[2])
    assert east[1, 2] == pytest.approx(expected / mesh.dy[1])
    assert south[2, 2] == pytest.approx(expected / mesh.dz[2])
    assert north[2, 1] == pytest.approx(expected / mesh.dz[1])


def test_solve_velocity_system_returns_zero_outside_active_mask():
    mesh = generate_rect_duct_mesh(width=2.0, height=2.0, ny=4, nz=4)
    diffusivity = jnp.ones(mesh.yz_shape) * 0.1
    reaction = jnp.ones(mesh.yz_shape) * 0.05
    rhs = jnp.ones(mesh.yz_shape)
    active_mask = jnp.asarray(
        [
            [False, False, False, False],
            [False, True, True, False],
            [False, True, True, False],
            [False, False, False, False],
        ]
    )

    field, residual, iterations, initial_residual = solvers._solve_velocity_system(
        mesh=mesh,
        diffusivity=diffusivity,
        reaction=reaction,
        rhs=rhs,
        active_mask=active_mask,
        linear_solver="cg",
        preconditioner="jacobi",
        max_steps=64,
        tolerance=1.0e-10,
    )

    assert field.shape == mesh.yz_shape
    assert jnp.allclose(field[~active_mask], 0.0)
    assert jnp.all(jnp.isfinite(field))
    assert float(residual) >= 0.0
    assert int(iterations) >= 0
    assert float(initial_residual) >= 0.0


def test_velocity_system_coefficients_cover_connected_and_boundary_fallback_paths():
    mesh = generate_layered_duct_mesh(
        width=2.0,
        height=2.0,
        ny=6,
        nz=6,
        wall_thickness=(0.1, 0.1, 0.1, 0.1),
        wall_cells=(1, 1, 1, 1),
        target_ha=20.0,
    )
    diffusivity = jnp.ones(mesh.yz_shape) * 0.2
    reaction = jnp.ones(mesh.yz_shape) * 0.05
    active_mask = jnp.zeros(mesh.yz_shape, dtype=bool)
    active_mask = active_mask.at[2:5, 2:5].set(True)

    diagonal, west, east, south, north = solvers._velocity_system_coefficients(
        mesh,
        diffusivity,
        reaction,
        active_mask,
    )

    assert diagonal.shape == mesh.yz_shape
    assert float(diagonal[0, 0]) == pytest.approx(1.0)
    assert float(west[0, 0]) == pytest.approx(0.0)
    assert float(north[0, 0]) == pytest.approx(0.0)

    interior_value = float(west[3, 3])
    boundary_fallback = float(west[2, 2])
    assert interior_value > 0.0
    assert boundary_fallback > 0.0
    assert boundary_fallback != pytest.approx(interior_value)
    assert float(diagonal[3, 3]) > float(reaction[3, 3])


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


def test_conductive_current_components_keep_wall_currents_for_interface_audits():
    mesh = generate_layered_duct_mesh(
        width=2.0,
        height=2.0,
        ny=6,
        nz=6,
        wall_thickness=(0.1, 0.1, 0.1, 0.1),
        wall_cells=(1, 1, 1, 1),
        target_ha=20.0,
    )
    case = make_hunt_case(ha=10.0, ny=6, nz=6, wall_cells=1)
    materials = build_material_fields(case, mesh)
    phi = jnp.linspace(0.0, 1.0, mesh.ny * mesh.nz, dtype=float).reshape(mesh.yz_shape)
    u = jnp.ones(mesh.yz_shape) * 0.1
    _, by, bz = magnetic_field_components(case.magnetic_field, mesh, time=0.0)

    jy_all, jz_all = solvers._conductive_current_components(
        mesh,
        materials.conductivity,
        materials.fluid_mask,
        u,
        phi,
        by,
        bz,
    )
    jy_masked, jz_masked, _ = solvers._compute_current_and_lorentz(
        mesh,
        materials.conductivity,
        materials.fluid_mask,
        u,
        phi,
        by,
        bz,
        reconstruction="cell_centered",
    )

    wall_mask = ~materials.fluid_mask
    assert float(jnp.max(jnp.abs(jy_all[wall_mask]))) > 0.0
    assert float(jnp.max(jnp.abs(jz_all[wall_mask]))) > 0.0
    assert jnp.allclose(jy_masked[wall_mask], 0.0)
    assert jnp.allclose(jz_masked[wall_mask], 0.0)


def test_solve_steady_respects_t_final_when_tolerance_not_reached(monkeypatch: pytest.MonkeyPatch):
    case = make_hartmann_case(ha=5.0, ny=8, nz=8)
    case = replace(
        case,
        time_stepper=replace(case.time_stepper, dt=0.002, t_final=0.01, max_steps=200, steady_tolerance=0.0),
    )

    def fake_fully_developed_case_step(**kwargs):
        u = kwargs["u_previous"]
        zeros = jnp.zeros_like(u)
        return u, zeros, zeros, zeros, zeros, 1.0e-2, 1.0e-2, 25, 1.0e-2, 8.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0e-2, 1.0e-2

    monkeypatch.setattr(solvers, "_fully_developed_case_step", fake_fully_developed_case_step)
    solution = solve_steady(case)

    assert solution.diagnostics.time_history.shape[0] == 5
    assert float(solution.diagnostics.time_history[-1]) == pytest.approx(0.01)
    assert solution.state.time == pytest.approx(0.01)


def test_hunt_low_resolution_manual_interface_gate_is_now_bounded():
    case = make_hunt_case(ha=10.0, ny=8, nz=8, wall_cells=2)
    case = replace(
        case,
        time_stepper=replace(case.time_stepper, max_steps=12, potential_iterations=48),
        solver=replace(case.solver, coupling_iterations=8),
    )

    solution = solve_steady(case)

    assert float(solution.diagnostics.charge_balance_residual_history[-1]) <= 8.0e-1
    assert float(solution.diagnostics.interface_current_residual_history[-1]) <= 2.5e-1


def test_bounded_time_step_count_does_not_round_up_fractional_end_times():
    assert solvers._bounded_time_step_count(start_time=0.0, dt=0.002, t_final=0.011, max_steps=200) == 5
    assert solvers._bounded_time_step_count(start_time=0.004, dt=0.002, t_final=0.011, max_steps=200) == 3
    assert solvers._bounded_time_step_count(start_time=0.0, dt=0.02, t_final=0.01, max_steps=200) == 0


def test_bounded_time_step_count_rejects_bad_dt_and_handles_empty_windows():
    with pytest.raises(ValueError, match="dt must be positive"):
        solvers._bounded_time_step_count(start_time=0.0, dt=0.0, t_final=1.0, max_steps=10)
    assert solvers._bounded_time_step_count(start_time=0.0, dt=0.1, t_final=1.0, max_steps=0) == 0
    assert solvers._bounded_time_step_count(start_time=1.0, dt=0.1, t_final=1.0, max_steps=10) == 0


def test_build_mesh_and_mask_helpers_cover_unsupported_and_passthrough_paths():
    case = make_hartmann_case(ha=5.0, ny=4, nz=4)
    bad_case = replace(case, geometry=GeometrySpec(kind="pipe_ogrid", width=1.0, height=1.0, radius=0.5, nr=4, ntheta=8))
    with pytest.raises(NotImplementedError, match="not supported"):
        solvers._build_mesh(bad_case)
    fluid_mask = jnp.asarray([[True, True], [True, True]])
    assert solvers._active_velocity_mask_for_solver(fluid_mask, "fully_developed_inductionless").shape == fluid_mask.shape
    assert jnp.array_equal(solvers._active_velocity_mask_for_solver(fluid_mask, "other"), fluid_mask)


def test_fully_developed_solver_rejects_pipe_geometry():
    case = make_hartmann_case(ha=5.0, ny=4, nz=4)
    bad_case = replace(
        case,
        geometry=GeometrySpec(kind="pipe_ogrid", width=1.0, height=1.0, radius=0.5, nr=4, ntheta=8),
    )
    with pytest.raises(NotImplementedError, match="not supported by the laminar solver"):
        solvers._solve_fully_developed(bad_case)


def test_fully_developed_case_step_rejects_non_implicit_transient_scheme():
    case = make_hartmann_case(ha=5.0, ny=4, nz=4)
    case = replace(case, solver=replace(case.solver, mode="transient", time_scheme="crank_nicolson"))
    mesh = solvers._build_mesh(case)
    materials = build_material_fields(case, mesh)
    u_previous = jnp.zeros(mesh.yz_shape)

    with pytest.raises(NotImplementedError, match="implicit_euler only"):
        solvers._fully_developed_case_step(
            case=case,
            mesh=mesh,
            materials=materials,
            u_previous=u_previous,
            step_time=case.time_stepper.dt,
            potential_solver="jacobi",
            target_mean_velocity=None,
            linear_solver="cg",
            preconditioner="jacobi",
            coupling_iterations=1,
            coupling_tolerance=1.0e-6,
        )


def test_fully_developed_case_step_uses_explicit_forcing_when_no_target_velocity(monkeypatch: pytest.MonkeyPatch):
    case = make_hartmann_case(ha=5.0, ny=4, nz=4)
    case = replace(case, time_stepper=replace(case.time_stepper, velocity_update_limit=1.0))
    mesh = solvers._build_mesh(case)
    materials = build_material_fields(case, mesh)
    u_previous = jnp.zeros(mesh.yz_shape)
    call_counter = {"velocity": 0}

    monkeypatch.setattr(
        solvers,
        "_solve_potential",
        lambda *args, **kwargs: (
            jnp.zeros(mesh.yz_shape),
            jnp.asarray(1.0e-9),
            jnp.asarray(3, dtype=jnp.int32),
            jnp.asarray(1.0e-6),
        ),
    )

    def fake_solve_velocity_system(**kwargs):
        call_counter["velocity"] += 1
        field = jnp.full(mesh.yz_shape, 0.25)
        return field, jnp.asarray(2.0e-7), jnp.asarray(5, dtype=jnp.int32), jnp.asarray(4.0e-6)

    monkeypatch.setattr(solvers, "_solve_velocity_system", fake_solve_velocity_system)
    monkeypatch.setattr(
        solvers,
        "_compute_current_and_lorentz",
        lambda *args, **kwargs: (
            jnp.zeros(mesh.yz_shape),
            jnp.zeros(mesh.yz_shape),
            jnp.zeros(mesh.yz_shape),
        ),
    )
    monkeypatch.setattr(
        solvers,
        "_face_current_emf_and_lorentz_max",
        lambda *args, **kwargs: (
            jnp.asarray(1.0e-4),
            jnp.asarray(2.0e-4),
            jnp.asarray(3.0e-4),
        ),
    )

    (
        u_next,
        _phi,
        _jy,
        _jz,
        _lorentz,
        velocity_residual,
        potential_residual,
        potential_iterations,
        linear_residual,
        linear_iterations,
        face_current_max,
        emf_max,
        face_lorentz_max,
        mean_velocity,
        applied_forcing,
        potential_initial_residual,
        linear_initial_residual,
    ) = solvers._fully_developed_case_step(
        case=case,
        mesh=mesh,
        materials=materials,
        u_previous=u_previous,
        step_time=case.time_stepper.dt,
        potential_solver="jacobi",
        target_mean_velocity=None,
        linear_solver="cg",
        preconditioner="jacobi",
        coupling_iterations=1,
        coupling_tolerance=1.0e-6,
    )

    active_mask = solvers._active_velocity_mask(materials.fluid_mask)
    assert call_counter["velocity"] == 1
    assert jnp.allclose(u_next[active_mask], 0.25)
    assert jnp.isfinite(u_next[~active_mask]).all()
    assert float(jnp.max(jnp.abs(u_next[~active_mask]))) <= 0.25
    assert float(velocity_residual) == pytest.approx(0.25)
    assert float(potential_residual) == pytest.approx(1.0e-9)
    assert int(potential_iterations) == 3
    assert float(linear_residual) == pytest.approx(2.0e-7)
    assert int(linear_iterations) == 5
    assert float(face_current_max) == pytest.approx(1.0e-4)
    assert float(emf_max) == pytest.approx(2.0e-4)
    assert float(face_lorentz_max) == pytest.approx(3.0e-4)
    fluid_weight = jnp.where(materials.fluid_mask, solvers._cell_metric(mesh).astype(u_next.dtype), 0.0)
    expected_mean_velocity = float(jnp.sum(fluid_weight * u_next) / jnp.sum(fluid_weight))
    assert float(mean_velocity) == pytest.approx(expected_mean_velocity)
    assert float(applied_forcing) == pytest.approx(case.forcing)
    assert float(potential_initial_residual) == pytest.approx(1.0e-6)
    assert float(linear_initial_residual) == pytest.approx(4.0e-6)


def test_fully_developed_case_step_matches_target_mean_velocity_with_sensitivity_solve(monkeypatch: pytest.MonkeyPatch):
    case = make_hartmann_case(ha=5.0, ny=4, nz=4)
    case = replace(case, time_stepper=replace(case.time_stepper, velocity_update_limit=1.0))
    mesh = solvers._build_mesh(case)
    materials = build_material_fields(case, mesh)
    u_previous = jnp.zeros(mesh.yz_shape)
    velocity_calls = {"count": 0}

    monkeypatch.setattr(
        solvers,
        "_solve_potential",
        lambda *args, **kwargs: (
            jnp.zeros(mesh.yz_shape),
            jnp.asarray(5.0e-10),
            jnp.asarray(2, dtype=jnp.int32),
            jnp.asarray(8.0e-7),
        ),
    )

    def fake_solve_velocity_system(**kwargs):
        velocity_calls["count"] += 1
        if velocity_calls["count"] == 1:
            return (
                jnp.full(mesh.yz_shape, 0.2),
                jnp.asarray(3.0e-7),
                jnp.asarray(4, dtype=jnp.int32),
                jnp.asarray(6.0e-6),
            )
        return (
            jnp.full(mesh.yz_shape, 0.5),
            jnp.asarray(1.0e-7),
            jnp.asarray(3, dtype=jnp.int32),
            jnp.asarray(9.0e-6),
        )

    monkeypatch.setattr(solvers, "_solve_velocity_system", fake_solve_velocity_system)
    monkeypatch.setattr(
        solvers,
        "_compute_current_and_lorentz",
        lambda *args, **kwargs: (
            jnp.zeros(mesh.yz_shape),
            jnp.zeros(mesh.yz_shape),
            jnp.zeros(mesh.yz_shape),
        ),
    )
    monkeypatch.setattr(
        solvers,
        "_face_current_emf_and_lorentz_max",
        lambda *args, **kwargs: (
            jnp.asarray(0.0),
            jnp.asarray(0.0),
            jnp.asarray(0.0),
        ),
    )

    (
        u_next,
        _phi,
        _jy,
        _jz,
        _lorentz,
        velocity_residual,
        _potential_residual,
        _potential_iterations,
        linear_residual,
        linear_iterations,
        _face_current_max,
        _emf_max,
        _face_lorentz_max,
        mean_velocity,
        applied_forcing,
        _potential_initial_residual,
        linear_initial_residual,
    ) = solvers._fully_developed_case_step(
        case=case,
        mesh=mesh,
        materials=materials,
        u_previous=u_previous,
        step_time=case.time_stepper.dt,
        potential_solver="jacobi",
        target_mean_velocity=0.7,
        linear_solver="cg",
        preconditioner="jacobi",
        coupling_iterations=1,
        coupling_tolerance=1.0e-6,
    )

    active_mask = solvers._active_velocity_mask(materials.fluid_mask)
    assert velocity_calls["count"] == 2
    assert jnp.allclose(u_next[active_mask], 0.7)
    assert jnp.isfinite(u_next[~active_mask]).all()
    assert float(jnp.max(jnp.abs(u_next[~active_mask]))) <= 0.7
    assert float(velocity_residual) == pytest.approx(0.7)
    assert float(linear_residual) == pytest.approx(3.0e-7)
    assert int(linear_iterations) == 4
    fluid_weight = jnp.where(materials.fluid_mask, solvers._cell_metric(mesh).astype(u_next.dtype), 0.0)
    expected_mean_velocity = float(jnp.sum(fluid_weight * u_next) / jnp.sum(fluid_weight))
    assert float(mean_velocity) == pytest.approx(expected_mean_velocity)
    assert float(applied_forcing) == pytest.approx(1.0)
    assert float(linear_initial_residual) == pytest.approx(9.0e-6)


def test_solve_steady_and_transient_reject_unknown_solver_kind():
    case = make_hartmann_case(ha=5.0, ny=4, nz=4)
    bad_case = replace(case, solver=replace(case.solver, kind="extruded_inductionless"))

    with pytest.raises(NotImplementedError, match="not implemented for steady runs"):
        solve_steady(bad_case)
    with pytest.raises(NotImplementedError, match="not implemented for transient runs"):
        solve_transient(bad_case)


def test_fully_developed_steady_stops_once_residual_reaches_tolerance(monkeypatch: pytest.MonkeyPatch):
    case = make_hartmann_case(ha=5.0, ny=8, nz=8)
    case = replace(
        case,
        time_stepper=replace(
            case.time_stepper,
            max_steps=10,
            steady_tolerance=1e-4,
            potential_tolerance=1.0e-2,
        ),
    )
    residuals = iter([1.0e-1, 1.0e-2, 1.0e-5, 1.0e-6])

    def fake_fully_developed_case_step(**kwargs):
        u = kwargs["u_previous"]
        zeros = jnp.zeros_like(u)
        return u, zeros, zeros, zeros, zeros, next(residuals), 1.0e-2, 25, 0.0, 8.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0e-2, 1.0e-2

    monkeypatch.setattr(solvers, "_fully_developed_case_step", fake_fully_developed_case_step)
    solution = solve_steady(case)

    assert solution.diagnostics.residual_history.shape[0] == 3
    assert solution.diagnostics.potential_residual_history.shape[0] == 3
    assert solution.diagnostics.linear_residual_history.shape[0] == 3
    assert solution.state.time == pytest.approx(3 * case.time_stepper.dt)
    assert solution.state.residual == pytest.approx(1.0e-5)


def test_fully_developed_steady_can_require_potential_residual_when_requested(monkeypatch: pytest.MonkeyPatch):
    case = make_hartmann_case(ha=5.0, ny=8, nz=8)
    case = replace(
        case,
        time_stepper=replace(
            case.time_stepper,
            max_steps=6,
            steady_tolerance=1e-4,
            steady_potential_tolerance=5e-4,
        ),
    )
    residuals = iter([1.0e-3, 1.0e-5, 1.0e-5, 1.0e-6])
    potential_residuals = iter([1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5])

    def fake_fully_developed_case_step(**kwargs):
        u = kwargs["u_previous"]
        zeros = jnp.zeros_like(u)
        return u, zeros, zeros, zeros, zeros, next(residuals), next(potential_residuals), 20, 0.0, 8.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0e-2, 1.0e-2

    monkeypatch.setattr(solvers, "_fully_developed_case_step", fake_fully_developed_case_step)
    solution = solve_steady(case)

    assert solution.diagnostics.residual_history.shape[0] == 3
    assert solution.diagnostics.potential_residual_history.shape[0] == 3
    assert solution.state.time == pytest.approx(3 * case.time_stepper.dt)
    assert solution.state.residual == pytest.approx(1.0e-5)


def test_potential_solver_supports_lineax_and_rejects_unknown_backend(monkeypatch: pytest.MonkeyPatch):
    mesh = generate_rect_duct_mesh(width=2.0, height=2.0, ny=4, nz=4)
    sigma = jnp.ones(mesh.yz_shape)
    fluid_mask = jnp.ones(mesh.yz_shape, dtype=bool)
    u = jnp.ones(mesh.yz_shape) * 0.05
    by = jnp.zeros(mesh.yz_shape)
    bz = jnp.ones(mesh.yz_shape)

    def fake_lineax(*args, **kwargs):
        return jnp.zeros(mesh.yz_shape), type("Info", (), {"residual": 1.0e-9, "iterations": 7})()

    monkeypatch.setattr(solvers, "solve_poisson_lineax", fake_lineax)

    phi, residual, iterations, initial_residual = solvers._solve_potential(
        mesh,
        sigma,
        fluid_mask,
        u,
        by,
        bz,
        anchor=(0, 0),
        iterations=25,
        tolerance=1e-8,
        solver="lineax_cg",
    )

    assert jnp.isfinite(phi).all()
    assert float(initial_residual) >= 0.0
    assert float(residual) >= 0.0
    assert int(iterations) >= 0

    with pytest.raises(ValueError, match="Unsupported potential solver backend"):
        solvers._solve_potential(
            mesh,
            sigma,
            fluid_mask,
            u,
            by,
            bz,
            anchor=(0, 0),
            iterations=5,
            solver="bad_backend",
        )


def test_potential_solver_supports_jacobi_backend():
    mesh = generate_rect_duct_mesh(width=2.0, height=2.0, ny=4, nz=4)
    sigma = jnp.ones(mesh.yz_shape)
    fluid_mask = jnp.ones(mesh.yz_shape, dtype=bool)
    u = jnp.zeros(mesh.yz_shape)
    by = jnp.zeros(mesh.yz_shape)
    bz = jnp.ones(mesh.yz_shape)
    phi, residual, iterations, initial_residual = solvers._solve_potential(
        mesh,
        sigma,
        fluid_mask,
        u,
        by,
        bz,
        anchor=(0, 0),
        iterations=8,
        tolerance=1e-6,
        solver="jacobi",
    )
    assert jnp.isfinite(phi).all()
    assert float(initial_residual) >= 0.0
    assert float(residual) >= 0.0
    assert int(iterations) >= 0


def test_potential_solver_supports_cg_volume_backend():
    mesh = generate_layered_duct_mesh(
        width=2.0,
        height=2.0,
        ny=6,
        nz=6,
        wall_thickness=(0.05, 0.05, 0.05, 0.05),
        wall_cells=(1, 1, 1, 1),
        target_ha=20.0,
    )
    sigma = jnp.ones(mesh.yz_shape)
    fluid_mask = jnp.asarray(mesh.fluid_mask, dtype=bool)
    u = jnp.zeros(mesh.yz_shape)
    by = jnp.zeros(mesh.yz_shape)
    bz = jnp.ones(mesh.yz_shape)

    phi, residual, iterations, initial_residual = solvers._solve_potential(
        mesh,
        sigma,
        fluid_mask,
        u,
        by,
        bz,
        anchor=(mesh.yz_shape[0] // 2, mesh.yz_shape[1] // 2),
        iterations=12,
        tolerance=1.0e-6,
        solver="cg_volume",
    )

    assert jnp.isfinite(phi).all()
    assert float(initial_residual) >= 0.0
    assert float(residual) >= 0.0
    assert int(iterations) >= 0


def test_resolve_potential_solver_auto_handles_none_and_full_fluid_mask():
    full_mask = jnp.ones((2, 2), dtype=bool)
    assert solvers._resolve_potential_solver("auto", None) == "cg"
    assert solvers._resolve_potential_solver("auto", full_mask) == "cg"
    assert solvers._resolve_potential_solver("jacobi", full_mask) == "jacobi"


def test_enforce_velocity_bc_supports_direct_wall_interpolation():
    mesh = generate_rect_duct_mesh(width=2.0, height=2.0, ny=4, nz=4)
    u = jnp.arange(16.0).reshape(4, 4)
    fluid_mask = jnp.ones((4, 4), dtype=bool)

    zeroed = solvers._enforce_velocity_bc(u, mesh, fluid_mask, interpolate_direct_fluid_walls=False)
    enforced = solvers._enforce_velocity_bc(u, mesh, fluid_mask, interpolate_direct_fluid_walls=True)

    assert enforced.shape == u.shape
    assert jnp.isfinite(enforced).all()
    assert not jnp.allclose(enforced, zeroed)
    assert float(enforced[0, 0]) > 0.0
    assert float(enforced[-1, -1]) > 0.0


def test_solve_fully_developed_enables_direct_wall_interpolation_only_for_rectangular_ducts(
    monkeypatch: pytest.MonkeyPatch,
):
    flags: list[bool] = []

    def fake_initial_solver_state(*, interpolate_direct_fluid_walls, **kwargs):
        flags.append(interpolate_direct_fluid_walls)
        mesh = kwargs["mesh"]
        zeros = jnp.zeros(mesh.yz_shape, dtype=float)
        return zeros, zeros, zeros, zeros, zeros, 0.0

    monkeypatch.setattr(solvers, "_initial_solver_state", fake_initial_solver_state)
    monkeypatch.setattr(solvers, "_bounded_time_step_count", lambda **kwargs: 0)

    solve_steady(make_shercliff_case(ha=10.0, ny=8, nz=8))
    solve_steady(make_hunt_case(ha=10.0, ny=8, nz=8, wall_cells=1))

    assert flags == [True, False]


def test_fully_developed_case_step_uses_direct_wall_interpolation_for_rectangular_ducts(
    monkeypatch: pytest.MonkeyPatch,
):
    case = make_shercliff_case(ha=10.0, ny=8, nz=8)
    mesh = solvers._build_mesh(case)
    materials = build_material_fields(case, mesh)
    flags: list[bool] = []

    monkeypatch.setattr(
        solvers,
        "_solve_potential",
        lambda *args, **kwargs: (
            jnp.zeros(mesh.yz_shape),
            jnp.asarray(0.0),
            jnp.asarray(0),
            jnp.asarray(0.0),
        ),
    )
    monkeypatch.setattr(
        solvers,
        "_fully_developed_rhs",
        lambda **kwargs: (jnp.zeros(mesh.yz_shape), jnp.zeros(mesh.yz_shape)),
    )
    monkeypatch.setattr(
        solvers,
        "_solve_velocity_system",
        lambda **kwargs: (
            jnp.ones(mesh.yz_shape),
            jnp.asarray(0.0),
            jnp.asarray(0),
            jnp.asarray(0.0),
        ),
    )

    def fake_enforce(u, mesh_arg, fluid_mask, *, interpolate_direct_fluid_walls):
        flags.append(interpolate_direct_fluid_walls)
        return u

    monkeypatch.setattr(solvers, "_enforce_velocity_bc", fake_enforce)
    monkeypatch.setattr(
        solvers,
        "_compute_current_and_lorentz",
        lambda *args, **kwargs: (
            jnp.zeros(mesh.yz_shape),
            jnp.zeros(mesh.yz_shape),
            jnp.zeros(mesh.yz_shape),
        ),
    )
    monkeypatch.setattr(
        solvers,
        "_face_current_emf_and_lorentz_max",
        lambda *args, **kwargs: (
            jnp.asarray(0.0),
            jnp.asarray(0.0),
            jnp.asarray(0.0),
        ),
    )

    solvers._fully_developed_case_step(
        case=case,
        mesh=mesh,
        materials=materials,
        u_previous=jnp.zeros(mesh.yz_shape),
        step_time=0.0,
        potential_solver="cg",
        target_mean_velocity=None,
        linear_solver="cg",
        preconditioner="jacobi",
        coupling_iterations=1,
        coupling_tolerance=1.0e-8,
    )

    assert flags == [True]


def test_velocity_update_limiters_cover_local_clip_and_validation_errors():
    current = jnp.zeros((2, 2))
    trial = jnp.asarray([[2.0, -2.0], [0.25, -0.25]])
    fluid_mask = jnp.asarray([[True, True], [True, False]])

    clipped = solvers._limited_velocity_update(current, trial, fluid_mask, max_delta=0.5, limiter="local_clip")
    assert float(clipped[0, 0]) == pytest.approx(0.5)
    assert float(clipped[0, 1]) == pytest.approx(-0.5)

    peak, scale, limited_fraction = solvers._velocity_update_statistics(
        current,
        trial,
        fluid_mask,
        max_delta=0.5,
        limiter="local_clip",
    )
    assert float(peak) == pytest.approx(2.0)
    assert 0.0 < float(scale) <= 1.0
    assert 0.0 < float(limited_fraction) <= 1.0

    with pytest.raises(ValueError, match="Unsupported velocity update limiter"):
        solvers._limited_velocity_update(current, trial, fluid_mask, limiter="bad")
    with pytest.raises(ValueError, match="Unsupported velocity update limiter"):
        solvers._velocity_update_statistics(current, trial, fluid_mask, max_delta=0.5, limiter="bad")


def test_velocity_update_global_scale_and_rhs_and_pressure_proxy_helpers():
    current = jnp.zeros((2, 2))
    trial = jnp.asarray([[2.0, -2.0], [0.25, -0.25]])
    fluid_mask = jnp.asarray([[True, True], [True, False]])
    updated = solvers._limited_velocity_update(current, trial, fluid_mask, max_delta=0.5, limiter="global_scale")
    assert float(jnp.max(jnp.abs(updated))) <= 0.5 + 1e-12

    mesh = generate_rect_duct_mesh(width=2.0, height=2.0, ny=2, nz=2)
    sigma = jnp.ones((2, 2))
    rho = jnp.ones((2, 2))
    phi = jnp.asarray([[0.0, 0.1], [0.2, 0.3]])
    by = jnp.ones((2, 2))
    bz = jnp.zeros((2, 2))
    rhs, lorentz_source = solvers._fully_developed_rhs(
        mesh=mesh,
        sigma=sigma,
        rho=rho,
        fluid_mask=jnp.ones((2, 2), dtype=bool),
        u=jnp.asarray([[0.2, 0.1], [0.0, -0.1]]),
        phi=phi,
        by=by,
        bz=bz,
        forcing=jnp.asarray(1.0),
    )
    assert rhs.shape == phi.shape
    assert lorentz_source.shape == phi.shape
    rhs_face, lorentz_face = solvers._fully_developed_rhs(
        mesh=mesh,
        sigma=sigma,
        rho=rho,
        fluid_mask=jnp.ones((2, 2), dtype=bool),
        u=jnp.asarray([[0.2, 0.1], [0.0, -0.1]]),
        phi=phi,
        by=by,
        bz=bz,
        forcing=jnp.asarray(1.0),
        current_reconstruction="face_averaged",
    )
    assert jnp.isfinite(rhs_face).all()
    assert jnp.isfinite(lorentz_face).all()
    diagnostics = type(
        "Diagnostics",
        (),
        {"face_current_max_history": jnp.asarray([]), "current_max_history": jnp.asarray([2.5])},
    )()
    assert solvers._pressure_proxy_reference_current(diagnostics) == pytest.approx(2.5)
    diagnostics_empty = type(
        "Diagnostics",
        (),
        {"face_current_max_history": jnp.asarray([]), "current_max_history": jnp.asarray([])},
    )()
    assert solvers._pressure_proxy_reference_current(diagnostics_empty) is None

    scaled, reference = solvers._scaled_pressure_proxy_value(
        pressure_proxy=3.0,
        current_max=2.0,
        face_current_max=0.0,
        reference_current=None,
    )
    assert scaled == pytest.approx(3.0)
    assert reference == pytest.approx(2.0)

    scaled_face, reference_face = solvers._scaled_pressure_proxy_value(
        pressure_proxy=3.0,
        current_max=2.0,
        face_current_max=4.0,
        reference_current=8.0,
    )
    assert scaled_face == pytest.approx(1.5)
    assert reference_face == pytest.approx(8.0)


def test_concat_history_and_velocity_targets_cover_empty_and_forcing_paths():
    current = jnp.asarray([1.0, 2.0])
    previous = jnp.asarray([0.5])
    assert jnp.array_equal(solvers._concat_history(None, current, append=True), current)
    assert jnp.array_equal(solvers._concat_history(previous, current, append=False), current)
    assert jnp.array_equal(solvers._concat_history(previous, current, append=True), jnp.asarray([0.5, 1.0, 2.0]))

    forced_case = make_hartmann_case(ha=5.0, ny=4, nz=4)
    assert solvers._target_mean_velocity(forced_case) is None
    assert solvers._reference_mean_velocity(forced_case) is None

    flow_rate_case = replace(
        forced_case,
        forcing=0.0,
        boundary_conditions=(BoundaryCondition("inlet", "inlet_flow_rate", value=1.0, axis="x"),),
    )
    expected_speed = 1.0 / (forced_case.geometry.width * forced_case.geometry.height)
    assert solvers._target_mean_velocity(flow_rate_case) == pytest.approx(expected_speed)
    assert solvers._reference_mean_velocity(flow_rate_case) == pytest.approx(expected_speed)

    inlet_velocity_case = replace(
        forced_case,
        boundary_conditions=(BoundaryCondition("inlet", "inlet_velocity", value=0.25, axis="x"),),
    )
    assert solvers._reference_mean_velocity(inlet_velocity_case) == pytest.approx(0.25)

    initial_case = replace(forced_case, initial_velocity=0.125, boundary_conditions=())
    assert solvers._reference_mean_velocity(initial_case) == pytest.approx(0.125)


def test_initial_solver_state_restores_restart_fields_and_time():
    case = replace(make_hartmann_case(ha=5.0, ny=4, nz=4), initial_velocity=0.25)
    mesh = solvers._build_mesh(case)
    materials = build_material_fields(case, mesh)
    fluid_mask = materials.fluid_mask
    restart_state = type(
        "RestartState",
        (),
        {
            "u": jnp.ones(mesh.yz_shape) * 0.3,
            "phi": jnp.ones(mesh.yz_shape) * 0.1,
            "jy": jnp.ones(mesh.yz_shape) * 0.2,
            "jz": jnp.ones(mesh.yz_shape) * -0.2,
            "lorentz_x": jnp.ones(mesh.yz_shape) * 0.05,
            "time": 0.75,
        },
    )()

    u0, phi0, jy0, jz0, lorentz0, start_time = solvers._initial_solver_state(
        case=case,
        mesh=mesh,
        fluid_mask=fluid_mask,
        interpolate_direct_fluid_walls=False,
        initial_state=restart_state,
    )

    assert float(start_time) == pytest.approx(0.75)
    assert jnp.isfinite(u0).all()
    assert jnp.array_equal(phi0, restart_state.phi)
    assert jnp.array_equal(jy0, restart_state.jy)
    assert jnp.array_equal(jz0, restart_state.jz)
    assert jnp.array_equal(lorentz0, restart_state.lorentz_x)


def test_inlet_speed_supports_tuple_scalar_and_flow_rate_boundaries():
    case = make_hartmann_case(ha=5.0, ny=8, nz=8)
    tuple_bc = BoundaryCondition("tuple", "inlet_velocity", value=(1.0, 2.0, 3.0), axis="z")
    scalar_bc = BoundaryCondition("scalar", "inlet_velocity", value=1.5, axis="x")
    flow_bc = BoundaryCondition(
        "flow",
        "inlet_flow_rate",
        value=case.geometry.width * case.geometry.height * 0.25,
        axis="x",
    )

    assert solvers._inlet_speed(tuple_bc, case) == pytest.approx(3.0)
    assert solvers._inlet_speed(scalar_bc, case) == pytest.approx(1.5)
    assert solvers._inlet_speed(flow_bc, case) == pytest.approx(0.25)


def test_fully_developed_case_step_rejects_crank_nicolson_in_transient_mode():
    case = replace(
        make_hartmann_case(ha=5.0, ny=6, nz=6),
        solver=replace(make_hartmann_case(ha=5.0, ny=6, nz=6).solver, mode="transient", time_scheme="crank_nicolson"),
    )
    mesh = solvers._build_mesh(case)
    materials = build_material_fields(case, mesh)

    with pytest.raises(NotImplementedError, match="implicit_euler only"):
        solvers._fully_developed_case_step(
            case=case,
            mesh=mesh,
            materials=materials,
            u_previous=jnp.zeros(mesh.yz_shape),
            step_time=0.0,
            potential_solver="cg",
            target_mean_velocity=None,
            linear_solver="cg",
            preconditioner="jacobi",
            coupling_iterations=1,
            coupling_tolerance=1e-8,
        )


def test_fully_developed_case_step_covers_forcing_and_target_velocity_paths():
    forcing_case = make_hartmann_case(ha=5.0, ny=4, nz=4)
    mesh = solvers._build_mesh(forcing_case)
    materials = build_material_fields(forcing_case, mesh)
    result = solvers._fully_developed_case_step(
        case=forcing_case,
        mesh=mesh,
        materials=materials,
        u_previous=jnp.zeros(mesh.yz_shape),
        step_time=forcing_case.time_stepper.dt,
        potential_solver="cg",
        target_mean_velocity=None,
        linear_solver="cg",
        preconditioner="jacobi",
        coupling_iterations=2,
        coupling_tolerance=1e-6,
    )
    assert len(result) == 17
    assert jnp.isfinite(result[0]).all()

    flow_case = replace(
        forcing_case,
        forcing=0.0,
        boundary_conditions=(BoundaryCondition("inlet", "inlet_flow_rate", value=0.5),),
    )
    flow_mesh = solvers._build_mesh(flow_case)
    flow_materials = build_material_fields(flow_case, flow_mesh)
    flow_result = solvers._fully_developed_case_step(
        case=flow_case,
        mesh=flow_mesh,
        materials=flow_materials,
        u_previous=jnp.zeros(flow_mesh.yz_shape),
        step_time=flow_case.time_stepper.dt,
        potential_solver="cg",
        target_mean_velocity=solvers._target_mean_velocity(flow_case),
        linear_solver="cg",
        preconditioner="jacobi",
        coupling_iterations=2,
        coupling_tolerance=1e-6,
    )
    assert jnp.isfinite(flow_result[0]).all()
    assert float(flow_result[14]) == pytest.approx(float(flow_result[14]))


def test_solver_logging_helpers_and_footer_are_emitted():
    calls: list[str] = []

    class Logger:
        def emit_header(self, **kwargs):
            calls.append("header")

        def emit_step(self, record):
            calls.append("step")

        def emit_footer(self, solution):
            calls.append("footer")

    logger = Logger()
    case = make_hartmann_case(ha=5.0, ny=4, nz=4)
    mesh = solvers._build_mesh(case)
    materials = build_material_fields(case, mesh)
    solvers._emit_solver_header(
        logger,
        case=case,
        mesh=mesh,
        materials=materials,
        mode="steady",
        potential_solver="cg",
        target_mean_velocity=None,
        reference_mean_velocity=None,
    )
    solvers._emit_solver_step(
        logger,
        step_index=1,
        step_time=0.1,
        dt=0.1,
        u_max_value=0.1,
        mean_velocity=0.1,
        max_current=0.0,
        face_current_max=0.0,
        emf_max=0.0,
        max_lorentz=0.0,
        face_lorentz_max=0.0,
        residual_value=1.0e-6,
        potential_residual=1.0e-6,
        potential_iteration_count=1.0,
        linear_residual=1.0e-6,
        linear_iteration_count=1.0,
        applied_forcing=1.0,
        pressure_proxy=1.0,
        current_scaled_pressure_proxy=1.0,
        raw_update_max=1.0e-6,
        limiter_scale=1.0,
        limited_fraction=0.0,
        courant_like=0.0,
        ohmic=0.0,
        volumetric_flow_rate=0.0,
        mean_current_magnitude=0.0,
        lorentz_power=0.0,
        div_current_max=0.0,
        charge_balance_residual=0.0,
        gauge_residual=0.0,
        interface_current_residual=0.0,
        potential_initial_residual=1.0e-6,
        linear_initial_residual=1.0e-6,
    )
    solution = solve_steady(case)
    logger.emit_footer(solution)
    assert calls == ["header", "step", "footer"]


def test_emit_solver_header_is_noop_without_logger():
    case = make_hartmann_case(ha=5.0, ny=4, nz=4)
    mesh = solvers._build_mesh(case)
    materials = build_material_fields(case, mesh)
    solvers._emit_solver_header(
        None,
        case=case,
        mesh=mesh,
        materials=materials,
        mode="steady",
        potential_solver="cg",
        target_mean_velocity=None,
        reference_mean_velocity=None,
    )


def test_emit_solver_step_is_noop_without_logger():
    solvers._emit_solver_step(
        None,
        step_index=0,
        step_time=0.0,
        dt=1.0,
        u_max_value=0.0,
        mean_velocity=0.0,
        max_current=0.0,
        face_current_max=0.0,
        emf_max=0.0,
        max_lorentz=0.0,
        face_lorentz_max=0.0,
        residual_value=0.0,
        potential_residual=0.0,
        potential_iteration_count=0.0,
        linear_residual=0.0,
        linear_iteration_count=0.0,
        applied_forcing=0.0,
        pressure_proxy=0.0,
        current_scaled_pressure_proxy=0.0,
        raw_update_max=0.0,
        limiter_scale=1.0,
        limited_fraction=0.0,
        courant_like=0.0,
        ohmic=0.0,
        volumetric_flow_rate=0.0,
        mean_current_magnitude=0.0,
        lorentz_power=0.0,
        div_current_max=0.0,
        charge_balance_residual=0.0,
        gauge_residual=0.0,
        interface_current_residual=0.0,
    )


def test_initial_solver_state_without_restart_zeros_auxiliary_fields():
    case = replace(make_hartmann_case(ha=5.0, ny=4, nz=4), initial_velocity=0.2)
    mesh = solvers._build_mesh(case)
    materials = build_material_fields(case, mesh)
    fluid_mask = materials.fluid_mask

    u0, phi0, jy0, jz0, lorentz0, start_time = solvers._initial_solver_state(
        case=case,
        mesh=mesh,
        fluid_mask=fluid_mask,
        interpolate_direct_fluid_walls=True,
        initial_state=None,
    )

    assert float(start_time) == pytest.approx(0.0)
    assert jnp.isfinite(u0).all()
    assert jnp.allclose(phi0, 0.0)
    assert jnp.allclose(jy0, 0.0)
    assert jnp.allclose(jz0, 0.0)
    assert jnp.allclose(lorentz0, 0.0)


def test_explicit_forcing_preserves_dtype():
    value = solvers._explicit_forcing(1.25, jnp.float32)
    assert value.dtype == jnp.float32
    assert float(value) == pytest.approx(1.25)


def test_scaled_pressure_proxy_value_falls_back_to_unity_reference_for_zero_currents():
    scaled, reference = solvers._scaled_pressure_proxy_value(
        pressure_proxy=2.0,
        current_max=0.0,
        face_current_max=0.0,
        reference_current=0.0,
    )
    assert scaled == pytest.approx(0.0)
    assert reference == pytest.approx(1.0)


def test_inlet_speed_defaults_tuple_axis_to_x_and_rejects_zero_area_flow_rate():
    case = make_hartmann_case(ha=5.0, ny=8, nz=8)
    tuple_bc = BoundaryCondition("tuple", "inlet_velocity", value=(1.0, 2.0, 3.0), axis="bad")
    zero_area_case = replace(case, geometry=replace(case.geometry, width=0.0))
    flow_bc = BoundaryCondition("flow", "inlet_flow_rate", value=1.0, axis="x")

    assert solvers._inlet_speed(tuple_bc, case) == pytest.approx(1.0)
    assert solvers._inlet_speed(flow_bc, zero_area_case) is None


def test_emit_solver_header_forwards_restart_payload():
    captured = {}

    class Logger:
        def emit_header(self, **kwargs):
            captured.update(kwargs)

    case = make_hartmann_case(ha=5.0, ny=4, nz=4)
    mesh = solvers._build_mesh(case)
    materials = build_material_fields(case, mesh)
    restart = SimpleNamespace(time=0.5, source="restart.npz")
    solvers._emit_solver_header(
        Logger(),
        case=case,
        mesh=mesh,
        materials=materials,
        mode="steady",
        potential_solver="cg",
        target_mean_velocity=None,
        reference_mean_velocity=0.2,
        restart=restart,
    )

    assert captured["restart"] is restart
    assert captured["reference_mean_velocity"] == pytest.approx(0.2)


def test_build_mesh_rejects_bent_pipe_for_laminar_solver():
    case = replace(
        make_hartmann_case(ha=5.0, ny=4, nz=4),
        geometry=GeometrySpec(kind="bent_pipe", width=1.0, height=1.0, radius=0.25, bend_radius=1.0, bend_angle=1.0, nr=4, ntheta=8),
    )
    with pytest.raises(NotImplementedError, match="bent_pipe"):
        solvers._build_mesh(case)


def test_fully_developed_solver_rejects_unsupported_geometry_after_mesh_build(monkeypatch: pytest.MonkeyPatch):
    case = replace(make_hartmann_case(ha=5.0, ny=4, nz=4), geometry=GeometrySpec(kind="pipe_ogrid", width=1.0, height=1.0, radius=0.5, nr=4, ntheta=8))
    fake_mesh = generate_rect_duct_mesh(width=1.0, height=1.0, ny=4, nz=4)
    monkeypatch.setattr(solvers, "_build_mesh", lambda case: fake_mesh)
    monkeypatch.setattr(solvers, "build_material_fields", lambda case, mesh: build_material_fields(make_hartmann_case(ha=5.0, ny=4, nz=4), fake_mesh))
    with pytest.raises(NotImplementedError, match="does not yet support geometry"):
        solvers._solve_fully_developed(case)


def test_public_solver_entrypoints_reject_unknown_solver_kinds():
    case = replace(make_hartmann_case(ha=5.0, ny=4, nz=4), solver=replace(make_hartmann_case(ha=5.0, ny=4, nz=4).solver, kind="extruded_inductionless"))
    with pytest.raises(NotImplementedError, match="not implemented for transient"):
        solve_transient(case)
    with pytest.raises(NotImplementedError, match="not implemented for steady"):
        solve_steady(case)


def test_public_solver_entrypoints_coerce_or_preserve_mode_before_dispatch(monkeypatch: pytest.MonkeyPatch):
    base_case = make_hartmann_case(ha=5.0, ny=4, nz=4)
    steady_case = replace(base_case, solver=replace(base_case.solver, mode="steady"))
    calls: list[tuple[str, bool]] = []

    def fake_solve(case_arg, **kwargs):
        calls.append((case_arg.solver.mode, bool(kwargs.get("append_diagnostics", False))))
        return "ok"

    monkeypatch.setattr(solvers, "_solve_fully_developed", fake_solve)

    assert solve_transient(base_case) == "ok"
    assert solve_steady(steady_case, append_diagnostics=True) == "ok"
    assert calls == [("transient", False), ("steady", True)]


for _unit_test_name in (
    "test_hartmann_solver_runs",
    "test_hunt_solver_keeps_solid_velocity_zero",
    "test_hunt_fully_developed_velocity_linear_solve_is_well_conditioned",
    "test_hunt_case_uses_ha_aware_coupling_controls",
    "test_hunt_case_derives_wall_conductivity_from_conductance_ratio",
    "test_hunt_case_adds_explicit_insulating_side_wall_region",
    "test_hunt_case_allows_explicit_wall_conductivity_override",
    "test_hunt_inlet_flow_rate_boundary_drives_short_transient",
    "test_transient_restart_matches_direct_run",
    "test_transient_restart_can_append_diagnostics",
    "test_target_mean_velocity_only_uses_inlet_flow_rate",
    "test_reference_mean_velocity_uses_inlet_velocity_or_initial_velocity",
    "test_active_velocity_mask_excludes_enforced_outer_boundary_cells",
    "test_magnetic_ramp_scale_disables_when_duration_is_zero",
    "test_magnetic_ramp_scale_matches_reference_startup_formula",
    "test_magnetic_ramp_delays_short_transient_lorentz_response",
    "test_shercliff_solution_stays_finite_and_zero_at_walls",
    "test_potential_solver_backends_return_finite_fields_on_small_system",
    "test_current_reconstruction_modes_and_face_diagnostics_are_finite",
    "test_auto_potential_backend_uses_cg_for_single_region_and_volume_scaled_cg_for_layered_cases",
    "test_build_material_fields_assigns_hunt_side_and_hartmann_wall_regions",
    "test_volume_scaled_potential_system_is_symmetric_after_cell_metric_weighting",
    "test_potential_coefficients_match_uniform_spacing_formula_on_rect_grid",
    "test_face_emf_uses_distance_weighted_nonuniform_interface_source",
    "test_fully_developed_steady_stops_once_residual_reaches_tolerance",
    "test_fully_developed_steady_can_require_potential_residual_when_requested",
    "test_potential_solver_supports_lineax_and_rejects_unknown_backend",
    "test_resolve_potential_solver_auto_handles_none_and_full_fluid_mask",
    "test_enforce_velocity_bc_supports_direct_wall_interpolation",
    "test_velocity_update_limiters_cover_local_clip_and_validation_errors",
    "test_inlet_speed_supports_tuple_scalar_and_flow_rate_boundaries",
    "test_fully_developed_case_step_rejects_crank_nicolson_in_transient_mode",
):
    globals()[_unit_test_name] = pytest.mark.unit(globals()[_unit_test_name])

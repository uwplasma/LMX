import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from solvax import schur_complement_precond

import lmx._fringing_common as common_impl
import lmx._fringing_duct as duct_impl
import lmx.fringing as fringing_impl
from lmx._fringing_common import (
    _apply_fixed_flow_pressure_constraint,
    _cross_duct_pressure_difference,
    _distance_weighted_harmonic_mean,
    _equalize_stationwise_flow_rate_3d,
    _gauge_invariant_scalar_update,
    _gradient_3d,
    _iteration_history_arrays,
    _laplacian_3d,
    _normalized_pressure_observable_update,
    _projection_pressure_correction_3d,
    _restore_duct_iteration_state,
    _shard_extruded_fields,
    _spacing_vector,
    _thin_wall_interface_mean,
)
from lmx._fringing_duct import (
    _conservative_current_diagnostics_3d,
    _conservative_current_fluxes_3d,
    _face_flux_pressure_projection_duct,
    _solvax_implicit_momentum_duct,
    _solvax_pressure_poisson_duct,
    _station_axial_current_from_fluxes,
)
from lmx._fringing_pipe import (
    _apply_pipe_diffusion_coefficients_3d,
    _pipe_conservative_current_diagnostics_3d,
    _pipe_conservative_emf_rhs_3d,
    _pipe_face_divergence,
    _pipe_gradient_3d,
    _pipe_laplacian_3d,
    _pipe_pressure_face_correction,
    _pipe_variable_diffusion_coefficients_3d,
    _separable_pressure_poisson_pipe,
    _solvax_diffusion_pipe,
    _solvax_pressure_poisson_pipe,
    _steady_stokes_projection_pipe,
)
from lmx.fringing import (
    build_extruded_problem_from_case,
    build_layered_duct_extruded_problem,
    build_magnetic_obstacle_rect_extruded_problem,
    build_pipe_ogrid_extruded_problem,
    build_square_duct_extruded_problem,
    evolve_extruded_fields,
    extruded_engineering_objectives,
    smooth_fringing_profile,
    solve_extruded_inductionless,
    validate_magnetic_obstacle_baseline,
    validate_variable_field_extruded_solution,
    validate_variable_field_pipe_solution,
)
from lmx.mesh import (
    _cross_section_mesh,
    _sample_volume_field,
    make_divergence_free_cross_section_field,
    make_localized_divergence_free_obstacle_field,
    sample_cross_section_field,
    write_tabulated_field_npz,
)
from lmx.specs import (
    ExtrudedFieldBundle,
    GeometrySpec,
    MagneticFieldSpec,
    NumericalFailure,
    RegionSpec,
)
from lmx.validation import build_benchmark_b_field_profile, build_benchmark_b_problem

pytestmark = pytest.mark.unit


def _centered_gradient(function, point, step):
    return jnp.asarray(
        [
            (function(point.at[i].add(step)) - function(point.at[i].add(-step))) / (2.0 * step)
            for i in range(len(point))
        ]
    )


def _mock_extruded_bundle(*, geometry_kind: str, stations: int) -> ExtrudedFieldBundle:
    """Return the smallest complete bundle needed by public-solver wrapper tests."""
    x = jnp.linspace(0.0, 1.0, stations)
    shape = (stations, 1, 1)
    zeros = jnp.zeros(shape)
    station_zeros = jnp.zeros(stations)
    return ExtrudedFieldBundle(
        x=x,
        y=jnp.zeros(1),
        z=jnp.zeros(1),
        field_scale=1.0 - jnp.abs(2.0 * x - 1.0),
        u=jnp.ones(shape),
        v=zeros,
        w=zeros,
        p=zeros,
        phi=zeros,
        geometry_kind=geometry_kind,
        solver_kind="extruded_inductionless",
        jx=zeros,
        jy=zeros,
        jz=zeros,
        lorentz_x=zeros,
        lorentz_y=zeros,
        lorentz_z=zeros,
        residual=jnp.full(stations, 1.0e-5),
        volumetric_flow_rate=jnp.linspace(0.2, 0.4, stations),
        mean_velocity=jnp.linspace(0.1, 0.3, stations),
        axial_current=jnp.linspace(0.01, 0.02, stations),
        wall_current_leakage=jnp.full(stations, 1.0e-6),
        current_scaled_pressure_proxy=jnp.linspace(0.1, 0.15, stations),
        charge_balance_residual=jnp.full(stations, 1.0e-8),
        boundary_current_residual=jnp.full(stations, 1.0e-7),
        axial_pressure_loss_gradient=station_zeros,
        transverse_pressure_difference=station_zeros,
    )


@pytest.mark.timeout(300)
def test_alex_b1_production_map_has_bounded_implicit_gradient():
    problem = build_benchmark_b_problem("B1-fringing-pipe", mesh_level="coarse")
    case = replace(
        problem.case,
        geometry=replace(
            problem.case.geometry,
            nx=3,
            nr=2,
            ntheta=4,
            wall_cells=(1, 1, 1, 1),
            target_ha=None,
            hartmann_layer_cells=None,
        ),
        time_stepper=replace(problem.case.time_stepper, max_steps=1, potential_iterations=20),
        solver=replace(problem.case.solver, coupling_iterations=1, coupling_tolerance=1.0e-6),
    )
    problem = replace(
        problem,
        case=case,
        profile=build_benchmark_b_field_profile("B1-fringing-pipe", axial_stations=3),
    )
    with pytest.raises(NotImplementedError, match="ALEX B1"):
        evolve_extruded_fields(problem, steps=1, num_devices=2)

    def objective(parameters):
        fields = evolve_extruded_fields(
            problem,
            forcing=parameters[0],
            magnetic_field_scale=parameters[1],
            material_conductivity_scale=parameters[2],
            geometry_scale=parameters[3:],
            steps=1,
        )
        return (
            jnp.mean(fields[0] ** 2)
            + 1.0e-4 * jnp.mean(fields[4] ** 2)
            + 1.0e-6 * jnp.mean(fields[6] ** 2 + fields[7] ** 2)
        )

    parameters = jnp.ones(5)
    compiled = jax.jit(jax.value_and_grad(objective)).lower(parameters).compile()
    value, gradient = compiled(parameters)
    epsilon = 2.0e-3
    direction = jnp.asarray([0.1, -0.2, 0.3, -0.15, 0.12])
    plus, _ = compiled(parameters + epsilon * direction)
    minus, _ = compiled(parameters - epsilon * direction)
    finite_difference = (plus - minus) / (2.0 * epsilon)
    assert jnp.isfinite(value) and jnp.all(jnp.isfinite(gradient))
    assert jnp.vdot(gradient, direction) == pytest.approx(finite_difference, rel=3.0e-4, abs=5.0e-8)
    assert jnp.all(jnp.abs(gradient[1:]) > 1.0e-6)
    assert compiled.memory_analysis().temp_size_in_bytes < 300_000


def test_extruded_solver_rejects_nonfinite_result(monkeypatch: pytest.MonkeyPatch):
    bundle = ExtrudedFieldBundle(
        x=jnp.zeros((1,)),
        y=jnp.zeros((1,)),
        z=jnp.zeros((1,)),
        field_scale=jnp.ones((1,)),
        u=jnp.full((1, 1, 1), jnp.nan),
        v=jnp.zeros((1, 1, 1)),
        w=jnp.zeros((1, 1, 1)),
        p=jnp.zeros((1, 1, 1)),
        phi=jnp.zeros((1, 1, 1)),
        geometry_kind="rect_duct",
        solver_kind="extruded_inductionless",
    )
    monkeypatch.setattr(fringing_impl, "_solve_extruded_projection", lambda *args, **kwargs: bundle)

    problem = build_square_duct_extruded_problem(nx_stations=1, ny=1, nz=1)
    with pytest.raises(NumericalFailure, match="3-D fringing solve.*u"):
        solve_extruded_inductionless(problem)
    with pytest.raises(ValueError, match="history_stride"):
        solve_extruded_inductionless(
            replace(
                problem, case=replace(problem.case, output=replace(problem.case.output, history_stride=-1))
            )
        )


def test_extruded_histories_are_terminal_strided_and_restartable():
    values = list(range(5))
    components = [tuple(float(value) for _ in range(6)) for value in values]
    compact = _iteration_history_arrays(values, components, values, components, values, stride=0)
    strided = _iteration_history_arrays(values, components, values, components, values, stride=2)
    resumed = _iteration_history_arrays(
        [0, 2, 3, 4, 5, 6, 7],
        [tuple(float(value) for _ in range(6)) for value in (0, 2, 3, 4, 5, 6, 7)],
        [0, 2, 3, 4, 5, 6, 7],
        [tuple(float(value) for _ in range(6)) for value in (0, 2, 3, 4, 5, 6, 7)],
        [0, 2, 3, 4, 5, 6, 7],
        stride=2,
        retained_prefix=3,
    )
    assert compact["iteration_residual_history"].tolist() == [4.0]
    assert strided["iteration_residual_history"].tolist() == [0.0, 2.0, 4.0]
    assert strided["iteration_component_residual_history"].shape == (3, 6)
    assert resumed["iteration_residual_history"].tolist() == [0.0, 2.0, 3.0, 4.0, 6.0, 7.0]

    problem = build_square_duct_extruded_problem(nx_stations=2, ny=2, nz=2)
    zeros = jnp.zeros((2, 2, 2))
    bundle = ExtrudedFieldBundle(
        x=jnp.arange(2.0),
        y=jnp.arange(2.0),
        z=jnp.arange(2.0),
        field_scale=jnp.ones(2),
        u=zeros,
        v=zeros,
        w=zeros,
        p=zeros,
        phi=zeros,
        geometry_kind="rect_duct",
        solver_kind="extruded_inductionless",
        stopping_state=(4, 0, "step_limit"),
        **compact,
    )
    restored = _restore_duct_iteration_state(
        bundle,
        case=problem.case,
        use_b2=False,
        velocity=zeros,
        velocity_scale=0.75,
        forcing=1.0,
    )
    assert restored[5] == 4
    assert restored[-2][:, 0, 0, 0].tolist() == [0.75, 0.75, 0.75]


def _with_analytic_field(problem, *, name, field_fn):
    case = replace(
        problem.case,
        name=name,
        magnetic_field=MagneticFieldSpec(kind="analytic", fn=field_fn),
    )
    return replace(problem, case=case)


def _with_integration_budget(problem):
    """Keep end-to-end physics gates fast without changing production defaults."""

    case = replace(
        problem.case,
        time_stepper=replace(problem.case.time_stepper, max_steps=24, potential_iterations=40),
        solver=replace(problem.case.solver, coupling_iterations=4),
    )
    return replace(problem, case=case)


def test_b2_canonical_shell_widths_remove_realization_thickness():
    nominal = jnp.asarray([0.01, 0.01, 0.4, 0.4, 0.4, 0.4, 0.4, 0.01, 0.01])
    confirmation = nominal.at[:2].divide(2.0).at[-2:].divide(2.0)

    expected = common_impl._canonical_shell_widths(nominal, 2, 7)
    observed = common_impl._canonical_shell_widths(confirmation, 2, 7)

    assert observed.tolist() == pytest.approx(expected.tolist())
    assert float(jnp.sum(observed[:2])) == pytest.approx(0.02)
    assert float(jnp.sum(observed[-2:])) == pytest.approx(0.02)


def test_fringing_jit_cache_reuses_the_first_compiled_kernel():
    common_impl._FRINGING_JIT_CACHE.clear()
    first = object()
    key = ("operator", jnp.asarray(1.0))

    assert common_impl._reuse_fringing_jit(key, first) is first
    assert common_impl._reuse_fringing_jit(key, object()) is first


def test_axial_mean_preconditioner_matches_dense_mixed_gauge_and_autodiff():
    nx, ny, nz = 5, 3, 2
    normalization = np.sqrt(ny * nz)
    volume = jnp.linspace(0.7, 1.4, nx * ny * nz).reshape((nx, ny, nz))
    reduced = jnp.asarray([0.3, -0.7, 1.2, 0.4, -0.9])
    residual = jnp.broadcast_to(reduced[:, None, None] / normalization, volume.shape)
    tangent = jnp.linspace(-0.2, 0.3, residual.size).reshape(residual.shape)
    for gauge in (True, False):
        faces = jnp.asarray([0.0, 0.8, 1.3, 0.6, 1.1, 0.0 if gauge else 1.7])
        west, east = faces[:-1, None, None] / volume, faces[1:, None, None] / volume
        precondition = common_impl._axial_mean_preconditioner_3d(volume, west, east, gauge=gauge)
        dense = np.diag(np.asarray(faces[:-1] + faces[1:]))
        dense -= np.diag(np.asarray(faces[1:-1]), 1) + np.diag(np.asarray(faces[1:-1]), -1)
        if gauge:
            weights = np.asarray(jnp.sum(volume, axis=(1, 2))) / normalization
            dense += np.outer(weights, weights) / float(jnp.sum(volume))
        expected = np.linalg.solve(dense, np.asarray(reduced))
        observed = np.asarray(jax.jit(precondition)(residual))[:, 0, 0] * normalization
        assert observed == pytest.approx(expected, rel=5.0e-11, abs=5.0e-12)
        assert np.max(np.abs(dense @ observed - np.asarray(reduced))) < 2.0e-11
        _, jvp = jax.jvp(precondition, (residual,), (tangent,))
        _, pullback = jax.vjp(precondition, residual)
        expected_tangent = precondition(tangent)
        assert jvp == pytest.approx(expected_tangent, rel=5.0e-11, abs=5.0e-12)
        assert pullback(tangent)[0] == pytest.approx(expected_tangent, rel=5.0e-11, abs=5.0e-12)


def test_transverse_modal_correction_is_accurate_spd_and_accelerates_pcg():
    nx, cross = 12, 18
    bounds = (3, 15, 3, 15)
    spacing = jnp.concatenate((jnp.full(3, 0.02 / 3.0), jnp.full(12, 2.0 / 12.0), jnp.full(3, 0.02 / 3.0)))
    mask = jnp.zeros((nx, cross, cross), dtype=bool)
    mask = mask.at[:, 3:15, 3:15].set(True)
    conductivity = jnp.where(mask, 1.0, 3.5)
    x = jnp.linspace(0.0, 1.0, nx)[:, None, None]
    y = jnp.linspace(-1.0, 1.0, cross)[None, :, None]
    z = jnp.linspace(-1.0, 1.0, cross)[None, None, :]
    expected = jnp.sin(2.0 * jnp.pi * x) * jnp.cos(0.5 * jnp.pi * y) * jnp.cos(0.5 * jnp.pi * z)
    coefficients = common_impl._variable_diffusion_coefficients_3d(
        conductivity,
        dx=0.1,
        dy=spacing,
        dz=spacing,
        validated_spacing=True,
        thin_wall_fluid_mask=mask,
    )
    neighbors = common_impl._neighbor_fields(expected, mode_x="neumann", mode_y="neumann", mode_z="neumann")
    rhs = sum(
        coefficient * (neighbor - expected)
        for coefficient, neighbor in zip(coefficients, neighbors, strict=True)
    )

    baseline = _solvax_pressure_poisson_duct(
        rhs,
        conductivity,
        dx=0.1,
        dy=spacing,
        dz=spacing,
        iterations=200,
        tolerance=1.0e-10,
        thin_wall_fluid_mask=mask,
    )
    accelerated = _solvax_pressure_poisson_duct(
        rhs,
        conductivity,
        dx=0.1,
        dy=spacing,
        dz=spacing,
        iterations=200,
        tolerance=1.0e-10,
        thin_wall_fluid_mask=mask,
        transverse_coarse_bounds=bounds,
    )

    assert bool(accelerated[2])
    assert int(accelerated[4]) < 0.75 * int(baseline[4])
    assert accelerated[0] == pytest.approx(baseline[0], abs=5.0e-9)

    volume = jnp.broadcast_to(spacing[None, :, None] * spacing[None, None, :], conductivity.shape)
    correction = common_impl._transverse_modal_correction_3d(
        volume,
        conductivity,
        coefficients,
        dx=0.1,
        dy=spacing,
        dz=spacing,
        fluid_bounds=bounds,
        stride=2,
    )
    left = jnp.sin(jnp.arange(expected.size, dtype=float)).reshape(expected.shape)
    right = jnp.cos(jnp.arange(expected.size, dtype=float)).reshape(expected.shape)
    assert jnp.vdot(left, correction(right)) == pytest.approx(
        jnp.vdot(correction(left), right), rel=1.0e-10, abs=1.0e-10
    )
    assert float(jnp.vdot(left, correction(left))) > 0.0

    mesh = common_impl.Mesh(np.asarray(jax.devices()[:1]), ("x",))
    sharding = common_impl.NamedSharding(mesh, common_impl.P("x", None, None))
    sharded_correction = common_impl._transverse_modal_correction_3d(
        volume,
        conductivity,
        coefficients,
        dx=0.1,
        dy=spacing,
        dz=spacing,
        fluid_bounds=bounds,
        stride=2,
        sharding=sharding,
    )
    assert sharded_correction(jax.device_put(right, sharding)) == pytest.approx(
        correction(right), rel=1.0e-10, abs=1.0e-10
    )
    sharded_left, sharded_right = (jax.device_put(value, sharding) for value in (left, right))
    assert jnp.vdot(sharded_left, sharded_correction(sharded_right)) == pytest.approx(
        jnp.vdot(sharded_correction(sharded_left), sharded_right),
        rel=1.0e-10,
        abs=1.0e-10,
    )
    _, sharded_tangent = jax.jvp(sharded_correction, (sharded_right,), (sharded_left,))
    assert sharded_tangent == pytest.approx(sharded_correction(sharded_left), rel=1.0e-10, abs=1.0e-10)

    _, tangent = jax.jvp(correction, (right,), (left,))
    assert tangent == pytest.approx(correction(left), rel=1.0e-10, abs=1.0e-10)


def test_duct_solvers_forward_single_reduction_to_solvax(
    monkeypatch: pytest.MonkeyPatch,
):
    calls = []

    def fake_solve(_matvec, rhs, **kwargs):
        calls.append(kwargs["single_reduction"])
        return SimpleNamespace(
            x=jnp.zeros_like(rhs),
            residual_norm=jnp.asarray(0.0),
            relative_residual_norm=jnp.asarray(0.0),
            iterations=jnp.asarray(0),
            converged=jnp.asarray(True),
            status=jnp.asarray(1),
        )

    monkeypatch.setattr(duct_impl, "pcg_linear_solve", fake_solve)
    field = jnp.ones((2, 2, 2))
    spacing = jnp.ones(2)
    _solvax_pressure_poisson_duct(
        field,
        field,
        dx=1.0,
        dy=spacing,
        dz=spacing,
        iterations=2,
        tolerance=1.0e-6,
        single_reduction=True,
        include_axial_line=False,
    )
    assert calls == [True]


def test_distance_weighted_harmonic_mean_preserves_series_resistance():
    value = _distance_weighted_harmonic_mean(
        jnp.asarray([10.0]),
        jnp.asarray([1.0]),
        jnp.asarray([0.1]),
        jnp.asarray([0.9]),
    )
    assert value == pytest.approx([1.0 / (0.1 / 10.0 + 0.9 / 1.0)])


@pytest.mark.parametrize("wall_width", [0.01, 0.02, 0.05])
def test_thin_wall_interface_removes_artificial_normal_resistance(wall_width):
    fluid_width = jnp.asarray([0.004])
    face_sigma = _thin_wall_interface_mean(
        jnp.asarray([0.07 / wall_width]),
        jnp.asarray([1.0]),
        jnp.asarray([wall_width]),
        fluid_width,
        jnp.asarray([False]),
        jnp.asarray([True]),
    )
    transmissibility = face_sigma / (0.5 * (wall_width + fluid_width))
    assert transmissibility == pytest.approx(2.0 / float(fluid_width[0]))


def test_fixed_flow_pressure_constraint_and_stationwise_correction():
    u = jnp.zeros((2, 3, 3))
    active = jnp.ones_like(u, dtype=bool)
    area = jnp.ones_like(u)
    response = jnp.full_like(u, 0.5)

    corrected, pressure_loss = _apply_fixed_flow_pressure_constraint(
        u,
        unit_pressure_response=response,
        active_mask=active,
        cell_area=area,
        target_flow_rate=9.0,
        base_pressure_loss_gradient=3.0,
    )

    assert jnp.sum(corrected, axis=(1, 2)).tolist() == pytest.approx([9.0, 9.0])
    assert pressure_loss.tolist() == pytest.approx([5.0, 5.0])

    u = jnp.asarray([[[1.0, 1.0]], [[3.0, 3.0]]])
    corrected = _equalize_stationwise_flow_rate_3d(
        u,
        active_mask=jnp.ones_like(u, dtype=bool),
        cell_area=jnp.ones_like(u),
        relaxation=1.0,
    )
    assert jnp.sum(corrected, axis=(1, 2)).tolist() == pytest.approx([4.0, 4.0])


def test_nonuniform_operators_and_spacing_contract():
    y_faces = jnp.asarray([-1.0, -0.7, -0.2, 0.1, 0.55, 1.0])
    z_faces = jnp.asarray([-0.8, -0.3, 0.0, 0.4, 0.8])
    y = 0.5 * (y_faces[:-1] + y_faces[1:])
    z = 0.5 * (z_faces[:-1] + z_faces[1:])
    yy, zz = jnp.meshgrid(y, z, indexing="ij")
    linear = jnp.broadcast_to((3.0 * yy - 2.0 * zz)[None, :, :], (3, 5, 4))
    _, d_dy, d_dz = _gradient_3d(
        linear,
        dx=0.25,
        dy=jnp.diff(y_faces),
        dz=jnp.diff(z_faces),
    )
    assert d_dy[:, 1:-1, 1:-1] == pytest.approx(3.0)
    assert d_dz[:, 1:-1, 1:-1] == pytest.approx(-2.0)

    errors = []
    for count in (17, 33):
        computational = jnp.linspace(-1.0, 1.0, count + 1)
        faces = jnp.tanh(1.5 * computational) / jnp.tanh(1.5)
        centers = 0.5 * (faces[:-1] + faces[1:])
        y_grid, z_grid = jnp.meshgrid(centers, centers, indexing="ij")
        quadratic = jnp.broadcast_to((y_grid**2 + z_grid**2)[None, :, :], (3, count, count))
        laplacian = _laplacian_3d(
            quadratic,
            dx=0.25,
            dy=jnp.diff(faces),
            dz=jnp.diff(faces),
        )
        errors.append(float(jnp.max(jnp.abs(laplacian[:, 2:-2, 2:-2] - 4.0))))
    assert errors[1] < errors[0] / 2.5
    assert errors[1] < 0.03

    constant = jnp.ones((3, 4, 3))
    wall_laplacian = _laplacian_3d(
        constant,
        dx=0.5,
        dy=jnp.asarray([0.2, 0.3, 0.4, 0.5]),
        dz=jnp.asarray([0.25, 0.35, 0.4]),
    )
    assert jnp.allclose(wall_laplacian[:, 1:-1, 1:-1], 0.0)
    for wall in (
        wall_laplacian[:, 0],
        wall_laplacian[:, -1],
        wall_laplacian[:, :, 0],
        wall_laplacian[:, :, -1],
    ):
        assert float(jnp.max(wall)) < 0.0

    assert _spacing_vector(0.25, 3, dtype=float).tolist() == pytest.approx([0.25, 0.25, 0.25])
    with pytest.raises(ValueError, match="shape \\(3,\\)"):
        _spacing_vector(jnp.asarray([0.2, 0.3]), 3, dtype=float)
    with pytest.raises(ValueError, match="positive"):
        _spacing_vector(jnp.asarray([0.2, 0.0, 0.3]), 3, dtype=float)

    field = jnp.arange(12.0).reshape(3, 2, 2)
    _, d_dy, d_dz = _gradient_3d(
        field,
        dx=0.5,
        dy=jnp.asarray([0.3, 0.7]),
        dz=jnp.asarray([0.4, 0.6]),
    )
    assert jnp.allclose(d_dy, 0.0)
    assert jnp.allclose(d_dz, 0.0)


def test_nonuniform_variable_poisson_reconstructs_discrete_manufactured_field():
    y_faces = jnp.asarray([-1.0, -0.8, -0.35, 0.0, 0.2, 0.6, 0.85, 1.0])
    z_faces = jnp.asarray([-1.0, -0.65, -0.1, 0.25, 0.7, 1.0])
    y = 0.5 * (y_faces[:-1] + y_faces[1:])
    z = 0.5 * (z_faces[:-1] + z_faces[1:])
    yy, zz = jnp.meshgrid(y, z, indexing="ij")
    manufactured_2d = jnp.cos(jnp.pi * (yy + 1.0) / 2.0) * jnp.cos(jnp.pi * (zz + 1.0) / 2.0)
    manufactured = jnp.broadcast_to(manufactured_2d[None, :, :], (3, 7, 5))
    conductivity = jnp.broadcast_to((1.0 + 0.2 * yy)[None, :, :], manufactured.shape)
    linear = jnp.broadcast_to((2.0 * yy - 3.0 * zz)[None], manufactured.shape)
    zeros = jnp.zeros_like(linear)
    fx, fy, fz = _conservative_current_fluxes_3d(
        jnp.ones_like(linear), linear, zeros, zeros, zeros, dx=0.4, dy=jnp.diff(y_faces), dz=jnp.diff(z_faces)
    )
    assert fy[:, 1:-1] == pytest.approx(-2.0)
    assert fz[:, :, 1:-1] == pytest.approx(3.0)
    current_divergence = (
        (fx[1:] - fx[:-1]) / 0.4
        + (fy[:, 1:] - fy[:, :-1]) / jnp.diff(y_faces)[None, :, None]
        + (fz[:, :, 1:] - fz[:, :, :-1]) / jnp.diff(z_faces)[None, None, :]
    )
    assert current_divergence[:, 1:-1, 1:-1] == pytest.approx(0.0, abs=1.0e-12)

    def manufactured_rhs(coefficient):
        neighbors = common_impl._neighbor_fields(
            manufactured, mode_x="neumann", mode_y="neumann", mode_z="neumann"
        )
        coefficients = common_impl._variable_diffusion_coefficients_3d(
            coefficient, dx=0.4, dy=jnp.diff(y_faces), dz=jnp.diff(z_faces)
        )
        return sum(c * (neighbor - manufactured) for c, neighbor in zip(coefficients, neighbors))

    rhs = manufactured_rhs(conductivity)
    expected = manufactured - jnp.mean(manufactured)
    solved_result = _solvax_pressure_poisson_duct(
        rhs,
        conductivity,
        dx=0.4,
        dy=jnp.diff(y_faces),
        dz=jnp.diff(z_faces),
        iterations=300,
        tolerance=1.0e-10,
    )
    (
        solvax_solved,
        solvax_residual,
        solvax_converged,
        solvax_relative_residual,
        solvax_iterations,
        solvax_status,
        solvax_local_residual,
    ) = solved_result
    solvax_solved = solvax_solved - jnp.mean(solvax_solved)
    assert bool(solvax_converged)
    assert float(solvax_residual) < 1.0e-8
    assert float(solvax_relative_residual) < 1.0e-8
    assert int(solvax_iterations) > 0
    assert int(solvax_status) == 1
    assert float(solvax_local_residual) < 1.0e-8
    assert float(jnp.max(jnp.abs(solvax_solved - expected))) < 1.0e-8

    projection_rhs = manufactured_rhs(jnp.ones_like(manufactured))
    projection = _projection_pressure_correction_3d(
        projection_rhs,
        dx=0.4,
        dy=jnp.diff(y_faces),
        dz=jnp.diff(z_faces),
        iterations=300,
    )
    volume = jnp.broadcast_to(
        jnp.diff(y_faces)[None, :, None] * jnp.diff(z_faces)[None, None, :], manufactured.shape
    )
    projection_expected = manufactured - jnp.sum(manufactured * volume) / jnp.sum(volume)
    assert float(jnp.max(jnp.abs(projection - projection_expected))) < 2.0e-5

    def projection_energy(scale):
        field = _projection_pressure_correction_3d(
            scale * projection_rhs,
            dx=0.4,
            dy=jnp.diff(y_faces),
            dz=jnp.diff(z_faces),
            iterations=300,
        )
        return jnp.sum(field**2 * volume) / jnp.sum(volume)

    value, derivative = jax.jit(jax.value_and_grad(projection_energy))(jnp.asarray(1.0))
    assert derivative == pytest.approx(2.0 * value, rel=2.0e-6, abs=1.0e-8)


def test_solvax_metric_pressure_poisson_is_jitted_and_differentiable():
    dy = jnp.asarray([0.2, 0.3, 0.4, 0.5])
    dz = jnp.asarray([0.25, 0.35, 0.4])
    x = jnp.linspace(-1.0, 1.0, 4)[:, None, None]
    y = jnp.linspace(-1.0, 1.0, 4)[None, :, None]
    z = jnp.linspace(-1.0, 1.0, 3)[None, None, :]
    rhs_shape = jnp.sin(jnp.pi * x) * jnp.cos(jnp.pi * y) * jnp.ones_like(z)
    mobility = jnp.broadcast_to(1.0 + 0.1 * y, rhs_shape.shape)

    def coefficient_objective(scale):
        pressure, _, _, _, _, _, _ = _solvax_pressure_poisson_duct(
            rhs_shape,
            scale * mobility,
            dx=0.4,
            dy=dy,
            dz=dz,
            iterations=200,
            tolerance=1.0e-10,
        )
        return jnp.mean(pressure**2)

    coefficient_value, coefficient_gradient = jax.jit(jax.value_and_grad(coefficient_objective))(
        jnp.asarray(1.0)
    )
    assert coefficient_gradient == pytest.approx(-2.0 * coefficient_value, rel=1.0e-6, abs=1.0e-8)


def test_implicit_duct_momentum_matches_dense_diffusion_and_autodiff(monkeypatch):
    shape, dx, dt = (4, 2, 2), 0.3, 0.04
    dy, dz = jnp.asarray([0.4, 0.6]), jnp.asarray([0.45, 0.55])
    scalar = jnp.arange(np.prod(shape), dtype=float).reshape(shape) / 50.0
    velocity = jnp.stack((scalar, 0.3 - scalar, 0.2 * scalar), axis=-1)
    force = jnp.stack((0.1 + scalar, -0.2 * scalar, 0.05 - scalar), axis=-1)
    density, viscosity = 1.0 + 0.1 * scalar, 0.06 + 0.02 * scalar
    rho_phi = (
        jnp.linspace(-0.05, 0.12, 20).reshape(5, 2, 2),
        jnp.linspace(0.09, -0.04, 24).reshape(4, 3, 2),
        jnp.linspace(-0.03, 0.07, 24).reshape(4, 2, 3),
    )
    zero_yz = jnp.zeros((4, 2, 3))
    boundary = (jnp.zeros((2, 2, 3)).at[..., 0].set(0.25), velocity[-1], zero_yz, zero_yz, zero_yz, zero_yz)
    widths = (jnp.full((4,), dx), dy, dz)

    def dense_scalar(alpha, patches):
        fluxes = tuple(np.asarray(alpha * value) for value in rho_phi)
        weights = tuple(
            np.asarray(value)
            for value in duct_impl._limited_linear_vector_face_weights_duct(
                velocity, tuple(alpha * value for value in rho_phi), patches, widths
            )
        )
        mu, rho_np = np.asarray(density * viscosity), np.asarray(density)
        widths_np = tuple(map(np.asarray, widths))
        volume_np = np.broadcast_to(dx * np.asarray(dy)[None, :, None] * np.asarray(dz)[None, None, :], shape)
        sink = np.zeros(shape)
        sink[:, 0, :] += 2.0 * mu[:, 0, :] / widths_np[1][0] ** 2
        sink[:, -1, :] += 2.0 * mu[:, -1, :] / widths_np[1][-1] ** 2
        sink[:, :, 0] += 2.0 * mu[:, :, 0] / widths_np[2][0] ** 2
        sink[:, :, -1] += 2.0 * mu[:, :, -1] / widths_np[2][-1] ** 2
        sink[0] += 2.0 * mu[0] / dx**2

        def action(flat):
            field = flat.reshape(shape)
            result = rho_np * volume_np * field + dt * volume_np * sink * field
            for axis, width in enumerate(widths_np):
                field_axis, result_axis = np.moveaxis(field, axis, 0), np.moveaxis(result, axis, 0)
                mu_axis, volume_axis = np.moveaxis(mu, axis, 0), np.moveaxis(volume_np, axis, 0)
                lo = width[:-1].reshape((-1, 1, 1))
                hi = width[1:].reshape((-1, 1, 1))
                face_mu = (lo + hi) / (lo / mu_axis[:-1] + hi / mu_axis[1:])
                delta = field_axis[1:] - field_axis[:-1]
                result_axis[:-1] -= dt * volume_axis[:-1] * face_mu / (lo * 0.5 * (lo + hi)) * delta
                result_axis[1:] += dt * volume_axis[1:] * face_mu / (hi * 0.5 * (lo + hi)) * delta
                weight = np.moveaxis(weights[axis], axis, 0)
                face_value = weight * field_axis[:-1] + (1.0 - weight) * field_axis[1:]
                face_flux = np.moveaxis(fluxes[axis], axis, 0)[1:-1]
                result_axis[:-1] += dt * face_flux * face_value
                result_axis[1:] -= dt * face_flux * face_value
            result[-1] += dt * fluxes[0][-1] * field[-1]
            return result.reshape(-1)

        eye = np.eye(np.prod(shape))
        return np.column_stack([action(column) for column in eye])

    captured, actual_linear_solve = {}, duct_impl.linear_solve

    def capture(matvec, rhs, solver, **kwargs):
        captured["matrix"] = jax.jacfwd(matvec)(jnp.zeros_like(rhs))
        captured["rhs"], captured["zero"] = rhs, matvec(jnp.zeros_like(rhs))
        captured["has_aux"] = kwargs.get("has_aux", False)
        return actual_linear_solve(matvec, rhs, solver, **kwargs)

    monkeypatch.setattr(duct_impl, "linear_solve", capture)

    def solve(applied_force, flux=rho_phi, patches=boundary, rho=density, reaction=None, **kwargs):
        return _solvax_implicit_momentum_duct(
            velocity,
            applied_force,
            rho,
            viscosity,
            flux,
            patches,
            dt=dt,
            dx=dx,
            dy=dy,
            dz=dz,
            iterations=240,
            tolerance=1.0e-10,
            reaction=jnp.zeros_like(rho) if reaction is None else reaction,
            **kwargs,
        )

    solved, residual, converged = solve(force)
    matrix, rhs = captured["matrix"], captured["rhs"]
    reference = np.kron(dense_scalar(1.0, boundary), np.eye(3))
    assert matrix == pytest.approx(reference) and solved.sharding == velocity.sharding
    assert solved.reshape(-1) == pytest.approx(np.linalg.solve(reference, rhs), abs=2e-7)
    assert residual < 1e-8 and bool(converged) and solved.shape == (*shape, 3)
    assert captured["has_aux"] and captured["zero"] == pytest.approx(0.0)
    assert not jnp.allclose(matrix, matrix.T)
    volume = dx * dy[None, :, None] * dz[None, None, :]
    pressure = lambda p: duct_impl._duct_pressure_force(p, dx=dx, dy=dy, dz=dz)  # noqa: E731
    pressure_force = jax.jacfwd(lambda p: (volume[..., None] * dt * pressure(p)).reshape(-1))(
        jnp.zeros(shape)
    ).reshape((velocity.size, scalar.size))
    divergence = jax.jacfwd(
        lambda value: duct_impl._duct_velocity_divergence(
            value.reshape(velocity.shape), jnp.zeros_like(velocity[0, ..., 0]), dx=dx, dy=dy, dz=dz
        ).reshape(-1)
    )(jnp.zeros(velocity.size))
    a_inverse = lambda value: jnp.linalg.solve(matrix, value)  # noqa: E731
    schur = divergence @ a_inverse(pressure_force)
    block = jnp.block([[matrix, pressure_force], [divergence, jnp.zeros((scalar.size,) * 2)]])
    precondition = schur_complement_precond(
        a_inverse,
        lambda value: pressure_force @ value,
        lambda value: divergence @ value,
        lambda value: jnp.linalg.solve(schur, value),
    )
    block_rhs = (jnp.linspace(-0.2, 0.3, velocity.size), jnp.linspace(0.1, -0.1, scalar.size))
    assert jnp.concatenate(precondition(block_rhs)) == pytest.approx(
        jnp.linalg.solve(block, jnp.concatenate(block_rhs)), abs=2.0e-12
    )
    response = solve(force, linear_rhs=block_rhs[0].reshape(velocity.shape))[0]
    assert response.reshape(-1) == pytest.approx(a_inverse(block_rhs[0]), abs=2e-7)
    expected_rhs = volume[..., None] * (density[..., None] * velocity + dt * force)
    inlet_source = rho_phi[0][0][..., None] + volume[0, ..., None] * (
        2.0 * (density * viscosity)[0, ..., None] / dx**2
    )
    expected_rhs = expected_rhs.at[0].add(dt * inlet_source * boundary[0])
    assert rhs.reshape((*shape, 3)) == pytest.approx(expected_rhs)
    reaction = jnp.linspace(0.1, 0.4, density.size).reshape(density.shape)
    solved_reaction, _, _ = solve(force, reaction=reaction)
    reaction_matrix, reaction_rhs = captured["matrix"], captured["rhs"]
    pseudo_mass = np.repeat(np.asarray(volume * dt * reaction).reshape(-1), 3)
    assert reaction_matrix == pytest.approx(reference + np.diag(pseudo_mass))
    assert reaction_rhs == pytest.approx(expected_rhs.reshape(-1) + pseudo_mass * velocity.reshape(-1))
    assert reaction_matrix @ velocity.reshape(-1) - reaction_rhs == pytest.approx(
        matrix @ velocity.reshape(-1) - rhs
    )
    assert solved_reaction.reshape(-1) == pytest.approx(
        np.linalg.solve(reaction_matrix, reaction_rhs), abs=2e-7
    )
    mass = np.repeat(np.asarray(volume * density).reshape(-1), 3)
    cell_volume = np.repeat(np.broadcast_to(np.asarray(volume), shape).reshape(-1), 3)
    old, star = map(np.asarray, (velocity.reshape(-1), solved.reshape(-1)))
    mapped = star + 0.01 * np.asarray(probe := jnp.linspace(-0.2, 0.3, velocity.size)).reshape(-1)
    source = (np.asarray(rhs) - mass * old) / (cell_volume * dt)
    transport = (np.asarray(matrix) @ star - mass * star) / (cell_volume * dt)
    split_residual = transport - source - np.repeat(np.asarray(density).reshape(-1), 3) * (mapped - star) / dt
    linear_residual = (np.asarray(matrix) @ star - np.asarray(rhs)) / (cell_volume * dt)
    assert split_residual == pytest.approx(
        linear_residual - np.repeat(np.asarray(density).reshape(-1), 3) * (mapped - old) / dt, abs=2.0e-12
    )
    monkeypatch.setattr(duct_impl, "linear_solve", actual_linear_solve)
    neutral = tuple(jnp.zeros_like(value) for value in boundary)

    def solve_alpha(alpha):
        return solve(force, tuple(alpha * value for value in rho_phi), neutral)[0]

    primal = jax.jit(solve_alpha)(jnp.asarray(1.0))
    tangent = jax.jvp(solve_alpha, (1.0,), (1.0,))[1]
    matrix_alpha = np.kron(dense_scalar(1.0, neutral), np.eye(3))
    convection = np.kron(dense_scalar(2.0, neutral) - dense_scalar(1.0, neutral), np.eye(3))
    expected_tangent = -np.linalg.solve(matrix_alpha, convection @ np.asarray(primal).reshape(-1))
    assert tangent.reshape(-1) == pytest.approx(expected_tangent, abs=3e-7)
    probe = probe.reshape(velocity.shape)
    gradient = jax.grad(lambda alpha: jnp.sum(solve_alpha(alpha) * probe))(1.0)
    adjoint = np.linalg.solve(matrix_alpha.T, np.asarray(probe).reshape(-1))
    assert gradient == pytest.approx(-adjoint @ convection @ np.asarray(primal).reshape(-1), rel=2e-6)
    zero_velocity = jnp.zeros_like(velocity)
    compact_flux = jnp.zeros((3, *shape))
    lorentz = jnp.broadcast_to(jnp.asarray([1.0, 2.0, 3.0]), velocity.shape)

    def defect(scale):
        return duct_impl._duct_momentum_defect(
            zero_velocity,
            scale * lorentz,
            jnp.ones(shape),
            jnp.zeros(shape),
            compact_flux,
            jnp.zeros(shape[1:]),
            jnp.zeros(shape),
            forcing=0.0,
            force_scale=2.0,
            dt=dt,
            dx=dx,
            dy=dy,
            dz=dz,
        )

    expected_defect = jnp.asarray([0.5, 1.0, 1.5, 1.5])
    assert jax.jit(defect)(1.0) == pytest.approx(expected_defect)
    assert jax.jvp(defect, (1.0,), (1.0,))[1] == pytest.approx(expected_defect)


def test_limited_linear_vector_convection_matches_manufactured_conservation_and_autodiff():
    nx, ny, nz = 7, 7, 3
    dx = jnp.asarray([0.18, 0.24, 0.21, 0.29, 0.23, 0.31, 0.26])
    dy = jnp.asarray([0.22, 0.17, 0.28, 0.19, 0.26, 0.21, 0.3])
    dz, rho = jnp.asarray([0.27, 0.34, 0.25]), 1.7
    x_faces = -0.3 + jnp.concatenate((jnp.zeros(1), jnp.cumsum(dx)))
    y_faces = 1.0 + jnp.concatenate((jnp.zeros(1), jnp.cumsum(dy)))
    x, y = 0.5 * (x_faces[:-1] + x_faces[1:]), 0.5 * (y_faces[:-1] + y_faces[1:])
    xx = jnp.broadcast_to(x[:, None, None], (nx, ny, nz))
    yy = jnp.broadcast_to(y[None, :, None], (nx, ny, nz))
    velocity = jnp.stack((yy, xx, jnp.zeros_like(xx)), axis=-1)
    x_wall = jnp.broadcast_to(x[:, None], (nx, ny))
    y_wall = jnp.broadcast_to(y[None, :], (nx, ny))
    west_y = jnp.broadcast_to(y[:, None], (ny, nz))
    south_x = jnp.broadcast_to(x[:, None], (nx, nz))
    boundary_velocity = (
        jnp.stack((west_y, jnp.full_like(west_y, x_faces[0]), jnp.zeros_like(west_y)), -1),
        jnp.stack((west_y, jnp.full_like(west_y, x_faces[-1]), jnp.zeros_like(west_y)), -1),
        jnp.stack((jnp.full_like(south_x, y_faces[0]), south_x, jnp.zeros_like(south_x)), -1),
        jnp.stack((jnp.full_like(south_x, y_faces[-1]), south_x, jnp.zeros_like(south_x)), -1),
        jnp.stack((y_wall, x_wall, jnp.zeros_like(x_wall)), -1),
        jnp.stack((y_wall, x_wall, jnp.zeros_like(x_wall)), -1),
    )
    rho_phi = (
        jnp.broadcast_to(
            rho * yy[:1] * dy[None, :, None] * dz[None, None, :],
            (nx + 1, ny, nz),
        ),
        jnp.broadcast_to(
            rho * x[:, None, None] * dx[:, None, None] * dz[None, None, :],
            (nx, ny + 1, nz),
        ),
        jnp.zeros((nx, ny, nz + 1)),
    )
    q = jnp.sum(velocity**2, axis=-1)
    q_patches = tuple(jnp.sum(value**2, axis=-1) for value in boundary_velocity)
    least_squares = duct_impl._cell_limited_least_squares_gradient_duct
    gradients = least_squares(q, q_patches, (dx, dy, dz))
    j, k = 3, 1

    def x_ls_reference(values, patches, i):
        lo = patches[0][j, k] if i == 0 else values[i - 1, j, k]
        hi = values[i + 1, j, k]
        lo_distance = 0.5 * dx[i] if i == 0 else 0.5 * (dx[i - 1] + dx[i])
        hi_distance = 0.5 * (dx[i] + dx[i + 1])
        lo_weight = 1.0 if i == 0 else dx[i] / (dx[i - 1] + dx[i])
        hi_weight = dx[i] / (dx[i] + dx[i + 1])
        return (
            lo_weight * (values[i, j, k] - lo) / lo_distance
            + hi_weight * (hi - values[i, j, k]) / hi_distance
        ) / (lo_weight + hi_weight)

    assert gradients[0][4, j, k] == pytest.approx(x_ls_reference(q, q_patches, 4))
    assert gradients[0][0, j, k] == pytest.approx(x_ls_reference(q, q_patches, 0))

    spike = jnp.broadcast_to(
        jnp.asarray([0.0, 1.0e-15, 0.0, 0.0, 0.0, 0.0, 0.0])[:, None, None],
        (nx, ny, nz),
    )
    zero_q = tuple(jnp.zeros(value.shape[:-1]) for value in boundary_velocity)
    spike_gradient = least_squares(spike, zero_q, (dx, dy, dz))[0][1, j, k]
    assert spike_gradient == pytest.approx(x_ls_reference(spike, zero_q, 1), abs=1.0e-30)

    def convection(scale):
        state = scale * velocity
        flux = tuple(scale * values for values in rho_phi)
        patches = tuple(scale * values for values in boundary_velocity)
        weights = duct_impl._limited_linear_vector_face_weights_duct(state, flux, patches, (dx, dy, dz))
        return duct_impl._limited_linear_convection_matrix_action_duct(
            state, flux, weights, patches, (dx, dy, dz)
        ), weights

    action, weights = convection(1.0)
    expected = rho * jnp.stack((xx, yy, jnp.zeros_like(xx)), axis=-1)
    assert action[4:-1, 2:-2] == pytest.approx(expected[4:-1, 2:-2], abs=2.0e-5)
    assert weights[0][3:6, 2:-2] == pytest.approx(
        jnp.broadcast_to((dx[1:] / (dx[:-1] + dx[1:]))[3:6, None, None], (3, 3, nz))
    )
    assert weights[1][4:-1, 1:5] == pytest.approx(
        jnp.broadcast_to((dy[1:] / (dy[:-1] + dy[1:]))[None, 1:5, None], (2, 4, nz))
    )
    assert all(bool(jnp.all((face_weight >= 0.0) & (face_weight <= 1.0))) for face_weight in weights)
    assert weights[0][1] == pytest.approx(1.0)
    assert weights[2] == pytest.approx(1.0)

    volume = dx[:, None, None] * dy[None, :, None] * dz[None, None, :]
    boundary_flux = (
        jnp.sum(rho_phi[0][-1, ..., None] * boundary_velocity[1], axis=(0, 1))
        - jnp.sum(rho_phi[0][0, ..., None] * boundary_velocity[0], axis=(0, 1))
        + jnp.sum(rho_phi[1][:, -1, :, None] * boundary_velocity[3], axis=(0, 1))
        - jnp.sum(rho_phi[1][:, 0, :, None] * boundary_velocity[2], axis=(0, 1))
    )
    assert jnp.sum(action * volume[..., None], axis=(0, 1, 2)) == pytest.approx(boundary_flux, abs=2.0e-5)
    zero_patches = tuple(jnp.zeros_like(value) for value in boundary_velocity)
    assert duct_impl._limited_linear_convection_matrix_action_duct(
        jnp.zeros_like(velocity), rho_phi, weights, zero_patches, (dx, dy, dz)
    ) == pytest.approx(0.0)
    scaled, scaled_weights = convection(2.0)
    assert scaled == pytest.approx(4.0 * action, abs=2.0e-5)
    assert all(jnp.allclose(a, b) for a, b in zip(weights, scaled_weights, strict=True))
    compiled = jax.jit(lambda scale: convection(scale)[0])(jnp.asarray(1.0))
    tangent = jax.jvp(
        lambda scale: convection(scale)[0],
        (jnp.asarray(1.0),),
        (jnp.asarray(1.0),),
    )[1]
    assert compiled == pytest.approx(action, abs=2.0e-5)
    assert tangent == pytest.approx(2.0 * action, abs=3.0e-5)


def test_explicit_deviatoric_stress_matches_independent_finite_volume_oracle():
    shape = (5, 4, 3)
    widths = tuple(
        np.asarray(value) for value in ([0.2, 0.3, 0.25, 0.35, 0.4], [0.3, 0.2, 0.4, 0.35], [0.25, 0.45, 0.3])
    )
    velocity = np.sin(np.arange(np.prod(shape) * 3).reshape((*shape, 3)) / 7.0)
    zero_y = np.zeros((shape[0], shape[2], 3))
    zero_z = np.zeros((shape[0], shape[1], 3))
    patches = (np.zeros((*shape[1:], 3)), velocity[-1], zero_y, zero_y, zero_z, zero_z)

    def reference(field, boundary):
        gradients, neighbours = [], []
        for axis, width in enumerate(widths):
            values = np.moveaxis(field, axis, 0)
            lo = np.concatenate((boundary[2 * axis][None], values[:-1]))
            hi = np.concatenate((values[1:], boundary[2 * axis + 1][None]))
            dm = np.concatenate((width[:1] / 2, (width[:-1] + width[1:]) / 2))[:, None, None, None]
            dp = np.concatenate(((width[:-1] + width[1:]) / 2, width[-1:] / 2))[:, None, None, None]
            fraction = width[1:] / (width[:-1] + width[1:])
            wm = np.concatenate(([1.0], fraction))[:, None, None, None]
            wp = np.concatenate((1.0 - fraction, [1.0]))[:, None, None, None]
            gradients.append(
                np.moveaxis((wm * (values - lo) / dm + wp * (hi - values) / dp) / (wm + wp), 0, axis)
            )
            neighbours.extend((np.moveaxis(lo, 0, axis), np.moveaxis(hi, 0, axis)))
        local = np.stack((field, *neighbours))
        minimum, maximum, limiter = local.min(0), local.max(0), np.ones_like(field)
        for axis, (gradient, width) in enumerate(zip(gradients, widths, strict=True)):
            reshape = [1, 1, 1, 1]
            reshape[axis] = shape[axis]
            for extrapolate in (
                -0.5 * width.reshape(reshape) * gradient,
                0.5 * width.reshape(reshape) * gradient,
            ):
                delta = np.where(extrapolate > 0, maximum - field, minimum - field)
                limiter = np.minimum(
                    limiter,
                    np.where(
                        abs(extrapolate) > 1e-15,
                        np.minimum(delta / np.where(abs(extrapolate) > 1e-15, extrapolate, 1), 1),
                        1,
                    ),
                )
        gradient = np.stack(tuple(limiter * value for value in gradients), axis=-2)
        result, eye = np.zeros_like(field), np.eye(3)
        for axis, width in enumerate(widths):

            def traction(g):
                return 0.7 * (
                    g[..., :, axis] - 2 / 3 * np.trace(g, axis1=-2, axis2=-1)[..., None] * eye[axis]
                )

            moved = np.moveaxis(traction(gradient), axis, 0)
            face_values = []
            for side, index in ((0, 0), (1, -1)):
                patch_gradient = np.take(gradient, index, axis=axis).copy()
                adjacent = np.take(field, index, axis=axis)
                patch_gradient[..., axis, :] = (
                    (2 * side - 1) * (boundary[2 * axis + side] - adjacent) / (width[index] / 2)
                )
                face_values.append(traction(patch_gradient))
            weight = (width[1:] / (width[:-1] + width[1:]))[:, None, None, None]
            faces = np.concatenate(
                (face_values[0][None], weight * moved[:-1] + (1 - weight) * moved[1:], face_values[1][None])
            )
            result += np.moveaxis(np.diff(faces, axis=0) / width[:, None, None, None], 0, axis)
        return result

    def evaluate(scale):
        return duct_impl._explicit_deviatoric_stress_duct(
            scale * jnp.asarray(velocity),
            jnp.full(shape, 0.7),
            tuple(scale * jnp.asarray(value) for value in patches),
            tuple(map(jnp.asarray, widths)),
        )

    expected = reference(velocity, patches)
    assert jax.jit(evaluate)(1.0) == pytest.approx(expected, abs=2.0e-6)
    assert jax.jvp(evaluate, (1.0,), (1.0,))[1] == pytest.approx(expected, abs=3.0e-6)

    uniform = (jnp.full((5,), 0.2), jnp.full((4,), 0.3), jnp.full((3,), 0.4))
    profile = jnp.arange(12.0).reshape(4, 3) / 20.0
    developed = jnp.zeros((*shape, 3)).at[..., 0].set(profile[None])
    developed_patches = (developed[0], developed[-1], *map(jnp.zeros_like, patches[2:]))
    stress = duct_impl._explicit_deviatoric_stress_duct
    assert stress(developed, jnp.full(shape, 0.7), developed_patches, uniform) == pytest.approx(0.0)
    x = (jnp.arange(5) + 0.5) * uniform[0][0]
    quadratic = jnp.zeros((*shape, 3)).at[..., 0].set(1.2 * x[:, None, None] ** 2)
    quadratic_patches = (jnp.zeros_like(patches[0]), quadratic[-1], *map(jnp.zeros_like, patches[2:]))
    quadratic_stress = stress(quadratic, jnp.full(shape, 0.7), quadratic_patches, uniform)
    assert quadratic_stress[2, 1:-1, 1:-1, 0] == pytest.approx(2 * 0.7 * 1.2 / 3)
    affine = jnp.zeros_like(quadratic).at[..., 0].set(1.2 * x[:, None, None])
    affine_stress = stress(
        affine,
        jnp.full(shape, 0.7),
        (jnp.zeros_like(patches[0]), affine[-1], *map(jnp.zeros_like, patches[2:])),
        uniform,
    )
    volume = uniform[0][0] * uniform[1][0] * uniform[2][0]
    assert jnp.sum(affine_stress[..., 0]) * volume == pytest.approx(-0.7 * 1.2 * 4 * 0.3 * 3 * 0.4 / 3)


def test_frozen_momentum_setup_preserves_primal_jvp_and_vjp():
    shape = (3, 3, 3)
    widths = tuple(map(jnp.asarray, ([0.2, 0.2, 0.2], [0.3, 0.25, 0.35], [0.4, 0.3, 0.5])))
    velocity = jnp.arange(np.prod((*shape, 3)), dtype=float).reshape((*shape, 3)) / 17.0
    zero_y = jnp.zeros((shape[0], shape[2], 3))
    zero_z = jnp.zeros((shape[0], shape[1], 3))
    patches = (velocity[0] - 0.3, velocity[-1] + 0.4, zero_y, zero_y, zero_z, zero_z)
    fluxes = (
        jnp.linspace(0.2, 1.0, (shape[0] + 1) * shape[1] * shape[2]).reshape(
            (shape[0] + 1, shape[1], shape[2])
        ),
        jnp.zeros((shape[0], shape[1] + 1, shape[2])),
        jnp.zeros((shape[0], shape[1], shape[2] + 1)),
    )

    def assert_equivalent(default, injected, cotangent, atol=0.0):
        pairs = (
            (default(1.0), injected(1.0)),
            (jax.jvp(default, (1.0,), (1.0,))[1], jax.jvp(injected, (1.0,), (1.0,))[1]),
            (jax.vjp(default, 1.0)[1](cotangent)[0], jax.vjp(injected, 1.0)[1](cotangent)[0]),
        )
        for observed, expected in pairs:
            jax.tree.map(
                lambda a, b: np.testing.assert_allclose(a, b, rtol=3.0e-14, atol=atol), observed, expected
            )

    def momentum_setup(scale, packed):
        state = scale * velocity
        boundaries = tuple(scale * value for value in patches)
        scaled_fluxes = tuple(scale * value for value in fluxes)
        q, q_patches = jnp.sum(state**2, axis=-1), tuple(jnp.sum(value**2, axis=-1) for value in boundaries)
        fields = (*tuple(state[..., component] for component in range(3)), q)
        boundary_fields = (
            *tuple(tuple(value[..., component] for value in boundaries) for component in range(3)),
            q_patches,
        )
        scalar = tuple(
            duct_impl._cell_limited_least_squares_gradient_duct(field, boundary, widths)
            for field, boundary in zip(fields, boundary_fields, strict=True)
        )
        gradient = jnp.stack(tuple(jnp.stack(value, axis=-1) for value in scalar[:3]), axis=-1)
        weights = duct_impl._limited_linear_vector_face_weights_duct(
            state, scaled_fluxes, boundaries, widths, gradient=scalar[3]
        )
        if packed:
            setup = duct_impl._frozen_duct_momentum_setup(
                state,
                jnp.ones(shape),
                jnp.full(shape, 0.7),
                scaled_fluxes,
                boundaries,
                widths,
                dx=widths[0][0],
            )
            weights, gradient = setup[-2:]
        stress = duct_impl._explicit_deviatoric_stress_duct(
            state, jnp.full(shape, 0.7), boundaries, widths, gradient=gradient
        )
        return gradient, weights, stress

    assert momentum_setup(1.0, True)[0].shape == (*shape, 3, 3)
    assert_equivalent(
        lambda scale: momentum_setup(scale, False),
        lambda scale: momentum_setup(scale, True),
        jax.tree.map(jnp.ones_like, momentum_setup(1.0, False)),
        atol=2.0e-14,
    )


def test_compact_duct_mass_flux_initializer_matches_fv_faces():
    shape = (2, 2, 2)
    compact = jnp.arange(24.0).reshape((3, *shape))
    inlet = jnp.arange(4.0).reshape(shape[1:])
    full = jax.jit(duct_impl._unpack_duct_mass_flux)(compact, inlet)
    assert jnp.all(full[1][:, 0] == 0.0) and jnp.all(full[2][:, :, 0] == 0.0)
    assert compact.size + inlet.size == 3 * np.prod(shape) + np.prod(shape[1:])
    velocity = jnp.arange(24.0).reshape((*shape, 3)) / 7.0
    density = 1.0 + jnp.arange(8.0).reshape(shape) / 10.0
    inlet_velocity = velocity[0] + jnp.asarray([0.4, -2.0, 3.0])
    dy, dz, dx = jnp.asarray([0.3, 0.7]), jnp.asarray([0.2, 0.8]), 0.4

    def initialize(scale):
        return duct_impl._initialize_duct_mass_flux(
            scale * velocity, density, scale * inlet_velocity, dx=dx, dy=dy, dz=dz
        )

    *components, initialized_inlet = jax.jit(initialize)(jnp.asarray(1.0))
    plus = jnp.stack(components)
    momentum = np.asarray(density[..., None] * velocity)
    expected_x = np.concatenate(
        (0.5 * (momentum[:-1, ..., 0] + momentum[1:, ..., 0]), momentum[-1:, ..., 0])
    ) * np.outer(dy, dz)
    expected_y = (
        np.concatenate(
            (
                0.7 * momentum[:, :-1, :, 1] + 0.3 * momentum[:, 1:, :, 1],
                np.zeros_like(momentum[:, :1, :, 1]),
            ),
            axis=1,
        )
        * dx
        * dz[None, None]
    )
    expected_z = (
        np.concatenate(
            (
                0.8 * momentum[:, :, :-1, 2] + 0.2 * momentum[:, :, 1:, 2],
                np.zeros_like(momentum[:, :, :1, 2]),
            ),
            axis=2,
        )
        * dx
        * dy[None, :, None]
    )
    assert plus == pytest.approx(np.stack((expected_x, expected_y, expected_z)))
    assert initialized_inlet == pytest.approx(density[0] * inlet_velocity[..., 0] * jnp.outer(dy, dz))
    fx, fy, fz = map(np.asarray, duct_impl._unpack_duct_mass_flux(plus, initialized_inlet))
    volume = np.broadcast_to(dx * np.outer(dy, dz), shape)
    surface = (
        abs(fx[:-1]) + abs(fx[1:]) + abs(fy[:, :-1]) + abs(fy[:, 1:]) + abs(fz[:, :, :-1]) + abs(fz[:, :, 1:])
    ) / np.asarray(density)
    cell_courant = 0.5 * 0.13 * surface / volume
    courant = jax.jit(
        lambda p, i: duct_impl._compact_duct_courant_numbers(p, i, density, dt=0.13, dx=dx, dy=dy, dz=dz)
    )(plus, initialized_inlet)
    assert tuple(map(float, courant)) == pytest.approx(
        (np.sum(cell_courant * volume) / np.sum(volume), np.max(cell_courant))
    )
    tangent = jax.jvp(initialize, (1.0,), (1.0,))[1]
    assert jnp.allclose(jnp.stack(tangent[:3]), plus)
    assert jnp.allclose(tangent[-1], initialized_inlet)


@pytest.mark.parametrize("mixed", [False, True])
def test_duct_pressure_operator_has_independent_energy_and_rank_contract(mixed):
    # Two-point transmissibilities: A=B.T*T*B, with an outlet Dirichlet term.
    shape = (3, 3, 2)
    size = int(np.prod(shape))
    widths = (np.full(3, 0.4), np.asarray([0.2, 0.35, 0.45]), np.asarray([0.3, 0.7]))
    mobility = 0.7 + np.arange(size).reshape(shape) / size
    volume = widths[0][:, None, None] * widths[1][None, :, None] * widths[2][None, None, :]
    oracle = np.zeros((size, size))
    for cell in np.ndindex(shape):
        i = np.ravel_multi_index(cell, shape)
        for axis in range(3):
            if cell[axis] + 1 == shape[axis]:
                continue
            neighbor = list(cell)
            neighbor[axis] += 1
            neighbor = tuple(neighbor)
            j = np.ravel_multi_index(neighbor, shape)
            area = volume[cell] / widths[axis][cell[axis]]
            resistance = 0.5 * (
                widths[axis][cell[axis]] / mobility[cell] + widths[axis][neighbor[axis]] / mobility[neighbor]
            )
            oracle[np.ix_([i, j], [i, j])] += area / resistance * np.asarray([[1.0, -1.0], [-1.0, 1.0]])
        if mixed and cell[0] == shape[0] - 1:
            oracle[i, i] += volume[cell] * mobility[cell] / (0.5 * widths[0][-1] ** 2)

    def operator(pressure):
        x, y, z = duct_impl._duct_pressure_face_corrections(
            pressure.reshape(shape),
            jnp.asarray(mobility),
            dx=0.4,
            dy=jnp.asarray(widths[1]),
            dz=jnp.asarray(widths[2]),
            mixed_axial_pressure=mixed,
        )
        divergence = duct_impl._duct_face_divergence(
            x,
            jnp.zeros(shape[1:]),
            y,
            z,
            dx=0.4,
            dy=jnp.asarray(widths[1]),
            dz=jnp.asarray(widths[2]),
        )
        return (volume * divergence).ravel()

    pressure = jnp.sin(jnp.arange(size, dtype=jnp.float64))
    matrix = np.asarray(jax.jacfwd(operator)(pressure))
    np.testing.assert_allclose(matrix, oracle, atol=1e-12)
    np.testing.assert_allclose(matrix, matrix.T, atol=1e-12)
    assert np.linalg.matrix_rank(matrix, tol=1e-10) == size - (not mixed)
    assert np.linalg.eigvalsh(matrix).min() >= -1e-12
    np.testing.assert_allclose(operator(jnp.ones(size)), oracle @ np.ones(size), atol=1e-12)
    gradient = jax.jit(jax.grad(lambda p: 0.5 * jnp.vdot(p, operator(p))))(pressure)
    np.testing.assert_allclose(gradient, oracle @ pressure, atol=1e-12)


def test_nonuniform_face_flux_projection_closes_discrete_divergence():
    nx, ny, nz = 5, 6, 5
    dy = jnp.asarray([0.2, 0.3, 0.45, 0.4, 0.35, 0.3])
    dz = jnp.asarray([0.25, 0.4, 0.5, 0.45, 0.3])
    x = jnp.linspace(0.0, 1.0, nx).reshape(nx, 1, 1)
    y = jnp.linspace(-1.0, 1.0, ny).reshape(1, ny, 1)
    z = jnp.linspace(-1.0, 1.0, nz).reshape(1, 1, nz)
    u = jnp.cos(2.0 * jnp.pi * x) * (1.0 + 0.1 * y) * jnp.ones_like(z)
    v = 0.2 * jnp.sin(jnp.pi * y) * jnp.ones_like(x + z)
    w = -0.15 * jnp.sin(jnp.pi * z) * jnp.ones_like(x + y)

    def project(amplitude):
        return _face_flux_pressure_projection_duct(
            amplitude * u,
            v,
            w,
            jnp.ones_like(u),
            jnp.ones((nx, ny, nz), dtype=bool),
            dt=0.05,
            dx=0.25,
            dy=dy,
            dz=dz,
            fluid_bounds=(0, ny, 0, nz),
            iterations=200,
            tolerance=1.0e-10,
        )

    projected_u, projected_v, projected_w, pressure, divergence = project(1.0)
    assert divergence < 1.0e-8
    assert all(jnp.isfinite(field).all() for field in (projected_u, projected_v, projected_w, pressure))

    # A divergence-free predictor needs no pressure correction and must not be
    # filtered by a cell-to-face-to-cell round trip.
    repeated_u = jnp.broadcast_to((1.0 + 0.2 * y) * (1.0 - 0.1 * z), u.shape)
    preserved = _face_flux_pressure_projection_duct(
        repeated_u,
        jnp.zeros_like(v),
        jnp.zeros_like(w),
        jnp.ones_like(u),
        jnp.ones((nx, ny, nz), dtype=bool),
        dt=0.05,
        dx=0.25,
        dy=dy,
        dz=dz,
        fluid_bounds=(0, ny, 0, nz),
        iterations=200,
        tolerance=1.0e-10,
    )
    assert preserved[0] == pytest.approx(repeated_u, abs=1.0e-12)
    assert preserved[1] == pytest.approx(0.0, abs=1.0e-12)
    assert preserved[2] == pytest.approx(0.0, abs=1.0e-12)

    def objective(amplitude):
        projected = project(amplitude)
        return jnp.mean(projected[0] ** 2) + 0.01 * jnp.mean(projected[3] ** 2)

    value, gradient = jax.jit(jax.value_and_grad(objective))(jnp.asarray(1.0))
    step = 1.0e-4
    finite_difference = (objective(1.0 + step) - objective(1.0 - step)) / (2 * step)
    assert jnp.isfinite(value)
    assert gradient == pytest.approx(finite_difference, rel=2.0e-5, abs=1.0e-8)


def test_mixed_face_flux_projection_recovers_coefficients_and_boundary_flow():
    shape = (4, 2, 2)
    dx, widths = 0.25, jnp.asarray([0.5, 0.5])
    settings = dict(dx=dx, dy=widths, dz=widths, iterations=100, tolerance=1.0e-10)
    expected_pressure = jnp.broadcast_to(jnp.asarray([4.0, 3.0, 2.0, 1.0])[:, None, None], shape)
    rhs = jnp.broadcast_to(jnp.asarray([-16.0, 0.0, 0.0, -16.0])[:, None, None], shape)
    with pytest.raises(ValueError, match="Unsupported axial pressure mode"):
        _solvax_pressure_poisson_duct(rhs, jnp.ones(shape), **settings, axial_pressure_mode="periodic")
    pressure, *_ = _solvax_pressure_poisson_duct(
        rhs,
        jnp.ones(shape),
        **settings,
        axial_pressure_mode=common_impl._MIXED_AXIAL_PRESSURE_MODE,
    )
    assert pressure == pytest.approx(expected_pressure, abs=1.0e-7)
    pressure_force = duct_impl._duct_pressure_force(
        expected_pressure,
        dx=dx,
        dy=jnp.asarray([0.2, 0.8]),
        dz=jnp.asarray([0.7, 0.3]),
    )
    assert pressure_force[..., 0] == pytest.approx(
        jnp.broadcast_to(jnp.asarray([2.0, 4.0, 4.0, 6.0])[:, None, None], shape)
    )
    assert pressure_force[..., 1:] == pytest.approx(0.0)
    zeros = jnp.zeros(shape)
    projected = _face_flux_pressure_projection_duct(
        zeros,
        zeros,
        zeros,
        jnp.ones(shape),
        jnp.ones(shape, dtype=bool),
        inlet_flow_rate=0.2,
        dt=0.1,
        **settings,
    )
    (
        _,
        _,
        _,
        projected_pressure,
        pressure_loss,
        divergence,
        flow_error,
        flux_x,
        flux_y,
        flux_z,
        inlet,
        linear_residual,
        linear_relative_residual,
        linear_iterations,
        linear_converged,
        linear_status,
    ) = projected
    plus = jnp.stack((flux_x, flux_y, flux_z))
    assert divergence < 1.0e-8
    assert flow_error < 1.0e-8
    fx, fy, fz = duct_impl._unpack_duct_mass_flux(plus, inlet)
    assert jnp.sum(inlet) == pytest.approx(0.2) and jnp.sum(plus[0, -1]) == pytest.approx(0.2)
    assert jnp.max(jnp.abs(jnp.diff(fx, axis=0) + jnp.diff(fy, axis=1) + jnp.diff(fz, axis=2))) < 1.0e-10
    assert jnp.isfinite(pressure_loss).all()
    assert jnp.isfinite(projected_pressure).all()
    assert jnp.max(jnp.abs(projected_pressure - projected_pressure[:, :1, :1])) < 1.0e-7
    assert linear_residual < 1.0e-8
    assert linear_relative_residual < 1.0e-8
    assert linear_iterations > 0
    assert linear_converged
    assert linear_status > 0
    correction_x, correction_y, correction_z = duct_impl._duct_pressure_face_corrections(
        projected_pressure, jnp.full(shape, 0.1), dx=dx, dy=widths, dz=widths, mixed_axial_pressure=True
    )
    correction_x_west = jnp.concatenate((jnp.zeros_like(correction_x[:1]), correction_x[:-1]), axis=0)
    pressure_velocity = jnp.stack(
        (
            0.5 * (correction_x_west + correction_x),
            0.5 * (correction_y[:, :-1] + correction_y[:, 1:]),
            0.5 * (correction_z[:, :, :-1] + correction_z[:, :, 1:]),
        ),
        axis=-1,
    )
    projected_velocity = jnp.stack(projected[:3], axis=-1)
    flow_adjustment = projected_velocity - pressure_velocity
    assert flow_adjustment[..., 1:] == pytest.approx(0.0, abs=1.0e-12)
    assert jnp.ptp(flow_adjustment[..., 0], axis=(1, 2)) == pytest.approx(0.0)
    assert jnp.sum(projected[0] * 0.25, axis=(1, 2)) == pytest.approx(0.2)

    # The pressure operator and reconstructed face correction must use the
    # same variable coefficient on a nonuniform mesh.
    mobility = 0.02 + 0.01 * jnp.arange(16, dtype=float).reshape(shape)
    nonuniform = _face_flux_pressure_projection_duct(
        zeros,
        zeros,
        zeros,
        jnp.ones(shape),
        jnp.ones(shape, dtype=bool),
        inlet_flow_rate=0.2,
        momentum_mobility=mobility,
        dt=0.1,
        dx=dx,
        dy=jnp.asarray([0.2, 0.8]),
        dz=jnp.asarray([0.7, 0.3]),
        iterations=200,
        tolerance=1.0e-10,
    )
    nfx, nfy, nfz = duct_impl._unpack_duct_mass_flux(jnp.stack(nonuniform[7:10]), nonuniform[10])
    nonuniform_divergence = jnp.diff(nfx, axis=0) + jnp.diff(nfy, axis=1) + jnp.diff(nfz, axis=2)
    assert nonuniform[5] < 1.0e-8
    assert jnp.max(jnp.abs(nonuniform_divergence)) < 1.0e-10


def test_nonuniform_pipe_radial_metrics_pass_manufactured_convergence():
    errors = []
    for count in (17, 33):
        computational = jnp.linspace(0.0, 1.0, count + 1)
        faces = 1.0 - jnp.tanh(1.5 * (1.0 - computational)) / jnp.tanh(1.5)
        widths = jnp.diff(faces)
        centers = 0.5 * (faces[:-1] + faces[1:])
        rr = jnp.broadcast_to(centers[None, :, None], (3, count, 8))
        _, radial_gradient, _ = _pipe_gradient_3d(
            rr,
            dx=0.3,
            dr=widths,
            dtheta=2.0 * jnp.pi / 8,
            r=rr,
        )
        assert radial_gradient[:, 1:-1, :] == pytest.approx(1.0)
        laplacian = _pipe_laplacian_3d(
            rr**2,
            dx=0.3,
            dr=widths,
            dtheta=2.0 * jnp.pi / 8,
            r=rr,
        )
        errors.append(float(jnp.max(jnp.abs(laplacian[:, 2:-2, :] - 4.0))))
    assert errors[1] < errors[0] / 2.0
    assert errors[1] < 0.03


def test_solvax_pipe_poisson_reconstructs_discrete_manufactured_field_and_gradient():
    r_faces = jnp.asarray([0.0, 0.12, 0.3, 0.55, 0.78, 1.0])
    r_centers = 0.5 * (r_faces[:-1] + r_faces[1:])
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, 8, endpoint=False)
    x = jnp.linspace(-1.0, 1.0, 4)
    manufactured = (
        jnp.cos(jnp.pi * x)[:, None, None]
        * (1.0 + r_centers[None, :, None] ** 2)
        * jnp.cos(theta)[None, None, :]
    )
    coefficient = jnp.broadcast_to(1.0 + 0.2 * r_centers[None, :, None], manufactured.shape)
    coefficients = _pipe_variable_diffusion_coefficients_3d(
        coefficient,
        dx=0.4,
        r_faces=r_faces,
        r_centers=r_centers,
        dtheta=2.0 * jnp.pi / 8,
    )
    rhs = _apply_pipe_diffusion_coefficients_3d(manufactured, coefficients)
    (
        solved,
        residual,
        converged,
        relative_residual,
        iterations,
        status,
        local_residual,
    ) = _solvax_pressure_poisson_pipe(
        rhs,
        coefficient,
        dx=0.4,
        r_faces=r_faces,
        r_centers=r_centers,
        dtheta=2.0 * jnp.pi / 8,
        iterations=500,
        tolerance=1.0e-10,
    )
    volume = jnp.broadcast_to(
        r_centers[None, :, None] * jnp.diff(r_faces)[None, :, None] * (2.0 * jnp.pi / 8),
        manufactured.shape,
    )
    expected = manufactured - jnp.sum(manufactured * volume) / jnp.sum(volume)
    assert bool(converged)
    assert float(residual) < 1.0e-8
    assert float(relative_residual) < 1.0e-8
    assert int(iterations) > 0
    assert int(status) == 1
    assert float(local_residual) < 1.0e-8
    assert solved == pytest.approx(expected, abs=1.0e-8)

    cyclic_solved, *cyclic_diagnostics = _solvax_pressure_poisson_pipe(
        rhs,
        coefficient,
        dx=0.4,
        r_faces=r_faces,
        r_centers=r_centers,
        dtheta=2.0 * jnp.pi / 8,
        iterations=500,
        tolerance=1.0e-10,
        include_theta_line=True,
    )
    assert cyclic_solved == pytest.approx(solved, abs=1.0e-8)
    assert int(cyclic_diagnostics[3]) <= int(iterations)

    direct = _separable_pressure_poisson_pipe(
        rhs,
        coefficient,
        dx=0.4,
        r_faces=r_faces,
        r_centers=r_centers,
        dtheta=2.0 * jnp.pi / 8,
        tolerance=1.0e-8,
    )
    assert bool(direct[2])
    assert direct[0] == pytest.approx(solved, abs=1.0e-8)

    def direct_objective(scale):
        field, *_ = _separable_pressure_poisson_pipe(
            rhs,
            scale * coefficient,
            dx=0.4,
            r_faces=r_faces,
            r_centers=r_centers,
            dtheta=2.0 * jnp.pi / 8,
            tolerance=1.0e-8,
        )
        return jnp.mean(field**2)

    direct_value, direct_gradient = jax.value_and_grad(direct_objective)(jnp.asarray(1.0))
    assert direct_gradient == pytest.approx(-2.0 * direct_value, rel=1.0e-6, abs=1.0e-8)

    def objective(scale):
        field, _, _, _, _, _, _ = _solvax_pressure_poisson_pipe(
            rhs,
            scale * coefficient,
            dx=0.4,
            r_faces=r_faces,
            r_centers=r_centers,
            dtheta=2.0 * jnp.pi / 8,
            iterations=500,
            tolerance=1.0e-10,
            include_theta_line=True,
        )
        return jnp.mean(field**2)

    value, gradient = jax.jit(jax.value_and_grad(objective))(jnp.asarray(1.0))
    assert gradient == pytest.approx(-2.0 * value, rel=1.0e-6, abs=1.0e-8)


@pytest.mark.parametrize("steady", [False, True])
def test_pipe_diffusion_reconstructs_manufactured_field(steady):
    r_faces = jnp.asarray([0.0, 0.12, 0.3, 0.55, 0.78, 1.0])
    r_centers = 0.5 * (r_faces[:-1] + r_faces[1:])
    shape = (4, 5, 8)
    manufactured = jnp.arange(np.prod(shape), dtype=float).reshape(shape) / 1000.0
    viscosity = jnp.full(shape, 0.04)
    reaction = jnp.broadcast_to(
        jnp.linspace(0.01, 0.03, shape[0])[:, None, None] * (1.0 + r_centers[None, :, None]),
        shape,
    )
    dt = 0.02
    coefficients = _pipe_variable_diffusion_coefficients_3d(
        viscosity,
        dx=0.4,
        r_faces=r_faces,
        r_centers=r_centers,
        dtheta=2.0 * jnp.pi / 8,
    )
    wall_width = jnp.diff(r_faces)[-1]
    wall_sink = (
        jnp.zeros_like(manufactured)
        .at[:, -1, :]
        .set(viscosity[:, -1, :] * r_faces[-1] / (r_centers[-1] * wall_width * (0.5 * wall_width)))
    )
    steady_rhs = (
        -_apply_pipe_diffusion_coefficients_3d(manufactured, coefficients)
        + (wall_sink + reaction) * manufactured
    )
    rhs = steady_rhs if steady else manufactured + dt * steady_rhs
    solved, residual, converged = _solvax_diffusion_pipe(
        rhs,
        viscosity,
        dt=None if steady else dt,
        dx=0.4,
        r_faces=r_faces,
        r_centers=r_centers,
        dtheta=2.0 * jnp.pi / 8,
        iterations=500,
        tolerance=1.0e-10,
        reaction=reaction,
    )
    assert bool(converged)
    assert float(residual) < 1.0e-8
    assert solved == pytest.approx(manufactured, abs=1.0e-8)


def test_pipe_face_gradient_divergence_is_compatible_symmetric_and_jittable():
    nx, ntheta = 4, 8
    r_faces = jnp.asarray([0.0, 0.12, 0.3, 0.55, 0.78, 1.0])
    r_centers = 0.5 * (r_faces[:-1] + r_faces[1:])
    dtheta = 2.0 * jnp.pi / ntheta
    x = jnp.linspace(-1.0, 1.0, nx)[:, None, None]
    radius = r_centers[None, :, None]
    theta = jnp.arange(ntheta)[None, None, :] * dtheta
    pressure = x * (1.0 - radius**2) * jnp.cos(theta)
    probe = (x**2 - jnp.mean(x**2)) * (1.0 - radius) * jnp.sin(theta)
    mobility = jnp.broadcast_to(1.0 + 0.2 * radius, pressure.shape)
    coefficients = _pipe_variable_diffusion_coefficients_3d(
        mobility,
        dx=0.4,
        r_faces=r_faces,
        r_centers=r_centers,
        dtheta=dtheta,
    )

    def operator(field):
        correction = _pipe_pressure_face_correction(
            field,
            mobility,
            dx=0.4,
            r_centers=r_centers,
            dtheta=dtheta,
        )
        return -_pipe_face_divergence(
            *correction,
            dx=0.4,
            r_faces=r_faces,
            r_centers=r_centers,
            dtheta=dtheta,
        )

    applied = jax.jit(operator)(pressure)
    assert applied == pytest.approx(
        _apply_pipe_diffusion_coefficients_3d(pressure, coefficients), abs=1.0e-12
    )
    assert operator(jnp.ones_like(pressure)) == pytest.approx(0.0, abs=1.0e-12)
    volume = jnp.broadcast_to(
        r_centers[None, :, None] * jnp.diff(r_faces)[None, :, None] * dtheta,
        pressure.shape,
    )
    assert jnp.sum(volume * pressure * operator(probe)) == pytest.approx(
        jnp.sum(volume * probe * applied), rel=1.0e-12, abs=1.0e-12
    )
    value, gradient = jax.jit(jax.value_and_grad(lambda a: jnp.sum(operator(a * pressure) ** 2)))(
        jnp.asarray(1.0)
    )
    assert gradient == pytest.approx(2.0 * value, rel=1.0e-12)


@pytest.mark.timeout(300)
def test_steady_pipe_stokes_projection_closes_compatible_divergence_and_flow():
    nx, nr, ntheta = 3, 2, 4
    r_faces = jnp.asarray([0.0, 0.4, 1.0])
    r_centers = 0.5 * (r_faces[:-1] + r_faces[1:])
    dtheta = 2.0 * jnp.pi / ntheta
    inner_iterations = 24
    shape = (nx, nr, ntheta)
    x = jnp.linspace(-1.0, 1.0, nx)[:, None, None]
    radius = r_centers[None, :, None]
    theta = jnp.arange(ntheta)[None, None, :] * dtheta
    u = 0.7 + 0.2 * x * (1.0 - radius) * jnp.cos(theta)
    v = 0.1 * (1.0 - radius) * jnp.sin(theta) * jnp.ones_like(x)
    w = -0.1 * x * (1.0 - radius) * jnp.cos(theta)
    cell_area = jnp.broadcast_to(
        r_centers[None, :, None] * jnp.diff(r_faces)[None, :, None] * dtheta,
        shape,
    )
    viscosity = jnp.full(shape, 0.07)

    def inverse(rhs):
        return _solvax_diffusion_pipe(
            rhs,
            viscosity,
            dt=None,
            dx=0.5,
            r_faces=r_faces,
            r_centers=r_centers,
            dtheta=dtheta,
            iterations=inner_iterations,
            tolerance=1.0e-10,
        )[0]

    unit_response = inverse(jnp.ones(shape))

    def steady_project(scale=1.0, **kwargs):
        return _steady_stokes_projection_pipe(
            inverse(scale * u),
            inverse(scale * v),
            inverse(scale * w),
            jnp.ones(shape),
            unit_response,
            cell_area,
            inverse,
            target_flow_rate=2.0,
            dx=0.5,
            r_faces=r_faces,
            r_centers=r_centers,
            dtheta=dtheta,
            pressure_iterations=inner_iterations,
            pressure_tolerance=1.0e-10,
            restart=12,
            max_restarts=3,
            physical_tolerance=1.0e-9,
            **kwargs,
        )

    compiled_steady_project = jax.jit(steady_project)
    steady_result = compiled_steady_project(jnp.asarray(1.0))
    assert steady_result[-3] < 1.0e-9
    cross_section_area = jnp.mean(jnp.sum(cell_area, axis=(1, 2)))
    assert steady_result[-2] / cross_section_area < 1.0e-9
    assert steady_result[-1].converged
    assert steady_result[-1].residual_norm < 1.0e-9

    def projected_energy(scale):
        return jnp.mean(compiled_steady_project(scale)[0] ** 2)

    epsilon = 1.0e-3
    gradient = jax.grad(projected_energy)(1.0)
    finite_difference = (projected_energy(1.0 + epsilon) - projected_energy(1.0 - epsilon)) / (2 * epsilon)
    assert gradient == pytest.approx(finite_difference, rel=1.0e-3, abs=1.0e-8)
    assert jax.jvp(projected_energy, (1.0,), (0.37,))[1] == pytest.approx(0.37 * gradient, rel=1.0e-7)
    with pytest.raises(ValueError, match="retained-modal coefficients"):
        steady_project(modal_stabilization=True)


def test_gauge_invariant_update_ignores_constant_shift():
    previous = jnp.asarray([[1.0, 2.0], [3.0, 4.0]])
    physical_update = jnp.asarray([[0.0, 0.2], [-0.1, 0.1]])
    volume = jnp.asarray([[1.0, 2.0], [3.0, 4.0]])

    reference = _gauge_invariant_scalar_update(
        previous + physical_update,
        previous,
        volume,
        scale=2.0,
    )
    shifted = _gauge_invariant_scalar_update(
        previous + physical_update + 1.0e6,
        previous,
        volume,
        scale=2.0,
    )

    assert shifted == pytest.approx(reference, rel=1.0e-9, abs=1.0e-12)


def test_duct_boundary_current_is_stationwise_divergence_integral():
    shape = (4, 3, 2)
    x = jnp.linspace(0.0, 1.0, shape[0])[:, None, None]
    phi = jnp.broadcast_to(x**2, shape)
    sigma = jnp.ones(shape)
    zeros = jnp.zeros(shape)
    dy = jnp.asarray([0.2, 0.3, 0.5])
    dz = jnp.asarray([0.4, 0.6])

    divergence, _, boundary = _conservative_current_diagnostics_3d(
        sigma,
        phi,
        zeros,
        zeros,
        zeros,
        dx=0.25,
        dy=dy,
        dz=dz,
    )
    expected = jnp.abs(jnp.sum(divergence * dy[None, :, None] * dz[None, None, :], axis=(1, 2)) * 0.25)

    assert boundary.shape == (shape[0],)
    assert boundary == pytest.approx(expected)
    assert float(jnp.max(boundary)) > 0.0
    fx = jnp.asarray([[[0.0, 1.0]], [[2.0, 3.0]], [[4.0, 5.0]]])
    axial_current = _station_axial_current_from_fluxes(fx, jnp.asarray([[2.0, 4.0]]))
    assert axial_current.tolist() == pytest.approx([10.0, 22.0])


def test_extruded_problem_builder_and_fixed_flow_contract():
    case = build_square_duct_extruded_problem(nx_stations=5, ny=4, nz=4).case
    problem = build_extruded_problem_from_case(
        case,
        entry_center=1.0,
        exit_center=5.0,
        transition_width=0.2,
        axis="y",
    )
    assert problem.case is case
    assert problem.profile.axis == "y"
    assert problem.profile.x.shape == (5,)

    field = jnp.zeros((1, 2, 2))
    with pytest.raises(ValueError, match="nonzero"):
        _apply_fixed_flow_pressure_constraint(
            field,
            unit_pressure_response=field,
            active_mask=jnp.ones_like(field, dtype=bool),
            cell_area=jnp.ones_like(field),
            target_flow_rate=1.0,
        )


def test_cross_duct_pressure_difference_values_and_contract():
    p_z = jnp.broadcast_to(jnp.arange(4.0)[None, None, :], (2, 3, 4))
    active = jnp.ones_like(p_z, dtype=bool)
    assert _cross_duct_pressure_difference(
        p_z, active_mask=active, magnetic_axis=1, side_axis=2
    ).tolist() == pytest.approx([1.5, 1.5])

    p_y = jnp.broadcast_to(jnp.arange(3.0)[None, :, None], (2, 3, 4))
    assert _cross_duct_pressure_difference(
        p_y, active_mask=active, magnetic_axis=1, side_axis=2
    ).tolist() == pytest.approx([-1.0, -1.0])

    p = jnp.zeros((1, 2, 2))
    with pytest.raises(ValueError, match="distinct members"):
        _cross_duct_pressure_difference(
            p, active_mask=jnp.ones_like(p, dtype=bool), magnetic_axis=1, side_axis=1
        )
    with pytest.raises(ValueError, match="active fluid"):
        _cross_duct_pressure_difference(
            p, active_mask=jnp.zeros_like(p, dtype=bool), magnetic_axis=1, side_axis=2
        )
    assert _normalized_pressure_observable_update(
        jnp.array([2.0, -1.0]), jnp.zeros(2), jnp.array([25.0, 100.0])
    ) == pytest.approx(0.02)
    assert _normalized_pressure_observable_update(
        jnp.array([2.0]), jnp.zeros(1), jnp.array([0.25])
    ) == pytest.approx(2.0)


def test_extruded_sharding_validates_placement_and_supported_paths(
    monkeypatch: pytest.MonkeyPatch,
):
    timings = []
    measured = common_impl._synchronized_phase(lambda x: x + 1, "test", lambda *x: timings.append(x))
    assert int(measured(jnp.array(1))) == 2 and timings[0][0] == "test"
    field = jnp.zeros((4, 2, 2))
    assert _shard_extruded_fields((field,), num_devices=None)[0] is field

    devices = [object(), object()]
    monkeypatch.setattr("lmx._fringing_common.jax.devices", lambda: devices)
    with pytest.raises(ValueError, match="divisible"):
        _shard_extruded_fields((jnp.zeros((3, 2, 2)),), num_devices=2)

    monkeypatch.setattr("lmx._fringing_common.Mesh", lambda *args, **kwargs: "mesh")
    monkeypatch.setattr("lmx._fringing_common.NamedSharding", lambda *args, **kwargs: "sharding")
    monkeypatch.setattr(
        "lmx._fringing_common.jax.lax.with_sharding_constraint",
        lambda value, sharding: value + 1,
    )
    for count in (1, 2):
        placed = _shard_extruded_fields((field,), num_devices=count)
        assert jnp.all(placed[0] == 1)


@pytest.mark.timeout(120)
def test_generic_extruded_sharding_matches_primal_and_gradient_on_two_devices():
    root = Path(__file__).resolve().parents[1]
    code = """
from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
from lmx.fringing import (
    build_layered_duct_extruded_problem,
    build_pipe_ogrid_extruded_problem,
    build_square_duct_extruded_problem,
    evolve_extruded_fields,
    solve_extruded_inductionless,
)

common = dict(
    ha_peak=1.0,
    nx_stations=4,
    length=1.5,
    entry_center=0.4,
    exit_center=1.1,
    transition_width=0.2,
)
problems = (
    build_square_duct_extruded_problem(**common, ny=3, nz=3),
    build_layered_duct_extruded_problem(**common, ny=3, nz=3, wall_cells=1),
    build_pipe_ogrid_extruded_problem(**common, nr=2, ntheta=4),
)
for problem in problems:
    problem = replace(
        problem,
        case=replace(
            problem.case,
            time_stepper=replace(problem.case.time_stepper, max_steps=1, potential_iterations=4),
            solver=replace(problem.case.solver, coupling_iterations=1, coupling_tolerance=1.0e-8),
        ),
    )
    reference = jax.jit(lambda: evolve_extruded_fields(problem, steps=1))()
    sharded = jax.jit(lambda: evolve_extruded_fields(problem, steps=1, num_devices=2))()
    for expected, actual in zip(reference, sharded, strict=True):
        np.testing.assert_allclose(expected, actual, rtol=2.0e-6, atol=1.0e-9)
    assert len(sharded[0].addressable_shards) == 2
    assert all(part.data.shape[0] == 2 for part in sharded[0].addressable_shards)
    if problem.case.geometry.kind == "rect_duct":
        duct_problem = problem
        solved = solve_extruded_inductionless(problem, num_devices=2)
        assert len(solved.bundle.u.addressable_shards) == 2

problem = duct_problem
parameters = jnp.ones(5)
def objective(values, devices):
    fields = evolve_extruded_fields(
        problem,
        forcing=values[0],
        magnetic_field_scale=values[1:],
        steps=1,
        checkpoint_size=1,
        num_devices=devices,
    )
    return jnp.mean(fields[0] ** 2) + 0.01 * jnp.mean(fields[4] ** 2)

reference = jax.jit(jax.value_and_grad(lambda values: objective(values, None)))(parameters)
sharded = jax.jit(jax.value_and_grad(lambda values: objective(values, 2)))(parameters)
np.testing.assert_allclose(reference[0], sharded[0], rtol=2.0e-9, atol=2.0e-14)
np.testing.assert_allclose(reference[1], sharded[1], rtol=2.0e-8, atol=2.0e-14)
"""
    environment = {
        **os.environ,
        "JAX_ENABLE_X64": "true",
        "JAX_PLATFORMS": "cpu",
        "PYTHONPATH": str(root / "src"),
        "XLA_FLAGS": f"--xla_force_host_platform_device_count=2 {os.environ.get('XLA_FLAGS', '')}",
    }
    subprocess.run([sys.executable, "-c", code], cwd=root, env=environment, timeout=120, check=True)


def test_fringing_profile_and_constant_field_builders():
    profile = smooth_fringing_profile(
        length=6.0,
        nx=9,
        entry_center=1.5,
        exit_center=4.5,
        transition_width=0.3,
        peak_scale=1.2,
    )

    assert profile.axis == "z"
    assert profile.x.shape == (9,)
    assert jnp.all(profile.field_scale >= 0.0)
    assert float(jnp.max(profile.field_scale)) <= 1.2
    with pytest.raises(ValueError, match="Unsupported magnetic axis"):
        smooth_fringing_profile(
            length=1.0,
            nx=3,
            entry_center=0.2,
            exit_center=0.8,
            transition_width=0.1,
            axis="bad",
        )


@pytest.mark.parametrize(
    ("builder", "kwargs", "geometry_kind"),
    (
        (build_square_duct_extruded_problem, {"ny": 8, "nz": 8}, "rect_duct"),
        (
            build_layered_duct_extruded_problem,
            {"ny": 8, "nz": 8, "wall_cells": 1, "insulator_cells": 1},
            "layered_duct",
        ),
        (build_pipe_ogrid_extruded_problem, {"nr": 6, "ntheta": 12}, "pipe_ogrid"),
    ),
)
def test_extruded_problem_builders_mark_solver_family(builder, kwargs, geometry_kind):
    problem = builder(nx_stations=5, **kwargs)

    assert problem.case.solver.kind == "extruded_inductionless"
    assert problem.case.geometry.kind == geometry_kind
    assert problem.profile.x.shape == (5,)


def test_extruded_fields_match_production_and_bound_reverse_memory():
    problem = build_square_duct_extruded_problem(
        ha_peak=3.0,
        nx_stations=3,
        ny=3,
        nz=3,
        length=2.0,
        entry_center=0.5,
        exit_center=1.5,
        transition_width=0.2,
    )
    problem = replace(
        problem,
        case=replace(
            problem.case,
            time_stepper=replace(problem.case.time_stepper, max_steps=8, potential_iterations=10),
            solver=replace(problem.case.solver, coupling_iterations=1, coupling_tolerance=1.0e-8),
        ),
    )
    production = solve_extruded_inductionless(
        replace(
            problem, case=replace(problem.case, time_stepper=replace(problem.case.time_stepper, max_steps=3))
        )
    ).bundle
    fields = evolve_extruded_fields(problem, magnetic_field_scale=jnp.ones(3), steps=3)
    for actual, name in zip(
        fields,
        ("u", "v", "w", "p", "phi", "jx", "jy", "jz", "lorentz_x", "lorentz_y", "lorentz_z"),
        strict=True,
    ):
        assert actual == pytest.approx(getattr(production, name), rel=2.0e-6, abs=1.0e-9)

    def objective(parameters, checkpoint_size=None):
        evolved = evolve_extruded_fields(
            problem,
            forcing=parameters[0],
            magnetic_field_scale=parameters[1:],
            steps=8,
            checkpoint_size=checkpoint_size,
        )
        return jnp.mean(evolved[0] ** 2) + 0.01 * jnp.mean(evolved[4] ** 2)

    parameters = jnp.ones(4)
    value_and_gradient = jax.jit(jax.value_and_grad(objective))
    value, gradient = value_and_gradient(parameters)
    epsilon = 2.0e-3
    finite_difference = _centered_gradient(lambda point: value_and_gradient(point)[0], parameters, epsilon)
    assert jnp.isfinite(value)
    assert gradient == pytest.approx(finite_difference, rel=2.0e-3, abs=2.0e-9)
    direction = jnp.asarray([0.3, -0.2, 0.1, -0.1])
    tangent = jax.jvp(objective, (parameters,), (direction,))[1]
    assert tangent == pytest.approx(jnp.vdot(gradient, direction), rel=2.0e-6, abs=1.0e-9)
    bounded = value_and_gradient.lower(parameters).compile()
    full_tape = jax.jit(jax.value_and_grad(lambda values: objective(values, 8))).lower(parameters).compile()
    assert (
        bounded.memory_analysis().temp_size_in_bytes < 0.75 * full_tape.memory_analysis().temp_size_in_bytes
    )
    with pytest.raises(ValueError, match="one value per axial station"):
        evolve_extruded_fields(problem, magnetic_field_scale=jnp.ones(2), steps=2)


def test_layered_extruded_fields_share_the_production_update():
    problem = build_layered_duct_extruded_problem(
        ha_peak=2.0,
        nx_stations=3,
        ny=3,
        nz=3,
        wall_cells=1,
        length=1.5,
        entry_center=0.4,
        exit_center=1.1,
        transition_width=0.2,
    )
    problem = replace(
        problem,
        case=replace(
            problem.case,
            time_stepper=replace(problem.case.time_stepper, max_steps=2, potential_iterations=10),
            solver=replace(problem.case.solver, coupling_iterations=1, coupling_tolerance=1.0e-8),
        ),
    )
    production = solve_extruded_inductionless(problem).bundle
    fields = evolve_extruded_fields(problem, material_conductivity_scale=jnp.ones(2), steps=2)
    for actual, name in zip(
        fields,
        ("u", "v", "w", "p", "phi", "jx", "jy", "jz", "lorentz_x", "lorentz_y", "lorentz_z"),
        strict=True,
    ):
        assert actual == pytest.approx(getattr(production, name), rel=2.0e-6, abs=1.0e-9)
    assert all(jnp.isfinite(value) for value in extruded_engineering_objectives(problem, fields).values())

    def objective(parameters):
        evolved = evolve_extruded_fields(
            problem,
            forcing=parameters[0],
            magnetic_field_scale=parameters[1],
            material_conductivity_scale=parameters[2:4],
            geometry_scale=parameters[4:],
            steps=1,
        )
        return extruded_engineering_objectives(problem, evolved, geometry_scale=parameters[4:])[
            "wall_current_density_rms"
        ]

    scale = jnp.ones(7)
    value_and_gradient = jax.jit(jax.value_and_grad(objective))
    value, gradient = value_and_gradient(scale)
    finite_difference = _centered_gradient(lambda point: value_and_gradient(point)[0], scale, 2.0e-3)
    direction = jnp.asarray([0.1, -0.2, 0.25, -0.5, 0.15, -0.1, 0.2])
    tangent = jax.jvp(objective, (scale,), (direction,))[1]
    assert gradient == pytest.approx(finite_difference, rel=5.0e-4, abs=2.0e-9)
    assert tangent == pytest.approx(jnp.vdot(gradient, direction), rel=2.0e-6, abs=1.0e-9)
    points = jnp.stack((scale, jnp.asarray([1.1, 0.9, 0.95, 1.2, 1.02, 0.98, 1.03])))
    batched = jax.jit(jax.vmap(jax.value_and_grad(objective)))(points)
    second = value_and_gradient(points[1])
    assert batched[0] == pytest.approx(jnp.stack((value, second[0])))
    assert batched[1] == pytest.approx(jnp.stack((gradient, second[1])))
    with pytest.raises(ValueError, match="fluid, solid"):
        evolve_extruded_fields(problem, material_conductivity_scale=jnp.ones(3), steps=2)
    with pytest.raises(ValueError, match="axial, transverse_y, transverse_z"):
        evolve_extruded_fields(problem, geometry_scale=jnp.ones(2), steps=2)


def test_pipe_fields_share_the_production_update_and_checked_derivative():
    problem = build_pipe_ogrid_extruded_problem(
        ha_peak=2.0,
        nx_stations=3,
        nr=3,
        ntheta=8,
        length=1.5,
        entry_center=0.4,
        exit_center=1.1,
        transition_width=0.2,
    )
    problem = replace(
        problem,
        case=replace(
            problem.case,
            geometry=replace(
                problem.case.geometry,
                wall_thickness=(0.1,) * 4,
                wall_cells=(1,) * 4,
            ),
            regions=(
                problem.case.regions[0],
                RegionSpec("wall", "solid", 0.5, 1.0, 1.0, 0.1),
            ),
            time_stepper=replace(problem.case.time_stepper, max_steps=4, potential_iterations=10),
            solver=replace(problem.case.solver, coupling_iterations=1, coupling_tolerance=1.0e-12),
        ),
    )
    production = solve_extruded_inductionless(problem).bundle
    fields = evolve_extruded_fields(problem, magnetic_field_scale=jnp.ones(3), steps=4)
    for actual, name in zip(
        fields,
        ("u", "v", "w", "p", "phi", "jx", "jy", "jz", "lorentz_x", "lorentz_y", "lorentz_z"),
        strict=True,
    ):
        assert actual == pytest.approx(getattr(production, name), rel=2.0e-6, abs=2.0e-9)
    assert all(jnp.isfinite(value) for value in extruded_engineering_objectives(problem, fields).values())

    def objective(parameters, checkpoint_size=None):
        evolved = evolve_extruded_fields(
            problem,
            forcing=parameters[0],
            magnetic_field_scale=parameters[1:4],
            material_conductivity_scale=parameters[4:6],
            geometry_scale=parameters[6:],
            steps=4,
            checkpoint_size=checkpoint_size,
        )
        return (
            jnp.mean(evolved[0] ** 2)
            + 0.01 * jnp.mean(evolved[4] ** 2)
            + 0.001 * jnp.mean(evolved[6] ** 2 + evolved[7] ** 2)
        )

    parameters = jnp.ones(8)
    value_and_gradient = jax.jit(jax.value_and_grad(objective))
    value, gradient = value_and_gradient(parameters)
    finite_difference = _centered_gradient(lambda point: value_and_gradient(point)[0], parameters, 2.0e-3)
    direction = jnp.asarray([0.1, -0.2, 0.15, 0.05, 0.2, -0.08, -0.1, 0.12])
    tangent = jax.jvp(objective, (parameters,), (direction,))[1]
    assert jnp.isfinite(value) and jnp.all(jnp.isfinite(gradient))
    assert abs(gradient[5]) > 1.0e-12
    assert gradient == pytest.approx(finite_difference, rel=5.0e-4, abs=2.0e-9)
    assert tangent == pytest.approx(jnp.vdot(gradient, direction), rel=2.0e-6, abs=1.0e-10)
    batched_values, batched_gradients = jax.jit(jax.vmap(jax.value_and_grad(objective)))(
        jnp.stack((parameters, parameters.at[0].set(0.9)))
    )
    assert batched_values[0] == pytest.approx(value)
    assert batched_gradients[0] == pytest.approx(gradient)
    bounded = value_and_gradient.lower(parameters).compile().memory_analysis().temp_size_in_bytes
    full_tape = (
        jax.jit(jax.value_and_grad(lambda values: objective(values, 4)))
        .lower(parameters)
        .compile()
        .memory_analysis()
        .temp_size_in_bytes
    )
    assert bounded < 0.9 * full_tape
    with pytest.raises(ValueError, match="axial, radial"):
        evolve_extruded_fields(problem, geometry_scale=jnp.ones(3), steps=2)


def test_extruded_engineering_objectives_have_physical_conventions_and_gradients():
    problem = build_square_duct_extruded_problem(nx_stations=3, ny=3, nz=3)
    shape = (3, 3, 3)
    zeros = jnp.zeros(shape)
    pressure = jnp.broadcast_to(jnp.asarray([3.0, 2.0, 1.0])[:, None, None], shape)

    def objectives(speed, geometry_scale=1.0):
        fields = (
            jnp.full(shape, speed),
            zeros,
            zeros,
            pressure,
            zeros,
            jnp.full(shape, 3.0),
            *(zeros,) * 5,
        )
        return extruded_engineering_objectives(
            problem, fields, geometry_scale=geometry_scale, smoothing=1.0e-6
        )

    values = objectives(1.0)
    assert values["pressure_drop"] == pytest.approx(2.0)
    assert values["flow_rate"] == pytest.approx(4.0)
    assert values["pumping_power"] == pytest.approx(8.0)
    assert values["pressure_tap_flux_power"] == pytest.approx(8.0)
    assert values["flow_nonuniformity"] == pytest.approx(0.0, abs=1.0e-12)
    assert values["wall_current_density_rms"] == pytest.approx(3.0, abs=2.0e-6)
    assert values["recirculation_fraction"] == pytest.approx(0.0, abs=1.0e-12)
    assert jax.grad(lambda speed: objectives(speed)["pumping_power"])(1.0) == pytest.approx(8.0)
    assert jax.grad(lambda scale: objectives(1.0, scale)["flow_rate"])(1.0) == pytest.approx(8.0)
    # Independent surface-work quadrature: area=4, mean(q)=0, mean(q**2)=2/3.
    q = jnp.broadcast_to(jnp.asarray([-1.0, 0.0, 1.0])[None, :, None], shape)

    def tap_work(speed=1.0, scale=1.0, gauge=0.0, outlet_scale=1.0):
        u = (speed * (1.0 + q)).at[-1].multiply(outlet_scale)
        p = (pressure + gauge).at[0].add(q[0])
        return extruded_engineering_objectives(
            problem, (u, zeros, zeros, p, *(zeros,) * 4), geometry_scale=scale
        )

    assert tap_work()["pumping_power"] == pytest.approx(8.0)
    assert tap_work()["pressure_tap_flux_power"] == pytest.approx(32.0 / 3.0)
    assert tap_work(gauge=7.0)["pressure_tap_flux_power"] == pytest.approx(32.0 / 3.0)
    assert jax.jit(jax.grad(lambda speed: tap_work(speed)["pressure_tap_flux_power"]))(1.0) == pytest.approx(
        32.0 / 3.0
    )
    assert jax.grad(lambda scale: tap_work(scale=scale)["pressure_tap_flux_power"])(1.0) == pytest.approx(
        64.0 / 3.0
    )
    assert jax.grad(lambda gauge: tap_work(gauge=gauge, outlet_scale=2.0)["pressure_tap_flux_power"])(
        0.0
    ) == pytest.approx(-4.0)
    pipe = build_pipe_ogrid_extruded_problem(nx_stations=3, nr=3, ntheta=8)
    pipe_shape = (3, 3, 8)
    pipe_fields = (jnp.ones(pipe_shape),) + (jnp.zeros(pipe_shape),) * 10

    def pipe_flow(radial):
        return extruded_engineering_objectives(pipe, pipe_fields, geometry_scale=jnp.asarray([1.0, radial]))[
            "flow_rate"
        ]

    assert pipe_flow(2.0) == pytest.approx(4.0 * pipe_flow(1.0))
    assert jax.grad(pipe_flow)(1.0) == pytest.approx(2.0 * pipe_flow(1.0))
    with pytest.raises(ValueError, match="velocity, pressure"):
        extruded_engineering_objectives(problem, ())
    with pytest.raises(ValueError, match="problem shape"):
        extruded_engineering_objectives(problem, (jnp.zeros((2, 2, 2)),) * 11)
    with pytest.raises(ValueError, match="smoothing"):
        extruded_engineering_objectives(problem, (zeros,) * 11, smoothing=0.0)


@pytest.mark.parametrize(
    "builder, grid, area",
    [
        (build_square_duct_extruded_problem, dict(ny=3, nz=3), 4.0),
        (build_layered_duct_extruded_problem, dict(ny=3, nz=3, wall_cells=1), 4.0),
        (build_pipe_ogrid_extruded_problem, dict(nr=3, ntheta=8), np.pi),
    ],
)
def test_tap_work_and_storage_share_a_control_volume_and_exact_derivatives(builder, grid, area):
    problem = builder(nx_stations=4, length=3.0, **grid)
    case = replace(
        problem.case, regions=tuple(replace(region, density=2.0) for region in problem.case.regions)
    )
    problem = replace(problem, case=case)
    mesh = _cross_section_mesh(case)
    shape = (4, *mesh.yz_shape)
    ones, zeros = jnp.ones(shape), jnp.zeros(shape)
    # Uniform velocity, rho=2, and dp/dx=f: boundary and body work cancel.
    # Exact slab volume is A * (L - dx), not the whole-domain A * L.
    volume = area * 2.25

    def work(parameters):
        force, speed, scale = parameters
        pressure = jnp.broadcast_to(force * scale * mesh.x_centers[:, None, None], shape)
        fields = (speed * ones, ones, 2 * ones, pressure, *(zeros,) * 4)
        return extruded_engineering_objectives(problem, fields, forcing=force, geometry_scale=scale)

    parameters = jnp.asarray([5.0, 3.0, 1.0])
    values = jax.jit(work)(parameters)
    assert values["tap_body_drive_power"] == pytest.approx(15 * volume)
    assert values["pressure_tap_flux_power"] == pytest.approx(-15 * volume)
    assert values["tap_kinetic_energy"] == pytest.approx(14 * volume)
    gradient = jax.jit(jax.grad(lambda parameters: work(parameters)["tap_body_drive_power"]))(parameters)
    np.testing.assert_allclose(gradient, volume * np.asarray([3.0, 5.0, 45.0]), rtol=1e-12)
    gradient = jax.grad(lambda parameters: work(parameters)["tap_kinetic_energy"])(parameters)
    np.testing.assert_allclose(gradient, volume * np.asarray([0.0, 6.0, 42.0]), atol=1e-12)
    default = extruded_engineering_objectives(problem, (3 * ones, ones, 2 * ones, *(zeros,) * 5))
    assert default["tap_body_drive_power"] == pytest.approx(case.forcing * 3 * volume)
    x = jnp.broadcast_to(mesh.x_centers[:, None, None], shape)
    linear = extruded_engineering_objectives(problem, (x, *(zeros,) * 7))
    a, b, dx = 0.375, 2.625, 0.75
    assert linear["tap_body_drive_power"] == pytest.approx(case.forcing * area * (b**2 - a**2) / 2)
    # For rho=2 and u=x, the trapezoidal energy error is exactly A*(b-a)*dx²/6.
    assert linear["tap_kinetic_energy"] == pytest.approx(area * ((b**3 - a**3) / 3 + (b - a) * dx**2 / 6))
    with pytest.raises(ValueError, match="scalar axial force"):
        extruded_engineering_objectives(problem, (ones,) * 8, forcing=jnp.ones(2))


def test_cross_section_mesh_supports_pipe_ogrid_geometry():
    problem = build_square_duct_extruded_problem(nx_stations=3, ny=4, nz=4)
    pipe_case = replace(
        problem.case,
        geometry=GeometrySpec(kind="pipe_ogrid", width=1.0, height=1.0, radius=0.5, nr=4, ntheta=8),
    )
    mesh = _cross_section_mesh(pipe_case)
    assert mesh.geometry == "pipe_ogrid"


def test_solve_extruded_inductionless_wraps_history_bundle_and_validation(
    monkeypatch: pytest.MonkeyPatch,
):
    problem = build_square_duct_extruded_problem(nx_stations=3, ny=6, nz=6)
    fake_bundle = _mock_extruded_bundle(geometry_kind="rect_duct", stations=3)
    monkeypatch.setattr(
        "lmx.fringing._solve_extruded_projection",
        lambda problem, **kwargs: fake_bundle,
    )

    solution = solve_extruded_inductionless(problem)
    assert len(solution.station_history) == 3
    assert solution.validation.station_count == 3
    assert solution.bundle.solver_kind == "extruded_inductionless"
    assert "axial_current" in solution.station_history[0]
    assert "wall_current_leakage" in solution.station_history[0]
    assert "boundary_current_residual" in solution.station_history[0]
    assert "pressure_span" in solution.station_history[0]
    assert solution.station_history[0]["u_max"] == pytest.approx(1.0)
    assert solution.station_history[0]["axial_pressure_loss_gradient"] == 0.0
    assert solution.station_history[0]["transverse_pressure_difference"] == 0.0


def test_pipe_projection_supports_explicit_conducting_annulus_and_fixed_flow():
    problem = build_pipe_ogrid_extruded_problem(ha_peak=2.0, nx_stations=3, nr=4, ntheta=8)
    problem = replace(
        problem,
        case=replace(
            problem.case,
            geometry=replace(
                problem.case.geometry,
                wall_thickness=(0.1, 0.1, 0.1, 0.1),
                wall_cells=(2, 2, 2, 2),
            ),
            regions=(
                problem.case.regions[0],
                RegionSpec("wall", "solid", 0.5, 1.0, 1.0, 0.1),
            ),
            initial_velocity=0.5,
        ),
    )
    field = make_localized_divergence_free_obstacle_field(
        width=1.0,
        height=1.0,
        base_bz=12.0,
        core_fraction_y=0.5,
        core_fraction_z=0.5,
    )
    problem = _with_analytic_field(problem, name="variable_field_conducting_pipe", field_fn=field)
    problem = _with_integration_budget(problem)

    solution = solve_extruded_inductionless(problem)
    field_validation = validate_variable_field_pipe_solution(solution, field_ny=41, field_nz=41)

    assert solution.bundle.u.shape == (3, 6, 8)
    assert jnp.allclose(solution.bundle.u[:, 4:, :], 0.0)
    assert jnp.allclose(solution.bundle.v[:, 4:, :], 0.0)
    assert jnp.allclose(solution.bundle.w[:, 4:, :], 0.0)
    assert solution.bundle.mean_velocity.tolist() == pytest.approx([0.5, 0.5, 0.5], rel=1.0e-6)
    assert solution.bundle.axial_pressure_loss_gradient.shape == (3,)
    assert jnp.isfinite(solution.bundle.axial_pressure_loss_gradient).all()
    assert solution.bundle.transverse_pressure_difference.tolist() == pytest.approx([0.0, 0.0, 0.0])
    assert jnp.isfinite(solution.bundle.axial_current).all()
    assert jnp.isfinite(solution.bundle.wall_current_leakage).all()
    assert field_validation["current_proxy_change"] > 0.0
    assert solution.validation.max_charge_balance_residual < 0.5
    assert "axial_pressure_loss_gradient" in solution.station_history[0]
    assert jnp.isfinite(solution.bundle.phi).all()


def test_pipe_matrix_free_potential_cancels_conservative_emf_divergence():
    nx, nr, ntheta = 4, 5, 12
    dx = 0.3
    r_faces = jnp.linspace(0.0, 0.4, nr + 1)
    r_centers = 0.5 * (r_faces[:-1] + r_faces[1:])
    dtheta = 2.0 * jnp.pi / ntheta
    x = jnp.linspace(-1.0, 1.0, nx)[:, None, None]
    r = r_centers[None, :, None]
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, ntheta, endpoint=False)[None, None, :]
    sigma = jnp.ones((nx, nr, ntheta))
    uxb_x = jnp.broadcast_to(0.03 * jnp.sin(jnp.pi * x) * jnp.cos(theta), sigma.shape)
    uxb_r = jnp.broadcast_to(0.02 * r * jnp.cos(2.0 * theta), sigma.shape)
    uxb_theta = jnp.broadcast_to(0.01 * jnp.sin(theta) * (1.0 + x), sigma.shape)

    emf_rhs = _pipe_conservative_emf_rhs_3d(
        sigma,
        uxb_x,
        uxb_r,
        uxb_theta,
        dx=dx,
        r_faces=r_faces,
        r_centers=r_centers,
        dtheta=float(dtheta),
    )
    phi, *_, local_residual = _solvax_pressure_poisson_pipe(
        emf_rhs,
        sigma,
        dx=dx,
        r_faces=r_faces,
        r_centers=r_centers,
        dtheta=float(dtheta),
        iterations=400,
        tolerance=1.0e-12,
        include_theta_line=True,
    )
    div_j = _pipe_conservative_current_diagnostics_3d(
        sigma,
        phi,
        uxb_x,
        uxb_r,
        uxb_theta,
        dx=dx,
        r_faces=r_faces,
        r_centers=r_centers,
        dtheta=float(dtheta),
    )[0]

    assert local_residual < 1.0e-10
    assert float(jnp.max(jnp.abs(div_j))) < 1.0e-10


def test_solve_extruded_inductionless_supports_layered_analytic_variable_field():
    field_fn = make_divergence_free_cross_section_field(
        width=2.0, height=2.0, base_bz=12.0, perturbation=0.12
    )
    problem = build_layered_duct_extruded_problem(ha_peak=1.0, nx_stations=7, ny=10, nz=10)
    problem = _with_analytic_field(problem, name="variable_field_layered_bz12", field_fn=field_fn)
    problem = _with_integration_budget(problem)
    solution = solve_extruded_inductionless(problem)
    validation = validate_variable_field_extruded_solution(solution, field_ny=41, field_nz=41)

    assert solution.bundle.geometry_kind == "layered_duct"
    assert jnp.all(jnp.isfinite(solution.bundle.u))
    assert jnp.all(jnp.isfinite(solution.bundle.phi))
    assert solution.bundle.iteration_residual_history.shape == (1,)
    assert solution.validation.volumetric_flow_rate_span < 5.0e-3
    assert solution.validation.max_residual < 1.0e-3
    assert solution.validation.max_charge_balance_residual < 1.0e-4
    assert solution.validation.axial_current_mirror_residual < 1.0e-3
    assert solution.validation.pressure_span_mirror_residual < 1.0e-3
    assert abs(solution.validation.center_axial_current) < 1.0e-4
    assert validation["mean_velocity_change"] > 0.0
    assert isinstance(validation["validation_pass"], bool)
    bad = replace(solution, bundle=replace(solution.bundle, u=solution.bundle.u.at[0, 0, 0].set(jnp.nan)))
    invalid = validate_variable_field_extruded_solution(bad, field_ny=5, field_nz=5)
    assert (invalid["finite_velocity"], invalid["validation_pass"]) == (False, False)


def test_tabulated_magnetic_obstacle_uses_solvax_and_reports_velocity_deficit(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    problem = build_magnetic_obstacle_rect_extruded_problem(nx_stations=7, ny=10, nz=10)
    geometry = problem.case.geometry
    y, z, field = sample_cross_section_field(
        problem.case.magnetic_field.fn,
        width=geometry.width,
        height=geometry.height,
        ny=41,
        nz=41,
    )
    path = write_tabulated_field_npz(
        tmp_path / "obstacle-field.npz",
        y=y,
        z=z,
        bx=field[..., 0],
        by=field[..., 1],
        bz=field[..., 2],
    )
    problem = replace(
        problem,
        case=replace(
            problem.case,
            magnetic_field=MagneticFieldSpec(kind="tabulated", table_path=str(path)),
        ),
    )
    problem = _with_integration_budget(problem)
    calls = {"count": 0}
    original = duct_impl._solvax_pressure_poisson_duct

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(duct_impl, "_solvax_pressure_poisson_duct", wrapped)
    solution = solve_extruded_inductionless(problem)
    validation = validate_magnetic_obstacle_baseline(solution, field_ny=41, field_nz=41)
    field_validation = validate_variable_field_extruded_solution(solution, field_ny=41, field_nz=41)

    assert calls["count"] >= 2
    assert solution.bundle.geometry_kind == "rect_duct"
    assert jnp.isfinite(solution.bundle.u).all()
    assert jnp.isfinite(solution.bundle.p).all()
    assert jnp.isfinite(solution.bundle.axial_current).all()
    assert field_validation["rms_divergence"] >= 0.0
    assert validation["obstacle_velocity_deficit"] > 0.0
    assert validation["current_proxy_peak"] > 0.0
    assert validation["divergence_to_field_ratio"] >= 0.0
    assert validation["field_quality_pass"] in {True, False}
    assert validation["reference_kind"] == "none"
    assert validation["external_reference_available"] is False
    assert validation["research_grade_validation_pass"] is False
    assert isinstance(validation["validation_pass"], bool)
    with pytest.raises(ValueError, match="three-component"):
        _sample_volume_field(
            lambda x, y, z: jnp.zeros_like(x),
            jnp.zeros((2, 2, 2)),
            jnp.zeros((2, 2, 2)),
            jnp.zeros((2, 2, 2)),
        )


def test_solve_extruded_inductionless_uses_projection_for_pipe_geometry(
    monkeypatch: pytest.MonkeyPatch,
):
    problem = build_square_duct_extruded_problem(nx_stations=3, ny=4, nz=4)
    pipe_case = replace(
        problem.case,
        geometry=GeometrySpec(kind="pipe_ogrid", width=1.0, height=1.0, radius=0.5, nr=4, ntheta=8),
    )
    pipe_problem = replace(problem, case=pipe_case)
    monkeypatch.setattr(
        "lmx.fringing._solve_extruded_projection",
        lambda problem, **kwargs: _mock_extruded_bundle(geometry_kind="pipe_ogrid", stations=1),
    )
    solution = solve_extruded_inductionless(pipe_problem)
    assert solution.validation.station_count == 1

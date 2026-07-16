import jax
import jax.numpy as jnp
import numpy as np
import pytest
from dataclasses import replace
from types import SimpleNamespace

import lmx.fringing as fringing_impl
from solvax import block_thomas_solve

from lmx.field_models import (
    make_divergence_free_cross_section_field,
    make_maxwell_consistent_fringe_field,
    sample_cross_section_field,
    write_tabulated_field_npz,
)
from lmx.fringing import (
    _apply_fixed_flow_pressure_constraint,
    _apply_pipe_diffusion_coefficients_3d,
    _cross_duct_pressure_difference,
    _conservative_current_diagnostics_3d,
    _distance_weighted_harmonic_mean,
    _thin_wall_interface_mean,
    _enforce_stationwise_flow_rate_3d,
    _gradient_3d,
    _gauge_invariant_scalar_update,
    _laplacian_3d,
    _normalized_pressure_observable_update,
    _sample_volume_field,
    _pipe_poisson_jacobi_3d,
    build_bent_pipe_extruded_problem,
    build_extruded_problem_from_case,
    build_layered_duct_extruded_problem,
    build_magnetic_obstacle_rect_extruded_problem,
    build_pipe_ogrid_extruded_problem,
    build_square_duct_extruded_problem,
    build_square_duct_fringing_benchmark,
    build_variable_field_duct_extruded_problem,
    build_variable_field_bent_pipe_extruded_problem,
    build_variable_field_layered_extruded_problem,
    build_variable_field_pipe_ogrid_extruded_problem,
    build_wham_mirror_pipe_extruded_problem,
    _cross_section_mesh,
    _pipe_conservative_current_diagnostics_3d,
    _pipe_conservative_emf_rhs_3d,
    _pipe_face_divergence,
    _pipe_pressure_face_correction,
    _pipe_gradient_3d,
    _pipe_laplacian_3d,
    _pipe_poisson_sparse_3d,
    _pipe_radial_fluid_count,
    _pipe_variable_diffusion_coefficients_3d,
    _rectangular_fluid_bounds,
    _solvax_pressure_poisson_duct,
    _shard_extruded_fields,
    _solvax_pressure_poisson_pipe,
    _solvax_implicit_momentum_duct,
    _solvax_diffusion_pipe,
    _separable_pressure_poisson_pipe,
    _steady_stokes_projection_pipe,
    _spacing_vector,
    _station_axial_current_from_fluxes,
    _poisson_jacobi_3d,
    _variable_coefficient_poisson_jacobi_3d,
    _variable_coefficient_poisson_sparse_3d,
    _variable_coefficient_residual_3d,
    _conservative_current_fluxes_3d,
    _face_flux_pressure_projection_duct,
    _face_flux_pressure_projection_pipe,
    _fixed_flow_face_flux_projection_pipe,
    clone_case_with_field,
    magnetic_obstacle_literature_reference_cases,
    run_extruded_inductionless_slice,
    run_fringing_station_sweep,
    solve_extruded_inductionless,
    smooth_fringing_profile,
    validate_bent_pipe_low_de_baseline,
    validate_extruded_inductionless_solution,
    validate_magnetic_obstacle_benchmark,
    validate_magnetic_obstacle_baseline,
    validate_magnetic_obstacle_external_readiness,
    validate_magnetic_obstacle_literature_slice,
    validate_wham_mirror_pipe_baseline,
    validate_variable_field_pipe_solution,
    validate_variable_field_extruded_solution,
)
from lmx.specs import GeometrySpec
from lmx.specs import MagneticFieldSpec, RegionSpec


pytestmark = pytest.mark.unit


def test_b2_canonical_shell_widths_remove_realization_thickness():
    nominal = jnp.asarray([0.01, 0.01, 0.4, 0.4, 0.4, 0.4, 0.4, 0.01, 0.01])
    confirmation = nominal.at[:2].divide(2.0).at[-2:].divide(2.0)

    expected = fringing_impl._canonical_shell_widths(nominal, 2, 7)
    observed = fringing_impl._canonical_shell_widths(confirmation, 2, 7)

    assert observed.tolist() == pytest.approx(expected.tolist())
    assert float(jnp.sum(observed[:2])) == pytest.approx(0.02)
    assert float(jnp.sum(observed[-2:])) == pytest.approx(0.02)


def test_fringing_jit_cache_reuses_the_first_compiled_kernel():
    fringing_impl._FRINGING_JIT_CACHE.clear()
    first = object()
    key = ("operator", "configuration")

    assert fringing_impl._reuse_fringing_jit(key, first) is first
    assert fringing_impl._reuse_fringing_jit(key, object()) is first


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
        precondition = fringing_impl._axial_mean_preconditioner_3d(
            volume, west, east, gauge=gauge)
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
    spacing = jnp.concatenate(
        (jnp.full(3, 0.02 / 3.0), jnp.full(12, 2.0 / 12.0), jnp.full(3, 0.02 / 3.0))
    )
    mask = jnp.zeros((nx, cross, cross), dtype=bool)
    mask = mask.at[:, 3:15, 3:15].set(True)
    conductivity = jnp.where(mask, 1.0, 3.5)
    x = jnp.linspace(0.0, 1.0, nx)[:, None, None]
    y = jnp.linspace(-1.0, 1.0, cross)[None, :, None]
    z = jnp.linspace(-1.0, 1.0, cross)[None, None, :]
    expected = (
        jnp.sin(2.0 * jnp.pi * x)
        * jnp.cos(0.5 * jnp.pi * y)
        * jnp.cos(0.5 * jnp.pi * z)
    )
    coefficients = fringing_impl._variable_diffusion_coefficients_3d(
        conductivity,
        dx=0.1,
        dy=spacing,
        dz=spacing,
        validated_spacing=True,
        thin_wall_fluid_mask=mask,
    )
    neighbors = fringing_impl._neighbor_fields(
        expected, mode_x="neumann", mode_y="neumann", mode_z="neumann"
    )
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

    volume = jnp.broadcast_to(
        spacing[None, :, None] * spacing[None, None, :], conductivity.shape
    )
    correction = fringing_impl._transverse_modal_correction_3d(
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

    mesh = fringing_impl.Mesh(np.asarray(jax.devices()[:1]), ("x",))
    sharding = fringing_impl.NamedSharding(mesh, fringing_impl.P("x", None, None))
    sharded_correction = fringing_impl._transverse_modal_correction_3d(
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
    sharded_left, sharded_right = (
        jax.device_put(value, sharding) for value in (left, right)
    )
    assert jnp.vdot(sharded_left, sharded_correction(sharded_right)) == pytest.approx(
        jnp.vdot(sharded_correction(sharded_left), sharded_right),
        rel=1.0e-10,
        abs=1.0e-10,
    )
    _, sharded_tangent = jax.jvp(
        sharded_correction, (sharded_right,), (sharded_left,)
    )
    assert sharded_tangent == pytest.approx(
        sharded_correction(sharded_left), rel=1.0e-10, abs=1.0e-10
    )

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

    monkeypatch.setattr(fringing_impl, "pcg_linear_solve", fake_solve)
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
    corrected = _enforce_stationwise_flow_rate_3d(
        u,
        active_mask=jnp.ones_like(u, dtype=bool),
        cell_area=jnp.ones_like(u),
        target_flow_rate=4.0,
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
        quadratic = jnp.broadcast_to(
            (y_grid**2 + z_grid**2)[None, :, :], (3, count, count)
        )
        laplacian = _laplacian_3d(
            quadratic,
            dx=0.25,
            dy=jnp.diff(faces),
            dz=jnp.diff(faces),
            mode_y="neumann",
            mode_z="neumann",
        )
        errors.append(float(jnp.max(jnp.abs(laplacian[:, 2:-2, 2:-2] - 4.0))))
    assert errors[1] < errors[0] / 2.5
    assert errors[1] < 0.03

    constant = jnp.ones((3, 4, 3))
    wall_laplacian = _laplacian_3d(
        constant, dx=0.5, dy=jnp.asarray([0.2, 0.3, 0.4, 0.5]),
        dz=jnp.asarray([0.25, 0.35, 0.4]), mode_y="dirichlet", mode_z="dirichlet"
    )
    assert jnp.allclose(wall_laplacian[:, 1:-1, 1:-1], 0.0)
    for wall in (wall_laplacian[:, 0], wall_laplacian[:, -1],
                 wall_laplacian[:, :, 0], wall_laplacian[:, :, -1]):
        assert float(jnp.max(wall)) < 0.0

    assert _spacing_vector(0.25, 3, dtype=float).tolist() == pytest.approx(
        [0.25, 0.25, 0.25]
    )
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
    manufactured_2d = jnp.cos(jnp.pi * (yy + 1.0) / 2.0) * jnp.cos(
        jnp.pi * (zz + 1.0) / 2.0
    )
    manufactured = jnp.broadcast_to(manufactured_2d[None, :, :], (3, 7, 5))
    conductivity = jnp.broadcast_to((1.0 + 0.2 * yy)[None, :, :], manufactured.shape)
    linear = jnp.broadcast_to((2.0 * yy - 3.0 * zz)[None], manufactured.shape)
    zeros = jnp.zeros_like(linear)
    fx, fy, fz = _conservative_current_fluxes_3d(
        jnp.ones_like(linear), linear, zeros, zeros, zeros,
        dx=0.4, dy=jnp.diff(y_faces), dz=jnp.diff(z_faces)
    )
    assert fy[:, 1:-1] == pytest.approx(-2.0)
    assert fz[:, :, 1:-1] == pytest.approx(3.0)
    current_divergence = (
        (fx[1:] - fx[:-1]) / 0.4
        + (fy[:, 1:] - fy[:, :-1]) / jnp.diff(y_faces)[None, :, None]
        + (fz[:, :, 1:] - fz[:, :, :-1]) / jnp.diff(z_faces)[None, None, :]
    )
    assert current_divergence[:, 1:-1, 1:-1] == pytest.approx(0.0, abs=1.0e-12)
    rhs = _variable_coefficient_residual_3d(
        manufactured,
        jnp.zeros_like(manufactured),
        conductivity,
        dx=0.4,
        dy=jnp.diff(y_faces),
        dz=jnp.diff(z_faces),
    )
    solved, residual, _, _ = _variable_coefficient_poisson_sparse_3d(
        rhs,
        conductivity,
        dx=0.4,
        dy=jnp.diff(y_faces),
        dz=jnp.diff(z_faces),
        iterations=200,
        tolerance=1.0e-10,
    )
    expected = manufactured - jnp.mean(manufactured)
    solved = solved - jnp.mean(solved)
    assert float(jnp.max(jnp.abs(solved - expected))) < 1.0e-9
    assert residual < 1.0e-8

    solved_result = _solvax_pressure_poisson_duct(
        rhs, conductivity, dx=0.4, dy=jnp.diff(y_faces), dz=jnp.diff(z_faces),
        iterations=300, tolerance=1.0e-10)
    (solvax_solved, solvax_residual, solvax_converged, solvax_relative_residual,
     solvax_iterations, solvax_status, solvax_local_residual) = solved_result
    solvax_solved = solvax_solved - jnp.mean(solvax_solved)
    assert bool(solvax_converged)
    assert float(solvax_residual) < 1.0e-8
    assert float(solvax_relative_residual) < 1.0e-8
    assert int(solvax_iterations) > 0
    assert int(solvax_status) == 1
    assert float(solvax_local_residual) < 1.0e-8
    assert float(jnp.max(jnp.abs(solvax_solved - expected))) < 1.0e-8

    warm, warm_residual, warm_iterations, warm_initial = _variable_coefficient_poisson_jacobi_3d(
        rhs, conductivity, dx=0.4, dy=jnp.diff(y_faces), dz=jnp.diff(z_faces),
        iterations=2, tolerance=1.0e-8, initial_field=manufactured + 7.0)
    assert warm_iterations == 1
    assert warm_initial < 1.0e-8
    assert warm_residual < 1.0e-8
    volume_weights = jnp.broadcast_to(jnp.diff(y_faces)[None, :, None]
        * jnp.diff(z_faces)[None, None, :], manufactured.shape)
    expected_weighted_gauge = manufactured - jnp.sum(manufactured * volume_weights) / jnp.sum(volume_weights)
    assert warm == pytest.approx(expected_weighted_gauge, abs=1.0e-8)

    zero_result = _poisson_jacobi_3d(
        jnp.zeros((3, 4, 3)), dx=0.4, dy=jnp.asarray([0.2, 0.3, 0.4, 0.5]),
        dz=jnp.asarray([0.25, 0.35, 0.4]), iterations=4, tolerance=1.0e-12
    )
    assert jnp.allclose(zero_result[0], 0.0)
    assert zero_result[1:] == pytest.approx((0.0, 1, 0.0))


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

    coefficient_value, coefficient_gradient = jax.jit(
        jax.value_and_grad(coefficient_objective)
    )(jnp.asarray(1.0))
    assert coefficient_gradient == pytest.approx(
        -2.0 * coefficient_value, rel=1.0e-6, abs=1.0e-8
    )


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
        jnp.linspace(-0.03, 0.07, 24).reshape(4, 2, 3))
    zero_yz = jnp.zeros((4, 2, 3))
    boundary = (
        jnp.zeros((2, 2, 3)).at[..., 0].set(0.25), velocity[-1],
        zero_yz, zero_yz, zero_yz, zero_yz)
    widths = (jnp.full((4,), dx), dy, dz)
    def dense_scalar(alpha, patches):
        fluxes = tuple(np.asarray(alpha * value) for value in rho_phi)
        weights = tuple(np.asarray(value) for value in
            fringing_impl._limited_linear_vector_face_weights_duct(
                velocity, tuple(alpha * value for value in rho_phi), patches, widths))
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
    captured, actual_linear_solve = {}, fringing_impl.linear_solve

    def capture(matvec, rhs, solver, **kwargs):
        captured["matrix"] = jax.jacfwd(matvec)(jnp.zeros_like(rhs))
        captured["rhs"], captured["zero"] = rhs, matvec(jnp.zeros_like(rhs))
        captured["has_aux"] = kwargs.get("has_aux", False)
        return actual_linear_solve(matvec, rhs, solver, **kwargs)
    monkeypatch.setattr(fringing_impl, "linear_solve", capture)

    def solve(applied_force, flux=rho_phi, patches=boundary, rho=density, prescribed=True):
        return _solvax_implicit_momentum_duct(
            velocity, applied_force, rho, viscosity, flux, patches, dt=dt, dx=dx,
            dy=dy, dz=dz, iterations=240, tolerance=1.0e-10,
            prescribed_inlet=prescribed)

    solved, residual, converged = solve(force)
    matrix, rhs = captured["matrix"], captured["rhs"]
    reference = np.kron(dense_scalar(1.0, boundary), np.eye(3))
    assert matrix == pytest.approx(reference) and solved.sharding == velocity.sharding
    assert solved.reshape(-1) == pytest.approx(np.linalg.solve(reference, rhs), abs=2e-7)
    assert residual < 1e-8 and bool(converged) and solved.shape == (*shape, 3)
    assert captured["has_aux"]
    assert captured["zero"] == pytest.approx(0.0) and not jnp.allclose(matrix, matrix.T)
    volume = dx * dy[None, :, None] * dz[None, None, :]
    expected_rhs = volume[..., None] * (density[..., None] * velocity + dt * force)
    inlet_source = rho_phi[0][0][..., None] + volume[0, ..., None] * (
        2.0 * (density * viscosity)[0, ..., None] / dx**2
    )
    expected_rhs = expected_rhs.at[0].add(dt * inlet_source * boundary[0])
    assert rhs.reshape((*shape, 3)) == pytest.approx(expected_rhs)
    mass = np.repeat(np.asarray(volume * density).reshape(-1), 3)
    cell_volume = np.repeat(np.broadcast_to(np.asarray(volume), shape).reshape(-1), 3)
    old, star = map(np.asarray, (velocity.reshape(-1), solved.reshape(-1)))
    mapped = star + 0.01 * np.asarray(probe := jnp.linspace(
        -0.2, 0.3, velocity.size)).reshape(-1)
    source = (np.asarray(rhs) - mass * old) / (cell_volume * dt)
    transport = (np.asarray(matrix) @ star - mass * star) / (cell_volume * dt)
    split_residual = transport - source - np.repeat(
        np.asarray(density).reshape(-1), 3) * (mapped - star) / dt
    linear_residual = (np.asarray(matrix) @ star - np.asarray(rhs)) / (cell_volume * dt)
    assert split_residual == pytest.approx(
        linear_residual - np.repeat(np.asarray(density).reshape(-1), 3)
        * (mapped - old) / dt, abs=2.0e-12)
    monkeypatch.setattr(fringing_impl, "linear_solve", actual_linear_solve)
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
    zero_flux = tuple(jnp.zeros_like(value) for value in rho_phi)
    diffused, *_ = solve(jnp.zeros_like(force), zero_flux, neutral, jnp.ones(shape), False)
    coefficients = fringing_impl._variable_diffusion_coefficients_3d(viscosity, dx=dx, dy=dy, dz=dz)
    neighbours = fringing_impl._neighbor_fields(diffused, mode_x="neumann", mode_y="neumann", mode_z="neumann")
    wall = jnp.zeros(shape).at[:, 0, :].add(viscosity[:, 0, :] / (0.5 * dy[0] ** 2))
    wall = wall.at[:, -1, :].add(viscosity[:, -1, :] / (0.5 * dy[-1] ** 2))
    wall = wall.at[:, :, 0].add(viscosity[:, :, 0] / (0.5 * dz[0] ** 2))
    wall = wall.at[:, :, -1].add(viscosity[:, :, -1] / (0.5 * dz[-1] ** 2))
    diffusion = sum(c[..., None] * (n - diffused) for c, n in zip(
        coefficients, neighbours, strict=True)) - wall[..., None] * diffused
    assert diffused - dt * diffusion == pytest.approx(velocity, abs=2e-7)

    zero_velocity = jnp.zeros_like(velocity)
    compact_flux = jnp.zeros((3, *shape))
    lorentz = jnp.broadcast_to(jnp.asarray([1.0, 2.0, 3.0]), velocity.shape)
    def defect(scale):
        return fringing_impl._duct_momentum_defect(
            zero_velocity, scale * lorentz, jnp.ones(shape), jnp.zeros(shape),
            compact_flux, jnp.zeros(shape[1:]), jnp.zeros(shape), forcing=0.0,
            force_scale=2.0, dt=dt, dx=dx, dy=dy, dz=dz)
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
        jnp.stack(
            (west_y, jnp.full_like(west_y, x_faces[0]), jnp.zeros_like(west_y)), -1
        ),
        jnp.stack(
            (west_y, jnp.full_like(west_y, x_faces[-1]), jnp.zeros_like(west_y)), -1
        ),
        jnp.stack(
            (jnp.full_like(south_x, y_faces[0]), south_x, jnp.zeros_like(south_x)), -1
        ),
        jnp.stack(
            (jnp.full_like(south_x, y_faces[-1]), south_x, jnp.zeros_like(south_x)), -1
        ),
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
    least_squares = fringing_impl._cell_limited_least_squares_gradient_duct
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
    assert spike_gradient == pytest.approx(
        x_ls_reference(spike, zero_q, 1), abs=1.0e-30
    )

    def convection(scale):
        state = scale * velocity
        flux = tuple(scale * values for values in rho_phi)
        patches = tuple(scale * values for values in boundary_velocity)
        weights = fringing_impl._limited_linear_vector_face_weights_duct(
            state, flux, patches, (dx, dy, dz)
        )
        return fringing_impl._limited_linear_convection_matrix_action_duct(
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
    assert all(
        bool(jnp.all((face_weight >= 0.0) & (face_weight <= 1.0)))
        for face_weight in weights
    )
    assert weights[0][1] == pytest.approx(1.0)
    assert weights[2] == pytest.approx(1.0)

    volume = dx[:, None, None] * dy[None, :, None] * dz[None, None, :]
    boundary_flux = (
        jnp.sum(rho_phi[0][-1, ..., None] * boundary_velocity[1], axis=(0, 1))
        - jnp.sum(rho_phi[0][0, ..., None] * boundary_velocity[0], axis=(0, 1))
        + jnp.sum(rho_phi[1][:, -1, :, None] * boundary_velocity[3], axis=(0, 1))
        - jnp.sum(rho_phi[1][:, 0, :, None] * boundary_velocity[2], axis=(0, 1))
    )
    assert jnp.sum(action * volume[..., None], axis=(0, 1, 2)) == pytest.approx(
        boundary_flux, abs=2.0e-5
    )
    zero_patches = tuple(jnp.zeros_like(value) for value in boundary_velocity)
    assert fringing_impl._limited_linear_convection_matrix_action_duct(
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
    widths = tuple(np.asarray(value) for value in (
        [0.2, 0.3, 0.25, 0.35, 0.4], [0.3, 0.2, 0.4, 0.35], [0.25, 0.45, 0.3]))
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
            gradients.append(np.moveaxis((wm * (values - lo) / dm + wp * (hi - values) / dp) / (wm + wp), 0, axis))
            neighbours.extend((np.moveaxis(lo, 0, axis), np.moveaxis(hi, 0, axis)))
        local = np.stack((field, *neighbours))
        minimum, maximum, limiter = local.min(0), local.max(0), np.ones_like(field)
        for axis, (gradient, width) in enumerate(zip(gradients, widths, strict=True)):
            reshape = [1, 1, 1, 1]
            reshape[axis] = shape[axis]
            for extrapolate in (-0.5 * width.reshape(reshape) * gradient,
                                0.5 * width.reshape(reshape) * gradient):
                delta = np.where(extrapolate > 0, maximum - field, minimum - field)
                limiter = np.minimum(limiter, np.where(abs(extrapolate) > 1e-15,
                    np.minimum(delta / np.where(abs(extrapolate) > 1e-15, extrapolate, 1), 1), 1))
        gradient = np.stack(tuple(limiter * value for value in gradients), axis=-2)
        result, eye = np.zeros_like(field), np.eye(3)
        for axis, width in enumerate(widths):
            def traction(g):
                return 0.7 * (g[..., :, axis] - 2 / 3 * np.trace(g, axis1=-2, axis2=-1)[..., None] * eye[axis])
            moved = np.moveaxis(traction(gradient), axis, 0)
            face_values = []
            for side, index in ((0, 0), (1, -1)):
                patch_gradient = np.take(gradient, index, axis=axis).copy()
                adjacent = np.take(field, index, axis=axis)
                patch_gradient[..., axis, :] = (2 * side - 1) * (boundary[2 * axis + side] - adjacent) / (width[index] / 2)
                face_values.append(traction(patch_gradient))
            weight = (width[1:] / (width[:-1] + width[1:]))[:, None, None, None]
            faces = np.concatenate((face_values[0][None], weight * moved[:-1] + (1 - weight) * moved[1:], face_values[1][None]))
            result += np.moveaxis(np.diff(faces, axis=0) / width[:, None, None, None], 0, axis)
        return result

    def evaluate(scale):
        return fringing_impl._explicit_deviatoric_stress_duct(
            scale * jnp.asarray(velocity), jnp.full(shape, 0.7),
            tuple(scale * jnp.asarray(value) for value in patches), tuple(map(jnp.asarray, widths)))
    expected = reference(velocity, patches)
    assert jax.jit(evaluate)(1.0) == pytest.approx(expected, abs=2.0e-6)
    assert jax.jvp(evaluate, (1.0,), (1.0,))[1] == pytest.approx(expected, abs=3.0e-6)

    uniform = (jnp.full((5,), 0.2), jnp.full((4,), 0.3), jnp.full((3,), 0.4))
    profile = jnp.arange(12.0).reshape(4, 3) / 20.0
    developed = jnp.zeros((*shape, 3)).at[..., 0].set(profile[None])
    developed_patches = (developed[0], developed[-1], *map(jnp.zeros_like, patches[2:]))
    stress = fringing_impl._explicit_deviatoric_stress_duct
    assert stress(developed, jnp.full(shape, 0.7), developed_patches, uniform) == pytest.approx(0.0)
    x = (jnp.arange(5) + 0.5) * uniform[0][0]
    quadratic = jnp.zeros((*shape, 3)).at[..., 0].set(1.2 * x[:, None, None] ** 2)
    quadratic_patches = (jnp.zeros_like(patches[0]), quadratic[-1], *map(jnp.zeros_like, patches[2:]))
    quadratic_stress = stress(quadratic, jnp.full(shape, 0.7), quadratic_patches, uniform)
    assert quadratic_stress[2, 1:-1, 1:-1, 0] == pytest.approx(2 * 0.7 * 1.2 / 3)
    affine = jnp.zeros_like(quadratic).at[..., 0].set(1.2 * x[:, None, None])
    affine_stress = stress(affine, jnp.full(shape, 0.7), (jnp.zeros_like(patches[0]), affine[-1], *map(jnp.zeros_like, patches[2:])), uniform)
    volume = uniform[0][0] * uniform[1][0] * uniform[2][0]
    assert jnp.sum(affine_stress[..., 0]) * volume == pytest.approx(-0.7 * 1.2 * 4 * 0.3 * 3 * 0.4 / 3)


def test_compact_duct_mass_flux_initializer_matches_fv_faces():
    shape = (2, 2, 2)
    compact = jnp.arange(24.0).reshape((3, *shape))
    inlet = jnp.arange(4.0).reshape(shape[1:])
    full = jax.jit(fringing_impl._unpack_duct_mass_flux)(compact, inlet)
    assert jnp.all(full[1][:, 0] == 0.0) and jnp.all(full[2][:, :, 0] == 0.0)
    assert compact.size + inlet.size == 3 * np.prod(shape) + np.prod(shape[1:])
    velocity = jnp.arange(24.0).reshape((*shape, 3)) / 7.0
    density = 1.0 + jnp.arange(8.0).reshape(shape) / 10.0
    inlet_velocity = velocity[0] + jnp.asarray([0.4, -2.0, 3.0])
    dy, dz, dx = jnp.asarray([0.3, 0.7]), jnp.asarray([0.2, 0.8]), 0.4
    def initialize(scale):
        return fringing_impl._initialize_duct_mass_flux(
            scale * velocity, density, scale * inlet_velocity, dx=dx, dy=dy, dz=dz)
    *components, initialized_inlet = jax.jit(initialize)(jnp.asarray(1.0))
    plus = jnp.stack(components)
    momentum = np.asarray(density[..., None] * velocity)
    expected_x = np.concatenate((0.5 * (momentum[:-1, ..., 0] + momentum[1:, ..., 0]),
        momentum[-1:, ..., 0])) * np.outer(dy, dz)
    expected_y = np.concatenate((0.7 * momentum[:, :-1, :, 1] + 0.3 * momentum[:, 1:, :, 1],
        np.zeros_like(momentum[:, :1, :, 1])), axis=1) * dx * dz[None, None]
    expected_z = np.concatenate((0.8 * momentum[:, :, :-1, 2] + 0.2 * momentum[:, :, 1:, 2],
        np.zeros_like(momentum[:, :, :1, 2])), axis=2) * dx * dy[None, :, None]
    assert plus == pytest.approx(np.stack((expected_x, expected_y, expected_z)))
    assert initialized_inlet == pytest.approx(density[0] * inlet_velocity[..., 0] * jnp.outer(dy, dz))
    fx, fy, fz = map(np.asarray, fringing_impl._unpack_duct_mass_flux(plus, initialized_inlet))
    volume = np.broadcast_to(dx * np.outer(dy, dz), shape)
    surface = (abs(fx[:-1]) + abs(fx[1:]) + abs(fy[:, :-1]) + abs(fy[:, 1:]) +
               abs(fz[:, :, :-1]) + abs(fz[:, :, 1:])) / np.asarray(density)
    cell_courant = 0.5 * 0.13 * surface / volume
    courant = jax.jit(lambda p, i: fringing_impl._compact_duct_courant_numbers(
        p, i, density, dt=0.13, dx=dx, dy=dy, dz=dz))(plus, initialized_inlet)
    assert tuple(map(float, courant)) == pytest.approx(
        (np.sum(cell_courant * volume) / np.sum(volume), np.max(cell_courant))
    )
    tangent = jax.jvp(initialize, (1.0,), (1.0,))[1]
    assert jnp.allclose(jnp.stack(tangent[:3]), plus)
    assert jnp.allclose(tangent[-1], initialized_inlet)


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
    assert all(
        jnp.isfinite(field).all()
        for field in (projected_u, projected_v, projected_w, pressure)
    )

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
    expected_pressure = jnp.broadcast_to(
        jnp.asarray([4.0, 3.0, 2.0, 1.0])[:, None, None], shape
    )
    rhs = jnp.broadcast_to(jnp.asarray([-16.0, 0.0, 0.0, -16.0])[:, None, None], shape)
    with pytest.raises(ValueError, match="Unsupported axial pressure mode"):
        _solvax_pressure_poisson_duct(rhs, jnp.ones(shape), **settings,
            axial_pressure_mode="periodic")
    pressure, *_ = _solvax_pressure_poisson_duct(
        rhs,
        jnp.ones(shape),
        **settings,
        axial_pressure_mode=fringing_impl._MIXED_AXIAL_PRESSURE_MODE,
    )
    assert pressure == pytest.approx(expected_pressure, abs=1.0e-7)
    zeros = jnp.zeros(shape)
    projected = _face_flux_pressure_projection_duct(
        zeros, zeros, zeros, jnp.ones(shape), jnp.ones(shape, dtype=bool),
        inlet_flow_rate=0.2, dt=0.1, **settings)
    (_, _, _, projected_pressure, pressure_loss, divergence, flow_error,
        flux_x, flux_y, flux_z, inlet, linear_residual, linear_relative_residual,
        linear_iterations, linear_converged, linear_status) = projected
    plus = jnp.stack((flux_x, flux_y, flux_z))
    assert divergence < 1.0e-8
    assert flow_error < 1.0e-8
    fx, fy, fz = fringing_impl._unpack_duct_mass_flux(plus, inlet)
    assert jnp.sum(inlet) == pytest.approx(0.2) and jnp.sum(plus[0, -1]) == pytest.approx(0.2)
    assert jnp.max(jnp.abs(jnp.diff(fx, axis=0) + jnp.diff(fy, axis=1)
        + jnp.diff(fz, axis=2))) < 1.0e-10
    assert jnp.isfinite(pressure_loss).all()
    assert jnp.isfinite(projected_pressure).all()
    assert jnp.max(jnp.abs(projected_pressure - projected_pressure[:, :1, :1])) < 1.0e-7
    assert linear_residual < 1.0e-8
    assert linear_relative_residual < 1.0e-8
    assert linear_iterations > 0
    assert linear_converged
    assert linear_status > 0
    correction_x, correction_y, correction_z = (
        fringing_impl._duct_pressure_face_corrections(
            projected_pressure, jnp.full(shape, 0.1), dx=dx, dy=widths, dz=widths,
            mixed_axial_pressure=True))
    correction_x_west = jnp.concatenate(
        (jnp.zeros_like(correction_x[:1]), correction_x[:-1]), axis=0)
    pressure_velocity = jnp.stack((
        0.5 * (correction_x_west + correction_x),
        0.5 * (correction_y[:, :-1] + correction_y[:, 1:]),
        0.5 * (correction_z[:, :, :-1] + correction_z[:, :, 1:])), axis=-1)
    projected_velocity = jnp.stack(projected[:3], axis=-1)
    flow_adjustment = projected_velocity - pressure_velocity
    assert flow_adjustment[..., 1:] == pytest.approx(0.0, abs=1.0e-12)
    assert jnp.ptp(flow_adjustment[..., 0], axis=(1, 2)) == pytest.approx(0.0)
    assert jnp.sum(projected[0] * 0.25, axis=(1, 2)) == pytest.approx(0.2)


def test_face_flux_projection_requires_nonempty_rectangular_fluid_mask():
    empty = jnp.zeros((3, 4, 4), dtype=bool)
    with pytest.raises(ValueError, match="nonempty"):
        _rectangular_fluid_bounds(empty)

    nonrectangular = empty.at[:, 1:3, 1:3].set(True).at[:, 1, 1].set(False)
    with pytest.raises(ValueError, match="rectangular"):
        _rectangular_fluid_bounds(nonrectangular)


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
            outer_dirichlet=False,
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
    coefficient = jnp.broadcast_to(
        1.0 + 0.2 * r_centers[None, :, None], manufactured.shape
    )
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
        r_centers[None, :, None]
        * jnp.diff(r_faces)[None, :, None]
        * (2.0 * jnp.pi / 8),
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

    direct_value, direct_gradient = jax.value_and_grad(direct_objective)(
        jnp.asarray(1.0)
    )
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


def test_periodic_line_preconditioner_matches_variable_coefficient_dense_solve():
    weights = jnp.asarray([0.4, 0.7, 0.5, 0.9, 0.6])
    upper_1d = -weights
    lower_1d = -jnp.roll(weights, 1)
    diagonal_1d = 1.0 + weights + jnp.roll(weights, 1)
    diagonal = diagonal_1d[None, None, :]
    lower = lower_1d[None, None, :]
    upper = upper_1d[None, None, :]
    rhs = jnp.asarray([0.3, -0.1, 0.8, 0.2, -0.4])[None, None, :]

    solve = fringing_impl._additive_line_preconditioner_3d(
        diagonal, (), periodic_last_axis=(lower, upper)
    )
    solved = solve(rhs)
    dense = jnp.diag(diagonal_1d)
    dense = dense.at[jnp.arange(1, 5), jnp.arange(4)].set(lower_1d[1:])
    dense = dense.at[jnp.arange(4), jnp.arange(1, 5)].set(upper_1d[:-1])
    dense = dense.at[0, -1].set(lower_1d[0])
    dense = dense.at[-1, 0].set(upper_1d[-1])
    assert solved[0, 0] == pytest.approx(
        jnp.linalg.solve(dense, rhs[0, 0]), abs=1.0e-12
    )

    def objective(scale):
        return jnp.sum(solve(scale * rhs) ** 2)

    value, gradient = jax.value_and_grad(objective)(jnp.asarray(1.0))
    assert gradient == pytest.approx(2.0 * value, rel=1.0e-12)


@pytest.mark.parametrize("steady", [False, True])
def test_pipe_diffusion_reconstructs_manufactured_field(steady):
    r_faces = jnp.asarray([0.0, 0.12, 0.3, 0.55, 0.78, 1.0])
    r_centers = 0.5 * (r_faces[:-1] + r_faces[1:])
    shape = (4, 5, 8)
    manufactured = jnp.arange(np.prod(shape), dtype=float).reshape(shape) / 1000.0
    viscosity = jnp.full(shape, 0.04)
    reaction = jnp.broadcast_to(
        jnp.linspace(0.01, 0.03, shape[0])[:, None, None]
        * (1.0 + r_centers[None, :, None]),
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
    wall_sink = jnp.zeros_like(manufactured).at[:, -1, :].set(
        viscosity[:, -1, :]
        * r_faces[-1]
        / (r_centers[-1] * wall_width * (0.5 * wall_width))
    )
    steady_rhs = -_apply_pipe_diffusion_coefficients_3d(
        manufactured, coefficients
    ) + (wall_sink + reaction) * manufactured
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


def test_pipe_block_jacobi_diffusion_decouples_axial_stations():
    r_faces = jnp.asarray([0.0, 0.2, 0.5, 0.75, 1.0])
    r_centers = 0.5 * (r_faces[:-1] + r_faces[1:])
    dx, viscosity = 0.4, 0.07
    rhs = jnp.zeros((3, 4, 8)).at[1].set(1.0)
    common = dict(
        dt=None,
        dx=dx,
        r_faces=r_faces,
        r_centers=r_centers,
        dtheta=2.0 * jnp.pi / 8,
        iterations=300,
        tolerance=1.0e-10,
    )
    solved, _, converged = _solvax_diffusion_pipe(
        rhs, jnp.full(rhs.shape, viscosity), decouple_axial=True, **common
    )
    local, _, local_converged = _solvax_diffusion_pipe(
        rhs[1:2],
        jnp.full(rhs[1:2].shape, viscosity),
        reaction=jnp.full(rhs[1:2].shape, 2.0 * viscosity / dx**2),
        **common,
    )
    assert bool(converged) and bool(local_converged)
    assert solved[jnp.asarray([0, 2])] == pytest.approx(0.0, abs=1.0e-12)
    assert solved[1] == pytest.approx(local[0], abs=1.0e-8)


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
    value, gradient = jax.jit(
        jax.value_and_grad(lambda a: jnp.sum(operator(a * pressure) ** 2))
    )(jnp.asarray(1.0))
    assert gradient == pytest.approx(2.0 * value, rel=1.0e-12)


@pytest.mark.timeout(300)
@pytest.mark.parametrize(
    "modal_stabilization",
    [False, True],
    ids=("base", "weighted-modal-rhie-chow"),
)
def test_steady_pipe_stokes_projection_closes_compatible_divergence_and_flow(
    modal_stabilization,
):
    nx, nr, ntheta = 5, 3, 8
    r_faces = jnp.asarray([0.0, 0.2, 0.55, 1.0])
    r_centers = 0.5 * (r_faces[:-1] + r_faces[1:])
    wall_width = jnp.diff(r_faces)[-1]
    dtheta = 2.0 * jnp.pi / ntheta
    # This tiny manufactured system reaches the strict gates within 32 steps;
    # a larger budget only lengthens compilation in routine coverage runs.
    inner_iterations = 32
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

    def modal_inverse(rhs):
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
            decouple_axial=True,
        )[0]

    modal_key = ("test_retained_modal_blocks",) if modal_stabilization else None
    unit_response = inverse(jnp.ones(shape))

    def steady_project(**kwargs):
        return _steady_stokes_projection_pipe(
            inverse(u),
            inverse(v),
            inverse(w),
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
            pressure_tolerance=1.0e-9,
            restart=24,
            max_restarts=3,
            apply_momentum_inverse_components=lambda forces: jnp.stack(
                tuple(inverse(force) for force in forces)
            ),
            apply_modal_momentum_inverse=(
                modal_inverse if modal_stabilization else None
            ),
            modal_stabilization=modal_stabilization,
            physical_tolerance=1.0e-7,
            **kwargs,
        )

    steady_result = steady_project(modal_factor_key=modal_key)
    assert steady_result[-3] < 1.0e-7
    assert steady_result[-2] < 1.0e-7
    assert bool(steady_result[-1].converged)

    if modal_stabilization:
        coefficients = _pipe_variable_diffusion_coefficients_3d(
            viscosity,
            dx=0.5,
            r_faces=r_faces,
            r_centers=r_centers,
            dtheta=dtheta,
        )
        wall_sink = (
            jnp.zeros(shape)
            .at[:, -1, :]
            .set(0.07 * r_faces[-1] / (r_centers[-1] * 0.5 * wall_width**2))
        )
        retained = fringing_impl._pipe_retained_modal_factors(
            jnp.ones(shape),
            jnp.maximum(unit_response, jnp.mean(unit_response) * 1.0e-8),
            cell_area,
            coefficients,
            wall_sink,
            dx=0.5,
            r_faces=r_faces,
            r_centers=r_centers,
            dtheta=dtheta,
            modes=(2,),
        )
        probed = fringing_impl._FRINGING_MODAL_FACTOR_CACHE.pop(modal_key)
        trial = jnp.arange(nx * (3 * nr - 1), dtype=float).reshape(nx, 3 * nr - 1)
        assert fringing_impl._solve_pipe_retained_modal_factors(
            retained, trial
        ) == pytest.approx(block_thomas_solve(probed, trial), rel=1.0e-10, abs=1.0e-10)


def test_pipe_face_projection_uses_fluid_wall_face():
    nx, nr, ntheta = 5, 5, 8
    r_faces = jnp.asarray([0.0, 0.12, 0.3, 0.55, 0.78, 1.0, 1.1])
    r_centers = 0.5 * (r_faces[:-1] + r_faces[1:])
    fluid_count = nr
    shape = (nx, nr + 1, ntheta)
    mask = jnp.zeros(shape, dtype=bool).at[:, :fluid_count, :].set(True)
    x = jnp.linspace(0.0, 1.0, nx)[:, None, None]
    r = r_centers[None, :, None]
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, ntheta, endpoint=False)[None, None, :]
    u = jnp.where(mask, 0.7 + 0.1 * jnp.cos(2.0 * jnp.pi * x), 0.0)
    v = jnp.where(mask, 0.1 * r * jnp.cos(theta), 0.0)
    w = jnp.where(mask, -0.1 * r * jnp.sin(theta), 0.0)
    rho = jnp.ones(shape)
    projected = _face_flux_pressure_projection_pipe(
        u,
        v,
        w,
        rho,
        mask,
        dt=0.05,
        dx=0.25,
        r_faces=r_faces,
        r_centers=r_centers,
        dtheta=2.0 * jnp.pi / ntheta,
        iterations=500,
        tolerance=1.0e-10,
        radial_fluid_count=fluid_count,
    )
    projected_u, projected_v, projected_w, pressure, divergence = projected
    assert divergence < 1.0e-8
    assert jnp.isfinite(pressure).all()
    assert jnp.allclose(projected_u[:, fluid_count:, :], 0.0)
    assert jnp.allclose(projected_v[:, fluid_count:, :], 0.0)
    assert jnp.allclose(projected_w[:, fluid_count:, :], 0.0)

    assert _pipe_radial_fluid_count(mask) == fluid_count

    def objective(amplitude):
        projected_fields = _face_flux_pressure_projection_pipe(
            amplitude * u,
            v,
            w,
            rho,
            mask,
            dt=0.05,
            dx=0.25,
            r_faces=r_faces,
            r_centers=r_centers,
            dtheta=2.0 * jnp.pi / ntheta,
            iterations=500,
            tolerance=1.0e-10,
            radial_fluid_count=fluid_count,
        )
        return jnp.mean(projected_fields[0] ** 2) + projected_fields[-1] ** 2

    value, gradient = jax.jit(jax.value_and_grad(objective))(jnp.asarray(1.0))
    step = 1.0e-4
    finite_difference = (objective(1.0 + step) - objective(1.0 - step)) / (2.0 * step)
    assert jnp.isfinite(value)
    assert gradient == pytest.approx(finite_difference, rel=2.0e-5, abs=1.0e-8)


def test_fixed_flow_pipe_projection_closes_flow_and_rejects_invalid_masks():
    nx, nr, ntheta = 4, 4, 8
    r_faces = jnp.asarray([0.0, 0.15, 0.4, 0.7, 1.0])
    r_centers = 0.5 * (r_faces[:-1] + r_faces[1:])
    shape = (nx, nr, ntheta)
    mask = jnp.ones(shape, dtype=bool)
    rho = jnp.ones(shape)
    u = jnp.linspace(0.6, 0.9, nx)[:, None, None] * jnp.ones(shape)
    zeros = jnp.zeros(shape)
    dtheta = 2.0 * jnp.pi / ntheta
    cell_area = jnp.broadcast_to(
        r_centers[None, :, None] * jnp.diff(r_faces)[None, :, None] * dtheta,
        shape,
    )
    response = jnp.full(shape, 0.05)
    target = 2.0
    result = _fixed_flow_face_flux_projection_pipe(
        u,
        zeros,
        zeros,
        rho,
        mask,
        response,
        cell_area,
        target_flow_rate=target,
        base_pressure_loss_gradient=0.0,
        dt=0.05,
        dx=0.3,
        r_faces=r_faces,
        r_centers=r_centers,
        dtheta=dtheta,
        iterations=300,
        tolerance=1.0e-10,
        radial_fluid_count=nr,
    )
    projected_u, _, _, pressure, pressure_loss, divergence, flow_error = result
    flow = jnp.sum(projected_u * cell_area, axis=(1, 2))
    assert flow == pytest.approx(target, abs=1.0e-8)
    assert divergence < 1.0e-8
    assert flow_error < 1.0e-8
    assert jnp.isfinite(pressure_loss).all()

    warm = _fixed_flow_face_flux_projection_pipe(
        u,
        zeros,
        zeros,
        rho,
        mask,
        response,
        cell_area,
        target_flow_rate=target,
        base_pressure_loss_gradient=0.0,
        dt=0.05,
        dx=0.3,
        r_faces=r_faces,
        r_centers=r_centers,
        dtheta=dtheta,
        iterations=300,
        tolerance=1.0e-10,
        radial_fluid_count=nr,
        initial_pressure=pressure,
    )
    assert warm[3] == pytest.approx(pressure, rel=1.0e-8, abs=1.0e-8)

    empty = jnp.zeros(shape, dtype=bool)
    with pytest.raises(ValueError, match="nonempty"):
        _pipe_radial_fluid_count(empty)
    disconnected = mask.at[:, 1, :].set(False)
    with pytest.raises(ValueError, match="contiguous"):
        _pipe_radial_fluid_count(disconnected)
    partial = mask.at[:, -1, 0].set(False)
    with pytest.raises(ValueError, match="full annular"):
        _pipe_radial_fluid_count(partial)


def test_pipe_pressure_fallback_and_gauge_invariant_update():
    rhs = jnp.zeros((3, 4, 8))
    r = jnp.linspace(0.1, 0.4, 4)[None, :, None]
    field, residual, iterations, initial = _pipe_poisson_jacobi_3d(
        rhs,
        dx=0.2,
        dr=0.1,
        dtheta=2.0 * jnp.pi / 8,
        r=r,
        iterations=4,
        tolerance=1.0e-12,
    )
    assert jnp.allclose(field, 0.0)
    assert residual == pytest.approx(0.0)
    assert initial == pytest.approx(0.0)
    assert iterations == 1

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
    expected = jnp.abs(
        jnp.sum(divergence * dy[None, :, None] * dz[None, None, :], axis=(1, 2)) * 0.25
    )

    assert boundary.shape == (shape[0],)
    assert boundary == pytest.approx(expected)
    assert float(jnp.max(boundary)) > 0.0
    fx = jnp.asarray([[[0.0, 1.0]], [[2.0, 3.0]], [[4.0, 5.0]]])
    axial_current = _station_axial_current_from_fluxes(
        fx, jnp.asarray([[2.0, 4.0]])
    )
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
    field = jnp.zeros((4, 2, 2))
    assert _shard_extruded_fields((field,), num_devices=None)[0] is field

    devices = [object(), object()]
    monkeypatch.setattr("lmx.fringing.jax.devices", lambda: devices)
    with pytest.raises(ValueError, match="divisible"):
        _shard_extruded_fields((jnp.zeros((3, 2, 2)),), num_devices=2)

    monkeypatch.setattr("lmx.fringing.Mesh", lambda *args, **kwargs: "mesh")
    monkeypatch.setattr(
        "lmx.fringing.NamedSharding", lambda *args, **kwargs: "sharding"
    )
    monkeypatch.setattr(
        "lmx.fringing.jax.device_put", lambda value, sharding: value + 1
    )
    for count in (1, 2):
        placed = _shard_extruded_fields((field,), num_devices=count)
        assert jnp.all(placed[0] == 1)
    problem = build_square_duct_extruded_problem(nx_stations=4, ny=4, nz=4)
    with pytest.raises(NotImplementedError, match="ALEX B2"):
        solve_extruded_inductionless(problem, num_devices=2)


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

    base_case, _ = build_square_duct_fringing_benchmark(nx_stations=5, ny=8, nz=8)
    shifted = clone_case_with_field(base_case, axis="y", magnitude=3.0, suffix="probe")
    assert shifted.name.endswith("probe")
    assert shifted.magnetic_field.value == (0.0, 3.0, 0.0)
    assert clone_case_with_field(
        base_case, axis="x", magnitude=2.0
    ).magnetic_field.value == (2.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="Unsupported magnetic axis"):
        clone_case_with_field(base_case, axis="bad", magnitude=1.0)


def test_run_fringing_station_sweep_chains_state_and_requires_constant_field(
    monkeypatch: pytest.MonkeyPatch,
):
    base_case, profile = build_square_duct_fringing_benchmark(nx_stations=3, ny=8, nz=8)
    calls: list[tuple[str, object]] = []

    class _State:
        def __init__(self, value: float):
            self.time = 0.0
            self.residual = value

    class _Solution:
        def __init__(self, value: float):
            self.state = _State(value)

    def fake_solver(case, initial_state=None):
        calls.append((case.name, initial_state))
        return _Solution(float(len(calls)))

    monkeypatch.setattr(
        "lmx.fringing.validation_summary",
        lambda solution, case_name, ha=None: {
            "u_max": 0.1,
            "mean_velocity": 0.2,
            "volumetric_flow_rate": 0.3,
            "current_scaled_pressure_proxy": 0.4,
        },
    )

    history = run_fringing_station_sweep(base_case, profile, solver=fake_solver)

    assert len(history) == 3
    assert calls[0][1] is None
    assert calls[1][1] is not None
    assert history[-1]["current_scaled_pressure_proxy"] == pytest.approx(0.4)

    bad_case = replace(
        base_case,
        magnetic_field=replace(
            base_case.magnetic_field,
            kind="analytic",
            fn=lambda y, z: jnp.zeros(y.shape + (3,)),
        ),
    )
    with pytest.raises(ValueError, match="constant-field"):
        run_fringing_station_sweep(bad_case, profile)


def test_run_extruded_inductionless_slice_stacks_station_fields():
    base_case, profile = build_square_duct_fringing_benchmark(nx_stations=4, ny=6, nz=6)
    shape = (base_case.geometry.ny, base_case.geometry.nz)
    y_centers = jnp.linspace(-1.0, 1.0, shape[0])
    z_centers = jnp.linspace(-1.0, 1.0, shape[1])

    class _State:
        def __init__(self, value: float):
            self.u = jnp.full(shape, value)
            self.phi = jnp.full(shape, 0.1 * value)
            self.jy = jnp.zeros(shape)
            self.jz = jnp.zeros(shape)
            self.lorentz_x = jnp.full(shape, 0.01 * value)
            self.time = 0.0
            self.residual = value

    class _Diagnostics:
        def __init__(self, value: float):
            self.volumetric_flow_rate_history = jnp.asarray([value])
            self.mean_velocity_history = jnp.asarray([0.5 * value])
            self.current_scaled_pressure_proxy_history = jnp.asarray([0.25 * value])
            self.charge_balance_residual_history = jnp.asarray([1.0e-6 * value])

    class _Solution:
        def __init__(self, value: float):
            self.mesh = type(
                "Mesh", (), {"y_centers": y_centers, "z_centers": z_centers}
            )()
            self.state = _State(value)
            self.diagnostics = _Diagnostics(value)

    call_count = {"value": 0}

    def fake_solver(case, initial_state=None):
        call_count["value"] += 1
        return _Solution(float(call_count["value"]))

    bundle = run_extruded_inductionless_slice(base_case, profile, solver=fake_solver)

    assert bundle.u.shape == (4, 6, 6)
    assert bundle.phi.shape == (4, 6, 6)
    assert bundle.v.shape == (4, 6, 6)
    assert bundle.w.shape == (4, 6, 6)
    assert bundle.p.shape == (4, 6, 6)
    assert bundle.jx.shape == (4, 6, 6)
    assert bundle.x.shape == (4,)
    assert bundle.y.shape == (6,)
    assert bundle.z.shape == (6,)
    assert bundle.geometry_kind == base_case.geometry.kind
    assert bundle.solver_kind == base_case.solver.kind
    assert jnp.isfinite(bundle.u).all()
    assert jnp.isfinite(bundle.axial_current).all()
    assert jnp.isfinite(bundle.wall_current_leakage).all()
    assert jnp.isfinite(bundle.charge_balance_residual).all()


def test_run_extruded_inductionless_slice_rejects_invalid_inputs():
    base_case, profile = build_square_duct_fringing_benchmark(nx_stations=3, ny=6, nz=6)
    bad_case = replace(
        base_case, geometry=replace(base_case.geometry, kind="pipe_ogrid")
    )
    with pytest.raises(ValueError, match="rectangular and layered ducts"):
        run_extruded_inductionless_slice(bad_case, profile)
    bad_field_case = replace(
        base_case,
        magnetic_field=replace(
            base_case.magnetic_field,
            kind="analytic",
            fn=lambda y, z: jnp.zeros(y.shape + (3,)),
        ),
    )
    with pytest.raises(ValueError, match="constant-field"):
        run_extruded_inductionless_slice(bad_field_case, profile)


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
        (build_bent_pipe_extruded_problem, {"nr": 6, "ntheta": 12}, "bent_pipe"),
    ),
)
def test_extruded_problem_builders_mark_solver_family(builder, kwargs, geometry_kind):
    problem = builder(nx_stations=5, **kwargs)

    assert problem.case.solver.kind == "extruded_inductionless"
    assert problem.case.geometry.kind == geometry_kind
    assert problem.profile.x.shape == (5,)


def test_build_wham_mirror_pipe_extruded_problem_marks_solver_family(tmp_path):
    x = np.linspace(-0.2, 0.2, 5)
    y = np.linspace(-0.1, 0.1, 5)
    z = np.linspace(-0.1, 0.1, 5)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    path = write_tabulated_field_npz(
        tmp_path / "field.npz",
        x=x,
        y=y,
        z=z,
        bx=np.zeros_like(xx),
        by=np.zeros_like(xx),
        bz=1.0 + 0.1 * np.cos(np.pi * xx / max(np.max(np.abs(x)), 1.0e-12)),
    )
    problem = build_wham_mirror_pipe_extruded_problem(
        table_path=str(path),
        radius=0.1,
        nr=4,
        ntheta=12,
        length=0.4,
        nx_stations=5,
    )
    assert problem.case.solver.kind == "extruded_inductionless"
    assert problem.case.geometry.kind == "pipe_ogrid"
    assert problem.case.magnetic_field.kind == "tabulated"
    assert np.allclose(np.asarray(problem.profile.field_scale, dtype=float), 1.0)


def test_cross_section_mesh_supports_pipe_ogrid_geometry():
    problem = build_square_duct_extruded_problem(nx_stations=3, ny=4, nz=4)
    pipe_case = replace(
        problem.case,
        geometry=GeometrySpec(
            kind="pipe_ogrid", width=1.0, height=1.0, radius=0.5, nr=4, ntheta=8
        ),
    )
    mesh = _cross_section_mesh(pipe_case)
    assert mesh.geometry == "pipe_ogrid"


def test_validate_extruded_inductionless_solution_reports_metrics():
    base_case, profile = build_square_duct_fringing_benchmark(nx_stations=4, ny=6, nz=6)
    bundle = run_extruded_inductionless_slice(
        base_case,
        profile,
        solver=lambda case, initial_state=None: type(
            "Solution",
            (),
            {
                "mesh": type(
                    "Mesh",
                    (),
                    {
                        "y_centers": jnp.linspace(-1.0, 1.0, 6),
                        "z_centers": jnp.linspace(-1.0, 1.0, 6),
                    },
                )(),
                "state": type(
                    "State",
                    (),
                    {
                        "u": jnp.ones((6, 6)),
                        "phi": jnp.zeros((6, 6)),
                        "jy": jnp.zeros((6, 6)),
                        "jz": jnp.zeros((6, 6)),
                        "lorentz_x": jnp.zeros((6, 6)),
                        "time": 0.0,
                        "residual": 1.0e-6,
                    },
                )(),
                "diagnostics": type(
                    "Diagnostics",
                    (),
                    {
                        "volumetric_flow_rate_history": jnp.asarray([1.0]),
                        "mean_velocity_history": jnp.asarray([0.5]),
                        "current_scaled_pressure_proxy_history": jnp.asarray([0.2]),
                        "charge_balance_residual_history": jnp.asarray([1.0e-7]),
                    },
                )(),
            },
        )(),
    )

    report = validate_extruded_inductionless_solution(bundle)
    assert report.station_count == 4
    assert report.max_charge_balance_residual >= 0.0
    assert report.axial_current_span >= 0.0
    assert report.peak_velocity_span >= 0.0
    assert report.pressure_span_range >= 0.0
    assert report.max_wall_current_leakage >= 0.0
    assert report.net_boundary_current_residual >= 0.0
    assert jnp.isfinite(report.field_mean_velocity_correlation)


def test_solve_extruded_inductionless_wraps_history_bundle_and_validation(
    monkeypatch: pytest.MonkeyPatch,
):
    problem = build_square_duct_extruded_problem(nx_stations=3, ny=6, nz=6)
    fake_bundle = type(
        "Bundle",
        (),
        {
            "x": jnp.asarray([0.0, 0.5, 1.0]),
            "field_scale": jnp.asarray([0.0, 1.0, 0.0]),
            "mean_velocity": jnp.asarray([0.1, 0.3, 0.12]),
            "volumetric_flow_rate": jnp.asarray([0.2, 0.4, 0.25]),
            "axial_current": jnp.asarray([0.01, 0.02, 0.015]),
            "wall_current_leakage": jnp.asarray([1.0e-6, 2.0e-6, 1.5e-6]),
            "boundary_current_residual": jnp.asarray([1.0e-7, 2.0e-7, 1.5e-7]),
            "residual": jnp.asarray([1.0e-4, 1.0e-5, 1.0e-5]),
            "charge_balance_residual": jnp.asarray([1.0e-8, 2.0e-8, 1.5e-8]),
            "y": jnp.asarray([0.0]),
            "z": jnp.asarray([0.0]),
            "u": jnp.ones((3, 1, 1)),
            "v": jnp.zeros((3, 1, 1)),
            "w": jnp.zeros((3, 1, 1)),
            "p": jnp.zeros((3, 1, 1)),
            "phi": jnp.zeros((3, 1, 1)),
            "jx": jnp.zeros((3, 1, 1)),
            "jy": jnp.zeros((3, 1, 1)),
            "jz": jnp.zeros((3, 1, 1)),
            "lorentz_x": jnp.zeros((3, 1, 1)),
            "lorentz_y": jnp.zeros((3, 1, 1)),
            "lorentz_z": jnp.zeros((3, 1, 1)),
            "current_scaled_pressure_proxy": jnp.asarray([0.1, 0.15, 0.1]),
            "geometry_kind": "rect_duct",
            "solver_kind": "extruded_inductionless",
        },
    )()
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


def test_solve_extruded_inductionless_projection_returns_finite_rectangular_bundle():
    problem = build_square_duct_extruded_problem(ha_peak=8.0, nx_stations=4, ny=4, nz=4)
    field = make_maxwell_consistent_fringe_field(
        peak_field=8.0, center=3.0, transition_width=0.5
    )
    problem = replace(problem, profile=replace(problem.profile, volume_field=field))

    solution = solve_extruded_inductionless(problem)

    assert solution.bundle.u.shape == (4, 4, 4)
    assert solution.bundle.p.shape == (4, 4, 4)
    assert solution.bundle.v.shape == (4, 4, 4)
    assert solution.bundle.w.shape == (4, 4, 4)
    assert solution.bundle.jx.shape == (4, 4, 4)
    assert jnp.isfinite(solution.bundle.u).all()
    assert jnp.isfinite(solution.bundle.p).all()
    assert jnp.isfinite(solution.bundle.axial_current).all()
    assert jnp.isfinite(solution.bundle.wall_current_leakage).all()
    assert solution.validation.max_wall_current_leakage >= 0.0
    assert solution.validation.net_boundary_current_residual >= 0.0
    assert solution.validation.station_count == 4

    with pytest.raises(ValueError, match="three-component"):
        _sample_volume_field(
            lambda x, y, z: jnp.zeros_like(x),
            jnp.zeros((2, 2, 2)),
            jnp.zeros((2, 2, 2)),
            jnp.zeros((2, 2, 2)),
        )


def test_rectangular_projection_uses_sparse_electric_solve(
    monkeypatch: pytest.MonkeyPatch,
):
    problem = build_square_duct_extruded_problem(ha_peak=8.0, nx_stations=3, ny=4, nz=4)
    sparse_calls = {"count": 0}
    jacobi_calls = {"count": 0}

    def wrapped_sparse(*args, **kwargs):
        sparse_calls["count"] += 1
        return _variable_coefficient_poisson_sparse_3d(*args, **kwargs)

    def wrapped_jacobi(*args, **kwargs):
        jacobi_calls["count"] += 1
        return _variable_coefficient_poisson_jacobi_3d(*args, **kwargs)

    monkeypatch.setattr(
        "lmx.fringing._variable_coefficient_poisson_sparse_3d", wrapped_sparse
    )
    monkeypatch.setattr(
        "lmx.fringing._variable_coefficient_poisson_jacobi_3d", wrapped_jacobi
    )

    solve_extruded_inductionless(problem)

    assert sparse_calls["count"] > 0
    assert jacobi_calls["count"] == 0


def test_projection_solver_can_break_early_with_loose_tolerance():
    problem = build_square_duct_extruded_problem(ha_peak=4.0, nx_stations=3, ny=4, nz=4)
    loose_problem = replace(
        problem,
        case=replace(
            problem.case, solver=replace(problem.case.solver, coupling_tolerance=10.0)
        ),
    )
    solution = solve_extruded_inductionless(loose_problem)
    assert solution.validation.max_residual <= 10.0


def test_solve_extruded_inductionless_projection_returns_finite_layered_bundle():
    problem = build_layered_duct_extruded_problem(
        ha_peak=8.0,
        nx_stations=4,
        ny=4,
        nz=4,
        wall_cells=1,
        insulator_cells=1,
    )

    solution = solve_extruded_inductionless(problem)

    assert solution.bundle.u.shape[0] == 4
    assert solution.bundle.geometry_kind == "layered_duct"
    assert jnp.isfinite(solution.bundle.u).all()
    assert jnp.isfinite(solution.bundle.phi).all()
    assert solution.validation.max_charge_balance_residual < 1.0e-4
    assert solution.validation.net_boundary_current_residual == pytest.approx(0.0)


def test_layered_projection_keeps_throughput_span_bounded_on_heavier_case():
    problem = build_layered_duct_extruded_problem(
        ha_peak=20.0,
        nx_stations=5,
        ny=6,
        nz=6,
        wall_cells=1,
        insulator_cells=1,
    )
    problem = replace(
        problem,
        case=replace(
            problem.case,
            time_stepper=replace(
                problem.case.time_stepper, max_steps=12, potential_iterations=48
            ),
            solver=replace(problem.case.solver, coupling_iterations=8),
        ),
    )

    solution = solve_extruded_inductionless(problem)

    assert solution.validation.volumetric_flow_rate_span < 5.0e-3
    # Mean throughput is constrained nearly stationwise; its correlation with
    # field strength is not a physical braking metric. Peak/profile response is
    # covered separately, while this regression closes residual and symmetry.
    assert abs(solution.validation.field_mean_velocity_correlation) <= 1.0
    assert solution.validation.max_residual < 1.0e-3
    assert solution.validation.max_charge_balance_residual < 1.0e-4
    assert solution.validation.axial_current_mirror_residual < 1.0e-3
    assert solution.validation.pressure_span_mirror_residual < 1.0e-3
    assert abs(solution.validation.center_axial_current) < 1.0e-4


def test_solve_extruded_inductionless_projection_returns_finite_pipe_bundle():
    problem = build_pipe_ogrid_extruded_problem(
        ha_peak=6.0, nx_stations=4, nr=4, ntheta=8
    )
    field = make_maxwell_consistent_fringe_field(
        peak_field=6.0, center=3.0, transition_width=0.5
    )
    problem = replace(problem, profile=replace(problem.profile, volume_field=field))

    solution = solve_extruded_inductionless(problem)

    assert solution.bundle.geometry_kind == "pipe_ogrid"
    assert solution.bundle.u.shape == (4, 4, 8)
    assert jnp.isfinite(solution.bundle.u).all()
    assert jnp.isfinite(solution.bundle.axial_current).all()
    assert jnp.isfinite(solution.bundle.wall_current_leakage).all()
    assert float(jnp.max(jnp.abs(solution.bundle.u[:, -1, :]))) > 0.0
    assert solution.validation.max_charge_balance_residual < 0.5
    assert solution.validation.net_boundary_current_residual == pytest.approx(0.0)


def test_pipe_projection_supports_explicit_conducting_annulus_and_fixed_flow():
    problem = build_pipe_ogrid_extruded_problem(
        ha_peak=2.0, nx_stations=3, nr=4, ntheta=8
    )
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

    solution = solve_extruded_inductionless(problem)

    assert solution.bundle.u.shape == (3, 6, 8)
    assert jnp.allclose(solution.bundle.u[:, 4:, :], 0.0)
    assert jnp.allclose(solution.bundle.v[:, 4:, :], 0.0)
    assert jnp.allclose(solution.bundle.w[:, 4:, :], 0.0)
    assert solution.bundle.mean_velocity.tolist() == pytest.approx(
        [0.5, 0.5, 0.5], rel=1.0e-6
    )
    assert solution.bundle.axial_pressure_loss_gradient.shape == (3,)
    assert jnp.isfinite(solution.bundle.axial_pressure_loss_gradient).all()
    assert solution.bundle.transverse_pressure_difference.tolist() == pytest.approx(
        [0.0, 0.0, 0.0]
    )
    assert "axial_pressure_loss_gradient" in solution.station_history[0]
    assert jnp.isfinite(solution.bundle.phi).all()


def test_pipe_sparse_potential_cancels_conservative_emf_divergence():
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
    phi, residual, _, _ = _pipe_poisson_sparse_3d(
        -emf_rhs,
        sigma,
        dx=dx,
        r_faces=r_faces,
        r_centers=r_centers,
        dtheta=float(dtheta),
        iterations=40,
        tolerance=1.0e-12,
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

    assert residual < 1.0e-10
    assert float(jnp.max(jnp.abs(div_j))) < 1.0e-10


def test_wham_mirror_pipe_baseline_reports_finite_metrics(tmp_path):
    x = np.linspace(-0.2, 0.2, 7)
    y = np.linspace(-0.12, 0.12, 7)
    z = np.linspace(-0.12, 0.12, 7)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    bz = 1.0 + 0.3 * np.exp(-((xx / 0.12) ** 2))
    path = write_tabulated_field_npz(
        tmp_path / "wham_small.npz",
        x=x,
        y=y,
        z=z,
        bx=np.zeros_like(xx),
        by=np.zeros_like(xx),
        bz=bz,
    )
    problem = build_wham_mirror_pipe_extruded_problem(
        table_path=str(path),
        radius=0.12,
        nr=4,
        ntheta=12,
        length=0.4,
        nx_stations=7,
    )
    problem = replace(
        problem,
        profile=replace(problem.profile, x=jnp.asarray(x, dtype=float)),
        case=replace(
            problem.case,
            time_stepper=replace(
                problem.case.time_stepper, max_steps=4, potential_iterations=12
            ),
            solver=replace(problem.case.solver, coupling_iterations=4),
        ),
    )
    solution = solve_extruded_inductionless(problem)
    metrics = validate_wham_mirror_pipe_baseline(solution)
    assert np.isfinite(metrics["pressure_drop_proxy"])
    assert np.isfinite(metrics["field_velocity_correlation"])
    assert np.isfinite(metrics["max_charge_balance_residual"])


def test_solve_extruded_inductionless_projection_returns_finite_bent_pipe_bundle():
    bent_problem = build_bent_pipe_extruded_problem(
        ha_peak=6.0,
        bend_radius=4.0,
        bend_angle=1.0,
        nx_stations=4,
        nr=4,
        ntheta=8,
    )
    straight_problem = build_pipe_ogrid_extruded_problem(
        ha_peak=6.0,
        radius=float(bent_problem.case.geometry.radius),
        length=float(bent_problem.case.geometry.length),
        nx_stations=4,
        nr=4,
        ntheta=8,
    )
    straight_problem = replace(straight_problem, profile=bent_problem.profile)

    bent_solution = solve_extruded_inductionless(bent_problem)
    straight_solution = solve_extruded_inductionless(straight_problem)
    validation = validate_bent_pipe_low_de_baseline(bent_solution, straight_solution)

    assert bent_solution.bundle.geometry_kind == "bent_pipe"
    assert jnp.isfinite(bent_solution.bundle.u).all()
    assert validation["dean_number"] >= 0.0
    assert validation["dean_vortex_observables_available"] is True
    assert validation["secondary_flow_rms_ratio"] >= 0.0
    assert validation["secondary_flow_peak_ratio"] >= 0.0
    assert np.isfinite(validation["normalized_velocity_centroid_shift"])
    assert np.isfinite(validation["inner_outer_velocity_ratio"])
    assert validation["research_grade_dean_validation_pass"] is False
    assert isinstance(validation["research_grade_charge_balance_pass"], bool)
    assert (
        validation["research_grade_charge_balance_tolerance"]
        < validation["bounded_charge_balance_tolerance"]
    )
    assert validation["cross_section_l2_error"] <= 0.2
    assert isinstance(validation["validation_pass"], bool)


def test_solve_extruded_inductionless_supports_analytic_variable_field():
    problem = build_variable_field_duct_extruded_problem(nx_stations=7, ny=10, nz=10)
    solution = solve_extruded_inductionless(problem)
    validation = validate_variable_field_extruded_solution(
        solution, field_ny=41, field_nz=41
    )

    assert solution.bundle.geometry_kind == "rect_duct"
    assert jnp.all(jnp.isfinite(solution.bundle.u))
    assert validation["mean_velocity_change"] > 0.0
    assert validation["current_proxy_change"] > 0.0
    assert isinstance(validation["validation_pass"], bool)


def test_solve_extruded_inductionless_supports_layered_analytic_variable_field():
    problem = build_variable_field_layered_extruded_problem(nx_stations=7, ny=10, nz=10)
    solution = solve_extruded_inductionless(problem)
    validation = validate_variable_field_extruded_solution(
        solution, field_ny=41, field_nz=41
    )

    assert solution.bundle.geometry_kind == "layered_duct"
    assert jnp.all(jnp.isfinite(solution.bundle.u))
    assert validation["mean_velocity_change"] > 0.0
    assert isinstance(validation["validation_pass"], bool)


def test_solve_extruded_inductionless_supports_tabulated_variable_field(tmp_path):
    field_fn = make_divergence_free_cross_section_field(
        width=2.4, height=1.6, base_bz=12.0, perturbation=0.12
    )
    y, z, field = sample_cross_section_field(
        field_fn, width=2.4, height=1.6, ny=41, nz=41
    )
    path = write_tabulated_field_npz(
        tmp_path / "field.npz",
        y=y,
        z=z,
        bx=field[..., 0],
        by=field[..., 1],
        bz=field[..., 2],
    )
    problem = build_square_duct_extruded_problem(
        nx_stations=7, ny=10, nz=10, width=2.4, height=1.6, ha_peak=12.0
    )
    problem = replace(
        problem,
        case=replace(
            problem.case,
            magnetic_field=MagneticFieldSpec(kind="tabulated", table_path=str(path)),
        ),
    )
    solution = solve_extruded_inductionless(problem)
    validation = validate_variable_field_extruded_solution(
        solution, field_ny=41, field_nz=41
    )

    assert solution.bundle.geometry_kind == "rect_duct"
    assert validation["rms_divergence"] >= 0.0
    assert isinstance(validation["validation_pass"], bool)


def test_solve_extruded_inductionless_supports_variable_field_pipe_and_bent_pipe():
    straight_problem = build_variable_field_pipe_ogrid_extruded_problem(
        nx_stations=5, nr=4, ntheta=8
    )
    bent_problem = build_variable_field_bent_pipe_extruded_problem(
        nx_stations=5, nr=4, ntheta=8
    )
    straight_solution = solve_extruded_inductionless(straight_problem)
    bent_solution = solve_extruded_inductionless(bent_problem)

    pipe_validation = validate_variable_field_pipe_solution(
        straight_solution, field_ny=41, field_nz=41
    )
    bent_field_validation = validate_variable_field_pipe_solution(
        bent_solution, field_ny=41, field_nz=41
    )
    bent_low_de_validation = validate_bent_pipe_low_de_baseline(
        bent_solution, straight_solution
    )

    assert straight_solution.bundle.geometry_kind == "pipe_ogrid"
    assert bent_solution.bundle.geometry_kind == "bent_pipe"
    assert pipe_validation["current_proxy_change"] > 0.0
    assert bent_field_validation["current_proxy_change"] > 0.0
    assert isinstance(bent_low_de_validation["validation_pass"], bool)


def test_magnetic_obstacle_baseline_reports_velocity_deficit():
    problem = build_magnetic_obstacle_rect_extruded_problem(nx_stations=9, ny=12, nz=12)
    solution = solve_extruded_inductionless(problem)
    validation = validate_magnetic_obstacle_baseline(solution, field_ny=41, field_nz=41)

    assert solution.bundle.geometry_kind == "rect_duct"
    assert validation["obstacle_velocity_deficit"] > 0.0
    assert validation["current_proxy_peak"] > 0.0
    assert validation["divergence_to_field_ratio"] >= 0.0
    assert validation["field_quality_pass"] in {True, False}
    assert validation["reference_kind"] == "none"
    assert validation["external_reference_available"] is False
    assert validation["research_grade_validation_pass"] is False
    assert isinstance(validation["validation_pass"], bool)


def test_magnetic_obstacle_benchmark_reports_normalized_response():
    problem = build_magnetic_obstacle_rect_extruded_problem(
        base_bz=60.0, nx_stations=9, ny=12, nz=12, forcing=2.0
    )
    problem = replace(
        problem,
        case=replace(
            problem.case,
            time_stepper=replace(
                problem.case.time_stepper, max_steps=12, potential_iterations=24
            ),
            solver=replace(problem.case.solver, coupling_iterations=6),
        ),
    )
    solution = solve_extruded_inductionless(problem)
    reference_problem = replace(
        problem,
        profile=replace(
            problem.profile, field_scale=jnp.zeros_like(problem.profile.field_scale)
        ),
    )
    reference_solution = solve_extruded_inductionless(reference_problem)
    validation = validate_magnetic_obstacle_benchmark(
        solution, reference_solution, field_ny=41, field_nz=41
    )

    assert solution.bundle.geometry_kind == "rect_duct"
    assert validation["current_proxy_peak"] > 0.0
    assert validation["peak_pressure_excess"] >= 0.0
    assert validation["peak_velocity_deficit_ratio"] >= 0.0
    assert validation["integrated_velocity_deficit_ratio"] >= 0.0
    assert validation["peak_centerline_deficit_ratio"] >= 0.0
    assert validation["peak_centerline_station_deficit_ratio"] >= 0.0
    assert validation["recovery_station"] >= float(solution.bundle.x[0])
    assert validation["y_l2_distortion"] > 0.0
    assert validation["z_l2_distortion"] > 0.0
    assert validation["y_peak_cut_abs_error"] >= 0.0
    assert validation["z_peak_cut_abs_error"] >= 0.0
    assert validation["peak_crosscut_distortion"] == pytest.approx(
        max(validation["y_l2_distortion"], validation["z_l2_distortion"])
    )
    assert validation["reference_kind"] == "matched_no_field_lmx"
    assert validation["external_reference_available"] is False
    assert validation["internal_response_pass"] == validation["benchmark_pass"]
    assert validation["research_grade_validation_pass"] is False
    assert isinstance(validation["benchmark_pass"], bool)


def test_magnetic_obstacle_literature_slice_reports_recovery_metrics():
    problem = build_magnetic_obstacle_rect_extruded_problem(
        base_bz=60.0, nx_stations=9, ny=12, nz=12, forcing=2.0
    )
    problem = replace(
        problem,
        case=replace(
            problem.case,
            time_stepper=replace(
                problem.case.time_stepper, max_steps=12, potential_iterations=24
            ),
            solver=replace(problem.case.solver, coupling_iterations=6),
        ),
    )
    solution = solve_extruded_inductionless(problem)
    reference_problem = replace(
        problem,
        profile=replace(
            problem.profile, field_scale=jnp.zeros_like(problem.profile.field_scale)
        ),
    )
    reference_solution = solve_extruded_inductionless(reference_problem)
    validation = validate_magnetic_obstacle_literature_slice(
        solution, reference_solution, field_ny=41, field_nz=41
    )
    references = magnetic_obstacle_literature_reference_cases()
    readiness = validate_magnetic_obstacle_external_readiness(
        solution, field_ny=41, field_nz=41
    )

    assert validation["peak_station"] >= float(solution.bundle.x[0])
    assert validation["recovery_distance"] >= 0.0
    assert 0.0 <= validation["normalized_recovery_distance"] <= 1.0
    assert validation["literature_shape_gate"] in {True, False}
    assert validation["literature_status"] == "internal_lmx_response_only"
    assert validation["external_reference_available"] is False
    assert validation["research_grade_validation_pass"] is False
    assert validation["literature_pass"] is False
    assert "votyakov_zienicke_kolesnikov_jfm" in references
    assert readiness["reference_case"] == "votyakov_zienicke_kolesnikov_jfm"
    assert "centerline_velocity_deficit_ratio" in readiness["observables"]
    assert "minimum_centerline_velocity_ratio" in readiness["observables"]
    assert "normalized_recovery_distance" in readiness["observables"]
    assert readiness["external_reference_available"] is False
    assert readiness["research_grade_validation_pass"] is False


def test_poisson_helpers_can_stop_early():
    rhs = jnp.zeros((2, 2, 2))
    field, residual, iterations, initial = _poisson_jacobi_3d(
        rhs, dx=1.0, dy=1.0, dz=1.0, iterations=4, tolerance=1.0
    )
    assert iterations == 1
    assert residual <= initial
    conductivity = jnp.ones((2, 2, 2))
    field_var, residual_var, iterations_var, initial_var = (
        _variable_coefficient_poisson_jacobi_3d(
            rhs,
            conductivity,
            dx=1.0,
            dy=1.0,
            dz=1.0,
            iterations=4,
            tolerance=1.0,
        )
    )
    assert iterations_var == 1
    assert residual_var <= initial_var
    field_sparse, residual_sparse, iterations_sparse, initial_sparse = (
        _variable_coefficient_poisson_sparse_3d(
            rhs,
            conductivity,
            dx=1.0,
            dy=1.0,
            dz=1.0,
            iterations=4,
            tolerance=1.0,
        )
    )
    assert iterations_sparse >= 1
    assert residual_sparse <= initial_sparse
    assert jnp.isfinite(field).all()
    assert jnp.isfinite(field_var).all()
    assert jnp.isfinite(field_sparse).all()


def test_solve_extruded_inductionless_uses_projection_for_pipe_geometry(
    monkeypatch: pytest.MonkeyPatch,
):
    problem = build_square_duct_extruded_problem(nx_stations=3, ny=4, nz=4)
    pipe_case = replace(
        problem.case,
        geometry=GeometrySpec(
            kind="pipe_ogrid", width=1.0, height=1.0, radius=0.5, nr=4, ntheta=8
        ),
    )
    pipe_problem = replace(problem, case=pipe_case)
    monkeypatch.setattr(
        "lmx.fringing._solve_extruded_projection",
        lambda problem, **kwargs: type(
            "Bundle",
            (),
            {
                "x": jnp.asarray([0.0]),
                "field_scale": jnp.asarray([1.0]),
                "mean_velocity": jnp.asarray([0.1]),
                "volumetric_flow_rate": jnp.asarray([0.2]),
                "axial_current": jnp.asarray([0.0]),
                "wall_current_leakage": jnp.asarray([0.0]),
                "boundary_current_residual": jnp.asarray([0.0]),
                "residual": jnp.asarray([1.0e-4]),
                "charge_balance_residual": jnp.asarray([1.0e-6]),
                "y": jnp.asarray([0.0]),
                "z": jnp.asarray([0.0]),
                "u": jnp.zeros((1, 1, 1)),
                "v": jnp.zeros((1, 1, 1)),
                "w": jnp.zeros((1, 1, 1)),
                "p": jnp.zeros((1, 1, 1)),
                "phi": jnp.zeros((1, 1, 1)),
                "jx": jnp.zeros((1, 1, 1)),
                "jy": jnp.zeros((1, 1, 1)),
                "jz": jnp.zeros((1, 1, 1)),
                "lorentz_x": jnp.zeros((1, 1, 1)),
                "lorentz_y": jnp.zeros((1, 1, 1)),
                "lorentz_z": jnp.zeros((1, 1, 1)),
                "current_scaled_pressure_proxy": jnp.asarray([0.0]),
                "geometry_kind": "pipe_ogrid",
                "solver_kind": "extruded_inductionless",
            },
        )(),
    )
    solution = solve_extruded_inductionless(pipe_problem)
    assert solution.validation.station_count == 1


def test_solve_extruded_inductionless_projection_accepts_matching_initial_bundle():
    problem = build_square_duct_extruded_problem(ha_peak=5.0, nx_stations=3, ny=4, nz=4)
    first = solve_extruded_inductionless(problem)

    resumed = solve_extruded_inductionless(problem, initial_bundle=first.bundle)

    assert resumed.bundle.u.shape == first.bundle.u.shape
    assert jnp.isfinite(resumed.bundle.u).all()
    assert resumed.validation.station_count == first.validation.station_count

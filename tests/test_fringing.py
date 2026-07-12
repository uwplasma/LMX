import jax
import jax.numpy as jnp
import numpy as np
import pytest
from dataclasses import replace

import lmx._fringing as fringing_impl

from lmx.field_models import (
    make_divergence_free_cross_section_field,
    sample_cross_section_field,
    write_tabulated_field_npz,
)
from lmx.fringing import (
    _apply_fixed_flow_pressure_constraint,
    _apply_pipe_diffusion_coefficients_3d,
    _cross_duct_pressure_difference,
    _distance_weighted_harmonic_mean,
    _enforce_stationwise_flow_rate_3d,
    _gradient_3d,
    _laplacian_3d,
    _masked_laplacian_duct,
    _net_boundary_current_residual,
    _normalized_pressure_observable_update,
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
    _pipe_gradient_3d,
    _pipe_laplacian_3d,
    _pipe_poisson_sparse_3d,
    _pipe_radial_fluid_count,
    _pipe_variable_diffusion_coefficients_3d,
    _rectangular_fluid_bounds,
    _solvax_pressure_poisson_duct,
    _shard_extruded_fields,
    _solvax_pressure_poisson_pipe,
    _solvax_implicit_diffusion_duct,
    _solvax_implicit_diffusion_pipe,
    _spacing_vector,
    _station_axial_current_from_fluxes,
    _poisson_jacobi_3d,
    _variable_coefficient_poisson_jacobi_3d,
    _variable_coefficient_poisson_sparse_3d,
    _variable_coefficient_residual_3d,
    _conservative_current_fluxes_3d,
    _face_flux_pressure_projection_duct,
    _fixed_flow_face_flux_projection_duct,
    _face_flux_pressure_projection_pipe,
    _fixed_flow_face_flux_projection_pipe,
    _masked_laplacian_pipe,
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


def test_b2_jit_cache_reuses_the_first_compiled_kernel():
    fringing_impl._B2_JIT_CACHE.clear()
    first = object()
    key = ("operator", "configuration")

    assert fringing_impl._reuse_b2_jit(key, first) is first
    assert fringing_impl._reuse_b2_jit(key, object()) is first


def test_distance_weighted_harmonic_mean_preserves_series_resistance():
    value = _distance_weighted_harmonic_mean(
        jnp.asarray([10.0]),
        jnp.asarray([1.0]),
        jnp.asarray([0.1]),
        jnp.asarray([0.9]),
    )
    assert value == pytest.approx([1.0 / (0.1 / 10.0 + 0.9 / 1.0)])


def test_fixed_flow_pressure_constraint_recovers_target_and_multiplier():
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


def test_stationwise_flow_correction_accepts_explicit_target():
    u = jnp.asarray([[[1.0, 1.0]], [[3.0, 3.0]]])
    corrected = _enforce_stationwise_flow_rate_3d(
        u,
        active_mask=jnp.ones_like(u, dtype=bool),
        cell_area=jnp.ones_like(u),
        target_flow_rate=4.0,
        relaxation=1.0,
    )
    assert jnp.sum(corrected, axis=(1, 2)).tolist() == pytest.approx([4.0, 4.0])


def test_nonuniform_gradient_and_laplacian_use_physical_cell_spacing():
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


def test_nonuniform_spacing_contract_rejects_invalid_metrics_and_handles_thin_axes():
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


def test_nonuniform_dirichlet_laplacian_uses_half_cell_wall_distance():
    field = jnp.ones((3, 4, 3))
    laplacian = _laplacian_3d(
        field,
        dx=0.5,
        dy=jnp.asarray([0.2, 0.3, 0.4, 0.5]),
        dz=jnp.asarray([0.25, 0.35, 0.4]),
        mode_y="dirichlet",
        mode_z="dirichlet",
    )
    assert jnp.all(laplacian[:, 1:-1, 1:-1] == pytest.approx(0.0))
    assert float(jnp.max(laplacian[:, 0, :])) < 0.0
    assert float(jnp.max(laplacian[:, -1, :])) < 0.0
    assert float(jnp.max(laplacian[:, :, 0])) < 0.0
    assert float(jnp.max(laplacian[:, :, -1])) < 0.0


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

    (
        solvax_solved,
        solvax_residual,
        solvax_converged,
        solvax_relative_residual,
        solvax_iterations,
        solvax_status,
        solvax_local_residual,
    ) = _solvax_pressure_poisson_duct(
        rhs,
        conductivity,
        dx=0.4,
        dy=jnp.diff(y_faces),
        dz=jnp.diff(z_faces),
        iterations=300,
        tolerance=1.0e-10,
    )
    solvax_solved = solvax_solved - jnp.mean(solvax_solved)
    assert bool(solvax_converged)
    assert float(solvax_residual) < 1.0e-8
    assert float(solvax_relative_residual) < 1.0e-8
    assert int(solvax_iterations) > 0
    assert int(solvax_status) == 1
    assert float(solvax_local_residual) < 1.0e-8
    assert float(jnp.max(jnp.abs(solvax_solved - expected))) < 1.0e-8

    warm, warm_residual, warm_iterations, warm_initial = (
        _variable_coefficient_poisson_jacobi_3d(
            rhs,
            conductivity,
            dx=0.4,
            dy=jnp.diff(y_faces),
            dz=jnp.diff(z_faces),
            iterations=2,
            tolerance=1.0e-8,
            initial_field=manufactured + 7.0,
        )
    )
    assert warm_iterations == 1
    assert warm_initial < 1.0e-8
    assert warm_residual < 1.0e-8
    volume_weights = jnp.broadcast_to(
        jnp.diff(y_faces)[None, :, None] * jnp.diff(z_faces)[None, None, :],
        manufactured.shape,
    )
    expected_weighted_gauge = manufactured - jnp.sum(
        manufactured * volume_weights
    ) / jnp.sum(volume_weights)
    assert warm == pytest.approx(expected_weighted_gauge, abs=1.0e-8)


def test_nonuniform_poisson_dispatches_to_metric_solver():
    rhs = jnp.zeros((3, 4, 3))
    field, residual, iterations, initial = _poisson_jacobi_3d(
        rhs,
        dx=0.4,
        dy=jnp.asarray([0.2, 0.3, 0.4, 0.5]),
        dz=jnp.asarray([0.25, 0.35, 0.4]),
        iterations=4,
        tolerance=1.0e-12,
    )
    assert jnp.allclose(field, 0.0)
    assert residual == pytest.approx(0.0)
    assert initial == pytest.approx(0.0)
    assert iterations == 1


def test_solvax_metric_pressure_poisson_is_jitted_and_differentiable():
    dy = jnp.asarray([0.2, 0.3, 0.4, 0.5])
    dz = jnp.asarray([0.25, 0.35, 0.4])
    x = jnp.linspace(-1.0, 1.0, 4)[:, None, None]
    y = jnp.linspace(-1.0, 1.0, 4)[None, :, None]
    z = jnp.linspace(-1.0, 1.0, 3)[None, None, :]
    rhs_shape = jnp.sin(jnp.pi * x) * jnp.cos(jnp.pi * y) * jnp.ones_like(z)
    mobility = jnp.broadcast_to(1.0 + 0.1 * y, rhs_shape.shape)

    def objective(amplitude):
        pressure, _, _, _, _, _, _ = _solvax_pressure_poisson_duct(
            amplitude * rhs_shape,
            mobility,
            dx=0.4,
            dy=dy,
            dz=dz,
            iterations=200,
            tolerance=1.0e-10,
        )
        return jnp.mean(pressure**2)

    compiled_value_and_grad = jax.jit(jax.value_and_grad(objective))
    value, gradient = compiled_value_and_grad(jnp.asarray(1.0))
    step = 1.0e-4
    finite_difference = (objective(1.0 + step) - objective(1.0 - step)) / (2.0 * step)
    assert jnp.isfinite(value)
    assert jnp.isfinite(gradient)
    assert gradient == pytest.approx(finite_difference, rel=1.0e-6, abs=1.0e-8)
    assert gradient == pytest.approx(2.0 * value, rel=1.0e-6, abs=1.0e-8)

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
    coefficient_finite_difference = (
        coefficient_objective(1.0 + step) - coefficient_objective(1.0 - step)
    ) / (2.0 * step)
    assert coefficient_gradient == pytest.approx(
        coefficient_finite_difference, rel=1.0e-6, abs=1.0e-8
    )
    assert coefficient_gradient == pytest.approx(
        -2.0 * coefficient_value, rel=1.0e-6, abs=1.0e-8
    )


def test_implicit_duct_diffusion_reconstructs_manufactured_field_and_gradient():
    dy = jnp.asarray([0.2, 0.3, 0.4, 0.5])
    dz = jnp.asarray([0.25, 0.35, 0.4])
    shape = (4, 4, 3)
    manufactured = jnp.arange(np.prod(shape), dtype=float).reshape(shape) / 100.0
    viscosity = jnp.full(shape, 0.07)
    mask = jnp.ones(shape, dtype=bool)
    dt = 0.03
    laplacian = _masked_laplacian_duct(manufactured, mask, dx=0.4, dy=dy, dz=dz)
    rhs = manufactured - dt * viscosity * laplacian
    solved, residual, converged = _solvax_implicit_diffusion_duct(
        rhs,
        viscosity,
        dt=dt,
        dx=0.4,
        dy=dy,
        dz=dz,
        iterations=300,
        tolerance=1.0e-10,
    )
    assert bool(converged)
    assert float(residual) < 1.0e-8
    assert solved == pytest.approx(manufactured, abs=1.0e-8)

    def objective(scale):
        field, _, _ = _solvax_implicit_diffusion_duct(
            scale * rhs,
            viscosity,
            dt=dt,
            dx=0.4,
            dy=dy,
            dz=dz,
            iterations=300,
            tolerance=1.0e-10,
        )
        return jnp.mean(field**2)

    value, gradient = jax.jit(jax.value_and_grad(objective))(jnp.asarray(1.0))
    assert gradient == pytest.approx(2.0 * value, rel=1.0e-6, abs=1.0e-8)


def test_metric_face_projection_has_jitted_implicit_gradient():
    nx, ny, nz = 4, 4, 3
    dy = jnp.asarray([0.2, 0.3, 0.4, 0.5])
    dz = jnp.asarray([0.25, 0.35, 0.4])
    base_u = jnp.arange(nx * ny * nz, dtype=float).reshape(nx, ny, nz) / 100.0
    zeros = jnp.zeros_like(base_u)
    rho = jnp.ones_like(base_u)
    mask = jnp.ones_like(base_u, dtype=bool)

    def objective(amplitude):
        projected_u, _, _, pressure, divergence = _face_flux_pressure_projection_duct(
            amplitude * base_u,
            zeros,
            zeros,
            rho,
            mask,
            dt=0.05,
            dx=0.4,
            dy=dy,
            dz=dz,
            iterations=200,
            tolerance=1.0e-10,
            fluid_bounds=(0, ny, 0, nz),
        )
        return jnp.mean(projected_u**2) + 0.01 * jnp.mean(pressure**2) + divergence**2

    value, gradient = jax.jit(jax.value_and_grad(objective))(jnp.asarray(1.0))
    step = 1.0e-4
    finite_difference = (objective(1.0 + step) - objective(1.0 - step)) / (2.0 * step)
    assert jnp.isfinite(value)
    assert jnp.isfinite(gradient)
    assert gradient == pytest.approx(finite_difference, rel=2.0e-5, abs=1.0e-8)


def test_nonuniform_conservative_current_uses_face_center_distances():
    y_faces = jnp.asarray([-1.0, -0.8, -0.35, 0.0, 0.2, 0.6, 0.85, 1.0])
    z_faces = jnp.asarray([-1.0, -0.65, -0.1, 0.25, 0.7, 1.0])
    y = 0.5 * (y_faces[:-1] + y_faces[1:])
    z = 0.5 * (z_faces[:-1] + z_faces[1:])
    yy, zz = jnp.meshgrid(y, z, indexing="ij")
    phi = jnp.broadcast_to((2.0 * yy - 3.0 * zz)[None, :, :], (3, 7, 5))
    sigma = jnp.ones_like(phi)
    zeros = jnp.zeros_like(phi)
    fx, fy, fz = _conservative_current_fluxes_3d(
        sigma,
        phi,
        zeros,
        zeros,
        zeros,
        dx=0.4,
        dy=jnp.diff(y_faces),
        dz=jnp.diff(z_faces),
    )
    assert fy[:, 1:-1, :] == pytest.approx(-2.0)
    assert fz[:, :, 1:-1] == pytest.approx(3.0)
    divergence = (
        (fx[1:] - fx[:-1]) / 0.4
        + (fy[:, 1:, :] - fy[:, :-1, :]) / jnp.diff(y_faces)[None, :, None]
        + (fz[:, :, 1:] - fz[:, :, :-1]) / jnp.diff(z_faces)[None, None, :]
    )
    assert divergence[:, 1:-1, 1:-1] == pytest.approx(0.0, abs=1.0e-12)


def test_masked_diffusion_places_no_slip_at_fluid_solid_face():
    field = jnp.ones((3, 6, 6))
    mask = jnp.zeros_like(field, dtype=bool).at[:, 1:5, 1:5].set(True)
    widths = jnp.asarray([0.2, 0.3, 0.4, 0.35, 0.25, 0.2])
    laplacian = _masked_laplacian_duct(field, mask, dx=0.5, dy=widths, dz=widths)
    assert jnp.allclose(jnp.where(mask, 0.0, laplacian), 0.0)
    assert float(jnp.max(laplacian[:, 2:4, 2:4])) == pytest.approx(0.0)
    assert float(jnp.min(laplacian[:, 1:5, 1:5])) < 0.0


def test_nonuniform_face_flux_projection_closes_discrete_divergence():
    nx, ny, nz = 5, 6, 5
    dy = jnp.asarray([0.2, 0.3, 0.45, 0.4, 0.35, 0.3])
    dz = jnp.asarray([0.25, 0.4, 0.5, 0.45, 0.3])
    x = jnp.linspace(0.0, 1.0, nx)[:, None, None]
    y = jnp.linspace(-1.0, 1.0, ny)[None, :, None]
    z = jnp.linspace(-1.0, 1.0, nz)[None, None, :]
    u = jnp.cos(2.0 * jnp.pi * x) * (1.0 + 0.1 * y) * jnp.ones_like(z)
    v = 0.2 * jnp.sin(jnp.pi * y) * jnp.ones_like(x + z)
    w = -0.15 * jnp.sin(jnp.pi * z) * jnp.ones_like(x + y)
    mask = jnp.ones((nx, ny, nz), dtype=bool)
    projected = _face_flux_pressure_projection_duct(
        u,
        v,
        w,
        jnp.ones_like(u),
        mask,
        dt=0.05,
        dx=0.25,
        dy=dy,
        dz=dz,
        iterations=200,
        tolerance=1.0e-10,
    )
    projected_u, projected_v, projected_w, pressure, divergence = projected
    assert divergence < 1.0e-8
    assert jnp.isfinite(projected_u).all()
    assert jnp.isfinite(projected_v).all()
    assert jnp.isfinite(projected_w).all()
    assert jnp.isfinite(pressure).all()


def test_face_flux_projection_requires_nonempty_rectangular_fluid_mask():
    empty = jnp.zeros((3, 4, 4), dtype=bool)
    with pytest.raises(ValueError, match="nonempty"):
        _rectangular_fluid_bounds(empty)

    nonrectangular = empty.at[:, 1:3, 1:3].set(True).at[:, 1, 1].set(False)
    with pytest.raises(ValueError, match="rectangular"):
        _rectangular_fluid_bounds(nonrectangular)


def test_fixed_flow_face_projection_closes_divergence_and_flow_constraint():
    nx, ny, nz = 5, 6, 5
    dy = jnp.asarray([0.2, 0.3, 0.45, 0.4, 0.35, 0.3])
    dz = jnp.asarray([0.25, 0.4, 0.5, 0.45, 0.3])
    x = jnp.linspace(0.0, 1.0, nx)[:, None, None]
    y = jnp.linspace(-1.0, 1.0, ny)[None, :, None]
    z = jnp.linspace(-1.0, 1.0, nz)[None, None, :]
    u = 0.7 + 0.1 * jnp.cos(2.0 * jnp.pi * x) * jnp.ones_like(y + z)
    v = 0.2 * jnp.sin(jnp.pi * y) * jnp.ones_like(x + z)
    w = -0.15 * jnp.sin(jnp.pi * z) * jnp.ones_like(x + y)
    mask = jnp.ones((nx, ny, nz), dtype=bool)
    rho = jnp.ones_like(u)
    cell_area = jnp.broadcast_to(dy[None, :, None] * dz[None, None, :], u.shape)
    dt = 0.05
    response = jnp.full_like(u, dt)
    target = 1.25
    projected = _fixed_flow_face_flux_projection_duct(
        u,
        v,
        w,
        rho,
        mask,
        response,
        cell_area,
        target_flow_rate=target,
        base_pressure_loss_gradient=0.0,
        dt=dt,
        dx=0.25,
        dy=dy,
        dz=dz,
        iterations=200,
        tolerance=1.0e-10,
    )
    projected_u, _, _, pressure, pressure_loss, divergence, flow_error = projected
    flow = jnp.sum(projected_u * cell_area, axis=(1, 2))
    assert flow == pytest.approx(target, abs=1.0e-8)
    assert divergence < 1.0e-8
    assert flow_error < 1.0e-8
    assert jnp.isfinite(pressure).all()
    assert jnp.isfinite(pressure_loss).all()

    warm = _fixed_flow_face_flux_projection_duct(
        u,
        v,
        w,
        rho,
        mask,
        response,
        cell_area,
        target_flow_rate=target,
        base_pressure_loss_gradient=0.0,
        dt=dt,
        dx=0.25,
        dy=dy,
        dz=dz,
        iterations=200,
        tolerance=1.0e-10,
        initial_pressure=pressure,
    )
    assert warm[3] == pytest.approx(pressure, rel=1.0e-8, abs=1.0e-8)

    varying_response = response.at[-1].multiply(2.0)
    with pytest.raises(ValueError, match="x-invariant"):
        _fixed_flow_face_flux_projection_duct(
            u,
            v,
            w,
            rho,
            mask,
            varying_response,
            cell_area,
            target_flow_rate=target,
            base_pressure_loss_gradient=0.0,
            dt=dt,
            dx=0.25,
            dy=dy,
            dz=dz,
            iterations=2,
            tolerance=1.0e-8,
        )


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
        )
        return jnp.mean(field**2)

    value, gradient = jax.jit(jax.value_and_grad(objective))(jnp.asarray(1.0))
    assert gradient == pytest.approx(-2.0 * value, rel=1.0e-6, abs=1.0e-8)


def test_implicit_pipe_diffusion_reconstructs_manufactured_field():
    r_faces = jnp.asarray([0.0, 0.12, 0.3, 0.55, 0.78, 1.0])
    r_centers = 0.5 * (r_faces[:-1] + r_faces[1:])
    shape = (4, 5, 8)
    manufactured = jnp.arange(np.prod(shape), dtype=float).reshape(shape) / 1000.0
    viscosity = jnp.full(shape, 0.04)
    mask = jnp.ones(shape, dtype=bool)
    dt = 0.02
    laplacian = _masked_laplacian_pipe(
        manufactured,
        mask,
        dx=0.4,
        r_faces=r_faces,
        r_centers=r_centers,
        dtheta=2.0 * jnp.pi / 8,
        radial_fluid_count=5,
    )
    rhs = manufactured - dt * viscosity * laplacian
    solved, residual, converged = _solvax_implicit_diffusion_pipe(
        rhs,
        viscosity,
        dt=dt,
        dx=0.4,
        r_faces=r_faces,
        r_centers=r_centers,
        dtheta=2.0 * jnp.pi / 8,
        iterations=500,
        tolerance=1.0e-10,
    )
    assert bool(converged)
    assert float(residual) < 1.0e-8
    assert solved == pytest.approx(manufactured, abs=1.0e-8)


def test_pipe_face_projection_and_masked_diffusion_use_fluid_wall_face():
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

    constant = jnp.where(mask, 1.0, 0.0)
    laplacian = _masked_laplacian_pipe(
        constant,
        mask,
        dx=0.25,
        r_faces=r_faces,
        r_centers=r_centers,
        dtheta=2.0 * jnp.pi / ntheta,
        radial_fluid_count=fluid_count,
    )
    assert jnp.allclose(laplacian[:, : fluid_count - 1, :], 0.0)
    assert float(jnp.max(laplacian[:, fluid_count - 1, :])) < 0.0
    assert jnp.allclose(laplacian[:, fluid_count:, :], 0.0)
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


def test_pipe_jacobi_pressure_fallback_solves_compatible_zero_rhs():
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


def test_net_boundary_current_residual_cancels_balanced_faces():
    field = jnp.ones((2, 2, 2))
    assert _net_boundary_current_residual(
        field,
        jnp.zeros_like(field),
        jnp.zeros_like(field),
        dx=1.0,
        dy=1.0,
        dz=1.0,
    ) == pytest.approx(0.0)


def test_build_extruded_problem_from_case_preserves_case_and_profile():
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


def test_fixed_flow_pressure_constraint_rejects_zero_response():
    field = jnp.zeros((1, 2, 2))
    with pytest.raises(ValueError, match="nonzero"):
        _apply_fixed_flow_pressure_constraint(
            field,
            unit_pressure_response=field,
            active_mask=jnp.ones_like(field, dtype=bool),
            cell_area=jnp.ones_like(field),
            target_flow_rate=1.0,
        )


def test_cross_duct_pressure_difference_samples_adjacent_wall_midpoints():
    p_z = jnp.broadcast_to(jnp.arange(4.0)[None, None, :], (2, 3, 4))
    active = jnp.ones_like(p_z, dtype=bool)
    assert _cross_duct_pressure_difference(
        p_z, active_mask=active, magnetic_axis=1, side_axis=2
    ).tolist() == pytest.approx([1.5, 1.5])

    p_y = jnp.broadcast_to(jnp.arange(3.0)[None, :, None], (2, 3, 4))
    assert _cross_duct_pressure_difference(
        p_y, active_mask=active, magnetic_axis=1, side_axis=2
    ).tolist() == pytest.approx([-1.0, -1.0])


def test_cross_duct_pressure_difference_rejects_invalid_contract():
    p = jnp.zeros((1, 2, 2))
    with pytest.raises(ValueError, match="distinct members"):
        _cross_duct_pressure_difference(
            p, active_mask=jnp.ones_like(p, dtype=bool), magnetic_axis=1, side_axis=1
        )
    with pytest.raises(ValueError, match="active fluid"):
        _cross_duct_pressure_difference(
            p, active_mask=jnp.zeros_like(p, dtype=bool), magnetic_axis=1, side_axis=2
        )


def test_extruded_sharding_validates_and_places_fields(
    monkeypatch: pytest.MonkeyPatch,
):
    field = jnp.zeros((4, 2, 2))
    assert _shard_extruded_fields((field,), num_devices=1)[0] is field

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
    placed = _shard_extruded_fields((field,), num_devices=2)
    assert jnp.all(placed[0] == 1)


def test_spatial_sharding_rejects_unimplemented_extruded_paths():
    problem = build_square_duct_extruded_problem(nx_stations=4, ny=4, nz=4)
    with pytest.raises(NotImplementedError, match="ALEX B2"):
        solve_extruded_inductionless(problem, num_devices=2)


def test_pressure_observable_update_uses_magnetic_pressure_normalization():
    assert _normalized_pressure_observable_update(
        jnp.array([2.0, -1.0]),
        jnp.zeros(2),
        jnp.array([25.0, 100.0]),
    ) == pytest.approx(0.02)
    assert _normalized_pressure_observable_update(
        jnp.array([2.0]), jnp.zeros(1), jnp.array([0.25])
    ) == pytest.approx(2.0)


def test_smooth_fringing_profile_produces_bounded_station_scales():
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


def test_smooth_fringing_profile_rejects_invalid_axis():
    with pytest.raises(ValueError, match="Unsupported magnetic axis"):
        smooth_fringing_profile(
            length=1.0,
            nx=3,
            entry_center=0.2,
            exit_center=0.8,
            transition_width=0.1,
            axis="bad",
        )


def test_clone_case_with_field_replaces_constant_field():
    base_case, _ = build_square_duct_fringing_benchmark(nx_stations=5, ny=8, nz=8)
    shifted = clone_case_with_field(base_case, axis="y", magnitude=3.0, suffix="probe")

    assert shifted.name.endswith("probe")
    assert shifted.magnetic_field.value == (0.0, 3.0, 0.0)


def test_clone_case_with_field_supports_x_axis():
    base_case, _ = build_square_duct_fringing_benchmark(nx_stations=5, ny=8, nz=8)
    shifted = clone_case_with_field(base_case, axis="x", magnitude=2.0)
    assert shifted.magnetic_field.value == (2.0, 0.0, 0.0)


def test_clone_case_with_field_rejects_invalid_axis():
    base_case, _ = build_square_duct_fringing_benchmark(nx_stations=5, ny=8, nz=8)
    with pytest.raises(ValueError, match="Unsupported magnetic axis"):
        clone_case_with_field(base_case, axis="bad", magnitude=1.0)


def test_station_axial_current_from_fluxes_averages_adjacent_x_faces():
    fx = jnp.asarray(
        [
            [[0.0, 1.0]],
            [[2.0, 3.0]],
            [[4.0, 5.0]],
        ]
    )
    cell_area = jnp.asarray([[2.0, 4.0]])

    axial_current = _station_axial_current_from_fluxes(fx, cell_area)

    assert axial_current.shape == (2,)
    assert axial_current[0] == pytest.approx(10.0)
    assert axial_current[1] == pytest.approx(22.0)


def test_run_fringing_station_sweep_chains_initial_state(
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


def test_run_fringing_station_sweep_requires_constant_field():
    base_case, profile = build_square_duct_fringing_benchmark(nx_stations=3, ny=8, nz=8)
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


def test_build_square_duct_extruded_problem_marks_solver_family():
    problem = build_square_duct_extruded_problem(nx_stations=5, ny=8, nz=8)

    assert problem.case.solver.kind == "extruded_inductionless"
    assert problem.profile.x.shape == (5,)


def test_build_layered_duct_extruded_problem_marks_solver_family():
    problem = build_layered_duct_extruded_problem(
        nx_stations=5, ny=8, nz=8, wall_cells=1, insulator_cells=1
    )

    assert problem.case.solver.kind == "extruded_inductionless"
    assert problem.case.geometry.kind == "layered_duct"
    assert problem.profile.x.shape == (5,)


def test_build_pipe_ogrid_extruded_problem_marks_solver_family():
    problem = build_pipe_ogrid_extruded_problem(nx_stations=5, nr=6, ntheta=12)

    assert problem.case.solver.kind == "extruded_inductionless"
    assert problem.case.geometry.kind == "pipe_ogrid"
    assert problem.profile.x.shape == (5,)


def test_build_bent_pipe_extruded_problem_marks_solver_family():
    problem = build_bent_pipe_extruded_problem(nx_stations=5, nr=6, ntheta=12)

    assert problem.case.solver.kind == "extruded_inductionless"
    assert problem.case.geometry.kind == "bent_pipe"
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
        lambda problem, initial_bundle=None: fake_bundle,
    )

    solution = solve_extruded_inductionless(problem)
    assert len(solution.station_history) == 3
    assert solution.validation.station_count == 3
    assert solution.bundle.solver_kind == "extruded_inductionless"
    assert "axial_current" in solution.station_history[0]
    assert "wall_current_leakage" in solution.station_history[0]
    assert "boundary_current_residual" in solution.station_history[0]
    assert "pressure_span" in solution.station_history[0]


def test_solve_extruded_inductionless_projection_returns_finite_rectangular_bundle():
    problem = build_square_duct_extruded_problem(ha_peak=8.0, nx_stations=4, ny=4, nz=4)

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
        lambda problem, initial_bundle=None: type(
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

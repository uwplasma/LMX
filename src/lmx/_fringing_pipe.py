"""Mapped-pipe 3-D kernels and pressure systems."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.scipy.fft import dct, idct
from solvax import (
    KrylovSolution,
    additive_tridiagonal_line_preconditioner,
    block_thomas_factor,
    block_thomas_solve,
    gmres,
    linear_solve,
    pcg_linear_solve,
    tridiagonal_solve,
)

from ._fringing_common import (
    _apply_fixed_flow_pressure_constraint,
    _enforce_stationwise_flow_rate_3d,
    _finalize_local_pressure_solve,
    _harmonic_mean,
    _nonuniform_axis_gradient,
    _reuse_fringing_jit,
    _reuse_modal_factors,
    _spacing_vector,
)


def _pipe_theta_neighbors(field: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    theta_prev = jnp.concatenate([field[:, :, -1:], field[:, :, :-1]], axis=2)
    theta_next = jnp.concatenate([field[:, :, 1:], field[:, :, :1]], axis=2)
    return theta_prev, theta_next


def _pipe_gradient_3d(
    field: jnp.ndarray,
    *,
    dx: float,
    dr: float | jnp.ndarray,
    dtheta: float,
    r: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    dr_values = jnp.asarray(dr)
    if dr_values.ndim:
        widths = _spacing_vector(dr_values, field.shape[1], dtype=field.dtype)
        safe_r = jnp.broadcast_to(jnp.asarray(r, dtype=field.dtype), field.shape)
    else:
        widths = None
        safe_r = jnp.maximum(r, 0.5 * dr_values)
    x_west = jnp.concatenate([field[:1], field[:-1]], axis=0)
    x_east = jnp.concatenate([field[1:], field[-1:]], axis=0)
    r_inner = jnp.concatenate([field[:, :1, :], field[:, :-1, :]], axis=1)
    r_outer = jnp.concatenate([field[:, 1:, :], field[:, -1:, :]], axis=1)
    theta_prev, theta_next = _pipe_theta_neighbors(field)
    d_dx = (x_east - x_west) / jnp.maximum(2.0 * dx, 1.0e-12)
    d_dr = (
        (r_outer - r_inner) / jnp.maximum(2.0 * dr_values, 1.0e-12)
        if widths is None
        else _nonuniform_axis_gradient(field, widths, axis=1)
    )
    d_dtheta = (theta_next - theta_prev) / jnp.maximum(2.0 * dtheta * safe_r, 1.0e-12)
    d_dr = d_dr.at[:, 0, :].set(0.0)
    d_dtheta = d_dtheta.at[:, 0, :].set(0.0)
    return d_dx, d_dr, d_dtheta


def _pipe_laplacian_3d(
    field: jnp.ndarray,
    *,
    dx: float,
    dr: float | jnp.ndarray,
    dtheta: float,
    r: jnp.ndarray,
    outer_dirichlet: bool = True,
) -> jnp.ndarray:
    dr_values = jnp.asarray(dr)
    if dr_values.ndim:
        widths = _spacing_vector(dr_values, field.shape[1], dtype=field.dtype)
        r_values = jnp.asarray(r, dtype=field.dtype)
        r_centers = r_values[0, :, 0] if r_values.ndim == 3 else r_values.reshape(-1)
        r_faces = jnp.concatenate([jnp.zeros((1,), dtype=field.dtype), jnp.cumsum(widths)])
        safe_r = jnp.broadcast_to(r_centers[None, :, None], field.shape)
    else:
        widths = None
        safe_r = jnp.maximum(r, 0.5 * dr_values)
    x_west = jnp.concatenate([field[:1], field[:-1]], axis=0)
    x_east = jnp.concatenate([field[1:], field[-1:]], axis=0)
    r_inner = jnp.concatenate([field[:, :1, :], field[:, :-1, :]], axis=1)
    outer_ghost = jnp.zeros_like(field[:, -1:, :]) if outer_dirichlet else field[:, -1:, :]
    r_outer = jnp.concatenate([field[:, 1:, :], outer_ghost], axis=1)
    theta_prev, theta_next = _pipe_theta_neighbors(field)
    dxx = (x_west - 2.0 * field + x_east) / jnp.maximum(dx**2, 1.0e-12)
    radial_step = dr_values if widths is None else jnp.asarray(1.0)
    drr = (r_inner - 2.0 * field + r_outer) / jnp.maximum(radial_step**2, 1.0e-12)
    d_dr = (r_outer - r_inner) / jnp.maximum(2.0 * radial_step, 1.0e-12)
    dtheta2 = (theta_prev - 2.0 * field + theta_next) / jnp.maximum((safe_r**2) * dtheta**2, 1.0e-12)
    if widths is None:
        lap = dxx + drr + d_dr / safe_r + dtheta2
        return lap.at[:, 0, :].set(
            dxx[:, 0, :] + 2.0 * (field[:, 1, :] - field[:, 0, :]) / jnp.maximum(dr_values**2, 1.0e-12)
        )
    radial_flux = jnp.zeros((field.shape[0], field.shape[1] + 1, field.shape[2]), dtype=field.dtype)
    center_distance = 0.5 * (widths[:-1] + widths[1:])
    radial_flux = radial_flux.at[:, 1:-1, :].set(
        r_faces[None, 1:-1, None] * (field[:, 1:, :] - field[:, :-1, :]) / center_distance[None, :, None]
    )
    if outer_dirichlet:
        radial_flux = radial_flux.at[:, -1, :].set(r_faces[-1] * (-field[:, -1, :]) / (0.5 * widths[-1]))
    radial_term = (radial_flux[:, 1:, :] - radial_flux[:, :-1, :]) / jnp.maximum(
        r_centers[None, :, None] * widths[None, :, None], 1.0e-20
    )
    return dxx + radial_term + dtheta2


def _pipe_divergence_3d(
    jx: jnp.ndarray,
    jr: jnp.ndarray,
    jtheta: jnp.ndarray,
    *,
    dx: float,
    dr: float | jnp.ndarray,
    dtheta: float,
    r: jnp.ndarray,
) -> jnp.ndarray:
    dr_values = jnp.asarray(dr)
    if dr_values.ndim:
        widths = _spacing_vector(dr_values, jx.shape[1], dtype=jx.dtype)
        r_values = jnp.asarray(r, dtype=jx.dtype)
        r_centers = r_values[0, :, 0] if r_values.ndim == 3 else r_values.reshape(-1)
        r_faces = jnp.concatenate([jnp.zeros((1,), dtype=jx.dtype), jnp.cumsum(widths)])
        safe_r = jnp.broadcast_to(r_centers[None, :, None], jx.shape)
    else:
        widths = None
        safe_r = jnp.maximum(r, 0.5 * dr_values)
    djx_dx = _pipe_gradient_3d(jx, dx=dx, dr=dr, dtheta=dtheta, r=r)[0]
    if widths is not None:
        radial_face = jnp.zeros((jr.shape[0], jr.shape[1] + 1, jr.shape[2]), dtype=jr.dtype)
        radial_face = radial_face.at[:, 1:-1, :].set(0.5 * (jr[:, 1:, :] + jr[:, :-1, :]))
        radial_term = (
            r_faces[None, 1:, None] * radial_face[:, 1:, :]
            - r_faces[None, :-1, None] * radial_face[:, :-1, :]
        ) / jnp.maximum(r_centers[None, :, None] * widths[None, :, None], 1.0e-20)
        theta_prev, theta_next = _pipe_theta_neighbors(jtheta)
        theta_term = (theta_next - theta_prev) / jnp.maximum(2.0 * dtheta * safe_r, 1.0e-12)
        return djx_dx + radial_term + theta_term
    rjr = safe_r * jr
    rjr_inner = jnp.concatenate([rjr[:, :1, :], rjr[:, :-1, :]], axis=1)
    rjr_outer = jnp.concatenate([rjr[:, 1:, :], rjr[:, -1:, :]], axis=1)
    radial_term = (rjr_outer - rjr_inner) / jnp.maximum(2.0 * dr_values * safe_r, 1.0e-12)
    theta_prev, theta_next = _pipe_theta_neighbors(jtheta)
    theta_term = (theta_next - theta_prev) / jnp.maximum(2.0 * dtheta * safe_r, 1.0e-12)
    divergence = djx_dx + radial_term + theta_term
    return divergence.at[:, 0, :].set(djx_dx[:, 0, :] + 2.0 * jr[:, 1, :] / jnp.maximum(dr_values, 1.0e-12))


def _pipe_variable_diffusion_coefficients_3d(
    coefficient: jnp.ndarray,
    *,
    dx: float,
    r_faces: jnp.ndarray,
    r_centers: jnp.ndarray,
    dtheta: float,
) -> tuple[jnp.ndarray, ...]:
    """Cylindrical FV coefficients for ``div(coefficient grad)``."""

    radial_widths = jnp.diff(r_faces)
    radial_distance = jnp.diff(r_centers)
    sigma_x = _harmonic_mean(coefficient[1:], coefficient[:-1])
    coef_x_w = jnp.concatenate(
        [jnp.zeros_like(coefficient[:1]), sigma_x / jnp.maximum(dx**2, 1.0e-12)],
        axis=0,
    )
    coef_x_e = jnp.concatenate(
        [sigma_x / jnp.maximum(dx**2, 1.0e-12), jnp.zeros_like(coefficient[-1:])],
        axis=0,
    )

    sigma_r = _harmonic_mean(coefficient[:, 1:, :], coefficient[:, :-1, :])
    radial_face_factor = r_faces[1:-1][None, :, None] / jnp.maximum(radial_distance[None, :, None], 1.0e-20)
    coef_r_inner = (
        jnp.zeros_like(coefficient)
        .at[:, 1:, :]
        .set(
            sigma_r
            * radial_face_factor
            / jnp.maximum(
                r_centers[None, 1:, None] * radial_widths[None, 1:, None],
                1.0e-20,
            )
        )
    )
    coef_r_outer = (
        jnp.zeros_like(coefficient)
        .at[:, :-1, :]
        .set(
            sigma_r
            * radial_face_factor
            / jnp.maximum(
                r_centers[None, :-1, None] * radial_widths[None, :-1, None],
                1.0e-20,
            )
        )
    )

    sigma_theta_out = _harmonic_mean(coefficient, jnp.roll(coefficient, -1, axis=2))
    coef_theta_out = sigma_theta_out / jnp.maximum(r_centers[None, :, None] ** 2 * dtheta**2, 1.0e-20)
    coef_theta_in = jnp.roll(coef_theta_out, 1, axis=2)
    return (
        coef_x_w,
        coef_x_e,
        coef_r_inner,
        coef_r_outer,
        coef_theta_in,
        coef_theta_out,
    )


def _apply_pipe_diffusion_coefficients_3d(
    field: jnp.ndarray, coefficients: tuple[jnp.ndarray, ...]
) -> jnp.ndarray:
    coef_x_w, coef_x_e, coef_r_i, coef_r_o, coef_t_i, coef_t_o = coefficients
    x_west = jnp.concatenate([field[:1], field[:-1]], axis=0)
    x_east = jnp.concatenate([field[1:], field[-1:]], axis=0)
    r_inner = jnp.concatenate([field[:, :1, :], field[:, :-1, :]], axis=1)
    r_outer = jnp.concatenate([field[:, 1:, :], field[:, -1:, :]], axis=1)
    theta_in = jnp.roll(field, 1, axis=2)
    theta_out = jnp.roll(field, -1, axis=2)
    return (
        coef_x_w * (x_west - field)
        + coef_x_e * (x_east - field)
        + coef_r_i * (r_inner - field)
        + coef_r_o * (r_outer - field)
        + coef_t_i * (theta_in - field)
        + coef_t_o * (theta_out - field)
    )


def _separable_pressure_poisson_pipe(
    rhs: jnp.ndarray,
    coefficient: jnp.ndarray,
    *,
    dx: float,
    r_faces: jnp.ndarray,
    r_centers: jnp.ndarray,
    dtheta: float,
    tolerance: float,
) -> tuple[jnp.ndarray, ...]:
    """Solve an axisymmetric cylindrical Neumann operator by x/theta modes."""

    nx, nr, ntheta = rhs.shape
    radial_widths = jnp.diff(r_faces)
    volume = jnp.broadcast_to(
        r_centers[None, :, None] * radial_widths[None, :, None] * dtheta,
        rhs.shape,
    )
    volume_sum = jnp.sum(volume)
    rhs_compatible = rhs - jnp.sum(rhs * volume) / volume_sum
    coefficients = _pipe_variable_diffusion_coefficients_3d(
        coefficient,
        dx=dx,
        r_faces=r_faces,
        r_centers=r_centers,
        dtheta=dtheta,
    )
    _, _, radial_inner, radial_outer, _, theta_outer = coefficients
    sigma = coefficient[0, :, 0]
    x_wave = jnp.pi * jnp.arange(nx, dtype=rhs.dtype) / nx
    theta_wave = 2.0 * jnp.pi * jnp.arange(ntheta // 2 + 1, dtype=rhs.dtype) / ntheta
    radial_inner_rate = radial_inner[0, :, 0]
    radial_outer_rate = radial_outer[0, :, 0]
    radial_rate = radial_inner_rate + radial_outer_rate
    modal_rate = (
        radial_rate[:, None, None]
        + 2.0 * sigma[:, None, None] * (1.0 - jnp.cos(x_wave))[None, :, None] / dx**2
        + 2.0 * theta_outer[0, :, 0][:, None, None] * (1.0 - jnp.cos(theta_wave))[None, None, :]
    )
    radial_volume = r_centers * radial_widths * dtheta
    diagonal = radial_volume[:, None, None] * modal_rate
    lower = jnp.broadcast_to(
        -radial_volume[:, None, None] * radial_inner_rate[:, None, None],
        diagonal.shape,
    )
    upper = jnp.broadcast_to(
        -radial_volume[:, None, None] * radial_outer_rate[:, None, None],
        diagonal.shape,
    )
    modal_rhs = jnp.moveaxis(
        jnp.fft.rfft(dct(-volume * rhs_compatible, type=2, norm="ortho", axis=0), axis=2),
        1,
        0,
    )
    # The compatible (kx, m) = (0, 0) radial system has one redundant row.
    # Pin its outer cell for the solve, then restore the volume-weighted gauge.
    lower = lower.at[-1, 0, 0].set(0.0)
    upper = upper.at[-1, 0, 0].set(0.0)
    diagonal = diagonal.at[-1, 0, 0].set(1.0)
    modal_rhs = modal_rhs.at[-1, 0, 0].set(0.0)
    modes = tridiagonal_solve(lower, diagonal, upper, modal_rhs.real) + 1j * (
        tridiagonal_solve(lower, diagonal, upper, modal_rhs.imag)
    )
    field = idct(
        jnp.fft.irfft(jnp.moveaxis(modes, 0, 1), n=ntheta, axis=2),
        type=2,
        norm="ortho",
        axis=0,
    )
    field = field - jnp.sum(field * volume) / volume_sum
    local_residual = jnp.max(
        jnp.abs(_apply_pipe_diffusion_coefficients_3d(field, coefficients) - rhs_compatible)
    )
    linear_rhs = -volume * rhs_compatible
    linear_residual = jnp.linalg.norm(
        -volume * _apply_pipe_diffusion_coefficients_3d(field, coefficients) - linear_rhs
    )
    relative_residual = linear_residual / jnp.maximum(jnp.linalg.norm(linear_rhs), 1.0e-30)
    converged = local_residual <= tolerance
    return (
        field,
        linear_residual,
        converged,
        relative_residual,
        jnp.asarray(1),
        jnp.where(converged, 1, 0),
        local_residual,
    )


def _solvax_pressure_poisson_pipe(
    rhs: jnp.ndarray,
    coefficient: jnp.ndarray,
    *,
    dx: float,
    r_faces: jnp.ndarray,
    r_centers: jnp.ndarray,
    dtheta: float,
    iterations: int,
    tolerance: float,
    initial_field: jnp.ndarray | None = None,
    local_tolerance: float | None = None,
    include_theta_line: bool = False,
) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
]:
    """Implicit SOLVAX solve for a cylindrical Neumann FV operator."""

    radial_widths = jnp.diff(r_faces)
    volume = jnp.broadcast_to(
        r_centers[None, :, None] * radial_widths[None, :, None] * jnp.asarray(dtheta, dtype=rhs.dtype),
        rhs.shape,
    )
    volume_sum = jnp.sum(volume)
    rhs_compatible = rhs - jnp.sum(rhs * volume) / volume_sum
    coefficients = _pipe_variable_diffusion_coefficients_3d(
        coefficient,
        dx=dx,
        r_faces=r_faces,
        r_centers=r_centers,
        dtheta=dtheta,
    )
    coef_x_w, coef_x_e, coef_r_i, coef_r_o, coef_t_i, coef_t_o = coefficients

    def matvec(field: jnp.ndarray) -> jnp.ndarray:
        diffusion = _apply_pipe_diffusion_coefficients_3d(field, coefficients)
        gauge = volume * jnp.sum(volume * field) / volume_sum
        return -volume * diffusion + gauge

    diagonal = volume * sum(coefficients) + volume**2 / volume_sum

    directions = (
        (0, -volume * coef_x_w, -volume * coef_x_e),
        (1, -volume * coef_r_i, -volume * coef_r_o),
    )
    precondition = additive_tridiagonal_line_preconditioner(
        diagonal,
        directions,
        periodic_last_axis=((-volume * coef_t_i, -volume * coef_t_o) if include_theta_line else None),
    )

    linear_rhs = -volume * rhs_compatible
    effective_rtol = tolerance
    effective_atol = tolerance
    if local_tolerance is not None:
        local_absolute_target = float(jnp.min(volume)) * local_tolerance
        effective_rtol = 0.0
        effective_atol = min(tolerance, local_absolute_target)
    solution = pcg_linear_solve(
        matvec,
        linear_rhs,
        x0=initial_field,
        precond=precondition,
        transpose_precond=precondition,
        rtol=effective_rtol,
        atol=effective_atol,
        max_steps=iterations,
        transpose_rtol=tolerance,
        transpose_atol=tolerance,
        transpose_max_steps=iterations,
    )
    return _finalize_local_pressure_solve(
        solution,
        linear_rhs=linear_rhs,
        matvec=matvec,
        local_residual_fn=lambda field: (
            _apply_pipe_diffusion_coefficients_3d(field, coefficients) - rhs_compatible
        ),
        volume=volume,
        precondition=precondition,
        iterations=iterations,
        effective_atol=effective_atol,
        local_tolerance=local_tolerance,
    )


def _solve_pipe_diffusion_system(
    linear_rhs: jnp.ndarray,
    volume: jnp.ndarray,
    coefficients: tuple[jnp.ndarray, ...],
    wall_sink: jnp.ndarray,
    initial_field: jnp.ndarray | None,
    *,
    mass_coefficient: float = 1.0,
    diffusion_coefficient: float = 1.0,
    iterations: int,
    tolerance: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Solve one prepared shifted cylindrical diffusion system."""

    coef_x_w, coef_x_e, coef_r_i, coef_r_o, _, _ = coefficients

    def matvec(field: jnp.ndarray) -> jnp.ndarray:
        diffusion = _apply_pipe_diffusion_coefficients_3d(field, coefficients) - wall_sink * field
        return volume * (mass_coefficient * field - diffusion_coefficient * diffusion)

    diagonal = volume * (mass_coefficient + diffusion_coefficient * (sum(coefficients) + wall_sink))
    directions = (
        (
            0,
            -volume * diffusion_coefficient * coef_x_w,
            -volume * diffusion_coefficient * coef_x_e,
        ),
        (
            1,
            -volume * diffusion_coefficient * coef_r_i,
            -volume * diffusion_coefficient * coef_r_o,
        ),
    )
    precondition = additive_tridiagonal_line_preconditioner(diagonal, directions)
    solution = pcg_linear_solve(
        matvec,
        linear_rhs,
        x0=initial_field,
        precond=precondition,
        transpose_precond=precondition,
        rtol=tolerance,
        atol=tolerance,
        max_steps=iterations,
        transpose_rtol=tolerance,
        transpose_atol=tolerance,
        transpose_max_steps=iterations,
    )
    return solution.x, solution.residual_norm, solution.converged


def _solvax_diffusion_pipe(
    rhs: jnp.ndarray,
    viscosity: jnp.ndarray,
    *,
    dt: float | None,
    dx: float,
    r_faces: jnp.ndarray,
    r_centers: jnp.ndarray,
    dtheta: float,
    iterations: int,
    tolerance: float,
    initial_field: jnp.ndarray | None = None,
    reaction: jnp.ndarray | None = None,
    decouple_axial: bool = False,
    _system_solve: Callable | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Solve steady or implicit cylindrical no-slip diffusion.

    ``dt=None`` solves ``-div(viscosity grad(field)) = rhs``. A positive
    ``dt`` solves the implicit update ``field - dt * div(...) = rhs``.
    ``decouple_axial`` retains the axial diagonal while dropping neighboring
    station couplings, providing a cross-section block-Jacobi inverse.
    """

    radial_widths = jnp.diff(r_faces)
    volume = jnp.broadcast_to(
        r_centers[None, :, None] * radial_widths[None, :, None] * dtheta,
        rhs.shape,
    )
    coefficients = _pipe_variable_diffusion_coefficients_3d(
        viscosity,
        dx=dx,
        r_faces=r_faces,
        r_centers=r_centers,
        dtheta=dtheta,
    )
    axial_sink = jnp.zeros_like(rhs)
    if decouple_axial:
        axial_sink = coefficients[0] + coefficients[1]
        zero = jnp.zeros_like(coefficients[0])
        coefficients = (zero, zero, *coefficients[2:])
    wall_sink = (
        jnp.zeros_like(rhs)
        .at[:, -1, :]
        .set(
            viscosity[:, -1, :]
            * r_faces[-1]
            / jnp.maximum(
                r_centers[-1] * radial_widths[-1] * (0.5 * radial_widths[-1]),
                1.0e-20,
            )
        )
    ) + axial_sink
    if reaction is not None:
        wall_sink = wall_sink + reaction
    system = (volume * rhs, volume, coefficients, wall_sink, initial_field)
    if _system_solve is not None:
        return _system_solve(*system)
    return _solve_pipe_diffusion_system(
        *system,
        mass_coefficient=float(dt is not None),
        diffusion_coefficient=1.0 if dt is None else dt,
        iterations=iterations,
        tolerance=tolerance,
    )


def _generic_pipe_step(
    state: tuple[jnp.ndarray, ...],
    *,
    material: tuple[jnp.ndarray, ...],
    field: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    forcing: jnp.ndarray,
    metric: tuple[float | jnp.ndarray, ...],
    solves: tuple[int | float, ...],
    limits: tuple[float, float],
    flow: tuple[float | jnp.ndarray | None, jnp.ndarray, int],
) -> tuple[tuple[jnp.ndarray, ...], tuple[jnp.ndarray, ...]]:
    """Advance the retained generic mapped-pipe equations once."""

    sigma, rho, nu, fluid_mask, cell_area = material
    bx, br, btheta = field
    dt, dx, dr, dtheta, r_centers, radius, r_faces = metric
    projection_iterations, projection_tolerance, electric_iterations, electric_tolerance = solves
    velocity_limit, scalar_limit = limits
    target_flow_rate, unit_pressure_response, radial_fluid_count = flow
    u, v, w, pressure, potential = state
    potential_gradient = _pipe_gradient_3d(potential, dx=dx, dr=dr, dtheta=dtheta, r=radius)
    uxb = (v * btheta - w * br, w * bx - u * btheta, u * br - v * bx)
    current = tuple(sigma * (-gradient + emf) for gradient, emf in zip(potential_gradient, uxb))
    jx, jr, jtheta = current
    lorentz = (jr * btheta - jtheta * br, jtheta * bx - jx * btheta, jx * br - jr * bx)
    pressure_gradient = _pipe_gradient_3d(pressure, dx=dx, dr=dr, dtheta=dtheta, r=radius)
    laplacian = tuple(
        _pipe_laplacian_3d(component, dx=dx, dr=dr, dtheta=dtheta, r=radius) for component in (u, v, w)
    )
    source = (forcing / rho, jnp.zeros_like(rho), jnp.zeros_like(rho))
    predicted = tuple(
        jnp.clip(
            component + dt * (nu * diffusion + drive + force / rho - gradient / rho),
            -velocity_limit,
            velocity_limit,
        )
        for component, diffusion, drive, force, gradient in zip(
            (u, v, w), laplacian, source, lorentz, pressure_gradient, strict=True
        )
    )
    predicted = _enforce_pipe_velocity_bc(
        *predicted,
        r_centers=r_centers,
        r_faces=r_faces,
        fluid_mask=fluid_mask,
        radial_fluid_count=radial_fluid_count,
    )
    divergence = _pipe_divergence_3d(*predicted, dx=dx, dr=dr, dtheta=dtheta, r=radius)
    correction, *_ = _solvax_pressure_poisson_pipe(
        (rho / jnp.maximum(dt, 1.0e-12)) * divergence,
        jnp.ones_like(rho),
        dx=dx,
        r_faces=r_faces,
        r_centers=r_centers,
        dtheta=dtheta,
        iterations=projection_iterations,
        tolerance=projection_tolerance,
        initial_field=pressure,
        include_theta_line=True,
    )
    correction = jnp.clip(correction, -scalar_limit, scalar_limit)
    correction_gradient = _pipe_gradient_3d(correction, dx=dx, dr=dr, dtheta=dtheta, r=radius)
    velocity = tuple(
        jnp.clip(value - (dt / rho) * gradient, -velocity_limit, velocity_limit)
        for value, gradient in zip(predicted, correction_gradient, strict=True)
    )
    velocity = _enforce_pipe_velocity_bc(
        *velocity,
        r_centers=r_centers,
        r_faces=r_faces,
        fluid_mask=fluid_mask,
        radial_fluid_count=radial_fluid_count,
    )
    if target_flow_rate is None:
        velocity = (
            _enforce_stationwise_flow_rate_3d(
                velocity[0], active_mask=fluid_mask, cell_area=cell_area, relaxation=0.25
            ),
            velocity[1],
            velocity[2],
        )
        pressure_loss = jnp.full((u.shape[0],), forcing, dtype=u.dtype)
    else:
        axial, pressure_loss = _apply_fixed_flow_pressure_constraint(
            velocity[0],
            unit_pressure_response=unit_pressure_response,
            active_mask=fluid_mask,
            cell_area=cell_area,
            target_flow_rate=target_flow_rate,
            base_pressure_loss_gradient=forcing,
        )
        velocity = (axial, velocity[1], velocity[2])
    u, v, w = _enforce_pipe_velocity_bc(
        *velocity,
        r_centers=r_centers,
        r_faces=r_faces,
        fluid_mask=fluid_mask,
        radial_fluid_count=radial_fluid_count,
    )
    pressure = jnp.clip(pressure + correction, -scalar_limit, scalar_limit)
    uxb = (v * btheta - w * br, w * bx - u * btheta, u * br - v * bx)
    potential, *electric_diagnostics = _solvax_pressure_poisson_pipe(
        _pipe_conservative_emf_rhs_3d(
            sigma, *uxb, dx=dx, r_faces=r_faces, r_centers=r_centers, dtheta=dtheta
        ),
        sigma,
        dx=dx,
        r_faces=r_faces,
        r_centers=r_centers,
        dtheta=dtheta,
        iterations=electric_iterations,
        tolerance=electric_tolerance,
        initial_field=potential,
        include_theta_line=True,
    )
    potential = jnp.clip(potential, -scalar_limit, scalar_limit)
    fluxes = _pipe_conservative_current_fluxes_3d(
        sigma, potential, *uxb, dx=dx, r_faces=r_faces, r_centers=r_centers, dtheta=dtheta
    )
    current = (
        0.5 * (fluxes[0][1:] + fluxes[0][:-1]),
        0.5 * (fluxes[1][:, 1:, :] + fluxes[1][:, :-1, :]),
        0.5 * (fluxes[2] + jnp.roll(fluxes[2], 1, axis=2)),
    )
    jx, jr, jtheta = (jnp.clip(value, -scalar_limit, scalar_limit) for value in current)
    lorentz = (jr * btheta - jtheta * br, jtheta * bx - jx * btheta, jx * br - jr * bx)
    div_j, _, _ = _pipe_conservative_current_diagnostics_3d(
        sigma,
        potential,
        *uxb,
        dx=dx,
        r_faces=r_faces,
        r_centers=r_centers,
        dtheta=dtheta,
        fluxes=fluxes,
    )
    projected_divergence = jnp.max(
        jnp.abs(_pipe_divergence_3d(u, v, w, dx=dx, dr=dr, dtheta=dtheta, r=radius))
    )
    return (u, v, w, pressure, potential), (
        jx,
        jr,
        jtheta,
        *lorentz,
        div_j,
        projected_divergence,
        pressure_loss,
        *electric_diagnostics,
    )


def _pipe_velocity_faces(
    u: jnp.ndarray, v: jnp.ndarray, w: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Interpolate cylindrical cell velocities to conservative faces."""

    nx, nr, ntheta = u.shape
    uf = jnp.zeros((nx + 1, nr, ntheta), dtype=u.dtype)
    uf = uf.at[1:-1].set(0.5 * (u[1:] + u[:-1]))
    uf = uf.at[0].set(u[0])
    uf = uf.at[-1].set(u[-1])
    vf = jnp.zeros((nx, nr + 1, ntheta), dtype=v.dtype)
    vf = vf.at[:, 1:-1, :].set(0.5 * (v[:, 1:, :] + v[:, :-1, :]))
    wf = 0.5 * (w + jnp.roll(w, -1, axis=2))
    return uf, vf, wf


def _pipe_face_divergence(
    uf: jnp.ndarray,
    vf: jnp.ndarray,
    wf: jnp.ndarray,
    *,
    dx: float,
    r_faces: jnp.ndarray,
    r_centers: jnp.ndarray,
    dtheta: float,
) -> jnp.ndarray:
    """Return finite-volume divergence from cylindrical face fluxes."""

    widths = jnp.diff(r_faces)
    return (
        (uf[1:] - uf[:-1]) / jnp.maximum(dx, 1.0e-12)
        + (r_faces[None, 1:, None] * vf[:, 1:, :] - r_faces[None, :-1, None] * vf[:, :-1, :])
        / jnp.maximum(r_centers[None, :, None] * widths[None, :, None], 1.0e-20)
        + (wf - jnp.roll(wf, 1, axis=2)) / jnp.maximum(r_centers[None, :, None] * dtheta, 1.0e-20)
    )


def _pipe_pressure_face_correction(
    pressure: jnp.ndarray,
    mobility: jnp.ndarray,
    *,
    dx: float,
    r_centers: jnp.ndarray,
    dtheta: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return ``-mobility grad(pressure)`` on cylindrical faces."""

    nx, nr, ntheta = pressure.shape
    correction_x = jnp.zeros((nx + 1, nr, ntheta), dtype=pressure.dtype)
    correction_x = correction_x.at[1:-1].set(
        -_harmonic_mean(mobility[1:], mobility[:-1])
        * (pressure[1:] - pressure[:-1])
        / jnp.maximum(dx, 1.0e-12)
    )
    correction_r = jnp.zeros((nx, nr + 1, ntheta), dtype=pressure.dtype)
    correction_r = correction_r.at[:, 1:-1, :].set(
        -_harmonic_mean(mobility[:, 1:, :], mobility[:, :-1, :])
        * (pressure[:, 1:, :] - pressure[:, :-1, :])
        / jnp.diff(r_centers)[None, :, None]
    )
    correction_theta = (
        -_harmonic_mean(mobility, jnp.roll(mobility, -1, axis=2))
        * (jnp.roll(pressure, -1, axis=2) - pressure)
        / jnp.maximum(r_centers[None, :, None] * dtheta, 1.0e-20)
    )
    return correction_x, correction_r, correction_theta


def _pipe_face_velocity_cells(
    uf: jnp.ndarray, vf: jnp.ndarray, wf: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Reconstruct cylindrical cell velocities from conservative faces."""

    return (
        0.5 * (uf[:-1] + uf[1:]),
        0.5 * (vf[:, :-1, :] + vf[:, 1:, :]),
        0.5 * (wf + jnp.roll(wf, 1, axis=2)),
    )


def _pipe_retained_modal_factors(
    mobility: jnp.ndarray,
    pressure_mobility: jnp.ndarray,
    cell_area: jnp.ndarray,
    momentum_coefficients: tuple[jnp.ndarray, ...],
    momentum_sink: jnp.ndarray,
    *,
    dx: float,
    r_faces: jnp.ndarray,
    r_centers: jnp.ndarray,
    dtheta: float,
    modes: tuple[int, ...] = (1, 2, 3, 4),
) -> tuple[object, tuple[object, ...]]:
    """Factor separated axisymmetric low-mode Schur blocks."""

    nx, nr, ntheta = mobility.shape
    radial_widths = jnp.diff(r_faces)
    mobility = mobility[:, :, 0]
    pressure_mobility = pressure_mobility[:, :, 0]
    cell_area = cell_area[:, :, 0]
    coefficients = tuple(coefficient[:, :, 0] for coefficient in momentum_coefficients)
    momentum_sink = momentum_sink[:, :, 0]
    face_area = jnp.concatenate((cell_area[:1], 0.5 * (cell_area[:-1] + cell_area[1:]), cell_area[-1:]))
    radial_weights = cell_area[0]

    def pressure_faces(pressure, coefficient, phase):
        axial = jnp.zeros((nx + 1, nr), dtype=pressure.dtype)
        axial = axial.at[1:-1].set(
            -_harmonic_mean(coefficient[1:], coefficient[:-1])
            * (pressure[1:] - pressure[:-1])
            / jnp.maximum(dx, 1.0e-12)
        )
        radial = jnp.zeros((nx, nr + 1), dtype=pressure.dtype)
        radial = radial.at[:, 1:-1].set(
            -_harmonic_mean(coefficient[:, 1:], coefficient[:, :-1])
            * (pressure[:, 1:] - pressure[:, :-1])
            / jnp.diff(r_centers)[None, :]
        )
        azimuthal = (
            -coefficient * (phase - 1.0) * pressure / jnp.maximum(r_centers[None, :] * dtheta, 1.0e-20)
        )
        return axial, radial, azimuthal

    def face_cells(axial, radial, azimuthal, phase):
        return (
            0.5 * (axial[:-1] + axial[1:]),
            0.5 * (radial[:, :-1] + radial[:, 1:]),
            0.5 * (1.0 + 1.0 / phase) * azimuthal,
        )

    def velocity_faces(axial, radial, azimuthal, phase):
        axial_faces = jnp.zeros((nx + 1, nr), dtype=axial.dtype)
        axial_faces = axial_faces.at[1:-1].set(0.5 * (axial[1:] + axial[:-1]))
        axial_faces = axial_faces.at[0].set(axial[0]).at[-1].set(axial[-1])
        radial_faces = jnp.zeros((nx, nr + 1), dtype=radial.dtype)
        radial_faces = radial_faces.at[:, 1:-1].set(0.5 * (radial[:, 1:] + radial[:, :-1]))
        return axial_faces, radial_faces, 0.5 * (1.0 + phase) * azimuthal

    def momentum_inverse(rhs, phase):
        coef_x_w, coef_x_e, coef_r_i, coef_r_o, coef_t_i, coef_t_o = coefficients
        diagonal = (
            coef_x_w
            + coef_x_e
            + coef_r_i
            + coef_r_o
            + coef_t_i * (1.0 - 1.0 / phase)
            + coef_t_o * (1.0 - phase)
            + momentum_sink
        ).real
        lower = -coef_r_i
        upper = -coef_r_o
        solve_args = tuple(jnp.swapaxes(array, 0, 1) for array in (lower, diagonal, upper, rhs))
        solved = tridiagonal_solve(*solve_args[:3], solve_args[3].real) + 1j * (
            tridiagonal_solve(*solve_args[:3], solve_args[3].imag)
        )
        return jnp.swapaxes(solved, 0, 1)

    def divergence(axial, radial, azimuthal, phase):
        return (
            (axial[1:] - axial[:-1]) / jnp.maximum(dx, 1.0e-12)
            + (r_faces[None, 1:] * radial[:, 1:] - r_faces[None, :-1] * radial[:, :-1])
            / jnp.maximum(r_centers[None, :] * radial_widths[None, :], 1.0e-20)
            + (1.0 - 1.0 / phase) * azimuthal / jnp.maximum(r_centers[None, :] * dtheta, 1.0e-20)
        )

    def modal_action(radial_pressure, source, phase, zero_mean):
        if zero_mean:
            final = -jnp.sum(radial_pressure * radial_weights[:-1]) / jnp.maximum(radial_weights[-1], 1.0e-20)
            radial_pressure = jnp.concatenate((radial_pressure, final[None]))
        pressure = jnp.zeros((nx, nr), dtype=jnp.result_type(mobility, 1j))
        pressure = pressure.at[source].set(radial_pressure)
        forcing_faces = pressure_faces(pressure, mobility, phase)
        forcing = face_cells(*forcing_faces, phase)
        response = tuple(momentum_inverse(force, phase) for force in forcing)

        direct_faces = pressure_faces(pressure, pressure_mobility, phase)
        reconstructed = velocity_faces(*face_cells(*direct_faces, phase), phase)
        stabilization = tuple(direct - recovered for direct, recovered in zip(direct_faces, reconstructed))
        axial = stabilization[0]
        if zero_mean:
            axial = axial - jnp.sum(axial * face_area, axis=1)[:, None] / jnp.maximum(
                jnp.sum(face_area, axis=1)[:, None], 1.0e-30
            )
        axial = axial.at[0].set(0.0).at[-1].set(0.0)
        response_faces = velocity_faces(*response, phase)
        result = divergence(
            response_faces[0] + axial,
            response_faces[1] + stabilization[1],
            response_faces[2] + stabilization[2],
            phase,
        )
        if zero_mean:
            result = result - jnp.sum(result * cell_area, axis=1)[:, None] / jnp.maximum(
                jnp.sum(cell_area, axis=1)[:, None], 1.0e-20
            )
            return result[:, :-1]
        return result

    def factor_mode(mode, size, zero_mean):
        phase = 1.0 if zero_mean else jnp.exp(2j * jnp.pi * mode / ntheta)
        jacobian = jax.jit(
            jax.jacfwd(lambda pressure, source: modal_action(pressure, source, phase, zero_mean))
        )
        actions = tuple(jacobian(jnp.zeros((size,), dtype=mobility.dtype), source) for source in range(nx))
        if zero_mean:
            actions = tuple(action.real for action in actions)
        empty = jnp.zeros((size, size), dtype=actions[0].dtype)
        blocks = (
            jnp.stack((empty, *(actions[i - 1][i] for i in range(1, nx)))),
            jnp.stack(tuple(actions[i][i] for i in range(nx))),
            jnp.stack((*(actions[i + 1][i] for i in range(nx - 1)), empty)),
        )
        return block_thomas_factor(*blocks)

    return (
        factor_mode(0, nr - 1, True),
        tuple(factor_mode(mode, nr, False) for mode in modes),
    )


def _solve_pipe_retained_modal_factors(factors, residual):
    """Solve real cosine/sine residuals with separated complex factors."""

    axisymmetric, modes = factors
    radial_size = axisymmetric[0].shape[-1]
    radial = block_thomas_solve(axisymmetric, residual[:, :radial_size])
    mode_rhs = residual[:, radial_size:].reshape((residual.shape[0], 2, len(modes), -1))
    solved = jnp.stack(
        tuple(
            block_thomas_solve(factor, mode_rhs[:, 0, i] - 1j * mode_rhs[:, 1, i])
            for i, factor in enumerate(modes)
        ),
        axis=1,
    )
    return jnp.concatenate(
        (
            radial,
            jnp.stack((solved.real, -solved.imag), axis=1).reshape((residual.shape[0], -1)),
        ),
        axis=1,
    )


def _steady_stokes_projection_pipe(
    u: jnp.ndarray,
    v: jnp.ndarray,
    w: jnp.ndarray,
    rho: jnp.ndarray,
    unit_flow_response: jnp.ndarray,
    cell_area: jnp.ndarray,
    apply_momentum_inverse: Callable[[jnp.ndarray], jnp.ndarray],
    *,
    target_flow_rate: float,
    dx: float,
    r_faces: jnp.ndarray,
    r_centers: jnp.ndarray,
    dtheta: float,
    pressure_iterations: int,
    pressure_tolerance: float,
    restart: int = 24,
    max_restarts: int = 8,
    flow_response_matrix: jnp.ndarray | None = None,
    pressure_preconditioner_mobility: jnp.ndarray | None = None,
    apply_momentum_inverse_components: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    modal_momentum_coefficients: tuple[jnp.ndarray, ...] | None = None,
    modal_momentum_sink: jnp.ndarray | None = None,
    modal_stabilization: bool = False,
    modal_factor_key: tuple[object, ...] | None = None,
    physical_tolerance: float | None = None,
) -> tuple[jnp.ndarray, ...]:
    """Apply the compatible steady ``D A^-1 G`` pipe projection."""

    response_flow = jnp.sum(unit_flow_response * cell_area, axis=(1, 2))
    area_sum = jnp.sum(cell_area, axis=(1, 2), keepdims=True)
    mobility = 1.0 / jnp.maximum(rho, 1.0e-20)
    pressure_mobility = (
        jnp.maximum(unit_flow_response, jnp.mean(unit_flow_response) * 1.0e-8)
        if pressure_preconditioner_mobility is None
        else pressure_preconditioner_mobility
    )
    cross_section_size = u.shape[1] * u.shape[2]
    pressure_size = u.shape[0] * (cross_section_size - 1)
    flat_area = cell_area.reshape((u.shape[0], cross_section_size))
    sqrt_area = jnp.sqrt(jnp.maximum(flat_area, 1.0e-30))
    gauge = sqrt_area / jnp.linalg.norm(sqrt_area, axis=1, keepdims=True)
    householder = gauge.at[:, -1].add(-1.0)
    householder_scale = 2.0 / jnp.maximum(jnp.sum(householder**2, axis=1, keepdims=True), 1.0e-30)

    def reflect(field):
        return field - householder_scale * householder * jnp.sum(householder * field, axis=1, keepdims=True)

    def unpack_pressure(reduced):
        reduced = reduced.reshape((u.shape[0], cross_section_size - 1))
        if modal_stabilization:
            transformed = reflect(
                jnp.concatenate((reduced, jnp.zeros((u.shape[0], 1), dtype=reduced.dtype)), axis=1)
            )
            return (transformed / sqrt_area).reshape(u.shape)
        final = -jnp.sum(reduced * flat_area[:, :-1], axis=1) / jnp.maximum(flat_area[:, -1], 1.0e-20)
        return jnp.concatenate((reduced, final[:, None]), axis=1).reshape(u.shape)

    def reduce_field(field):
        if modal_stabilization:
            transformed = reflect(sqrt_area * field.reshape(flat_area.shape))
            return transformed[:, :-1].reshape(-1)
        return field.reshape((u.shape[0], cross_section_size))[:, :-1].reshape(-1)

    def velocity_response(state):
        pressure = unpack_pressure(state[:pressure_size])
        pressure_loss = state[pressure_size:]
        face_force = _pipe_pressure_face_correction(
            pressure,
            mobility,
            dx=dx,
            r_centers=r_centers,
            dtheta=dtheta,
        )
        force_u, force_v, force_w = _pipe_face_velocity_cells(*face_force)
        force_u = force_u + pressure_loss[:, None, None] * mobility
        forces = jnp.stack((force_u, force_v, force_w))
        responses = (
            jnp.stack(tuple(apply_momentum_inverse(force) for force in forces))
            if apply_momentum_inverse_components is None
            else apply_momentum_inverse_components(forces)
        )
        return tuple(responses)

    def rhie_chow_faces(pressure, response):
        pressure_faces = _pipe_pressure_face_correction(
            pressure,
            pressure_mobility,
            dx=dx,
            r_centers=r_centers,
            dtheta=dtheta,
        )
        reconstructed_faces = _pipe_velocity_faces(*_pipe_face_velocity_cells(*pressure_faces))
        stabilization = tuple(
            direct - reconstructed for direct, reconstructed in zip(pressure_faces, reconstructed_faces)
        )
        face_area = jnp.concatenate(
            (
                cell_area[:1],
                0.5 * (cell_area[:-1] + cell_area[1:]),
                cell_area[-1:],
            ),
            axis=0,
        )
        axial_mean = jnp.sum(stabilization[0] * face_area, axis=(1, 2)) / jnp.maximum(
            jnp.sum(face_area, axis=(1, 2)), 1.0e-30
        )
        axial = stabilization[0] - axial_mean[:, None, None]
        axial = axial.at[0].set(0.0).at[-1].set(0.0)
        stabilization = (axial, *stabilization[1:])
        return tuple(
            exact + correction for exact, correction in zip(_pipe_velocity_faces(*response), stabilization)
        )

    def constraints(state_u, state_v, state_w, *, faces=None):
        divergence = _pipe_face_divergence(
            *(_pipe_velocity_faces(state_u, state_v, state_w) if faces is None else faces),
            dx=dx,
            r_faces=r_faces,
            r_centers=r_centers,
            dtheta=dtheta,
        )
        mean_divergence = jnp.sum(divergence * cell_area, axis=(1, 2), keepdims=True) / jnp.maximum(
            area_sum, 1.0e-20
        )
        flow = jnp.sum(state_u * cell_area, axis=(1, 2))
        return jnp.concatenate((reduce_field(divergence - mean_divergence), flow))

    base_constraints = constraints(u, v, w)
    rhs = -base_constraints.at[pressure_size:].add(-target_flow_rate)

    def schur(state):
        response = velocity_response(state)
        if not modal_stabilization:
            return constraints(*response)
        pressure = unpack_pressure(state[:pressure_size])
        return constraints(*response, faces=rhie_chow_faces(pressure, response))

    def local_precondition(residual):
        divergence = unpack_pressure(residual[:pressure_size])
        pressure, *_ = _solvax_pressure_poisson_pipe(
            -divergence,
            pressure_mobility,
            dx=dx,
            r_faces=r_faces,
            r_centers=r_centers,
            dtheta=dtheta,
            iterations=pressure_iterations,
            tolerance=pressure_tolerance,
            include_theta_line=True,
        )
        flow_residual = residual[pressure_size:]
        pressure_loss = (
            jnp.linalg.solve(flow_response_matrix, flow_residual)
            if flow_response_matrix is not None
            else flow_residual / jnp.maximum(jnp.mean(response_flow), 1.0e-20)
        )
        return jnp.concatenate((reduce_field(pressure), pressure_loss))

    if modal_stabilization:
        if modal_momentum_coefficients is None or modal_momentum_sink is None:
            raise ValueError("modal stabilization requires retained-modal coefficients and sink")
        radial_weights = jnp.sum(cell_area, axis=2)
        coarse_pressure_size = u.shape[0] * (u.shape[1] - 1)
        modal_modes = (1, 2, 3, 4)
        mode_size = 2 * len(modal_modes) * u.shape[0] * u.shape[1]
        coarse_size = coarse_pressure_size + mode_size + u.shape[0]
        theta = jnp.arange(u.shape[2], dtype=u.dtype) * dtheta
        mode_angles = jnp.asarray(modal_modes, dtype=u.dtype)[:, None] * theta
        mode_cosine = jnp.cos(mode_angles)
        mode_sine = jnp.sin(mode_angles)

        def prolong(coarse):
            radial = coarse[:coarse_pressure_size].reshape((u.shape[0], u.shape[1] - 1))
            final = -jnp.sum(radial * radial_weights[:, :-1], axis=1) / jnp.maximum(
                radial_weights[:, -1], 1.0e-20
            )
            pressure = jnp.broadcast_to(
                jnp.concatenate((radial, final[:, None]), axis=1)[:, :, None],
                u.shape,
            )
            offset = coarse_pressure_size
            modes = coarse[offset : offset + mode_size].reshape((2, len(modal_modes), u.shape[0], u.shape[1]))
            pressure = (
                pressure
                + jnp.einsum("mxr,mt->xrt", modes[0], mode_cosine)
                + jnp.einsum("mxr,mt->xrt", modes[1], mode_sine)
            )
            return jnp.concatenate(
                (
                    reduce_field(pressure),
                    coarse[coarse_pressure_size + mode_size :],
                )
            )

        def restrict(residual):
            divergence = unpack_pressure(residual[:pressure_size])
            return jnp.concatenate(
                (
                    jnp.mean(divergence, axis=2)[:, :-1].reshape(-1),
                    (
                        2.0
                        / u.shape[2]
                        * jnp.stack(
                            (
                                jnp.einsum("xrt,mt->mxr", divergence, mode_cosine),
                                jnp.einsum("xrt,mt->mxr", divergence, mode_sine),
                            )
                        )
                    ).reshape(-1),
                    residual[pressure_size:],
                )
            )

        def modal_restrict(residual):
            coarse = restrict(residual)
            radial = coarse[:coarse_pressure_size].reshape((u.shape[0], u.shape[1] - 1))
            modes = coarse[coarse_pressure_size : coarse_pressure_size + mode_size].reshape(
                (2, len(modal_modes), u.shape[0], u.shape[1])
            )
            return jnp.concatenate(
                (radial, jnp.transpose(modes, (2, 0, 1, 3)).reshape((u.shape[0], -1))),
                axis=1,
            )

        def build_modal_factors():
            return _pipe_retained_modal_factors(
                mobility,
                pressure_mobility,
                cell_area,
                modal_momentum_coefficients,
                modal_momentum_sink,
                dx=dx,
                r_faces=r_faces,
                r_centers=r_centers,
                dtheta=dtheta,
                modes=modal_modes,
            )

        modal_factors = (
            build_modal_factors()
            if modal_factor_key is None
            else _reuse_modal_factors(modal_factor_key, build_modal_factors)
        )

        def modal_prolong(local):
            coarse = jnp.zeros((coarse_size,), dtype=u.dtype)
            coarse = coarse.at[:coarse_pressure_size].set(local[:, : u.shape[1] - 1].reshape(-1))
            modes = local[:, u.shape[1] - 1 :].reshape((u.shape[0], 2, len(modal_modes), u.shape[1]))
            coarse = coarse.at[coarse_pressure_size : coarse_pressure_size + mode_size].set(
                jnp.transpose(modes, (1, 2, 0, 3)).reshape(-1)
            )
            return prolong(coarse)

        def precondition(residual):
            local = local_precondition(residual)
            modal_residual = modal_restrict(residual - schur(local))
            modal_correction = _solve_pipe_retained_modal_factors(modal_factors, modal_residual)
            candidate = local + modal_prolong(modal_correction)
            flow_residual = (residual - schur(candidate))[pressure_size:]
            flow_response = (
                jnp.linalg.solve(flow_response_matrix, flow_residual)
                if flow_response_matrix is not None
                else flow_residual / jnp.maximum(jnp.mean(response_flow), 1.0e-20)
            )
            return candidate + jnp.zeros_like(candidate).at[pressure_size:].set(flow_response)

    else:
        precondition = local_precondition

    if modal_factor_key is not None:
        pressure_kernel_key = (
            jax.default_backend(),
            modal_factor_key,
            pressure_iterations,
            pressure_tolerance,
        )
        schur = _reuse_fringing_jit(
            ("b1_steady_schur", *pressure_kernel_key),
            jax.jit(schur, inline=False),
        )
        precondition = _reuse_fringing_jit(
            ("b1_steady_preconditioner", *pressure_kernel_key),
            jax.jit(precondition, inline=False),
        )

    def physical_constraint_residual(operator, state, linear_rhs):
        residual = operator(state) - linear_rhs
        divergence = unpack_pressure(residual[:pressure_size])
        flow = residual[pressure_size:] / jnp.maximum(jnp.mean(area_sum), 1.0e-30)
        return jnp.maximum(jnp.max(jnp.abs(divergence)), jnp.max(jnp.abs(flow)))

    def solve_operator(operator, linear_rhs, initial, operator_precondition=precondition):
        if physical_tolerance is None:
            return gmres(
                operator,
                linear_rhs,
                x0=initial,
                precond=operator_precondition,
                restart=restart,
                rtol=pressure_tolerance,
                atol=pressure_tolerance,
                max_restarts=max_restarts,
            )
        pilot_tolerance = max(pressure_tolerance, min(1.0e-6, physical_tolerance))
        pilot = gmres(
            operator,
            linear_rhs,
            x0=initial,
            precond=operator_precondition,
            restart=restart,
            rtol=pilot_tolerance,
            atol=pilot_tolerance,
            max_restarts=1,
        )
        pilot_passes = physical_constraint_residual(operator, pilot.x, linear_rhs) <= physical_tolerance

        def accept_pilot(_):
            return pilot

        def refine_pilot(_):
            refined = gmres(
                operator,
                linear_rhs,
                x0=pilot.x,
                precond=operator_precondition,
                restart=restart,
                rtol=pressure_tolerance,
                atol=pressure_tolerance,
                max_restarts=max_restarts,
            )
            return KrylovSolution(
                refined.x,
                refined.residual_norm,
                pilot.iterations + refined.iterations,
                refined.converged,
                refined.recycle,
            )

        return jax.lax.cond(pilot_passes, accept_pilot, refine_pilot, operand=None)

    def solve_pressure(linear_rhs, initial):
        return solve_operator(schur, linear_rhs, initial)

    if modal_factor_key is not None:
        solve_pressure = _reuse_fringing_jit(
            (
                "b1_steady_pressure",
                jax.default_backend(),
                modal_factor_key,
                pressure_iterations,
                pressure_tolerance,
                restart,
                max_restarts,
                physical_tolerance,
            ),
            jax.jit(solve_pressure),
        )

    def implicit_solver(_, linear_rhs):
        solution = solve_pressure(linear_rhs, precondition(linear_rhs))
        return solution.x, (solution.residual_norm, solution.iterations, solution.converged)

    def implicit_transpose_solver(operator, linear_rhs):
        transpose = jax.linear_transpose(precondition, jnp.zeros_like(linear_rhs))

        def transpose_precondition(value):
            return transpose(value)[0]

        initial = transpose_precondition(linear_rhs)
        solution = solve_operator(operator, linear_rhs, initial, transpose_precondition)
        return solution.x, (solution.residual_norm, solution.iterations, solution.converged)

    pressure_state, pressure_aux = linear_solve(
        schur,
        rhs,
        implicit_solver,
        transpose_solver=implicit_transpose_solver,
        has_aux=True,
    )
    pressure_solution = KrylovSolution(pressure_state, *pressure_aux)
    pressure = unpack_pressure(pressure_solution.x[:pressure_size])
    pressure_loss = pressure_solution.x[pressure_size:]
    correction = velocity_response(pressure_solution.x)
    projected = tuple(base + delta for base, delta in zip((u, v, w), correction))
    if flow_response_matrix is not None:
        flow_delta = target_flow_rate - jnp.sum(projected[0] * cell_area, axis=(1, 2))
        flow_refinement_state = (
            jnp.zeros_like(pressure_solution.x)
            .at[pressure_size:]
            .set(jnp.linalg.solve(flow_response_matrix, flow_delta))
        )
        flow_refinement = velocity_response(flow_refinement_state)
        correction = tuple(base + delta for base, delta in zip(correction, flow_refinement))
        projected = tuple(base + delta for base, delta in zip((u, v, w), correction))
        pressure_loss = pressure_loss + flow_refinement_state[pressure_size:]
        refined_state = pressure_solution.x + flow_refinement_state
        refined_residual = jnp.linalg.norm(schur(refined_state) - rhs)
        pressure_solution = KrylovSolution(
            refined_state,
            refined_residual,
            pressure_solution.iterations,
            refined_residual <= pressure_tolerance,
            pressure_solution.recycle,
        )
    final_flow = jnp.sum(projected[0] * cell_area, axis=(1, 2))
    projected_faces = _pipe_velocity_faces(*projected)
    if modal_stabilization:
        response_faces = rhie_chow_faces(pressure, correction)
        projected_faces = tuple(
            base + response for base, response in zip(_pipe_velocity_faces(u, v, w), response_faces)
        )
    final_divergence = _pipe_face_divergence(
        *projected_faces,
        dx=dx,
        r_faces=r_faces,
        r_centers=r_centers,
        dtheta=dtheta,
    )
    final_mean_divergence = jnp.sum(final_divergence * cell_area, axis=(1, 2), keepdims=True) / jnp.maximum(
        area_sum, 1.0e-30
    )
    final_mean_free_divergence = final_divergence - final_mean_divergence
    divergence_residual = jnp.max(jnp.abs(final_mean_free_divergence))
    flow_residual = jnp.max(jnp.abs(final_flow - target_flow_rate))
    normalized_flow_residual = flow_residual / jnp.maximum(jnp.mean(area_sum), 1.0e-30)
    physical_residual = jnp.maximum(divergence_residual, normalized_flow_residual)
    convergence_tolerance = pressure_tolerance if physical_tolerance is None else physical_tolerance
    pressure_solution = KrylovSolution(
        pressure_solution.x,
        physical_residual,
        pressure_solution.iterations,
        physical_residual <= convergence_tolerance,
        pressure_solution.recycle,
    )
    return (
        *projected,
        pressure,
        pressure_loss,
        divergence_residual,
        flow_residual,
        pressure_solution,
    )


def _pipe_conservative_current_fluxes_3d(
    sigma: jnp.ndarray,
    phi: jnp.ndarray,
    uxb_x: jnp.ndarray,
    uxb_r: jnp.ndarray,
    uxb_theta: jnp.ndarray,
    *,
    dx: float,
    r_faces: jnp.ndarray,
    r_centers: jnp.ndarray,
    dtheta: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    nx, nr, ntheta = phi.shape
    fx = jnp.zeros((nx + 1, nr, ntheta), dtype=phi.dtype)
    fr = jnp.zeros((nx, nr + 1, ntheta), dtype=phi.dtype)
    ftheta = jnp.zeros((nx, nr, ntheta), dtype=phi.dtype)

    sigma_x = _harmonic_mean(sigma[1:], sigma[:-1])
    phi_grad_x = (phi[1:] - phi[:-1]) / jnp.maximum(dx, 1.0e-12)
    uxb_face_x = 0.5 * (uxb_x[1:] + uxb_x[:-1])
    fx = fx.at[1:-1].set(sigma_x * (-phi_grad_x + uxb_face_x))

    dr_centers = jnp.maximum(0.5 * (jnp.diff(r_faces)[1:] + jnp.diff(r_faces)[:-1]), 1.0e-12)
    sigma_r = _harmonic_mean(sigma[:, 1:, :], sigma[:, :-1, :])
    phi_grad_r = (phi[:, 1:, :] - phi[:, :-1, :]) / dr_centers[None, :, None]
    uxb_face_r = 0.5 * (uxb_r[:, 1:, :] + uxb_r[:, :-1, :])
    fr = fr.at[:, 1:-1, :].set(sigma_r * (-phi_grad_r + uxb_face_r))

    sigma_theta = _harmonic_mean(sigma, jnp.roll(sigma, -1, axis=2))
    phi_grad_theta = (jnp.roll(phi, -1, axis=2) - phi) / jnp.maximum(
        r_centers[None, :, None] * dtheta, 1.0e-12
    )
    uxb_face_theta = 0.5 * (uxb_theta + jnp.roll(uxb_theta, -1, axis=2))
    ftheta = sigma_theta * (-phi_grad_theta + uxb_face_theta)
    return fx, fr, ftheta


def _pipe_conservative_current_diagnostics_3d(
    sigma: jnp.ndarray,
    phi: jnp.ndarray,
    uxb_x: jnp.ndarray,
    uxb_r: jnp.ndarray,
    uxb_theta: jnp.ndarray,
    *,
    dx: float,
    r_faces: jnp.ndarray,
    r_centers: jnp.ndarray,
    dtheta: float,
    fluxes: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    if fluxes is None:
        fluxes = _pipe_conservative_current_fluxes_3d(
            sigma,
            phi,
            uxb_x,
            uxb_r,
            uxb_theta,
            dx=dx,
            r_faces=r_faces,
            r_centers=r_centers,
            dtheta=dtheta,
        )
    fx, fr, ftheta = fluxes
    dr = jnp.diff(r_faces)
    radial_term = (
        r_faces[None, 1:, None] * fr[:, 1:, :] - r_faces[None, :-1, None] * fr[:, :-1, :]
    ) / jnp.maximum(r_centers[None, :, None] * dr[None, :, None], 1.0e-12)
    theta_term = (ftheta - jnp.roll(ftheta, 1, axis=2)) / jnp.maximum(
        r_centers[None, :, None] * dtheta, 1.0e-12
    )
    div_j = (fx[1:] - fx[:-1]) / jnp.maximum(dx, 1.0e-12) + radial_term + theta_term
    wall_area = r_faces[-1] * dx * dtheta
    wall_leakage = jnp.sum(jnp.abs(fr[:, -1, :]) * wall_area, axis=1)
    yz_area = r_centers[:, None] * dr[:, None] * dtheta
    boundary_residual = jnp.abs(
        -jnp.sum(fx[0] * yz_area) + jnp.sum(fx[-1] * yz_area) + jnp.sum(fr[:, -1, :] * wall_area, axis=1)
    )
    return div_j, wall_leakage, boundary_residual


def _pipe_conservative_emf_rhs_3d(
    sigma: jnp.ndarray,
    uxb_x: jnp.ndarray,
    uxb_r: jnp.ndarray,
    uxb_theta: jnp.ndarray,
    *,
    dx: float,
    r_faces: jnp.ndarray,
    r_centers: jnp.ndarray,
    dtheta: float,
) -> jnp.ndarray:
    zeros = jnp.zeros_like(uxb_x)
    fx, fr, ftheta = _pipe_conservative_current_fluxes_3d(
        sigma,
        zeros,
        uxb_x,
        uxb_r,
        uxb_theta,
        dx=dx,
        r_faces=r_faces,
        r_centers=r_centers,
        dtheta=dtheta,
    )
    dr = jnp.diff(r_faces)
    radial_term = (
        r_faces[None, 1:, None] * fr[:, 1:, :] - r_faces[None, :-1, None] * fr[:, :-1, :]
    ) / jnp.maximum(r_centers[None, :, None] * dr[None, :, None], 1.0e-12)
    theta_term = (ftheta - jnp.roll(ftheta, 1, axis=2)) / jnp.maximum(
        r_centers[None, :, None] * dtheta, 1.0e-12
    )
    return (fx[1:] - fx[:-1]) / jnp.maximum(dx, 1.0e-12) + radial_term + theta_term


def _enforce_pipe_velocity_bc(
    u: jnp.ndarray,
    v: jnp.ndarray,
    w: jnp.ndarray,
    *,
    r_centers: jnp.ndarray | None = None,
    r_faces: jnp.ndarray | None = None,
    fluid_mask: jnp.ndarray | None = None,
    radial_fluid_count: int | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    if u.shape[1] > 1:
        u = u.at[:, 0, :].set(u[:, 1, :])
        w = w.at[:, 0, :].set(w[:, 1, :])
    v = v.at[:, 0, :].set(0.0)
    if fluid_mask is not None:
        active = jnp.asarray(fluid_mask, dtype=bool)
        u = jnp.where(active, u, 0.0)
        v = jnp.where(active, v, 0.0)
        w = jnp.where(active, w, 0.0)
        interface_index = (
            int(jnp.sum(jnp.any(active, axis=(0, 2)))) if radial_fluid_count is None else radial_fluid_count
        ) - 1
        if interface_index > 0 and r_centers is not None and r_faces is not None:
            fluid_radius = r_faces[interface_index + 1]
            ratio = (
                0.9
                * (fluid_radius - r_centers[interface_index])
                / jnp.maximum(fluid_radius - r_centers[interface_index - 1], 1.0e-12)
            )
            u = u.at[:, interface_index, :].set(ratio * u[:, interface_index - 1, :])
            w = w.at[:, interface_index, :].set(ratio * w[:, interface_index - 1, :])
            v = v.at[:, interface_index, :].set(0.0)
    elif r_centers is not None and r_faces is not None and u.shape[1] > 1:
        outer_ratio = 0.9 * (r_faces[-1] - r_centers[-1]) / jnp.maximum(r_faces[-1] - r_centers[-2], 1.0e-12)
        u = u.at[:, -1, :].set(outer_ratio * u[:, -2, :])
        w = w.at[:, -1, :].set(outer_ratio * w[:, -2, :])
    else:
        u = u.at[:, -1, :].set(0.0)
        w = w.at[:, -1, :].set(0.0)
    v = v.at[:, -1, :].set(0.0)
    if u.shape[0] > 1:
        u = u.at[0, :, :].set(u[1, :, :])
        u = u.at[-1, :, :].set(u[-2, :, :])
        v = v.at[0, :, :].set(v[1, :, :])
        v = v.at[-1, :, :].set(v[-2, :, :])
        w = w.at[0, :, :].set(w[1, :, :])
        w = w.at[-1, :, :].set(w[-2, :, :])
    return u, v, w

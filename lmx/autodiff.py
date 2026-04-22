from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .field_models import wham_mirror_station_scale
from .linear import solve_poisson_jacobi_state
from .mesh import StructuredMesh, generate_rect_duct_mesh
from .operators import gradient_scalar
from .solvers import (
    _enforce_velocity_bc,
    _face_emf_y,
    _face_emf_z,
    _fully_developed_rhs,
    _potential_coefficients,
    _velocity_system_coefficients,
)


@dataclass(frozen=True)
class FringingAutodiffProblem:
    base_problem: HartmannAutodiffProblem
    x: jnp.ndarray


@dataclass(frozen=True)
class HartmannAutodiffProblem:
    mesh: StructuredMesh
    sigma: jnp.ndarray
    rho: jnp.ndarray
    nu: jnp.ndarray
    fluid_mask: jnp.ndarray
    anchor: tuple[int, int] = (0, 0)
    potential_iterations: int = 80
    velocity_iterations: int = 120
    macro_iterations: int = 8
    relaxation: float = 0.9


def build_hartmann_autodiff_problem(
    *,
    ny: int = 48,
    nz: int = 48,
    width: float = 2.0,
    height: float = 2.0,
    conductivity: float = 1.0,
    density: float = 1.0,
    viscosity: float = 0.01,
    potential_iterations: int = 80,
    velocity_iterations: int = 120,
    macro_iterations: int = 8,
    relaxation: float = 0.9,
) -> HartmannAutodiffProblem:
    mesh = generate_rect_duct_mesh(width=width, height=height, ny=ny, nz=nz)
    yz_shape = mesh.yz_shape
    return HartmannAutodiffProblem(
        mesh=mesh,
        sigma=jnp.full(yz_shape, conductivity),
        rho=jnp.full(yz_shape, density),
        nu=jnp.full(yz_shape, viscosity),
        fluid_mask=jnp.ones(yz_shape, dtype=bool),
        potential_iterations=potential_iterations,
        velocity_iterations=velocity_iterations,
        macro_iterations=macro_iterations,
        relaxation=relaxation,
    )


def build_fringing_autodiff_problem(
    *,
    nx_stations: int = 15,
    length: float = 6.0,
    **kwargs,
) -> FringingAutodiffProblem:
    return FringingAutodiffProblem(
        base_problem=build_hartmann_autodiff_problem(**kwargs),
        x=jnp.linspace(0.0, length, nx_stations),
    )


def _smooth_fringing_scale(
    x: jnp.ndarray,
    *,
    entry_center: float | jnp.ndarray,
    exit_center: float | jnp.ndarray,
    transition_width: float | jnp.ndarray,
    peak_scale: float | jnp.ndarray = 1.0,
) -> jnp.ndarray:
    width = jnp.maximum(jnp.asarray(transition_width), 1.0e-6)
    rise = 0.5 * (1.0 + jnp.tanh((x - jnp.asarray(entry_center)) / width))
    fall = 0.5 * (1.0 - jnp.tanh((x - jnp.asarray(exit_center)) / width))
    return jnp.asarray(peak_scale) * rise * fall


def _extruded_rect_response_history_from_field_scale(
    problem: FringingAutodiffProblem,
    *,
    forcing: float | jnp.ndarray,
    peak_hartmann_number: float | jnp.ndarray,
    field_scale: jnp.ndarray,
) -> dict[str, jnp.ndarray]:
    mesh = problem.base_problem.mesh
    dy = float(jnp.mean(mesh.dy))
    dz = float(jnp.mean(mesh.dz))
    scale = jnp.asarray(field_scale, dtype=problem.base_problem.sigma.dtype)
    bz = jnp.broadcast_to(jnp.asarray(peak_hartmann_number) * scale[:, None, None], (scale.shape[0], *mesh.yz_shape))
    sigma = jnp.broadcast_to(problem.base_problem.sigma[None, :, :], bz.shape)
    forcing_value = jnp.asarray(forcing, dtype=sigma.dtype)

    def station_response(bz_slice: jnp.ndarray) -> dict[str, jnp.ndarray]:
        u, phi = solve_differentiable_hartmann(problem.base_problem, forcing=forcing_value, hartmann_number=bz_slice[0, 0])
        dphi_dy, dphi_dz = gradient_scalar(phi, mesh)
        uxb_y = -u * bz_slice
        jx = jnp.zeros_like(u)
        jy = sigma[0] * (-dphi_dy + uxb_y)
        jz = sigma[0] * (-dphi_dz)
        div_j = jnp.gradient(jy, float(jnp.mean(mesh.dy)), axis=0) + jnp.gradient(jz, float(jnp.mean(mesh.dz)), axis=1)
        boundary_current_residual = jnp.abs(
            -jnp.sum(jy[0, :]) * dz
            + jnp.sum(jy[-1, :]) * dz
            - jnp.sum(jz[:, 0]) * dy
            + jnp.sum(jz[:, -1]) * dy
        )
        return {
            "mean_velocity": jnp.mean(u),
            "current_proxy": jnp.mean(jnp.abs(jy)),
            "charge_balance_residual": jnp.max(jnp.abs(div_j)),
            "boundary_current_residual": boundary_current_residual,
        }

    payload = jax.vmap(station_response)(bz)
    return {
        "x": problem.x,
        "field_scale": scale,
        **payload,
    }


def _extruded_rect_projection_history_from_field_scale(
    problem: FringingAutodiffProblem,
    *,
    forcing: float | jnp.ndarray,
    peak_hartmann_number: float | jnp.ndarray,
    field_scale: jnp.ndarray,
) -> dict[str, jnp.ndarray]:
    mesh = problem.base_problem.mesh
    ny, nz = mesh.yz_shape
    nx = int(problem.x.shape[0])
    dx = float((problem.x[-1] - problem.x[0]) / max(nx - 1, 1)) if nx > 1 else 1.0
    dy = float(jnp.mean(mesh.dy))
    dz = float(jnp.mean(mesh.dz))
    sigma = jnp.broadcast_to(problem.base_problem.sigma[None, :, :], (nx, ny, nz))
    rho = jnp.broadcast_to(problem.base_problem.rho[None, :, :], (nx, ny, nz))
    nu = jnp.broadcast_to(problem.base_problem.nu[None, :, :], (nx, ny, nz))
    scale = jnp.asarray(field_scale, dtype=sigma.dtype)
    bz = jnp.broadcast_to(jnp.asarray(peak_hartmann_number) * scale[:, None, None], (nx, ny, nz))
    forcing_value = jnp.asarray(forcing, dtype=sigma.dtype)
    inverse_diffusive_scale = jnp.max(nu) * (1.0 / max(dx**2, 1.0e-12) + 1.0 / max(dy**2, 1.0e-12) + 1.0 / max(dz**2, 1.0e-12))
    dt = jnp.asarray(0.2) / jnp.maximum(inverse_diffusive_scale, 1.0e-12)
    pressure_iterations = max(4, problem.base_problem.potential_iterations // 2)

    def macro_body(_, state):
        u, v, w, p, phi = state
        dphi_dx, dphi_dy, dphi_dz = _extruded_gradient(phi, dx=dx, dy=dy, dz=dz)
        uxb_x = v * bz
        uxb_y = -u * bz
        uxb_z = jnp.zeros_like(u)
        jx = sigma * (-dphi_dx + uxb_x)
        jy = sigma * (-dphi_dy + uxb_y)
        lorentz_x = jy * bz
        lorentz_y = -jx * bz
        lorentz_z = jnp.zeros_like(u)

        dp_dx, dp_dy, dp_dz = _extruded_gradient(p, dx=dx, dy=dy, dz=dz)
        u_star = _extruded_enforce_velocity_bc(
            u + dt * (nu * _extruded_laplacian(u, dx=dx, dy=dy, dz=dz) + forcing_value / jnp.maximum(rho, 1.0e-12) + lorentz_x / jnp.maximum(rho, 1.0e-12) - dp_dx / jnp.maximum(rho, 1.0e-12))
        )
        v_star = _extruded_enforce_velocity_bc(
            v + dt * (nu * _extruded_laplacian(v, dx=dx, dy=dy, dz=dz) + lorentz_y / jnp.maximum(rho, 1.0e-12) - dp_dy / jnp.maximum(rho, 1.0e-12))
        )
        w_star = _extruded_enforce_velocity_bc(
            w + dt * (nu * _extruded_laplacian(w, dx=dx, dy=dy, dz=dz) + lorentz_z / jnp.maximum(rho, 1.0e-12) - dp_dz / jnp.maximum(rho, 1.0e-12))
        )

        du_dx, _, _ = _extruded_gradient(u_star, dx=dx, dy=dy, dz=dz)
        _, dv_dy, _ = _extruded_gradient(v_star, dx=dx, dy=dy, dz=dz)
        _, _, dw_dz = _extruded_gradient(w_star, dx=dx, dy=dy, dz=dz)
        divergence = du_dx + dv_dy + dw_dz
        p_corr = _extruded_poisson_jacobi((rho / jnp.maximum(dt, 1.0e-12)) * divergence, dx=dx, dy=dy, dz=dz, iterations=pressure_iterations)
        dpc_dx, dpc_dy, dpc_dz = _extruded_gradient(p_corr, dx=dx, dy=dy, dz=dz)
        u_next = _extruded_enforce_velocity_bc(u_star - (dt / jnp.maximum(rho, 1.0e-12)) * dpc_dx)
        v_next = _extruded_enforce_velocity_bc(v_star - (dt / jnp.maximum(rho, 1.0e-12)) * dpc_dy)
        w_next = _extruded_enforce_velocity_bc(w_star - (dt / jnp.maximum(rho, 1.0e-12)) * dpc_dz)
        p_next = p + p_corr

        uxb_x = v_next * bz
        uxb_y = -u_next * bz
        uxb_z = jnp.zeros_like(u_next)
        rhs_phi = _extruded_conservative_emf_rhs(sigma, uxb_x, uxb_y, uxb_z, dx=dx, dy=dy, dz=dz)
        phi_next = _extruded_poisson_jacobi(
            rhs_phi,
            dx=dx,
            dy=dy,
            dz=dz,
            iterations=problem.base_problem.potential_iterations,
        )
        return u_next, v_next, w_next, p_next, phi_next

    initial_state = (
        jnp.zeros((nx, ny, nz), dtype=problem.base_problem.sigma.dtype),
        jnp.zeros((nx, ny, nz), dtype=problem.base_problem.sigma.dtype),
        jnp.zeros((nx, ny, nz), dtype=problem.base_problem.sigma.dtype),
        jnp.zeros((nx, ny, nz), dtype=problem.base_problem.sigma.dtype),
        jnp.zeros((nx, ny, nz), dtype=problem.base_problem.sigma.dtype),
    )
    u, v, w, p, phi = jax.lax.fori_loop(0, problem.base_problem.macro_iterations, macro_body, initial_state)

    dphi_dx, dphi_dy, dphi_dz = _extruded_gradient(phi, dx=dx, dy=dy, dz=dz)
    uxb_x = v * bz
    uxb_y = -u * bz
    uxb_z = jnp.zeros_like(u)
    jx = sigma * (-dphi_dx + uxb_x)
    jy = sigma * (-dphi_dy + uxb_y)
    fx, fy, fz = _extruded_conservative_current_fluxes(sigma, phi, uxb_x, uxb_y, uxb_z, dx=dx, dy=dy, dz=dz)
    div_j = (
        (fx[1:] - fx[:-1]) / max(dx, 1.0e-12)
        + (fy[:, 1:, :] - fy[:, :-1, :]) / max(dy, 1.0e-12)
        + (fz[:, :, 1:] - fz[:, :, :-1]) / max(dz, 1.0e-12)
    )
    boundary_current_residual = jnp.abs(
        -jnp.sum(fx[0], axis=(0, 1)) * dy * dz
        + jnp.sum(fx[-1], axis=(0, 1)) * dy * dz
        - jnp.sum(fy[:, 0, :], axis=1) * dx * dz
        + jnp.sum(fy[:, -1, :], axis=1) * dx * dz
        - jnp.sum(fz[:, :, 0], axis=1) * dx * dy
        + jnp.sum(fz[:, :, -1], axis=1) * dx * dy
    )
    wall_current_leakage = (
        jnp.sum(jnp.abs(fy[:, 0, :]), axis=1) * dx * dz
        + jnp.sum(jnp.abs(fy[:, -1, :]), axis=1) * dx * dz
        + jnp.sum(jnp.abs(fz[:, :, 0]), axis=1) * dx * dy
        + jnp.sum(jnp.abs(fz[:, :, -1]), axis=1) * dx * dy
    )
    pressure_span = jnp.max(p, axis=(1, 2)) - jnp.min(p, axis=(1, 2))
    transverse_kinetic_energy = jnp.mean(v**2 + w**2, axis=(1, 2))
    return {
        "x": problem.x,
        "field_scale": scale,
        "mean_velocity": jnp.mean(u, axis=(1, 2)),
        "current_proxy": jnp.mean(jnp.abs(jy), axis=(1, 2)),
        "charge_balance_residual": jnp.max(jnp.abs(div_j), axis=(1, 2)),
        "boundary_current_residual": boundary_current_residual,
        "pressure_span": pressure_span,
        "transverse_kinetic_energy": transverse_kinetic_energy,
        "wall_current_leakage": wall_current_leakage,
        "axial_current": jnp.sum(jx, axis=(1, 2)) * dy * dz,
        "u_field": u,
        "v_field": v,
        "w_field": w,
        "pressure_field": p,
        "phi_field": phi,
        "jy_field": jy,
    }


def _extruded_neighbor_fields(field: jnp.ndarray) -> tuple[jnp.ndarray, ...]:
    x_west = jnp.concatenate([field[:1], field[:-1]], axis=0)
    x_east = jnp.concatenate([field[1:], field[-1:]], axis=0)
    y_south = jnp.concatenate([jnp.zeros_like(field[:, :1, :]), field[:, :-1, :]], axis=1)
    y_north = jnp.concatenate([field[:, 1:, :], jnp.zeros_like(field[:, -1:, :])], axis=1)
    z_bottom = jnp.concatenate([jnp.zeros_like(field[:, :, :1]), field[:, :, :-1]], axis=2)
    z_top = jnp.concatenate([field[:, :, 1:], jnp.zeros_like(field[:, :, -1:])], axis=2)
    return x_west, x_east, y_south, y_north, z_bottom, z_top


def _extruded_gradient(field: jnp.ndarray, *, dx: float, dy: float, dz: float) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    x_west, x_east, y_south, y_north, z_bottom, z_top = _extruded_neighbor_fields(field)
    d_dx = (x_east - x_west) / max(2.0 * dx, 1.0e-12)
    d_dy = (y_north - y_south) / max(2.0 * dy, 1.0e-12)
    d_dz = (z_top - z_bottom) / max(2.0 * dz, 1.0e-12)
    return d_dx, d_dy, d_dz


def _extruded_laplacian(field: jnp.ndarray, *, dx: float, dy: float, dz: float) -> jnp.ndarray:
    x_west, x_east, y_south, y_north, z_bottom, z_top = _extruded_neighbor_fields(field)
    return (
        (x_west - 2.0 * field + x_east) / max(dx**2, 1.0e-12)
        + (y_south - 2.0 * field + y_north) / max(dy**2, 1.0e-12)
        + (z_bottom - 2.0 * field + z_top) / max(dz**2, 1.0e-12)
    )


def _extruded_enforce_velocity_bc(field: jnp.ndarray) -> jnp.ndarray:
    bounded = field.at[:, 0, :].set(0.0)
    bounded = bounded.at[:, -1, :].set(0.0)
    bounded = bounded.at[:, :, 0].set(0.0)
    bounded = bounded.at[:, :, -1].set(0.0)
    bounded = bounded.at[0, :, :].set(bounded[1, :, :]) if bounded.shape[0] > 1 else bounded
    bounded = bounded.at[-1, :, :].set(bounded[-2, :, :]) if bounded.shape[0] > 1 else bounded
    return bounded


def _extruded_poisson_jacobi(rhs: jnp.ndarray, *, dx: float, dy: float, dz: float, iterations: int) -> jnp.ndarray:
    rhs_compatible = rhs - jnp.mean(rhs)
    diagonal = 2.0 / max(dx**2, 1.0e-12) + 2.0 / max(dy**2, 1.0e-12) + 2.0 / max(dz**2, 1.0e-12)

    def body_fun(_, field):
        x_west, x_east, y_south, y_north, z_bottom, z_top = _extruded_neighbor_fields(field)
        updated = (
            (x_west + x_east) / max(dx**2, 1.0e-12)
            + (y_south + y_north) / max(dy**2, 1.0e-12)
            + (z_bottom + z_top) / max(dz**2, 1.0e-12)
            - rhs_compatible
        ) / diagonal
        return updated - jnp.mean(updated)

    return jax.lax.fori_loop(0, iterations, body_fun, jnp.zeros_like(rhs_compatible))


def _extruded_harmonic_mean(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    denom = jnp.maximum(a + b, 1.0e-20)
    return 2.0 * a * b / denom


def _extruded_conservative_current_fluxes(
    sigma: jnp.ndarray,
    phi: jnp.ndarray,
    uxb_x: jnp.ndarray,
    uxb_y: jnp.ndarray,
    uxb_z: jnp.ndarray,
    *,
    dx: float,
    dy: float,
    dz: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    nx, ny, nz = phi.shape
    fx = jnp.zeros((nx + 1, ny, nz), dtype=phi.dtype)
    fy = jnp.zeros((nx, ny + 1, nz), dtype=phi.dtype)
    fz = jnp.zeros((nx, ny, nz + 1), dtype=phi.dtype)

    sigma_x = _extruded_harmonic_mean(sigma[1:], sigma[:-1])
    phi_grad_x = (phi[1:] - phi[:-1]) / max(dx, 1.0e-12)
    uxb_face_x = 0.5 * (uxb_x[1:] + uxb_x[:-1])
    fx = fx.at[1:-1].set(sigma_x * (-phi_grad_x + uxb_face_x))

    sigma_y = _extruded_harmonic_mean(sigma[:, 1:, :], sigma[:, :-1, :])
    phi_grad_y = (phi[:, 1:, :] - phi[:, :-1, :]) / max(dy, 1.0e-12)
    uxb_face_y = 0.5 * (uxb_y[:, 1:, :] + uxb_y[:, :-1, :])
    fy = fy.at[:, 1:-1, :].set(sigma_y * (-phi_grad_y + uxb_face_y))

    sigma_z = _extruded_harmonic_mean(sigma[:, :, 1:], sigma[:, :, :-1])
    phi_grad_z = (phi[:, :, 1:] - phi[:, :, :-1]) / max(dz, 1.0e-12)
    uxb_face_z = 0.5 * (uxb_z[:, :, 1:] + uxb_z[:, :, :-1])
    fz = fz.at[:, :, 1:-1].set(sigma_z * (-phi_grad_z + uxb_face_z))
    return fx, fy, fz


def _extruded_conservative_emf_rhs(
    sigma: jnp.ndarray,
    uxb_x: jnp.ndarray,
    uxb_y: jnp.ndarray,
    uxb_z: jnp.ndarray,
    *,
    dx: float,
    dy: float,
    dz: float,
) -> jnp.ndarray:
    zeros = jnp.zeros_like(uxb_x)
    fx, fy, fz = _extruded_conservative_current_fluxes(
        sigma,
        zeros,
        uxb_x,
        uxb_y,
        uxb_z,
        dx=dx,
        dy=dy,
        dz=dz,
    )
    return (
        (fx[1:] - fx[:-1]) / max(dx, 1.0e-12)
        + (fy[:, 1:, :] - fy[:, :-1, :]) / max(dy, 1.0e-12)
        + (fz[:, :, 1:] - fz[:, :, :-1]) / max(dz, 1.0e-12)
    )


def _solve_extruded_velocity_jacobi(
    *,
    rhs: jnp.ndarray,
    diffusivity: jnp.ndarray,
    reaction: jnp.ndarray,
    dx: float,
    dy: float,
    dz: float,
    iterations: int,
    relaxation: float,
) -> jnp.ndarray:
    omega = jnp.asarray(relaxation, dtype=rhs.dtype)
    diagonal = reaction + 2.0 * diffusivity * (
        1.0 / max(dx**2, 1.0e-12) + 1.0 / max(dy**2, 1.0e-12) + 1.0 / max(dz**2, 1.0e-12)
    )
    diagonal = jnp.maximum(diagonal, 1.0e-12)

    def body_fun(_, field):
        x_west, x_east, y_south, y_north, z_bottom, z_top = _extruded_neighbor_fields(field)
        updated = (
            rhs
            + diffusivity
            * (
                (x_west + x_east) / max(dx**2, 1.0e-12)
                + (y_south + y_north) / max(dy**2, 1.0e-12)
                + (z_bottom + z_top) / max(dz**2, 1.0e-12)
            )
        ) / diagonal
        blended = (1.0 - omega) * field + omega * updated
        return _extruded_enforce_velocity_bc(blended)

    return jax.lax.fori_loop(0, iterations, body_fun, jnp.zeros_like(rhs))


def _solve_velocity_jacobi_state(
    *,
    mesh: StructuredMesh,
    diffusivity: jnp.ndarray,
    reaction: jnp.ndarray,
    rhs: jnp.ndarray,
    active_mask: jnp.ndarray,
    iterations: int,
    relaxation: float,
) -> jnp.ndarray:
    diagonal, west, east, south, north = _velocity_system_coefficients(mesh, diffusivity, reaction, active_mask)
    diagonal = jnp.maximum(diagonal, 1.0e-12)
    omega = jnp.asarray(relaxation, dtype=rhs.dtype)
    field0 = jnp.zeros_like(rhs)

    def body_fun(_, field):
        west_field = jnp.pad(field[:-1, :], ((1, 0), (0, 0)))
        east_field = jnp.pad(field[1:, :], ((0, 1), (0, 0)))
        south_field = jnp.pad(field[:, :-1], ((0, 0), (1, 0)))
        north_field = jnp.pad(field[:, 1:], ((0, 0), (0, 1)))
        updated = (rhs + west * west_field + east * east_field + south * south_field + north * north_field) / diagonal
        blended = (1.0 - omega) * field + omega * updated
        return jnp.where(active_mask, blended, 0.0)

    return jax.lax.fori_loop(0, iterations, body_fun, field0)


def solve_differentiable_hartmann(
    problem: HartmannAutodiffProblem,
    *,
    forcing: float | jnp.ndarray,
    hartmann_number: float | jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    mesh = problem.mesh
    sigma = problem.sigma
    rho = problem.rho
    nu = problem.nu
    fluid_mask = problem.fluid_mask
    by = jnp.zeros(mesh.yz_shape, dtype=sigma.dtype)
    bz = jnp.full(mesh.yz_shape, hartmann_number, dtype=sigma.dtype)
    forcing_value = jnp.asarray(forcing, dtype=sigma.dtype)

    def macro_body(_, u_iter):
        uxb_y = jnp.where(fluid_mask, -u_iter * bz, 0.0)
        uxb_z = jnp.where(fluid_mask, u_iter * by, 0.0)
        conv_y = _face_emf_y(mesh, sigma, uxb_y)
        conv_z = _face_emf_z(mesh, sigma, uxb_z)
        face_conv_y = jnp.pad(conv_y, ((1, 1), (0, 0)))
        face_conv_z = jnp.pad(conv_z, ((0, 0), (1, 1)))
        rhs_phi = -(
            (face_conv_y[1:, :] - face_conv_y[:-1, :]) / mesh.dy[:, None]
            + (face_conv_z[:, 1:] - face_conv_z[:, :-1]) / mesh.dz[None, :]
        )
        diagonal, west, east, south, north = _potential_coefficients(mesh, sigma)
        phi, _, _ = solve_poisson_jacobi_state(
            diagonal,
            west,
            east,
            south,
            north,
            rhs_phi,
            problem.anchor,
            problem.potential_iterations,
            tolerance=None,
            relaxation=problem.relaxation,
        )
        reaction = sigma * (bz**2 + by**2) / rho
        rhs_u, _ = _fully_developed_rhs(
            mesh=mesh,
            sigma=sigma,
            rho=rho,
            fluid_mask=fluid_mask,
            u=u_iter,
            phi=phi,
            by=by,
            bz=bz,
            forcing=forcing_value,
        )
        u_next = _solve_velocity_jacobi_state(
            mesh=mesh,
            diffusivity=nu,
            reaction=reaction,
            rhs=rhs_u,
            active_mask=fluid_mask,
            iterations=problem.velocity_iterations,
            relaxation=problem.relaxation,
        )
        return _enforce_velocity_bc(u_next, mesh, fluid_mask, interpolate_direct_fluid_walls=False)

    u = jax.lax.fori_loop(0, problem.macro_iterations, macro_body, jnp.zeros(mesh.yz_shape, dtype=sigma.dtype))

    uxb_y = jnp.where(fluid_mask, -u * bz, 0.0)
    uxb_z = jnp.where(fluid_mask, u * by, 0.0)
    conv_y = _face_emf_y(mesh, sigma, uxb_y)
    conv_z = _face_emf_z(mesh, sigma, uxb_z)
    face_conv_y = jnp.pad(conv_y, ((1, 1), (0, 0)))
    face_conv_z = jnp.pad(conv_z, ((0, 0), (1, 1)))
    rhs_phi = -(
        (face_conv_y[1:, :] - face_conv_y[:-1, :]) / mesh.dy[:, None]
        + (face_conv_z[:, 1:] - face_conv_z[:, :-1]) / mesh.dz[None, :]
    )
    diagonal, west, east, south, north = _potential_coefficients(mesh, sigma)
    phi, _, _ = solve_poisson_jacobi_state(
        diagonal,
        west,
        east,
        south,
        north,
        rhs_phi,
        problem.anchor,
        problem.potential_iterations,
        tolerance=None,
        relaxation=problem.relaxation,
    )
    return u, phi


def fringing_mean_velocity_history(
    problem: FringingAutodiffProblem,
    *,
    forcing: float | jnp.ndarray,
    peak_hartmann_number: float | jnp.ndarray,
    entry_center: float | jnp.ndarray,
    exit_center: float | jnp.ndarray,
    transition_width: float | jnp.ndarray,
) -> dict[str, jnp.ndarray]:
    field_scale = _smooth_fringing_scale(
        problem.x,
        entry_center=entry_center,
        exit_center=exit_center,
        transition_width=transition_width,
        peak_scale=1.0,
    )

    def single_station(scale_value):
        return hartmann_mean_velocity(
            problem.base_problem,
            forcing=forcing,
            hartmann_number=jnp.asarray(peak_hartmann_number) * scale_value,
        )

    mean_velocity = jax.vmap(single_station)(field_scale)
    return {
        "x": problem.x,
        "field_scale": field_scale,
        "mean_velocity": mean_velocity,
    }


def hartmann_current_proxy(
    problem: HartmannAutodiffProblem,
    *,
    forcing: float | jnp.ndarray,
    hartmann_number: float | jnp.ndarray,
) -> jnp.ndarray:
    u, phi = solve_differentiable_hartmann(problem, forcing=forcing, hartmann_number=hartmann_number)
    dphi_dy, _ = gradient_scalar(phi, problem.mesh)
    jy = -dphi_dy - u * jnp.asarray(hartmann_number, dtype=u.dtype)
    return jnp.mean(jnp.abs(jy))


def fringing_response_history(
    problem: FringingAutodiffProblem,
    *,
    forcing: float | jnp.ndarray,
    peak_hartmann_number: float | jnp.ndarray,
    entry_center: float | jnp.ndarray,
    exit_center: float | jnp.ndarray,
    transition_width: float | jnp.ndarray,
) -> dict[str, jnp.ndarray]:
    field_scale = _smooth_fringing_scale(
        problem.x,
        entry_center=entry_center,
        exit_center=exit_center,
        transition_width=transition_width,
        peak_scale=1.0,
    )

    def single_station(scale_value):
        station_ha = jnp.asarray(peak_hartmann_number) * scale_value
        return (
            hartmann_mean_velocity(
                problem.base_problem,
                forcing=forcing,
                hartmann_number=station_ha,
            ),
            hartmann_current_proxy(
                problem.base_problem,
                forcing=forcing,
                hartmann_number=station_ha,
            ),
        )

    mean_velocity, current_proxy = jax.vmap(single_station)(field_scale)
    return {
        "x": problem.x,
        "field_scale": field_scale,
        "mean_velocity": mean_velocity,
        "current_proxy": current_proxy,
    }


def extruded_rect_response_history(
    problem: FringingAutodiffProblem,
    *,
    forcing: float | jnp.ndarray,
    peak_hartmann_number: float | jnp.ndarray,
    entry_center: float | jnp.ndarray,
    exit_center: float | jnp.ndarray,
    transition_width: float | jnp.ndarray,
) -> dict[str, jnp.ndarray]:
    field_scale = _smooth_fringing_scale(
        problem.x,
        entry_center=entry_center,
        exit_center=exit_center,
        transition_width=transition_width,
        peak_scale=1.0,
    )
    return _extruded_rect_response_history_from_field_scale(
        problem,
        forcing=forcing,
        peak_hartmann_number=peak_hartmann_number,
        field_scale=field_scale,
    )


def extruded_rect_projection_history(
    problem: FringingAutodiffProblem,
    *,
    forcing: float | jnp.ndarray,
    peak_hartmann_number: float | jnp.ndarray,
    entry_center: float | jnp.ndarray,
    exit_center: float | jnp.ndarray,
    transition_width: float | jnp.ndarray,
) -> dict[str, jnp.ndarray]:
    field_scale = _smooth_fringing_scale(
        problem.x,
        entry_center=entry_center,
        exit_center=exit_center,
        transition_width=transition_width,
        peak_scale=1.0,
    )
    return _extruded_rect_projection_history_from_field_scale(
        problem,
        forcing=forcing,
        peak_hartmann_number=peak_hartmann_number,
        field_scale=field_scale,
    )


def wham_mirror_pressure_drop_history(
    problem: FringingAutodiffProblem,
    *,
    forcing: float | jnp.ndarray,
    peak_hartmann_number: float | jnp.ndarray,
    coil_separation: float | jnp.ndarray,
    center_offset: float | jnp.ndarray = 0.0,
    current_scale: float = 2000.0 * 17.0 / 17.51,
    inner_radius: float = 0.5 * 86.0e-3,
    outer_radius: float = 0.5 * 730.0e-3,
    coil_axial_thickness: float = 14.3e-3 * 8.0,
    radial_loops: int = 24,
    axial_loops: int = 8,
) -> dict[str, jnp.ndarray]:
    centered_x = problem.x - 0.5 * (problem.x[0] + problem.x[-1]) - jnp.asarray(center_offset, dtype=jnp.float32)
    field_scale = wham_mirror_station_scale(
        centered_x,
        coil_separation=coil_separation,
        current_scale=current_scale,
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        coil_axial_thickness=coil_axial_thickness,
        radial_loops=radial_loops,
        axial_loops=axial_loops,
    )
    return _extruded_rect_projection_history_from_field_scale(
        problem,
        forcing=forcing,
        peak_hartmann_number=peak_hartmann_number,
        field_scale=field_scale,
    )


def wham_mirror_pressure_drop_sensitivity(
    problem: FringingAutodiffProblem,
    *,
    forcing: float | jnp.ndarray,
    peak_hartmann_number: float | jnp.ndarray,
    coil_separation: float | jnp.ndarray,
    center_offset: float | jnp.ndarray = 0.0,
    current_scale: float = 2000.0 * 17.0 / 17.51,
    inner_radius: float = 0.5 * 86.0e-3,
    outer_radius: float = 0.5 * 730.0e-3,
    coil_axial_thickness: float = 14.3e-3 * 8.0,
    radial_loops: int = 24,
    axial_loops: int = 8,
) -> dict[str, jnp.ndarray]:
    def objective(separation_value):
        response = wham_mirror_pressure_drop_history(
            problem,
            forcing=forcing,
            peak_hartmann_number=peak_hartmann_number,
            coil_separation=separation_value,
            center_offset=center_offset,
            current_scale=current_scale,
            inner_radius=inner_radius,
            outer_radius=outer_radius,
            coil_axial_thickness=coil_axial_thickness,
            radial_loops=radial_loops,
            axial_loops=axial_loops,
        )
        x = response["x"]
        pressure_span = response["pressure_span"]
        dx = x[1:] - x[:-1]
        return jnp.sum(0.5 * (pressure_span[1:] + pressure_span[:-1]) * dx) / jnp.maximum(x[-1] - x[0], 1.0e-12)

    pressure_drop_proxy = objective(coil_separation)
    response = wham_mirror_pressure_drop_history(
        problem,
        forcing=forcing,
        peak_hartmann_number=peak_hartmann_number,
        coil_separation=coil_separation,
        center_offset=center_offset,
        current_scale=current_scale,
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        coil_axial_thickness=coil_axial_thickness,
        radial_loops=radial_loops,
        axial_loops=axial_loops,
    )
    return {
        "pressure_drop_proxy": pressure_drop_proxy,
        "d_pressure_drop_d_separation": jax.grad(objective)(coil_separation),
        "x": response["x"],
        "field_scale": response["field_scale"],
        "pressure_span": response["pressure_span"],
        "mean_velocity": response["mean_velocity"],
        "current_proxy": response["current_proxy"],
    }


def extruded_rect_projection_iteration_history(
    problem: FringingAutodiffProblem,
    *,
    forcing: float | jnp.ndarray,
    peak_hartmann_number: float | jnp.ndarray,
    entry_center: float | jnp.ndarray,
    exit_center: float | jnp.ndarray,
    transition_width: float | jnp.ndarray,
    station_indices: jnp.ndarray,
) -> dict[str, jnp.ndarray]:
    mesh = problem.base_problem.mesh
    ny, nz = mesh.yz_shape
    nx = int(problem.x.shape[0])
    dx = float((problem.x[-1] - problem.x[0]) / max(nx - 1, 1)) if nx > 1 else 1.0
    dy = float(jnp.mean(mesh.dy))
    dz = float(jnp.mean(mesh.dz))
    sigma = jnp.broadcast_to(problem.base_problem.sigma[None, :, :], (nx, ny, nz))
    rho = jnp.broadcast_to(problem.base_problem.rho[None, :, :], (nx, ny, nz))
    nu = jnp.broadcast_to(problem.base_problem.nu[None, :, :], (nx, ny, nz))
    field_scale = _smooth_fringing_scale(
        problem.x,
        entry_center=entry_center,
        exit_center=exit_center,
        transition_width=transition_width,
        peak_scale=1.0,
    )
    bz = jnp.broadcast_to(jnp.asarray(peak_hartmann_number) * field_scale[:, None, None], (nx, ny, nz))
    forcing_value = jnp.asarray(forcing, dtype=sigma.dtype)
    inverse_diffusive_scale = jnp.max(nu) * (1.0 / max(dx**2, 1.0e-12) + 1.0 / max(dy**2, 1.0e-12) + 1.0 / max(dz**2, 1.0e-12))
    dt = jnp.asarray(0.2) / jnp.maximum(inverse_diffusive_scale, 1.0e-12)
    pressure_iterations = max(4, problem.base_problem.potential_iterations // 2)
    station_ids = jnp.asarray(station_indices, dtype=jnp.int32)

    def step(state, _):
        u, v, w, p, phi = state
        dphi_dx, dphi_dy, dphi_dz = _extruded_gradient(phi, dx=dx, dy=dy, dz=dz)
        uxb_x = v * bz
        uxb_y = -u * bz
        uxb_z = jnp.zeros_like(u)
        jx = sigma * (-dphi_dx + uxb_x)
        jy = sigma * (-dphi_dy + uxb_y)
        jz = sigma * (-dphi_dz + uxb_z)
        lorentz_x = jy * bz
        lorentz_y = -jx * bz
        lorentz_z = jnp.zeros_like(u)

        dp_dx, dp_dy, dp_dz = _extruded_gradient(p, dx=dx, dy=dy, dz=dz)
        u_star = _extruded_enforce_velocity_bc(
            u + dt * (nu * _extruded_laplacian(u, dx=dx, dy=dy, dz=dz) + forcing_value / jnp.maximum(rho, 1.0e-12) + lorentz_x / jnp.maximum(rho, 1.0e-12) - dp_dx / jnp.maximum(rho, 1.0e-12))
        )
        v_star = _extruded_enforce_velocity_bc(
            v + dt * (nu * _extruded_laplacian(v, dx=dx, dy=dy, dz=dz) + lorentz_y / jnp.maximum(rho, 1.0e-12) - dp_dy / jnp.maximum(rho, 1.0e-12))
        )
        w_star = _extruded_enforce_velocity_bc(
            w + dt * (nu * _extruded_laplacian(w, dx=dx, dy=dy, dz=dz) + lorentz_z / jnp.maximum(rho, 1.0e-12) - dp_dz / jnp.maximum(rho, 1.0e-12))
        )

        du_dx, _, _ = _extruded_gradient(u_star, dx=dx, dy=dy, dz=dz)
        _, dv_dy, _ = _extruded_gradient(v_star, dx=dx, dy=dy, dz=dz)
        _, _, dw_dz = _extruded_gradient(w_star, dx=dx, dy=dy, dz=dz)
        divergence = du_dx + dv_dy + dw_dz
        p_corr = _extruded_poisson_jacobi((rho / jnp.maximum(dt, 1.0e-12)) * divergence, dx=dx, dy=dy, dz=dz, iterations=pressure_iterations)
        dpc_dx, dpc_dy, dpc_dz = _extruded_gradient(p_corr, dx=dx, dy=dy, dz=dz)
        u_next = _extruded_enforce_velocity_bc(u_star - (dt / jnp.maximum(rho, 1.0e-12)) * dpc_dx)
        v_next = _extruded_enforce_velocity_bc(v_star - (dt / jnp.maximum(rho, 1.0e-12)) * dpc_dy)
        w_next = _extruded_enforce_velocity_bc(w_star - (dt / jnp.maximum(rho, 1.0e-12)) * dpc_dz)
        p_next = p + p_corr

        uxb_x = v_next * bz
        uxb_y = -u_next * bz
        uxb_z = jnp.zeros_like(u_next)
        rhs_phi = _extruded_conservative_emf_rhs(
            sigma,
            uxb_x,
            uxb_y,
            uxb_z,
            dx=dx,
            dy=dy,
            dz=dz,
        )
        phi_next = _extruded_poisson_jacobi(
            rhs_phi,
            dx=dx,
            dy=dy,
            dz=dz,
            iterations=problem.base_problem.potential_iterations,
        )
        dphi_dx, dphi_dy, dphi_dz = _extruded_gradient(phi_next, dx=dx, dy=dy, dz=dz)
        jx = sigma * (-dphi_dx + uxb_x)
        jy = sigma * (-dphi_dy + uxb_y)
        fx, fy, fz = _extruded_conservative_current_fluxes(
            sigma,
            phi_next,
            uxb_x,
            uxb_y,
            uxb_z,
            dx=dx,
            dy=dy,
            dz=dz,
        )
        div_j = (
            (fx[1:] - fx[:-1]) / max(dx, 1.0e-12)
            + (fy[:, 1:, :] - fy[:, :-1, :]) / max(dy, 1.0e-12)
            + (fz[:, :, 1:] - fz[:, :, :-1]) / max(dz, 1.0e-12)
        )
        boundary_current_residual = jnp.abs(
            -jnp.sum(fx[0], axis=(0, 1)) * dy * dz
            + jnp.sum(fx[-1], axis=(0, 1)) * dy * dz
            - jnp.sum(fy[:, 0, :], axis=1) * dx * dz
            + jnp.sum(fy[:, -1, :], axis=1) * dx * dz
            - jnp.sum(fz[:, :, 0], axis=1) * dx * dy
            + jnp.sum(fz[:, :, -1], axis=1) * dx * dy
        )
        sample = {
            "u_field": u_next[station_ids],
            "phi_field": phi_next[station_ids],
            "jy_field": jy[station_ids],
            "pressure_field": p_next[station_ids],
            "charge_balance_residual": jnp.max(jnp.abs(div_j), axis=(1, 2))[station_ids],
            "boundary_current_residual": boundary_current_residual[station_ids],
        }
        return (u_next, v_next, w_next, p_next, phi_next), sample

    initial_state = (
        jnp.zeros((nx, ny, nz), dtype=problem.base_problem.sigma.dtype),
        jnp.zeros((nx, ny, nz), dtype=problem.base_problem.sigma.dtype),
        jnp.zeros((nx, ny, nz), dtype=problem.base_problem.sigma.dtype),
        jnp.zeros((nx, ny, nz), dtype=problem.base_problem.sigma.dtype),
        jnp.zeros((nx, ny, nz), dtype=problem.base_problem.sigma.dtype),
    )
    _, history = jax.lax.scan(step, initial_state, xs=None, length=problem.base_problem.macro_iterations)
    history["station_indices"] = station_ids
    history["field_scale"] = field_scale[station_ids]
    history["x"] = problem.x[station_ids]
    return history


def hartmann_mean_velocity(
    problem: HartmannAutodiffProblem,
    *,
    forcing: float | jnp.ndarray,
    hartmann_number: float | jnp.ndarray,
) -> jnp.ndarray:
    u, _ = solve_differentiable_hartmann(problem, forcing=forcing, hartmann_number=hartmann_number)
    return jnp.mean(u)


def hartmann_mean_velocity_gradients(
    problem: HartmannAutodiffProblem,
    *,
    forcing: float | jnp.ndarray,
    hartmann_number: float | jnp.ndarray,
) -> dict[str, jnp.ndarray]:
    objective = lambda force_value, ha_value: hartmann_mean_velocity(
        problem,
        forcing=force_value,
        hartmann_number=ha_value,
    )
    mean_velocity = objective(forcing, hartmann_number)
    d_mean_velocity_d_forcing, d_mean_velocity_d_ha = jax.grad(objective, argnums=(0, 1))(forcing, hartmann_number)
    return {
        "mean_velocity": mean_velocity,
        "d_mean_velocity_d_forcing": d_mean_velocity_d_forcing,
        "d_mean_velocity_d_ha": d_mean_velocity_d_ha,
    }


def hartmann_mean_velocity_finite_difference_gradients(
    problem: HartmannAutodiffProblem,
    *,
    forcing: float | jnp.ndarray,
    hartmann_number: float | jnp.ndarray,
    delta_forcing: float = 1.0e-3,
    delta_ha: float = 1.0e-3,
) -> dict[str, jnp.ndarray]:
    mean_velocity = hartmann_mean_velocity(problem, forcing=forcing, hartmann_number=hartmann_number)
    plus_forcing = hartmann_mean_velocity(problem, forcing=jnp.asarray(forcing) + delta_forcing, hartmann_number=hartmann_number)
    minus_forcing = hartmann_mean_velocity(problem, forcing=jnp.asarray(forcing) - delta_forcing, hartmann_number=hartmann_number)
    plus_ha = hartmann_mean_velocity(problem, forcing=forcing, hartmann_number=jnp.asarray(hartmann_number) + delta_ha)
    minus_ha = hartmann_mean_velocity(problem, forcing=forcing, hartmann_number=jnp.asarray(hartmann_number) - delta_ha)
    return {
        "mean_velocity": mean_velocity,
        "d_mean_velocity_d_forcing": (plus_forcing - minus_forcing) / (2.0 * delta_forcing),
        "d_mean_velocity_d_ha": (plus_ha - minus_ha) / (2.0 * delta_ha),
    }


def hartmann_profile_loss(
    problem: HartmannAutodiffProblem,
    *,
    forcing: float | jnp.ndarray,
    hartmann_number: float | jnp.ndarray,
    target_profile: jnp.ndarray,
) -> jnp.ndarray:
    u, _ = solve_differentiable_hartmann(problem, forcing=forcing, hartmann_number=hartmann_number)
    centerline = u[:, u.shape[1] // 2]
    centerline_scale = jnp.maximum(jnp.max(jnp.abs(centerline)), 1.0e-12)
    target_scale = jnp.maximum(jnp.max(jnp.abs(target_profile)), 1.0e-12)
    return jnp.mean((centerline / centerline_scale - target_profile / target_scale) ** 2)


def fringing_history_loss(
    problem: FringingAutodiffProblem,
    *,
    forcing: float | jnp.ndarray,
    peak_hartmann_number: float | jnp.ndarray,
    entry_center: float | jnp.ndarray,
    exit_center: float | jnp.ndarray,
    transition_width: float | jnp.ndarray,
    target_mean_velocity: jnp.ndarray,
) -> jnp.ndarray:
    history = fringing_mean_velocity_history(
        problem,
        forcing=forcing,
        peak_hartmann_number=peak_hartmann_number,
        entry_center=entry_center,
        exit_center=exit_center,
        transition_width=transition_width,
    )["mean_velocity"]
    scale = jnp.maximum(jnp.max(jnp.abs(target_mean_velocity)), 1.0e-12)
    return jnp.mean(((history - target_mean_velocity) / scale) ** 2)


def fringing_response_loss(
    problem: FringingAutodiffProblem,
    *,
    forcing: float | jnp.ndarray,
    peak_hartmann_number: float | jnp.ndarray,
    entry_center: float | jnp.ndarray,
    exit_center: float | jnp.ndarray,
    transition_width: float | jnp.ndarray,
    target_mean_velocity: jnp.ndarray,
    target_current_proxy: jnp.ndarray,
    current_weight: float = 1.0,
) -> jnp.ndarray:
    response = fringing_response_history(
        problem,
        forcing=forcing,
        peak_hartmann_number=peak_hartmann_number,
        entry_center=entry_center,
        exit_center=exit_center,
        transition_width=transition_width,
    )
    velocity_scale = jnp.maximum(jnp.max(jnp.abs(target_mean_velocity)), 1.0e-12)
    current_scale = jnp.maximum(jnp.max(jnp.abs(target_current_proxy)), 1.0e-12)
    velocity_loss = jnp.mean(((response["mean_velocity"] - target_mean_velocity) / velocity_scale) ** 2)
    current_loss = jnp.mean(((response["current_proxy"] - target_current_proxy) / current_scale) ** 2)
    return velocity_loss + current_weight * current_loss


def extruded_rect_response_loss(
    problem: FringingAutodiffProblem,
    *,
    forcing: float | jnp.ndarray,
    peak_hartmann_number: float | jnp.ndarray,
    entry_center: float | jnp.ndarray,
    exit_center: float | jnp.ndarray,
    transition_width: float | jnp.ndarray,
    target_mean_velocity: jnp.ndarray,
    target_current_proxy: jnp.ndarray,
    target_charge_balance: jnp.ndarray,
    target_boundary_current: jnp.ndarray,
    current_weight: float = 1.0,
    charge_balance_weight: float = 0.1,
    boundary_current_weight: float = 0.1,
) -> jnp.ndarray:
    response = extruded_rect_response_history(
        problem,
        forcing=forcing,
        peak_hartmann_number=peak_hartmann_number,
        entry_center=entry_center,
        exit_center=exit_center,
        transition_width=transition_width,
    )
    velocity_scale = jnp.maximum(jnp.max(jnp.abs(target_mean_velocity)), 1.0e-12)
    current_scale = jnp.maximum(jnp.max(jnp.abs(target_current_proxy)), 1.0e-12)
    charge_scale = jnp.maximum(jnp.max(jnp.abs(target_charge_balance)), 1.0e-12)
    boundary_scale = jnp.maximum(jnp.max(jnp.abs(target_boundary_current)), 1.0e-12)
    velocity_loss = jnp.mean(((response["mean_velocity"] - target_mean_velocity) / velocity_scale) ** 2)
    current_loss = jnp.mean(((response["current_proxy"] - target_current_proxy) / current_scale) ** 2)
    charge_loss = jnp.mean(((response["charge_balance_residual"] - target_charge_balance) / charge_scale) ** 2)
    boundary_loss = jnp.mean(((response["boundary_current_residual"] - target_boundary_current) / boundary_scale) ** 2)
    return velocity_loss + current_weight * current_loss + charge_balance_weight * charge_loss + boundary_current_weight * boundary_loss


def extruded_rect_projection_loss(
    problem: FringingAutodiffProblem,
    *,
    forcing: float | jnp.ndarray,
    peak_hartmann_number: float | jnp.ndarray,
    entry_center: float | jnp.ndarray,
    exit_center: float | jnp.ndarray,
    transition_width: float | jnp.ndarray,
    target_mean_velocity: jnp.ndarray,
    target_current_proxy: jnp.ndarray,
    target_charge_balance: jnp.ndarray,
    target_boundary_current: jnp.ndarray,
    target_pressure_span: jnp.ndarray,
    current_weight: float = 1.0,
    charge_balance_weight: float = 0.1,
    boundary_current_weight: float = 0.1,
    pressure_span_weight: float = 0.25,
) -> jnp.ndarray:
    response = extruded_rect_projection_history(
        problem,
        forcing=forcing,
        peak_hartmann_number=peak_hartmann_number,
        entry_center=entry_center,
        exit_center=exit_center,
        transition_width=transition_width,
    )
    velocity_scale = jnp.maximum(jnp.max(jnp.abs(target_mean_velocity)), 1.0e-12)
    current_scale = jnp.maximum(jnp.max(jnp.abs(target_current_proxy)), 1.0e-12)
    charge_scale = jnp.maximum(jnp.max(jnp.abs(target_charge_balance)), 1.0e-12)
    boundary_scale = jnp.maximum(jnp.max(jnp.abs(target_boundary_current)), 1.0e-12)
    pressure_scale = jnp.maximum(jnp.max(jnp.abs(target_pressure_span)), 1.0e-12)
    velocity_loss = jnp.mean(((response["mean_velocity"] - target_mean_velocity) / velocity_scale) ** 2)
    current_loss = jnp.mean(((response["current_proxy"] - target_current_proxy) / current_scale) ** 2)
    charge_loss = jnp.mean(((response["charge_balance_residual"] - target_charge_balance) / charge_scale) ** 2)
    boundary_loss = jnp.mean(((response["boundary_current_residual"] - target_boundary_current) / boundary_scale) ** 2)
    pressure_loss = jnp.mean(((response["pressure_span"] - target_pressure_span) / pressure_scale) ** 2)
    return (
        velocity_loss
        + current_weight * current_loss
        + charge_balance_weight * charge_loss
        + boundary_current_weight * boundary_loss
        + pressure_span_weight * pressure_loss
    )


def extruded_rect_projection_field_loss(
    problem: FringingAutodiffProblem,
    *,
    forcing: float | jnp.ndarray,
    peak_hartmann_number: float | jnp.ndarray,
    entry_center: float | jnp.ndarray,
    exit_center: float | jnp.ndarray,
    transition_width: float | jnp.ndarray,
    target_u_field: jnp.ndarray,
    target_phi_field: jnp.ndarray,
    target_jy_field: jnp.ndarray,
    target_pressure_field: jnp.ndarray,
    station_indices: jnp.ndarray,
    u_weight: float = 1.0,
    phi_weight: float = 0.25,
    jy_weight: float = 0.5,
    pressure_weight: float = 0.25,
) -> jnp.ndarray:
    response = extruded_rect_projection_history(
        problem,
        forcing=forcing,
        peak_hartmann_number=peak_hartmann_number,
        entry_center=entry_center,
        exit_center=exit_center,
        transition_width=transition_width,
    )
    station_ids = jnp.asarray(station_indices, dtype=jnp.int32)
    u_field = response["u_field"][station_ids]
    phi_field = response["phi_field"][station_ids]
    jy_field = response["jy_field"][station_ids]
    pressure_field = response["pressure_field"][station_ids]
    u_scale = jnp.maximum(jnp.max(jnp.abs(target_u_field)), 1.0e-12)
    phi_scale = jnp.maximum(jnp.max(jnp.abs(target_phi_field)), 1.0e-12)
    jy_scale = jnp.maximum(jnp.max(jnp.abs(target_jy_field)), 1.0e-12)
    pressure_scale = jnp.maximum(jnp.max(jnp.abs(target_pressure_field)), 1.0e-12)
    u_loss = jnp.mean(((u_field - target_u_field) / u_scale) ** 2)
    phi_loss = jnp.mean(((phi_field - target_phi_field) / phi_scale) ** 2)
    jy_loss = jnp.mean(((jy_field - target_jy_field) / jy_scale) ** 2)
    pressure_loss = jnp.mean(((pressure_field - target_pressure_field) / pressure_scale) ** 2)
    return u_weight * u_loss + phi_weight * phi_loss + jy_weight * jy_loss + pressure_weight * pressure_loss


def extruded_rect_projection_trajectory_loss(
    problem: FringingAutodiffProblem,
    *,
    forcing: float | jnp.ndarray,
    peak_hartmann_number: float | jnp.ndarray,
    entry_center: float | jnp.ndarray,
    exit_center: float | jnp.ndarray,
    transition_width: float | jnp.ndarray,
    target_u_history: jnp.ndarray,
    target_phi_history: jnp.ndarray,
    target_jy_history: jnp.ndarray,
    target_pressure_history: jnp.ndarray,
    target_charge_balance_history: jnp.ndarray,
    target_boundary_current_history: jnp.ndarray,
    station_indices: jnp.ndarray,
    u_weight: float = 1.0,
    phi_weight: float = 0.25,
    jy_weight: float = 0.5,
    pressure_weight: float = 0.25,
    charge_balance_weight: float = 0.1,
    boundary_current_weight: float = 0.1,
) -> jnp.ndarray:
    trajectory = extruded_rect_projection_iteration_history(
        problem,
        forcing=forcing,
        peak_hartmann_number=peak_hartmann_number,
        entry_center=entry_center,
        exit_center=exit_center,
        transition_width=transition_width,
        station_indices=station_indices,
    )
    u_scale = jnp.maximum(jnp.max(jnp.abs(target_u_history)), 1.0e-12)
    phi_scale = jnp.maximum(jnp.max(jnp.abs(target_phi_history)), 1.0e-12)
    jy_scale = jnp.maximum(jnp.max(jnp.abs(target_jy_history)), 1.0e-12)
    pressure_scale = jnp.maximum(jnp.max(jnp.abs(target_pressure_history)), 1.0e-12)
    charge_scale = jnp.maximum(jnp.max(jnp.abs(target_charge_balance_history)), 1.0e-12)
    boundary_scale = jnp.maximum(jnp.max(jnp.abs(target_boundary_current_history)), 1.0e-12)
    u_loss = jnp.mean(((trajectory["u_field"] - target_u_history) / u_scale) ** 2)
    phi_loss = jnp.mean(((trajectory["phi_field"] - target_phi_history) / phi_scale) ** 2)
    jy_loss = jnp.mean(((trajectory["jy_field"] - target_jy_history) / jy_scale) ** 2)
    pressure_loss = jnp.mean(((trajectory["pressure_field"] - target_pressure_history) / pressure_scale) ** 2)
    charge_loss = jnp.mean(((trajectory["charge_balance_residual"] - target_charge_balance_history) / charge_scale) ** 2)
    boundary_loss = jnp.mean(((trajectory["boundary_current_residual"] - target_boundary_current_history) / boundary_scale) ** 2)
    return (
        u_weight * u_loss
        + phi_weight * phi_loss
        + jy_weight * jy_loss
        + pressure_weight * pressure_loss
        + charge_balance_weight * charge_loss
        + boundary_current_weight * boundary_loss
    )


def hartmann_profile_loss_gradients(
    problem: HartmannAutodiffProblem,
    *,
    forcing: float | jnp.ndarray,
    hartmann_number: float | jnp.ndarray,
    target_profile: jnp.ndarray,
) -> dict[str, jnp.ndarray]:
    objective = lambda force_value, ha_value: hartmann_profile_loss(
        problem,
        forcing=force_value,
        hartmann_number=ha_value,
        target_profile=target_profile,
    )
    loss = objective(forcing, hartmann_number)
    d_loss_d_forcing, d_loss_d_ha = jax.grad(objective, argnums=(0, 1))(forcing, hartmann_number)
    return {
        "loss": loss,
        "d_loss_d_forcing": d_loss_d_forcing,
        "d_loss_d_ha": d_loss_d_ha,
    }


def fringing_history_loss_gradients(
    problem: FringingAutodiffProblem,
    *,
    forcing: float | jnp.ndarray,
    peak_hartmann_number: float | jnp.ndarray,
    entry_center: float | jnp.ndarray,
    exit_center: float | jnp.ndarray,
    transition_width: float | jnp.ndarray,
    target_mean_velocity: jnp.ndarray,
) -> dict[str, jnp.ndarray]:
    objective = lambda peak_ha, entry, exit_, width: fringing_history_loss(
        problem,
        forcing=forcing,
        peak_hartmann_number=peak_ha,
        entry_center=entry,
        exit_center=exit_,
        transition_width=width,
        target_mean_velocity=target_mean_velocity,
    )
    loss = objective(peak_hartmann_number, entry_center, exit_center, transition_width)
    d_peak_ha, d_entry, d_exit, d_width = jax.grad(objective, argnums=(0, 1, 2, 3))(
        peak_hartmann_number,
        entry_center,
        exit_center,
        transition_width,
    )
    return {
        "loss": loss,
        "d_peak_hartmann_number": d_peak_ha,
        "d_entry_center": d_entry,
        "d_exit_center": d_exit,
        "d_transition_width": d_width,
    }


def fringing_response_loss_gradients(
    problem: FringingAutodiffProblem,
    *,
    forcing: float | jnp.ndarray,
    peak_hartmann_number: float | jnp.ndarray,
    entry_center: float | jnp.ndarray,
    exit_center: float | jnp.ndarray,
    transition_width: float | jnp.ndarray,
    target_mean_velocity: jnp.ndarray,
    target_current_proxy: jnp.ndarray,
    current_weight: float = 1.0,
) -> dict[str, jnp.ndarray]:
    objective = lambda peak_ha, entry, exit_, width: fringing_response_loss(
        problem,
        forcing=forcing,
        peak_hartmann_number=peak_ha,
        entry_center=entry,
        exit_center=exit_,
        transition_width=width,
        target_mean_velocity=target_mean_velocity,
        target_current_proxy=target_current_proxy,
        current_weight=current_weight,
    )
    loss = objective(peak_hartmann_number, entry_center, exit_center, transition_width)
    d_peak_ha, d_entry, d_exit, d_width = jax.grad(objective, argnums=(0, 1, 2, 3))(
        peak_hartmann_number,
        entry_center,
        exit_center,
        transition_width,
    )
    return {
        "loss": loss,
        "d_peak_hartmann_number": d_peak_ha,
        "d_entry_center": d_entry,
        "d_exit_center": d_exit,
        "d_transition_width": d_width,
    }


def extruded_rect_response_loss_gradients(
    problem: FringingAutodiffProblem,
    *,
    forcing: float | jnp.ndarray,
    peak_hartmann_number: float | jnp.ndarray,
    entry_center: float | jnp.ndarray,
    exit_center: float | jnp.ndarray,
    transition_width: float | jnp.ndarray,
    target_mean_velocity: jnp.ndarray,
    target_current_proxy: jnp.ndarray,
    target_charge_balance: jnp.ndarray,
    target_boundary_current: jnp.ndarray,
    current_weight: float = 1.0,
    charge_balance_weight: float = 0.1,
    boundary_current_weight: float = 0.1,
) -> dict[str, jnp.ndarray]:
    objective = lambda peak_ha, entry, exit_, width: extruded_rect_response_loss(
        problem,
        forcing=forcing,
        peak_hartmann_number=peak_ha,
        entry_center=entry,
        exit_center=exit_,
        transition_width=width,
        target_mean_velocity=target_mean_velocity,
        target_current_proxy=target_current_proxy,
        target_charge_balance=target_charge_balance,
        target_boundary_current=target_boundary_current,
        current_weight=current_weight,
        charge_balance_weight=charge_balance_weight,
        boundary_current_weight=boundary_current_weight,
    )
    loss = objective(peak_hartmann_number, entry_center, exit_center, transition_width)
    d_peak_ha, d_entry, d_exit, d_width = jax.grad(objective, argnums=(0, 1, 2, 3))(
        peak_hartmann_number,
        entry_center,
        exit_center,
        transition_width,
    )
    return {
        "loss": loss,
        "d_peak_hartmann_number": d_peak_ha,
        "d_entry_center": d_entry,
        "d_exit_center": d_exit,
        "d_transition_width": d_width,
    }


def extruded_rect_projection_loss_gradients(
    problem: FringingAutodiffProblem,
    *,
    forcing: float | jnp.ndarray,
    peak_hartmann_number: float | jnp.ndarray,
    entry_center: float | jnp.ndarray,
    exit_center: float | jnp.ndarray,
    transition_width: float | jnp.ndarray,
    target_mean_velocity: jnp.ndarray,
    target_current_proxy: jnp.ndarray,
    target_charge_balance: jnp.ndarray,
    target_boundary_current: jnp.ndarray,
    target_pressure_span: jnp.ndarray,
    current_weight: float = 1.0,
    charge_balance_weight: float = 0.1,
    boundary_current_weight: float = 0.1,
    pressure_span_weight: float = 0.25,
) -> dict[str, jnp.ndarray]:
    objective = lambda peak_ha, entry, exit_, width: extruded_rect_projection_loss(
        problem,
        forcing=forcing,
        peak_hartmann_number=peak_ha,
        entry_center=entry,
        exit_center=exit_,
        transition_width=width,
        target_mean_velocity=target_mean_velocity,
        target_current_proxy=target_current_proxy,
        target_charge_balance=target_charge_balance,
        target_boundary_current=target_boundary_current,
        target_pressure_span=target_pressure_span,
        current_weight=current_weight,
        charge_balance_weight=charge_balance_weight,
        boundary_current_weight=boundary_current_weight,
        pressure_span_weight=pressure_span_weight,
    )
    loss = objective(peak_hartmann_number, entry_center, exit_center, transition_width)
    d_peak_ha, d_entry, d_exit, d_width = jax.grad(objective, argnums=(0, 1, 2, 3))(
        peak_hartmann_number,
        entry_center,
        exit_center,
        transition_width,
    )
    return {
        "loss": loss,
        "d_peak_hartmann_number": d_peak_ha,
        "d_entry_center": d_entry,
        "d_exit_center": d_exit,
        "d_transition_width": d_width,
    }


def extruded_rect_projection_field_loss_gradients(
    problem: FringingAutodiffProblem,
    *,
    forcing: float | jnp.ndarray,
    peak_hartmann_number: float | jnp.ndarray,
    entry_center: float | jnp.ndarray,
    exit_center: float | jnp.ndarray,
    transition_width: float | jnp.ndarray,
    target_u_field: jnp.ndarray,
    target_phi_field: jnp.ndarray,
    target_jy_field: jnp.ndarray,
    target_pressure_field: jnp.ndarray,
    station_indices: jnp.ndarray,
    u_weight: float = 1.0,
    phi_weight: float = 0.25,
    jy_weight: float = 0.5,
    pressure_weight: float = 0.25,
) -> dict[str, jnp.ndarray]:
    objective = lambda peak_ha, entry, exit_, width: extruded_rect_projection_field_loss(
        problem,
        forcing=forcing,
        peak_hartmann_number=peak_ha,
        entry_center=entry,
        exit_center=exit_,
        transition_width=width,
        target_u_field=target_u_field,
        target_phi_field=target_phi_field,
        target_jy_field=target_jy_field,
        target_pressure_field=target_pressure_field,
        station_indices=station_indices,
        u_weight=u_weight,
        phi_weight=phi_weight,
        jy_weight=jy_weight,
        pressure_weight=pressure_weight,
    )
    loss = objective(peak_hartmann_number, entry_center, exit_center, transition_width)
    d_peak_ha, d_entry, d_exit, d_width = jax.grad(objective, argnums=(0, 1, 2, 3))(
        peak_hartmann_number,
        entry_center,
        exit_center,
        transition_width,
    )
    return {
        "loss": loss,
        "d_peak_hartmann_number": d_peak_ha,
        "d_entry_center": d_entry,
        "d_exit_center": d_exit,
        "d_transition_width": d_width,
    }


def extruded_rect_projection_trajectory_loss_gradients(
    problem: FringingAutodiffProblem,
    *,
    forcing: float | jnp.ndarray,
    peak_hartmann_number: float | jnp.ndarray,
    entry_center: float | jnp.ndarray,
    exit_center: float | jnp.ndarray,
    transition_width: float | jnp.ndarray,
    target_u_history: jnp.ndarray,
    target_phi_history: jnp.ndarray,
    target_jy_history: jnp.ndarray,
    target_pressure_history: jnp.ndarray,
    target_charge_balance_history: jnp.ndarray,
    target_boundary_current_history: jnp.ndarray,
    station_indices: jnp.ndarray,
    u_weight: float = 1.0,
    phi_weight: float = 0.25,
    jy_weight: float = 0.5,
    pressure_weight: float = 0.25,
    charge_balance_weight: float = 0.1,
    boundary_current_weight: float = 0.1,
) -> dict[str, jnp.ndarray]:
    objective = lambda peak_ha, entry, exit_, width: extruded_rect_projection_trajectory_loss(
        problem,
        forcing=forcing,
        peak_hartmann_number=peak_ha,
        entry_center=entry,
        exit_center=exit_,
        transition_width=width,
        target_u_history=target_u_history,
        target_phi_history=target_phi_history,
        target_jy_history=target_jy_history,
        target_pressure_history=target_pressure_history,
        target_charge_balance_history=target_charge_balance_history,
        target_boundary_current_history=target_boundary_current_history,
        station_indices=station_indices,
        u_weight=u_weight,
        phi_weight=phi_weight,
        jy_weight=jy_weight,
        pressure_weight=pressure_weight,
        charge_balance_weight=charge_balance_weight,
        boundary_current_weight=boundary_current_weight,
    )
    loss = objective(peak_hartmann_number, entry_center, exit_center, transition_width)
    d_peak_ha, d_entry, d_exit, d_width = jax.grad(objective, argnums=(0, 1, 2, 3))(
        peak_hartmann_number,
        entry_center,
        exit_center,
        transition_width,
    )
    return {
        "loss": loss,
        "d_peak_hartmann_number": d_peak_ha,
        "d_entry_center": d_entry,
        "d_exit_center": d_exit,
        "d_transition_width": d_width,
    }


def run_hartmann_profile_inverse_design(
    problem: HartmannAutodiffProblem,
    *,
    target_profile: jnp.ndarray,
    forcing_init: float,
    hartmann_init: float,
    learning_rate_forcing: float = 20.0,
    learning_rate_ha: float = 5.0,
    steps: int = 24,
) -> dict[str, object]:
    forcing = jnp.asarray(forcing_init, dtype=jnp.float32)
    hartmann_number = jnp.asarray(hartmann_init, dtype=jnp.float32)
    history: list[dict[str, float]] = []
    for step in range(steps):
        gradients = hartmann_profile_loss_gradients(
            problem,
            forcing=forcing,
            hartmann_number=hartmann_number,
            target_profile=target_profile,
        )
        history.append(
            {
                "iteration": float(step),
                "forcing": float(forcing),
                "hartmann_number": float(hartmann_number),
                "loss": float(gradients["loss"]),
                "d_loss_d_forcing": float(gradients["d_loss_d_forcing"]),
                "d_loss_d_ha": float(gradients["d_loss_d_ha"]),
            }
        )
        forcing = jnp.clip(forcing - learning_rate_forcing * gradients["d_loss_d_forcing"], 0.05, 5.0)
        hartmann_number = jnp.clip(hartmann_number - learning_rate_ha * gradients["d_loss_d_ha"], 0.5, 40.0)

    recovered_u, recovered_phi = solve_differentiable_hartmann(
        problem,
        forcing=forcing,
        hartmann_number=hartmann_number,
    )
    recovered_profile = recovered_u[:, recovered_u.shape[1] // 2]
    return {
        "forcing": float(forcing),
        "hartmann_number": float(hartmann_number),
        "history": history,
        "recovered_profile": recovered_profile,
        "recovered_phi_max": float(jnp.max(jnp.abs(recovered_phi))),
    }


def run_fringing_history_inverse_design(
    problem: FringingAutodiffProblem,
    *,
    target_mean_velocity: jnp.ndarray,
    forcing: float,
    peak_hartmann_init: float,
    entry_center_init: float,
    exit_center_init: float,
    transition_width_init: float,
    learning_rate_peak_ha: float = 1.0,
    learning_rate_entry: float = 0.2,
    learning_rate_exit: float = 0.2,
    learning_rate_width: float = 0.1,
    steps: int = 16,
) -> dict[str, object]:
    peak_hartmann_number = jnp.asarray(peak_hartmann_init, dtype=jnp.float32)
    entry_center = jnp.asarray(entry_center_init, dtype=jnp.float32)
    exit_center = jnp.asarray(exit_center_init, dtype=jnp.float32)
    transition_width = jnp.asarray(transition_width_init, dtype=jnp.float32)
    history: list[dict[str, float]] = []

    for step in range(steps):
        gradients = fringing_history_loss_gradients(
            problem,
            forcing=forcing,
            peak_hartmann_number=peak_hartmann_number,
            entry_center=entry_center,
            exit_center=exit_center,
            transition_width=transition_width,
            target_mean_velocity=target_mean_velocity,
        )
        history.append(
            {
                "iteration": float(step),
                "peak_hartmann_number": float(peak_hartmann_number),
                "entry_center": float(entry_center),
                "exit_center": float(exit_center),
                "transition_width": float(transition_width),
                "loss": float(gradients["loss"]),
            }
        )
        peak_hartmann_number = jnp.clip(
            peak_hartmann_number - learning_rate_peak_ha * gradients["d_peak_hartmann_number"], 0.5, 60.0
        )
        entry_center = jnp.clip(entry_center - learning_rate_entry * gradients["d_entry_center"], 0.0, float(problem.x[-1]))
        exit_center = jnp.clip(exit_center - learning_rate_exit * gradients["d_exit_center"], 0.0, float(problem.x[-1]))
        transition_width = jnp.clip(
            transition_width - learning_rate_width * gradients["d_transition_width"], 0.05, float(problem.x[-1])
        )
        exit_center = jnp.maximum(exit_center, entry_center + 0.2)

    recovered = fringing_mean_velocity_history(
        problem,
        forcing=forcing,
        peak_hartmann_number=peak_hartmann_number,
        entry_center=entry_center,
        exit_center=exit_center,
        transition_width=transition_width,
    )
    return {
        "peak_hartmann_number": float(peak_hartmann_number),
        "entry_center": float(entry_center),
        "exit_center": float(exit_center),
        "transition_width": float(transition_width),
        "history": history,
        "recovered_mean_velocity": recovered["mean_velocity"],
        "recovered_field_scale": recovered["field_scale"],
        "x": recovered["x"],
    }


def run_fringing_response_inverse_design(
    problem: FringingAutodiffProblem,
    *,
    target_mean_velocity: jnp.ndarray,
    target_current_proxy: jnp.ndarray,
    forcing: float,
    peak_hartmann_init: float,
    entry_center_init: float,
    exit_center_init: float,
    transition_width_init: float,
    current_weight: float = 1.0,
    learning_rate_peak_ha: float = 1.0,
    learning_rate_entry: float = 0.2,
    learning_rate_exit: float = 0.2,
    learning_rate_width: float = 0.1,
    steps: int = 16,
) -> dict[str, object]:
    peak_hartmann_number = jnp.asarray(peak_hartmann_init, dtype=jnp.float32)
    entry_center = jnp.asarray(entry_center_init, dtype=jnp.float32)
    exit_center = jnp.asarray(exit_center_init, dtype=jnp.float32)
    transition_width = jnp.asarray(transition_width_init, dtype=jnp.float32)
    history: list[dict[str, float]] = []

    for step in range(steps):
        gradients = fringing_response_loss_gradients(
            problem,
            forcing=forcing,
            peak_hartmann_number=peak_hartmann_number,
            entry_center=entry_center,
            exit_center=exit_center,
            transition_width=transition_width,
            target_mean_velocity=target_mean_velocity,
            target_current_proxy=target_current_proxy,
            current_weight=current_weight,
        )
        history.append(
            {
                "iteration": float(step),
                "peak_hartmann_number": float(peak_hartmann_number),
                "entry_center": float(entry_center),
                "exit_center": float(exit_center),
                "transition_width": float(transition_width),
                "loss": float(gradients["loss"]),
            }
        )
        peak_hartmann_number = jnp.clip(
            peak_hartmann_number - learning_rate_peak_ha * gradients["d_peak_hartmann_number"], 0.5, 60.0
        )
        entry_center = jnp.clip(entry_center - learning_rate_entry * gradients["d_entry_center"], 0.0, float(problem.x[-1]))
        exit_center = jnp.clip(exit_center - learning_rate_exit * gradients["d_exit_center"], 0.0, float(problem.x[-1]))
        transition_width = jnp.clip(
            transition_width - learning_rate_width * gradients["d_transition_width"], 0.05, float(problem.x[-1])
        )
        exit_center = jnp.maximum(exit_center, entry_center + 0.2)

    recovered = fringing_response_history(
        problem,
        forcing=forcing,
        peak_hartmann_number=peak_hartmann_number,
        entry_center=entry_center,
        exit_center=exit_center,
        transition_width=transition_width,
    )
    return {
        "peak_hartmann_number": float(peak_hartmann_number),
        "entry_center": float(entry_center),
        "exit_center": float(exit_center),
        "transition_width": float(transition_width),
        "history": history,
        "recovered_mean_velocity": recovered["mean_velocity"],
        "recovered_current_proxy": recovered["current_proxy"],
        "recovered_field_scale": recovered["field_scale"],
        "x": recovered["x"],
    }


def run_extruded_rect_inverse_design(
    problem: FringingAutodiffProblem,
    *,
    target_mean_velocity: jnp.ndarray,
    target_current_proxy: jnp.ndarray,
    target_charge_balance: jnp.ndarray,
    target_boundary_current: jnp.ndarray,
    forcing: float,
    peak_hartmann_init: float,
    entry_center_init: float,
    exit_center_init: float,
    transition_width_init: float,
    current_weight: float = 1.0,
    charge_balance_weight: float = 0.1,
    boundary_current_weight: float = 0.1,
    learning_rate_peak_ha: float = 0.8,
    learning_rate_entry: float = 0.15,
    learning_rate_exit: float = 0.15,
    learning_rate_width: float = 0.08,
    steps: int = 12,
) -> dict[str, object]:
    peak_hartmann_number = jnp.asarray(peak_hartmann_init, dtype=jnp.float32)
    entry_center = jnp.asarray(entry_center_init, dtype=jnp.float32)
    exit_center = jnp.asarray(exit_center_init, dtype=jnp.float32)
    transition_width = jnp.asarray(transition_width_init, dtype=jnp.float32)
    history: list[dict[str, float]] = []
    for step in range(steps):
        gradients = extruded_rect_response_loss_gradients(
            problem,
            forcing=forcing,
            peak_hartmann_number=peak_hartmann_number,
            entry_center=entry_center,
            exit_center=exit_center,
            transition_width=transition_width,
            target_mean_velocity=target_mean_velocity,
            target_current_proxy=target_current_proxy,
            target_charge_balance=target_charge_balance,
            target_boundary_current=target_boundary_current,
            current_weight=current_weight,
            charge_balance_weight=charge_balance_weight,
            boundary_current_weight=boundary_current_weight,
        )
        history.append(
            {
                "iteration": float(step),
                "peak_hartmann_number": float(peak_hartmann_number),
                "entry_center": float(entry_center),
                "exit_center": float(exit_center),
                "transition_width": float(transition_width),
                "loss": float(gradients["loss"]),
            }
        )
        peak_hartmann_number = jnp.clip(
            peak_hartmann_number - learning_rate_peak_ha * gradients["d_peak_hartmann_number"], 0.5, 60.0
        )
        entry_center = jnp.clip(entry_center - learning_rate_entry * gradients["d_entry_center"], 0.0, float(problem.x[-1]))
        exit_center = jnp.clip(exit_center - learning_rate_exit * gradients["d_exit_center"], 0.0, float(problem.x[-1]))
        transition_width = jnp.clip(
            transition_width - learning_rate_width * gradients["d_transition_width"], 0.05, float(problem.x[-1])
        )
        exit_center = jnp.maximum(exit_center, entry_center + 0.2)
    recovered = extruded_rect_response_history(
        problem,
        forcing=forcing,
        peak_hartmann_number=peak_hartmann_number,
        entry_center=entry_center,
        exit_center=exit_center,
        transition_width=transition_width,
    )
    return {
        "peak_hartmann_number": float(peak_hartmann_number),
        "entry_center": float(entry_center),
        "exit_center": float(exit_center),
        "transition_width": float(transition_width),
        "history": history,
        "recovered_mean_velocity": recovered["mean_velocity"],
        "recovered_current_proxy": recovered["current_proxy"],
        "recovered_charge_balance": recovered["charge_balance_residual"],
        "recovered_boundary_current": recovered["boundary_current_residual"],
        "recovered_field_scale": recovered["field_scale"],
        "x": recovered["x"],
        "model": "direct_extruded_rect",
    }


def run_extruded_rect_projection_inverse_design(
    problem: FringingAutodiffProblem,
    *,
    target_mean_velocity: jnp.ndarray,
    target_current_proxy: jnp.ndarray,
    target_charge_balance: jnp.ndarray,
    target_boundary_current: jnp.ndarray,
    target_pressure_span: jnp.ndarray,
    forcing: float,
    peak_hartmann_init: float,
    entry_center_init: float,
    exit_center_init: float,
    transition_width_init: float,
    current_weight: float = 1.0,
    charge_balance_weight: float = 0.1,
    boundary_current_weight: float = 0.1,
    pressure_span_weight: float = 0.25,
    learning_rate_peak_ha: float = 0.4,
    learning_rate_entry: float = 0.1,
    learning_rate_exit: float = 0.1,
    learning_rate_width: float = 0.05,
    steps: int = 8,
) -> dict[str, object]:
    peak_hartmann_number = jnp.asarray(peak_hartmann_init, dtype=jnp.float32)
    entry_center = jnp.asarray(entry_center_init, dtype=jnp.float32)
    exit_center = jnp.asarray(exit_center_init, dtype=jnp.float32)
    transition_width = jnp.asarray(transition_width_init, dtype=jnp.float32)
    history: list[dict[str, float]] = []
    for step in range(steps):
        gradients = extruded_rect_projection_loss_gradients(
            problem,
            forcing=forcing,
            peak_hartmann_number=peak_hartmann_number,
            entry_center=entry_center,
            exit_center=exit_center,
            transition_width=transition_width,
            target_mean_velocity=target_mean_velocity,
            target_current_proxy=target_current_proxy,
            target_charge_balance=target_charge_balance,
            target_boundary_current=target_boundary_current,
            target_pressure_span=target_pressure_span,
            current_weight=current_weight,
            charge_balance_weight=charge_balance_weight,
            boundary_current_weight=boundary_current_weight,
            pressure_span_weight=pressure_span_weight,
        )
        history.append(
            {
                "iteration": float(step),
                "peak_hartmann_number": float(peak_hartmann_number),
                "entry_center": float(entry_center),
                "exit_center": float(exit_center),
                "transition_width": float(transition_width),
                "loss": float(gradients["loss"]),
            }
        )
        peak_hartmann_number = jnp.clip(
            peak_hartmann_number - learning_rate_peak_ha * gradients["d_peak_hartmann_number"], 0.5, 60.0
        )
        entry_center = jnp.clip(entry_center - learning_rate_entry * gradients["d_entry_center"], 0.0, float(problem.x[-1]))
        exit_center = jnp.clip(exit_center - learning_rate_exit * gradients["d_exit_center"], 0.0, float(problem.x[-1]))
        transition_width = jnp.clip(
            transition_width - learning_rate_width * gradients["d_transition_width"], 0.05, float(problem.x[-1])
        )
        exit_center = jnp.maximum(exit_center, entry_center + 0.2)
    recovered = extruded_rect_projection_history(
        problem,
        forcing=forcing,
        peak_hartmann_number=peak_hartmann_number,
        entry_center=entry_center,
        exit_center=exit_center,
        transition_width=transition_width,
    )
    return {
        "peak_hartmann_number": float(peak_hartmann_number),
        "entry_center": float(entry_center),
        "exit_center": float(exit_center),
        "transition_width": float(transition_width),
        "history": history,
        "recovered_mean_velocity": recovered["mean_velocity"],
        "recovered_current_proxy": recovered["current_proxy"],
        "recovered_charge_balance": recovered["charge_balance_residual"],
        "recovered_boundary_current": recovered["boundary_current_residual"],
        "recovered_pressure_span": recovered["pressure_span"],
        "recovered_transverse_kinetic_energy": recovered["transverse_kinetic_energy"],
        "recovered_field_scale": recovered["field_scale"],
        "x": recovered["x"],
        "model": "direct_extruded_projection",
    }


def run_extruded_rect_projection_field_inverse_design(
    problem: FringingAutodiffProblem,
    *,
    target_u_field: jnp.ndarray,
    target_phi_field: jnp.ndarray,
    target_jy_field: jnp.ndarray,
    target_pressure_field: jnp.ndarray,
    station_indices: jnp.ndarray,
    forcing: float,
    peak_hartmann_init: float,
    entry_center_init: float,
    exit_center_init: float,
    transition_width_init: float,
    u_weight: float = 1.0,
    phi_weight: float = 0.25,
    jy_weight: float = 0.5,
    pressure_weight: float = 0.25,
    learning_rate_peak_ha: float = 0.4,
    learning_rate_entry: float = 0.1,
    learning_rate_exit: float = 0.1,
    learning_rate_width: float = 0.05,
    steps: int = 8,
) -> dict[str, object]:
    peak_hartmann_number = jnp.asarray(peak_hartmann_init, dtype=jnp.float32)
    entry_center = jnp.asarray(entry_center_init, dtype=jnp.float32)
    exit_center = jnp.asarray(exit_center_init, dtype=jnp.float32)
    transition_width = jnp.asarray(transition_width_init, dtype=jnp.float32)
    station_ids = jnp.asarray(station_indices, dtype=jnp.int32)
    history: list[dict[str, float]] = []
    for step in range(steps):
        gradients = extruded_rect_projection_field_loss_gradients(
            problem,
            forcing=forcing,
            peak_hartmann_number=peak_hartmann_number,
            entry_center=entry_center,
            exit_center=exit_center,
            transition_width=transition_width,
            target_u_field=target_u_field,
            target_phi_field=target_phi_field,
            target_jy_field=target_jy_field,
            target_pressure_field=target_pressure_field,
            station_indices=station_ids,
            u_weight=u_weight,
            phi_weight=phi_weight,
            jy_weight=jy_weight,
            pressure_weight=pressure_weight,
        )
        history.append(
            {
                "iteration": float(step),
                "peak_hartmann_number": float(peak_hartmann_number),
                "entry_center": float(entry_center),
                "exit_center": float(exit_center),
                "transition_width": float(transition_width),
                "loss": float(gradients["loss"]),
            }
        )
        peak_hartmann_number = jnp.clip(
            peak_hartmann_number - learning_rate_peak_ha * gradients["d_peak_hartmann_number"], 0.5, 60.0
        )
        entry_center = jnp.clip(entry_center - learning_rate_entry * gradients["d_entry_center"], 0.0, float(problem.x[-1]))
        exit_center = jnp.clip(exit_center - learning_rate_exit * gradients["d_exit_center"], 0.0, float(problem.x[-1]))
        transition_width = jnp.clip(
            transition_width - learning_rate_width * gradients["d_transition_width"], 0.05, float(problem.x[-1])
        )
        exit_center = jnp.maximum(exit_center, entry_center + 0.2)
    recovered = extruded_rect_projection_history(
        problem,
        forcing=forcing,
        peak_hartmann_number=peak_hartmann_number,
        entry_center=entry_center,
        exit_center=exit_center,
        transition_width=transition_width,
    )
    return {
        "peak_hartmann_number": float(peak_hartmann_number),
        "entry_center": float(entry_center),
        "exit_center": float(exit_center),
        "transition_width": float(transition_width),
        "history": history,
        "station_indices": jnp.asarray(station_ids).tolist(),
        "recovered_u_field": recovered["u_field"][station_ids],
        "recovered_phi_field": recovered["phi_field"][station_ids],
        "recovered_jy_field": recovered["jy_field"][station_ids],
        "recovered_pressure_field": recovered["pressure_field"][station_ids],
        "x": recovered["x"],
        "field_scale": recovered["field_scale"],
        "model": "direct_extruded_projection_fields",
    }


def run_extruded_rect_projection_trajectory_inverse_design(
    problem: FringingAutodiffProblem,
    *,
    target_u_history: jnp.ndarray,
    target_phi_history: jnp.ndarray,
    target_jy_history: jnp.ndarray,
    target_pressure_history: jnp.ndarray,
    target_charge_balance_history: jnp.ndarray,
    target_boundary_current_history: jnp.ndarray,
    station_indices: jnp.ndarray,
    forcing: float,
    peak_hartmann_init: float,
    entry_center_init: float,
    exit_center_init: float,
    transition_width_init: float,
    u_weight: float = 1.0,
    phi_weight: float = 0.25,
    jy_weight: float = 0.5,
    pressure_weight: float = 0.25,
    charge_balance_weight: float = 0.1,
    boundary_current_weight: float = 0.1,
    learning_rate_peak_ha: float = 0.35,
    learning_rate_entry: float = 0.08,
    learning_rate_exit: float = 0.08,
    learning_rate_width: float = 0.04,
    steps: int = 8,
) -> dict[str, object]:
    peak_hartmann_number = jnp.asarray(peak_hartmann_init, dtype=jnp.float32)
    entry_center = jnp.asarray(entry_center_init, dtype=jnp.float32)
    exit_center = jnp.asarray(exit_center_init, dtype=jnp.float32)
    transition_width = jnp.asarray(transition_width_init, dtype=jnp.float32)
    station_ids = jnp.asarray(station_indices, dtype=jnp.int32)
    history: list[dict[str, float]] = []
    for step in range(steps):
        gradients = extruded_rect_projection_trajectory_loss_gradients(
            problem,
            forcing=forcing,
            peak_hartmann_number=peak_hartmann_number,
            entry_center=entry_center,
            exit_center=exit_center,
            transition_width=transition_width,
            target_u_history=target_u_history,
            target_phi_history=target_phi_history,
            target_jy_history=target_jy_history,
            target_pressure_history=target_pressure_history,
            target_charge_balance_history=target_charge_balance_history,
            target_boundary_current_history=target_boundary_current_history,
            station_indices=station_ids,
            u_weight=u_weight,
            phi_weight=phi_weight,
            jy_weight=jy_weight,
            pressure_weight=pressure_weight,
            charge_balance_weight=charge_balance_weight,
            boundary_current_weight=boundary_current_weight,
        )
        history.append(
            {
                "iteration": float(step),
                "peak_hartmann_number": float(peak_hartmann_number),
                "entry_center": float(entry_center),
                "exit_center": float(exit_center),
                "transition_width": float(transition_width),
                "loss": float(gradients["loss"]),
            }
        )
        peak_hartmann_number = jnp.clip(
            peak_hartmann_number - learning_rate_peak_ha * gradients["d_peak_hartmann_number"], 0.5, 60.0
        )
        entry_center = jnp.clip(entry_center - learning_rate_entry * gradients["d_entry_center"], 0.0, float(problem.x[-1]))
        exit_center = jnp.clip(exit_center - learning_rate_exit * gradients["d_exit_center"], 0.0, float(problem.x[-1]))
        transition_width = jnp.clip(
            transition_width - learning_rate_width * gradients["d_transition_width"], 0.05, float(problem.x[-1])
        )
        exit_center = jnp.maximum(exit_center, entry_center + 0.2)
    recovered = extruded_rect_projection_iteration_history(
        problem,
        forcing=forcing,
        peak_hartmann_number=peak_hartmann_number,
        entry_center=entry_center,
        exit_center=exit_center,
        transition_width=transition_width,
        station_indices=station_ids,
    )
    return {
        "peak_hartmann_number": float(peak_hartmann_number),
        "entry_center": float(entry_center),
        "exit_center": float(exit_center),
        "transition_width": float(transition_width),
        "history": history,
        "station_indices": jnp.asarray(station_ids).tolist(),
        "recovered_u_history": recovered["u_field"],
        "recovered_phi_history": recovered["phi_field"],
        "recovered_jy_history": recovered["jy_field"],
        "recovered_pressure_history": recovered["pressure_field"],
        "recovered_charge_balance_history": recovered["charge_balance_residual"],
        "recovered_boundary_current_history": recovered["boundary_current_residual"],
        "x": recovered["x"],
        "field_scale": recovered["field_scale"],
        "model": "direct_extruded_projection_trajectory",
    }


def build_extruded_response_targets(extruded_solution) -> dict[str, jnp.ndarray]:
    bundle = extruded_solution.bundle
    return {
        "x": jnp.asarray(bundle.x, dtype=jnp.float32),
        "field_scale": jnp.asarray(bundle.field_scale, dtype=jnp.float32),
        "mean_velocity": jnp.asarray(bundle.mean_velocity, dtype=jnp.float32),
        "current_proxy": jnp.asarray(bundle.current_scaled_pressure_proxy, dtype=jnp.float32),
        "charge_balance_residual": jnp.asarray(bundle.charge_balance_residual, dtype=jnp.float32),
        "boundary_current_residual": jnp.asarray(bundle.boundary_current_residual, dtype=jnp.float32),
        "wall_current_leakage": jnp.asarray(bundle.wall_current_leakage, dtype=jnp.float32),
        "axial_current": jnp.asarray(bundle.axial_current, dtype=jnp.float32),
        "pressure_span": jnp.asarray(jnp.max(bundle.p, axis=(1, 2)) - jnp.min(bundle.p, axis=(1, 2)), dtype=jnp.float32),
        "transverse_kinetic_energy": jnp.asarray(jnp.mean(bundle.v**2 + bundle.w**2, axis=(1, 2)), dtype=jnp.float32),
        "u_field": jnp.asarray(bundle.u, dtype=jnp.float32),
        "phi_field": jnp.asarray(bundle.phi, dtype=jnp.float32),
        "jy_field": jnp.asarray(bundle.jy, dtype=jnp.float32),
        "pressure_field": jnp.asarray(bundle.p, dtype=jnp.float32),
    }


def run_extruded_target_inverse_design(
    extruded_solution,
    *,
    ny: int = 12,
    nz: int = 12,
    potential_iterations: int = 12,
    velocity_iterations: int = 16,
    macro_iterations: int = 3,
    peak_hartmann_init: float = 10.0,
    entry_center_init: float = 1.0,
    exit_center_init: float = 5.0,
    transition_width_init: float = 0.7,
    current_weight: float = 0.5,
    steps: int = 16,
) -> dict[str, object]:
    targets = build_extruded_response_targets(extruded_solution)
    x = targets["x"]
    problem = build_fringing_autodiff_problem(
        nx_stations=int(x.shape[0]),
        length=float(x[-1] - x[0]) if x.shape[0] > 1 else 1.0,
        ny=ny,
        nz=nz,
        potential_iterations=potential_iterations,
        velocity_iterations=velocity_iterations,
        macro_iterations=macro_iterations,
    )
    if str(extruded_solution.problem.case.geometry.kind) == "rect_duct":
        result = run_extruded_rect_projection_inverse_design(
            problem,
            target_mean_velocity=targets["mean_velocity"],
            target_current_proxy=targets["current_proxy"],
            target_charge_balance=targets["charge_balance_residual"],
            target_boundary_current=targets["boundary_current_residual"],
            target_pressure_span=targets["pressure_span"],
            forcing=float(extruded_solution.problem.case.forcing),
            peak_hartmann_init=peak_hartmann_init,
            entry_center_init=entry_center_init,
            exit_center_init=exit_center_init,
            transition_width_init=transition_width_init,
            current_weight=current_weight,
            steps=steps,
        )
    else:
        result = run_fringing_response_inverse_design(
            problem,
            target_mean_velocity=targets["mean_velocity"],
            target_current_proxy=targets["current_proxy"],
            forcing=float(extruded_solution.problem.case.forcing),
            peak_hartmann_init=peak_hartmann_init,
            entry_center_init=entry_center_init,
            exit_center_init=exit_center_init,
            transition_width_init=transition_width_init,
            current_weight=current_weight,
            steps=steps,
        )
        result["model"] = "fringing_response_surrogate"
    return {
        "target": targets,
        "recovered": result,
        "geometry_kind": extruded_solution.problem.case.geometry.kind,
        "forcing": float(extruded_solution.problem.case.forcing),
    }

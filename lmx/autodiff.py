from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .linear import solve_poisson_jacobi_state
from .mesh import StructuredMesh, generate_rect_duct_mesh
from .solvers import (
    _enforce_velocity_bc,
    _face_emf_y,
    _face_emf_z,
    _fully_developed_rhs,
    _potential_coefficients,
    _velocity_system_coefficients,
)


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

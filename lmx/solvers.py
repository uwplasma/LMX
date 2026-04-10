from __future__ import annotations

from dataclasses import asdict
import numpy as np

import jax
import jax.numpy as jnp

from .core import Diagnostics, MHDState, Solution
from .linear import (
    poisson_residual_norm,
    solve_five_point_system,
    solve_poisson_cg_state,
    solve_poisson_jacobi_state,
    solve_poisson_lineax,
)
from .mesh import StructuredMesh, generate_layered_duct_mesh, generate_rect_duct_mesh
from .operators import gradient_scalar, laplacian_scalar
from .physics import build_material_fields, magnetic_field_components
from .runtime_logging import RestartLogInfo, SolverStepRecord
from .specs import BoundaryCondition, CaseSpec


def _build_mesh(case: CaseSpec) -> StructuredMesh:
    g = case.geometry
    if g.kind == "rect_duct":
        return generate_rect_duct_mesh(width=g.width, height=g.height, length=g.length, nx=g.nx, ny=g.ny, nz=g.nz)
    if g.kind == "layered_duct":
        return generate_layered_duct_mesh(
            width=g.width,
            height=g.height,
            length=g.length,
            nx=g.nx,
            ny=g.ny,
            nz=g.nz,
            wall_thickness=g.wall_thickness,
            wall_cells=g.wall_cells,
            target_ha=g.target_ha,
        )
    raise NotImplementedError(f"Geometry {g.kind} is not supported by the laminar solver yet.")


def _interface_conductance_y(mesh: StructuredMesh, sigma: jnp.ndarray) -> jnp.ndarray:
    left_distance = 0.5 * mesh.dy[:-1, None]
    right_distance = 0.5 * mesh.dy[1:, None]
    sigma_left = jnp.maximum(sigma[:-1, :], 1e-12)
    sigma_right = jnp.maximum(sigma[1:, :], 1e-12)
    return 1.0 / jnp.maximum(left_distance / sigma_left + right_distance / sigma_right, 1e-12)


def _face_conductance_y(mesh: StructuredMesh, sigma: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    conductance = _interface_conductance_y(mesh, sigma)
    west = jnp.pad(conductance, ((1, 0), (0, 0))) / mesh.dy[:, None]
    east = jnp.pad(conductance, ((0, 1), (0, 0))) / mesh.dy[:, None]
    return west, east


def _interface_conductance_z(mesh: StructuredMesh, sigma: jnp.ndarray) -> jnp.ndarray:
    left_distance = 0.5 * mesh.dz[None, :-1]
    right_distance = 0.5 * mesh.dz[None, 1:]
    sigma_left = jnp.maximum(sigma[:, :-1], 1e-12)
    sigma_right = jnp.maximum(sigma[:, 1:], 1e-12)
    return 1.0 / jnp.maximum(left_distance / sigma_left + right_distance / sigma_right, 1e-12)


def _face_conductance_z(mesh: StructuredMesh, sigma: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    conductance = _interface_conductance_z(mesh, sigma)
    south = jnp.pad(conductance, ((0, 0), (1, 0))) / mesh.dz[None, :]
    north = jnp.pad(conductance, ((0, 0), (0, 1))) / mesh.dz[None, :]
    return south, north


def _face_emf_y(mesh: StructuredMesh, sigma: jnp.ndarray, source: jnp.ndarray) -> jnp.ndarray:
    left_distance = 0.5 * mesh.dy[:-1, None]
    right_distance = 0.5 * mesh.dy[1:, None]
    conductance = _interface_conductance_y(mesh, sigma)
    return conductance * (left_distance * source[:-1, :] + right_distance * source[1:, :])


def _face_emf_z(mesh: StructuredMesh, sigma: jnp.ndarray, source: jnp.ndarray) -> jnp.ndarray:
    left_distance = 0.5 * mesh.dz[None, :-1]
    right_distance = 0.5 * mesh.dz[None, 1:]
    conductance = _interface_conductance_z(mesh, sigma)
    return conductance * (left_distance * source[:, :-1] + right_distance * source[:, 1:])


def _potential_coefficients(mesh: StructuredMesh, sigma: jnp.ndarray) -> tuple[jnp.ndarray, ...]:
    west, east = _face_conductance_y(mesh, sigma)
    south, north = _face_conductance_z(mesh, sigma)
    diagonal = west + east + south + north
    diagonal = jnp.where(diagonal > 0.0, diagonal, 1.0)
    return diagonal, west, east, south, north


def _cell_metric(mesh: StructuredMesh) -> jnp.ndarray:
    return mesh.dy[:, None] * mesh.dz[None, :]


def _active_velocity_mask_for_solver(fluid_mask: jnp.ndarray, solver_kind: str) -> jnp.ndarray:
    if solver_kind == "fully_developed_inductionless":
        return _active_velocity_mask(fluid_mask)
    return fluid_mask


def _connected_interface_diffusivity_y(mesh: StructuredMesh, diffusivity: jnp.ndarray, active_mask: jnp.ndarray) -> jnp.ndarray:
    connected = active_mask[:-1, :] & active_mask[1:, :]
    left_distance = 0.5 * mesh.dy[:-1, None]
    right_distance = 0.5 * mesh.dy[1:, None]
    diffusivity_left = jnp.maximum(diffusivity[:-1, :], 1e-12)
    diffusivity_right = jnp.maximum(diffusivity[1:, :], 1e-12)
    conductance = 1.0 / jnp.maximum(left_distance / diffusivity_left + right_distance / diffusivity_right, 1e-12)
    return jnp.where(connected, conductance, 0.0)


def _connected_interface_diffusivity_z(mesh: StructuredMesh, diffusivity: jnp.ndarray, active_mask: jnp.ndarray) -> jnp.ndarray:
    connected = active_mask[:, :-1] & active_mask[:, 1:]
    left_distance = 0.5 * mesh.dz[None, :-1]
    right_distance = 0.5 * mesh.dz[None, 1:]
    diffusivity_left = jnp.maximum(diffusivity[:, :-1], 1e-12)
    diffusivity_right = jnp.maximum(diffusivity[:, 1:], 1e-12)
    conductance = 1.0 / jnp.maximum(left_distance / diffusivity_left + right_distance / diffusivity_right, 1e-12)
    return jnp.where(connected, conductance, 0.0)


def _velocity_system_coefficients(
    mesh: StructuredMesh,
    diffusivity: jnp.ndarray,
    reaction: jnp.ndarray,
    active_mask: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    dy = mesh.dy[:, None]
    dz = mesh.dz[None, :]
    active = active_mask.astype(diffusivity.dtype)

    west_connected = active_mask & jnp.pad(active_mask[:-1, :], ((1, 0), (0, 0)))
    east_connected = active_mask & jnp.pad(active_mask[1:, :], ((0, 1), (0, 0)))
    south_connected = active_mask & jnp.pad(active_mask[:, :-1], ((0, 0), (1, 0)))
    north_connected = active_mask & jnp.pad(active_mask[:, 1:], ((0, 0), (0, 1)))

    interface_y = _connected_interface_diffusivity_y(mesh, diffusivity, active_mask)
    interface_z = _connected_interface_diffusivity_z(mesh, diffusivity, active_mask)
    west = jnp.where(
        west_connected,
        jnp.pad(interface_y, ((1, 0), (0, 0))) / jnp.maximum(dy, 1e-12),
        jnp.where(active_mask, 2.0 * diffusivity / jnp.maximum(dy**2, 1e-12), 0.0),
    )
    east = jnp.where(
        east_connected,
        jnp.pad(interface_y, ((0, 1), (0, 0))) / jnp.maximum(dy, 1e-12),
        jnp.where(active_mask, 2.0 * diffusivity / jnp.maximum(dy**2, 1e-12), 0.0),
    )
    south = jnp.where(
        south_connected,
        jnp.pad(interface_z, ((0, 0), (1, 0))) / jnp.maximum(dz, 1e-12),
        jnp.where(active_mask, 2.0 * diffusivity / jnp.maximum(dz**2, 1e-12), 0.0),
    )
    north = jnp.where(
        north_connected,
        jnp.pad(interface_z, ((0, 0), (0, 1))) / jnp.maximum(dz, 1e-12),
        jnp.where(active_mask, 2.0 * diffusivity / jnp.maximum(dz**2, 1e-12), 0.0),
    )
    diagonal = west + east + south + north + jnp.where(active_mask, reaction, 1.0 - active)
    diagonal = jnp.where(active_mask, diagonal, 1.0)
    west = jnp.where(active_mask, west, 0.0)
    east = jnp.where(active_mask, east, 0.0)
    south = jnp.where(active_mask, south, 0.0)
    north = jnp.where(active_mask, north, 0.0)
    return diagonal, west, east, south, north


def _solve_velocity_system(
    *,
    mesh: StructuredMesh,
    diffusivity: jnp.ndarray,
    reaction: jnp.ndarray,
    rhs: jnp.ndarray,
    active_mask: jnp.ndarray,
    linear_solver: str,
    preconditioner: str,
    max_steps: int,
    tolerance: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    diagonal, west, east, south, north = _velocity_system_coefficients(mesh, diffusivity, reaction, active_mask)
    rhs_masked = jnp.where(active_mask, rhs, 0.0)
    cell_metric = _cell_metric(mesh).astype(rhs_masked.dtype)
    field, info = solve_five_point_system(
        diagonal * cell_metric,
        west * cell_metric,
        east * cell_metric,
        south * cell_metric,
        north * cell_metric,
        rhs_masked * cell_metric,
        linear_solver=linear_solver,
        preconditioner=preconditioner,
        tolerance=tolerance,
        max_steps=max_steps,
    )
    field = jnp.where(active_mask, field, 0.0)
    return field, jnp.asarray(info.residual, dtype=rhs.dtype), jnp.asarray(info.iterations, dtype=jnp.int32)


def _fully_developed_rhs(
    *,
    mesh: StructuredMesh,
    sigma: jnp.ndarray,
    rho: jnp.ndarray,
    fluid_mask: jnp.ndarray,
    phi: jnp.ndarray,
    by: jnp.ndarray,
    bz: jnp.ndarray,
    forcing: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    dphi_dy, dphi_dz = gradient_scalar(phi, mesh)
    lorentz_source = sigma * (-(bz * dphi_dy) - (by * dphi_dz))
    rhs = jnp.where(fluid_mask, (forcing + lorentz_source) / rho, 0.0)
    return rhs, lorentz_source


def _volume_scaled_potential_system(
    mesh: StructuredMesh,
    diagonal: jnp.ndarray,
    west: jnp.ndarray,
    east: jnp.ndarray,
    south: jnp.ndarray,
    north: jnp.ndarray,
    rhs: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    cell_metric = _cell_metric(mesh)
    return (
        diagonal * cell_metric,
        west * cell_metric,
        east * cell_metric,
        south * cell_metric,
        north * cell_metric,
        rhs * cell_metric,
    )


def _solve_potential(
    mesh: StructuredMesh,
    sigma: jnp.ndarray,
    fluid_mask: jnp.ndarray,
    u: jnp.ndarray,
    by: jnp.ndarray,
    bz: jnp.ndarray,
    anchor: tuple[int, int],
    iterations: int,
    tolerance: float | None = None,
    relaxation: float = 1.0,
    solver: str = "jacobi",
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    uxb_y = jnp.where(fluid_mask, -u * bz, 0.0)
    uxb_z = jnp.where(fluid_mask, u * by, 0.0)
    conv_y = _face_emf_y(mesh, sigma, uxb_y)
    conv_z = _face_emf_z(mesh, sigma, uxb_z)
    face_conv_y = jnp.pad(conv_y, ((1, 1), (0, 0)))
    face_conv_z = jnp.pad(conv_z, ((0, 0), (1, 1)))
    rhs = -(
        (face_conv_y[1:, :] - face_conv_y[:-1, :]) / mesh.dy[:, None]
        + (face_conv_z[:, 1:] - face_conv_z[:, :-1]) / mesh.dz[None, :]
    )

    diagonal, west, east, south, north = _potential_coefficients(mesh, sigma)
    if solver == "jacobi":
        phi, residual, iteration_count = solve_poisson_jacobi_state(
            diagonal,
            west,
            east,
            south,
            north,
            rhs,
            anchor,
            iterations,
            tolerance=tolerance,
            relaxation=relaxation,
        )
    elif solver == "cg":
        phi, residual, iteration_count = solve_poisson_cg_state(
            diagonal,
            west,
            east,
            south,
            north,
            rhs,
            anchor,
            iterations,
            tolerance=tolerance,
        )
    elif solver == "cg_volume":
        diagonal_scaled, west_scaled, east_scaled, south_scaled, north_scaled, rhs_scaled = _volume_scaled_potential_system(
            mesh,
            diagonal,
            west,
            east,
            south,
            north,
            rhs,
        )
        phi, _, iteration_count = solve_poisson_cg_state(
            diagonal_scaled,
            west_scaled,
            east_scaled,
            south_scaled,
            north_scaled,
            rhs_scaled,
            anchor,
            iterations,
            tolerance=tolerance,
        )
        residual = poisson_residual_norm(diagonal, west, east, south, north, rhs, phi, anchor)
    elif solver == "lineax_cg":
        phi, info = solve_poisson_lineax(
            diagonal,
            west,
            east,
            south,
            north,
            rhs,
            anchor,
            fallback_iterations=iterations,
            max_steps=iterations,
        )
        residual = jnp.asarray(info.residual, dtype=rhs.dtype)
        iteration_count = jnp.asarray(info.iterations, dtype=jnp.int32)
    else:
        raise ValueError(f"Unsupported potential solver backend {solver!r}")
    return phi, residual, iteration_count


def _resolve_potential_solver(solver: str, fluid_mask: jnp.ndarray | None) -> str:
    if solver != "auto":
        return solver
    if fluid_mask is None:
        return "cg"
    return "cg" if bool(np.asarray(fluid_mask).all()) else "cg_volume"


def _compute_current_and_lorentz(
    mesh: StructuredMesh,
    sigma: jnp.ndarray,
    fluid_mask: jnp.ndarray,
    u: jnp.ndarray,
    phi: jnp.ndarray,
    by: jnp.ndarray,
    bz: jnp.ndarray,
    reconstruction: str = "cell_centered",
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    use_face_currents = reconstruction in {"face_averaged", "hybrid_face_lorentz"}
    if use_face_currents:
        uxb_y = jnp.where(fluid_mask, -u * bz, 0.0)
        uxb_z = jnp.where(fluid_mask, u * by, 0.0)
        face_jy = _interface_conductance_y(mesh, sigma) * (phi[:-1, :] - phi[1:, :]) + _face_emf_y(mesh, sigma, uxb_y)
        face_jz = _interface_conductance_z(mesh, sigma) * (phi[:, :-1] - phi[:, 1:]) + _face_emf_z(mesh, sigma, uxb_z)
        face_jy_centered = 0.5 * (jnp.pad(face_jy, ((1, 0), (0, 0))) + jnp.pad(face_jy, ((0, 1), (0, 0))))
        face_jz_centered = 0.5 * (jnp.pad(face_jz, ((0, 0), (1, 0))) + jnp.pad(face_jz, ((0, 0), (0, 1))))
    if reconstruction == "face_averaged":
        jy = face_jy_centered
        jz = face_jz_centered
        lorentz_x = jy * bz - jz * by
    else:
        dphi_dy, dphi_dz = gradient_scalar(phi, mesh)
        jy = sigma * (-dphi_dy - u * bz)
        jz = sigma * (-dphi_dz + u * by)
        lorentz_x = jy * bz - jz * by
        if reconstruction == "hybrid_face_lorentz":
            lorentz_x = face_jy_centered * bz - face_jz_centered * by
    jy = jnp.where(fluid_mask, jy, 0.0)
    jz = jnp.where(fluid_mask, jz, 0.0)
    lorentz_x = jnp.where(fluid_mask, lorentz_x, 0.0)
    return jy, jz, lorentz_x


def _face_current_components(
    mesh: StructuredMesh,
    sigma: jnp.ndarray,
    fluid_mask: jnp.ndarray,
    u: jnp.ndarray,
    phi: jnp.ndarray,
    by: jnp.ndarray,
    bz: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    uxb_y = jnp.where(fluid_mask, -u * bz, 0.0)
    uxb_z = jnp.where(fluid_mask, u * by, 0.0)
    emf_y = _face_emf_y(mesh, sigma, uxb_y)
    emf_z = _face_emf_z(mesh, sigma, uxb_z)
    face_jy = _interface_conductance_y(mesh, sigma) * (phi[:-1, :] - phi[1:, :]) + emf_y
    face_jz = _interface_conductance_z(mesh, sigma) * (phi[:, :-1] - phi[:, 1:]) + emf_z
    return face_jy, face_jz, emf_y, emf_z


def _face_current_emf_and_lorentz_max(
    mesh: StructuredMesh,
    sigma: jnp.ndarray,
    fluid_mask: jnp.ndarray,
    u: jnp.ndarray,
    phi: jnp.ndarray,
    by: jnp.ndarray,
    bz: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    face_jy, face_jz, emf_y, emf_z = _face_current_components(mesh, sigma, fluid_mask, u, phi, by, bz)
    max_face_current = jnp.maximum(jnp.max(jnp.abs(face_jy)), jnp.max(jnp.abs(face_jz)))
    max_emf = jnp.maximum(jnp.max(jnp.abs(emf_y)), jnp.max(jnp.abs(emf_z)))
    face_bz = 0.5 * (bz[:-1, :] + bz[1:, :])
    face_by = 0.5 * (by[:, :-1] + by[:, 1:])
    max_face_lorentz = jnp.maximum(
        jnp.max(jnp.abs(face_jy * face_bz)),
        jnp.max(jnp.abs(face_jz * face_by)),
    )
    return max_face_current, max_emf, max_face_lorentz


def _integral_diagnostics(
    *,
    mesh: StructuredMesh,
    sigma: jnp.ndarray,
    fluid_mask: jnp.ndarray,
    u: jnp.ndarray,
    phi: jnp.ndarray,
    jy: jnp.ndarray,
    jz: jnp.ndarray,
    lorentz: jnp.ndarray,
    by: jnp.ndarray,
    bz: jnp.ndarray,
    anchor: tuple[int, int],
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    cell_metric = _cell_metric(mesh).astype(u.dtype)
    fluid_weight = jnp.where(fluid_mask, cell_metric, 0.0)
    fluid_total_weight = jnp.maximum(jnp.sum(fluid_weight), 1e-20)
    volumetric_flow_rate = jnp.sum(fluid_weight * u)
    mean_current_magnitude = jnp.sum(fluid_weight * jnp.sqrt(jy**2 + jz**2)) / fluid_total_weight
    lorentz_power = jnp.sum(fluid_weight * lorentz * u)
    face_jy, face_jz, _, _ = _face_current_components(mesh, sigma, fluid_mask, u, phi, by, bz)
    padded_face_jy = jnp.pad(face_jy, ((1, 1), (0, 0)))
    padded_face_jz = jnp.pad(face_jz, ((0, 0), (1, 1)))
    div_current = (
        (padded_face_jy[1:, :] - padded_face_jy[:-1, :]) / mesh.dy[:, None]
        + (padded_face_jz[:, 1:] - padded_face_jz[:, :-1]) / mesh.dz[None, :]
    )
    div_current_max = jnp.max(jnp.where(fluid_mask, jnp.abs(div_current), 0.0))
    gauge_residual = jnp.abs(phi[anchor])
    conductivity_jump_y = jnp.abs(sigma[:-1, :] - sigma[1:, :]) > 1e-12
    conductivity_jump_z = jnp.abs(sigma[:, :-1] - sigma[:, 1:]) > 1e-12
    interface_mask_y = conductivity_jump_y | (fluid_mask[:-1, :] != fluid_mask[1:, :])
    interface_mask_z = conductivity_jump_z | (fluid_mask[:, :-1] != fluid_mask[:, 1:])
    interface_residual_y = jnp.where(
        interface_mask_y,
        jnp.abs(face_jy - 0.5 * (jy[:-1, :] + jy[1:, :])),
        0.0,
    )
    interface_residual_z = jnp.where(
        interface_mask_z,
        jnp.abs(face_jz - 0.5 * (jz[:, :-1] + jz[:, 1:])),
        0.0,
    )
    interface_current_residual = jnp.maximum(
        jnp.max(interface_residual_y, initial=0.0),
        jnp.max(interface_residual_z, initial=0.0),
    )
    return (
        volumetric_flow_rate,
        mean_current_magnitude,
        lorentz_power,
        div_current_max,
        gauge_residual,
        interface_current_residual,
    )


def _enforce_velocity_bc(
    u: jnp.ndarray,
    mesh: StructuredMesh,
    fluid_mask: jnp.ndarray,
    *,
    interpolate_direct_fluid_walls: bool = False,
) -> jnp.ndarray:
    u = jnp.where(fluid_mask, u, 0.0)
    if not interpolate_direct_fluid_walls:
        u = u.at[0, :].set(0.0)
        u = u.at[-1, :].set(0.0)
        u = u.at[:, 0].set(0.0)
        u = u.at[:, -1].set(0.0)
        return u
    direct_west = fluid_mask[0, :]
    direct_east = fluid_mask[-1, :]
    direct_south = fluid_mask[:, 0]
    direct_north = fluid_mask[:, -1]
    if u.shape[0] > 1:
        west_ratio = (mesh.y_centers[0] - mesh.y_faces[0]) / jnp.maximum(mesh.y_centers[1] - mesh.y_faces[0], 1e-12)
        east_ratio = (mesh.y_faces[-1] - mesh.y_centers[-1]) / jnp.maximum(mesh.y_faces[-1] - mesh.y_centers[-2], 1e-12)
        west_scale = jnp.where(
            direct_west,
            west_ratio * fluid_mask[1, :].astype(u.dtype),
            0.0,
        )
        east_scale = jnp.where(
            direct_east,
            east_ratio * fluid_mask[-2, :].astype(u.dtype),
            0.0,
        )
        u = u.at[0, :].set(jnp.where(direct_west, west_scale * u[1, :], u[0, :]))
        u = u.at[-1, :].set(jnp.where(direct_east, east_scale * u[-2, :], u[-1, :]))
    if u.shape[1] > 1:
        south_ratio = (mesh.z_centers[0] - mesh.z_faces[0]) / jnp.maximum(mesh.z_centers[1] - mesh.z_faces[0], 1e-12)
        north_ratio = (mesh.z_faces[-1] - mesh.z_centers[-1]) / jnp.maximum(mesh.z_faces[-1] - mesh.z_centers[-2], 1e-12)
        south_scale = jnp.where(
            direct_south,
            south_ratio * fluid_mask[:, 1].astype(u.dtype),
            0.0,
        )
        north_scale = jnp.where(
            direct_north,
            north_ratio * fluid_mask[:, -2].astype(u.dtype),
            0.0,
        )
        u = u.at[:, 0].set(jnp.where(direct_south, south_scale * u[:, 1], u[:, 0]))
        u = u.at[:, -1].set(jnp.where(direct_north, north_scale * u[:, -2], u[:, -1]))
    return u


def _limited_velocity_update(
    current: jnp.ndarray,
    trial: jnp.ndarray,
    fluid_mask: jnp.ndarray,
    max_delta: float = 1e-3,
    limiter: str = "global_scale",
) -> jnp.ndarray:
    delta = jnp.where(fluid_mask, trial - current, 0.0)
    if limiter == "local_clip":
        clipped_delta = jnp.clip(delta, -max_delta, max_delta)
        return jnp.where(fluid_mask, current + clipped_delta, 0.0)
    if limiter != "global_scale":
        raise ValueError(f"Unsupported velocity update limiter {limiter!r}")
    peak_delta = jnp.max(jnp.abs(delta))
    scale = jnp.minimum(1.0, max_delta / jnp.maximum(peak_delta, 1e-12))
    return jnp.where(fluid_mask, current + scale * delta, 0.0)


def _velocity_update_statistics(
    current: jnp.ndarray,
    trial: jnp.ndarray,
    fluid_mask: jnp.ndarray,
    *,
    max_delta: float,
    limiter: str,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    delta = jnp.where(fluid_mask, trial - current, 0.0)
    peak_delta = jnp.max(jnp.abs(delta))
    active_count = jnp.maximum(jnp.sum(fluid_mask.astype(delta.dtype)), 1.0)
    limited_fraction = jnp.sum(
        jnp.where(fluid_mask, (jnp.abs(delta) > max_delta).astype(delta.dtype), 0.0)
    ) / active_count
    if limiter not in {"global_scale", "local_clip"}:
        raise ValueError(f"Unsupported velocity update limiter {limiter!r}")
    scale = jnp.minimum(1.0, max_delta / jnp.maximum(peak_delta, 1e-12))
    return peak_delta, scale, limited_fraction


def _active_velocity_mask(fluid_mask: jnp.ndarray) -> jnp.ndarray:
    active = jnp.array(fluid_mask, copy=True)
    active = active.at[0, :].set(False)
    active = active.at[-1, :].set(False)
    active = active.at[:, 0].set(False)
    active = active.at[:, -1].set(False)
    return active


def _inlet_speed(boundary: BoundaryCondition, case: CaseSpec) -> float | None:
    if boundary.kind == "inlet_velocity":
        value = boundary.value
        if isinstance(value, tuple):
            axis = (boundary.axis or "x").lower()
            component = {"x": 0, "y": 1, "z": 2}.get(axis, 0)
            return float(value[component])
        if isinstance(value, (int, float)):
            return float(value)
    if boundary.kind == "inlet_flow_rate" and isinstance(boundary.value, (int, float)):
        area = case.geometry.width * case.geometry.height
        if area > 0.0:
            return float(boundary.value) / area
    return None


def _target_mean_velocity(case: CaseSpec) -> float | None:
    if abs(case.forcing) != 0.0:
        return None
    for boundary in case.boundary_conditions:
        if boundary.kind != "inlet_flow_rate":
            continue
        speed = _inlet_speed(boundary, case)
        if speed is not None:
            return speed
    return None


def _reference_mean_velocity(case: CaseSpec) -> float | None:
    for boundary in case.boundary_conditions:
        speed = _inlet_speed(boundary, case)
        if speed is not None:
            return speed
    if abs(case.initial_velocity) > 0.0:
        return float(case.initial_velocity)
    return None


def _explicit_forcing(explicit_forcing: float, dtype: jnp.dtype) -> jnp.ndarray:
    return jnp.asarray(explicit_forcing, dtype=dtype)


def _step(
    u: jnp.ndarray,
    mesh: StructuredMesh,
    sigma: jnp.ndarray,
    rho: jnp.ndarray,
    nu: jnp.ndarray,
    fluid_mask: jnp.ndarray,
    by: jnp.ndarray,
    bz: jnp.ndarray,
    dt: float,
    forcing: float,
    target_mean_velocity: float | None,
    reference_mean_velocity: float | None,
    anchor: tuple[int, int],
    outer_iterations: int,
    potential_iterations: int,
    potential_tolerance: float | None,
    potential_relaxation: float,
    potential_solver: str,
    relaxation: float,
    velocity_update_limit: float,
    velocity_update_limiter: str,
    current_reconstruction: str,
    post_update_potential_refresh: bool,
    interpolate_direct_fluid_walls: bool,
) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    float,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
]:
    def outer_body(_, carry):
        u_iter = carry[0]
        phi, potential_residual, potential_iteration_count = _solve_potential(
            mesh,
            sigma,
            fluid_mask,
            u_iter,
            by,
            bz,
            anchor,
            potential_iterations,
            tolerance=potential_tolerance,
            relaxation=potential_relaxation,
            solver=potential_solver,
        )
        phi = jnp.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0)
        jy, jz, lorentz = _compute_current_and_lorentz(
            mesh,
            sigma,
            fluid_mask,
            u_iter,
            phi,
            by,
            bz,
            reconstruction=current_reconstruction,
        )
        lorentz = jnp.nan_to_num(lorentz, nan=0.0, posinf=0.0, neginf=0.0)
        lap_u = laplacian_scalar(u_iter, mesh, mask=fluid_mask)
        lorentz_damping = jnp.where(fluid_mask, sigma * (by**2 + bz**2), 0.0)
        lorentz_explicit = lorentz + lorentz_damping * u_iter
        implicit_scale = 1.0 + dt * lorentz_damping / rho
        base_trial = jnp.where(fluid_mask, (u + dt * (nu * lap_u + lorentz_explicit / rho)) / implicit_scale, 0.0)
        pressure_sensitivity = jnp.where(fluid_mask, (dt / rho) / implicit_scale, 0.0)
        active_mask = _active_velocity_mask(fluid_mask)
        cell_metric = _cell_metric(mesh).astype(u.dtype)
        active_weight = jnp.where(active_mask, cell_metric, 0.0)
        active_total_weight = jnp.maximum(jnp.sum(active_weight), 1e-20)
        control_mask = jnp.where(
            target_mean_velocity is None,
            active_mask,
            fluid_mask,
        )
        control_weight = jnp.where(control_mask, cell_metric, 0.0)
        control_total_weight = jnp.maximum(jnp.sum(control_weight), 1e-20)
        mean_base = jnp.sum(control_weight * base_trial) / control_total_weight
        mean_sensitivity = jnp.sum(control_weight * pressure_sensitivity) / control_total_weight
        forcing_value = jnp.asarray(forcing, dtype=u.dtype)
        pressure_proxy = forcing_value
        if reference_mean_velocity is not None:
            reference_target = jnp.asarray(reference_mean_velocity, dtype=u.dtype)
            pressure_proxy = jnp.where(
                mean_sensitivity > 1e-20,
                (reference_target - mean_base) / mean_sensitivity,
                forcing_value,
            )
        if target_mean_velocity is not None:
            target = jnp.asarray(target_mean_velocity, dtype=u.dtype)
            forcing_value = jnp.where(
                mean_sensitivity > 1e-20,
                (target - mean_base) / mean_sensitivity,
                forcing_value,
            )
        u_trial = jnp.where(fluid_mask, base_trial + pressure_sensitivity * forcing_value, 0.0)
        relaxed = jnp.where(fluid_mask, (1.0 - relaxation) * u_iter + relaxation * u_trial, 0.0)
        raw_update_max, limiter_scale, limited_fraction = _velocity_update_statistics(
            u_iter,
            relaxed,
            fluid_mask,
            max_delta=velocity_update_limit,
            limiter=velocity_update_limiter,
        )
        u_next = _limited_velocity_update(
            u_iter,
            relaxed,
            fluid_mask,
            max_delta=velocity_update_limit,
            limiter=velocity_update_limiter,
        )
        u_next = _enforce_velocity_bc(
            u_next,
            mesh,
            fluid_mask,
            interpolate_direct_fluid_walls=interpolate_direct_fluid_walls,
        )
        u_next = jnp.nan_to_num(u_next, nan=0.0, posinf=5.0, neginf=-5.0)
        u_next = jnp.clip(u_next, -5.0, 5.0)
        if post_update_potential_refresh:
            phi, potential_residual, potential_iteration_count = _solve_potential(
                mesh,
                sigma,
                fluid_mask,
                u_next,
                by,
                bz,
                anchor,
                potential_iterations,
                tolerance=potential_tolerance,
                relaxation=potential_relaxation,
                solver=potential_solver,
            )
            phi = jnp.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0)
        jy, jz, lorentz = _compute_current_and_lorentz(
            mesh,
            sigma,
            fluid_mask,
            u_next,
            phi,
            by,
            bz,
            reconstruction=current_reconstruction,
        )
        face_current_max, emf_max, face_lorentz_max = _face_current_emf_and_lorentz_max(
            mesh,
            sigma,
            fluid_mask,
            u_next,
            phi,
            by,
            bz,
        )
        mean_velocity = jnp.sum(active_weight * u_next) / active_total_weight
        return (
            u_next,
            phi,
            jy,
            jz,
            lorentz,
            potential_residual,
            potential_iteration_count,
            face_current_max,
            emf_max,
            face_lorentz_max,
            mean_velocity,
            forcing_value,
            pressure_proxy,
            raw_update_max,
            limiter_scale,
            limited_fraction,
        )

    u_init = _enforce_velocity_bc(
        u,
        mesh,
        fluid_mask,
        interpolate_direct_fluid_walls=interpolate_direct_fluid_walls,
    )
    phi0 = jnp.zeros_like(u)
    j0 = jnp.zeros_like(u)
    l0 = jnp.zeros_like(u)
    r0 = jnp.asarray(0.0, dtype=u.dtype)
    i0 = jnp.asarray(0, dtype=jnp.int32)
    f0 = jnp.asarray(0.0, dtype=u.dtype)
    e0 = jnp.asarray(0.0, dtype=u.dtype)
    h0 = jnp.asarray(0.0, dtype=u.dtype)
    m0 = jnp.asarray(0.0, dtype=u.dtype)
    g0 = jnp.asarray(0.0, dtype=u.dtype)
    p0 = jnp.asarray(0.0, dtype=u.dtype)
    d0 = jnp.asarray(0.0, dtype=u.dtype)
    s0 = jnp.asarray(1.0, dtype=u.dtype)
    q0 = jnp.asarray(0.0, dtype=u.dtype)
    outer_count = max(1, outer_iterations)
    (
        u_next,
        phi,
        jy,
        jz,
        lorentz,
        potential_residual,
        potential_iteration_count,
        face_current_max,
        emf_max,
        face_lorentz_max,
        mean_velocity,
        applied_forcing,
        pressure_proxy,
        raw_update_max,
        limiter_scale,
        limited_fraction,
    ) = jax.lax.fori_loop(
        0,
        outer_count,
        outer_body,
        (u_init, phi0, j0, j0, l0, r0, i0, f0, e0, h0, m0, g0, p0, d0, s0, q0),
    )
    residual = jnp.max(jnp.abs(u_next - u))
    return (
        u_next,
        phi,
        jy,
        jz,
        lorentz,
        residual,
        potential_residual,
        potential_iteration_count,
        face_current_max,
        emf_max,
        face_lorentz_max,
        mean_velocity,
        applied_forcing,
        pressure_proxy,
        raw_update_max,
        limiter_scale,
        limited_fraction,
    )

def _emit_solver_header(
    logger,
    *,
    case,
    mesh,
    materials,
    mode,
    potential_solver,
    target_mean_velocity,
    reference_mean_velocity,
    restart: RestartLogInfo | None = None,
):
    if logger is None:
        return
    logger.emit_header(
        case=case,
        mesh=mesh,
        materials=materials,
        mode=mode,
        potential_solver=potential_solver,
        target_mean_velocity=target_mean_velocity,
        reference_mean_velocity=reference_mean_velocity,
        restart=restart,
    )


def _initial_solver_state(
    *,
    case: CaseSpec,
    mesh: StructuredMesh,
    fluid_mask: jnp.ndarray,
    interpolate_direct_fluid_walls: bool,
    initial_state: MHDState | None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, float]:
    if initial_state is None:
        initial_u = jnp.where(fluid_mask, case.initial_velocity, 0.0)
        initial_u = _enforce_velocity_bc(
            initial_u,
            mesh,
            fluid_mask,
            interpolate_direct_fluid_walls=interpolate_direct_fluid_walls,
        )
        zeros = jnp.zeros_like(initial_u)
        return initial_u, zeros, zeros, zeros, zeros, 0.0
    initial_u = _enforce_velocity_bc(
        jnp.asarray(initial_state.u),
        mesh,
        fluid_mask,
        interpolate_direct_fluid_walls=interpolate_direct_fluid_walls,
    )
    return (
        initial_u,
        jnp.asarray(initial_state.phi),
        jnp.asarray(initial_state.jy),
        jnp.asarray(initial_state.jz),
        jnp.asarray(initial_state.lorentz_x),
        float(initial_state.time),
    )


def _concat_history(
    previous: jnp.ndarray | None,
    current: jnp.ndarray,
    *,
    append: bool,
) -> jnp.ndarray:
    if not append or previous is None or previous.size == 0:
        return current
    return jnp.concatenate((previous, current))


def _pressure_proxy_reference_current(diagnostics: Diagnostics | None) -> float | None:
    if diagnostics is None:
        return None
    if diagnostics.face_current_max_history.size:
        return float(diagnostics.face_current_max_history[0])
    if diagnostics.current_max_history.size:
        return float(diagnostics.current_max_history[0])
    return None


def _scaled_pressure_proxy_value(
    pressure_proxy: float,
    current_max: float,
    face_current_max: float,
    reference_current: float | None,
) -> tuple[float, float]:
    current_source = float(face_current_max) if abs(float(face_current_max)) > 0.0 else float(current_max)
    if reference_current is None or abs(reference_current) < 1e-20:
        reference_current = current_source if abs(current_source) >= 1e-20 else 1.0
    scaled = float(pressure_proxy) * current_source / reference_current
    return scaled, reference_current


def _emit_solver_step(
    logger,
    *,
    step_index: int,
    step_time: float,
    dt: float,
    u_max_value: float,
    mean_velocity: float,
    max_current: float,
    face_current_max: float,
    emf_max: float,
    max_lorentz: float,
    face_lorentz_max: float,
    residual_value: float,
    potential_residual: float,
    potential_iteration_count: float,
    linear_residual: float,
    linear_iteration_count: float,
    applied_forcing: float,
    pressure_proxy: float,
    current_scaled_pressure_proxy: float,
    raw_update_max: float,
    limiter_scale: float,
    limited_fraction: float,
    courant_like: float,
    ohmic: float,
    volumetric_flow_rate: float,
    mean_current_magnitude: float,
    lorentz_power: float,
    div_current_max: float,
    gauge_residual: float,
    interface_current_residual: float,
):
    if logger is None:
        return
    logger.emit_step(
        SolverStepRecord(
            step_index=step_index,
            time=step_time,
            dt=dt,
            u_max=u_max_value,
            mean_velocity=mean_velocity,
            current_max=max_current,
            face_current_max=face_current_max,
            emf_max=emf_max,
            lorentz_max=max_lorentz,
            face_lorentz_max=face_lorentz_max,
            residual=residual_value,
            potential_residual=potential_residual,
            potential_iterations=potential_iteration_count,
            linear_residual=linear_residual,
            linear_iterations=linear_iteration_count,
            applied_forcing=applied_forcing,
            pressure_proxy=pressure_proxy,
            current_scaled_pressure_proxy=current_scaled_pressure_proxy,
            raw_update_max=raw_update_max,
            limiter_scale=limiter_scale,
            limited_fraction=limited_fraction,
            courant_like=courant_like,
            ohmic_power=ohmic,
            volumetric_flow_rate=volumetric_flow_rate,
            mean_current_magnitude=mean_current_magnitude,
            lorentz_power=lorentz_power,
            div_current_max=div_current_max,
            gauge_residual=gauge_residual,
            interface_current_residual=interface_current_residual,
        )
        )


def _fully_developed_case_step(
    *,
    case: CaseSpec,
    mesh: StructuredMesh,
    materials,
    u_previous: jnp.ndarray,
    step_time: float,
    potential_solver: str,
    target_mean_velocity: float | None,
    linear_solver: str,
    preconditioner: str,
    coupling_iterations: int,
    coupling_tolerance: float,
) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
]:
    _, by, bz = magnetic_field_components(case.magnetic_field, mesh, time=step_time)
    forcing = _explicit_forcing(case.forcing, by.dtype)
    fluid_mask = materials.fluid_mask
    active_mask = fluid_mask
    u_iter = u_previous
    dt = case.time_stepper.dt
    fluid_weight = jnp.where(fluid_mask, _cell_metric(mesh).astype(u_previous.dtype), 0.0)
    fluid_total_weight = jnp.maximum(jnp.sum(fluid_weight), 1e-20)
    velocity_residual = jnp.asarray(jnp.inf, dtype=u_previous.dtype)
    potential_residual = jnp.asarray(jnp.inf, dtype=u_previous.dtype)
    potential_iteration_count = jnp.asarray(0, dtype=jnp.int32)
    linear_iteration_count = jnp.asarray(0, dtype=jnp.int32)
    linear_residual = jnp.asarray(jnp.inf, dtype=u_previous.dtype)
    applied_forcing = forcing
    steady_mode = case.solver.mode == "steady"

    if case.solver.time_scheme != "implicit_euler" and not steady_mode:
        raise NotImplementedError("fully_developed_inductionless currently supports implicit_euler only")

    for _ in range(max(1, coupling_iterations)):
        phi, potential_residual, potential_iteration_count = _solve_potential(
            mesh,
            materials.conductivity,
            fluid_mask,
            u_iter,
            by,
            bz,
            case.reference_phi_cell,
            case.time_stepper.potential_iterations,
            tolerance=case.time_stepper.potential_tolerance,
            relaxation=case.time_stepper.potential_relaxation,
            solver=potential_solver,
        )
        phi = jnp.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0)
        reaction = jnp.where(
            active_mask,
            materials.conductivity * (by**2 + bz**2) / materials.density,
            0.0,
        )
        if not steady_mode:
            reaction = reaction + jnp.where(active_mask, 1.0 / dt, 0.0)
        rhs_base, _ = _fully_developed_rhs(
            mesh=mesh,
            sigma=materials.conductivity,
            rho=materials.density,
            fluid_mask=fluid_mask,
            phi=phi,
            by=by,
            bz=bz,
            forcing=jnp.asarray(0.0, dtype=u_previous.dtype),
        )
        if not steady_mode:
            rhs_base = rhs_base + jnp.where(active_mask, u_previous / dt, 0.0)
        if target_mean_velocity is None:
            rhs = rhs_base + jnp.where(active_mask, forcing / materials.density, 0.0)
            u_next, velocity_linear_residual, linear_iteration_count = _solve_velocity_system(
                mesh=mesh,
                diffusivity=materials.viscosity,
                reaction=reaction,
                rhs=rhs,
                active_mask=active_mask,
                linear_solver=linear_solver,
                preconditioner=preconditioner,
                max_steps=max(case.time_stepper.max_steps, case.solver.coupling_iterations * 25),
                tolerance=min(coupling_tolerance, 1e-10),
            )
            applied_forcing = forcing
        else:
            unit_rhs = jnp.where(active_mask, 1.0 / materials.density, 0.0)
            u_base, velocity_linear_residual, linear_iteration_count = _solve_velocity_system(
                mesh=mesh,
                diffusivity=materials.viscosity,
                reaction=reaction,
                rhs=rhs_base,
                active_mask=active_mask,
                linear_solver=linear_solver,
                preconditioner=preconditioner,
                max_steps=max(case.time_stepper.max_steps, case.solver.coupling_iterations * 25),
                tolerance=min(coupling_tolerance, 1e-10),
            )
            u_sensitivity, _, _ = _solve_velocity_system(
                mesh=mesh,
                diffusivity=materials.viscosity,
                reaction=reaction,
                rhs=unit_rhs,
                active_mask=active_mask,
                linear_solver=linear_solver,
                preconditioner=preconditioner,
                max_steps=max(case.time_stepper.max_steps, case.solver.coupling_iterations * 25),
                tolerance=min(coupling_tolerance, 1e-10),
            )
            mean_base = jnp.sum(fluid_weight * u_base) / fluid_total_weight
            mean_sensitivity = jnp.sum(fluid_weight * u_sensitivity) / fluid_total_weight
            applied_forcing = jnp.where(
                mean_sensitivity > 1e-20,
                (jnp.asarray(target_mean_velocity, dtype=u_previous.dtype) - mean_base) / mean_sensitivity,
                jnp.asarray(0.0, dtype=u_previous.dtype),
            )
            u_next = u_base + applied_forcing * u_sensitivity
        linear_residual = velocity_linear_residual
        u_next = _enforce_velocity_bc(
            jnp.where(fluid_mask, u_next, 0.0),
            mesh,
            fluid_mask,
            interpolate_direct_fluid_walls=False,
        )
        velocity_residual = jnp.max(jnp.abs(u_next - u_iter))
        u_iter = u_next
        if (
            float(velocity_residual) <= float(coupling_tolerance)
            and float(potential_residual) <= float(case.time_stepper.potential_tolerance or coupling_tolerance)
            and float(linear_residual) <= float(coupling_tolerance)
        ):
            break

    phi, potential_residual, potential_iteration_count = _solve_potential(
        mesh,
        materials.conductivity,
        fluid_mask,
        u_iter,
        by,
        bz,
        case.reference_phi_cell,
        case.time_stepper.potential_iterations,
        tolerance=case.time_stepper.potential_tolerance,
        relaxation=case.time_stepper.potential_relaxation,
        solver=potential_solver,
    )
    jy, jz, lorentz = _compute_current_and_lorentz(
        mesh,
        materials.conductivity,
        fluid_mask,
        u_iter,
        phi,
        by,
        bz,
        reconstruction="cell_centered",
    )
    face_current_max, emf_max, face_lorentz_max = _face_current_emf_and_lorentz_max(
        mesh,
        materials.conductivity,
        fluid_mask,
        u_iter,
        phi,
        by,
        bz,
    )
    mean_velocity = jnp.sum(fluid_weight * u_iter) / fluid_total_weight
    return (
        u_iter,
        phi,
        jy,
        jz,
        lorentz,
        velocity_residual,
        potential_residual,
        potential_iteration_count,
        linear_residual,
        linear_iteration_count,
        face_current_max,
        emf_max,
        face_lorentz_max,
        mean_velocity,
        applied_forcing,
    )


def _solve_fully_developed(
    case: CaseSpec,
    logger=None,
    *,
    initial_state: MHDState | None = None,
    initial_diagnostics: Diagnostics | None = None,
    append_diagnostics: bool = False,
    restart_info: RestartLogInfo | None = None,
) -> Solution:
    mesh = _build_mesh(case)
    materials = build_material_fields(case, mesh)
    target_mean_velocity = _target_mean_velocity(case)
    reference_mean_velocity = _reference_mean_velocity(case)
    potential_solver = _resolve_potential_solver(case.time_stepper.potential_solver, materials.fluid_mask)
    linear_solver = "cg" if case.solver.linear_solver == "auto" else case.solver.linear_solver
    if case.geometry.kind not in {"rect_duct", "layered_duct"}:
        raise NotImplementedError(f"Solver {case.solver.kind!r} does not yet support geometry {case.geometry.kind!r}")
    interpolate_direct_fluid_walls = False
    initial_u, initial_phi, initial_jy, initial_jz, initial_lorentz, start_time = _initial_solver_state(
        case=case,
        mesh=mesh,
        fluid_mask=materials.fluid_mask,
        interpolate_direct_fluid_walls=interpolate_direct_fluid_walls,
        initial_state=initial_state,
    )
    dt = case.time_stepper.dt
    steady_mode = case.solver.mode == "steady"
    if steady_mode:
        steps = max(1, case.time_stepper.max_steps)
        step_coupling_iterations = case.solver.coupling_iterations
        step_coupling_tolerance = float(case.time_stepper.steady_tolerance)
    else:
        remaining_time = max(0.0, case.time_stepper.t_final - start_time)
        requested_steps = int(round(remaining_time / dt)) if remaining_time > 0.0 else 0
        steps = min(case.time_stepper.max_steps, requested_steps)
        step_coupling_iterations = case.solver.coupling_iterations
        step_coupling_tolerance = case.solver.coupling_tolerance
    _emit_solver_header(
        logger,
        case=case,
        mesh=mesh,
        materials=materials,
        mode=case.solver.mode,
        potential_solver=f"{potential_solver} / {linear_solver}",
        target_mean_velocity=target_mean_velocity,
        reference_mean_velocity=reference_mean_velocity,
        restart=restart_info,
    )

    u = initial_u
    phi = initial_phi
    jy = initial_jy
    jz = initial_jz
    lorentz = initial_lorentz
    time_history: list[float] = []
    u_max_history: list[float] = []
    mean_velocity_history: list[float] = []
    applied_forcing_history: list[float] = []
    pressure_proxy_history: list[float] = []
    current_scaled_pressure_proxy_history: list[float] = []
    residual_history: list[float] = []
    courant_history: list[float] = []
    ohmic_history: list[float] = []
    current_max_history: list[float] = []
    face_current_max_history: list[float] = []
    emf_max_history: list[float] = []
    lorentz_max_history: list[float] = []
    face_lorentz_max_history: list[float] = []
    potential_history: list[float] = []
    potential_iteration_history: list[float] = []
    linear_residual_history: list[float] = []
    linear_iteration_history: list[float] = []
    volumetric_flow_rate_history: list[float] = []
    mean_current_magnitude_history: list[float] = []
    lorentz_power_history: list[float] = []
    div_current_max_history: list[float] = []
    gauge_residual_history: list[float] = []
    interface_current_residual_history: list[float] = []
    raw_update_max_history: list[float] = []
    limiter_scale_history: list[float] = []
    limited_fraction_history: list[float] = []
    pressure_proxy_reference_current = _pressure_proxy_reference_current(initial_diagnostics if append_diagnostics else None)
    residual_value = float(initial_state.residual if initial_state is not None else 0.0)
    step_count = 0

    for step_index in range(steps):
        step_time = float(start_time + (step_index + 1) * dt)
        (
            u,
            phi,
            jy,
            jz,
            lorentz,
            residual,
            potential_residual,
            potential_iteration_count,
            linear_residual,
            _linear_iteration_count,
            face_current_max,
            emf_max,
            face_lorentz_max,
            mean_velocity,
            applied_forcing,
        ) = _fully_developed_case_step(
            case=case,
            mesh=mesh,
            materials=materials,
            u_previous=u,
            step_time=step_time,
            potential_solver=potential_solver,
            target_mean_velocity=target_mean_velocity,
            linear_solver=linear_solver,
            preconditioner=case.solver.preconditioner,
            coupling_iterations=step_coupling_iterations,
            coupling_tolerance=step_coupling_tolerance,
        )
        residual_value = float(residual)
        u_max_value = float(jnp.max(jnp.abs(u)))
        courant_like = float(u_max_value * dt / jnp.min(mesh.dy))
        ohmic = float(jnp.mean(jy**2 + jz**2))
        max_current = float(jnp.max(jnp.sqrt(jy**2 + jz**2)))
        max_lorentz = float(jnp.max(jnp.abs(lorentz)))
        _, by_step, bz_step = magnetic_field_components(case.magnetic_field, mesh, time=step_time)
        (
            volumetric_flow_rate,
            mean_current_magnitude,
            lorentz_power,
            div_current_max,
            gauge_residual,
            interface_current_residual,
        ) = _integral_diagnostics(
            mesh=mesh,
            sigma=materials.conductivity,
            fluid_mask=materials.fluid_mask,
            u=u,
            phi=phi,
            jy=jy,
            jz=jz,
            lorentz=lorentz,
            by=by_step,
            bz=bz_step,
            anchor=case.reference_phi_cell,
        )
        time_history.append(step_time)
        u_max_history.append(u_max_value)
        mean_velocity_history.append(float(mean_velocity))
        applied_forcing_history.append(float(applied_forcing))
        pressure_proxy_history.append(float(applied_forcing))
        residual_history.append(residual_value)
        courant_history.append(courant_like)
        ohmic_history.append(ohmic)
        current_max_history.append(max_current)
        face_current_max_history.append(float(face_current_max))
        emf_max_history.append(float(emf_max))
        lorentz_max_history.append(max_lorentz)
        face_lorentz_max_history.append(float(face_lorentz_max))
        potential_history.append(float(potential_residual))
        potential_iteration_history.append(float(potential_iteration_count))
        linear_residual_history.append(float(linear_residual))
        linear_iteration_history.append(float(_linear_iteration_count))
        volumetric_flow_rate_history.append(float(volumetric_flow_rate))
        mean_current_magnitude_history.append(float(mean_current_magnitude))
        lorentz_power_history.append(float(lorentz_power))
        div_current_max_history.append(float(div_current_max))
        gauge_residual_history.append(float(gauge_residual))
        interface_current_residual_history.append(float(interface_current_residual))
        raw_update_max_history.append(residual_value)
        limiter_scale_history.append(1.0)
        limited_fraction_history.append(0.0)
        current_scaled_pressure_proxy, pressure_proxy_reference_current = _scaled_pressure_proxy_value(
            float(applied_forcing),
            max_current,
            float(face_current_max),
            pressure_proxy_reference_current,
        )
        current_scaled_pressure_proxy_history.append(float(current_scaled_pressure_proxy))
        _emit_solver_step(
            logger,
            step_index=step_index + 1,
            step_time=step_time,
            dt=dt,
            u_max_value=u_max_value,
            mean_velocity=float(mean_velocity),
            max_current=max_current,
            face_current_max=float(face_current_max),
            emf_max=float(emf_max),
            max_lorentz=max_lorentz,
            face_lorentz_max=float(face_lorentz_max),
            residual_value=residual_value,
            potential_residual=float(potential_residual),
            potential_iteration_count=float(potential_iteration_count),
            linear_residual=float(linear_residual),
            linear_iteration_count=float(_linear_iteration_count),
            applied_forcing=float(applied_forcing),
            pressure_proxy=float(applied_forcing),
            current_scaled_pressure_proxy=float(current_scaled_pressure_proxy),
            raw_update_max=residual_value,
            limiter_scale=1.0,
            limited_fraction=0.0,
            courant_like=courant_like,
            ohmic=ohmic,
            volumetric_flow_rate=float(volumetric_flow_rate),
            mean_current_magnitude=float(mean_current_magnitude),
            lorentz_power=float(lorentz_power),
            div_current_max=float(div_current_max),
            gauge_residual=float(gauge_residual),
            interface_current_residual=float(interface_current_residual),
        )
        step_count = step_index + 1
        potential_gate = case.time_stepper.steady_potential_tolerance
        if (
            steady_mode
            and residual_value <= float(case.time_stepper.steady_tolerance)
            and float(linear_residual) <= float(case.time_stepper.steady_tolerance)
            and (
                potential_gate is None
                or float(potential_residual) <= float(potential_gate)
            )
        ):
            break

    state = MHDState(
        u=u,
        phi=phi,
        jy=jy,
        jz=jz,
        lorentz_x=lorentz,
        time=float(start_time + step_count * dt),
        residual=residual_value,
    )
    diagnostics = Diagnostics(
        time_history=_concat_history(initial_diagnostics.time_history if initial_diagnostics is not None else None, jnp.asarray(time_history, dtype=float), append=append_diagnostics),
        u_max_history=_concat_history(initial_diagnostics.u_max_history if initial_diagnostics is not None else None, jnp.asarray(u_max_history, dtype=float), append=append_diagnostics),
        mean_velocity_history=_concat_history(initial_diagnostics.mean_velocity_history if initial_diagnostics is not None else None, jnp.asarray(mean_velocity_history, dtype=float), append=append_diagnostics),
        applied_forcing_history=_concat_history(initial_diagnostics.applied_forcing_history if initial_diagnostics is not None else None, jnp.asarray(applied_forcing_history, dtype=float), append=append_diagnostics),
        pressure_proxy_history=_concat_history(initial_diagnostics.pressure_proxy_history if initial_diagnostics is not None else None, jnp.asarray(pressure_proxy_history, dtype=float), append=append_diagnostics),
        current_scaled_pressure_proxy_history=_concat_history(initial_diagnostics.current_scaled_pressure_proxy_history if initial_diagnostics is not None else None, jnp.asarray(current_scaled_pressure_proxy_history, dtype=float), append=append_diagnostics),
        raw_update_max_history=_concat_history(initial_diagnostics.raw_update_max_history if initial_diagnostics is not None else None, jnp.asarray(raw_update_max_history, dtype=float), append=append_diagnostics),
        limiter_scale_history=_concat_history(initial_diagnostics.limiter_scale_history if initial_diagnostics is not None else None, jnp.asarray(limiter_scale_history, dtype=float), append=append_diagnostics),
        limited_fraction_history=_concat_history(initial_diagnostics.limited_fraction_history if initial_diagnostics is not None else None, jnp.asarray(limited_fraction_history, dtype=float), append=append_diagnostics),
        residual_history=_concat_history(initial_diagnostics.residual_history if initial_diagnostics is not None else None, jnp.asarray(residual_history, dtype=float), append=append_diagnostics),
        courant_like=_concat_history(initial_diagnostics.courant_like if initial_diagnostics is not None else None, jnp.asarray(courant_history, dtype=float), append=append_diagnostics),
        ohmic_power=_concat_history(initial_diagnostics.ohmic_power if initial_diagnostics is not None else None, jnp.asarray(ohmic_history, dtype=float), append=append_diagnostics),
        current_max_history=_concat_history(initial_diagnostics.current_max_history if initial_diagnostics is not None else None, jnp.asarray(current_max_history, dtype=float), append=append_diagnostics),
        face_current_max_history=_concat_history(initial_diagnostics.face_current_max_history if initial_diagnostics is not None else None, jnp.asarray(face_current_max_history, dtype=float), append=append_diagnostics),
        emf_max_history=_concat_history(initial_diagnostics.emf_max_history if initial_diagnostics is not None else None, jnp.asarray(emf_max_history, dtype=float), append=append_diagnostics),
        lorentz_max_history=_concat_history(initial_diagnostics.lorentz_max_history if initial_diagnostics is not None else None, jnp.asarray(lorentz_max_history, dtype=float), append=append_diagnostics),
        face_lorentz_max_history=_concat_history(initial_diagnostics.face_lorentz_max_history if initial_diagnostics is not None else None, jnp.asarray(face_lorentz_max_history, dtype=float), append=append_diagnostics),
        potential_residual_history=_concat_history(initial_diagnostics.potential_residual_history if initial_diagnostics is not None else None, jnp.asarray(potential_history, dtype=float), append=append_diagnostics),
        potential_iterations_history=_concat_history(initial_diagnostics.potential_iterations_history if initial_diagnostics is not None else None, jnp.asarray(potential_iteration_history, dtype=float), append=append_diagnostics),
        linear_residual_history=_concat_history(initial_diagnostics.linear_residual_history if initial_diagnostics is not None else None, jnp.asarray(linear_residual_history, dtype=float), append=append_diagnostics),
        linear_iterations_history=_concat_history(initial_diagnostics.linear_iterations_history if initial_diagnostics is not None else None, jnp.asarray(linear_iteration_history, dtype=float), append=append_diagnostics),
        volumetric_flow_rate_history=_concat_history(initial_diagnostics.volumetric_flow_rate_history if initial_diagnostics is not None else None, jnp.asarray(volumetric_flow_rate_history, dtype=float), append=append_diagnostics),
        mean_current_magnitude_history=_concat_history(initial_diagnostics.mean_current_magnitude_history if initial_diagnostics is not None else None, jnp.asarray(mean_current_magnitude_history, dtype=float), append=append_diagnostics),
        lorentz_power_history=_concat_history(initial_diagnostics.lorentz_power_history if initial_diagnostics is not None else None, jnp.asarray(lorentz_power_history, dtype=float), append=append_diagnostics),
        div_current_max_history=_concat_history(initial_diagnostics.div_current_max_history if initial_diagnostics is not None else None, jnp.asarray(div_current_max_history, dtype=float), append=append_diagnostics),
        gauge_residual_history=_concat_history(initial_diagnostics.gauge_residual_history if initial_diagnostics is not None else None, jnp.asarray(gauge_residual_history, dtype=float), append=append_diagnostics),
        interface_current_residual_history=_concat_history(initial_diagnostics.interface_current_residual_history if initial_diagnostics is not None else None, jnp.asarray(interface_current_residual_history, dtype=float), append=append_diagnostics),
    )
    solution = Solution(mesh=mesh, state=state, diagnostics=diagnostics, case_name=case.name)
    if logger is not None:
        logger.emit_footer(solution)
    return solution


def _solve_transient_legacy(
    case: CaseSpec,
    logger=None,
    *,
    initial_state: MHDState | None = None,
    initial_diagnostics: Diagnostics | None = None,
    append_diagnostics: bool = False,
    restart_info: RestartLogInfo | None = None,
) -> Solution:
    mesh = _build_mesh(case)
    materials = build_material_fields(case, mesh)
    target_mean_velocity = _target_mean_velocity(case)
    reference_mean_velocity = _reference_mean_velocity(case)
    potential_solver = _resolve_potential_solver(case.time_stepper.potential_solver, materials.fluid_mask)
    interpolate_direct_fluid_walls = not bool(jnp.all(materials.fluid_mask))

    initial_u, initial_phi, initial_jy, initial_jz, initial_lorentz, start_time = _initial_solver_state(
        case=case,
        mesh=mesh,
        fluid_mask=materials.fluid_mask,
        interpolate_direct_fluid_walls=interpolate_direct_fluid_walls,
        initial_state=initial_state,
    )
    dt = case.time_stepper.dt
    remaining_time = max(0.0, case.time_stepper.t_final - start_time)
    requested_steps = int(round(remaining_time / dt)) if remaining_time > 0.0 else 0
    steps = min(case.time_stepper.max_steps, requested_steps)
    _emit_solver_header(
        logger,
        case=case,
        mesh=mesh,
        materials=materials,
        mode="transient",
        potential_solver=potential_solver,
        target_mean_velocity=target_mean_velocity,
        reference_mean_velocity=reference_mean_velocity,
        restart=restart_info,
    )

    if steps == 0:
        state = MHDState(
            u=initial_u,
            phi=initial_phi,
            jy=initial_jy,
            jz=initial_jz,
            lorentz_x=initial_lorentz,
            time=float(start_time),
            residual=float(initial_state.residual if initial_state is not None else 0.0),
        )
        diagnostics = initial_diagnostics if (append_diagnostics and initial_diagnostics is not None) else Diagnostics(
            residual_history=jnp.zeros((0,)),
            courant_like=jnp.zeros((0,)),
            ohmic_power=jnp.zeros((0,)),
        )
        solution = Solution(mesh=mesh, state=state, diagnostics=diagnostics, case_name=case.name)
        if logger is not None:
            logger.emit_footer(solution)
        return solution

    if logger is None:
        def scan_step(carry, _):
            u, time = carry
            step_time = time + dt
            _, by, bz = magnetic_field_components(case.magnetic_field, mesh, time=step_time)
            forcing = _explicit_forcing(case.forcing, by.dtype)
            (
                u_next,
                phi,
                jy,
                jz,
                lorentz,
                residual,
                potential_residual,
                potential_iteration_count,
                face_current_max,
                emf_max,
                face_lorentz_max,
                mean_velocity,
                applied_forcing,
                pressure_proxy,
                raw_update_max,
                limiter_scale,
                limited_fraction,
            ) = _step(
                u=u,
                mesh=mesh,
                sigma=materials.conductivity,
                rho=materials.density,
                nu=materials.viscosity,
                fluid_mask=materials.fluid_mask,
                by=by,
                bz=bz,
                dt=dt,
                forcing=forcing,
                target_mean_velocity=target_mean_velocity,
                reference_mean_velocity=reference_mean_velocity,
                anchor=case.reference_phi_cell,
                outer_iterations=case.time_stepper.outer_iterations,
                potential_iterations=case.time_stepper.potential_iterations,
                potential_tolerance=case.time_stepper.potential_tolerance,
                potential_relaxation=case.time_stepper.potential_relaxation,
                potential_solver=potential_solver,
                relaxation=case.time_stepper.relaxation,
                velocity_update_limit=case.time_stepper.velocity_update_limit,
                velocity_update_limiter=case.time_stepper.velocity_update_limiter,
                current_reconstruction=case.time_stepper.current_reconstruction,
                post_update_potential_refresh=case.time_stepper.post_update_potential_refresh,
                interpolate_direct_fluid_walls=interpolate_direct_fluid_walls,
            )
            courant_like = jnp.max(jnp.abs(u_next)) * dt / jnp.min(mesh.dy)
            ohmic = jnp.mean(jy**2 + jz**2)
            max_current = jnp.max(jnp.sqrt(jy**2 + jz**2))
            max_lorentz = jnp.max(jnp.abs(lorentz))
            sample = jnp.asarray(
                [
                    step_time,
                    jnp.max(jnp.abs(u_next)),
                    mean_velocity,
                    applied_forcing,
                    pressure_proxy,
                    residual,
                    courant_like,
                    ohmic,
                    max_current,
                    face_current_max,
                    emf_max,
                    max_lorentz,
                    face_lorentz_max,
                    potential_residual,
                    potential_iteration_count,
                    raw_update_max,
                    limiter_scale,
                    limited_fraction,
                ],
                dtype=float,
            )
            return (u_next, step_time), (u_next, phi, jy, jz, lorentz, sample)

        (u_final, time_final), history = jax.lax.scan(scan_step, (initial_u, start_time), xs=None, length=steps)
        u_hist, phi_hist, jy_hist, jz_hist, lorentz_hist, samples = history
        reference_current = _pressure_proxy_reference_current(initial_diagnostics if append_diagnostics else None)
        face_current_column = samples[:, 9]
        current_column = samples[:, 8]
        if reference_current is None:
            reference_current = float(face_current_column[0]) if abs(float(face_current_column[0])) > 0.0 else float(current_column[0])
            if abs(reference_current) < 1e-20:
                reference_current = 1.0
        current_scale_column = face_current_column if float(jnp.max(jnp.abs(face_current_column))) > 0.0 else current_column
        current_scaled_pressure_proxy = samples[:, 4] * current_scale_column / reference_current

        state = MHDState(
            u=u_final,
            phi=phi_hist[-1],
            jy=jy_hist[-1],
            jz=jz_hist[-1],
            lorentz_x=lorentz_hist[-1],
            time=float(time_final),
            residual=float(samples[-1, 5]),
        )
        diagnostics = Diagnostics(
            time_history=_concat_history(
                initial_diagnostics.time_history if initial_diagnostics is not None else None,
                samples[:, 0],
                append=append_diagnostics,
            ),
            u_max_history=_concat_history(
                initial_diagnostics.u_max_history if initial_diagnostics is not None else None,
                samples[:, 1],
                append=append_diagnostics,
            ),
            mean_velocity_history=_concat_history(
                initial_diagnostics.mean_velocity_history if initial_diagnostics is not None else None,
                samples[:, 2],
                append=append_diagnostics,
            ),
            applied_forcing_history=_concat_history(
                initial_diagnostics.applied_forcing_history if initial_diagnostics is not None else None,
                samples[:, 3],
                append=append_diagnostics,
            ),
            pressure_proxy_history=_concat_history(
                initial_diagnostics.pressure_proxy_history if initial_diagnostics is not None else None,
                samples[:, 4],
                append=append_diagnostics,
            ),
            current_scaled_pressure_proxy_history=_concat_history(
                initial_diagnostics.current_scaled_pressure_proxy_history if initial_diagnostics is not None else None,
                current_scaled_pressure_proxy,
                append=append_diagnostics,
            ),
            residual_history=_concat_history(
                initial_diagnostics.residual_history if initial_diagnostics is not None else None,
                samples[:, 5],
                append=append_diagnostics,
            ),
            courant_like=_concat_history(
                initial_diagnostics.courant_like if initial_diagnostics is not None else None,
                samples[:, 6],
                append=append_diagnostics,
            ),
            ohmic_power=_concat_history(
                initial_diagnostics.ohmic_power if initial_diagnostics is not None else None,
                samples[:, 7],
                append=append_diagnostics,
            ),
            current_max_history=_concat_history(
                initial_diagnostics.current_max_history if initial_diagnostics is not None else None,
                samples[:, 8],
                append=append_diagnostics,
            ),
            face_current_max_history=_concat_history(
                initial_diagnostics.face_current_max_history if initial_diagnostics is not None else None,
                samples[:, 9],
                append=append_diagnostics,
            ),
            emf_max_history=_concat_history(
                initial_diagnostics.emf_max_history if initial_diagnostics is not None else None,
                samples[:, 10],
                append=append_diagnostics,
            ),
            lorentz_max_history=_concat_history(
                initial_diagnostics.lorentz_max_history if initial_diagnostics is not None else None,
                samples[:, 11],
                append=append_diagnostics,
            ),
            face_lorentz_max_history=_concat_history(
                initial_diagnostics.face_lorentz_max_history if initial_diagnostics is not None else None,
                samples[:, 12],
                append=append_diagnostics,
            ),
            potential_residual_history=_concat_history(
                initial_diagnostics.potential_residual_history if initial_diagnostics is not None else None,
                samples[:, 13],
                append=append_diagnostics,
            ),
            potential_iterations_history=_concat_history(
                initial_diagnostics.potential_iterations_history if initial_diagnostics is not None else None,
                samples[:, 14],
                append=append_diagnostics,
            ),
            raw_update_max_history=_concat_history(
                initial_diagnostics.raw_update_max_history if initial_diagnostics is not None else None,
                samples[:, 15],
                append=append_diagnostics,
            ),
            limiter_scale_history=_concat_history(
                initial_diagnostics.limiter_scale_history if initial_diagnostics is not None else None,
                samples[:, 16],
                append=append_diagnostics,
            ),
            limited_fraction_history=_concat_history(
                initial_diagnostics.limited_fraction_history if initial_diagnostics is not None else None,
                samples[:, 17],
                append=append_diagnostics,
            ),
        )
        solution = Solution(mesh=mesh, state=state, diagnostics=diagnostics, case_name=case.name)
        return solution

    def compiled_step(u: jnp.ndarray, time: float):
        _, by, bz = magnetic_field_components(case.magnetic_field, mesh, time=time)
        forcing = _explicit_forcing(case.forcing, by.dtype)
        return _step(
            u=u,
            mesh=mesh,
            sigma=materials.conductivity,
            rho=materials.density,
            nu=materials.viscosity,
            fluid_mask=materials.fluid_mask,
            by=by,
            bz=bz,
            dt=dt,
            forcing=forcing,
            target_mean_velocity=target_mean_velocity,
            reference_mean_velocity=reference_mean_velocity,
            anchor=case.reference_phi_cell,
            outer_iterations=case.time_stepper.outer_iterations,
            potential_iterations=case.time_stepper.potential_iterations,
            potential_tolerance=case.time_stepper.potential_tolerance,
            potential_relaxation=case.time_stepper.potential_relaxation,
            potential_solver=potential_solver,
            relaxation=case.time_stepper.relaxation,
            velocity_update_limit=case.time_stepper.velocity_update_limit,
            velocity_update_limiter=case.time_stepper.velocity_update_limiter,
            current_reconstruction=case.time_stepper.current_reconstruction,
            post_update_potential_refresh=case.time_stepper.post_update_potential_refresh,
            interpolate_direct_fluid_walls=interpolate_direct_fluid_walls,
        )

    compiled_step = jax.jit(compiled_step)

    u = initial_u
    phi = initial_phi
    jy = initial_jy
    jz = initial_jz
    lorentz = initial_lorentz
    time_history: list[float] = []
    u_max_history: list[float] = []
    mean_velocity_history: list[float] = []
    applied_forcing_history: list[float] = []
    pressure_proxy_history: list[float] = []
    residual_history: list[float] = []
    courant_history: list[float] = []
    ohmic_history: list[float] = []
    current_max_history: list[float] = []
    face_current_max_history: list[float] = []
    emf_max_history: list[float] = []
    lorentz_max_history: list[float] = []
    face_lorentz_max_history: list[float] = []
    current_scaled_pressure_proxy_history: list[float] = []
    raw_update_max_history: list[float] = []
    limiter_scale_history: list[float] = []
    limited_fraction_history: list[float] = []
    potential_history: list[float] = []
    potential_iteration_history: list[float] = []
    residual_value = float("inf")
    pressure_proxy_reference_current = _pressure_proxy_reference_current(initial_diagnostics if append_diagnostics else None)

    for step_index in range(steps):
        step_time = float(start_time + (step_index + 1) * dt)
        (
            u,
            phi,
            jy,
            jz,
            lorentz,
            residual,
            potential_residual,
            potential_iteration_count,
            face_current_max,
            emf_max,
            face_lorentz_max,
            mean_velocity,
            applied_forcing,
            pressure_proxy,
            raw_update_max,
            limiter_scale,
            limited_fraction,
        ) = compiled_step(u, step_time)
        residual_value = float(residual)
        u_max_value = float(jnp.max(jnp.abs(u)))
        courant_like = float(u_max_value * dt / jnp.min(mesh.dy))
        ohmic = float(jnp.mean(jy**2 + jz**2))
        max_current = float(jnp.max(jnp.sqrt(jy**2 + jz**2)))
        max_lorentz = float(jnp.max(jnp.abs(lorentz)))
        _, by_step, bz_step = magnetic_field_components(case.magnetic_field, mesh, time=step_time)
        (
            volumetric_flow_rate,
            mean_current_magnitude,
            lorentz_power,
            div_current_max,
            gauge_residual,
            interface_current_residual,
        ) = _integral_diagnostics(
            mesh=mesh,
            sigma=materials.conductivity,
            fluid_mask=materials.fluid_mask,
            u=u,
            phi=phi,
            jy=jy,
            jz=jz,
            lorentz=lorentz,
            by=by_step,
            bz=bz_step,
            anchor=case.reference_phi_cell,
        )
        time_history.append(step_time)
        u_max_history.append(u_max_value)
        mean_velocity_history.append(float(mean_velocity))
        applied_forcing_history.append(float(applied_forcing))
        pressure_proxy_history.append(float(pressure_proxy))
        residual_history.append(residual_value)
        courant_history.append(courant_like)
        ohmic_history.append(ohmic)
        current_max_history.append(max_current)
        face_current_max_history.append(float(face_current_max))
        emf_max_history.append(float(emf_max))
        lorentz_max_history.append(max_lorentz)
        face_lorentz_max_history.append(float(face_lorentz_max))
        raw_update_max_history.append(float(raw_update_max))
        limiter_scale_history.append(float(limiter_scale))
        limited_fraction_history.append(float(limited_fraction))
        current_scaled_pressure_proxy, pressure_proxy_reference_current = _scaled_pressure_proxy_value(
            float(pressure_proxy),
            max_current,
            float(face_current_max),
            pressure_proxy_reference_current,
        )
        current_scaled_pressure_proxy_history.append(float(current_scaled_pressure_proxy))
        potential_history.append(float(potential_residual))
        potential_iteration_history.append(float(potential_iteration_count))
        _emit_solver_step(
            logger,
            step_index=step_index + 1,
            step_time=step_time,
            dt=dt,
            u_max_value=u_max_value,
            mean_velocity=float(mean_velocity),
            max_current=max_current,
            face_current_max=float(face_current_max),
            emf_max=float(emf_max),
            max_lorentz=max_lorentz,
            face_lorentz_max=float(face_lorentz_max),
            residual_value=residual_value,
            potential_residual=float(potential_residual),
            potential_iteration_count=float(potential_iteration_count),
            linear_residual=0.0,
            linear_iteration_count=0.0,
            applied_forcing=float(applied_forcing),
            pressure_proxy=float(pressure_proxy),
            current_scaled_pressure_proxy=float(current_scaled_pressure_proxy),
            raw_update_max=float(raw_update_max),
            limiter_scale=float(limiter_scale),
            limited_fraction=float(limited_fraction),
            courant_like=courant_like,
            ohmic=ohmic,
            volumetric_flow_rate=float(volumetric_flow_rate),
            mean_current_magnitude=float(mean_current_magnitude),
            lorentz_power=float(lorentz_power),
            div_current_max=float(div_current_max),
            gauge_residual=float(gauge_residual),
            interface_current_residual=float(interface_current_residual),
        )

    state = MHDState(
        u=u,
        phi=phi,
        jy=jy,
        jz=jz,
        lorentz_x=lorentz,
        time=float(start_time + steps * dt),
        residual=residual_value,
    )
    diagnostics = Diagnostics(
        time_history=_concat_history(
            initial_diagnostics.time_history if initial_diagnostics is not None else None,
            jnp.asarray(time_history, dtype=float),
            append=append_diagnostics,
        ),
        u_max_history=_concat_history(
            initial_diagnostics.u_max_history if initial_diagnostics is not None else None,
            jnp.asarray(u_max_history, dtype=float),
            append=append_diagnostics,
        ),
        mean_velocity_history=_concat_history(
            initial_diagnostics.mean_velocity_history if initial_diagnostics is not None else None,
            jnp.asarray(mean_velocity_history, dtype=float),
            append=append_diagnostics,
        ),
        applied_forcing_history=_concat_history(
            initial_diagnostics.applied_forcing_history if initial_diagnostics is not None else None,
            jnp.asarray(applied_forcing_history, dtype=float),
            append=append_diagnostics,
        ),
        pressure_proxy_history=_concat_history(
            initial_diagnostics.pressure_proxy_history if initial_diagnostics is not None else None,
            jnp.asarray(pressure_proxy_history, dtype=float),
            append=append_diagnostics,
        ),
        current_scaled_pressure_proxy_history=_concat_history(
            initial_diagnostics.current_scaled_pressure_proxy_history if initial_diagnostics is not None else None,
            jnp.asarray(current_scaled_pressure_proxy_history, dtype=float),
            append=append_diagnostics,
        ),
        raw_update_max_history=_concat_history(
            initial_diagnostics.raw_update_max_history if initial_diagnostics is not None else None,
            jnp.asarray(raw_update_max_history, dtype=float),
            append=append_diagnostics,
        ),
        limiter_scale_history=_concat_history(
            initial_diagnostics.limiter_scale_history if initial_diagnostics is not None else None,
            jnp.asarray(limiter_scale_history, dtype=float),
            append=append_diagnostics,
        ),
        limited_fraction_history=_concat_history(
            initial_diagnostics.limited_fraction_history if initial_diagnostics is not None else None,
            jnp.asarray(limited_fraction_history, dtype=float),
            append=append_diagnostics,
        ),
        residual_history=_concat_history(
            initial_diagnostics.residual_history if initial_diagnostics is not None else None,
            jnp.asarray(residual_history, dtype=float),
            append=append_diagnostics,
        ),
        courant_like=_concat_history(
            initial_diagnostics.courant_like if initial_diagnostics is not None else None,
            jnp.asarray(courant_history, dtype=float),
            append=append_diagnostics,
        ),
        ohmic_power=_concat_history(
            initial_diagnostics.ohmic_power if initial_diagnostics is not None else None,
            jnp.asarray(ohmic_history, dtype=float),
            append=append_diagnostics,
        ),
        current_max_history=_concat_history(
            initial_diagnostics.current_max_history if initial_diagnostics is not None else None,
            jnp.asarray(current_max_history, dtype=float),
            append=append_diagnostics,
        ),
        face_current_max_history=_concat_history(
            initial_diagnostics.face_current_max_history if initial_diagnostics is not None else None,
            jnp.asarray(face_current_max_history, dtype=float),
            append=append_diagnostics,
        ),
        emf_max_history=_concat_history(
            initial_diagnostics.emf_max_history if initial_diagnostics is not None else None,
            jnp.asarray(emf_max_history, dtype=float),
            append=append_diagnostics,
        ),
        lorentz_max_history=_concat_history(
            initial_diagnostics.lorentz_max_history if initial_diagnostics is not None else None,
            jnp.asarray(lorentz_max_history, dtype=float),
            append=append_diagnostics,
        ),
        face_lorentz_max_history=_concat_history(
            initial_diagnostics.face_lorentz_max_history if initial_diagnostics is not None else None,
            jnp.asarray(face_lorentz_max_history, dtype=float),
            append=append_diagnostics,
        ),
        potential_residual_history=_concat_history(
            initial_diagnostics.potential_residual_history if initial_diagnostics is not None else None,
            jnp.asarray(potential_history, dtype=float),
            append=append_diagnostics,
        ),
        potential_iterations_history=_concat_history(
            initial_diagnostics.potential_iterations_history if initial_diagnostics is not None else None,
            jnp.asarray(potential_iteration_history, dtype=float),
            append=append_diagnostics,
        ),
    )
    solution = Solution(mesh=mesh, state=state, diagnostics=diagnostics, case_name=case.name)
    if logger is not None:
        logger.emit_footer(solution)
    return solution


def _solve_steady_legacy(
    case: CaseSpec,
    logger=None,
    *,
    initial_state: MHDState | None = None,
    initial_diagnostics: Diagnostics | None = None,
    append_diagnostics: bool = False,
    restart_info: RestartLogInfo | None = None,
) -> Solution:
    mesh = _build_mesh(case)
    materials = build_material_fields(case, mesh)
    target_mean_velocity = _target_mean_velocity(case)
    reference_mean_velocity = _reference_mean_velocity(case)
    potential_solver = _resolve_potential_solver(case.time_stepper.potential_solver, materials.fluid_mask)
    interpolate_direct_fluid_walls = not bool(jnp.all(materials.fluid_mask))

    initial_u, initial_phi, initial_jy, initial_jz, initial_lorentz, start_time = _initial_solver_state(
        case=case,
        mesh=mesh,
        fluid_mask=materials.fluid_mask,
        interpolate_direct_fluid_walls=interpolate_direct_fluid_walls,
        initial_state=initial_state,
    )
    dt = case.time_stepper.dt
    max_steps = max(1, case.time_stepper.max_steps)
    tolerance = float(case.time_stepper.steady_tolerance)
    potential_tolerance = case.time_stepper.steady_potential_tolerance
    _emit_solver_header(
        logger,
        case=case,
        mesh=mesh,
        materials=materials,
        mode="steady",
        potential_solver=potential_solver,
        target_mean_velocity=target_mean_velocity,
        reference_mean_velocity=reference_mean_velocity,
        restart=restart_info,
    )

    def compiled_step(
        u: jnp.ndarray,
        time: float,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, float, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        _, by, bz = magnetic_field_components(case.magnetic_field, mesh, time=time)
        forcing = _explicit_forcing(case.forcing, by.dtype)
        return _step(
            u=u,
            mesh=mesh,
            sigma=materials.conductivity,
            rho=materials.density,
            nu=materials.viscosity,
            fluid_mask=materials.fluid_mask,
            by=by,
            bz=bz,
            dt=dt,
            forcing=forcing,
            target_mean_velocity=target_mean_velocity,
            reference_mean_velocity=reference_mean_velocity,
            anchor=case.reference_phi_cell,
            outer_iterations=case.time_stepper.outer_iterations,
            potential_iterations=case.time_stepper.potential_iterations,
            potential_tolerance=case.time_stepper.potential_tolerance,
            potential_relaxation=case.time_stepper.potential_relaxation,
            potential_solver=potential_solver,
            relaxation=case.time_stepper.relaxation,
            velocity_update_limit=case.time_stepper.velocity_update_limit,
            velocity_update_limiter=case.time_stepper.velocity_update_limiter,
            current_reconstruction=case.time_stepper.current_reconstruction,
            post_update_potential_refresh=case.time_stepper.post_update_potential_refresh,
            interpolate_direct_fluid_walls=interpolate_direct_fluid_walls,
        )

    step_fn = jax.jit(compiled_step)

    u = initial_u
    phi = initial_phi
    jy = initial_jy
    jz = initial_jz
    lorentz = initial_lorentz
    residual_value = float("inf")
    residual_history: list[float] = []
    courant_history: list[float] = []
    ohmic_history: list[float] = []
    time_history: list[float] = []
    u_max_history: list[float] = []
    mean_velocity_history: list[float] = []
    applied_forcing_history: list[float] = []
    pressure_proxy_history: list[float] = []
    current_max_history: list[float] = []
    face_current_max_history: list[float] = []
    emf_max_history: list[float] = []
    lorentz_max_history: list[float] = []
    face_lorentz_max_history: list[float] = []
    current_scaled_pressure_proxy_history: list[float] = []
    raw_update_max_history: list[float] = []
    limiter_scale_history: list[float] = []
    limited_fraction_history: list[float] = []
    potential_history: list[float] = []
    potential_iteration_history: list[float] = []
    step_count = 0
    pressure_proxy_reference_current = _pressure_proxy_reference_current(initial_diagnostics if append_diagnostics else None)

    for step_index in range(max_steps):
        step_time = float(start_time + (step_index + 1) * dt)
        (
            u,
            phi,
            jy,
            jz,
            lorentz,
            residual,
            potential_residual,
            potential_iteration_count,
            face_current_max,
            emf_max,
            face_lorentz_max,
            mean_velocity,
            applied_forcing,
            pressure_proxy,
            raw_update_max,
            limiter_scale,
            limited_fraction,
        ) = step_fn(
            u,
            step_time,
        )
        residual_value = float(residual)
        u_max_value = float(jnp.max(jnp.abs(u)))
        courant_like = float(u_max_value * dt / jnp.min(mesh.dy))
        ohmic = float(jnp.mean(jy**2 + jz**2))
        max_current = float(jnp.max(jnp.sqrt(jy**2 + jz**2)))
        max_lorentz = float(jnp.max(jnp.abs(lorentz)))
        _, by_step, bz_step = magnetic_field_components(case.magnetic_field, mesh, time=step_time)
        (
            volumetric_flow_rate,
            mean_current_magnitude,
            lorentz_power,
            div_current_max,
            gauge_residual,
            interface_current_residual,
        ) = _integral_diagnostics(
            mesh=mesh,
            sigma=materials.conductivity,
            fluid_mask=materials.fluid_mask,
            u=u,
            phi=phi,
            jy=jy,
            jz=jz,
            lorentz=lorentz,
            by=by_step,
            bz=bz_step,
            anchor=case.reference_phi_cell,
        )
        time_history.append(step_time)
        u_max_history.append(u_max_value)
        mean_velocity_history.append(float(mean_velocity))
        applied_forcing_history.append(float(applied_forcing))
        pressure_proxy_history.append(float(pressure_proxy))
        residual_history.append(residual_value)
        courant_history.append(courant_like)
        ohmic_history.append(ohmic)
        current_max_history.append(max_current)
        face_current_max_history.append(float(face_current_max))
        emf_max_history.append(float(emf_max))
        lorentz_max_history.append(max_lorentz)
        face_lorentz_max_history.append(float(face_lorentz_max))
        raw_update_max_history.append(float(raw_update_max))
        limiter_scale_history.append(float(limiter_scale))
        limited_fraction_history.append(float(limited_fraction))
        current_scaled_pressure_proxy, pressure_proxy_reference_current = _scaled_pressure_proxy_value(
            float(pressure_proxy),
            max_current,
            float(face_current_max),
            pressure_proxy_reference_current,
        )
        current_scaled_pressure_proxy_history.append(float(current_scaled_pressure_proxy))
        potential_history.append(float(potential_residual))
        potential_iteration_history.append(float(potential_iteration_count))
        _emit_solver_step(
            logger,
            step_index=step_index + 1,
            step_time=step_time,
            dt=dt,
            u_max_value=u_max_value,
            mean_velocity=float(mean_velocity),
            max_current=max_current,
            face_current_max=float(face_current_max),
            emf_max=float(emf_max),
            max_lorentz=max_lorentz,
            face_lorentz_max=float(face_lorentz_max),
            residual_value=residual_value,
            potential_residual=float(potential_residual),
            potential_iteration_count=float(potential_iteration_count),
            linear_residual=0.0,
            linear_iteration_count=0.0,
            applied_forcing=float(applied_forcing),
            pressure_proxy=float(pressure_proxy),
            current_scaled_pressure_proxy=float(current_scaled_pressure_proxy),
            raw_update_max=float(raw_update_max),
            limiter_scale=float(limiter_scale),
            limited_fraction=float(limited_fraction),
            courant_like=courant_like,
            ohmic=ohmic,
            volumetric_flow_rate=float(volumetric_flow_rate),
            mean_current_magnitude=float(mean_current_magnitude),
            lorentz_power=float(lorentz_power),
            div_current_max=float(div_current_max),
            gauge_residual=float(gauge_residual),
            interface_current_residual=float(interface_current_residual),
        )
        step_count = step_index + 1
        velocity_converged = residual_value <= tolerance
        potential_converged = True if potential_tolerance is None else float(potential_residual) <= float(potential_tolerance)
        if velocity_converged and potential_converged:
            break

    state = MHDState(
        u=u,
        phi=phi,
        jy=jy,
        jz=jz,
        lorentz_x=lorentz,
        time=float(start_time + step_count * dt),
        residual=residual_value,
    )
    diagnostics = Diagnostics(
        time_history=_concat_history(
            initial_diagnostics.time_history if initial_diagnostics is not None else None,
            jnp.asarray(time_history, dtype=float),
            append=append_diagnostics,
        ),
        u_max_history=_concat_history(
            initial_diagnostics.u_max_history if initial_diagnostics is not None else None,
            jnp.asarray(u_max_history, dtype=float),
            append=append_diagnostics,
        ),
        mean_velocity_history=_concat_history(
            initial_diagnostics.mean_velocity_history if initial_diagnostics is not None else None,
            jnp.asarray(mean_velocity_history, dtype=float),
            append=append_diagnostics,
        ),
        applied_forcing_history=_concat_history(
            initial_diagnostics.applied_forcing_history if initial_diagnostics is not None else None,
            jnp.asarray(applied_forcing_history, dtype=float),
            append=append_diagnostics,
        ),
        pressure_proxy_history=_concat_history(
            initial_diagnostics.pressure_proxy_history if initial_diagnostics is not None else None,
            jnp.asarray(pressure_proxy_history, dtype=float),
            append=append_diagnostics,
        ),
        current_scaled_pressure_proxy_history=_concat_history(
            initial_diagnostics.current_scaled_pressure_proxy_history if initial_diagnostics is not None else None,
            jnp.asarray(current_scaled_pressure_proxy_history, dtype=float),
            append=append_diagnostics,
        ),
        raw_update_max_history=_concat_history(
            initial_diagnostics.raw_update_max_history if initial_diagnostics is not None else None,
            jnp.asarray(raw_update_max_history, dtype=float),
            append=append_diagnostics,
        ),
        limiter_scale_history=_concat_history(
            initial_diagnostics.limiter_scale_history if initial_diagnostics is not None else None,
            jnp.asarray(limiter_scale_history, dtype=float),
            append=append_diagnostics,
        ),
        limited_fraction_history=_concat_history(
            initial_diagnostics.limited_fraction_history if initial_diagnostics is not None else None,
            jnp.asarray(limited_fraction_history, dtype=float),
            append=append_diagnostics,
        ),
        residual_history=_concat_history(
            initial_diagnostics.residual_history if initial_diagnostics is not None else None,
            jnp.asarray(residual_history, dtype=float),
            append=append_diagnostics,
        ),
        courant_like=_concat_history(
            initial_diagnostics.courant_like if initial_diagnostics is not None else None,
            jnp.asarray(courant_history, dtype=float),
            append=append_diagnostics,
        ),
        ohmic_power=_concat_history(
            initial_diagnostics.ohmic_power if initial_diagnostics is not None else None,
            jnp.asarray(ohmic_history, dtype=float),
            append=append_diagnostics,
        ),
        current_max_history=_concat_history(
            initial_diagnostics.current_max_history if initial_diagnostics is not None else None,
            jnp.asarray(current_max_history, dtype=float),
            append=append_diagnostics,
        ),
        face_current_max_history=_concat_history(
            initial_diagnostics.face_current_max_history if initial_diagnostics is not None else None,
            jnp.asarray(face_current_max_history, dtype=float),
            append=append_diagnostics,
        ),
        emf_max_history=_concat_history(
            initial_diagnostics.emf_max_history if initial_diagnostics is not None else None,
            jnp.asarray(emf_max_history, dtype=float),
            append=append_diagnostics,
        ),
        lorentz_max_history=_concat_history(
            initial_diagnostics.lorentz_max_history if initial_diagnostics is not None else None,
            jnp.asarray(lorentz_max_history, dtype=float),
            append=append_diagnostics,
        ),
        face_lorentz_max_history=_concat_history(
            initial_diagnostics.face_lorentz_max_history if initial_diagnostics is not None else None,
            jnp.asarray(face_lorentz_max_history, dtype=float),
            append=append_diagnostics,
        ),
        potential_residual_history=_concat_history(
            initial_diagnostics.potential_residual_history if initial_diagnostics is not None else None,
            jnp.asarray(potential_history, dtype=float),
            append=append_diagnostics,
        ),
        potential_iterations_history=_concat_history(
            initial_diagnostics.potential_iterations_history if initial_diagnostics is not None else None,
            jnp.asarray(potential_iteration_history, dtype=float),
            append=append_diagnostics,
        ),
    )
    solution = Solution(mesh=mesh, state=state, diagnostics=diagnostics, case_name=case.name)
    if logger is not None:
        logger.emit_footer(solution)
    return solution


def solve_transient(
    case: CaseSpec,
    logger=None,
    *,
    initial_state: MHDState | None = None,
    initial_diagnostics: Diagnostics | None = None,
    append_diagnostics: bool = False,
    restart_info: RestartLogInfo | None = None,
) -> Solution:
    solver_kind = getattr(getattr(case, "solver", None), "kind", "fully_developed_inductionless")
    if solver_kind == "legacy_reduced":
        return _solve_transient_legacy(
            case,
            logger=logger,
            initial_state=initial_state,
            initial_diagnostics=initial_diagnostics,
            append_diagnostics=append_diagnostics,
            restart_info=restart_info,
        )
    if solver_kind == "fully_developed_inductionless":
        transient_case = case if case.solver.mode == "transient" else case.__class__(**{**case.__dict__, "solver": case.solver.__class__(**{**case.solver.__dict__, "mode": "transient"})})
        return _solve_fully_developed(
            transient_case,
            logger=logger,
            initial_state=initial_state,
            initial_diagnostics=initial_diagnostics,
            append_diagnostics=append_diagnostics,
            restart_info=restart_info,
        )
    raise NotImplementedError(f"Solver kind {solver_kind!r} is not implemented for transient runs")


def solve_steady(
    case: CaseSpec,
    logger=None,
    *,
    initial_state: MHDState | None = None,
    initial_diagnostics: Diagnostics | None = None,
    append_diagnostics: bool = False,
    restart_info: RestartLogInfo | None = None,
) -> Solution:
    solver_kind = getattr(getattr(case, "solver", None), "kind", "fully_developed_inductionless")
    if solver_kind == "legacy_reduced":
        return _solve_steady_legacy(
            case,
            logger=logger,
            initial_state=initial_state,
            initial_diagnostics=initial_diagnostics,
            append_diagnostics=append_diagnostics,
            restart_info=restart_info,
        )
    if solver_kind == "fully_developed_inductionless":
        steady_case = case if case.solver.mode == "steady" else case.__class__(**{**case.__dict__, "solver": case.solver.__class__(**{**case.solver.__dict__, "mode": "steady"})})
        return _solve_fully_developed(
            steady_case,
            logger=logger,
            initial_state=initial_state,
            initial_diagnostics=initial_diagnostics,
            append_diagnostics=append_diagnostics,
            restart_info=restart_info,
        )
    raise NotImplementedError(f"Solver kind {solver_kind!r} is not implemented for steady runs")

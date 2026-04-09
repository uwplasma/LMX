from __future__ import annotations

from dataclasses import asdict
import numpy as np

import jax
import jax.numpy as jnp

from .core import Diagnostics, MHDState, Solution
from .linear import poisson_residual_norm, solve_poisson_cg_state, solve_poisson_jacobi_state, solve_poisson_lineax
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


def _face_current_emf_and_lorentz_max(
    mesh: StructuredMesh,
    sigma: jnp.ndarray,
    fluid_mask: jnp.ndarray,
    u: jnp.ndarray,
    phi: jnp.ndarray,
    by: jnp.ndarray,
    bz: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    uxb_y = jnp.where(fluid_mask, -u * bz, 0.0)
    uxb_z = jnp.where(fluid_mask, u * by, 0.0)
    emf_y = _face_emf_y(mesh, sigma, uxb_y)
    emf_z = _face_emf_z(mesh, sigma, uxb_z)
    face_jy = _interface_conductance_y(mesh, sigma) * (phi[:-1, :] - phi[1:, :]) + emf_y
    face_jz = _interface_conductance_z(mesh, sigma) * (phi[:, :-1] - phi[:, 1:]) + emf_z
    max_face_current = jnp.maximum(jnp.max(jnp.abs(face_jy)), jnp.max(jnp.abs(face_jz)))
    max_emf = jnp.maximum(jnp.max(jnp.abs(emf_y)), jnp.max(jnp.abs(emf_z)))
    face_bz = 0.5 * (bz[:-1, :] + bz[1:, :])
    face_by = 0.5 * (by[:, :-1] + by[:, 1:])
    max_face_lorentz = jnp.maximum(
        jnp.max(jnp.abs(face_jy * face_bz)),
        jnp.max(jnp.abs(face_jz * face_by)),
    )
    return max_face_current, max_emf, max_face_lorentz


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
    interpolate_direct_fluid_walls: bool,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, float, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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
    outer_count = max(1, outer_iterations)
    u_next, phi, jy, jz, lorentz, potential_residual, potential_iteration_count, face_current_max, emf_max, face_lorentz_max, mean_velocity, applied_forcing, pressure_proxy = jax.lax.fori_loop(
        0,
        outer_count,
        outer_body,
        (u_init, phi0, j0, j0, l0, r0, i0, f0, e0, h0, m0, g0, p0),
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
    applied_forcing: float,
    pressure_proxy: float,
    courant_like: float,
    ohmic: float,
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
            applied_forcing=applied_forcing,
            pressure_proxy=pressure_proxy,
            courant_like=courant_like,
            ohmic_power=ohmic,
        )
    )


def solve_transient(
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
            u_next, phi, jy, jz, lorentz, residual, potential_residual, potential_iteration_count, face_current_max, emf_max, face_lorentz_max, mean_velocity, applied_forcing, pressure_proxy = _step(
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
                ],
                dtype=float,
            )
            return (u_next, step_time), (u_next, phi, jy, jz, lorentz, sample)

        (u_final, time_final), history = jax.lax.scan(scan_step, (initial_u, start_time), xs=None, length=steps)
        u_hist, phi_hist, jy_hist, jz_hist, lorentz_hist, samples = history

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
    potential_history: list[float] = []
    potential_iteration_history: list[float] = []
    residual_value = float("inf")

    for step_index in range(steps):
        step_time = float(start_time + (step_index + 1) * dt)
        u, phi, jy, jz, lorentz, residual, potential_residual, potential_iteration_count, face_current_max, emf_max, face_lorentz_max, mean_velocity, applied_forcing, pressure_proxy = compiled_step(u, step_time)
        residual_value = float(residual)
        u_max_value = float(jnp.max(jnp.abs(u)))
        courant_like = float(u_max_value * dt / jnp.min(mesh.dy))
        ohmic = float(jnp.mean(jy**2 + jz**2))
        max_current = float(jnp.max(jnp.sqrt(jy**2 + jz**2)))
        max_lorentz = float(jnp.max(jnp.abs(lorentz)))
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
            applied_forcing=float(applied_forcing),
            pressure_proxy=float(pressure_proxy),
            courant_like=courant_like,
            ohmic=ohmic,
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


def solve_steady(
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
    potential_history: list[float] = []
    potential_iteration_history: list[float] = []
    step_count = 0

    for step_index in range(max_steps):
        step_time = float(start_time + (step_index + 1) * dt)
        u, phi, jy, jz, lorentz, residual, potential_residual, potential_iteration_count, face_current_max, emf_max, face_lorentz_max, mean_velocity, applied_forcing, pressure_proxy = step_fn(
            u,
            step_time,
        )
        residual_value = float(residual)
        u_max_value = float(jnp.max(jnp.abs(u)))
        courant_like = float(u_max_value * dt / jnp.min(mesh.dy))
        ohmic = float(jnp.mean(jy**2 + jz**2))
        max_current = float(jnp.max(jnp.sqrt(jy**2 + jz**2)))
        max_lorentz = float(jnp.max(jnp.abs(lorentz)))
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
            applied_forcing=float(applied_forcing),
            pressure_proxy=float(pressure_proxy),
            courant_like=courant_like,
            ohmic=ohmic,
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

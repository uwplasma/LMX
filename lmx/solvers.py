from __future__ import annotations

from dataclasses import asdict

import jax
import jax.numpy as jnp

from .core import Diagnostics, MHDState, Solution
from .linear import solve_poisson_jacobi
from .mesh import StructuredMesh, generate_layered_duct_mesh, generate_rect_duct_mesh
from .operators import face_average_x, face_average_z, laplacian_scalar
from .physics import build_material_fields, magnetic_field_components
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


def _face_conductivity_y(sigma: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    sigma_face = 2.0 * sigma[:-1, :] * sigma[1:, :] / jnp.maximum(sigma[:-1, :] + sigma[1:, :], 1e-12)
    west = jnp.pad(sigma_face, ((1, 0), (0, 0)))
    east = jnp.pad(sigma_face, ((0, 1), (0, 0)))
    return west, east


def _face_conductivity_z(sigma: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    sigma_face = 2.0 * sigma[:, :-1] * sigma[:, 1:] / jnp.maximum(sigma[:, :-1] + sigma[:, 1:], 1e-12)
    south = jnp.pad(sigma_face, ((0, 0), (1, 0)))
    north = jnp.pad(sigma_face, ((0, 0), (0, 1)))
    return south, north


def _potential_coefficients(mesh: StructuredMesh, sigma: jnp.ndarray) -> tuple[jnp.ndarray, ...]:
    dy = mesh.dy[:, None]
    dz = mesh.dz[None, :]
    west_sigma, east_sigma = _face_conductivity_y(sigma)
    south_sigma, north_sigma = _face_conductivity_z(sigma)
    west = west_sigma / (dy**2)
    east = east_sigma / (dy**2)
    south = south_sigma / (dz**2)
    north = north_sigma / (dz**2)
    diagonal = west + east + south + north
    diagonal = jnp.where(diagonal > 0.0, diagonal, 1.0)
    return diagonal, west, east, south, north


def _solve_potential(
    mesh: StructuredMesh,
    sigma: jnp.ndarray,
    fluid_mask: jnp.ndarray,
    u: jnp.ndarray,
    by: jnp.ndarray,
    bz: jnp.ndarray,
    anchor: tuple[int, int],
    iterations: int,
) -> jnp.ndarray:
    uxb_y = jnp.where(fluid_mask, -u * bz, 0.0)
    uxb_z = jnp.where(fluid_mask, u * by, 0.0)

    sigma_y = 2.0 * sigma[:-1, :] * sigma[1:, :] / jnp.maximum(sigma[:-1, :] + sigma[1:, :], 1e-12)
    sigma_z = 2.0 * sigma[:, :-1] * sigma[:, 1:] / jnp.maximum(sigma[:, :-1] + sigma[:, 1:], 1e-12)
    conv_y = sigma_y * face_average_x(uxb_y)
    conv_z = sigma_z * face_average_z(uxb_z)
    face_conv_y = jnp.pad(conv_y, ((1, 1), (0, 0)))
    face_conv_z = jnp.pad(conv_z, ((0, 0), (1, 1)))
    rhs = -(
        (face_conv_y[1:, :] - face_conv_y[:-1, :]) / mesh.dy[:, None]
        + (face_conv_z[:, 1:] - face_conv_z[:, :-1]) / mesh.dz[None, :]
    )

    diagonal, west, east, south, north = _potential_coefficients(mesh, sigma)
    phi, _ = solve_poisson_jacobi(diagonal, west, east, south, north, rhs, anchor, iterations)
    return phi


def _compute_current_and_lorentz(
    mesh: StructuredMesh,
    sigma: jnp.ndarray,
    fluid_mask: jnp.ndarray,
    u: jnp.ndarray,
    phi: jnp.ndarray,
    by: jnp.ndarray,
    bz: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    west_phi = jnp.pad(phi[:-1, :], ((1, 0), (0, 0)))
    east_phi = jnp.pad(phi[1:, :], ((0, 1), (0, 0)))
    south_phi = jnp.pad(phi[:, :-1], ((0, 0), (1, 0)))
    north_phi = jnp.pad(phi[:, 1:], ((0, 0), (0, 1)))
    dphi_dy = (east_phi - west_phi) / (2.0 * mesh.dy[:, None])
    dphi_dz = (north_phi - south_phi) / (2.0 * mesh.dz[None, :])
    jy = sigma * (-dphi_dy - u * bz)
    jz = sigma * (-dphi_dz + u * by)
    jy = jnp.where(fluid_mask, jy, 0.0)
    jz = jnp.where(fluid_mask, jz, 0.0)
    lorentz_x = jy * bz - jz * by
    return jy, jz, lorentz_x


def _enforce_velocity_bc(u: jnp.ndarray, fluid_mask: jnp.ndarray) -> jnp.ndarray:
    u = jnp.where(fluid_mask, u, 0.0)
    u = u.at[0, :].set(0.0)
    u = u.at[-1, :].set(0.0)
    u = u.at[:, 0].set(0.0)
    u = u.at[:, -1].set(0.0)
    return u


def _limited_velocity_update(
    current: jnp.ndarray,
    trial: jnp.ndarray,
    fluid_mask: jnp.ndarray,
    max_delta: float = 1e-3,
) -> jnp.ndarray:
    delta = jnp.where(fluid_mask, trial - current, 0.0)
    peak_delta = jnp.max(jnp.abs(delta))
    scale = jnp.minimum(1.0, max_delta / jnp.maximum(peak_delta, 1e-12))
    return jnp.where(fluid_mask, current + scale * delta, 0.0)


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


def _effective_forcing(
    case: CaseSpec,
    sigma: jnp.ndarray,
    fluid_mask: jnp.ndarray,
    by: jnp.ndarray,
    bz: jnp.ndarray,
) -> float:
    if abs(case.forcing) > 0.0:
        return case.forcing
    inlet_velocity = next(
        (speed for boundary in case.boundary_conditions if (speed := _inlet_speed(boundary, case)) is not None),
        None,
    )
    if inlet_velocity is None:
        return 0.0
    magnetic_loading = jnp.where(fluid_mask, sigma * (by**2 + bz**2), 0.0)
    fluid_count = jnp.maximum(jnp.sum(fluid_mask.astype(float)), 1.0)
    mean_loading = float(jnp.sum(magnetic_loading) / fluid_count)
    return mean_loading * inlet_velocity


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
    anchor: tuple[int, int],
    outer_iterations: int,
    potential_iterations: int,
    relaxation: float,
    velocity_update_limit: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, float]:
    def outer_body(_, carry):
        u_iter, _, _, _, _ = carry
        phi = _solve_potential(mesh, sigma, fluid_mask, u_iter, by, bz, anchor, potential_iterations)
        phi = jnp.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0)
        jy, jz, lorentz = _compute_current_and_lorentz(mesh, sigma, fluid_mask, u_iter, phi, by, bz)
        lorentz = jnp.nan_to_num(lorentz, nan=0.0, posinf=0.0, neginf=0.0)
        lap_u = laplacian_scalar(u_iter, mesh, mask=fluid_mask)
        lorentz_damping = jnp.where(fluid_mask, sigma * (by**2 + bz**2), 0.0)
        lorentz_explicit = lorentz + lorentz_damping * u_iter
        rhs = nu * lap_u + (forcing + lorentz_explicit) / rho
        implicit_scale = 1.0 + dt * lorentz_damping / rho
        u_trial = jnp.where(fluid_mask, (u + dt * rhs) / implicit_scale, 0.0)
        relaxed = jnp.where(fluid_mask, (1.0 - relaxation) * u_iter + relaxation * u_trial, 0.0)
        u_next = _limited_velocity_update(u_iter, relaxed, fluid_mask, max_delta=velocity_update_limit)
        u_next = _enforce_velocity_bc(u_next, fluid_mask)
        u_next = jnp.nan_to_num(u_next, nan=0.0, posinf=5.0, neginf=-5.0)
        u_next = jnp.clip(u_next, -5.0, 5.0)
        return (u_next, phi, jy, jz, lorentz)

    u_init = _enforce_velocity_bc(u, fluid_mask)
    phi0 = jnp.zeros_like(u)
    j0 = jnp.zeros_like(u)
    l0 = jnp.zeros_like(u)
    outer_count = max(1, outer_iterations)
    u_next, phi, jy, jz, lorentz = jax.lax.fori_loop(
        0,
        outer_count,
        outer_body,
        (u_init, phi0, j0, j0, l0),
    )
    residual = jnp.max(jnp.abs(u_next - u))
    return u_next, phi, jy, jz, lorentz, residual


def solve_transient(case: CaseSpec) -> Solution:
    mesh = _build_mesh(case)
    materials = build_material_fields(case, mesh)
    _, by, bz = magnetic_field_components(case.magnetic_field, mesh)
    forcing = _effective_forcing(case, materials.conductivity, materials.fluid_mask, by, bz)

    initial_u = jnp.where(materials.fluid_mask, case.initial_velocity, 0.0)
    initial_u = _enforce_velocity_bc(initial_u, materials.fluid_mask)
    dt = case.time_stepper.dt
    steps = min(case.time_stepper.max_steps, max(1, int(round(case.time_stepper.t_final / dt))))

    def scan_step(carry, _):
        u, time = carry
        u_next, phi, jy, jz, lorentz, residual = _step(
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
            anchor=case.reference_phi_cell,
            outer_iterations=case.time_stepper.outer_iterations,
            potential_iterations=case.time_stepper.potential_iterations,
            relaxation=case.time_stepper.relaxation,
            velocity_update_limit=case.time_stepper.velocity_update_limit,
        )
        courant_like = jnp.max(jnp.abs(u_next)) * dt / jnp.min(mesh.dy)
        ohmic = jnp.mean(jy**2 + jz**2)
        sample = jnp.asarray([residual, courant_like, ohmic], dtype=float)
        return (u_next, time + dt), (u_next, phi, jy, jz, lorentz, sample)

    (u_final, time_final), history = jax.lax.scan(scan_step, (initial_u, 0.0), xs=None, length=steps)
    u_hist, phi_hist, jy_hist, jz_hist, lorentz_hist, samples = history

    state = MHDState(
        u=u_final,
        phi=phi_hist[-1],
        jy=jy_hist[-1],
        jz=jz_hist[-1],
        lorentz_x=lorentz_hist[-1],
        time=float(time_final),
        residual=float(samples[-1, 0]),
    )
    diagnostics = Diagnostics(
        residual_history=samples[:, 0],
        courant_like=samples[:, 1],
        ohmic_power=samples[:, 2],
    )
    return Solution(mesh=mesh, state=state, diagnostics=diagnostics, case_name=case.name)


def solve_steady(case: CaseSpec) -> Solution:
    mesh = _build_mesh(case)
    materials = build_material_fields(case, mesh)
    _, by, bz = magnetic_field_components(case.magnetic_field, mesh)
    forcing = _effective_forcing(case, materials.conductivity, materials.fluid_mask, by, bz)

    initial_u = jnp.where(materials.fluid_mask, case.initial_velocity, 0.0)
    initial_u = _enforce_velocity_bc(initial_u, materials.fluid_mask)
    dt = case.time_stepper.dt
    max_steps = max(1, case.time_stepper.max_steps)
    tolerance = float(case.time_stepper.steady_tolerance)

    def compiled_step(u: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, float]:
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
            anchor=case.reference_phi_cell,
            outer_iterations=case.time_stepper.outer_iterations,
            potential_iterations=case.time_stepper.potential_iterations,
            relaxation=case.time_stepper.relaxation,
            velocity_update_limit=case.time_stepper.velocity_update_limit,
        )

    step_fn = jax.jit(compiled_step)

    u = initial_u
    phi = jnp.zeros_like(u)
    jy = jnp.zeros_like(u)
    jz = jnp.zeros_like(u)
    lorentz = jnp.zeros_like(u)
    residual_value = float("inf")
    residual_history: list[float] = []
    courant_history: list[float] = []
    ohmic_history: list[float] = []
    step_count = 0

    for step_index in range(max_steps):
        u, phi, jy, jz, lorentz, residual = step_fn(u)
        residual_value = float(residual)
        courant_like = float(jnp.max(jnp.abs(u)) * dt / jnp.min(mesh.dy))
        ohmic = float(jnp.mean(jy**2 + jz**2))
        residual_history.append(residual_value)
        courant_history.append(courant_like)
        ohmic_history.append(ohmic)
        step_count = step_index + 1
        if residual_value <= tolerance:
            break

    state = MHDState(
        u=u,
        phi=phi,
        jy=jy,
        jz=jz,
        lorentz_x=lorentz,
        time=float(step_count * dt),
        residual=residual_value,
    )
    diagnostics = Diagnostics(
        residual_history=jnp.asarray(residual_history, dtype=float),
        courant_like=jnp.asarray(courant_history, dtype=float),
        ohmic_power=jnp.asarray(ohmic_history, dtype=float),
    )
    return Solution(mesh=mesh, state=state, diagnostics=diagnostics, case_name=case.name)

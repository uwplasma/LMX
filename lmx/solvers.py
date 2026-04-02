from __future__ import annotations

from dataclasses import asdict

import jax
import jax.numpy as jnp

from .core import Diagnostics, MHDState, Solution
from .linear import poisson_residual_norm, solve_poisson_cg_state, solve_poisson_jacobi_state, solve_poisson_lineax
from .mesh import StructuredMesh, generate_layered_duct_mesh, generate_rect_duct_mesh
from .operators import gradient_scalar, laplacian_scalar
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
    return "cg" if bool(jnp.all(fluid_mask)) else "cg_volume"


def _compute_current_and_lorentz(
    mesh: StructuredMesh,
    sigma: jnp.ndarray,
    fluid_mask: jnp.ndarray,
    u: jnp.ndarray,
    phi: jnp.ndarray,
    by: jnp.ndarray,
    bz: jnp.ndarray,
    use_face_currents: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    if use_face_currents:
        uxb_y = jnp.where(fluid_mask, -u * bz, 0.0)
        uxb_z = jnp.where(fluid_mask, u * by, 0.0)
        face_jy = _interface_conductance_y(mesh, sigma) * (phi[:-1, :] - phi[1:, :]) + _face_emf_y(mesh, sigma, uxb_y)
        face_jz = _interface_conductance_z(mesh, sigma) * (phi[:, :-1] - phi[:, 1:]) + _face_emf_z(mesh, sigma, uxb_z)
        jy = 0.5 * (jnp.pad(face_jy, ((1, 0), (0, 0))) + jnp.pad(face_jy, ((0, 1), (0, 0))))
        jz = 0.5 * (jnp.pad(face_jz, ((0, 0), (1, 0))) + jnp.pad(face_jz, ((0, 0), (0, 1))))
    else:
        dphi_dy, dphi_dz = gradient_scalar(phi, mesh)
        jy = sigma * (-dphi_dy - u * bz)
        jz = sigma * (-dphi_dz + u * by)
    jy = jnp.where(fluid_mask, jy, 0.0)
    jz = jnp.where(fluid_mask, jz, 0.0)
    lorentz_x = jy * bz - jz * by
    return jy, jz, lorentz_x


def _face_current_and_emf_max(
    mesh: StructuredMesh,
    sigma: jnp.ndarray,
    fluid_mask: jnp.ndarray,
    u: jnp.ndarray,
    phi: jnp.ndarray,
    by: jnp.ndarray,
    bz: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    uxb_y = jnp.where(fluid_mask, -u * bz, 0.0)
    uxb_z = jnp.where(fluid_mask, u * by, 0.0)
    emf_y = _face_emf_y(mesh, sigma, uxb_y)
    emf_z = _face_emf_z(mesh, sigma, uxb_z)
    face_jy = _interface_conductance_y(mesh, sigma) * (phi[:-1, :] - phi[1:, :]) + emf_y
    face_jz = _interface_conductance_z(mesh, sigma) * (phi[:, :-1] - phi[:, 1:]) + emf_z
    max_face_current = jnp.maximum(jnp.max(jnp.abs(face_jy)), jnp.max(jnp.abs(face_jz)))
    max_emf = jnp.maximum(jnp.max(jnp.abs(emf_y)), jnp.max(jnp.abs(emf_z)))
    return max_face_current, max_emf


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
    anchor: tuple[int, int],
    outer_iterations: int,
    potential_iterations: int,
    potential_tolerance: float | None,
    potential_relaxation: float,
    potential_solver: str,
    relaxation: float,
    velocity_update_limit: float,
    current_reconstruction: str,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, float, jnp.ndarray, jnp.ndarray]:
    def outer_body(_, carry):
        u_iter, _, _, _, _, _, _, _, _ = carry
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
            use_face_currents=current_reconstruction == "face_averaged",
        )
        lorentz = jnp.nan_to_num(lorentz, nan=0.0, posinf=0.0, neginf=0.0)
        lap_u = laplacian_scalar(u_iter, mesh, mask=fluid_mask)
        lorentz_damping = jnp.where(fluid_mask, sigma * (by**2 + bz**2), 0.0)
        lorentz_explicit = lorentz + lorentz_damping * u_iter
        implicit_scale = 1.0 + dt * lorentz_damping / rho
        base_trial = jnp.where(fluid_mask, (u + dt * (nu * lap_u + lorentz_explicit / rho)) / implicit_scale, 0.0)
        pressure_sensitivity = jnp.where(fluid_mask, (dt / rho) / implicit_scale, 0.0)
        forcing_value = jnp.asarray(forcing, dtype=u.dtype)
        if target_mean_velocity is not None:
            active_mask = _active_velocity_mask(fluid_mask)
            fluid_count = jnp.maximum(jnp.sum(active_mask.astype(u.dtype)), 1.0)
            target = jnp.asarray(target_mean_velocity, dtype=u.dtype)
            mean_base = jnp.sum(jnp.where(active_mask, base_trial, 0.0)) / fluid_count
            mean_sensitivity = jnp.sum(jnp.where(active_mask, pressure_sensitivity, 0.0)) / fluid_count
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
        u_next = _enforce_velocity_bc(u_next, fluid_mask)
        u_next = jnp.nan_to_num(u_next, nan=0.0, posinf=5.0, neginf=-5.0)
        u_next = jnp.clip(u_next, -5.0, 5.0)
        face_current_max, emf_max = _face_current_and_emf_max(mesh, sigma, fluid_mask, u_iter, phi, by, bz)
        return (u_next, phi, jy, jz, lorentz, potential_residual, potential_iteration_count, face_current_max, emf_max)

    u_init = _enforce_velocity_bc(u, fluid_mask)
    phi0 = jnp.zeros_like(u)
    j0 = jnp.zeros_like(u)
    l0 = jnp.zeros_like(u)
    r0 = jnp.asarray(0.0, dtype=u.dtype)
    i0 = jnp.asarray(0, dtype=jnp.int32)
    f0 = jnp.asarray(0.0, dtype=u.dtype)
    e0 = jnp.asarray(0.0, dtype=u.dtype)
    outer_count = max(1, outer_iterations)
    u_next, phi, jy, jz, lorentz, potential_residual, potential_iteration_count, face_current_max, emf_max = jax.lax.fori_loop(
        0,
        outer_count,
        outer_body,
        (u_init, phi0, j0, j0, l0, r0, i0, f0, e0),
    )
    residual = jnp.max(jnp.abs(u_next - u))
    return u_next, phi, jy, jz, lorentz, residual, potential_residual, potential_iteration_count, face_current_max, emf_max


def solve_transient(case: CaseSpec) -> Solution:
    mesh = _build_mesh(case)
    materials = build_material_fields(case, mesh)
    target_mean_velocity = _target_mean_velocity(case)
    potential_solver = _resolve_potential_solver(case.time_stepper.potential_solver, materials.fluid_mask)

    initial_u = jnp.where(materials.fluid_mask, case.initial_velocity, 0.0)
    initial_u = _enforce_velocity_bc(initial_u, materials.fluid_mask)
    dt = case.time_stepper.dt
    steps = min(case.time_stepper.max_steps, max(1, int(round(case.time_stepper.t_final / dt))))

    def scan_step(carry, _):
        u, time = carry
        step_time = time + dt
        _, by, bz = magnetic_field_components(case.magnetic_field, mesh, time=step_time)
        forcing = _explicit_forcing(case.forcing, by.dtype)
        u_next, phi, jy, jz, lorentz, residual, potential_residual, potential_iteration_count, face_current_max, emf_max = _step(
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
        )
        courant_like = jnp.max(jnp.abs(u_next)) * dt / jnp.min(mesh.dy)
        ohmic = jnp.mean(jy**2 + jz**2)
        max_current = jnp.max(jnp.sqrt(jy**2 + jz**2))
        max_lorentz = jnp.max(jnp.abs(lorentz))
        sample = jnp.asarray(
            [
                step_time,
                jnp.max(jnp.abs(u_next)),
                residual,
                courant_like,
                ohmic,
                max_current,
                face_current_max,
                emf_max,
                max_lorentz,
                potential_residual,
                potential_iteration_count,
            ],
            dtype=float,
        )
        return (u_next, step_time), (u_next, phi, jy, jz, lorentz, sample)

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
        time_history=samples[:, 0],
        u_max_history=samples[:, 1],
        residual_history=samples[:, 2],
        courant_like=samples[:, 3],
        ohmic_power=samples[:, 4],
        current_max_history=samples[:, 5],
        face_current_max_history=samples[:, 6],
        emf_max_history=samples[:, 7],
        lorentz_max_history=samples[:, 8],
        potential_residual_history=samples[:, 9],
        potential_iterations_history=samples[:, 10],
    )
    return Solution(mesh=mesh, state=state, diagnostics=diagnostics, case_name=case.name)


def solve_steady(case: CaseSpec) -> Solution:
    mesh = _build_mesh(case)
    materials = build_material_fields(case, mesh)
    target_mean_velocity = _target_mean_velocity(case)
    potential_solver = _resolve_potential_solver(case.time_stepper.potential_solver, materials.fluid_mask)

    initial_u = jnp.where(materials.fluid_mask, case.initial_velocity, 0.0)
    initial_u = _enforce_velocity_bc(initial_u, materials.fluid_mask)
    dt = case.time_stepper.dt
    max_steps = max(1, case.time_stepper.max_steps)
    tolerance = float(case.time_stepper.steady_tolerance)
    potential_tolerance = case.time_stepper.steady_potential_tolerance

    def compiled_step(
        u: jnp.ndarray,
        time: float,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, float, jnp.ndarray, jnp.ndarray]:
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
    time_history: list[float] = []
    u_max_history: list[float] = []
    current_max_history: list[float] = []
    face_current_max_history: list[float] = []
    emf_max_history: list[float] = []
    lorentz_max_history: list[float] = []
    potential_history: list[float] = []
    potential_iteration_history: list[float] = []
    step_count = 0

    for step_index in range(max_steps):
        step_time = float((step_index + 1) * dt)
        u, phi, jy, jz, lorentz, residual, potential_residual, potential_iteration_count, face_current_max, emf_max = step_fn(
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
        residual_history.append(residual_value)
        courant_history.append(courant_like)
        ohmic_history.append(ohmic)
        current_max_history.append(max_current)
        face_current_max_history.append(float(face_current_max))
        emf_max_history.append(float(emf_max))
        lorentz_max_history.append(max_lorentz)
        potential_history.append(float(potential_residual))
        potential_iteration_history.append(float(potential_iteration_count))
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
        time=float(step_count * dt),
        residual=residual_value,
    )
    diagnostics = Diagnostics(
        time_history=jnp.asarray(time_history, dtype=float),
        u_max_history=jnp.asarray(u_max_history, dtype=float),
        residual_history=jnp.asarray(residual_history, dtype=float),
        courant_like=jnp.asarray(courant_history, dtype=float),
        ohmic_power=jnp.asarray(ohmic_history, dtype=float),
        current_max_history=jnp.asarray(current_max_history, dtype=float),
        face_current_max_history=jnp.asarray(face_current_max_history, dtype=float),
        emf_max_history=jnp.asarray(emf_max_history, dtype=float),
        lorentz_max_history=jnp.asarray(lorentz_max_history, dtype=float),
        potential_residual_history=jnp.asarray(potential_history, dtype=float),
        potential_iterations_history=jnp.asarray(potential_iteration_history, dtype=float),
    )
    return Solution(mesh=mesh, state=state, diagnostics=diagnostics, case_name=case.name)

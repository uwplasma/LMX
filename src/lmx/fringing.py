"""Public 3-D fringing problems, applications, solves, and validation."""

from __future__ import annotations

import math
from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from solvax import (
    aitken_relaxation,
    anderson_mixing,
    checkpointed_fori_loop,
)

from . import _fringing_common as common
from . import _fringing_duct as duct
from . import _fringing_pipe as pipe
from ._fringing_common import (
    _EXTRUDED_NUMERICAL_RESULTS,
    _coordinate_scale,
)
from .cases import (
    build_extruded_problem_from_case,
    build_layered_duct_extruded_problem,
    build_magnetic_obstacle_rect_extruded_problem,
    build_pipe_ogrid_extruded_problem,
    build_square_duct_extruded_problem,
    smooth_fringing_profile,
)
from .mesh import (
    _cross_section_mesh,
    _sample_station_magnetic_field,
)
from .physics import build_material_fields
from .specs import (
    ExtrudedFieldBundle,
    ExtrudedInductionlessProblem,
    ExtrudedInductionlessSolution,
    ExtrudedIterationProgress,
    require_finite,
)
from .validation import (
    _bundle_station_history,
    validate_extruded_inductionless_solution,
    validate_magnetic_obstacle_baseline,
    validate_variable_field_extruded_solution,
    validate_variable_field_pipe_solution,
)

__all__ = (
    "build_extruded_problem_from_case",
    "build_layered_duct_extruded_problem",
    "build_magnetic_obstacle_rect_extruded_problem",
    "build_pipe_ogrid_extruded_problem",
    "build_square_duct_extruded_problem",
    "smooth_fringing_profile",
    "validate_extruded_inductionless_solution",
    "validate_magnetic_obstacle_baseline",
    "validate_variable_field_extruded_solution",
    "validate_variable_field_pipe_solution",
    "solve_extruded_inductionless",
    "evolve_extruded_fields",
    "extruded_engineering_objectives",
)


def _solve_extruded_projection(
    problem: ExtrudedInductionlessProblem,
    *,
    initial_bundle: ExtrudedFieldBundle | None = None,
    num_devices: int | None = None,
    progress_callback: Callable[[ExtrudedIterationProgress], None] | None = None,
    phase_timing_callback: Callable[[str, float], None] | None = None,
    checkpoint_interval: int | None = None,
    design_parameters: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, int, int | None]
    | None = None,
) -> ExtrudedFieldBundle | tuple[jnp.ndarray, ...]:
    case = problem.case
    with jax.ensure_compile_time_eval():
        mesh = _cross_section_mesh(case)
    use_alex_b2_finite_volume = (
        case.name.startswith("alex_b2-fringing-square_") and case.geometry.kind == "layered_duct"
    )
    use_alex_b1_finite_volume = (
        case.name.startswith("alex_b1-fringing-pipe_") and case.geometry.kind == "pipe_ogrid"
    )
    if design_parameters is not None and (
        case.geometry.kind not in {"rect_duct", "layered_duct", "pipe_ogrid"} or use_alex_b2_finite_volume
    ):
        raise NotImplementedError("differentiable extruded fields do not yet support ALEX B2")
    if case.name.startswith("alex_") and not (use_alex_b1_finite_volume or use_alex_b2_finite_volume):
        raise NotImplementedError(
            "Unsupported ALEX production case; only the frozen B1 pipe and B2 square "
            "finite-volume paths are implemented"
        )
    if num_devices is not None and num_devices > 1 and not use_alex_b2_finite_volume:
        raise NotImplementedError("Production spatial sharding currently supports the ALEX B2 duct path")
    if (
        use_alex_b2_finite_volume
        and case.solver.coupling_acceleration == "anderson"
        and case.solver.coupling_history_depth != 2
    ):
        raise ValueError("B2 conservative Anderson mixing requires history depth 2")
    runtime = (
        initial_bundle,
        num_devices,
        progress_callback,
        phase_timing_callback,
        checkpoint_interval,
        design_parameters,
    )
    if case.geometry.kind == "pipe_ogrid":
        return _solve_pipe_projection(problem, mesh, use_alex_b1_finite_volume, runtime)
    return _solve_duct_projection(problem, mesh, use_alex_b2_finite_volume, runtime)


def _solve_pipe_projection(
    problem: ExtrudedInductionlessProblem,
    mesh,
    use_alex_b1_finite_volume: bool,
    runtime,
) -> ExtrudedFieldBundle | tuple[jnp.ndarray, ...]:
    initial_bundle, num_devices, progress_callback, _, checkpoint_interval, design_parameters = runtime
    case = problem.case
    if case.geometry.kind == "pipe_ogrid":
        with jax.ensure_compile_time_eval():
            materials = build_material_fields(case, mesh)
        if design_parameters is None:
            forcing, magnetic_scale, conductivity_scale = case.forcing, 1.0, 1.0
            axial_scale = radial_scale = 1.0
            checkpoint_size = None
        else:
            forcing, magnetic_scale, conductivity_scale, geometry_scale, outer_steps, checkpoint_size = (
                design_parameters
            )
            geometry_scale = jnp.asarray(geometry_scale, dtype=float)
            if geometry_scale.ndim == 0:
                axial_scale = radial_scale = geometry_scale
            elif geometry_scale.shape == (2,):
                axial_scale, radial_scale = geometry_scale
            else:
                raise ValueError("pipe geometry_scale must be scalar or (axial, radial)")
        base_x = jnp.asarray(mesh.x_centers, dtype=float)
        base_r_faces = jnp.asarray(mesh.y_faces, dtype=float)
        base_r = jnp.asarray(mesh.y_centers, dtype=float)
        x = axial_scale * base_x
        r_faces = radial_scale * base_r_faces
        r = radial_scale * base_r
        theta = jnp.asarray(mesh.z_centers, dtype=float)
        nx, nr, ntheta = len(x), len(r), len(theta)
        base_dx = case.geometry.length / nx
        dx = axial_scale * base_dx
        base_dr = (
            (case.geometry.radius or 0.5 * case.geometry.width) + max(case.geometry.wall_thickness)
        ) / nr
        dr = radial_scale * base_dr
        dr_widths = radial_scale * jnp.asarray(mesh.dy, dtype=float)
        dtheta = 2.0 * math.pi / ntheta
        sigma = common._broadcast_cross_section(materials.conductivity, nx)
        rho = common._broadcast_cross_section(materials.density, nx)
        nu = common._broadcast_cross_section(materials.viscosity, nx)
        fluid_mask = common._broadcast_cross_section(materials.fluid_mask.astype(float), nx) > 0.5
        radial_fluid_count = case.geometry.nr or case.geometry.ny
        rr = jnp.broadcast_to(jnp.maximum(r[None, :, None], 0.5 * dr), (nx, nr, ntheta))
        theta_grid = jnp.broadcast_to(theta[None, None, :], (nx, nr, ntheta))
        field_scale = jnp.asarray(problem.profile.field_scale, dtype=float)
        with jax.ensure_compile_time_eval():
            static_theta = jnp.asarray(mesh.z_centers, dtype=float)
            static_theta_grid = jnp.broadcast_to(static_theta[None, None, :], (nx, nr, ntheta))
            base_rr = jnp.broadcast_to(jnp.maximum(base_r[None, :, None], 0.5 * base_dr), (nx, nr, ntheta))
            bx, by, bz = _sample_station_magnetic_field(
                case,
                field_scale=field_scale,
                x=problem.profile.x,
                y=base_rr[0] * jnp.cos(static_theta_grid[0]),
                z=base_rr[0] * jnp.sin(static_theta_grid[0]),
                volume_field=problem.profile.volume_field,
            )
        stability_bx, stability_by, stability_bz = bx, by, bz
        (bx, by, bz), sigma = common._scale_projection_properties(
            (bx, by, bz), sigma, fluid_mask, magnetic_scale, conductivity_scale
        )
        br = by * jnp.cos(theta_grid) + bz * jnp.sin(theta_grid)
        btheta = -by * jnp.sin(theta_grid) + bz * jnp.cos(theta_grid)

        u, v, w, p, phi = common._initial_projection_fields(case, fluid_mask, initial_bundle)

        (
            u,
            v,
            w,
            p,
            phi,
            sigma,
            rho,
            nu,
            fluid_mask,
            bx,
            by,
            bz,
            br,
            btheta,
            rr,
            theta_grid,
        ) = common._shard_extruded_fields(
            (
                u,
                v,
                w,
                p,
                phi,
                sigma,
                rho,
                nu,
                fluid_mask,
                bx,
                by,
                bz,
                br,
                btheta,
                rr,
                theta_grid,
            ),
            num_devices=num_devices,
        )

        with jax.ensure_compile_time_eval():
            min_dr = float(jnp.min(mesh.dy))
            static_r = jnp.asarray(mesh.y_centers, dtype=float)
            min_arc = (
                float(jnp.min(jnp.maximum(static_r[1:], 0.5 * min_dr))) * dtheta
                if nr > 1
                else max(float(static_r[0]) * dtheta, 0.5 * min_dr * dtheta)
            )
            static_sigma = common._broadcast_cross_section(materials.conductivity, nx)
            static_rho = common._broadcast_cross_section(materials.density, nx)
            static_nu = common._broadcast_cross_section(materials.viscosity, nx)
            static_mask = common._broadcast_cross_section(materials.fluid_mask.astype(float), nx) > 0.5
            static_faces = jnp.asarray(mesh.y_faces, dtype=float)
            static_radius = jnp.maximum(static_r, 0.5 * base_dr)
            reference_flow_area = float(
                jnp.sum(
                    jnp.where(
                        static_mask[0],
                        static_radius[:, None] * jnp.diff(static_faces)[:, None] * dtheta,
                        0.0,
                    )
                )
            )
            inverse_diffusive_scale = float(
                jnp.max(static_nu)
                * (
                    1.0 / max(base_dx**2, 1.0e-12)
                    + 1.0 / max(min_dr**2, 1.0e-12)
                    + 1.0 / max(min_arc**2, 1.0e-12)
                )
            )
            inverse_electromagnetic_scale = float(
                jnp.max(
                    jnp.where(
                        static_mask,
                        static_sigma * (stability_bx**2 + stability_by**2 + stability_bz**2) / static_rho,
                        0.0,
                    )
                )
            )
            field_energy_max = float(jnp.max(stability_bx**2 + stability_by**2 + stability_bz**2))
        stability_safety = (
            0.001
            if use_alex_b1_finite_volume
            else (0.01 if float(case.geometry.target_ha or 0.0) >= 100.0 else 0.1)
        )
        stable_dt = stability_safety / max(
            inverse_electromagnetic_scale
            if use_alex_b1_finite_volume
            else inverse_diffusive_scale + inverse_electromagnetic_scale,
            1.0e-12,
        )
        dt = min(float(case.time_stepper.dt), stable_dt)
        cell_area = rr * jnp.diff(r_faces)[None, :, None] * dtheta
        fluid_cell_area = jnp.where(fluid_mask, cell_area, 0.0)
        target_flow_rate = (
            float(jnp.mean(jnp.sum(u * fluid_cell_area, axis=(1, 2))))
            if initial_bundle is not None
            else case.initial_velocity * reference_flow_area
            if case.initial_velocity != 0.0
            else None
        )
        if design_parameters is None:
            outer_steps = min(case.time_stepper.max_steps, max(6, case.solver.coupling_iterations * 2))
        poisson_iterations = (
            case.time_stepper.potential_iterations
            if use_alex_b1_finite_volume
            else min(case.time_stepper.potential_iterations, 80)
        )
        poisson_tolerance = case.solver.coupling_tolerance
        electric_iterations = max(poisson_iterations, 4000)
        electric_tolerance = min(poisson_tolerance, 1.0e-12)
        projection_iterations = max(poisson_iterations, 4000)
        projection_tolerance = min(poisson_tolerance, 1.0e-12)
        momentum_iterations = max(poisson_iterations, 2000 if use_alex_b1_finite_volume else 400)
        momentum_tolerance = min(poisson_tolerance, 1.0e-10)
        velocity_limit = max(5.0, 2.0 * math.sqrt(float(case.geometry.target_ha or 1.0)))
        scalar_limit = max(20.0, 2.0 * field_energy_max)
        electric_potential_scale = max(1.0, math.sqrt(field_energy_max))
        residual_by_step: list[float] = []
        component_residual_by_step: list[tuple[float, ...]] = []
        pressure_residual_by_step: list[float] = []
        electric_linear_by_step: list[tuple[float, ...]] = []
        potential_residual_by_step: list[float] = []
        fixed_point_iterates: list[jnp.ndarray] = []
        fixed_point_residuals: list[jnp.ndarray] = []
        previous_fixed_point_residual: jnp.ndarray | None = None
        fixed_point_relaxation = jnp.asarray(1.0, dtype=u.dtype)
        fixed_point_scale = jnp.asarray(
            [
                velocity_limit,
                velocity_limit,
                velocity_limit,
                electric_potential_scale,
            ],
            dtype=u.dtype,
        )[:, None, None, None]
        if use_alex_b1_finite_volume:
            count = radial_fluid_count
            faces = r_faces[: count + 1]
            centers = r[:count]
            steady_reaction = (
                2.0
                * sigma[:, :count, :]
                * (bx[:, :count, :] ** 2 + br[:, :count, :] ** 2 + btheta[:, :count, :] ** 2)
                / rho[:, :count, :]
            )
            steady_coefficients = pipe._pipe_variable_diffusion_coefficients_3d(
                nu[:, :count, :], dx=dx, r_faces=faces, r_centers=centers, dtheta=dtheta
            )
            radial_widths = jnp.diff(faces)
            wall_sink = (
                jnp.zeros_like(steady_reaction)
                .at[:, -1, :]
                .set(
                    nu[:, count - 1, :]
                    * faces[-1]
                    / jnp.maximum(centers[-1] * radial_widths[-1] * (0.5 * radial_widths[-1]), 1.0e-20)
                )
            )
            steady_rate_diagonal = sum(steady_coefficients) + wall_sink + steady_reaction
            pressure_preconditioner_mobility = 1.0 / jnp.maximum(
                rho[:, :count, :] * steady_rate_diagonal, 1.0e-20
            )
            momentum_viscosity = nu[:, :count, :]

            def momentum_solve(rhs, initial):
                return pipe._solvax_diffusion_pipe(
                    rhs,
                    momentum_viscosity,
                    dt=None,
                    dx=dx,
                    r_faces=faces,
                    r_centers=centers,
                    dtheta=dtheta,
                    iterations=momentum_iterations,
                    tolerance=momentum_tolerance,
                    initial_field=initial,
                    reaction=steady_reaction,
                )

            modal_factor_key = None
            if design_parameters is None:
                kernel_key = (
                    "b1_diffusion",
                    u.shape,
                    count,
                    dt,
                    dx,
                    tuple(np.asarray(faces)),
                    tuple(np.asarray(centers)),
                    dtheta,
                    momentum_iterations,
                    momentum_tolerance,
                )
                parameter_key = common._array_fingerprint(
                    rho[:, :count, :],
                    momentum_viscosity,
                    steady_reaction,
                    fluid_cell_area[:, :count, :],
                )
                momentum_solve = common._reuse_fringing_jit(
                    ("b1_momentum", jax.default_backend(), kernel_key, parameter_key),
                    jax.jit(momentum_solve),
                )
                modal_factor_key = (
                    "b1_modal_factors",
                    "retained",
                    jax.default_backend(),
                    u.dtype.str,
                    kernel_key,
                    parameter_key,
                )
            response_rhs = 1.0 / rho[:, :count, :]
            response_fluid, _, _ = momentum_solve(response_rhs, jnp.zeros_like(response_rhs))
            basis_rhs = jnp.eye(nx, dtype=u.dtype)[:, :, None, None] / rho[None, :, :count, :]
            zero = jnp.zeros_like(response_fluid)
            basis_response = jnp.stack(tuple(momentum_solve(rhs, zero)[0] for rhs in basis_rhs))
            flow_response_matrix = jnp.sum(
                basis_response * fluid_cell_area[None, :, :count, :], axis=(2, 3)
            ).T
            unit_pressure_response = jnp.zeros_like(u).at[:, :count, :].set(response_fluid)
        else:
            unit_pressure_response, _, _ = pipe._enforce_pipe_velocity_bc(
                jnp.where(fluid_mask, dt / rho, 0.0),
                jnp.zeros_like(u),
                jnp.zeros_like(u),
                r_centers=r,
                r_faces=r_faces,
                fluid_mask=fluid_mask,
                radial_fluid_count=radial_fluid_count,
            )
        generic_step = (
            None
            if use_alex_b1_finite_volume
            else partial(
                pipe._generic_pipe_step,
                material=(sigma, rho, nu, fluid_mask, cell_area),
                field=(bx, br, btheta),
                forcing=jnp.asarray(forcing),
                metric=(dt, dx, dr, dtheta, r, rr, r_faces),
                solves=(
                    projection_iterations,
                    projection_tolerance,
                    electric_iterations,
                    electric_tolerance,
                ),
                limits=(velocity_limit, scalar_limit),
                flow=(target_flow_rate, unit_pressure_response, radial_fluid_count),
            )
        )
        b1_step = None
        if use_alex_b1_finite_volume:
            if target_flow_rate is None:
                raise ValueError("ALEX B1 requires its frozen fixed mean flow rate")
            fluid = (slice(None), slice(0, count), slice(None))

            def b1_step(state):
                u0, v0, w0, _, potential = state
                potential_gradient = pipe._pipe_gradient_3d(
                    potential, dx=dx, dr=dr_widths, dtheta=dtheta, r=rr
                )
                emf = common._cross((u0, v0, w0), (bx, br, btheta))
                current = tuple(
                    sigma * (-gradient + source)
                    for gradient, source in zip(potential_gradient, emf, strict=True)
                )
                jx0, jr0, jtheta0 = current
                lorentz = common._cross(current, (bx, br, btheta))
                previous = (u0[fluid], v0[fluid], w0[fluid])
                drives = (forcing + lorentz[0][fluid], lorentz[1][fluid], lorentz[2][fluid])
                predicted = tuple(
                    momentum_solve(drive / rho[fluid] + steady_reaction * value, value)[0]
                    for drive, value in zip(drives, previous, strict=True)
                )
                projection = pipe._steady_stokes_projection_pipe(
                    *predicted,
                    rho[fluid],
                    response_fluid,
                    fluid_cell_area[fluid],
                    lambda rhs: momentum_solve(rhs, zero)[0],
                    target_flow_rate=target_flow_rate,
                    dx=dx,
                    r_faces=faces,
                    r_centers=centers,
                    dtheta=dtheta,
                    pressure_iterations=projection_iterations,
                    pressure_tolerance=momentum_tolerance,
                    flow_response_matrix=flow_response_matrix,
                    pressure_preconditioner_mobility=pressure_preconditioner_mobility,
                    modal_momentum_coefficients=steady_coefficients,
                    modal_momentum_sink=wall_sink + steady_reaction,
                    modal_stabilization=True,
                    modal_factor_key=modal_factor_key,
                    physical_tolerance=common.ALEX_BALANCE_TOLERANCE,
                )
                velocity = tuple(
                    jnp.zeros_like(value).at[fluid].set(projected)
                    for value, projected in zip((u0, v0, w0), projection[:3], strict=True)
                )
                pressure = jnp.zeros_like(potential).at[fluid].set(projection[3])
                u1, v1, w1 = velocity
                emf = common._cross(velocity, (bx, br, btheta))
                potential, *electric = pipe._separable_pressure_poisson_pipe(
                    pipe._pipe_conservative_emf_rhs_3d(
                        sigma, *emf, dx=dx, r_faces=r_faces, r_centers=r, dtheta=dtheta
                    ),
                    sigma,
                    dx=dx,
                    r_faces=r_faces,
                    r_centers=r,
                    dtheta=dtheta,
                    tolerance=electric_tolerance,
                )
                potential = jnp.clip(potential, -scalar_limit, scalar_limit)
                fluxes = pipe._pipe_conservative_current_fluxes_3d(
                    sigma, potential, *emf, dx=dx, r_faces=r_faces, r_centers=r, dtheta=dtheta
                )
                current = tuple(
                    jnp.clip(value, -scalar_limit, scalar_limit)
                    for value in (
                        0.5 * (fluxes[0][1:] + fluxes[0][:-1]),
                        0.5 * (fluxes[1][:, 1:, :] + fluxes[1][:, :-1, :]),
                        0.5 * (fluxes[2] + jnp.roll(fluxes[2], 1, axis=2)),
                    )
                )
                jx, jr, jtheta = current
                lorentz = common._cross(current, (bx, br, btheta))
                div_j, _, _ = pipe._pipe_conservative_current_diagnostics_3d(
                    sigma,
                    potential,
                    *emf,
                    dx=dx,
                    r_faces=r_faces,
                    r_centers=r,
                    dtheta=dtheta,
                    fluxes=fluxes,
                )
                observables = (
                    *current,
                    *lorentz,
                    div_j,
                    projection[5],
                    forcing + projection[4],
                    projection[6],
                    *electric,
                )
                return (*velocity, pressure, potential), observables

        step_function = b1_step or generic_step
        if design_parameters is not None:
            state = (u, v, w, p, phi)

            def advance_pipe(_, current):
                return step_function(current)[0]

            state = checkpointed_fori_loop(
                0,
                outer_steps - 1,
                advance_pipe,
                state,
                checkpoint_size=checkpoint_size,
            )
            state, observables = step_function(state)
            return (*state, *observables[:6])
        axial_pressure_loss_gradient = (
            jnp.asarray(initial_bundle.axial_pressure_loss_gradient, dtype=float)
            if initial_bundle is not None
            and initial_bundle.axial_pressure_loss_gradient is not None
            and initial_bundle.axial_pressure_loss_gradient.shape == (nx,)
            else jnp.full((nx,), forcing, dtype=float)
        )

        def pipe_checkpoint(iteration, terminal):
            return common._iteration_checkpoint_bundle(
                case=case,
                coordinates=(x, r, theta, field_scale),
                fields=(u, v, w, p, phi),
                axial_pressure_loss_gradient=axial_pressure_loss_gradient,
                transverse_pressure_difference=None,
                residual_history=residual_by_step,
                component_history=component_residual_by_step,
                pressure_history=pressure_residual_by_step,
                electric_history=electric_linear_by_step,
                potential_history=potential_residual_by_step,
                stopping_state=(
                    iteration,
                    0,
                    "converged" if terminal else "step_limit" if iteration == outer_steps else "in_progress",
                ),
            )

        for step in range(outer_steps):
            phi_previous = phi
            pressure_gradient_previous = axial_pressure_loss_gradient
            previous_velocity = u, v, w
            (u_next, v_next, w_next, p, phi), observables = step_function((u, v, w, p, phi))
            (
                jx,
                jr,
                jtheta,
                lorentz_x,
                lorentz_r,
                lorentz_theta,
                div_j,
                projected_divergence_norm,
                axial_pressure_loss_gradient,
                fixed_flow_error,
                electric_residual,
                electric_converged,
                electric_relative_residual,
                electric_iteration_count,
                electric_status,
                electric_local_residual,
            ) = observables
            potential_update = float(
                common._gauge_invariant_scalar_update(
                    phi, phi_previous, cell_area, scale=electric_potential_scale
                )
            )
            electric_linear_by_step.append(
                tuple(
                    map(
                        float,
                        (
                            electric_residual,
                            electric_relative_residual,
                            electric_local_residual,
                            electric_iteration_count,
                            electric_converged,
                            electric_status,
                        ),
                    )
                )
            )
            projected_divergence_max = float(projected_divergence_norm)
            updates = tuple(
                float(jnp.max(jnp.abs(current - previous)))
                for current, previous in zip((u_next, v_next, w_next), previous_velocity, strict=True)
            )
            flow_error_value = float(fixed_flow_error) if use_alex_b1_finite_volume else 0.0
            pressure_update = (
                common._normalized_pressure_observable_update(
                    axial_pressure_loss_gradient,
                    pressure_gradient_previous,
                    bx**2 + by**2 + bz**2,
                )
                if use_alex_b1_finite_volume
                else 0.0
            )
            update_residual = max(*updates, pressure_update, potential_update)
            charge_balance = float(jnp.max(jnp.abs(div_j)))
            components = (*updates, projected_divergence_max, flow_error_value, charge_balance)
            residual_by_step.append(update_residual)
            pressure_residual_by_step.append(pressure_update)
            potential_residual_by_step.append(potential_update)
            component_residual_by_step.append(components)
            converged = (
                update_residual <= case.solver.coupling_tolerance
                and projected_divergence_max <= common.ALEX_BALANCE_TOLERANCE
                and (not use_alex_b1_finite_volume or flow_error_value <= common.ALEX_BALANCE_TOLERANCE)
                and charge_balance <= common.ALEX_BALANCE_TOLERANCE
            )
            if use_alex_b1_finite_volume and not converged and step + 1 < outer_steps:
                current_state = jnp.stack((u, v, w, phi_previous)) / fixed_point_scale
                mapped_state = jnp.stack((u_next, v_next, w_next, phi)) / fixed_point_scale
                fixed_point_residual = mapped_state - current_state
                if case.solver.coupling_acceleration == "anderson":
                    fixed_point_iterates.append(current_state)
                    fixed_point_residuals.append(fixed_point_residual)
                    del fixed_point_iterates[: -case.solver.coupling_history_depth]
                    del fixed_point_residuals[: -case.solver.coupling_history_depth]
                    depth = case.solver.coupling_history_depth
                    accelerated = anderson_mixing(
                        jnp.stack(fixed_point_iterates[-depth:]),
                        jnp.stack(fixed_point_residuals[-depth:]),
                        regularization=case.solver.coupling_regularization,
                        damping=case.solver.coupling_damping,
                    )
                elif case.solver.coupling_acceleration == "aitken":
                    if previous_fixed_point_residual is not None:
                        fixed_point_relaxation = aitken_relaxation(
                            previous_fixed_point_residual,
                            fixed_point_residual,
                            fixed_point_relaxation,
                            min_relaxation=case.solver.coupling_min_relaxation,
                            max_relaxation=case.solver.coupling_max_relaxation,
                        )
                    accelerated = current_state + fixed_point_relaxation * fixed_point_residual
                    previous_fixed_point_residual = fixed_point_residual
                else:
                    accelerated = mapped_state
                u, v, w, phi = accelerated * fixed_point_scale
            else:
                u, v, w = u_next, v_next, w_next
            common._emit_iteration_progress(
                progress_callback,
                checkpoint_interval=checkpoint_interval,
                step=step + 1,
                total_steps=outer_steps,
                converged=converged,
                residual=update_residual,
                component_residuals=component_residual_by_step[-1],
                pressure_residual=pressure_update,
                potential_residual=potential_update,
                checkpoint_factory=partial(pipe_checkpoint, step + 1, converged),
            )
            if converged:
                break

        final_step_residual = residual_by_step[-1] if residual_by_step else 0.0
        residual = jnp.full((nx,), final_step_residual, dtype=float)
        cross_section_area = jnp.maximum(jnp.sum(fluid_cell_area, axis=(1, 2)), 1.0e-20)
        volumetric_flow_rate = jnp.sum(u * fluid_cell_area, axis=(1, 2))
        mean_velocity = volumetric_flow_rate / cross_section_area
        axial_current = jnp.sum(jx * cell_area, axis=(1, 2))
        uxb_x, uxb_r, uxb_theta = common._cross((u, v, w), (bx, br, btheta))
        final_div_j, wall_current_leakage, boundary_current_residual = (
            pipe._pipe_conservative_current_diagnostics_3d(
                sigma,
                phi,
                uxb_x,
                uxb_r,
                uxb_theta,
                dx=dx,
                r_faces=r_faces,
                r_centers=r,
                dtheta=dtheta,
            )
        )
        current_scaled_pressure_proxy = jnp.max(jnp.abs(jr), axis=(1, 2)) * jnp.maximum(
            jnp.max(jnp.abs(bx) + jnp.abs(br) + jnp.abs(btheta), axis=(1, 2)),
            1.0e-12,
        )
        charge_balance_residual = jnp.max(jnp.abs(final_div_j), axis=(1, 2))
        return ExtrudedFieldBundle.from_groups(
            (x, r, theta, field_scale),
            (u, v, w, p, phi),
            (jx, jr, jtheta),
            (lorentz_x, lorentz_r, lorentz_theta),
            (
                residual,
                volumetric_flow_rate,
                mean_velocity,
                axial_current,
                wall_current_leakage,
                current_scaled_pressure_proxy,
                charge_balance_residual,
                boundary_current_residual,
                axial_pressure_loss_gradient,
                jnp.zeros((nx,), dtype=float),
            ),
            geometry_kind=case.geometry.kind,
            solver_kind=case.solver.kind,
            stopping_state=(
                len(residual_by_step),
                0,
                "converged" if converged else "step_limit",
            ),
            **common._iteration_history_arrays(
                residual_by_step,
                component_residual_by_step,
                pressure_residual_by_step,
                electric_linear_by_step,
                potential_residual_by_step,
                stride=case.output.history_stride,
            ),
        )


def _solve_duct_projection(
    problem: ExtrudedInductionlessProblem,
    mesh,
    use_alex_b2_finite_volume: bool,
    runtime,
) -> ExtrudedFieldBundle | tuple[jnp.ndarray, ...]:
    (
        initial_bundle,
        num_devices,
        progress_callback,
        phase_timing_callback,
        checkpoint_interval,
        design_parameters,
    ) = runtime
    case = problem.case
    with jax.ensure_compile_time_eval():
        materials = build_material_fields(case, mesh)
    if design_parameters is None:
        forcing, magnetic_scale, conductivity_scale, geometry_scale = case.forcing, 1.0, 1.0, 1.0
    else:
        forcing, magnetic_scale, conductivity_scale, geometry_scale, outer_steps, checkpoint_size = (
            design_parameters
        )
    axial_scale, transverse_y_scale, transverse_z_scale = common._coordinate_scale(geometry_scale)
    x = axial_scale * jnp.asarray(mesh.x_centers, dtype=float)
    y = transverse_y_scale * jnp.asarray(mesh.y_centers, dtype=float)
    z = transverse_z_scale * jnp.asarray(mesh.z_centers, dtype=float)
    nx, ny, nz = len(x), len(y), len(z)
    dy = transverse_y_scale * jnp.asarray(mesh.dy, dtype=float)
    dz = transverse_z_scale * jnp.asarray(mesh.dz, dtype=float)
    walls = case.geometry.wall_thickness
    base_dx = case.geometry.length / nx
    dx = axial_scale * base_dx
    dy_momentum = transverse_y_scale * (case.geometry.width + walls[0] + walls[1]) / ny
    dz_momentum = transverse_z_scale * (case.geometry.height + walls[2] + walls[3]) / nz
    sigma = common._broadcast_cross_section(materials.conductivity, nx)
    rho = common._broadcast_cross_section(materials.density, nx)
    nu = common._broadcast_cross_section(materials.viscosity, nx)
    with jax.ensure_compile_time_eval():
        fluid_mask = common._broadcast_cross_section(materials.fluid_mask.astype(float), nx) > 0.5
        fluid_bounds = (
            common._rectangular_fluid_bounds(fluid_mask)
            if use_alex_b2_finite_volume
            or (design_parameters is not None and case.geometry.kind == "layered_duct")
            else None
        )
    if use_alex_b2_finite_volume:
        y0, y1, z0, z1 = fluid_bounds
        dy = common._canonical_shell_widths(dy, y0, y1)
        dz = common._canonical_shell_widths(dz, z0, z1)
        wall = next(region for region in case.regions if region.kind == "solid")
        sheet_conductance = wall.conductivity * wall.wall_thickness
        sigma = jnp.where(
            fluid_mask,
            sigma,
            sheet_conductance / common.ALEX_B2_CANONICAL_SHELL_THICKNESS,
        )
    cell_area = common._broadcast_cross_section(dy[:, None] * dz[None, :], nx)
    with jax.ensure_compile_time_eval():
        field_scale = jnp.asarray(problem.profile.field_scale, dtype=float)
        field_y, field_z = jnp.meshgrid(mesh.y_centers, mesh.z_centers, indexing="ij")
        base_field = _sample_station_magnetic_field(
            case,
            field_scale=field_scale,
            x=mesh.x_centers,
            y=field_y,
            z=field_z,
            volume_field=problem.profile.volume_field,
        )
    (bx, by, bz), sigma = common._scale_projection_properties(
        base_field, sigma, fluid_mask, magnetic_scale, conductivity_scale
    )
    u, v, w, p, phi = common._initial_projection_fields(case, fluid_mask, initial_bundle)

    (
        u,
        v,
        w,
        p,
        phi,
        sigma,
        rho,
        nu,
        fluid_mask,
        cell_area,
        bx,
        by,
        bz,
    ) = common._shard_extruded_fields(
        (
            u,
            v,
            w,
            p,
            phi,
            sigma,
            rho,
            nu,
            fluid_mask,
            cell_area,
            bx,
            by,
            bz,
        ),
        num_devices=num_devices,
    )

    stability_field = (bx, by, bz) if design_parameters is None else base_field
    with jax.ensure_compile_time_eval():
        static_sigma = common._broadcast_cross_section(materials.conductivity, nx)
        static_rho = common._broadcast_cross_section(materials.density, nx)
        static_nu = common._broadcast_cross_section(materials.viscosity, nx)
        static_mask = common._broadcast_cross_section(materials.fluid_mask.astype(float), nx) > 0.5
        inverse_diffusive_scale = float(
            jnp.max(static_nu)
            * (
                1.0 / max(base_dx**2, 1.0e-12)
                + 1.0 / max(float(jnp.min(mesh.dy)) ** 2, 1.0e-12)
                + 1.0 / max(float(jnp.min(mesh.dz)) ** 2, 1.0e-12)
            )
        )
        inverse_electromagnetic_scale = float(
            jnp.max(
                jnp.where(
                    static_mask,
                    static_sigma * sum(component**2 for component in stability_field) / static_rho,
                    0.0,
                )
            )
        )
    stability_safety = (
        common.ALEX_B2_MAGNETIC_STABILITY_SAFETY
        if use_alex_b2_finite_volume
        else (0.01 if float(case.geometry.target_ha or 0.0) >= 100.0 else 0.1)
    )
    stable_dt = stability_safety / max(
        inverse_electromagnetic_scale
        if use_alex_b2_finite_volume
        else inverse_diffusive_scale + inverse_electromagnetic_scale,
        1.0e-12,
    )
    dt = min(float(case.time_stepper.dt), stable_dt)
    if use_alex_b2_finite_volume:
        inlet = [bc for bc in case.boundary_conditions if bc.kind == "inlet_flow_rate"]
        outlet = [bc for bc in case.boundary_conditions if bc.kind == "outlet_pressure"]
        if (
            len(inlet) != 1
            or len(outlet) != 1
            or not isinstance(inlet[0].value, (int, float))
            or outlet[0].value != 0.0
        ):
            raise ValueError("ALEX B2 requires one inlet flow rate and zero outlet pressure")
        target_flow_rate = float(inlet[0].value)
        reference_area = float(jnp.mean(jnp.sum(jnp.where(fluid_mask, cell_area, 0.0), axis=(1, 2))))
        fixed_point_velocity_scale = target_flow_rate / reference_area
    else:
        with jax.ensure_compile_time_eval():
            target_flow_rate = (
                float(jnp.mean(jnp.sum(jnp.where(fluid_mask, u * cell_area, 0.0), axis=(1, 2))))
                if initial_bundle is not None or case.initial_velocity != 0.0
                else None
            )
        fixed_point_velocity_scale = 1.0
    if design_parameters is None:
        outer_steps = min(case.time_stepper.max_steps, max(6, case.solver.coupling_iterations * 2))
    poisson_iterations = (
        case.time_stepper.potential_iterations
        if use_alex_b2_finite_volume
        else min(case.time_stepper.potential_iterations, 80)
    )
    poisson_tolerance = case.solver.coupling_tolerance
    electric_iterations = max(poisson_iterations, 600)
    # Generic traced fields need a roundoff-level primal for their implicit VJP.
    # The primal-only B2 path instead certifies its stricter-than-physics linear solve directly.
    electric_tolerance = min(
        poisson_tolerance,
        1.0e-10 if use_alex_b2_finite_volume else 8.0 * np.finfo(np.float64).eps,
    )
    projection_iterations = max(poisson_iterations, 4000)
    projection_tolerance = min(poisson_tolerance, 1.0e-12)
    momentum_iterations = max(poisson_iterations, 400)
    momentum_tolerance = min(poisson_tolerance, 1.0e-10)
    velocity_limit = max(5.0, 2.0 * math.sqrt(float(case.geometry.target_ha or 1.0)))
    with jax.ensure_compile_time_eval():
        stability_energy = sum(component**2 for component in stability_field)
        scalar_limit = max(20.0, 2.0 * float(jnp.max(stability_energy)))
        electric_potential_scale = max(1.0, math.sqrt(float(jnp.max(stability_energy))))
    generic_step = None
    if not use_alex_b2_finite_volume:
        unit_response = common._enforce_velocity_bc_3d(jnp.where(fluid_mask, dt / rho, 0.0), fluid_mask)

        def generic_step(current):
            return common._generic_duct_step(
                current,
                material=(sigma, rho, nu, fluid_mask, cell_area),
                field=(bx, by, bz),
                forcing=forcing,
                spacing=(dt, dx, dy, dz, dy_momentum, dz_momentum),
                solves=(
                    poisson_iterations,
                    electric_iterations,
                    electric_tolerance,
                ),
                limits=(velocity_limit, scalar_limit),
                flow=(
                    target_flow_rate,
                    unit_response,
                    0.6 if case.geometry.kind == "layered_duct" else 0.0,
                ),
                operators=(
                    partial(duct._solvax_pressure_poisson_duct, transverse_coarse_bounds=fluid_bounds)
                    if fluid_bounds is not None
                    else duct._solvax_pressure_poisson_duct,
                    duct._conservative_emf_rhs_3d,
                    duct._conservative_current_diagnostics_3d,
                ),
            )

    if design_parameters is not None:
        assert generic_step is not None

        def advance(_, current):
            return generic_step(current)[0]

        state = (u, v, w, p, phi)
        state = checkpointed_fori_loop(
            0,
            outer_steps - 1,
            advance,
            state,
            checkpoint_size=checkpoint_size,
        )
        state, observables = generic_step(state)
        return (*state, *observables[:6])
    (
        residual_by_step,
        component_residual_by_step,
        pressure_residual_by_step,
        electric_linear_by_step,
        potential_residual_by_step,
        completed_steps,
        momentum_defect_by_step,
        pressure_linear_by_step,
        courant_by_step,
        previous_fixed_point_residual,
        fixed_aitken_relaxation,
        steady_streak,
        fixed_point_relaxation,
        previous_anderson_mapped,
        previous_anderson_residual,
        previous_anderson_flux,
        previous_anderson_inlet,
        fixed_point_scale,
        axial_pressure_loss_gradient,
    ) = common._restore_duct_iteration_state(
        initial_bundle,
        case=case,
        use_b2=use_alex_b2_finite_volume,
        velocity=u,
        velocity_scale=fixed_point_velocity_scale,
        potential_scale=electric_potential_scale,
        forcing=forcing,
    )
    retained_history_steps = len(residual_by_step)
    if use_alex_b2_finite_volume:
        (
            initialize_flux,
            momentum_solve,
            momentum_defect,
            embed_velocity,
            courant_numbers,
            pack_flux,
            unpack_flux,
            pack_vector,
            relax_flux,
            field_sharding,
            replicated_sharding,
            flux_sharding,
            kernel_key,
            current_rho_phi_inlet,
            current_rho_phi_plus,
            previous_anderson_flux,
            previous_anderson_inlet,
        ) = duct._prepare_b2_momentum_runtime(
            case=case,
            velocity=(u, v, w),
            density=rho,
            fluid_bounds=fluid_bounds,
            num_devices=num_devices,
            dy=dy,
            dz=dz,
            dt=dt,
            dx=dx,
            momentum_iterations=momentum_iterations,
            momentum_tolerance=momentum_tolerance,
            projection_iterations=projection_iterations,
            projection_tolerance=projection_tolerance,
            electric_iterations=electric_iterations,
            electric_tolerance=electric_tolerance,
            forcing=forcing,
            target_flow_rate=target_flow_rate,
            initial=initial_bundle,
            previous_anderson_flux=previous_anderson_flux,
            previous_anderson_inlet=previous_anderson_inlet,
        )
    if use_alex_b2_finite_volume:
        # The strict corner bound oversolves PCG; the local residual gates physics.
        electric_volume_min = 4.0 * float(jnp.min(dy) * jnp.min(dz))
        (
            mixed_boundary_projection,
            electric_solve,
            emf_operator,
            reconstruct_electric,
            lorentz_operator,
            scaled_state,
            state_difference,
            unscaled_state,
            mix_anderson,
        ) = duct._jit_b2_coupling_functions(
            duct._b2_coupling_functions(
                case=case,
                target_flow_rate=target_flow_rate,
                metric=(dt, dx, dy, dz),
                projection_iterations=projection_iterations,
                projection_tolerance=projection_tolerance,
                electric_iterations=electric_iterations,
                electric_tolerance=electric_tolerance,
                electric_volume_min=electric_volume_min,
                fluid_bounds=fluid_bounds,
                field_sharding=field_sharding,
                fixed_point_scale=fixed_point_scale,
            ),
            field_sharding=field_sharding,
            replicated_sharding=replicated_sharding,
            flux_sharding=flux_sharding,
            kernel_key=kernel_key,
        )
        if field_sharding is not None:  # pragma: no cover - hardware gate
            state_sharding = NamedSharding(field_sharding.mesh, P(None, "x", None, None))
            if previous_fixed_point_residual is not None:
                previous_fixed_point_residual = jax.device_put(
                    np.asarray(previous_fixed_point_residual), state_sharding
                )
            if previous_anderson_mapped is not None:
                previous_anderson_mapped = jax.device_put(
                    np.asarray(previous_anderson_mapped), state_sharding
                )
                previous_anderson_residual = jax.device_put(
                    np.asarray(previous_anderson_residual), state_sharding
                )
        momentum_solve = common._synchronized_phase(momentum_solve, "momentum", phase_timing_callback)
        momentum_defect = common._synchronized_phase(
            momentum_defect, "momentum_defect", phase_timing_callback
        )
        mixed_boundary_projection = common._synchronized_phase(
            mixed_boundary_projection, "projection", phase_timing_callback
        )
        electric_solve = common._synchronized_phase(electric_solve, "electric", phase_timing_callback)
        emf_operator = common._synchronized_phase(emf_operator, "emf", phase_timing_callback)
        reconstruct_electric = common._synchronized_phase(
            reconstruct_electric, "reconstruction", phase_timing_callback
        )
        mix_anderson = common._synchronized_phase(mix_anderson, "anderson", phase_timing_callback)

    stop_step = completed_steps + outer_steps

    def accelerator_state():
        return {
            "rho_phi_plus": current_rho_phi_plus if use_alex_b2_finite_volume else None,
            "rho_phi_inlet": current_rho_phi_inlet if use_alex_b2_finite_volume else None,
            "aitken_state": (
                (previous_fixed_point_residual, fixed_point_relaxation, steady_streak)
                if use_alex_b2_finite_volume and case.solver.coupling_acceleration == "aitken"
                else None
            ),
            "anderson_state": (
                (
                    previous_anderson_mapped,
                    previous_anderson_residual,
                    previous_anderson_flux,
                    previous_anderson_inlet,
                )
                if use_alex_b2_finite_volume and case.solver.coupling_acceleration == "anderson"
                else None
            ),
        }

    if use_alex_b2_finite_volume:

        def b2_map(state, flux):
            """Apply one complete conservative B2 fixed-point map."""
            u0, v0, w0, p0, phi0 = state
            rho_phi_plus, rho_phi_inlet = flux
            emf = common._cross((u0, v0, w0), (bx, by, bz))
            _, _, _, lorentz_x0, lorentz_y0, lorentz_z0 = lorentz_operator(phi0, sigma, *emf, bx, by, bz)
            momentum_force = pack_vector(lorentz_x0 + forcing, lorentz_y0, lorentz_z0)
            u1, v1, w1, pressure = u0, v0, w0, p0
            mapped_flux, mapped_inlet = rho_phi_plus, rho_phi_inlet
            for _ in range(common.ALEX_B2_PRESSURE_CORRECTORS):
                velocity_fluid, _, momentum_converged, momentum_mobility = momentum_solve(
                    pack_vector(u1, v1, w1),
                    momentum_force,
                    rho,
                    nu,
                    mapped_flux,
                    mapped_inlet,
                    pressure,
                )
                predicted = embed_velocity(velocity_fluid, fluid_mask)
                projection = mixed_boundary_projection(
                    *predicted, pressure, rho, fluid_mask, momentum_mobility
                )
                u1, v1, w1, pressure = projection[:4]
                mapped_flux, mapped_inlet = pack_flux(*projection[7:10]), projection[10]
            emf = common._cross((u1, v1, w1), (bx, by, bz))
            electric = electric_solve(emf_operator(sigma, *emf, fluid_mask), phi0, sigma, fluid_mask)
            potential = electric[0]
            current = reconstruct_electric(potential, sigma, *emf, bx, by, bz, fluid_mask)
            jx0, jy0, jz0, div_j0, lorentz_x1, lorentz_y1, lorentz_z1 = current
            defect = momentum_defect(
                pack_vector(u1, v1, w1),
                pack_vector(lorentz_x1, lorentz_y1, lorentz_z1),
                rho,
                nu,
                mapped_flux,
                mapped_inlet,
                pressure,
            )
            return (
                (u1, v1, w1, jnp.where(fluid_mask, pressure, 0.0), potential),
                (mapped_flux, mapped_inlet),
                (
                    (jx0, jy0, jz0, div_j0, lorentz_x1, lorentz_y1, lorentz_z1),
                    (projection[4], projection[5], projection[6]),
                    defect,
                    momentum_converged,
                    projection[11:],
                    electric[1:],
                ),
            )

        b2_map = jax.named_call(b2_map, name="lmx.b2.map")

    for step in range(completed_steps, stop_step):
        flux_relaxation = jnp.asarray(1.0, dtype=u.dtype)
        step_courant = (
            courant_numbers(*unpack_flux(current_rho_phi_plus), current_rho_phi_inlet, rho)
            if use_alex_b2_finite_volume
            else (-1.0, -1.0)
        )
        phi_previous = phi
        pressure_observable_previous = (
            common._cross_duct_pressure_difference(p, active_mask=fluid_mask, magnetic_axis=1, side_axis=2)
            if use_alex_b2_finite_volume
            else jnp.zeros((nx,), dtype=p.dtype)
        )
        pressure_linear = jnp.asarray((jnp.nan, jnp.nan, 0.0, 0.0, -1.0))
        momentum_linear_converged = jnp.asarray(True)
        if not use_alex_b2_finite_volume:
            assert generic_step is not None
            (u_next, v_next, w_next, p, phi), generic = generic_step((u, v, w, p, phi))
            (
                jx,
                jy,
                jz,
                lorentz_x,
                lorentz_y,
                lorentz_z,
                div_j,
                projected_divergence_max,
                axial_pressure_loss_gradient,
                *electric_linear,
            ) = generic
            fixed_flow_error = jnp.asarray(0.0)
            momentum_defect_components = jnp.full((4,), jnp.nan)
        else:
            (u_next, v_next, w_next, p, phi), (mapped_rho_phi_plus, mapped_rho_phi_inlet), b2 = b2_map(
                (u, v, w, p, phi), (current_rho_phi_plus, current_rho_phi_inlet)
            )
            (
                (jx, jy, jz, div_j, lorentz_x, lorentz_y, lorentz_z),
                (axial_pressure_loss_gradient, projected_divergence_norm, fixed_flow_error),
                momentum_defect_components,
                momentum_linear_converged,
                pressure_linear,
                electric_linear,
            ) = b2
            mapped_flux_components = unpack_flux(mapped_rho_phi_plus)
            valid = all(bool(jnp.all(jnp.isfinite(field))) for field in (u_next, v_next, w_next, p, phi))
            valid &= all(
                bool(jnp.all(jnp.abs(field) <= velocity_limit)) for field in (u_next, v_next, w_next)
            )
            if not valid:
                state = tuple(
                    (name, bool(jnp.all(jnp.isfinite(field))), float(jnp.nanmax(jnp.abs(field))))
                    for name, field in zip(("u", "v", "w", "p"), (u_next, v_next, w_next, p))
                )
                raise FloatingPointError(f"ALEX B2 projection inactive guard: {state}")

        potential_update = common._gauge_invariant_scalar_update(
            phi,
            phi_previous,
            cell_area,
            scale=electric_potential_scale,
        )

        if use_alex_b2_finite_volume:
            projected_divergence_max = projected_divergence_norm

        pressure_update = (
            common._normalized_pressure_observable_update(
                common._cross_duct_pressure_difference(
                    p,
                    active_mask=fluid_mask,
                    magnetic_axis=1,
                    side_axis=2,
                ),
                pressure_observable_previous,
                bx**2 + by**2 + bz**2,
            )
            if use_alex_b2_finite_volume
            else jnp.asarray(0.0)
        )
        diagnostics = np.asarray(
            jnp.stack(
                (
                    jnp.max(jnp.abs(u_next - u)),
                    jnp.max(jnp.abs(v_next - v)),
                    jnp.max(jnp.abs(w_next - w)),
                    projected_divergence_max,
                    fixed_flow_error,
                    jnp.max(jnp.abs(div_j)),
                    momentum_defect_components[-1],
                    pressure_update,
                    potential_update,
                    *pressure_linear,
                    electric_linear[0],
                    electric_linear[2],
                    electric_linear[5],
                    electric_linear[3],
                    electric_linear[1],
                    electric_linear[4],
                )
            )
        )
        (
            u_update,
            v_update,
            w_update,
            projected_divergence_max,
            flow_error_value,
            charge_balance,
            momentum_defect_value,
            pressure_update,
            potential_update,
            *linear_diagnostics,
        ) = map(float, diagnostics)
        pressure_linear_by_step.append(tuple(linear_diagnostics[:5]))
        electric_diagnostics = linear_diagnostics[5:]
        electric_linear_by_step.append(tuple(electric_diagnostics))
        courant_by_step.append((dt if use_alex_b2_finite_volume else -1.0, *map(float, step_courant)))
        update_residual = max(u_update, v_update, w_update, pressure_update, potential_update)
        stopping_residual = (
            max(u_update, v_update, w_update)
            / (inverse_electromagnetic_scale * dt * common.ALEX_B2_PRESSURE_CORRECTORS)
            if use_alex_b2_finite_volume
            else update_residual
        )
        residual_by_step.append(update_residual)
        pressure_residual_by_step.append(pressure_update)
        potential_residual_by_step.append(potential_update)
        if use_alex_b2_finite_volume:
            momentum_defect_by_step.append(momentum_defect_value)
        component_residual_by_step.append(
            (
                u_update,
                v_update,
                w_update,
                projected_divergence_max,
                flow_error_value,
                charge_balance,
            )
        )
        instantaneous_convergence = (
            stopping_residual <= case.solver.coupling_tolerance
            and projected_divergence_max <= common.ALEX_BALANCE_TOLERANCE
            and flow_error_value <= common.ALEX_BALANCE_TOLERANCE
            and charge_balance <= common.ALEX_BALANCE_TOLERANCE
            and (
                not use_alex_b2_finite_volume
                or all(map(bool, (momentum_linear_converged, pressure_linear[3], electric_linear[1])))
            )
        )
        accepted_state_converged = (
            max(u_update, v_update, w_update, potential_update) <= case.time_stepper.steady_tolerance
        )
        if use_alex_b2_finite_volume:
            # Require repeated passing updates before accepting an oscillatory map.
            steady_streak = steady_streak + 1 if instantaneous_convergence else 0
            converged = steady_streak >= common.ALEX_B2_STEADY_STEPS
        else:
            converged = instantaneous_convergence
        if use_alex_b2_finite_volume and not converged:
            current_state = scaled_state(u, v, w, phi_previous)
            mapped_state = scaled_state(u_next, v_next, w_next, phi)
            fixed_point_residual = state_difference(mapped_state, current_state)
            if case.solver.coupling_acceleration == "aitken":
                if fixed_aitken_relaxation is not None:
                    relaxation_value = 1.0 if step == 0 else fixed_aitken_relaxation
                    fixed_point_relaxation = jnp.asarray(relaxation_value, dtype=u.dtype)
                    accelerated = (
                        mapped_state if step == 0 else current_state + relaxation_value * fixed_point_residual
                    )
                    flux_relaxation = relaxation_value
                elif accepted_state_converged:
                    # Avoid reduction noise after settling while retaining a
                    # conservative, empirically monotone coupled acceleration.
                    accelerated = current_state + common.ALEX_B2_SETTLED_RELAXATION * fixed_point_residual
                    previous_fixed_point_residual = None
                    fixed_point_relaxation = jnp.asarray(1.0, dtype=u.dtype)
                    flux_relaxation = common.ALEX_B2_SETTLED_RELAXATION
                elif previous_fixed_point_residual is not None:
                    fixed_point_relaxation = aitken_relaxation(
                        previous_fixed_point_residual,
                        fixed_point_residual,
                        fixed_point_relaxation,
                        min_relaxation=case.solver.coupling_min_relaxation,
                        max_relaxation=case.solver.coupling_max_relaxation,
                    )
                    accelerated = current_state + fixed_point_relaxation * fixed_point_residual
                    flux_relaxation = fixed_point_relaxation
                else:
                    accelerated = mapped_state
                if not accepted_state_converged and fixed_aitken_relaxation is None:
                    previous_fixed_point_residual = fixed_point_residual
            else:
                mapped_flux = pack_flux(*mapped_flux_components)
                if previous_anderson_mapped is None:
                    accelerated = mapped_state
                    accelerated_flux = mapped_flux
                    accelerated_inlet = mapped_rho_phi_inlet
                else:
                    accelerated, accelerated_flux, accelerated_inlet = mix_anderson(
                        previous_anderson_mapped,
                        previous_anderson_residual,
                        previous_anderson_flux,
                        previous_anderson_inlet,
                        mapped_state,
                        fixed_point_residual,
                        mapped_flux,
                        mapped_rho_phi_inlet,
                    )
                # Keep only the latest raw map: schema 6 is exactly depth two.
                previous_anderson_mapped = mapped_state
                previous_anderson_residual = fixed_point_residual
                previous_anderson_flux = mapped_flux
                previous_anderson_inlet = mapped_rho_phi_inlet
            u, v, w, phi = unscaled_state(accelerated)
        else:
            u, v, w = u_next, v_next, w_next
        if use_alex_b2_finite_volume:
            if case.solver.coupling_acceleration == "anderson":
                current_rho_phi_plus = pack_flux(*mapped_flux_components) if converged else accelerated_flux
                current_rho_phi_inlet = mapped_rho_phi_inlet if converged else accelerated_inlet
            else:
                current_rho_phi_plus, current_rho_phi_inlet = relax_flux(
                    current_rho_phi_plus,
                    current_rho_phi_inlet,
                    pack_flux(*mapped_flux_components),
                    mapped_rho_phi_inlet,
                    flux_relaxation,
                )
        common._emit_iteration_progress(
            progress_callback,
            checkpoint_interval=checkpoint_interval,
            step=step + 1,
            total_steps=stop_step,
            converged=converged,
            residual=update_residual,
            component_residuals=component_residual_by_step[-1],
            pressure_residual=pressure_update,
            potential_residual=potential_update,
            checkpoint_factory=lambda: common._iteration_checkpoint_bundle(
                case=case,
                coordinates=(x, y, z, field_scale),
                fields=(u, v, w, p, phi),
                axial_pressure_loss_gradient=axial_pressure_loss_gradient,
                transverse_pressure_difference=None,
                residual_history=residual_by_step,
                component_history=component_residual_by_step,
                pressure_history=pressure_residual_by_step,
                electric_history=electric_linear_by_step,
                potential_history=potential_residual_by_step,
                pressure_linear_history=pressure_linear_by_step,
                **accelerator_state(),
                stopping_state=(
                    step + 1,
                    steady_streak,
                    "converged" if converged else "step_limit" if step + 1 == stop_step else "in_progress",
                ),
                courant_history=courant_by_step,
                momentum_defect_history=(momentum_defect_by_step if use_alex_b2_finite_volume else None),
            ),
        )
        if converged:
            break

    # Acceleration changes the accepted state after current reconstruction.
    uxb_x, uxb_y, uxb_z = common._cross((u, v, w), (bx, by, bz))
    if use_alex_b2_finite_volume:
        jx, jy, jz, div_j, lorentz_x, lorentz_y, lorentz_z = reconstruct_electric(
            phi, sigma, uxb_x, uxb_y, uxb_z, bx, by, bz, fluid_mask
        )

    final_step_residual = residual_by_step[-1] if residual_by_step else 0.0
    residual = jnp.full((nx,), final_step_residual, dtype=float)
    fluid_area = jnp.maximum(jnp.sum(jnp.where(fluid_mask, cell_area, 0.0), axis=(1, 2)), 1.0e-20)
    volumetric_flow_rate = jnp.sum(jnp.where(fluid_mask, u * cell_area, 0.0), axis=(1, 2))
    mean_velocity = volumetric_flow_rate / fluid_area
    fx, _, _ = duct._conservative_current_fluxes_3d(
        sigma,
        phi,
        uxb_x,
        uxb_y,
        uxb_z,
        dx=dx,
        dy=dy,
        dz=dz,
        thin_wall_fluid_mask=fluid_mask if use_alex_b2_finite_volume else None,
    )
    axial_current = duct._station_axial_current_from_fluxes(fx, cell_area[0])
    if use_alex_b2_finite_volume:
        wall_current_leakage = jnp.zeros((nx,), dtype=div_j.dtype)
        boundary_current_residual = jnp.abs(jnp.sum(div_j * cell_area, axis=(1, 2)) * dx)
    else:
        div_j, wall_current_leakage, boundary_current_residual = duct._conservative_current_diagnostics_3d(
            sigma,
            phi,
            uxb_x,
            uxb_y,
            uxb_z,
            dx=dx,
            dy=dy,
            dz=dz,
        )
    current_scaled_pressure_proxy = jnp.max(jnp.abs(jy), axis=(1, 2)) * jnp.maximum(
        jnp.max(jnp.abs(bx) + jnp.abs(by) + jnp.abs(bz), axis=(1, 2)), 1.0e-12
    )
    charge_balance_residual = jnp.max(jnp.abs(div_j), axis=(1, 2))
    transverse_pressure_difference = common._cross_duct_pressure_difference(
        p, active_mask=fluid_mask, magnetic_axis=1, side_axis=2
    )
    return ExtrudedFieldBundle.from_groups(
        (x, y, z, field_scale),
        (u, v, w, p, phi),
        (jx, jy, jz),
        (lorentz_x, lorentz_y, lorentz_z),
        (
            residual,
            volumetric_flow_rate,
            mean_velocity,
            axial_current,
            wall_current_leakage,
            current_scaled_pressure_proxy,
            charge_balance_residual,
            boundary_current_residual,
            axial_pressure_loss_gradient,
            transverse_pressure_difference,
        ),
        geometry_kind=case.geometry.kind,
        solver_kind=case.solver.kind,
        **accelerator_state(),
        stopping_state=(step + 1, steady_streak, "converged" if converged else "step_limit"),
        **common._iteration_history_arrays(
            residual_by_step,
            component_residual_by_step,
            pressure_residual_by_step,
            electric_linear_by_step,
            potential_residual_by_step,
            courant_by_step,
            pressure_linear=pressure_linear_by_step,
            momentum_defect=(momentum_defect_by_step if use_alex_b2_finite_volume else None),
            stride=case.output.history_stride,
            retained_prefix=retained_history_steps,
        ),
    )


def solve_extruded_inductionless(
    problem: ExtrudedInductionlessProblem,
    *,
    initial_bundle: ExtrudedFieldBundle | None = None,
    num_devices: int | None = None,
    progress_callback: Callable[[ExtrudedIterationProgress], None] | None = None,
    phase_timing_callback: Callable[[str, float], None] | None = None,
    checkpoint_interval: int | None = None,
) -> ExtrudedInductionlessSolution:
    """Solve an extruded problem with optional sharding and progress checkpoints.

    ``progress_callback`` is called after every outer iteration. Its progress
    object contains a restart-capable bundle at ``checkpoint_interval`` steps
    and on convergence; no checkpoint arrays are materialized otherwise.

    ``phase_timing_callback`` is a diagnostic hook that inserts completion
    barriers around B2 solver phases and reports ``(name, wall_seconds)``.
    Leave it unset for ordinary asynchronous execution and scaling timings.
    """

    if problem.case.output.history_stride < 0:
        raise ValueError("history_stride must be non-negative")
    if checkpoint_interval is not None and checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be positive")

    projection_kwargs = {
        "initial_bundle": initial_bundle,
        "progress_callback": progress_callback,
        "phase_timing_callback": phase_timing_callback,
        "checkpoint_interval": checkpoint_interval,
    }
    if num_devices is not None:
        projection_kwargs["num_devices"] = num_devices
    bundle = _solve_extruded_projection(problem, **projection_kwargs)
    require_finite(
        "3-D fringing solve",
        **{name: getattr(bundle, name) for name in _EXTRUDED_NUMERICAL_RESULTS},
    )
    station_history = _bundle_station_history(bundle)
    validation = validate_extruded_inductionless_solution(bundle, station_history=station_history)
    return ExtrudedInductionlessSolution(
        problem=problem,
        bundle=bundle,
        station_history=tuple(station_history),
        validation=validation,
    )


def evolve_extruded_fields(
    problem: ExtrudedInductionlessProblem,
    *,
    forcing: float | jnp.ndarray | None = None,
    magnetic_field_scale: float | jnp.ndarray = 1.0,
    material_conductivity_scale: float | jnp.ndarray = 1.0,
    geometry_scale: float | jnp.ndarray = 1.0,
    steps: int | None = None,
    checkpoint_size: int | None = None,
) -> tuple[jnp.ndarray, ...]:
    """Return differentiable 3-D duct or straight-pipe production fields.

    Returns velocity, pressure, potential, current, and Lorentz-force fields.
    Pressure forcing, imposed-field scale, and material conductivity are
    continuous. Field scale may contain one coefficient per axial station;
    conductivity scale may be scalar or ``(fluid, solid)``. Geometry scale may
    be scalar, ``(axial, transverse_y, transverse_z)`` for a duct, or
    ``(axial, radial)`` for a pipe. It maps the fixed reference mesh without
    changing topology or imposed-field samples; callers keep scale factors
    positive. Step controls are static. SOLVAX supplies implicit elliptic VJPs
    and exact checkpointing. ALEX B1 uses its production finite-volume map;
    specialized ALEX B2 fields are not yet exposed here.
    """

    steps = (
        min(problem.case.time_stepper.max_steps, max(6, problem.case.solver.coupling_iterations * 2))
        if steps is None
        else steps
    )
    if steps < 1:
        raise ValueError("steps must be positive")
    if checkpoint_size is not None and checkpoint_size < 1:
        raise ValueError("checkpoint_size must be positive")
    source = problem.case.forcing if forcing is None else forcing
    return _solve_extruded_projection(
        problem,
        design_parameters=(
            jnp.asarray(source),
            jnp.asarray(magnetic_field_scale),
            jnp.asarray(material_conductivity_scale),
            jnp.asarray(geometry_scale),
            steps,
            checkpoint_size,
        ),
    )


def extruded_engineering_objectives(
    problem: ExtrudedInductionlessProblem,
    fields: tuple[jnp.ndarray, ...],
    *,
    geometry_scale: float | jnp.ndarray = 1.0,
    smoothing: float = 1.0e-8,
) -> dict[str, jnp.ndarray]:
    """Reduce differentiable 3-D fields to scalar design objectives.

    Values retain the units of ``problem``. Pass the same fixed-topology
    ``geometry_scale`` used to evolve the fields. Lower is better except for
    flow rate; wall current is a cell-centered design proxy, not a validation
    flux.
    """
    geometry_kind = problem.case.geometry.kind
    if geometry_kind not in {"rect_duct", "layered_duct", "pipe_ogrid"}:
        raise NotImplementedError("engineering objectives require a generic duct or straight pipe")
    if len(fields) < 8:
        raise ValueError("fields must contain velocity, pressure, potential, and current")
    if smoothing <= 0.0:
        raise ValueError("smoothing must be positive")
    u, _, _, pressure, _, jx, jy, jz = fields[:8]
    with jax.ensure_compile_time_eval():
        mesh = _cross_section_mesh(problem.case)
        fluid = np.asarray(build_material_fields(problem.case, mesh).fluid_mask)
        area = (
            np.asarray(mesh.y_centers)[:, None] * np.asarray(mesh.dy)[:, None] * float(np.mean(mesh.dz))
            if geometry_kind == "pipe_ogrid"
            else np.asarray(mesh.dy)[:, None] * np.asarray(mesh.dz)[None, :]
        )
    expected_shape = (len(mesh.x_centers), *mesh.yz_shape)
    if any(value.shape != expected_shape for value in (u, pressure, jx, jy, jz)):
        raise ValueError(f"field arrays must share the problem shape {expected_shape}")
    if geometry_kind == "pipe_ogrid":
        scale = jnp.asarray(geometry_scale)
        if scale.ndim and scale.shape != (2,):
            raise ValueError("pipe geometry_scale must be scalar or (axial, radial)")
        radial_scale = scale if scale.ndim == 0 else scale[1]
        area_scale = radial_scale**2
    else:
        _, transverse_y_scale, transverse_z_scale = _coordinate_scale(geometry_scale)
        area_scale = transverse_y_scale * transverse_z_scale
    weights = area_scale * jnp.asarray(fluid * area)
    area_sum = jnp.sum(weights)
    flow = jnp.sum(weights * u, axis=(1, 2))
    mean_u = flow / area_sum
    mean_pressure = jnp.sum(weights * pressure, axis=(1, 2)) / area_sum
    pressure_drop = mean_pressure[0] - mean_pressure[-1]
    outlet_variance = jnp.sum(weights * (u[-1] - mean_u[-1]) ** 2) / area_sum
    flow_nonuniformity = outlet_variance / (mean_u[-1] ** 2 + smoothing**2)
    wall = ~fluid
    if not wall.any():
        wall[-1, :] = True
        if geometry_kind != "pipe_ogrid":
            wall[0, :] = True
            wall[:, [0, -1]] = True
    wall_weights = jnp.asarray(wall) * area
    current_squared = jx**2 + jy**2 + jz**2
    wall_current_rms = (
        jnp.sqrt(
            jnp.sum(wall_weights * current_squared) / (u.shape[0] * jnp.sum(wall_weights)) + smoothing**2
        )
        - smoothing
    )
    speed = jnp.sqrt(u**2 + smoothing**2)
    recirculation_fraction = 0.5 * jnp.sum(weights * (speed - u)) / jnp.sum(weights * speed)
    return {
        "pressure_drop": pressure_drop,
        "flow_rate": flow[-1],
        "pumping_power": pressure_drop * flow[-1],
        "flow_nonuniformity": flow_nonuniformity,
        "wall_current_density_rms": wall_current_rms,
        "recirculation_fraction": recirculation_fraction,
    }

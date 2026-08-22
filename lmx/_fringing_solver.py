"""Extruded 3-D solve orchestration and restart progress."""

from __future__ import annotations

import math
from collections.abc import Callable
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from solvax import (
    aitken_relaxation,
    anderson_mixing,
    anderson_weights,
)

try:
    from scipy import sparse
except Exception:  # pragma: no cover - SciPy should be present in shipped environments.
    sparse = None
from ._fringing_common import (
    ALEX_B2_CANONICAL_SHELL_THICKNESS,
    ALEX_B2_MAGNETIC_STABILITY_SAFETY,
    ALEX_B2_SETTLED_RELAXATION,
    ALEX_B2_STEADY_STEPS,
    ALEX_BALANCE_TOLERANCE,
    _apply_fixed_flow_pressure_constraint,
    _array_fingerprint,
    _broadcast_cross_section,
    _canonical_shell_widths,
    _cross_duct_pressure_difference,
    _enforce_stationwise_flow_rate_3d,
    _enforce_velocity_bc_3d,
    _gauge_invariant_scalar_update,
    _gradient_3d,
    _laplacian_3d,
    _normalized_pressure_observable_update,
    _poisson_jacobi_3d,
    _rectangular_fluid_bounds,
    _reuse_fringing_jit,
    _variable_coefficient_poisson_jacobi_3d,
    _variable_coefficient_poisson_sparse_3d,
)
from ._fringing_duct import (
    _clip_state,
    _compact_duct_courant_numbers,
    _conservative_current_diagnostics_3d,
    _conservative_current_fluxes_3d,
    _conservative_emf_rhs_3d,
    _cross_section_mesh,
    _duct_momentum_defect,
    _explicit_deviatoric_stress_duct,
    _face_flux_pressure_projection_duct,
    _flow_rate_inlet_profile,
    _frozen_duct_momentum_setup,
    _initialize_duct_mass_flux,
    _sample_station_magnetic_field,
    _solvax_implicit_momentum_duct,
    _solvax_pressure_poisson_duct,
    _station_axial_current_from_fluxes,
    _unpack_duct_mass_flux,
)
from ._fringing_pipe import (
    _enforce_pipe_velocity_bc,
    _fixed_flow_face_flux_projection_pipe,
    _pipe_conservative_current_diagnostics_3d,
    _pipe_conservative_current_fluxes_3d,
    _pipe_conservative_emf_rhs_3d,
    _pipe_divergence_3d,
    _pipe_gradient_3d,
    _pipe_laplacian_3d,
    _pipe_poisson_sparse_3d,
    _pipe_radial_fluid_count,
    _pipe_variable_diffusion_coefficients_3d,
    _separable_pressure_poisson_pipe,
    _solvax_diffusion_pipe,
    _solvax_pressure_poisson_pipe,
    _solve_pipe_diffusion_system,
    _steady_stokes_projection_pipe,
)
from .physics import build_material_fields
from .specs import (
    EXTRUDED_HISTORY_WIDTHS,
    CaseSpec,
    ExtrudedFieldBundle,
    ExtrudedInductionlessProblem,
    ExtrudedIterationProgress,
)


def _shard_extruded_fields(
    fields: tuple[jnp.ndarray, ...], *, num_devices: int | None
) -> tuple[jnp.ndarray, ...]:
    """Place 3-D extruded fields on an axial JAX device mesh.

    JAX propagates this named sharding through the production operators and
    inserts the required neighbor communication at axial stencil boundaries.
    An explicit one-device request uses the same named-sharding kernels as a
    multi-device run, which keeps strong-scaling baselines comparable.
    """

    if num_devices is None:
        return fields
    devices = jax.devices()
    if not 1 <= num_devices <= len(devices):
        raise ValueError(f"Requested {num_devices} devices, but only {len(devices)} are visible.")
    axial_size = fields[0].shape[0]
    if axial_size % num_devices:
        raise ValueError(f"Axial cell count {axial_size} must be divisible by {num_devices} devices.")
    sharding = _axial_field_sharding(num_devices)
    # JAX 0.6.x CUDA can leave non-primary shards uninitialized when directly
    # resharding a single-GPU array. Stage each global initial field once on the
    # host; all subsequent production iterations remain device-resident.
    return tuple(jax.device_put(np.asarray(field), sharding) for field in fields)


def _iteration_history_arrays(
    residual,
    component,
    pressure,
    electric,
    potential,
    courant=None,
    pressure_linear=None,
    momentum_defect=None,
):
    """Build consistently shaped outer-iteration histories."""
    values = (
        residual,
        momentum_defect or (),
        component,
        pressure,
        pressure_linear or (),
        electric,
        potential,
        courant or (),
    )
    return {
        name: jnp.asarray(value, dtype=float).reshape((-1, width))
        if width
        else jnp.asarray(value, dtype=float)
        for (name, width), value in zip(EXTRUDED_HISTORY_WIDTHS, values, strict=True)
    }


def _iteration_checkpoint_bundle(
    *,
    case: CaseSpec,
    x: jnp.ndarray,
    y: jnp.ndarray,
    z: jnp.ndarray,
    field_scale: jnp.ndarray,
    u: jnp.ndarray,
    v: jnp.ndarray,
    w: jnp.ndarray,
    p: jnp.ndarray,
    phi: jnp.ndarray,
    axial_pressure_loss_gradient: jnp.ndarray | None,
    transverse_pressure_difference: jnp.ndarray | None,
    residual_history: list[float],
    component_history: list[tuple[float, ...]],
    pressure_history: list[float],
    electric_history: list[tuple[float, ...]],
    potential_history: list[float],
    pressure_linear_history: list[tuple[float, ...]] | None = None,
    rho_phi_plus: jnp.ndarray | None = None,
    rho_phi_inlet: jnp.ndarray | None = None,
    aitken_state: tuple[jnp.ndarray | None, float, int] | None = None,
    anderson_state: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray] | None = None,
    stopping_state: tuple[int, int, str] = (0, 0, "not_recorded"),
    courant_history: list[tuple[float, float, float]] | None = None,
    momentum_defect_history: list[float] | None = None,
) -> ExtrudedFieldBundle:
    """Build the minimal existing-schema bundle needed to resume a solve."""

    return ExtrudedFieldBundle(
        x=x,
        y=y,
        z=z,
        field_scale=field_scale,
        u=u,
        v=v,
        w=w,
        p=p,
        phi=phi,
        rho_phi_plus=rho_phi_plus,
        rho_phi_inlet=rho_phi_inlet,
        aitken_state=aitken_state,
        anderson_state=anderson_state,
        stopping_state=stopping_state,
        geometry_kind=case.geometry.kind,
        solver_kind=case.solver.kind,
        axial_pressure_loss_gradient=(
            jnp.zeros_like(x) if axial_pressure_loss_gradient is None else axial_pressure_loss_gradient
        ),
        transverse_pressure_difference=(
            jnp.zeros_like(x) if transverse_pressure_difference is None else transverse_pressure_difference
        ),
        **_iteration_history_arrays(
            residual_history,
            component_history,
            pressure_history,
            electric_history,
            potential_history,
            courant_history,
            pressure_linear_history,
            momentum_defect_history,
        ),
    )


def _emit_iteration_progress(
    callback: Callable[[ExtrudedIterationProgress], None] | None,
    *,
    checkpoint_interval: int | None,
    step: int,
    total_steps: int,
    converged: bool,
    residual: float,
    component_residuals: tuple[float, ...],
    pressure_residual: float,
    potential_residual: float,
    checkpoint_factory: Callable[[], ExtrudedFieldBundle],
) -> None:
    if callback is None:
        return
    write_checkpoint = bool(
        checkpoint_interval and (step % checkpoint_interval == 0 or converged or step == total_steps)
    )
    callback(
        ExtrudedIterationProgress(
            step=step,
            total_steps=total_steps,
            residual=float(residual),
            component_residuals=tuple(float(value) for value in component_residuals),
            pressure_residual=float(pressure_residual),
            potential_residual=float(potential_residual),
            checkpoint=checkpoint_factory() if write_checkpoint else None,
        )
    )


def _synchronized_phase(
    function: Callable,
    name: str,
    callback: Callable[[str, float], None] | None,
) -> Callable:
    """Wrap one diagnostic phase with a completion barrier and wall timer."""

    if callback is None:
        return function

    def measured(*args):
        # Do not charge queued producer work to the named phase.
        jax.block_until_ready(args)
        started = perf_counter()
        result = function(*args)
        jax.block_until_ready(result)
        callback(name, perf_counter() - started)
        return result

    return measured


def _axial_field_sharding(num_devices: int) -> NamedSharding:
    """Return one process-stable axial mesh for compilation and repeat reuse."""

    devices = jax.devices()
    if not 1 <= num_devices <= len(devices):
        raise ValueError(f"Requested {num_devices} devices, but only {len(devices)} are visible.")
    mesh = Mesh(np.asarray(devices[:num_devices], dtype=object), ("x",))
    return NamedSharding(mesh, P("x", None, None))


def _solve_extruded_projection(
    problem: ExtrudedInductionlessProblem,
    *,
    initial_bundle: ExtrudedFieldBundle | None = None,
    num_devices: int | None = None,
    progress_callback: Callable[[ExtrudedIterationProgress], None] | None = None,
    phase_timing_callback: Callable[[str, float], None] | None = None,
    checkpoint_interval: int | None = None,
) -> ExtrudedFieldBundle:
    case = problem.case
    mesh = _cross_section_mesh(case)
    use_alex_b2_finite_volume = (
        case.name.startswith("alex_b2-fringing-square_") and case.geometry.kind == "layered_duct"
    )
    use_alex_b1_finite_volume = (
        case.name.startswith("alex_b1-fringing-pipe_") and case.geometry.kind == "pipe_ogrid"
    )
    use_compatible_steady_b1 = use_alex_b1_finite_volume
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
    if case.geometry.kind in {"pipe_ogrid", "bent_pipe"}:
        materials = build_material_fields(case, mesh)
        x = jnp.asarray(mesh.x_centers, dtype=float)
        r_faces = jnp.asarray(mesh.y_faces, dtype=float)
        r = jnp.asarray(mesh.y_centers, dtype=float)
        theta = jnp.asarray(mesh.z_centers, dtype=float)
        nx, nr, ntheta = len(x), len(r), len(theta)
        dx = float(jnp.mean(mesh.dx))
        dr = float(jnp.mean(mesh.dy))
        dr_widths = jnp.asarray(mesh.dy, dtype=float)
        dtheta = float(jnp.mean(mesh.dz))
        sigma = _broadcast_cross_section(materials.conductivity, nx)
        rho = _broadcast_cross_section(materials.density, nx)
        nu = _broadcast_cross_section(materials.viscosity, nx)
        fluid_mask = _broadcast_cross_section(materials.fluid_mask.astype(float), nx) > 0.5
        radial_fluid_count = _pipe_radial_fluid_count(fluid_mask) if use_alex_b1_finite_volume else None
        rr = jnp.broadcast_to(jnp.maximum(r[None, :, None], 0.5 * dr), (nx, nr, ntheta))
        theta_grid = jnp.broadcast_to(theta[None, None, :], (nx, nr, ntheta))
        forcing = float(case.forcing)
        field_scale = jnp.asarray(problem.profile.field_scale, dtype=float)
        bx, by, bz = _sample_station_magnetic_field(
            case,
            field_scale=field_scale,
            x=problem.profile.x,
            y=rr[0] * jnp.cos(theta_grid[0]),
            z=rr[0] * jnp.sin(theta_grid[0]),
            volume_field=problem.profile.volume_field,
        )
        br = by * jnp.cos(theta_grid) + bz * jnp.sin(theta_grid)
        btheta = -by * jnp.sin(theta_grid) + bz * jnp.cos(theta_grid)

        if initial_bundle is not None:
            if initial_bundle.u.shape != (nx, nr, ntheta):
                raise ValueError(
                    "Extruded restart bundle shape does not match the current mapped-pipe problem"
                )
            u = jnp.asarray(initial_bundle.u, dtype=float)
            v = jnp.asarray(initial_bundle.v, dtype=float)
            w = jnp.asarray(initial_bundle.w, dtype=float)
            p = jnp.asarray(initial_bundle.p, dtype=float)
            phi = jnp.asarray(initial_bundle.phi, dtype=float)
        else:
            u = jnp.where(
                fluid_mask,
                jnp.asarray(case.initial_velocity, dtype=float),
                0.0,
            )
            v = jnp.zeros_like(u)
            w = jnp.zeros_like(u)
            p = jnp.zeros_like(u)
            phi = jnp.zeros_like(u)

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
        ) = _shard_extruded_fields(
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

        min_dr = float(jnp.min(mesh.dy))
        min_arc = (
            float(jnp.min(jnp.maximum(r[1:], 0.5 * min_dr))) * dtheta
            if nr > 1
            else max(float(r[0]) * dtheta, 0.5 * min_dr * dtheta)
        )
        inverse_diffusive_scale = float(
            jnp.max(nu)
            * (1.0 / max(dx**2, 1.0e-12) + 1.0 / max(min_dr**2, 1.0e-12) + 1.0 / max(min_arc**2, 1.0e-12))
        )
        inverse_electromagnetic_scale = float(
            jnp.max(
                jnp.where(
                    fluid_mask,
                    sigma * (bx**2 + br**2 + btheta**2) / rho,
                    0.0,
                )
            )
        )
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
            if initial_bundle is not None or case.initial_velocity != 0.0
            else None
        )
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
        momentum_iterations = max(poisson_iterations, 2000 if use_compatible_steady_b1 else 400)
        momentum_tolerance = min(poisson_tolerance, 1.0e-10)
        velocity_limit = max(5.0, 2.0 * math.sqrt(float(case.geometry.target_ha or 1.0)))
        scalar_limit = max(
            20.0,
            2.0 * float(jnp.max(bx**2 + by**2 + bz**2)),
        )
        electric_potential_scale = max(1.0, math.sqrt(float(jnp.max(bx**2 + by**2 + bz**2))))
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
                use_compatible_steady_b1,
            )

            def diffusion_system_solve(linear_rhs, volume, coefficients, wall_sink, initial):
                return _solve_pipe_diffusion_system(
                    linear_rhs,
                    volume,
                    coefficients,
                    wall_sink,
                    initial,
                    mass_coefficient=0.0 if use_compatible_steady_b1 else 1.0,
                    diffusion_coefficient=1.0 if use_compatible_steady_b1 else dt,
                    iterations=momentum_iterations,
                    tolerance=momentum_tolerance,
                )

            diffusion_system_solve = _reuse_fringing_jit(kernel_key, jax.jit(diffusion_system_solve))
            steady_reaction = (
                2.0
                * sigma[:, :count, :]
                * (bx[:, :count, :] ** 2 + br[:, :count, :] ** 2 + btheta[:, :count, :] ** 2)
                / rho[:, :count, :]
                if use_compatible_steady_b1
                else None
            )
            if use_compatible_steady_b1:
                steady_coefficients = _pipe_variable_diffusion_coefficients_3d(
                    nu[:, :count, :],
                    dx=dx,
                    r_faces=faces,
                    r_centers=centers,
                    dtheta=dtheta,
                )
                radial_widths = jnp.diff(faces)
                wall_sink = (
                    jnp.zeros_like(steady_reaction)
                    .at[:, -1, :]
                    .set(
                        nu[:, count - 1, :]
                        * faces[-1]
                        / jnp.maximum(
                            centers[-1] * radial_widths[-1] * (0.5 * radial_widths[-1]),
                            1.0e-20,
                        )
                    )
                )
                steady_rate_diagonal = sum(steady_coefficients) + wall_sink + steady_reaction
                pressure_preconditioner_mobility = 1.0 / jnp.maximum(
                    rho[:, :count, :] * steady_rate_diagonal, 1.0e-20
                )
                modal_factor_key = (
                    "b1_modal_factors",
                    "retained",
                    jax.default_backend(),
                    u.dtype.str,
                    kernel_key,
                    _array_fingerprint(
                        rho[:, :count, :],
                        nu[:, :count, :],
                        steady_reaction,
                        fluid_cell_area[:, :count, :],
                    ),
                )
            else:
                pressure_preconditioner_mobility = None
                modal_factor_key = None

            momentum_viscosity = nu[:, :count, :]

            def momentum_solve(rhs, initial):
                return _solvax_diffusion_pipe(
                    rhs,
                    momentum_viscosity,
                    dt=None if use_compatible_steady_b1 else dt,
                    dx=dx,
                    r_faces=faces,
                    r_centers=centers,
                    dtheta=dtheta,
                    iterations=momentum_iterations,
                    tolerance=momentum_tolerance,
                    initial_field=initial,
                    reaction=steady_reaction,
                    _system_solve=(None if use_compatible_steady_b1 else diffusion_system_solve),
                )

            if use_compatible_steady_b1:
                momentum_solve = _reuse_fringing_jit(
                    (
                        "b1_momentum",
                        jax.default_backend(),
                        kernel_key,
                        _array_fingerprint(momentum_viscosity, steady_reaction),
                    ),
                    jax.jit(momentum_solve),
                )
            response_rhs = (1.0 if use_compatible_steady_b1 else dt) / rho[:, :count, :]
            response_fluid, _, _ = momentum_solve(response_rhs, jnp.zeros_like(response_rhs))
            if not use_compatible_steady_b1:
                response_cross_section = jnp.mean(response_fluid, axis=0, keepdims=True)
                response_fluid = jnp.broadcast_to(response_cross_section, response_fluid.shape)
                flow_response_matrix = None
            else:
                basis_rhs = jnp.eye(nx, dtype=u.dtype)[:, :, None, None] / rho[None, :, :count, :]
                zero = jnp.zeros_like(response_fluid)
                basis_response = jnp.stack(tuple(momentum_solve(rhs, zero)[0] for rhs in basis_rhs))
                flow_response_matrix = jnp.sum(
                    basis_response * fluid_cell_area[None, :, :count, :], axis=(2, 3)
                ).T
            unit_pressure_response = jnp.zeros_like(u).at[:, :count, :].set(response_fluid)
        else:
            unit_pressure_response, _, _ = _enforce_pipe_velocity_bc(
                jnp.where(fluid_mask, dt / rho, 0.0),
                jnp.zeros_like(u),
                jnp.zeros_like(u),
                r_centers=r,
                r_faces=r_faces,
                fluid_mask=fluid_mask,
            )
        axial_pressure_loss_gradient = (
            jnp.asarray(initial_bundle.axial_pressure_loss_gradient, dtype=float)
            if initial_bundle is not None
            and initial_bundle.axial_pressure_loss_gradient is not None
            and initial_bundle.axial_pressure_loss_gradient.shape == (nx,)
            else jnp.full((nx,), forcing, dtype=float)
        )

        for step in range(outer_steps):
            phi_previous = phi
            pressure_gradient_previous = axial_pressure_loss_gradient
            dphi_dx, dphi_dr, dphi_dtheta = _pipe_gradient_3d(
                phi,
                dx=dx,
                dr=dr_widths if use_alex_b1_finite_volume else dr,
                dtheta=dtheta,
                r=rr,
            )
            uxb_x = v * btheta - w * br
            uxb_r = w * bx - u * btheta
            uxb_theta = u * br - v * bx
            jx = sigma * (-dphi_dx + uxb_x)
            jr = sigma * (-dphi_dr + uxb_r)
            jtheta = sigma * (-dphi_dtheta + uxb_theta)
            lorentz_x = jr * btheta - jtheta * br
            lorentz_r = jtheta * bx - jx * btheta
            lorentz_theta = jx * br - jr * bx

            if use_alex_b1_finite_volume:
                dp_dx = jnp.zeros_like(p)
                dp_dr = jnp.zeros_like(p)
                dp_dtheta = jnp.zeros_like(p)
            else:
                dp_dx, dp_dr, dp_dtheta = _pipe_gradient_3d(p, dx=dx, dr=dr, dtheta=dtheta, r=rr)
                laplacian_u = _pipe_laplacian_3d(u, dx=dx, dr=dr, dtheta=dtheta, r=rr)
                laplacian_v = _pipe_laplacian_3d(v, dx=dx, dr=dr, dtheta=dtheta, r=rr)
                laplacian_w = _pipe_laplacian_3d(w, dx=dx, dr=dr, dtheta=dtheta, r=rr)
            if use_alex_b1_finite_volume:
                count = radial_fluid_count
                faces = r_faces[: count + 1]
                centers = r[:count]
                rhs_u = u[:, :count, :] + dt * (
                    forcing / rho[:, :count, :] + lorentz_x[:, :count, :] / rho[:, :count, :]
                )
                rhs_v = v[:, :count, :] + dt * (lorentz_r[:, :count, :] / rho[:, :count, :])
                rhs_w = w[:, :count, :] + dt * (lorentz_theta[:, :count, :] / rho[:, :count, :])
                if use_compatible_steady_b1:
                    rhs_u = (forcing + lorentz_x[:, :count, :]) / rho[:, :count, :] + steady_reaction * u[
                        :, :count, :
                    ]
                    rhs_v = lorentz_r[:, :count, :] / rho[:, :count, :] + steady_reaction * v[:, :count, :]
                    rhs_w = (
                        lorentz_theta[:, :count, :] / rho[:, :count, :] + steady_reaction * w[:, :count, :]
                    )
                u_fluid, _, _ = momentum_solve(rhs_u, u[:, :count, :])
                v_fluid, _, _ = momentum_solve(rhs_v, v[:, :count, :])
                w_fluid, _, _ = momentum_solve(rhs_w, w[:, :count, :])
                u_star = jnp.zeros_like(u).at[:, :count, :].set(u_fluid)
                v_star = jnp.zeros_like(v).at[:, :count, :].set(v_fluid)
                w_star = jnp.zeros_like(w).at[:, :count, :].set(w_fluid)
            else:
                u_star = u + dt * (laplacian_u * nu + forcing / rho + lorentz_x / rho - dp_dx / rho)
                v_star = v + dt * (laplacian_v * nu + lorentz_r / rho - dp_dr / rho)
                w_star = w + dt * (laplacian_w * nu + lorentz_theta / rho - dp_dtheta / rho)
            if not use_compatible_steady_b1:
                u_star = _clip_state(u_star, velocity_limit)
                v_star = _clip_state(v_star, velocity_limit)
                w_star = _clip_state(w_star, velocity_limit)
            if use_alex_b1_finite_volume:
                u_star = jnp.where(fluid_mask, u_star, 0.0)
                v_star = jnp.where(fluid_mask, v_star, 0.0)
                w_star = jnp.where(fluid_mask, w_star, 0.0)
                if target_flow_rate is None:
                    raise ValueError("ALEX B1 requires its frozen fixed mean flow rate")
                if use_compatible_steady_b1:
                    zero = jnp.zeros_like(u_fluid)
                    steady_projection = _steady_stokes_projection_pipe(
                        u_fluid,
                        v_fluid,
                        w_fluid,
                        rho[:, :count, :],
                        response_fluid,
                        fluid_cell_area[:, :count, :],
                        lambda rhs: momentum_solve(rhs, zero)[0],
                        target_flow_rate=target_flow_rate,
                        dx=dx,
                        r_faces=faces,
                        r_centers=centers,
                        dtheta=dtheta,
                        pressure_iterations=projection_iterations,
                        pressure_tolerance=momentum_tolerance,
                        flow_response_matrix=flow_response_matrix,
                        pressure_preconditioner_mobility=(pressure_preconditioner_mobility),
                        modal_momentum_coefficients=steady_coefficients,
                        modal_momentum_sink=wall_sink + steady_reaction,
                        modal_stabilization=True,
                        modal_factor_key=modal_factor_key,
                        physical_tolerance=ALEX_BALANCE_TOLERANCE,
                    )
                    u_next = jnp.zeros_like(u).at[:, :count, :].set(steady_projection[0])
                    v_next = jnp.zeros_like(v).at[:, :count, :].set(steady_projection[1])
                    w_next = jnp.zeros_like(w).at[:, :count, :].set(steady_projection[2])
                    p_corr = jnp.zeros_like(p).at[:, :count, :].set(steady_projection[3])
                    axial_pressure_loss_gradient = forcing + steady_projection[4]
                    projected_divergence_norm = steady_projection[5]
                    fixed_flow_error = steady_projection[6]
                else:
                    (
                        u_next,
                        v_next,
                        w_next,
                        p_corr,
                        axial_pressure_loss_gradient,
                        projected_divergence_norm,
                        fixed_flow_error,
                    ) = _fixed_flow_face_flux_projection_pipe(
                        u_star,
                        v_star,
                        w_star,
                        rho,
                        fluid_mask,
                        unit_pressure_response,
                        fluid_cell_area,
                        target_flow_rate=target_flow_rate,
                        base_pressure_loss_gradient=forcing,
                        dt=dt,
                        dx=dx,
                        r_faces=r_faces,
                        r_centers=r,
                        dtheta=dtheta,
                        iterations=projection_iterations,
                        tolerance=projection_tolerance,
                        radial_fluid_count=radial_fluid_count,
                        initial_pressure=p,
                        include_theta_line=True,
                    )
                if not use_compatible_steady_b1:
                    p_corr = _clip_state(p_corr, scalar_limit)
                    u_next = _clip_state(u_next, velocity_limit)
                    v_next = _clip_state(v_next, velocity_limit)
                    w_next = _clip_state(w_next, velocity_limit)
            else:
                u_star, v_star, w_star = _enforce_pipe_velocity_bc(
                    u_star,
                    v_star,
                    w_star,
                    r_centers=r,
                    r_faces=r_faces,
                    fluid_mask=fluid_mask,
                )
                divergence = _pipe_divergence_3d(u_star, v_star, w_star, dx=dx, dr=dr, dtheta=dtheta, r=rr)
                pressure_rhs = (rho / max(dt, 1.0e-12)) * divergence
                p_corr, _, _, _ = _pipe_poisson_sparse_3d(
                    -pressure_rhs,
                    jnp.ones_like(rho),
                    dx=dx,
                    r_faces=r_faces,
                    r_centers=r,
                    dtheta=dtheta,
                    iterations=electric_iterations,
                    tolerance=electric_tolerance,
                    initial_field=phi,
                )
                p_corr = _clip_state(p_corr, scalar_limit)
                dpc_dx, dpc_dr, dpc_dtheta = _pipe_gradient_3d(p_corr, dx=dx, dr=dr, dtheta=dtheta, r=rr)
                u_next = _clip_state(u_star - (dt / rho) * dpc_dx, velocity_limit)
                v_next = _clip_state(v_star - (dt / rho) * dpc_dr, velocity_limit)
                w_next = _clip_state(w_star - (dt / rho) * dpc_dtheta, velocity_limit)
                u_next, v_next, w_next = _enforce_pipe_velocity_bc(
                    u_next,
                    v_next,
                    w_next,
                    r_centers=r,
                    r_faces=r_faces,
                    fluid_mask=fluid_mask,
                )
                if target_flow_rate is None:
                    u_next = _enforce_stationwise_flow_rate_3d(
                        u_next,
                        active_mask=fluid_mask,
                        cell_area=fluid_cell_area,
                        relaxation=0.25,
                    )
                    axial_pressure_loss_gradient = jnp.full((nx,), forcing, dtype=float)
                else:
                    u_next, axial_pressure_loss_gradient = _apply_fixed_flow_pressure_constraint(
                        u_next,
                        unit_pressure_response=unit_pressure_response,
                        active_mask=fluid_mask,
                        cell_area=fluid_cell_area,
                        target_flow_rate=target_flow_rate,
                        base_pressure_loss_gradient=forcing,
                    )
                u_next, v_next, w_next = _enforce_pipe_velocity_bc(
                    u_next,
                    v_next,
                    w_next,
                    r_centers=r,
                    r_faces=r_faces,
                    fluid_mask=fluid_mask,
                )
                projected_divergence_norm = jnp.asarray(jnp.nan)
                fixed_flow_error = jnp.asarray(0.0)
            p = _clip_state(p_corr if use_alex_b1_finite_volume else p + p_corr, scalar_limit)

            uxb_x = v_next * btheta - w_next * br
            uxb_r = w_next * bx - u_next * btheta
            uxb_theta = u_next * br - v_next * bx
            emf_rhs = _pipe_conservative_emf_rhs_3d(
                sigma,
                uxb_x,
                uxb_r,
                uxb_theta,
                dx=dx,
                r_faces=r_faces,
                r_centers=r,
                dtheta=dtheta,
            )
            if use_compatible_steady_b1:
                (
                    phi,
                    electric_residual,
                    electric_converged,
                    electric_relative_residual,
                    electric_iteration_count,
                    electric_status,
                    electric_local_residual,
                ) = _separable_pressure_poisson_pipe(
                    emf_rhs,
                    sigma,
                    dx=dx,
                    r_faces=r_faces,
                    r_centers=r,
                    dtheta=dtheta,
                    tolerance=electric_tolerance,
                )
            elif use_alex_b1_finite_volume:
                (
                    phi,
                    electric_residual,
                    electric_converged,
                    electric_relative_residual,
                    electric_iteration_count,
                    electric_status,
                    electric_local_residual,
                ) = _solvax_pressure_poisson_pipe(
                    emf_rhs,
                    sigma,
                    dx=dx,
                    r_faces=r_faces,
                    r_centers=r,
                    dtheta=dtheta,
                    iterations=electric_iterations,
                    tolerance=electric_tolerance,
                    initial_field=phi,
                    local_tolerance=ALEX_BALANCE_TOLERANCE,
                    include_theta_line=True,
                )
            else:
                # The sparse pipe operator represents -div(sigma grad(phi)); J
                # is sigma(-grad(phi) + u x B), hence the opposite source sign.
                phi, _, _, _ = _pipe_poisson_sparse_3d(
                    -emf_rhs,
                    sigma,
                    dx=dx,
                    r_faces=r_faces,
                    r_centers=r,
                    dtheta=dtheta,
                    iterations=poisson_iterations,
                    tolerance=poisson_tolerance,
                    initial_field=phi,
                )
                electric_residual = jnp.asarray(jnp.nan)
                electric_relative_residual = jnp.asarray(jnp.nan)
                electric_iteration_count = jnp.asarray(0)
                electric_converged = jnp.asarray(False)
                electric_status = jnp.asarray(-1)
                electric_local_residual = jnp.asarray(jnp.nan)
            phi = _clip_state(phi, scalar_limit)
            potential_update = _gauge_invariant_scalar_update(
                phi,
                phi_previous,
                cell_area,
                scale=electric_potential_scale,
            )
            electric_linear_by_step.append(
                (
                    float(electric_residual),
                    float(electric_relative_residual),
                    float(electric_local_residual),
                    float(electric_iteration_count),
                    float(electric_converged),
                    float(electric_status),
                )
            )

            fx, fr, ftheta = _pipe_conservative_current_fluxes_3d(
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
            div_j, _, _ = _pipe_conservative_current_diagnostics_3d(
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
            jx = _clip_state(0.5 * (fx[1:] + fx[:-1]), scalar_limit)
            jr = _clip_state(0.5 * (fr[:, 1:, :] + fr[:, :-1, :]), scalar_limit)
            jtheta = _clip_state(0.5 * (ftheta + jnp.roll(ftheta, 1, axis=2)), scalar_limit)
            lorentz_x = jr * btheta - jtheta * br
            lorentz_r = jtheta * bx - jx * btheta
            lorentz_theta = jx * br - jr * bx
            if use_alex_b1_finite_volume:
                projected_divergence_max = float(projected_divergence_norm)
            else:
                projected_divergence = _pipe_divergence_3d(
                    u_next,
                    v_next,
                    w_next,
                    dx=dx,
                    dr=dr,
                    dtheta=dtheta,
                    r=rr,
                )
                projected_divergence_max = float(jnp.max(jnp.abs(projected_divergence)))
            u_update = float(jnp.max(jnp.abs(u_next - u)))
            v_update = float(jnp.max(jnp.abs(v_next - v)))
            w_update = float(jnp.max(jnp.abs(w_next - w)))
            flow_error_value = float(fixed_flow_error)
            pressure_update = (
                _normalized_pressure_observable_update(
                    axial_pressure_loss_gradient,
                    pressure_gradient_previous,
                    bx**2 + by**2 + bz**2,
                )
                if use_alex_b1_finite_volume
                else 0.0
            )
            update_residual = max(
                u_update,
                v_update,
                w_update,
                pressure_update,
                potential_update,
            )
            charge_balance = float(jnp.max(jnp.abs(div_j)))
            residual_by_step.append(update_residual)
            pressure_residual_by_step.append(pressure_update)
            potential_residual_by_step.append(potential_update)
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
            converged = (
                update_residual <= case.solver.coupling_tolerance
                and projected_divergence_max <= ALEX_BALANCE_TOLERANCE
                and flow_error_value <= ALEX_BALANCE_TOLERANCE
                and charge_balance <= ALEX_BALANCE_TOLERANCE
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
            _emit_iteration_progress(
                progress_callback,
                checkpoint_interval=checkpoint_interval,
                step=step + 1,
                total_steps=outer_steps,
                converged=converged,
                residual=update_residual,
                component_residuals=component_residual_by_step[-1],
                pressure_residual=pressure_update,
                potential_residual=potential_update,
                checkpoint_factory=lambda: _iteration_checkpoint_bundle(
                    case=case,
                    x=x,
                    y=r,
                    z=theta,
                    field_scale=field_scale,
                    u=u,
                    v=v,
                    w=w,
                    p=p,
                    phi=phi,
                    axial_pressure_loss_gradient=axial_pressure_loss_gradient,
                    transverse_pressure_difference=None,
                    residual_history=residual_by_step,
                    component_history=component_residual_by_step,
                    pressure_history=pressure_residual_by_step,
                    electric_history=electric_linear_by_step,
                    potential_history=potential_residual_by_step,
                    stopping_state=(
                        step + 1,
                        0,
                        "converged"
                        if converged
                        else "step_limit"
                        if step + 1 == outer_steps
                        else "in_progress",
                    ),
                ),
            )
            if converged:
                break

        final_step_residual = residual_by_step[-1] if residual_by_step else 0.0
        residual = jnp.full((nx,), final_step_residual, dtype=float)
        cross_section_area = jnp.maximum(jnp.sum(fluid_cell_area, axis=(1, 2)), 1.0e-20)
        volumetric_flow_rate = jnp.sum(u * fluid_cell_area, axis=(1, 2))
        mean_velocity = volumetric_flow_rate / cross_section_area
        axial_current = jnp.sum(jx * cell_area, axis=(1, 2))
        final_div_j, wall_current_leakage, boundary_current_residual = (
            _pipe_conservative_current_diagnostics_3d(
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
        return ExtrudedFieldBundle(
            x=x,
            y=r,
            z=theta,
            field_scale=field_scale,
            u=u,
            v=v,
            w=w,
            p=p,
            phi=phi,
            jx=jx,
            jy=jr,
            jz=jtheta,
            lorentz_x=lorentz_x,
            lorentz_y=lorentz_r,
            lorentz_z=lorentz_theta,
            residual=residual,
            volumetric_flow_rate=volumetric_flow_rate,
            mean_velocity=mean_velocity,
            axial_current=axial_current,
            wall_current_leakage=wall_current_leakage,
            current_scaled_pressure_proxy=current_scaled_pressure_proxy,
            charge_balance_residual=charge_balance_residual,
            boundary_current_residual=boundary_current_residual,
            geometry_kind=case.geometry.kind,
            solver_kind=case.solver.kind,
            stopping_state=(
                len(residual_by_step),
                0,
                "converged" if converged else "step_limit",
            ),
            axial_pressure_loss_gradient=axial_pressure_loss_gradient,
            transverse_pressure_difference=jnp.zeros((nx,), dtype=float),
            **_iteration_history_arrays(
                residual_by_step,
                component_residual_by_step,
                pressure_residual_by_step,
                electric_linear_by_step,
                potential_residual_by_step,
            ),
        )
    materials = build_material_fields(case, mesh)
    x = jnp.asarray(mesh.x_centers, dtype=float)
    y = jnp.asarray(mesh.y_centers, dtype=float)
    z = jnp.asarray(mesh.z_centers, dtype=float)
    nx, ny, nz = len(x), len(y), len(z)
    dx = float(jnp.mean(mesh.dx))
    dy = jnp.asarray(mesh.dy, dtype=float)
    dz = jnp.asarray(mesh.dz, dtype=float)
    dy_momentum = float(jnp.mean(dy))
    dz_momentum = float(jnp.mean(dz))
    sigma = _broadcast_cross_section(materials.conductivity, nx)
    rho = _broadcast_cross_section(materials.density, nx)
    nu = _broadcast_cross_section(materials.viscosity, nx)
    fluid_mask = _broadcast_cross_section(materials.fluid_mask.astype(float), nx) > 0.5
    fluid_bounds = _rectangular_fluid_bounds(fluid_mask) if use_alex_b2_finite_volume else None
    if use_alex_b2_finite_volume:
        y0, y1, z0, z1 = fluid_bounds
        dy = _canonical_shell_widths(dy, y0, y1)
        dz = _canonical_shell_widths(dz, z0, z1)
        wall = next(region for region in case.regions if region.kind == "solid")
        sheet_conductance = wall.conductivity * wall.wall_thickness
        sigma = jnp.where(
            fluid_mask,
            sigma,
            sheet_conductance / ALEX_B2_CANONICAL_SHELL_THICKNESS,
        )
    cell_area = _broadcast_cross_section(dy[:, None] * dz[None, :], nx)
    forcing = float(case.forcing)
    field_scale = jnp.asarray(problem.profile.field_scale, dtype=float)
    field_y, field_z = jnp.meshgrid(y, z, indexing="ij")
    bx, by, bz = _sample_station_magnetic_field(
        case,
        field_scale=field_scale,
        x=x,
        y=field_y,
        z=field_z,
        volume_field=problem.profile.volume_field,
    )

    if initial_bundle is not None:
        if initial_bundle.u.shape != (nx, ny, nz):
            raise ValueError("Extruded restart bundle shape does not match the current duct problem")
        u = jnp.asarray(initial_bundle.u, dtype=float)
        v = jnp.asarray(initial_bundle.v, dtype=float)
        w = jnp.asarray(initial_bundle.w, dtype=float)
        p = jnp.asarray(initial_bundle.p, dtype=float)
        phi = jnp.asarray(initial_bundle.phi, dtype=float)
    else:
        u = jnp.where(
            fluid_mask,
            jnp.asarray(case.initial_velocity, dtype=float),
            0.0,
        )
        v = jnp.zeros_like(u)
        w = jnp.zeros_like(u)
        p = jnp.zeros_like(u)
        phi = jnp.zeros_like(u)

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
    ) = _shard_extruded_fields(
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

    inverse_diffusive_scale = float(
        jnp.max(nu)
        * (
            1.0 / max(dx**2, 1.0e-12)
            + 1.0 / max(float(jnp.min(dy)) ** 2, 1.0e-12)
            + 1.0 / max(float(jnp.min(dz)) ** 2, 1.0e-12)
        )
    )
    inverse_electromagnetic_scale = float(
        jnp.max(
            jnp.where(
                fluid_mask,
                sigma * (bx**2 + by**2 + bz**2) / rho,
                0.0,
            )
        )
    )
    stability_safety = (
        ALEX_B2_MAGNETIC_STABILITY_SAFETY
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
    else:
        target_flow_rate = (
            float(jnp.mean(jnp.sum(jnp.where(fluid_mask, u * cell_area, 0.0), axis=(1, 2))))
            if initial_bundle is not None or case.initial_velocity != 0.0
            else None
        )
    outer_steps = min(case.time_stepper.max_steps, max(6, case.solver.coupling_iterations * 2))
    poisson_iterations = (
        case.time_stepper.potential_iterations
        if use_alex_b2_finite_volume
        else min(case.time_stepper.potential_iterations, 80)
    )
    poisson_tolerance = case.solver.coupling_tolerance
    electric_iterations = max(poisson_iterations, 600)
    electric_tolerance = min(poisson_tolerance, 1.0e-12)
    projection_iterations = max(poisson_iterations, 4000)
    projection_tolerance = min(poisson_tolerance, 1.0e-12)
    momentum_iterations = max(poisson_iterations, 400)
    momentum_tolerance = min(poisson_tolerance, 1.0e-10)
    velocity_limit = max(5.0, 2.0 * math.sqrt(float(case.geometry.target_ha or 1.0)))
    scalar_limit = max(
        20.0,
        2.0 * float(jnp.max(bx**2 + by**2 + bz**2)),
    )
    electric_potential_scale = max(1.0, math.sqrt(float(jnp.max(bx**2 + by**2 + bz**2))))
    history_names = (
        "iteration_residual_history",
        "iteration_component_residual_history",
        "iteration_pressure_residual_history",
        "iteration_electric_linear_history",
        "iteration_potential_residual_history",
    )
    histories = tuple(
        [] if initial_bundle is None else np.asarray(getattr(initial_bundle, name, jnp.zeros((0,)))).tolist()
        for name in history_names
    )
    if len({len(history) for history in histories}) != 1:
        raise ValueError("B2 restart iteration histories have inconsistent lengths")
    (
        residual_by_step,
        component_residual_by_step,
        pressure_residual_by_step,
        electric_linear_by_step,
        potential_residual_by_step,
    ) = histories
    completed_steps = len(residual_by_step)
    momentum_defect_by_step = (
        []
        if initial_bundle is None
        else np.asarray(initial_bundle.iteration_momentum_defect_history).tolist()
    )
    if use_alex_b2_finite_volume and len(momentum_defect_by_step) != completed_steps:
        raise ValueError("B2 restart predates the electromagnetic momentum-defect contract")
    pressure_linear_by_step = (
        []
        if initial_bundle is None
        else np.asarray(
            getattr(initial_bundle, "iteration_pressure_linear_history", jnp.zeros((0, 5)))
        ).tolist()
    )
    if completed_steps and not pressure_linear_by_step:
        pressure_linear_by_step = [[math.nan, math.nan, 0.0, 0.0, -1.0]] * completed_steps
    if len(pressure_linear_by_step) != completed_steps:
        raise ValueError("B2 restart pressure-linear history has inconsistent length")
    courant_by_step = (
        []
        if initial_bundle is None
        else np.asarray(getattr(initial_bundle, "iteration_courant_history", jnp.zeros((0, 3)))).tolist()
    )
    if completed_steps and not courant_by_step:
        courant_by_step = [[-1.0, -1.0, -1.0]] * completed_steps
    if len(courant_by_step) != completed_steps:
        raise ValueError("B2 restart CFL histories have inconsistent lengths")
    previous_fixed_point_residual: jnp.ndarray | None = None
    fixed_aitken_relaxation = (
        float(case.solver.coupling_min_relaxation)
        if use_alex_b2_finite_volume
        and case.solver.coupling_acceleration == "aitken"
        and case.solver.coupling_min_relaxation == case.solver.coupling_max_relaxation
        else None
    )
    restart_stopping = (
        (0, 0, "not_recorded")
        if initial_bundle is None
        else getattr(initial_bundle, "stopping_state", (0, 0, "not_recorded"))
    )
    if restart_stopping[0] not in (0, completed_steps):
        raise ValueError("B2 restart stopping state has inconsistent step count")
    steady_streak = int(restart_stopping[1])
    fixed_point_relaxation = jnp.asarray(1.0, dtype=u.dtype)
    restart_aitken = None if initial_bundle is None else getattr(initial_bundle, "aitken_state", None)
    if use_alex_b2_finite_volume and restart_aitken is not None:
        previous_fixed_point_residual, fixed_point_relaxation, stored_streak = restart_aitken
        if restart_stopping[2] == "not_recorded":
            steady_streak = stored_streak
        elif steady_streak != stored_streak:
            raise ValueError("B2 restart stopping and Aitken streaks disagree")
        fixed_point_relaxation = jnp.asarray(fixed_point_relaxation, dtype=u.dtype)
        if fixed_aitken_relaxation is not None:
            previous_fixed_point_residual = None
        elif previous_fixed_point_residual is not None:
            previous_fixed_point_residual = jnp.asarray(previous_fixed_point_residual, dtype=u.dtype)
            if previous_fixed_point_residual.shape != (4, nx, ny, nz):
                raise ValueError("B2 restart Aitken residual has inconsistent shape")
    previous_anderson_mapped = previous_anderson_residual = None
    previous_anderson_flux = previous_anderson_inlet = None
    restart_anderson = None if initial_bundle is None else getattr(initial_bundle, "anderson_state", None)
    if use_alex_b2_finite_volume and case.solver.coupling_acceleration == "anderson":
        if completed_steps and restart_anderson is None:
            raise ValueError("B2 Anderson restart is missing accelerator state")
        if restart_anderson is not None:
            if len(restart_anderson) != 4 or any(value is None for value in restart_anderson):
                raise ValueError("B2 Anderson restart state must be all-or-none")
            (
                previous_anderson_mapped,
                previous_anderson_residual,
                previous_anderson_flux,
                previous_anderson_inlet,
            ) = (jnp.asarray(value, dtype=u.dtype) for value in restart_anderson)
            if previous_anderson_mapped.shape != (4, nx, ny, nz) or previous_anderson_residual.shape != (
                4,
                nx,
                ny,
                nz,
            ):
                raise ValueError("B2 restart Anderson field state has inconsistent shape")
    fixed_point_scale = jnp.asarray(
        [
            velocity_limit,
            velocity_limit,
            velocity_limit,
            electric_potential_scale,
        ],
        dtype=u.dtype,
    )[:, None, None, None]
    axial_pressure_loss_gradient = jnp.full((nx,), forcing, dtype=float)
    if use_alex_b2_finite_volume:
        y0, y1, z0, z1 = fluid_bounds
        local_dy = dy[y0:y1]
        local_dz = dz[z0:z1]
        field_sharding = u.sharding if num_devices is not None else None
        replicated_sharding = None if field_sharding is None else NamedSharding(field_sharding.mesh, P())
        flux_sharding = (
            None if field_sharding is None else NamedSharding(field_sharding.mesh, P(None, "x", None, None))
        )
        vector_sharding = (
            None if field_sharding is None else NamedSharding(field_sharding.mesh, P("x", None, None, None))
        )
        kernel_key = (
            field_sharding,
            u.shape,
            fluid_bounds,
            dt,
            dx,
            tuple(np.asarray(dy)),
            tuple(np.asarray(dz)),
            momentum_iterations,
            momentum_tolerance,
            projection_iterations,
            projection_tolerance,
            electric_iterations,
            electric_tolerance,
            forcing,
            target_flow_rate,
        )

        face_area = local_dy[:, None] * local_dz[None, :]
        fluid = next(region for region in case.regions if region.kind == "fluid")
        prescribed_field = case.magnetic_field.value or (0.0, 0.0, 0.0)
        reference_speed = target_flow_rate / float(jnp.sum(face_area))
        electromagnetic_force_scale = (
            fluid.conductivity
            * reference_speed
            * sum(float(component) ** 2 for component in prescribed_field)
        )
        if electromagnetic_force_scale <= 0.0:
            raise ValueError("ALEX B2 requires a positive electromagnetic force scale")
        kernel_key = (
            *kernel_key,
            electromagnetic_force_scale,
            case.solver.coupling_regularization,
            case.solver.coupling_damping,
        )
        restart_flux = None if initial_bundle is None else initial_bundle.rho_phi_plus
        restart_inlet = None if initial_bundle is None else initial_bundle.rho_phi_inlet
        if (restart_flux is None) != (restart_inlet is None):
            raise ValueError("B2 restart requires both compact flux arrays")

        def initialize_flux(u0, v0, w0, density):
            velocity = jnp.stack((u0[:, y0:y1, z0:z1], v0[:, y0:y1, z0:z1], w0[:, y0:y1, z0:z1]), axis=-1)
            density = density[:, y0:y1, z0:z1]
            inlet = (
                velocity[0]
                .at[..., 0]
                .set(_flow_rate_inlet_profile(velocity[0, ..., 0], face_area, target_flow_rate))
            )
            return _initialize_duct_mass_flux(
                velocity, density, inlet, dx=dx, dy=local_dy, dz=local_dz, sharding=field_sharding
            )

        def momentum_solve(velocity, force, density, viscosity, rho_phi_plus, rho_phi_inlet):
            local_velocity, local_density, local_viscosity = (
                field[:, y0:y1, z0:z1] for field in (velocity, density, viscosity)
            )
            inlet_patch = local_velocity[0].at[..., 0].set(rho_phi_inlet / (local_density[0] * face_area))
            zero_y, zero_z = (jnp.zeros_like(local_velocity[:, 0]), jnp.zeros_like(local_velocity[:, :, 0]))
            boundary_velocity = (inlet_patch, local_velocity[-1], zero_y, zero_y, zero_z, zero_z)
            widths = (jnp.full((nx,), dx), local_dy, local_dz)
            rho_phi = _unpack_duct_mass_flux(rho_phi_plus, rho_phi_inlet)
            setup = _frozen_duct_momentum_setup(
                local_velocity, local_density, local_viscosity, rho_phi, boundary_velocity, widths, dx=dx
            )
            local_force = force[:, y0:y1, z0:z1] + _explicit_deviatoric_stress_duct(
                local_velocity, setup[0], boundary_velocity, widths, gradient=setup[-1]
            )
            return _solvax_implicit_momentum_duct(
                local_velocity,
                local_force,
                local_density,
                local_viscosity,
                rho_phi,
                boundary_velocity,
                dt=dt,
                dx=dx,
                dy=local_dy,
                dz=local_dz,
                iterations=momentum_iterations,
                tolerance=momentum_tolerance,
                frozen_setup=setup,
            )

        def momentum_defect(
            velocity, lorentz_force, density, viscosity, rho_phi_plus, rho_phi_inlet, pressure
        ):
            return _duct_momentum_defect(
                velocity[:, y0:y1, z0:z1],
                lorentz_force[:, y0:y1, z0:z1],
                density[:, y0:y1, z0:z1],
                viscosity[:, y0:y1, z0:z1],
                rho_phi_plus,
                rho_phi_inlet,
                pressure[:, y0:y1, z0:z1],
                forcing=forcing,
                force_scale=electromagnetic_force_scale,
                dt=dt,
                dx=dx,
                dy=local_dy,
                dz=local_dz,
                field_sharding=field_sharding,
            )

        def embed_velocity(local_velocity, mask):
            embedded = jnp.pad(local_velocity, ((0, 0), (y0, ny - y1), (z0, nz - z1), (0, 0)))
            return tuple(jnp.where(mask, embedded[..., i], 0.0) for i in range(3))

        def courant_numbers(east, north, top, inlet, density):
            return _compact_duct_courant_numbers(
                (east, north, top),
                inlet,
                density[:, y0:y1, z0:z1],
                dt=dt,
                dx=dx,
                dy=local_dy,
                dz=local_dz,
                sharding=field_sharding,
            )

        def pack_flux(x, y, z):
            return jnp.stack((x, y, z))

        def unpack_flux(flux):
            return tuple(flux)

        def pack_vector(x, y, z):
            return jnp.stack((x, y, z), axis=-1)

        def relax_flux(
            current_x,
            current_y,
            current_z,
            current_inlet,
            mapped_x,
            mapped_y,
            mapped_z,
            mapped_inlet,
            relaxation,
        ):
            components = tuple(
                current + relaxation * (mapped - current)
                for current, mapped in zip(
                    (current_x, current_y, current_z), (mapped_x, mapped_y, mapped_z), strict=True
                )
            )
            return (*components, current_inlet + relaxation * (mapped_inlet - current_inlet))

        momentum_solve = jax.named_call(momentum_solve, name="lmx.b2.momentum")
        momentum_defect = jax.named_call(momentum_defect, name="lmx.b2.momentum_defect")

        if field_sharding is not None:  # pragma: no cover - hardware gate
            initialize_flux = jax.jit(
                initialize_flux,
                in_shardings=(field_sharding,) * 4,
                out_shardings=(field_sharding,) * 3 + (replicated_sharding,),
            )
            initialize_flux = _reuse_fringing_jit(("initialize_flux", *kernel_key), initialize_flux)
            momentum_solve = jax.jit(
                momentum_solve,
                in_shardings=(vector_sharding,) * 2
                + (field_sharding,) * 2
                + (flux_sharding, replicated_sharding),
                out_shardings=(vector_sharding, replicated_sharding, replicated_sharding),
            )
            momentum_solve = _reuse_fringing_jit(("momentum", *kernel_key), momentum_solve)
            momentum_defect = jax.jit(
                momentum_defect,
                in_shardings=(vector_sharding,) * 2
                + (field_sharding,) * 2
                + (flux_sharding, replicated_sharding, field_sharding),
                out_shardings=replicated_sharding,
            )
            momentum_defect = _reuse_fringing_jit(("momentum_defect", *kernel_key), momentum_defect)
            embed_velocity = jax.jit(
                embed_velocity,
                in_shardings=(vector_sharding, field_sharding),
                out_shardings=(field_sharding,) * 3,
            )
            embed_velocity = _reuse_fringing_jit(("embed_velocity", *kernel_key), embed_velocity)
            courant_numbers = jax.jit(
                courant_numbers,
                in_shardings=(field_sharding,) * 3 + (replicated_sharding, field_sharding),
                out_shardings=(replicated_sharding, replicated_sharding),
            )
            courant_numbers = _reuse_fringing_jit(("courant", *kernel_key), courant_numbers)
            pack_flux = jax.jit(pack_flux, in_shardings=(field_sharding,) * 3, out_shardings=flux_sharding)
            pack_flux = _reuse_fringing_jit(("pack_flux", *kernel_key), pack_flux)
            unpack_flux = jax.jit(
                unpack_flux, in_shardings=flux_sharding, out_shardings=(field_sharding,) * 3
            )
            unpack_flux = _reuse_fringing_jit(("unpack_flux", *kernel_key), unpack_flux)
            pack_vector = jax.jit(
                pack_vector, in_shardings=(field_sharding,) * 3, out_shardings=vector_sharding
            )
            pack_vector = _reuse_fringing_jit(("pack_vector", *kernel_key), pack_vector)
            relax_flux = jax.jit(
                relax_flux,
                in_shardings=(field_sharding,) * 3
                + (replicated_sharding,)
                + (field_sharding,) * 3
                + (replicated_sharding,) * 2,
                out_shardings=(field_sharding,) * 3 + (replicated_sharding,),
            )
            relax_flux = _reuse_fringing_jit(("relax_flux", *kernel_key), relax_flux)

        if restart_flux is None:
            (*current_flux_components, current_rho_phi_inlet) = initialize_flux(u, v, w, rho)
        else:
            restart_flux = np.asarray(restart_flux, dtype=np.dtype(u.dtype))
            current_flux_components = tuple(
                jnp.asarray(value) if field_sharding is None else jax.device_put(value, field_sharding)
                for value in restart_flux
            )
            current_rho_phi_inlet = jnp.asarray(restart_inlet, dtype=u.dtype)
            if replicated_sharding is not None:
                current_rho_phi_inlet = jax.device_put(np.asarray(current_rho_phi_inlet), replicated_sharding)
        current_rho_phi_plus = pack_flux(*current_flux_components)
        if previous_anderson_flux is not None:
            if (
                previous_anderson_flux.shape != current_rho_phi_plus.shape
                or previous_anderson_inlet.shape != current_rho_phi_inlet.shape
            ):
                raise ValueError("B2 restart Anderson flux state has inconsistent shape")
            if flux_sharding is not None:  # pragma: no cover - hardware gate
                previous_anderson_flux = jax.device_put(np.asarray(previous_anderson_flux), flux_sharding)
                previous_anderson_inlet = jax.device_put(
                    np.asarray(previous_anderson_inlet), replicated_sharding
                )

    else:
        unit_pressure_response = _enforce_velocity_bc_3d(jnp.where(fluid_mask, dt / rho, 0.0), fluid_mask)

    if use_alex_b2_finite_volume:
        # The strict corner bound oversolves PCG; the local residual gates physics.
        electric_volume_min = 4.0 * float(jnp.min(dy) * jnp.min(dz))
        # Transverse lines retain wall coupling; axial lines regress shard scaling.
        use_axial_line_preconditioner = False

        def mixed_boundary_projection(u0, v0, w0, pressure0, rho0, mask0):
            return _face_flux_pressure_projection_duct(
                u0,
                v0,
                w0,
                rho0,
                mask0,
                inlet_flow_rate=target_flow_rate,
                dt=dt,
                dx=dx,
                dy=dy,
                dz=dz,
                iterations=projection_iterations,
                tolerance=projection_tolerance,
                fluid_bounds=fluid_bounds,
                initial_pressure=pressure0,
                single_reduction=field_sharding is not None,
                include_axial_line=use_axial_line_preconditioner,
                field_sharding=field_sharding,
            )

        def electric_solve(rhs, initial, conductivity, mask):
            return _solvax_pressure_poisson_duct(
                rhs,
                conductivity,
                dx=dx,
                dy=dy,
                dz=dz,
                iterations=electric_iterations,
                tolerance=electric_tolerance,
                initial_field=initial,
                local_tolerance=ALEX_BALANCE_TOLERANCE,
                local_volume_min=electric_volume_min,
                single_reduction=field_sharding is not None,
                include_axial_line=use_axial_line_preconditioner,
                thin_wall_fluid_mask=mask,
                transverse_coarse_bounds=fluid_bounds,
                field_sharding=field_sharding,
            )

        def emf_operator(conductivity, emf_x, emf_y, emf_z, mask):
            return _conservative_emf_rhs_3d(
                conductivity,
                emf_x,
                emf_y,
                emf_z,
                dx=dx,
                dy=dy,
                dz=dz,
                thin_wall_fluid_mask=mask,
            )

        def reconstruct_electric(
            potential,
            conductivity,
            emf_x,
            emf_y,
            emf_z,
            field_x,
            field_y,
            field_z,
            mask,
        ):
            dphi_dx, dphi_dy, dphi_dz = _gradient_3d(potential, dx=dx, dy=dy, dz=dz)
            current_x = conductivity * (-dphi_dx + emf_x)
            current_y = conductivity * (-dphi_dy + emf_y)
            current_z = conductivity * (-dphi_dz + emf_z)
            divergence, _, _ = _conservative_current_diagnostics_3d(
                conductivity,
                potential,
                emf_x,
                emf_y,
                emf_z,
                dx=dx,
                dy=dy,
                dz=dz,
                thin_wall_fluid_mask=mask,
            )
            return (
                current_x,
                current_y,
                current_z,
                divergence,
                current_y * field_z - current_z * field_y,
                current_z * field_x - current_x * field_z,
                current_x * field_y - current_y * field_x,
            )

        def lorentz_operator(
            potential,
            conductivity,
            emf_x,
            emf_y,
            emf_z,
            field_x,
            field_y,
            field_z,
        ):
            dphi_dx, dphi_dy, dphi_dz = _gradient_3d(potential, dx=dx, dy=dy, dz=dz)
            current_x = conductivity * (-dphi_dx + emf_x)
            current_y = conductivity * (-dphi_dy + emf_y)
            current_z = conductivity * (-dphi_dz + emf_z)
            return (
                current_x,
                current_y,
                current_z,
                current_y * field_z - current_z * field_y,
                current_z * field_x - current_x * field_z,
                current_x * field_y - current_y * field_x,
            )

        def scaled_state(u0, v0, w0, potential0):
            return jnp.stack((u0, v0, w0, potential0)) / fixed_point_scale

        def state_difference(mapped, current):
            return mapped - current

        def unscaled_state(state):
            values = state * fixed_point_scale
            return values[0], values[1], values[2], values[3]

        def mix_anderson(mapped0, residual0, flux0, inlet0, mapped1, residual1, flux1, inlet1):
            """Apply one shared SOLVAX weight vector to the coupled B2 record."""

            weights = anderson_weights(
                jnp.stack((residual0, residual1)),
                regularization=case.solver.coupling_regularization,
            )
            damping = case.solver.coupling_damping

            def mix(previous, current):
                weighted = jnp.tensordot(weights, jnp.stack((previous, current)), axes=(0, 0))
                return current + damping * (weighted - current)

            return (
                mix(mapped0, mapped1),
                mix(flux0, flux1),
                mix(inlet0, inlet1),
            )

        mixed_boundary_projection = jax.named_call(mixed_boundary_projection, name="lmx.b2.projection")
        electric_solve = jax.named_call(electric_solve, name="lmx.b2.electric")
        emf_operator = jax.named_call(emf_operator, name="lmx.b2.emf")
        reconstruct_electric = jax.named_call(reconstruct_electric, name="lmx.b2.reconstruction")
        mix_anderson = jax.named_call(mix_anderson, name="lmx.b2.anderson")

        if field_sharding is not None:  # pragma: no cover - hardware gate
            axial_sharding = NamedSharding(field_sharding.mesh, P("x"))
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
            mixed_boundary_projection = jax.jit(
                mixed_boundary_projection,
                in_shardings=(field_sharding,) * 6,
                out_shardings=(
                    field_sharding,
                    field_sharding,
                    field_sharding,
                    field_sharding,
                    axial_sharding,
                    replicated_sharding,
                    replicated_sharding,
                    field_sharding,
                    field_sharding,
                    field_sharding,
                    replicated_sharding,
                    replicated_sharding,
                    replicated_sharding,
                    replicated_sharding,
                    replicated_sharding,
                    replicated_sharding,
                ),
            )
            electric_solve = jax.jit(
                electric_solve,
                in_shardings=(field_sharding,) * 4,
                out_shardings=(field_sharding,) + (replicated_sharding,) * 6,
            )
            emf_operator = jax.jit(
                emf_operator,
                in_shardings=(field_sharding,) * 5,
                out_shardings=field_sharding,
            )
            reconstruct_electric = jax.jit(
                reconstruct_electric,
                in_shardings=(field_sharding,) * 9,
                out_shardings=(field_sharding,) * 7,
            )
            lorentz_operator = jax.jit(
                lorentz_operator,
                in_shardings=(field_sharding,) * 8,
                out_shardings=(field_sharding,) * 6,
            )
            scaled_state = jax.jit(
                scaled_state,
                in_shardings=(field_sharding,) * 4,
                out_shardings=state_sharding,
            )
            state_difference = jax.jit(
                state_difference,
                in_shardings=(state_sharding, state_sharding),
                out_shardings=state_sharding,
            )
            unscaled_state = jax.jit(
                unscaled_state,
                in_shardings=state_sharding,
                out_shardings=(field_sharding,) * 4,
            )
            mix_anderson = jax.jit(
                mix_anderson,
                in_shardings=(state_sharding, state_sharding, flux_sharding, replicated_sharding) * 2,
                out_shardings=(state_sharding, flux_sharding, replicated_sharding),
            )
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
            ) = tuple(
                _reuse_fringing_jit((name, *kernel_key), function)
                for name, function in (
                    ("mixed_boundary", mixed_boundary_projection),
                    ("electric", electric_solve),
                    ("emf", emf_operator),
                    ("reconstruct", reconstruct_electric),
                    ("lorentz", lorentz_operator),
                    ("scale_state", scaled_state),
                    ("state_difference", state_difference),
                    ("unscale_state", unscaled_state),
                    ("anderson", mix_anderson),
                )
            )
        momentum_solve = _synchronized_phase(momentum_solve, "momentum", phase_timing_callback)
        momentum_defect = _synchronized_phase(momentum_defect, "momentum_defect", phase_timing_callback)
        mixed_boundary_projection = _synchronized_phase(
            mixed_boundary_projection, "projection", phase_timing_callback
        )
        electric_solve = _synchronized_phase(electric_solve, "electric", phase_timing_callback)
        emf_operator = _synchronized_phase(emf_operator, "emf", phase_timing_callback)
        reconstruct_electric = _synchronized_phase(
            reconstruct_electric, "reconstruction", phase_timing_callback
        )
        mix_anderson = _synchronized_phase(mix_anderson, "anderson", phase_timing_callback)

    stop_step = completed_steps + outer_steps
    for step in range(completed_steps, stop_step):
        flux_relaxation = jnp.asarray(1.0, dtype=u.dtype)
        step_courant = (
            courant_numbers(*current_flux_components, current_rho_phi_inlet, rho)
            if use_alex_b2_finite_volume
            else (-1.0, -1.0)
        )
        phi_previous = phi
        pressure_observable_previous = (
            _cross_duct_pressure_difference(p, active_mask=fluid_mask, magnetic_axis=1, side_axis=2)
            if use_alex_b2_finite_volume
            else jnp.zeros((nx,), dtype=p.dtype)
        )
        uxb_x = v * bz - w * by
        uxb_y = w * bx - u * bz
        uxb_z = u * by - v * bx
        pressure_linear_residual = jnp.asarray(jnp.nan)
        pressure_linear_relative_residual = jnp.asarray(jnp.nan)
        pressure_linear_iterations = jnp.asarray(0)
        pressure_linear_converged = jnp.asarray(False)
        pressure_linear_status = jnp.asarray(-1)
        momentum_linear_converged = jnp.asarray(True)
        if use_alex_b2_finite_volume:
            jx, jy, jz, lorentz_x, lorentz_y, lorentz_z = lorentz_operator(
                phi, sigma, uxb_x, uxb_y, uxb_z, bx, by, bz
            )
        else:
            dphi_dx, dphi_dy, dphi_dz = _gradient_3d(phi, dx=dx, dy=dy, dz=dz)
            jx = sigma * (-dphi_dx + uxb_x)
            jy = sigma * (-dphi_dy + uxb_y)
            jz = sigma * (-dphi_dz + uxb_z)
            lorentz_x = jy * bz - jz * by
            lorentz_y = jz * bx - jx * bz
            lorentz_z = jx * by - jy * bx

        if not use_alex_b2_finite_volume:
            dp_dx, dp_dy, dp_dz = _gradient_3d(p, dx=dx, dy=dy_momentum, dz=dz_momentum)
            laplacian_u = _laplacian_3d(u, dx=dx, dy=dy_momentum, dz=dz_momentum)
            laplacian_v = _laplacian_3d(v, dx=dx, dy=dy_momentum, dz=dz_momentum)
            laplacian_w = _laplacian_3d(w, dx=dx, dy=dy_momentum, dz=dz_momentum)
        if use_alex_b2_finite_volume:
            velocity = pack_vector(u, v, w)
            momentum_force = pack_vector(lorentz_x + forcing, lorentz_y, lorentz_z)
            velocity_fluid, _, momentum_linear_converged = momentum_solve(
                velocity, momentum_force, rho, nu, current_rho_phi_plus, current_rho_phi_inlet
            )
            u_star, v_star, w_star = embed_velocity(velocity_fluid, fluid_mask)
        else:
            u_star = u + dt * (nu * laplacian_u + forcing / rho + lorentz_x / rho - dp_dx / rho)
            v_star = v + dt * (nu * laplacian_v + lorentz_y / rho - dp_dy / rho)
            w_star = w + dt * (nu * laplacian_w + lorentz_z / rho - dp_dz / rho)
        if not use_alex_b2_finite_volume:
            u_star = _clip_state(u_star, velocity_limit)
            v_star = _clip_state(v_star, velocity_limit)
            w_star = _clip_state(w_star, velocity_limit)
            u_star = _enforce_velocity_bc_3d(u_star, fluid_mask)
            v_star = _enforce_velocity_bc_3d(v_star, fluid_mask)
            w_star = _enforce_velocity_bc_3d(w_star, fluid_mask)

        if use_alex_b2_finite_volume:
            (
                u_next,
                v_next,
                w_next,
                p_corr,
                axial_pressure_loss_gradient,
                projected_divergence_norm,
                fixed_flow_error,
                mapped_rho_phi_x,
                mapped_rho_phi_y,
                mapped_rho_phi_z,
                mapped_rho_phi_inlet,
                pressure_linear_residual,
                pressure_linear_relative_residual,
                pressure_linear_iterations,
                pressure_linear_converged,
                pressure_linear_status,
            ) = mixed_boundary_projection(u_star, v_star, w_star, p, rho, fluid_mask)
            mapped_flux_components = (mapped_rho_phi_x, mapped_rho_phi_y, mapped_rho_phi_z)
            valid = all(bool(jnp.all(jnp.isfinite(field))) for field in (u_next, v_next, w_next, p_corr))
            valid &= all(
                bool(jnp.all(jnp.abs(field) <= velocity_limit)) for field in (u_next, v_next, w_next)
            )
            if not valid:
                state = tuple(
                    (name, bool(jnp.all(jnp.isfinite(field))), float(jnp.nanmax(jnp.abs(field))))
                    for name, field in zip(("u", "v", "w", "p"), (u_next, v_next, w_next, p_corr))
                )
                raise FloatingPointError(f"ALEX B2 projection inactive guard: {state}")
        else:
            du_dx, _, _ = _gradient_3d(u_star, dx=dx, dy=dy_momentum, dz=dz_momentum)
            _, dv_dy, _ = _gradient_3d(v_star, dx=dx, dy=dy_momentum, dz=dz_momentum)
            _, _, dw_dz = _gradient_3d(w_star, dx=dx, dy=dy_momentum, dz=dz_momentum)
            divergence = jnp.where(fluid_mask, du_dx + dv_dy + dw_dz, 0.0)
            p_corr, _, _, _ = _poisson_jacobi_3d(
                (rho / max(dt, 1.0e-12)) * divergence,
                dx=dx,
                dy=dy_momentum,
                dz=dz_momentum,
                iterations=poisson_iterations,
                tolerance=poisson_tolerance,
            )
            p_corr = _clip_state(jnp.where(fluid_mask, p_corr, 0.0), scalar_limit)
            dpc_dx, dpc_dy, dpc_dz = _gradient_3d(p_corr, dx=dx, dy=dy_momentum, dz=dz_momentum)
            u_next = _enforce_velocity_bc_3d(u_star - (dt / rho) * dpc_dx, fluid_mask)
            v_next = _enforce_velocity_bc_3d(v_star - (dt / rho) * dpc_dy, fluid_mask)
            w_next = _enforce_velocity_bc_3d(w_star - (dt / rho) * dpc_dz, fluid_mask)
            u_next = _clip_state(u_next, velocity_limit)
            v_next = _clip_state(v_next, velocity_limit)
            w_next = _clip_state(w_next, velocity_limit)
            if target_flow_rate is None:
                u_next = _enforce_stationwise_flow_rate_3d(
                    u_next,
                    active_mask=fluid_mask,
                    cell_area=cell_area,
                    relaxation=0.6 if case.geometry.kind == "layered_duct" else 0.0,
                )
                axial_pressure_loss_gradient = jnp.full((nx,), forcing, dtype=float)
            else:
                u_next, axial_pressure_loss_gradient = _apply_fixed_flow_pressure_constraint(
                    u_next,
                    unit_pressure_response=unit_pressure_response,
                    active_mask=fluid_mask,
                    cell_area=cell_area,
                    target_flow_rate=target_flow_rate,
                    base_pressure_loss_gradient=forcing,
                )
            u_next = _enforce_velocity_bc_3d(u_next, fluid_mask)
            projected_divergence_norm = float("nan")
            fixed_flow_error = 0.0
        p = jnp.where(
            fluid_mask,
            p_corr if use_alex_b2_finite_volume else _clip_state(p + p_corr, scalar_limit),
            0.0,
        )

        uxb_x = v_next * bz - w_next * by
        uxb_y = w_next * bx - u_next * bz
        uxb_z = u_next * by - v_next * bx
        emf_rhs = (
            emf_operator(sigma, uxb_x, uxb_y, uxb_z, fluid_mask)
            if use_alex_b2_finite_volume
            else _conservative_emf_rhs_3d(
                sigma,
                uxb_x,
                uxb_y,
                uxb_z,
                dx=dx,
                dy=dy,
                dz=dz,
            )
        )
        if use_alex_b2_finite_volume:
            (
                phi,
                electric_residual,
                electric_converged,
                electric_relative_residual,
                electric_iteration_count,
                electric_status,
                electric_local_residual,
            ) = electric_solve(emf_rhs, phi, sigma, fluid_mask)
        else:
            electric_solver = (
                _variable_coefficient_poisson_sparse_3d
                if case.geometry.kind in {"rect_duct", "layered_duct"}
                else _variable_coefficient_poisson_jacobi_3d
            )
            phi, _, _, _ = electric_solver(
                emf_rhs,
                sigma,
                dx=dx,
                dy=dy,
                dz=dz,
                iterations=poisson_iterations,
                tolerance=poisson_tolerance,
                initial_field=phi,
            )
            electric_residual = jnp.asarray(jnp.nan)
            electric_relative_residual = jnp.asarray(jnp.nan)
            electric_iteration_count = jnp.asarray(0)
            electric_converged = jnp.asarray(False)
            electric_status = jnp.asarray(-1)
            electric_local_residual = jnp.asarray(jnp.nan)
        if use_alex_b2_finite_volume and not bool(jnp.all(jnp.isfinite(phi))):
            raise FloatingPointError("ALEX B2 electric solve produced non-finite potential")
        phi = phi if use_alex_b2_finite_volume else _clip_state(phi, scalar_limit)
        potential_update = _gauge_invariant_scalar_update(
            phi,
            phi_previous,
            cell_area,
            scale=electric_potential_scale,
        )

        if use_alex_b2_finite_volume:
            jx, jy, jz, div_j, lorentz_x, lorentz_y, lorentz_z = reconstruct_electric(
                phi, sigma, uxb_x, uxb_y, uxb_z, bx, by, bz, fluid_mask
            )
            momentum_defect_components = momentum_defect(
                pack_vector(u_next, v_next, w_next),
                pack_vector(lorentz_x, lorentz_y, lorentz_z),
                rho,
                nu,
                pack_flux(*mapped_flux_components),
                mapped_rho_phi_inlet,
                p_corr,
            )
        else:
            dphi_dx, dphi_dy, dphi_dz = _gradient_3d(phi, dx=dx, dy=dy, dz=dz)
            jx = _clip_state(sigma * (-dphi_dx + uxb_x), scalar_limit)
            jy = _clip_state(sigma * (-dphi_dy + uxb_y), scalar_limit)
            jz = _clip_state(sigma * (-dphi_dz + uxb_z), scalar_limit)
            div_j, _, _ = _conservative_current_diagnostics_3d(
                sigma,
                phi,
                uxb_x,
                uxb_y,
                uxb_z,
                dx=dx,
                dy=dy,
                dz=dz,
            )
            lorentz_x = jy * bz - jz * by
            lorentz_y = jz * bx - jx * bz
            lorentz_z = jx * by - jy * bx
            momentum_defect_components = jnp.full((4,), jnp.nan)

        if use_alex_b2_finite_volume:
            projected_divergence_max = projected_divergence_norm
        else:
            du_dx, _, _ = _gradient_3d(u_next, dx=dx, dy=dy_momentum, dz=dz_momentum)
            _, dv_dy, _ = _gradient_3d(v_next, dx=dx, dy=dy_momentum, dz=dz_momentum)
            _, _, dw_dz = _gradient_3d(w_next, dx=dx, dy=dy_momentum, dz=dz_momentum)
            projected_divergence = jnp.where(fluid_mask, du_dx + dv_dy + dw_dz, 0.0)
            projected_divergence_max = jnp.max(jnp.abs(projected_divergence))

        pressure_update = (
            _normalized_pressure_observable_update(
                _cross_duct_pressure_difference(
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
                    pressure_linear_residual,
                    pressure_linear_relative_residual,
                    pressure_linear_iterations,
                    pressure_linear_converged,
                    pressure_linear_status,
                    electric_residual,
                    electric_relative_residual,
                    electric_local_residual,
                    electric_iteration_count,
                    electric_converged,
                    electric_status,
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
            max(u_update, v_update, w_update) / (inverse_electromagnetic_scale * dt)
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
            and projected_divergence_max <= ALEX_BALANCE_TOLERANCE
            and flow_error_value <= ALEX_BALANCE_TOLERANCE
            and charge_balance <= ALEX_BALANCE_TOLERANCE
            and (
                not use_alex_b2_finite_volume
                or all(map(bool, (momentum_linear_converged, pressure_linear_converged, electric_converged)))
            )
        )
        accepted_state_converged = (
            max(u_update, v_update, w_update, potential_update) <= case.time_stepper.steady_tolerance
        )
        if use_alex_b2_finite_volume:
            # Require repeated passing updates before accepting an oscillatory map.
            steady_streak = steady_streak + 1 if instantaneous_convergence else 0
            converged = steady_streak >= ALEX_B2_STEADY_STEPS
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
                    accelerated = current_state + ALEX_B2_SETTLED_RELAXATION * fixed_point_residual
                    previous_fixed_point_residual = None
                    fixed_point_relaxation = jnp.asarray(1.0, dtype=u.dtype)
                    flux_relaxation = ALEX_B2_SETTLED_RELAXATION
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
                # Preserve component placement across older GPU JAX releases.
                current_flux_components = unpack_flux(current_rho_phi_plus)
                current_rho_phi_inlet = mapped_rho_phi_inlet if converged else accelerated_inlet
            else:
                (*current_flux_components, current_rho_phi_inlet) = relax_flux(
                    *current_flux_components,
                    current_rho_phi_inlet,
                    *mapped_flux_components,
                    mapped_rho_phi_inlet,
                    flux_relaxation,
                )
                current_rho_phi_plus = pack_flux(*current_flux_components)
        _emit_iteration_progress(
            progress_callback,
            checkpoint_interval=checkpoint_interval,
            step=step + 1,
            total_steps=stop_step,
            converged=converged,
            residual=update_residual,
            component_residuals=component_residual_by_step[-1],
            pressure_residual=pressure_update,
            potential_residual=potential_update,
            checkpoint_factory=lambda: _iteration_checkpoint_bundle(
                case=case,
                x=x,
                y=y,
                z=z,
                field_scale=field_scale,
                u=u,
                v=v,
                w=w,
                p=p,
                phi=phi,
                axial_pressure_loss_gradient=axial_pressure_loss_gradient,
                transverse_pressure_difference=None,
                residual_history=residual_by_step,
                component_history=component_residual_by_step,
                pressure_history=pressure_residual_by_step,
                electric_history=electric_linear_by_step,
                potential_history=potential_residual_by_step,
                pressure_linear_history=pressure_linear_by_step,
                rho_phi_plus=(current_rho_phi_plus if use_alex_b2_finite_volume else None),
                rho_phi_inlet=(current_rho_phi_inlet if use_alex_b2_finite_volume else None),
                aitken_state=(
                    (previous_fixed_point_residual, fixed_point_relaxation, steady_streak)
                    if use_alex_b2_finite_volume and case.solver.coupling_acceleration == "aitken"
                    else None
                ),
                anderson_state=(
                    (
                        previous_anderson_mapped,
                        previous_anderson_residual,
                        previous_anderson_flux,
                        previous_anderson_inlet,
                    )
                    if use_alex_b2_finite_volume and case.solver.coupling_acceleration == "anderson"
                    else None
                ),
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

    if use_alex_b2_finite_volume:
        # Acceleration changes the accepted state after current reconstruction.
        uxb_x = v * bz - w * by
        uxb_y = w * bx - u * bz
        uxb_z = u * by - v * bx
        jx, jy, jz, div_j, lorentz_x, lorentz_y, lorentz_z = reconstruct_electric(
            phi, sigma, uxb_x, uxb_y, uxb_z, bx, by, bz, fluid_mask
        )

    final_step_residual = residual_by_step[-1] if residual_by_step else 0.0
    residual = jnp.full((nx,), final_step_residual, dtype=float)
    fluid_area = jnp.maximum(jnp.sum(jnp.where(fluid_mask, cell_area, 0.0), axis=(1, 2)), 1.0e-20)
    volumetric_flow_rate = jnp.sum(jnp.where(fluid_mask, u * cell_area, 0.0), axis=(1, 2))
    mean_velocity = volumetric_flow_rate / fluid_area
    fx, _, _ = _conservative_current_fluxes_3d(
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
    axial_current = _station_axial_current_from_fluxes(fx, cell_area[0])
    if use_alex_b2_finite_volume:
        wall_current_leakage = jnp.zeros((nx,), dtype=div_j.dtype)
        boundary_current_residual = jnp.abs(jnp.sum(div_j * cell_area, axis=(1, 2)) * dx)
    else:
        div_j, wall_current_leakage, boundary_current_residual = _conservative_current_diagnostics_3d(
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
    transverse_pressure_difference = _cross_duct_pressure_difference(
        p, active_mask=fluid_mask, magnetic_axis=1, side_axis=2
    )
    return ExtrudedFieldBundle(
        x=x,
        y=y,
        z=z,
        field_scale=field_scale,
        u=u,
        v=v,
        w=w,
        p=p,
        phi=phi,
        rho_phi_plus=(current_rho_phi_plus if use_alex_b2_finite_volume else None),
        rho_phi_inlet=(current_rho_phi_inlet if use_alex_b2_finite_volume else None),
        aitken_state=(
            (previous_fixed_point_residual, fixed_point_relaxation, steady_streak)
            if use_alex_b2_finite_volume and case.solver.coupling_acceleration == "aitken"
            else None
        ),
        anderson_state=(
            (
                previous_anderson_mapped,
                previous_anderson_residual,
                previous_anderson_flux,
                previous_anderson_inlet,
            )
            if use_alex_b2_finite_volume and case.solver.coupling_acceleration == "anderson"
            else None
        ),
        stopping_state=(len(residual_by_step), steady_streak, "converged" if converged else "step_limit"),
        jx=jx,
        jy=jy,
        jz=jz,
        lorentz_x=lorentz_x,
        lorentz_y=lorentz_y,
        lorentz_z=lorentz_z,
        residual=jnp.asarray(residual, dtype=float),
        volumetric_flow_rate=jnp.asarray(volumetric_flow_rate, dtype=float),
        mean_velocity=jnp.asarray(mean_velocity, dtype=float),
        axial_current=jnp.asarray(axial_current, dtype=float),
        wall_current_leakage=jnp.asarray(wall_current_leakage, dtype=float),
        current_scaled_pressure_proxy=jnp.asarray(current_scaled_pressure_proxy, dtype=float),
        charge_balance_residual=jnp.asarray(charge_balance_residual, dtype=float),
        boundary_current_residual=jnp.asarray(boundary_current_residual, dtype=float),
        geometry_kind=case.geometry.kind,
        solver_kind=case.solver.kind,
        axial_pressure_loss_gradient=jnp.asarray(axial_pressure_loss_gradient, dtype=float),
        transverse_pressure_difference=jnp.asarray(transverse_pressure_difference, dtype=float),
        **_iteration_history_arrays(
            residual_by_step,
            component_residual_by_step,
            pressure_residual_by_step,
            electric_linear_by_step,
            potential_residual_by_step,
            courant_by_step,
            pressure_linear=pressure_linear_by_step,
            momentum_defect=(momentum_defect_by_step if use_alex_b2_finite_volume else None),
        ),
    )

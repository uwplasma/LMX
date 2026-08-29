"""Private rectangular-duct momentum and conservative-current kernels."""

from __future__ import annotations

import math
from collections.abc import Callable
from operator import sub

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from solvax import (
    additive_tridiagonal_line_preconditioner,
    anderson_weights,
    gmres,
    linear_solve,
    pcg_linear_solve,
)

from ._fringing_common import (
    _MIXED_AXIAL_PRESSURE_MODE,
    ALEX_B2_ANDERSON_CONDITION_LIMIT,
    ALEX_B2_PRESSURE_RELAXATION,
    ALEX_BALANCE_TOLERANCE,
    _axial_mean_preconditioner_3d,
    _coerce_spacing_vector,
    _cross,
    _distance_weighted_harmonic_mean,
    _finalize_local_pressure_solve,
    _gradient_3d,
    _harmonic_mean,
    _neighbor_fields,
    _rectangular_fluid_bounds,
    _reuse_fringing_jit,
    _spacing_vector,
    _thin_wall_interface_mean,
    _transverse_modal_correction_3d,
    _variable_diffusion_coefficients_3d,
)
from .specs import (
    CaseSpec,
    ExtrudedFieldBundle,
)


def _solvax_pressure_poisson_duct(
    rhs: jnp.ndarray,
    mobility: jnp.ndarray,
    *,
    dx: float,
    dy: jnp.ndarray,
    dz: jnp.ndarray,
    iterations: int,
    tolerance: float,
    initial_field: jnp.ndarray | None = None,
    local_tolerance: float | None = None,
    local_volume_min: float | None = None,
    single_reduction: bool = False,
    include_axial_line: bool = True,
    thin_wall_fluid_mask: jnp.ndarray | None = None,
    transverse_coarse_bounds: tuple[int, int, int, int] | None = None,
    field_sharding: NamedSharding | None = None,
    axial_pressure_mode: str = "neumann",
) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
]:
    """Solve the finite-volume pressure equation with implicit PCG.

    The default all-Neumann system retains its compatible RHS and symmetric
    rank-one gauge.  The private mixed mode instead applies zero pressure at
    the outlet face and zero pressure gradient at the inlet.
    """

    if axial_pressure_mode not in {"neumann", _MIXED_AXIAL_PRESSURE_MODE}:
        raise ValueError(f"Unsupported axial pressure mode {axial_pressure_mode!r}")
    mixed_axial_pressure = axial_pressure_mode == _MIXED_AXIAL_PRESSURE_MODE
    if mixed_axial_pressure and transverse_coarse_bounds is not None:
        raise ValueError("Mixed axial pressure does not support the Neumann coarse correction")

    dy_widths = _coerce_spacing_vector(dy, rhs.shape[1], dtype=rhs.dtype)
    dz_widths = _coerce_spacing_vector(dz, rhs.shape[2], dtype=rhs.dtype)
    volume = jnp.broadcast_to(dy_widths[None, :, None] * dz_widths[None, None, :], rhs.shape)
    volume_sum = jnp.sum(volume)
    solved_rhs = rhs if mixed_axial_pressure else rhs - jnp.sum(rhs * volume) / volume_sum
    coefficients = _variable_diffusion_coefficients_3d(
        mobility,
        dx=dx,
        dy=dy_widths,
        dz=dz_widths,
        validated_spacing=True,
        thin_wall_fluid_mask=thin_wall_fluid_mask,
    )
    coef_x_w, coef_x_e, coef_y_s, coef_y_n, coef_z_b, coef_z_t = coefficients
    if mixed_axial_pressure:
        outlet = jnp.arange(rhs.shape[0])[:, None, None] == rhs.shape[0] - 1
        coef_x_e = jnp.where(outlet, 2.0 * mobility[-1] / jnp.maximum(dx**2, 1.0e-12), coef_x_e)
        coefficients = (
            coef_x_w,
            coef_x_e,
            coef_y_s,
            coef_y_n,
            coef_z_b,
            coef_z_t,
        )

    def diffusion_operator(field: jnp.ndarray) -> jnp.ndarray:
        x_west, x_east, y_south, y_north, z_bottom, z_top = _neighbor_fields(
            field,
            mode_x="neumann",
            mode_y="neumann",
            mode_z="neumann",
            sharding=field_sharding,
        )
        if mixed_axial_pressure:
            x_east = jnp.where(outlet, 0.0, x_east)
        return (
            coef_x_w * (x_west - field)
            + coef_x_e * (x_east - field)
            + coef_y_s * (y_south - field)
            + coef_y_n * (y_north - field)
            + coef_z_b * (z_bottom - field)
            + coef_z_t * (z_top - field)
        )

    def matvec(field: jnp.ndarray) -> jnp.ndarray:
        diffusion = diffusion_operator(field)
        if mixed_axial_pressure:
            return -volume * diffusion
        gauge_term = volume * jnp.sum(volume * field) / volume_sum
        return -volume * diffusion + gauge_term

    diagonal = volume * sum(coefficients)
    if not mixed_axial_pressure:
        diagonal = diagonal + volume**2 / volume_sum

    directions = (
        (0, -volume * coef_x_w, -volume * coef_x_e),
        (1, -volume * coef_y_s, -volume * coef_y_n),
        (2, -volume * coef_z_b, -volume * coef_z_t),
    )
    if not include_axial_line:
        directions = directions[1:]
    line_precondition = additive_tridiagonal_line_preconditioner(diagonal, directions)
    axial_precondition = _axial_mean_preconditioner_3d(
        volume,
        coef_x_w,
        coef_x_e,
        gauge=not mixed_axial_pressure,
        field_sharding=field_sharding,
    )

    def precondition(residual):
        return line_precondition(residual) + axial_precondition(residual)

    if transverse_coarse_bounds is not None:
        fluid_cells = min(
            transverse_coarse_bounds[1] - transverse_coarse_bounds[0],
            transverse_coarse_bounds[3] - transverse_coarse_bounds[2],
        )
        coarse_correction = _transverse_modal_correction_3d(
            volume,
            mobility,
            coefficients,
            dx=dx,
            dy=dy_widths,
            dz=dz_widths,
            fluid_bounds=transverse_coarse_bounds,
            stride=max(2, math.ceil(fluid_cells / 13)),
            sharding=field_sharding,
        )

        def precondition(residual):
            return line_precondition(residual) + axial_precondition(residual) + coarse_correction(residual)

    linear_rhs = -volume * solved_rhs
    effective_rtol = tolerance
    effective_atol = tolerance
    if local_tolerance is not None:
        volume_min = float(jnp.min(volume)) if local_volume_min is None else local_volume_min
        local_absolute_target = volume_min * local_tolerance
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
        transpose_rtol=max(tolerance, 1.0e-12),
        transpose_atol=max(tolerance, 1.0e-12),
        transpose_max_steps=iterations,
        single_reduction=single_reduction,
    )
    return _finalize_local_pressure_solve(
        solution,
        linear_rhs=linear_rhs,
        matvec=matvec,
        local_residual_fn=lambda field: diffusion_operator(field) - solved_rhs,
        volume=volume,
        precondition=precondition,
        iterations=iterations,
        effective_atol=effective_atol,
        local_tolerance=local_tolerance,
        single_reduction=single_reduction,
        gauge=not mixed_axial_pressure,
    )


def _solvax_implicit_momentum_duct(
    velocity: jnp.ndarray,
    force: jnp.ndarray,
    density: jnp.ndarray,
    viscosity: jnp.ndarray,
    rho_phi: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    boundary_velocity: tuple[jnp.ndarray, ...],
    *,
    dt: float,
    dx: float,
    dy: jnp.ndarray,
    dz: jnp.ndarray,
    iterations: int,
    tolerance: float,
    reaction: jnp.ndarray,
    frozen_setup=None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Solve one frozen, conservative three-component momentum system.

    ``force`` includes explicit deviatoric stresses and body forces. The
    positive ``reaction`` is added to both sides at the old velocity, changing
    pseudo-time conditioning without changing a fixed point. The inlet is
    prescribed, the outlet is zero-gradient, and affine terms stay outside GMRES.
    """

    shape = velocity.shape
    reaction = jnp.asarray(reaction)
    if shape != (*density.shape, 3) or force.shape != shape or reaction.shape != density.shape:
        raise ValueError("Momentum fields must share one (nx, ny, nz, 3) shape")
    dy_widths = _coerce_spacing_vector(dy, shape[1], dtype=velocity.dtype)
    dz_widths = _coerce_spacing_vector(dz, shape[2], dtype=velocity.dtype)
    dx_widths = jnp.full((shape[0],), dx, dtype=velocity.dtype)
    volume = dx_widths[:, None, None] * dy_widths[None, :, None] * dz_widths[None, None, :]
    widths = (dx_widths, dy_widths, dz_widths)
    setup = frozen_setup or _frozen_duct_momentum_setup(
        velocity,
        density,
        viscosity,
        rho_phi,
        boundary_velocity,
        widths,
        dx=dx,
    )
    _, coefficients, diffusion_sink, inlet_sink, weights, _ = setup
    inlet_cells = jnp.arange(shape[0])[:, None, None] == 0
    zero_patches = tuple(jnp.zeros_like(value) for value in boundary_velocity)
    prescribed_patches = (boundary_velocity[0], zero_patches[1], *boundary_velocity[2:])
    boundary_action, _ = _duct_momentum_transport(
        jnp.zeros_like(velocity), rho_phi, weights, prescribed_patches, widths, coefficients, diffusion_sink
    )

    def matvec(field: jnp.ndarray) -> jnp.ndarray:
        homogeneous_patches = (zero_patches[0], field[-1], *zero_patches[2:])
        convection, diffusion = _duct_momentum_transport(
            field, rho_phi, weights, homogeneous_patches, widths, coefficients, diffusion_sink
        )
        return volume[..., None] * (
            density[..., None] * field + dt * (convection - diffusion + reaction[..., None] * field)
        )

    diagonal = volume * (density + dt * (reaction + sum(coefficients) + diffusion_sink))

    def precondition(flat: jnp.ndarray) -> jnp.ndarray:
        # The transient mass term dominates this operator; diagonal scaling
        # avoids thousands of GPU line solves while GMRES certifies the result.
        return (flat.reshape(shape) / diagonal[..., None]).reshape(-1)

    inlet_velocity = jnp.where(inlet_cells[..., None], boundary_velocity[0], 0.0)
    source = force - boundary_action + inlet_sink[..., None] * inlet_velocity + reaction[..., None] * velocity
    linear_rhs = volume[..., None] * (density[..., None] * velocity + dt * source)
    flat_rhs = linear_rhs.reshape(-1)

    def flat_matvec(flat: jnp.ndarray) -> jnp.ndarray:
        return matvec(flat.reshape(shape)).reshape(-1)

    restart = min(12, flat_rhs.size)

    def krylov(operator, rhs):
        solution = gmres(
            operator,
            rhs,
            x0=jnp.zeros_like(rhs),
            precond=precondition,
            restart=restart,
            rtol=tolerance,
            atol=tolerance,
            max_restarts=max(1, math.ceil(iterations / restart)),
        )
        return solution.x, (solution.residual_norm, solution.converged)

    solved, (residual, converged) = linear_solve(flat_matvec, flat_rhs, krylov, has_aux=True)
    return solved.reshape(shape), residual, converged


def _cell_limited_least_squares_gradient_duct(
    field: jnp.ndarray,
    boundary_values: tuple[jnp.ndarray, ...],
    widths: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
) -> tuple[jnp.ndarray, ...]:
    """Return cell-limited least-squares gradients."""
    gradients, neighbours = [], []
    for axis, width in enumerate(widths):
        values = jnp.moveaxis(field, axis, 0)
        lo = jnp.concatenate((boundary_values[2 * axis][None], values[:-1]))
        hi = jnp.concatenate((values[1:], boundary_values[2 * axis + 1][None]))
        centers = 0.5 * (width[:-1] + width[1:])
        trailing = (None,) * (values.ndim - 1)
        dm = jnp.concatenate((0.5 * width[:1], centers))[(slice(None), *trailing)]
        dp = jnp.concatenate((centers, 0.5 * width[-1:]))[(slice(None), *trailing)]
        fraction = width[1:] / (width[:-1] + width[1:])
        lo_weight = jnp.concatenate((jnp.ones(1), fraction))[(slice(None), *trailing)]
        hi_weight = jnp.concatenate((1.0 - fraction, jnp.ones(1)))[(slice(None), *trailing)]
        gradient = (lo_weight * (values - lo) / dm + hi_weight * (hi - values) / dp) / (lo_weight + hi_weight)
        gradients.append(jnp.moveaxis(gradient, 0, axis))
        neighbours.extend((jnp.moveaxis(lo, 0, axis), jnp.moveaxis(hi, 0, axis)))
    local = jnp.stack((field, *neighbours))
    minimum, maximum = jnp.min(local, axis=0), jnp.max(local, axis=0)
    limiter = jnp.ones_like(field)
    for axis, (gradient, width) in enumerate(zip(gradients, widths, strict=True)):
        shape = [1] * field.ndim
        shape[axis] = field.shape[axis]
        half_step = 0.5 * width.reshape(shape) * gradient
        for extrapolate in (-half_step, half_step):
            delta = jnp.where(extrapolate > 0.0, maximum - field, minimum - field)
            ratio = delta / jnp.where(jnp.abs(extrapolate) > 1.0e-15, extrapolate, 1.0)
            bounded = jnp.where(jnp.abs(extrapolate) > 1.0e-15, jnp.minimum(ratio, 1.0), 1.0)
            limiter = jnp.minimum(limiter, bounded)
    return tuple(limiter * gradient for gradient in gradients)


def _explicit_deviatoric_stress_duct(
    velocity: jnp.ndarray,
    dynamic_viscosity: jnp.ndarray,
    boundary_velocity: tuple[jnp.ndarray, ...],
    widths: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    *,
    gradient: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Return limited ``div(mu*dev2(T(grad(U))))``."""
    if gradient is None:
        gradient = jnp.stack(
            _cell_limited_least_squares_gradient_duct(velocity, boundary_velocity, widths), axis=-2
        )
    identity = jnp.eye(3, dtype=velocity.dtype)

    def traction(value, coefficient, axis):
        trace = jnp.trace(value, axis1=-2, axis2=-1)
        deviatoric = value[..., :, axis] - (2.0 / 3.0) * trace[..., None] * identity[axis]
        return coefficient[..., None] * deviatoric

    correction = jnp.zeros_like(velocity)
    for axis, width in enumerate(widths):
        cell_traction = traction(gradient, dynamic_viscosity, axis)
        patch_tractions = []
        for side, index in ((0, 0), (1, -1)):
            normal = (
                (2 * side - 1)
                * (boundary_velocity[2 * axis + side] - jnp.take(velocity, index, axis=axis))
                / (0.5 * width[index])
            )
            patch_tractions.append(
                traction(
                    jnp.take(gradient, index, axis=axis).at[..., axis, :].set(normal),
                    jnp.take(dynamic_viscosity, index, axis=axis),
                    axis,
                )
            )
        moved = jnp.moveaxis(cell_traction, axis, 0)
        centered = (width[1:] / (width[:-1] + width[1:]))[:, None, None, None]
        faces = jnp.concatenate(
            (
                patch_tractions[0][None],
                centered * moved[:-1] + (1.0 - centered) * moved[1:],
                patch_tractions[1][None],
            )
        )
        correction += jnp.moveaxis(jnp.diff(faces, axis=0) / width[:, None, None, None], 0, axis)
    return correction


def _limited_linear_vector_face_weights_duct(
    velocity: jnp.ndarray,
    rho_phi: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    boundary_velocity: tuple[jnp.ndarray, ...],
    widths: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    *,
    gradient: tuple[jnp.ndarray, ...] | None = None,
) -> tuple[jnp.ndarray, ...]:
    """Freeze v2206 vector weights, optionally from precomputed gradients."""
    q = jnp.sum(velocity**2, axis=-1)
    gradients = gradient
    if gradients is None:
        gradients = _cell_limited_least_squares_gradient_duct(
            q, tuple(jnp.sum(value**2, axis=-1) for value in boundary_velocity), widths
        )
    weights = []
    for axis, (width, gradient, face_flux) in enumerate(zip(widths, gradients, rho_phi, strict=True)):
        values = jnp.moveaxis(q, axis, 0)
        gradient = jnp.moveaxis(gradient, axis, 0)
        distance = 0.5 * (width[:-1] + width[1:])[:, None, None]
        internal_flux = jnp.moveaxis(face_flux, axis, 0)[1:-1]
        gradf = values[1:] - values[:-1]
        upwind_gradient = jnp.where(internal_flux > 0.0, gradient[:-1], gradient[1:])
        gradcf = upwind_gradient * distance
        guarded = 2000.0 * jnp.sign(gradcf) * jnp.sign(gradf) - 1.0
        ratio = 2.0 * gradcf / gradf - 1.0
        r = jnp.where(jnp.abs(gradcf) >= 1000.0 * jnp.abs(gradf), guarded, ratio)
        limited = jnp.clip(2.0 * r, 0.0, 1.0)
        centered = (width[1:] / (width[:-1] + width[1:]))[:, None, None]
        face_weight = limited * centered + (1.0 - limited) * (internal_flux >= 0.0)
        weights.append(jnp.moveaxis(face_weight, 0, axis))
    return tuple(weights)


def _limited_linear_convection_matrix_action_duct(
    field,
    rho_phi,
    weights,
    boundary_values,
    widths,
) -> jnp.ndarray:
    """Return the limited-linear convection divergence."""
    dx, dy, dz = widths
    volume = dx[:, None, None] * dy[None, :, None] * dz[None, None, :]
    action = jnp.zeros_like(field)
    for axis, (face_flux, weight) in enumerate(zip(rho_phi, weights, strict=True)):
        values = jnp.moveaxis(field, axis, 0)
        weight = jnp.moveaxis(weight, axis, 0)[..., None]
        interpolated = weight * values[:-1] + (1.0 - weight) * values[1:]
        faces = jnp.concatenate(
            (boundary_values[2 * axis][None], interpolated, boundary_values[2 * axis + 1][None])
        )
        divergence = jnp.diff(jnp.moveaxis(face_flux, axis, 0)[..., None] * faces, axis=0)
        action += jnp.moveaxis(divergence, 0, axis) / volume[..., None]
    return action


def _frozen_duct_momentum_setup(
    velocity,
    density,
    viscosity,
    rho_phi,
    boundary_velocity,
    widths,
    *,
    dx,
):
    """Own frozen diffusion, limiter, and packed velocity/q gradients."""
    q = jnp.sum(velocity**2, axis=-1)
    packed = jnp.concatenate((velocity, q[..., None]), axis=-1)
    packed_patches = tuple(
        jnp.concatenate((patch, jnp.sum(patch**2, axis=-1, keepdims=True)), axis=-1)
        for patch in boundary_velocity
    )
    gradients = _cell_limited_least_squares_gradient_duct(packed, packed_patches, widths)
    dynamic_viscosity = density * viscosity
    coefficients = _variable_diffusion_coefficients_3d(
        dynamic_viscosity, dx=dx, dy=widths[1], dz=widths[2], validated_spacing=True
    )
    wall_sink = jnp.zeros_like(density)
    for axis, width in ((1, widths[1]), (2, widths[2])):
        for index in (0, -1):
            cells = [slice(None)] * 3
            cells[axis] = index
            cells = tuple(cells)
            wall_sink = wall_sink.at[cells].add(dynamic_viscosity[cells] / (0.5 * width[index] ** 2))
    inlet_cells = jnp.arange(density.shape[0])[:, None, None] == 0
    inlet_sink = jnp.where(inlet_cells, 2.0 * dynamic_viscosity[0] / dx**2, 0.0)
    weights = _limited_linear_vector_face_weights_duct(
        velocity, rho_phi, boundary_velocity, widths, gradient=tuple(value[..., 3] for value in gradients)
    )
    velocity_gradient = jnp.stack(gradients, axis=-2)[..., :3]
    return (dynamic_viscosity, coefficients, wall_sink + inlet_sink, inlet_sink, weights, velocity_gradient)


def _duct_momentum_transport(field, rho_phi, weights, boundary_values, widths, coefficients, diffusion_sink):
    """Return the shared frozen convection and homogeneous diffusion actions."""
    neighbours = _neighbor_fields(field, mode_x="neumann", mode_y="neumann", mode_z="neumann")
    diffusion = (
        sum(c[..., None] * (n - field) for c, n in zip(coefficients, neighbours, strict=True))
        - diffusion_sink[..., None] * field
    )
    convection = _limited_linear_convection_matrix_action_duct(
        field, rho_phi, weights, boundary_values, widths
    )
    return convection, diffusion


def _unpack_duct_mass_flux(rho_phi_plus, rho_phi_inlet):
    """Restore full face fluxes; lower transverse walls are implicit zeros."""
    return (
        jnp.concatenate((rho_phi_inlet[None], rho_phi_plus[0]), axis=0),
        jnp.concatenate((jnp.zeros_like(rho_phi_plus[1, :, :1]), rho_phi_plus[1]), axis=1),
        jnp.concatenate((jnp.zeros_like(rho_phi_plus[2, :, :, :1]), rho_phi_plus[2]), axis=2),
    )


def _compact_duct_courant_numbers(
    rho_phi_plus,
    rho_phi_inlet,
    density,
    *,
    dt,
    dx,
    dy,
    dz,
    sharding=None,
):
    """Return OpenFOAM-style volume-mean and maximum mass-flux Courant numbers."""

    east, north, top = rho_phi_plus
    west = _neighbor_fields(east, mode_x="dirichlet", mode_y="neumann", mode_z="neumann", sharding=sharding)[
        0
    ]
    west = jnp.where(jnp.arange(density.shape[0])[:, None, None] == 0, rho_phi_inlet, west)
    south = jnp.concatenate((jnp.zeros_like(north[:, :1]), north[:, :-1]), axis=1)
    bottom = jnp.concatenate((jnp.zeros_like(top[:, :, :1]), top[:, :, :-1]), axis=2)
    widths = _coerce_spacing_vector(dx, density.shape[0], dtype=density.dtype)
    volume = jnp.ones_like(density) * widths[:, None, None] * dy[None, :, None] * dz[None, None, :]
    sum_phi = sum(map(jnp.abs, (west, east, south, north, bottom, top))) / density
    if sharding is not None:  # pragma: no cover - forced-device/hardware gates

        def reduce(local_sum, local_volume):
            total_sum = jax.lax.psum(jnp.sum(local_sum), "x")
            total_volume = jax.lax.psum(jnp.sum(local_volume), "x")
            maximum = jax.lax.pmax(jnp.max(local_sum / local_volume), "x")
            return 0.5 * dt * total_sum / total_volume, 0.5 * dt * maximum

        return jax.shard_map(
            reduce,
            mesh=sharding.mesh,
            in_specs=(sharding.spec, sharding.spec),
            out_specs=(P(), P()),
            check_vma=False,
        )(sum_phi, volume)
    return 0.5 * dt * jnp.sum(sum_phi) / jnp.sum(volume), 0.5 * dt * jnp.max(sum_phi / volume)


def _initialize_duct_mass_flux(velocity, density, inlet_velocity, *, dx, dy, dz, sharding=None):
    """Pack oriented, area-integrated ``linearInterpolate(rho*U)&Sf`` faces."""
    momentum = density[..., None] * velocity
    area_x = dy[:, None] * dz[None, :]
    inlet = density[0] * inlet_velocity[..., 0] * area_x
    axial = momentum[..., 0]
    plus_x = (
        0.5
        * (
            axial
            + _neighbor_fields(
                axial, mode_x="neumann", mode_y="neumann", mode_z="neumann", sharding=sharding
            )[1]
        )
        * area_x
    )
    wy = (dy[1:] / (dy[:-1] + dy[1:]))[None, :, None]
    plus_y = jnp.concatenate(
        (
            wy * momentum[:, :-1, :, 1] + (1.0 - wy) * momentum[:, 1:, :, 1],
            jnp.zeros_like(momentum[:, :1, :, 1]),
        ),
        axis=1,
    ) * (dx * dz[None, None, :])
    wz = (dz[1:] / (dz[:-1] + dz[1:]))[None, None, :]
    plus_z = jnp.concatenate(
        (
            wz * momentum[:, :, :-1, 2] + (1.0 - wz) * momentum[:, :, 1:, 2],
            jnp.zeros_like(momentum[:, :, :1, 2]),
        ),
        axis=2,
    ) * (dx * dy[None, :, None])
    return plus_x, plus_y, plus_z, inlet


def _flow_rate_inlet_profile(axial_velocity, face_area, target):
    """Evaluate the OpenFOAM-style flow-rate inlet normal velocity."""
    profile = jnp.maximum(axial_velocity, 0.0)
    estimated = jnp.sum(profile * face_area)
    return jnp.where(
        estimated > 0.5 * target,
        profile * target / jnp.maximum(estimated, 1.0e-20),
        profile + (target - estimated) / jnp.sum(face_area),
    )


def _duct_pressure_face_corrections(
    pressure,
    mobility,
    *,
    dx,
    dy,
    dz,
    mixed_axial_pressure,
    field_sharding=None,
):
    """Return the projection's three oriented pressure-velocity corrections."""

    def axial_neighbors(value):
        return _neighbor_fields(
            value, mode_x="neumann", mode_y="neumann", mode_z="neumann", sharding=field_sharding
        )[:2]

    pressure_east, mobility_east = (axial_neighbors(value)[1] for value in (pressure, mobility))
    correction_x = (
        -_harmonic_mean(mobility_east, mobility) * (pressure_east - pressure) / jnp.maximum(dx, 1.0e-12)
    )
    if mixed_axial_pressure:
        outlet_correction = mobility[-1] * pressure[-1] / jnp.maximum(0.5 * dx, 1.0e-12)
        outlet_cells = jnp.arange(pressure.shape[0])[:, None, None] == pressure.shape[0] - 1
        correction_x = jnp.where(outlet_cells, outlet_correction, correction_x)

    transverse = []
    for axis, width in ((1, dy), (2, dz)):
        values, coefficient = (jnp.moveaxis(field, axis, 0) for field in (pressure, mobility))
        face_coefficient = _distance_weighted_harmonic_mean(
            coefficient[1:],
            coefficient[:-1],
            width[1:].reshape((-1,) + (1,) * (values.ndim - 1)),
            width[:-1].reshape((-1,) + (1,) * (values.ndim - 1)),
        )
        distance = 0.5 * (width[:-1] + width[1:])
        correction = jnp.zeros((values.shape[0] + 1, *values.shape[1:]), dtype=pressure.dtype)
        correction = correction.at[1:-1].set(
            -face_coefficient
            * (values[1:] - values[:-1])
            / distance.reshape((-1,) + (1,) * (values.ndim - 1))
        )
        transverse.append(jnp.moveaxis(correction, 0, axis))
    return correction_x, *transverse


def _duct_face_divergence(plus_x, inlet_x, plus_y, plus_z, *, dx, dy, dz, field_sharding=None):
    """Apply the conservative mixed-boundary divergence to oriented face fields."""
    west = _neighbor_fields(
        plus_x,
        mode_x="dirichlet",
        mode_y="neumann",
        mode_z="neumann",
        sharding=field_sharding,
    )[0]
    west = jnp.where(jnp.arange(plus_x.shape[0])[:, None, None] == 0, inlet_x, west)
    return (
        (plus_x - west) / jnp.maximum(dx, 1.0e-12)
        + (plus_y[:, 1:] - plus_y[:, :-1]) / dy[None, :, None]
        + (plus_z[:, :, 1:] - plus_z[:, :, :-1]) / dz[None, None, :]
    )


def _duct_velocity_divergence(velocity, inlet_x, *, dx, dy, dz, field_sharding=None):
    """Map cell velocity to the conservative mixed-boundary divergence."""
    u, v, w = jnp.moveaxis(velocity, -1, 0)
    east = _neighbor_fields(u, mode_x="neumann", mode_y="neumann", mode_z="neumann", sharding=field_sharding)[
        1
    ]
    plus_x = 0.5 * (u + east)
    plus_y = jnp.zeros((u.shape[0], u.shape[1] + 1, u.shape[2]), dtype=u.dtype)
    plus_y = plus_y.at[:, 1:-1].set(0.5 * (v[:, 1:] + v[:, :-1]))
    plus_z = jnp.zeros((u.shape[0], u.shape[1], u.shape[2] + 1), dtype=u.dtype)
    plus_z = plus_z.at[:, :, 1:-1].set(0.5 * (w[:, :, 1:] + w[:, :, :-1]))
    return _duct_face_divergence(
        plus_x, inlet_x, plus_y, plus_z, dx=dx, dy=dy, dz=dz, field_sharding=field_sharding
    )


def _cell_pressure_correction_duct(correction_x, correction_y, correction_z, *, field_sharding=None):
    """Interpolate conservative face pressure corrections to cell centres."""
    correction_x_west = _neighbor_fields(
        correction_x, mode_x="neumann", mode_y="neumann", mode_z="neumann", sharding=field_sharding
    )[0]
    correction_x_west = jnp.where(
        jnp.arange(correction_x.shape[0])[:, None, None] == 0, 0.0, correction_x_west
    )
    return jnp.stack(
        (
            0.5 * (correction_x_west + correction_x),
            0.5 * (correction_y[:, :-1] + correction_y[:, 1:]),
            0.5 * (correction_z[:, :, :-1] + correction_z[:, :, 1:]),
        ),
        axis=-1,
    )


def _duct_pressure_force(pressure, *, dx, dy, dz, field_sharding=None):
    """Return the cell pressure force from the conservative face gradient."""
    corrections = _duct_pressure_face_corrections(
        pressure,
        jnp.ones_like(pressure),
        dx=dx,
        dy=dy,
        dz=dz,
        mixed_axial_pressure=True,
        field_sharding=field_sharding,
    )
    return _cell_pressure_correction_duct(*corrections, field_sharding=field_sharding)


def _face_flux_pressure_projection_duct(
    u: jnp.ndarray,
    v: jnp.ndarray,
    w: jnp.ndarray,
    rho: jnp.ndarray,
    fluid_mask: jnp.ndarray,
    *,
    dt: float,
    dx: float,
    dy: jnp.ndarray,
    dz: jnp.ndarray,
    iterations: int,
    tolerance: float,
    fluid_bounds: tuple[int, int, int, int] | None = None,
    base_pressure: jnp.ndarray | None = None,
    single_reduction: bool = False,
    include_axial_line: bool = True,
    inlet_flow_rate: float | None = None,
    field_sharding: NamedSharding | None = None,
    momentum_mobility: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, ...]:
    """Project duct face fluxes; mixed boundaries also return flow diagnostics."""
    mixed_axial_pressure = inlet_flow_rate is not None
    if inlet_flow_rate is not None and inlet_flow_rate <= 0.0:
        raise ValueError("Inlet flow rate must be positive")

    y0, y1, z0, z1 = _rectangular_fluid_bounds(fluid_mask) if fluid_bounds is None else fluid_bounds
    us = u[:, y0:y1, z0:z1]
    vs = v[:, y0:y1, z0:z1]
    ws = w[:, y0:y1, z0:z1]
    rhos = rho[:, y0:y1, z0:z1]
    dys = dy[y0:y1]
    dzs = dz[z0:z1]
    nx, ny, nz = us.shape
    outlet_cells = jnp.arange(nx)[:, None, None] == nx - 1

    def axial_neighbors(value):
        return _neighbor_fields(
            value, mode_x="neumann", mode_y="neumann", mode_z="neumann", sharding=field_sharding
        )[:2]

    uf_plus = 0.5 * (us + axial_neighbors(us)[1])
    uf_inlet = us[0]
    vf = jnp.zeros((nx, ny + 1, nz), dtype=v.dtype)
    vf = vf.at[:, 1:-1, :].set(0.5 * (vs[:, 1:, :] + vs[:, :-1, :]))
    wf = jnp.zeros((nx, ny, nz + 1), dtype=w.dtype)
    wf = wf.at[:, :, 1:-1].set(0.5 * (ws[:, :, 1:] + ws[:, :, :-1]))
    face_area = dys[:, None] * dzs[None, :]
    if inlet_flow_rate is not None:
        target = jnp.asarray(inlet_flow_rate, dtype=u.dtype)
        uf_inlet = _flow_rate_inlet_profile(us[0], face_area, target)

    divergence = _duct_face_divergence(
        uf_plus, uf_inlet, vf, wf, dx=dx, dy=dys, dz=dzs, field_sharding=field_sharding
    )
    mobility = dt / jnp.maximum(rhos, 1.0e-20) if momentum_mobility is None else momentum_mobility
    if mobility.shape != rhos.shape:
        raise ValueError("Momentum mobility must match the projected velocity")
    pressure_result = _solvax_pressure_poisson_duct(
        divergence,
        mobility,
        dx=dx,
        dy=dys,
        dz=dzs,
        iterations=iterations,
        tolerance=tolerance,
        initial_field=None,
        single_reduction=single_reduction,
        include_axial_line=include_axial_line,
        axial_pressure_mode=(_MIXED_AXIAL_PRESSURE_MODE if mixed_axial_pressure else "neumann"),
        field_sharding=field_sharding,
    )
    pressure_correction = pressure_result[0]
    pressure = (
        pressure_correction
        if base_pressure is None
        else (base_pressure[:, y0:y1, z0:z1] + ALEX_B2_PRESSURE_RELAXATION * pressure_correction)
    )

    correction_x, correction_y, correction_z = _duct_pressure_face_corrections(
        pressure_correction,
        mobility,
        dx=dx,
        dy=dys,
        dz=dzs,
        mixed_axial_pressure=mixed_axial_pressure,
        field_sharding=field_sharding,
    )
    uf_plus += correction_x
    vf += correction_y
    wf += correction_z
    divergence_after = _duct_face_divergence(
        uf_plus, uf_inlet, vf, wf, dx=dx, dy=dys, dz=dzs, field_sharding=field_sharding
    )
    # Reconstruct only pressure correction; reconstructing the predictor filters it.
    cell_correction = _cell_pressure_correction_duct(
        correction_x, correction_y, correction_z, field_sharding=field_sharding
    )
    projected_u, projected_v, projected_w = jnp.moveaxis(
        jnp.stack((us, vs, ws), axis=-1) + cell_correction, -1, 0
    )
    if mixed_axial_pressure:
        cell_flow = jnp.sum(projected_u * face_area, axis=(1, 2))
        projected_u += ((target - cell_flow) / jnp.sum(face_area))[:, None, None]
    full_u = jnp.zeros_like(u).at[:, y0:y1, z0:z1].set(projected_u)
    full_v = jnp.zeros_like(v).at[:, y0:y1, z0:z1].set(projected_v)
    full_w = jnp.zeros_like(w).at[:, y0:y1, z0:z1].set(projected_w)
    full_p = jnp.zeros_like(u).at[:, y0:y1, z0:z1].set(pressure)
    divergence_norm = jnp.max(jnp.abs(divergence_after))
    if not mixed_axial_pressure:
        return full_u, full_v, full_w, full_p, divergence_norm
    inlet_flow = jnp.sum(uf_inlet * face_area)
    outlet_flow = jnp.sum(uf_plus[-1] * face_area)
    active_mask = fluid_mask[:, y0:y1, z0:z1]
    area = face_area[None, :, :]
    active_area = jnp.sum(jnp.where(active_mask, area, 0.0), axis=(1, 2))
    mean_pressure = jnp.sum(jnp.where(active_mask, pressure * area, 0.0), axis=(1, 2)) / active_area
    mean_pressure_east = axial_neighbors(mean_pressure[:, None, None])[1][:, 0, 0]
    pressure_loss_plus = -(mean_pressure_east - mean_pressure) / jnp.maximum(dx, 1.0e-12)
    pressure_loss_plus = jnp.where(
        outlet_cells[:, 0, 0], mean_pressure[-1] / jnp.maximum(0.5 * dx, 1.0e-12), pressure_loss_plus
    )
    pressure_loss = 0.5 * (
        jnp.concatenate((jnp.zeros((1,), dtype=pressure.dtype), pressure_loss_plus[:-1])) + pressure_loss_plus
    )
    flow_error = jnp.maximum(jnp.abs(inlet_flow - inlet_flow_rate), jnp.abs(outlet_flow - inlet_flow_rate))
    rho_x = 0.5 * (rhos + axial_neighbors(rhos)[1])
    wy = (dys[1:] / (dys[:-1] + dys[1:]))[None, :, None]
    rho_y = jnp.concatenate((wy * rhos[:, :-1] + (1.0 - wy) * rhos[:, 1:], rhos[:, -1:]), axis=1)
    wz = (dzs[1:] / (dzs[:-1] + dzs[1:]))[None, None, :]
    rho_z = jnp.concatenate((wz * rhos[:, :, :-1] + (1.0 - wz) * rhos[:, :, 1:], rhos[:, :, -1:]), axis=2)
    rho_phi_x = rho_x * uf_plus * face_area
    rho_phi_y = rho_y * vf[:, 1:] * (dx * dzs[None, None, :])
    rho_phi_z = rho_z * wf[:, :, 1:] * (dx * dys[None, :, None])
    rho_phi_inlet = rhos[0] * uf_inlet * face_area
    linear_diagnostics = (
        pressure_result[1],
        pressure_result[3],
        pressure_result[4],
        pressure_result[2],
        pressure_result[5],
    )
    return (
        full_u,
        full_v,
        full_w,
        full_p,
        pressure_loss,
        divergence_norm,
        flow_error,
        rho_phi_x,
        rho_phi_y,
        rho_phi_z,
        rho_phi_inlet,
        *linear_diagnostics,
    )


def _duct_momentum_defect(
    velocity: jnp.ndarray,
    lorentz_force: jnp.ndarray,
    density: jnp.ndarray,
    viscosity: jnp.ndarray,
    rho_phi_plus: jnp.ndarray,
    rho_phi_inlet: jnp.ndarray,
    pressure: jnp.ndarray,
    *,
    forcing: float,
    force_scale: float,
    dt: float,
    dx: float,
    dy: jnp.ndarray,
    dz: jnp.ndarray,
    field_sharding: NamedSharding | None = None,
) -> jnp.ndarray:
    """Return componentwise and total post-map electromagnetic momentum defects."""

    shape = velocity.shape
    dx_widths = jnp.full((shape[0],), dx, dtype=velocity.dtype)
    widths = (dx_widths, dy, dz)
    face_area = dy[:, None] * dz[None, :]
    inlet_patch = velocity[0].at[..., 0].set(rho_phi_inlet / (density[0] * face_area))
    zero_y, zero_z = (jnp.zeros_like(velocity[:, 0]), jnp.zeros_like(velocity[:, :, 0]))
    boundary_velocity = (inlet_patch, velocity[-1], zero_y, zero_y, zero_z, zero_z)
    rho_phi = _unpack_duct_mass_flux(rho_phi_plus, rho_phi_inlet)
    setup = _frozen_duct_momentum_setup(
        velocity, density, viscosity, rho_phi, boundary_velocity, widths, dx=dx
    )
    dynamic_viscosity, coefficients, diffusion_sink, inlet_sink, weights, gradient = setup
    convection, diffusion = _duct_momentum_transport(
        velocity, rho_phi, weights, boundary_velocity, widths, coefficients, diffusion_sink
    )
    inlet_cells = jnp.arange(shape[0])[:, None, None] == 0
    inlet_velocity = jnp.where(inlet_cells[..., None], inlet_patch, 0.0)
    diffusion += inlet_sink[..., None] * inlet_velocity
    deviatoric_stress = _explicit_deviatoric_stress_duct(
        velocity, dynamic_viscosity, boundary_velocity, widths, gradient=gradient
    )

    pressure_force = _duct_pressure_force(pressure, dx=dx, dy=dy, dz=dz, field_sharding=field_sharding)
    residual = (
        convection - diffusion - deviatoric_stress - lorentz_force.at[..., 0].add(forcing) - pressure_force
    )
    component_maxima = jnp.max(jnp.abs(residual), axis=(0, 1, 2)) / force_scale
    return jnp.concatenate((component_maxima, jnp.max(component_maxima)[None]))


def _conservative_current_fluxes_3d(
    sigma: jnp.ndarray,
    phi: jnp.ndarray,
    uxb_x: jnp.ndarray,
    uxb_y: jnp.ndarray,
    uxb_z: jnp.ndarray,
    *,
    dx: float,
    dy: float | jnp.ndarray,
    dz: float | jnp.ndarray,
    thin_wall_fluid_mask: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    nx, ny, nz = phi.shape
    fx = jnp.zeros((nx + 1, ny, nz), dtype=phi.dtype)
    fy = jnp.zeros((nx, ny + 1, nz), dtype=phi.dtype)
    fz = jnp.zeros((nx, ny, nz + 1), dtype=phi.dtype)
    dy_widths = _spacing_vector(dy, ny, dtype=phi.dtype)
    dz_widths = _spacing_vector(dz, nz, dtype=phi.dtype)
    dy_centers = 0.5 * (dy_widths[:-1] + dy_widths[1:])
    dz_centers = 0.5 * (dz_widths[:-1] + dz_widths[1:])

    sigma_x = _harmonic_mean(sigma[1:], sigma[:-1])
    phi_grad_x = (phi[1:] - phi[:-1]) / jnp.maximum(dx, 1.0e-12)
    uxb_face_x = 0.5 * (uxb_x[1:] + uxb_x[:-1])
    fx = fx.at[1:-1].set(sigma_x * (-phi_grad_x + uxb_face_x))

    sigma_y = _distance_weighted_harmonic_mean(
        sigma[:, 1:, :],
        sigma[:, :-1, :],
        dy_widths[None, 1:, None],
        dy_widths[None, :-1, None],
    )
    if thin_wall_fluid_mask is not None:
        sigma_y = _thin_wall_interface_mean(
            sigma[:, 1:, :],
            sigma[:, :-1, :],
            dy_widths[None, 1:, None],
            dy_widths[None, :-1, None],
            thin_wall_fluid_mask[:, 1:, :],
            thin_wall_fluid_mask[:, :-1, :],
        )
    phi_grad_y = (phi[:, 1:, :] - phi[:, :-1, :]) / dy_centers[None, :, None]
    uxb_face_y = 0.5 * (uxb_y[:, 1:, :] + uxb_y[:, :-1, :])
    fy = fy.at[:, 1:-1, :].set(sigma_y * (-phi_grad_y + uxb_face_y))

    sigma_z = _distance_weighted_harmonic_mean(
        sigma[:, :, 1:],
        sigma[:, :, :-1],
        dz_widths[None, None, 1:],
        dz_widths[None, None, :-1],
    )
    if thin_wall_fluid_mask is not None:
        sigma_z = _thin_wall_interface_mean(
            sigma[:, :, 1:],
            sigma[:, :, :-1],
            dz_widths[None, None, 1:],
            dz_widths[None, None, :-1],
            thin_wall_fluid_mask[:, :, 1:],
            thin_wall_fluid_mask[:, :, :-1],
        )
    phi_grad_z = (phi[:, :, 1:] - phi[:, :, :-1]) / dz_centers[None, None, :]
    uxb_face_z = 0.5 * (uxb_z[:, :, 1:] + uxb_z[:, :, :-1])
    fz = fz.at[:, :, 1:-1].set(sigma_z * (-phi_grad_z + uxb_face_z))
    return fx, fy, fz


def _station_axial_current_from_fluxes(fx: jnp.ndarray, cell_area: jnp.ndarray) -> jnp.ndarray:
    face_axial_current = jnp.sum(fx * cell_area[None, :, :], axis=(1, 2))
    return 0.5 * (face_axial_current[1:] + face_axial_current[:-1])


def _conservative_current_diagnostics_3d(
    sigma: jnp.ndarray,
    phi: jnp.ndarray,
    uxb_x: jnp.ndarray,
    uxb_y: jnp.ndarray,
    uxb_z: jnp.ndarray,
    *,
    dx: float,
    dy: float | jnp.ndarray,
    dz: float | jnp.ndarray,
    thin_wall_fluid_mask: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    fx, fy, fz = _conservative_current_fluxes_3d(
        sigma,
        phi,
        uxb_x,
        uxb_y,
        uxb_z,
        dx=dx,
        dy=dy,
        dz=dz,
        thin_wall_fluid_mask=thin_wall_fluid_mask,
    )
    dy_widths = _spacing_vector(dy, phi.shape[1], dtype=phi.dtype)
    dz_widths = _spacing_vector(dz, phi.shape[2], dtype=phi.dtype)
    div_j = (
        (fx[1:] - fx[:-1]) / jnp.maximum(dx, 1.0e-12)
        + (fy[:, 1:, :] - fy[:, :-1, :]) / dy_widths[None, :, None]
        + (fz[:, :, 1:] - fz[:, :, :-1]) / dz_widths[None, None, :]
    )
    wall_leakage = (
        jnp.sum(jnp.abs(fy[:, 0, :]) * dz_widths[None, :], axis=1) * dx
        + jnp.sum(jnp.abs(fy[:, -1, :]) * dz_widths[None, :], axis=1) * dx
        + jnp.sum(jnp.abs(fz[:, :, 0]) * dy_widths[None, :], axis=1) * dx
        + jnp.sum(jnp.abs(fz[:, :, -1]) * dy_widths[None, :], axis=1) * dx
    )
    cross_section_area = dy_widths[:, None] * dz_widths[None, :]
    # Integrating the conservative cell divergence gives the complete boundary
    # flux for each axial control-volume slab. This is the discrete divergence
    # theorem and avoids mixing global inlet/outlet fluxes into every station.
    boundary_residual = jnp.abs(jnp.sum(div_j * cross_section_area[None, :, :], axis=(1, 2)) * dx)
    return div_j, wall_leakage, boundary_residual


def _conservative_emf_rhs_3d(
    sigma: jnp.ndarray,
    uxb_x: jnp.ndarray,
    uxb_y: jnp.ndarray,
    uxb_z: jnp.ndarray,
    *,
    dx: float,
    dy: float | jnp.ndarray,
    dz: float | jnp.ndarray,
    thin_wall_fluid_mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    zeros = jnp.zeros_like(uxb_x)
    fx, fy, fz = _conservative_current_fluxes_3d(
        sigma,
        zeros,
        uxb_x,
        uxb_y,
        uxb_z,
        dx=dx,
        dy=dy,
        dz=dz,
        thin_wall_fluid_mask=thin_wall_fluid_mask,
    )
    dy_widths = _spacing_vector(dy, sigma.shape[1], dtype=sigma.dtype)
    dz_widths = _spacing_vector(dz, sigma.shape[2], dtype=sigma.dtype)
    return (
        (fx[1:] - fx[:-1]) / jnp.maximum(dx, 1.0e-12)
        + (fy[:, 1:, :] - fy[:, :-1, :]) / dy_widths[None, :, None]
        + (fz[:, :, 1:] - fz[:, :, :-1]) / dz_widths[None, None, :]
    )


def _b2_momentum_functions(
    *,
    case: CaseSpec,
    shape: tuple[int, int, int],
    fluid_bounds: tuple[int, int, int, int],
    metric: tuple[float, float, jnp.ndarray, jnp.ndarray],
    target_flow_rate: float,
    electromagnetic_force_scale: float,
    momentum_iterations: int,
    momentum_tolerance: float,
    field_sharding: NamedSharding | None,
) -> tuple[Callable, ...]:
    """Build the conservative B2 momentum and compact-flux operations."""

    dt, dx, dy, dz = metric
    nx, ny, nz = shape
    y0, y1, z0, z1 = fluid_bounds
    local_dy, local_dz = dy[y0:y1], dz[z0:z1]
    face_area = local_dy[:, None] * local_dz[None, :]

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

    def momentum_solve(velocity, force, density, viscosity, rho_phi_plus, rho_phi_inlet, pressure, reaction):
        local_velocity, local_density, local_viscosity, local_reaction = (
            field[:, y0:y1, z0:z1] for field in (velocity, density, viscosity, reaction)
        )
        inlet_patch = local_velocity[0].at[..., 0].set(rho_phi_inlet / (local_density[0] * face_area))
        zero_y, zero_z = jnp.zeros_like(local_velocity[:, 0]), jnp.zeros_like(local_velocity[:, :, 0])
        boundary_velocity = (inlet_patch, local_velocity[-1], zero_y, zero_y, zero_z, zero_z)
        widths = (jnp.full((nx,), dx), local_dy, local_dz)
        rho_phi = _unpack_duct_mass_flux(rho_phi_plus, rho_phi_inlet)
        setup = _frozen_duct_momentum_setup(
            local_velocity, local_density, local_viscosity, rho_phi, boundary_velocity, widths, dx=dx
        )
        local_force = force[:, y0:y1, z0:z1] + _explicit_deviatoric_stress_duct(
            local_velocity, setup[0], boundary_velocity, widths, gradient=setup[-1]
        )
        local_force += _duct_pressure_force(
            pressure[:, y0:y1, z0:z1],
            dx=dx,
            dy=local_dy,
            dz=local_dz,
            field_sharding=field_sharding,
        )
        solved = _solvax_implicit_momentum_duct(
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
            reaction=local_reaction,
        )
        momentum_mobility = dt / jnp.maximum(
            local_density + dt * (local_reaction + sum(setup[1]) + setup[2]), 1.0e-20
        )
        return (*solved, momentum_mobility)

    def momentum_defect(velocity, force, density, viscosity, rho_phi_plus, rho_phi_inlet, pressure):
        return _duct_momentum_defect(
            velocity[:, y0:y1, z0:z1],
            force[:, y0:y1, z0:z1],
            density[:, y0:y1, z0:z1],
            viscosity[:, y0:y1, z0:z1],
            rho_phi_plus,
            rho_phi_inlet,
            pressure[:, y0:y1, z0:z1],
            forcing=float(case.forcing),
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

    def relax_flux(current, current_inlet, mapped, mapped_inlet, relaxation):
        return (
            current + relaxation * (mapped - current),
            current_inlet + relaxation * (mapped_inlet - current_inlet),
        )

    return (
        initialize_flux,
        jax.named_call(momentum_solve, name="lmx.b2.momentum"),
        jax.named_call(momentum_defect, name="lmx.b2.momentum_defect"),
        embed_velocity,
        courant_numbers,
        pack_flux,
        unpack_flux,
        pack_vector,
        relax_flux,
    )


def _jit_b2_functions(functions, shardings, kernel_key):  # pragma: no cover - hardware gate
    return tuple(
        _reuse_fringing_jit(
            (function.__name__, *kernel_key),
            jax.jit(function, in_shardings=input_sharding, out_shardings=output_sharding),
        )
        for function, (input_sharding, output_sharding) in zip(functions, shardings, strict=True)
    )


def _jit_b2_momentum_functions(
    functions: tuple[Callable, ...],
    *,
    field_sharding: NamedSharding | None,
    replicated_sharding: NamedSharding | None,
    flux_sharding: NamedSharding | None,
    vector_sharding: NamedSharding | None,
    kernel_key: tuple[object, ...],
) -> tuple[Callable, ...]:  # pragma: no cover - requires a multi-device mesh
    if field_sharding is None:
        return functions
    shardings = (
        ((field_sharding,) * 4, (field_sharding,) * 3 + (replicated_sharding,)),
        (
            (vector_sharding,) * 2
            + (field_sharding,) * 2
            + (flux_sharding, replicated_sharding)
            + (field_sharding,) * 2,
            (vector_sharding, replicated_sharding, replicated_sharding, field_sharding),
        ),
        (
            (vector_sharding,) * 2
            + (field_sharding,) * 2
            + (flux_sharding, replicated_sharding, field_sharding),
            replicated_sharding,
        ),
        ((vector_sharding, field_sharding), (field_sharding,) * 3),
        (
            (field_sharding,) * 3 + (replicated_sharding, field_sharding),
            (replicated_sharding, replicated_sharding),
        ),
        ((field_sharding,) * 3, flux_sharding),
        (flux_sharding, (field_sharding,) * 3),
        ((field_sharding,) * 3, vector_sharding),
        (
            (flux_sharding, replicated_sharding, flux_sharding) + (replicated_sharding,) * 2,
            (flux_sharding, replicated_sharding),
        ),
    )
    return _jit_b2_functions(functions, shardings, kernel_key)


def _prepare_b2_momentum_runtime(
    *,
    case: CaseSpec,
    velocity: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    density: jnp.ndarray,
    fluid_bounds: tuple[int, int, int, int],
    num_devices: int | None,
    dy: jnp.ndarray,
    dz: jnp.ndarray,
    dt: float,
    dx: float,
    momentum_iterations: int,
    momentum_tolerance: float,
    projection_iterations: int,
    projection_tolerance: float,
    electric_iterations: int,
    electric_tolerance: float,
    forcing: float,
    target_flow_rate: float,
    initial: ExtrudedFieldBundle | None,
    previous_anderson_flux: jnp.ndarray | None,
    previous_anderson_inlet: jnp.ndarray | None,
):
    """Prepare sharding, compiled kernels, and compact B2 flux restart state."""

    u, v, w = velocity
    y0, y1, z0, z1 = fluid_bounds
    field_sharding = u.sharding if num_devices is not None else None
    replicated = None if field_sharding is None else NamedSharding(field_sharding.mesh, P())
    flux_sharding = (
        None if field_sharding is None else NamedSharding(field_sharding.mesh, P(None, "x", None, None))
    )
    vector_sharding = (
        None if field_sharding is None else NamedSharding(field_sharding.mesh, P("x", None, None, None))
    )
    force_scale = (
        next(region for region in case.regions if region.kind == "fluid").conductivity
        * target_flow_rate
        / float(jnp.sum(dy[y0:y1, None] * dz[None, z0:z1]))
        * sum(float(component) ** 2 for component in (case.magnetic_field.value or (0.0, 0.0, 0.0)))
    )
    if force_scale <= 0.0:
        raise ValueError("ALEX B2 requires a positive electromagnetic force scale")
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
        force_scale,
        case.solver.coupling_regularization,
        case.solver.coupling_damping,
    )
    functions = _jit_b2_momentum_functions(
        _b2_momentum_functions(
            case=case,
            shape=u.shape,
            fluid_bounds=fluid_bounds,
            metric=(dt, dx, dy, dz),
            target_flow_rate=target_flow_rate,
            electromagnetic_force_scale=force_scale,
            momentum_iterations=momentum_iterations,
            momentum_tolerance=momentum_tolerance,
            field_sharding=field_sharding,
        ),
        field_sharding=field_sharding,
        replicated_sharding=replicated,
        flux_sharding=flux_sharding,
        vector_sharding=vector_sharding,
        kernel_key=kernel_key,
    )
    initialize_flux, *_, pack_flux, _, _, _ = functions
    restart_flux = None if initial is None else initial.rho_phi_plus
    restart_inlet = None if initial is None else initial.rho_phi_inlet
    if (restart_flux is None) != (restart_inlet is None):
        raise ValueError("B2 restart requires both compact flux arrays")
    if restart_flux is None:
        *current_flux, current_inlet = initialize_flux(u, v, w, density)
    else:
        restart_flux = np.asarray(restart_flux, dtype=np.dtype(u.dtype))
        current_flux = tuple(
            jnp.asarray(value) if field_sharding is None else jax.device_put(value, field_sharding)
            for value in restart_flux
        )
        current_inlet = jnp.asarray(restart_inlet, dtype=u.dtype)
        if replicated is not None:
            current_inlet = jax.device_put(np.asarray(current_inlet), replicated)
    current_plus = pack_flux(*current_flux)
    if previous_anderson_flux is not None:
        if (
            previous_anderson_flux.shape != current_plus.shape
            or previous_anderson_inlet.shape != current_inlet.shape
        ):
            raise ValueError("B2 restart Anderson flux state has inconsistent shape")
        if flux_sharding is not None:  # pragma: no cover - hardware gate
            previous_anderson_flux = jax.device_put(np.asarray(previous_anderson_flux), flux_sharding)
            previous_anderson_inlet = jax.device_put(np.asarray(previous_anderson_inlet), replicated)
    return (
        *functions,
        field_sharding,
        replicated,
        flux_sharding,
        kernel_key,
        current_inlet,
        current_plus,
        previous_anderson_flux,
        previous_anderson_inlet,
    )


def _b2_coupling_functions(
    *,
    case: CaseSpec,
    target_flow_rate: float,
    metric: tuple[float, float, jnp.ndarray, jnp.ndarray],
    projection_iterations: int,
    projection_tolerance: float,
    electric_iterations: int,
    electric_tolerance: float,
    electric_volume_min: float,
    fluid_bounds: tuple[int, int, int, int],
    field_sharding: NamedSharding | None,
    fixed_point_scale: jnp.ndarray,
) -> tuple[Callable, ...]:
    """Build B2 pressure, electric, Lorentz, and coupling operations."""

    dt, dx, dy, dz = metric

    def mixed_boundary_projection(u, v, w, pressure, density, mask, momentum_mobility):
        return _face_flux_pressure_projection_duct(
            u,
            v,
            w,
            density,
            mask,
            inlet_flow_rate=target_flow_rate,
            dt=dt,
            dx=dx,
            dy=dy,
            dz=dz,
            iterations=projection_iterations,
            tolerance=projection_tolerance,
            fluid_bounds=fluid_bounds,
            base_pressure=pressure,
            single_reduction=field_sharding is not None,
            include_axial_line=False,
            field_sharding=field_sharding,
            momentum_mobility=momentum_mobility,
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
            include_axial_line=False,
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

    def electric_lorentz(potential, conductivity, emf_x, emf_y, emf_z, bx, by, bz):
        dphi_dx, dphi_dy, dphi_dz = _gradient_3d(potential, dx=dx, dy=dy, dz=dz)
        current_x = conductivity * (-dphi_dx + emf_x)
        current_y = conductivity * (-dphi_dy + emf_y)
        current_z = conductivity * (-dphi_dz + emf_z)
        return current_x, current_y, current_z, *_cross((current_x, current_y, current_z), (bx, by, bz))

    def reconstruct_electric(potential, conductivity, emf_x, emf_y, emf_z, bx, by, bz, mask):
        fields = electric_lorentz(potential, conductivity, emf_x, emf_y, emf_z, bx, by, bz)
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
        return (*fields[:3], divergence, *fields[3:])

    def lorentz_operator(potential, conductivity, emf_x, emf_y, emf_z, bx, by, bz):
        return electric_lorentz(potential, conductivity, emf_x, emf_y, emf_z, bx, by, bz)

    def scaled_state(u, v, w):
        return jnp.stack((u, v, w)) / fixed_point_scale

    def unscaled_state(state):
        values = state * fixed_point_scale
        return values[0], values[1], values[2]

    def mix_anderson(mapped0, residual0, flux0, inlet0, mapped1, residual1, flux1, inlet1):
        weights = anderson_weights(
            jnp.stack((residual0, residual1)),
            regularization=case.solver.coupling_regularization,
            condition_limit=ALEX_B2_ANDERSON_CONDITION_LIMIT,
        )
        damping = case.solver.coupling_damping

        def mix(previous, current):
            weighted = jnp.tensordot(weights, jnp.stack((previous, current)), axes=(0, 0))
            return current + damping * (weighted - current)

        return mix(mapped0, mapped1), mix(flux0, flux1), mix(inlet0, inlet1)

    return (
        jax.named_call(mixed_boundary_projection, name="lmx.b2.projection"),
        jax.named_call(electric_solve, name="lmx.b2.electric"),
        jax.named_call(emf_operator, name="lmx.b2.emf"),
        jax.named_call(reconstruct_electric, name="lmx.b2.reconstruction"),
        lorentz_operator,
        scaled_state,
        sub,
        unscaled_state,
        jax.named_call(mix_anderson, name="lmx.b2.anderson"),
    )


def _jit_b2_coupling_functions(
    functions: tuple[Callable, ...],
    *,
    field_sharding: NamedSharding | None,
    replicated_sharding: NamedSharding | None,
    flux_sharding: NamedSharding | None,
    kernel_key: tuple[object, ...],
) -> tuple[Callable, ...]:  # pragma: no cover - requires a multi-device mesh
    if field_sharding is None:
        return functions
    axial_sharding = NamedSharding(field_sharding.mesh, P("x"))
    state_sharding = NamedSharding(field_sharding.mesh, P(None, "x", None, None))
    shardings = (
        (
            (field_sharding,) * 7,
            (field_sharding,) * 4
            + (axial_sharding, replicated_sharding, replicated_sharding)
            + (field_sharding,) * 3
            + (replicated_sharding,) * 6,
        ),
        ((field_sharding,) * 4, (field_sharding,) + (replicated_sharding,) * 6),
        ((field_sharding,) * 5, field_sharding),
        ((field_sharding,) * 9, (field_sharding,) * 7),
        ((field_sharding,) * 8, (field_sharding,) * 6),
        ((field_sharding,) * 3, state_sharding),
        ((state_sharding, state_sharding), state_sharding),
        (state_sharding, (field_sharding,) * 3),
        (
            (state_sharding, state_sharding, flux_sharding, replicated_sharding) * 2,
            (state_sharding, flux_sharding, replicated_sharding),
        ),
    )
    return _jit_b2_functions(functions, shardings, kernel_key)

"""Shared rectangular 3-D kernels and pressure systems."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.fft import dct, idct
from jax.scipy.linalg import solve_triangular
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from solvax import (
    fixed_point_iteration,
    pcg_linear_solve,
    tridiagonal_solve,
)

from .specs import (
    EXTRUDED_HISTORY_WIDTHS,
    CaseSpec,
    ExtrudedFieldBundle,
    ExtrudedIterationProgress,
)

_EXTRUDED_NUMERICAL_RESULTS = (
    "x",
    "y",
    "z",
    "field_scale",
    "u",
    "v",
    "w",
    "p",
    "phi",
    "jx",
    "jy",
    "jz",
    "lorentz_x",
    "lorentz_y",
    "lorentz_z",
    "residual",
    "volumetric_flow_rate",
    "mean_velocity",
    "axial_current",
    "wall_current_leakage",
    "current_scaled_pressure_proxy",
    "charge_balance_residual",
    "boundary_current_residual",
    "axial_pressure_loss_gradient",
    "transverse_pressure_difference",
)


_FRINGING_JIT_CACHE: dict[tuple[object, ...], Callable] = {}


_FRINGING_MODAL_FACTOR_CACHE: dict[tuple[object, ...], object] = {}


_MIXED_AXIAL_PRESSURE_MODE = "inlet_neumann_outlet_dirichlet_zero"


def _reuse_fringing_jit(key: tuple[object, ...], function: Callable) -> Callable:
    """Reuse an identical compiled production kernel across repeated solves."""

    return _FRINGING_JIT_CACHE.setdefault(key, function)


def _array_fingerprint(*arrays: jnp.ndarray) -> str:
    """Return a compact cache key for immutable operator coefficients."""

    digest = hashlib.blake2b(digest_size=16)
    for array in arrays:
        host = np.ascontiguousarray(np.asarray(array))
        digest.update(host.dtype.str.encode())
        digest.update(np.asarray(host.shape, dtype=np.int64).tobytes())
        digest.update(host.tobytes())
    return digest.hexdigest()


def _reuse_modal_factors(key: tuple[object, ...], factory: Callable):
    """Build expensive B1 modal factors once per backend and operator."""

    factors = _FRINGING_MODAL_FACTOR_CACHE.get(key)
    if factors is None:
        factors = factory()
        if len(_FRINGING_MODAL_FACTOR_CACHE) >= 8:
            _FRINGING_MODAL_FACTOR_CACHE.pop(next(iter(_FRINGING_MODAL_FACTOR_CACHE)))
        _FRINGING_MODAL_FACTOR_CACHE[key] = factors
    return factors


# Frozen Benchmark B mass/current closure and stable-map controls.
ALEX_BALANCE_TOLERANCE = 1.0e-3
ALEX_B2_STEADY_STEPS = 3
ALEX_B2_CANONICAL_SHELL_THICKNESS = 0.02
ALEX_B2_MAGNETIC_STABILITY_SAFETY = 0.064
ALEX_B2_SETTLED_RELAXATION = 2.0


def _axial_field_sharding(num_devices: int) -> NamedSharding:
    """Return one process-stable axial mesh for compilation and repeat reuse."""

    devices = jax.devices()
    if not 1 <= num_devices <= len(devices):
        raise ValueError(f"Requested {num_devices} devices, but only {len(devices)} are visible.")
    return NamedSharding(Mesh(np.asarray(devices[:num_devices], dtype=object), ("x",)), P("x", None, None))


def _shard_extruded_fields(
    fields: tuple[jnp.ndarray, ...], *, num_devices: int | None
) -> tuple[jnp.ndarray, ...]:
    """Place 3-D fields on an axial mesh, staging initial values through the host."""

    if num_devices is None:
        return fields
    devices = jax.devices()
    if not 1 <= num_devices <= len(devices):
        raise ValueError(f"Requested {num_devices} devices, but only {len(devices)} are visible.")
    axial_size = fields[0].shape[0]
    if axial_size % num_devices:
        raise ValueError(f"Axial cell count {axial_size} must be divisible by {num_devices} devices.")
    sharding = _axial_field_sharding(num_devices)
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
    *,
    stride=1,
    retained_prefix=0,
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
    arrays = {
        name: jnp.asarray(value, dtype=float).reshape((-1, width))
        if width
        else jnp.asarray(value, dtype=float)
        for (name, width), value in zip(EXTRUDED_HISTORY_WIDTHS, values, strict=True)
    }
    if stride == 1:
        return arrays
    for name, array in arrays.items():
        if not len(array):
            continue
        if stride == 0:
            arrays[name] = array[-1:]
            continue
        prefix, segment = array[:retained_prefix], array[retained_prefix:]
        if not len(segment):
            arrays[name] = prefix
            continue
        sampled = segment[::stride]
        if (len(segment) - 1) % stride:
            sampled = jnp.concatenate((sampled, segment[-1:]))
        arrays[name] = jnp.concatenate((prefix, sampled)) if len(prefix) else sampled
    return arrays


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
    """Build the minimal bundle needed to resume a solve."""

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
    function: Callable, name: str, callback: Callable[[str, float], None] | None
) -> Callable:
    """Wrap one diagnostic phase with a completion barrier and wall timer."""

    if callback is None:
        return function

    def measured(*args):
        jax.block_until_ready(args)
        started = perf_counter()
        result = function(*args)
        jax.block_until_ready(result)
        callback(name, perf_counter() - started)
        return result

    return measured


def _restore_duct_iteration_state(
    initial: ExtrudedFieldBundle | None,
    *,
    case: CaseSpec,
    use_b2: bool,
    velocity: jnp.ndarray,
    velocity_limit: float,
    potential_scale: float,
    forcing: float,
):
    """Restore current duct histories and fixed-point accelerator state."""

    history_names = (
        "iteration_residual_history",
        "iteration_component_residual_history",
        "iteration_pressure_residual_history",
        "iteration_electric_linear_history",
        "iteration_potential_residual_history",
    )
    histories = tuple(
        [] if initial is None else np.asarray(getattr(initial, name, jnp.zeros((0,)))).tolist()
        for name in history_names
    )
    if len({len(history) for history in histories}) != 1:
        raise ValueError("B2 restart iteration histories have inconsistent lengths")
    stored_steps = len(histories[0])
    restart_stopping = (
        (0, 0, "not_recorded")
        if initial is None
        else getattr(initial, "stopping_state", (0, 0, "not_recorded"))
    )
    completed_steps = int(restart_stopping[0])
    if stored_steps > completed_steps or bool(stored_steps) != bool(completed_steps):
        raise ValueError("B2 restart stopping state has inconsistent step count")
    momentum_defect = (
        [] if initial is None else np.asarray(initial.iteration_momentum_defect_history).tolist()
    )
    if use_b2 and len(momentum_defect) != stored_steps:
        raise ValueError("B2 restart predates the electromagnetic momentum-defect contract")
    pressure_linear = (
        []
        if initial is None
        else np.asarray(getattr(initial, "iteration_pressure_linear_history", jnp.zeros((0, 5)))).tolist()
    )
    if stored_steps and not pressure_linear:
        pressure_linear = [[math.nan, math.nan, 0.0, 0.0, -1.0]] * stored_steps
    if len(pressure_linear) != stored_steps:
        raise ValueError("B2 restart pressure-linear history has inconsistent length")
    courant = (
        []
        if initial is None
        else np.asarray(getattr(initial, "iteration_courant_history", jnp.zeros((0, 3)))).tolist()
    )
    if stored_steps and not courant:
        courant = [[-1.0, -1.0, -1.0]] * stored_steps
    if len(courant) != stored_steps:
        raise ValueError("B2 restart CFL histories have inconsistent lengths")

    fixed_aitken = (
        float(case.solver.coupling_min_relaxation)
        if use_b2
        and case.solver.coupling_acceleration == "aitken"
        and case.solver.coupling_min_relaxation == case.solver.coupling_max_relaxation
        else None
    )
    steady_streak = int(restart_stopping[1])
    fixed_relaxation = jnp.asarray(1.0, dtype=velocity.dtype)
    previous_fixed_residual = None
    restart_aitken = None if initial is None else getattr(initial, "aitken_state", None)
    if use_b2 and restart_aitken is not None:
        previous_fixed_residual, fixed_relaxation, stored_streak = restart_aitken
        if restart_stopping[2] == "not_recorded":
            steady_streak = stored_streak
        elif steady_streak != stored_streak:
            raise ValueError("B2 restart stopping and Aitken streaks disagree")
        fixed_relaxation = jnp.asarray(fixed_relaxation, dtype=velocity.dtype)
        if fixed_aitken is not None:
            previous_fixed_residual = None
        elif previous_fixed_residual is not None:
            previous_fixed_residual = jnp.asarray(previous_fixed_residual, dtype=velocity.dtype)
            if previous_fixed_residual.shape != (4, *velocity.shape):
                raise ValueError("B2 restart Aitken residual has inconsistent shape")

    previous_mapped = previous_anderson_residual = None
    previous_flux = previous_inlet = None
    restart_anderson = None if initial is None else getattr(initial, "anderson_state", None)
    if use_b2 and case.solver.coupling_acceleration == "anderson":
        if completed_steps and restart_anderson is None:
            raise ValueError("B2 Anderson restart is missing accelerator state")
        if restart_anderson is not None:
            if len(restart_anderson) != 4 or any(value is None for value in restart_anderson):
                raise ValueError("B2 Anderson restart state must be all-or-none")
            previous_mapped, previous_anderson_residual, previous_flux, previous_inlet = (
                jnp.asarray(value, dtype=velocity.dtype) for value in restart_anderson
            )
            expected = (4, *velocity.shape)
            if previous_mapped.shape != expected or previous_anderson_residual.shape != expected:
                raise ValueError("B2 restart Anderson field state has inconsistent shape")

    fixed_scale = jnp.asarray(
        [velocity_limit, velocity_limit, velocity_limit, potential_scale], dtype=velocity.dtype
    )[:, None, None, None]
    return (
        *histories,
        completed_steps,
        momentum_defect,
        pressure_linear,
        courant,
        previous_fixed_residual,
        fixed_aitken,
        steady_streak,
        fixed_relaxation,
        previous_mapped,
        previous_anderson_residual,
        previous_flux,
        previous_inlet,
        fixed_scale,
        jnp.full((velocity.shape[0],), forcing, dtype=float),
    )


def _canonical_shell_widths(widths: jnp.ndarray, lower: int, upper: int) -> jnp.ndarray:
    """Map explicit wall cells to the frozen B2 mixed-dimensional shell."""

    if lower:
        widths = widths.at[:lower].multiply(ALEX_B2_CANONICAL_SHELL_THICKNESS / jnp.sum(widths[:lower]))
    if upper < widths.size:
        widths = widths.at[upper:].multiply(ALEX_B2_CANONICAL_SHELL_THICKNESS / jnp.sum(widths[upper:]))
    return widths


def _broadcast_cross_section(values: jnp.ndarray, nx: int) -> jnp.ndarray:
    return jnp.broadcast_to(jnp.asarray(values, dtype=float)[None, :, :], (nx,) + tuple(values.shape))


def _harmonic_mean(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    denom = jnp.maximum(a + b, 1.0e-20)
    return 2.0 * a * b / denom


def _distance_weighted_harmonic_mean(
    a: jnp.ndarray,
    b: jnp.ndarray,
    a_width: jnp.ndarray,
    b_width: jnp.ndarray,
) -> jnp.ndarray:
    """Return the face coefficient for two unequal finite-volume half-cells."""

    total_width = a_width + b_width
    resistance = a_width / jnp.maximum(a, 1.0e-20) + b_width / jnp.maximum(b, 1.0e-20)
    return total_width / jnp.maximum(resistance, 1.0e-20)


def _thin_wall_interface_mean(
    a: jnp.ndarray,
    b: jnp.ndarray,
    a_width: jnp.ndarray,
    b_width: jnp.ndarray,
    a_fluid: jnp.ndarray,
    b_fluid: jnp.ndarray,
) -> jnp.ndarray:
    """Collapse wall-normal half-cell resistance at a thin-wall interface."""

    base = _distance_weighted_harmonic_mean(a, b, a_width, b_width)
    interface = a_fluid != b_fluid
    fluid_sigma = jnp.where(a_fluid, a, b)
    fluid_width = jnp.where(a_fluid, a_width, b_width)
    collapsed = (a_width + b_width) * fluid_sigma / jnp.maximum(fluid_width, 1.0e-20)
    return jnp.where(interface, collapsed, base)


def _coerce_spacing_vector(spacing: float | jnp.ndarray, size: int, *, dtype) -> jnp.ndarray:
    values = jnp.asarray(spacing, dtype=dtype)
    if values.ndim == 0:
        return jnp.full((size,), values, dtype=dtype)
    if values.shape != (size,):
        raise ValueError(f"spacing must be scalar or have shape ({size},)")
    return values


def _spacing_vector(spacing: float | jnp.ndarray, size: int, *, dtype) -> jnp.ndarray:
    values = _coerce_spacing_vector(spacing, size, dtype=dtype)
    minimum = jnp.min(values)
    if not isinstance(minimum, jax.core.Tracer) and float(minimum) <= 0.0:
        raise ValueError("spacing values must be positive")
    return values


def _nonuniform_axis_gradient(field: jnp.ndarray, widths: jnp.ndarray, *, axis: int) -> jnp.ndarray:
    """Cell-centred derivative with homogeneous-Neumann boundary values."""

    result = jnp.zeros_like(field)
    if field.shape[axis] <= 2:
        return result
    denominator = 0.5 * (widths[:-2] + 2.0 * widths[1:-1] + widths[2:])
    shape = [1, 1, 1]
    shape[axis] = denominator.size
    if axis == 1:
        interior = (field[:, 2:, :] - field[:, :-2, :]) / denominator.reshape(shape)
        return result.at[:, 1:-1, :].set(interior)
    interior = (field[:, :, 2:] - field[:, :, :-2]) / denominator.reshape(shape)
    return result.at[:, :, 1:-1].set(interior)


def _nonuniform_axis_laplacian(
    field: jnp.ndarray, widths: jnp.ndarray, *, axis: int, mode: str
) -> jnp.ndarray:
    """Finite-volume second derivative using physical face/center distances."""

    center_distance = 0.5 * (widths[:-1] + widths[1:])
    if axis == 1:
        flux = jnp.zeros((field.shape[0], field.shape[1] + 1, field.shape[2]), dtype=field.dtype)
        flux = flux.at[:, 1:-1, :].set((field[:, 1:, :] - field[:, :-1, :]) / center_distance[None, :, None])
        if mode == "dirichlet":
            flux = flux.at[:, 0, :].set(field[:, 0, :] / (0.5 * widths[0]))
            flux = flux.at[:, -1, :].set(-field[:, -1, :] / (0.5 * widths[-1]))
        return (flux[:, 1:, :] - flux[:, :-1, :]) / widths[None, :, None]
    flux = jnp.zeros((field.shape[0], field.shape[1], field.shape[2] + 1), dtype=field.dtype)
    flux = flux.at[:, :, 1:-1].set((field[:, :, 1:] - field[:, :, :-1]) / center_distance[None, None, :])
    if mode == "dirichlet":
        flux = flux.at[:, :, 0].set(field[:, :, 0] / (0.5 * widths[0]))
        flux = flux.at[:, :, -1].set(-field[:, :, -1] / (0.5 * widths[-1]))
    return (flux[:, :, 1:] - flux[:, :, :-1]) / widths[None, None, :]


def _neighbor_fields(
    field: jnp.ndarray,
    *,
    mode_x: str,
    mode_y: str,
    mode_z: str,
    sharding: NamedSharding | None = None,
) -> tuple[jnp.ndarray, ...]:
    def shifted(value, axis, forward, mode):
        moved = jnp.moveaxis(value, axis, 0)
        edge = moved[-1 if forward else 0]
        boundary = edge if mode == "neumann" else jnp.zeros_like(edge)
        parts = (moved[1:], boundary[None]) if forward else (boundary[None], moved[:-1])
        return jnp.moveaxis(jnp.concatenate(parts), 0, axis)

    if sharding is None:
        x_west, x_east = (shifted(field, 0, side, mode_x) for side in (False, True))
    else:  # pragma: no cover - forced-device/hardware gates
        count = sharding.mesh.size

        def axial(value):
            index = jax.lax.axis_index("x")
            west = jax.lax.ppermute(value[-1], "x", tuple((i, i + 1) for i in range(count - 1)))
            east = jax.lax.ppermute(value[0], "x", tuple((i, i - 1) for i in range(1, count)))
            if mode_x == "neumann":
                west, east = (
                    jnp.where(index == 0, value[0], west),
                    jnp.where(index == count - 1, value[-1], east),
                )
            return (jnp.concatenate((west[None], value[:-1])), jnp.concatenate((value[1:], east[None])))

        x_west, x_east = jax.shard_map(
            axial,
            mesh=sharding.mesh,
            in_specs=sharding.spec,
            out_specs=(sharding.spec, sharding.spec),
            check_vma=False,
        )(field)
    y_south, y_north = (shifted(field, 1, side, mode_y) for side in (False, True))
    z_bottom, z_top = (shifted(field, 2, side, mode_z) for side in (False, True))
    return x_west, x_east, y_south, y_north, z_bottom, z_top


def _laplacian_3d(
    field: jnp.ndarray,
    *,
    dx: float,
    dy: float | jnp.ndarray,
    dz: float | jnp.ndarray,
    mode_x: str = "neumann",
    mode_y: str = "dirichlet",
    mode_z: str = "dirichlet",
) -> jnp.ndarray:
    x_west, x_east, y_south, y_north, z_bottom, z_top = _neighbor_fields(
        field,
        mode_x=mode_x,
        mode_y=mode_y,
        mode_z=mode_z,
    )
    x_term = (x_west - 2.0 * field + x_east) / max(dx**2, 1.0e-12)
    dy_values = jnp.asarray(dy)
    dz_values = jnp.asarray(dz)
    y_term = (
        (y_south - 2.0 * field + y_north) / jnp.maximum(dy_values**2, 1.0e-12)
        if dy_values.ndim == 0
        else _nonuniform_axis_laplacian(
            field,
            _spacing_vector(dy_values, field.shape[1], dtype=field.dtype),
            axis=1,
            mode=mode_y,
        )
    )
    z_term = (
        (z_bottom - 2.0 * field + z_top) / jnp.maximum(dz_values**2, 1.0e-12)
        if dz_values.ndim == 0
        else _nonuniform_axis_laplacian(
            field,
            _spacing_vector(dz_values, field.shape[2], dtype=field.dtype),
            axis=2,
            mode=mode_z,
        )
    )
    return x_term + y_term + z_term


def _gradient_3d(
    field: jnp.ndarray,
    *,
    dx: float,
    dy: float | jnp.ndarray,
    dz: float | jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    x_west, x_east, y_south, y_north, z_bottom, z_top = _neighbor_fields(
        field,
        mode_x="neumann",
        mode_y="neumann",
        mode_z="neumann",
    )
    d_dx = (x_east - x_west) / max(2.0 * dx, 1.0e-12)
    dy_values = jnp.asarray(dy)
    dz_values = jnp.asarray(dz)
    d_dy = (
        (y_north - y_south) / max(2.0 * float(dy_values), 1.0e-12)
        if dy_values.ndim == 0
        else _nonuniform_axis_gradient(
            field,
            _spacing_vector(dy_values, field.shape[1], dtype=field.dtype),
            axis=1,
        )
    )
    d_dz = (
        (z_top - z_bottom) / max(2.0 * float(dz_values), 1.0e-12)
        if dz_values.ndim == 0
        else _nonuniform_axis_gradient(
            field,
            _spacing_vector(dz_values, field.shape[2], dtype=field.dtype),
            axis=2,
        )
    )
    return d_dx, d_dy, d_dz


def _enforce_velocity_bc_3d(field: jnp.ndarray, active_mask: jnp.ndarray | None = None) -> jnp.ndarray:
    bounded = field.at[:, 0, :].set(0.0)
    bounded = bounded.at[:, -1, :].set(0.0)
    bounded = bounded.at[:, :, 0].set(0.0)
    bounded = bounded.at[:, :, -1].set(0.0)
    if bounded.shape[0] > 1:
        bounded = bounded.at[0, :, :].set(bounded[1, :, :])
        bounded = bounded.at[-1, :, :].set(bounded[-2, :, :])
    if active_mask is not None:
        bounded = jnp.where(active_mask, bounded, 0.0)
    return bounded


def _enforce_stationwise_flow_rate_3d(
    u: jnp.ndarray,
    *,
    active_mask: jnp.ndarray,
    cell_area: jnp.ndarray,
    target_flow_rate: float | None = None,
    relaxation: float = 1.0,
) -> jnp.ndarray:
    active_area = jnp.maximum(jnp.sum(jnp.where(active_mask, cell_area, 0.0), axis=(1, 2)), 1.0e-20)
    station_flow_rate = jnp.sum(jnp.where(active_mask, u * cell_area, 0.0), axis=(1, 2))
    if target_flow_rate is None:
        target = jnp.mean(station_flow_rate)
    else:
        target = jnp.asarray(target_flow_rate, dtype=u.dtype)
    correction = (relaxation * (station_flow_rate - target) / active_area)[:, None, None]
    corrected = jnp.where(active_mask, u - correction, 0.0)
    return corrected


def _apply_fixed_flow_pressure_constraint(
    u: jnp.ndarray,
    *,
    unit_pressure_response: jnp.ndarray,
    active_mask: jnp.ndarray,
    cell_area: jnp.ndarray,
    target_flow_rate: float,
    base_pressure_loss_gradient: float | jnp.ndarray = 0.0,
    validate_response: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply the stationwise pressure multiplier that enforces fixed flow.

    ``unit_pressure_response`` is the boundary-conditioned velocity increment
    caused by one unit of positive pressure-loss gradient over the current time
    step.  The returned gradient therefore has the unambiguous sign convention
    ``-dp/dx > 0`` for a pressure-driven flow.
    """

    flow = jnp.sum(jnp.where(active_mask, u * cell_area, 0.0), axis=(1, 2))
    response = jnp.sum(
        jnp.where(active_mask, unit_pressure_response * cell_area, 0.0),
        axis=(1, 2),
    )
    if validate_response and float(jnp.min(jnp.abs(response))) <= 1.0e-20:
        raise ValueError("Fixed-flow pressure response must be nonzero")
    multiplier = (jnp.asarray(target_flow_rate, dtype=u.dtype) - flow) / response
    corrected = jnp.where(
        active_mask,
        u + multiplier[:, None, None] * unit_pressure_response,
        0.0,
    )
    pressure_loss_gradient = jnp.asarray(base_pressure_loss_gradient, dtype=u.dtype) + multiplier
    return corrected, pressure_loss_gradient


def _cross_duct_pressure_difference(
    p: jnp.ndarray,
    *,
    active_mask: jnp.ndarray,
    magnetic_axis: int = 1,
    side_axis: int = 2,
) -> jnp.ndarray:
    """Return side-minus-top pressure at adjacent wall-midpoint taps."""

    if {magnetic_axis, side_axis} != {1, 2}:
        raise ValueError("magnetic_axis and side_axis must be distinct members of {1, 2}")

    if len(active_mask.addressable_shards) == 1 and not bool(jnp.any(active_mask)):
        raise ValueError("Pressure difference requires active fluid boundary cells")
    return _cross_duct_pressure_difference_kernel(
        p,
        active_mask,
        magnetic_axis=magnetic_axis,
        side_axis=side_axis,
    )


def _cross_duct_pressure_difference_kernel(
    p: jnp.ndarray,
    active_mask: jnp.ndarray,
    *,
    magnetic_axis: int,
    side_axis: int,
) -> jnp.ndarray:
    """JIT the wall-tap reductions so every device produces its local rows."""

    def high_wall_midpoint(wall_axis: int) -> jnp.ndarray:
        orthogonal_axis = 2 if wall_axis == 1 else 1
        active_line = jnp.any(active_mask, axis=orthogonal_axis)
        indices = jnp.arange(p.shape[wall_axis])
        high = jnp.max(jnp.where(active_line, indices[None, :], -1), axis=1)
        if wall_axis == 2:
            values = jnp.take_along_axis(p, high[:, None, None], axis=2)[:, :, 0]
            mask = jnp.take_along_axis(active_mask, high[:, None, None], axis=2)[:, :, 0]
        else:
            values = jnp.take_along_axis(p, high[:, None, None], axis=1)[:, 0, :]
            mask = jnp.take_along_axis(active_mask, high[:, None, None], axis=1)[:, 0, :]
        orthogonal_size = values.shape[1]
        orthogonal_index = jnp.arange(orthogonal_size, dtype=p.dtype)[None, :]
        center_index = 0.5 * (orthogonal_size - 1)
        distance = jnp.where(mask, jnp.abs(orthogonal_index - center_index), jnp.inf)
        closest = jnp.min(distance, axis=1, keepdims=True)
        tap_mask = mask & (distance <= closest + 1.0e-12)
        return jnp.sum(jnp.where(tap_mask, values, 0.0), axis=1) / jnp.maximum(jnp.sum(tap_mask, axis=1), 1)

    top_pressure = high_wall_midpoint(magnetic_axis)
    side_pressure = high_wall_midpoint(side_axis)
    return side_pressure - top_pressure


def _normalized_pressure_observable_update(
    current: jnp.ndarray,
    previous: jnp.ndarray,
    magnetic_energy: jnp.ndarray,
) -> jnp.ndarray:
    """Measure a pressure-observable update in magnetic-pressure units."""

    scale = jnp.maximum(1.0, jnp.max(jnp.asarray(magnetic_energy)))
    return jnp.max(jnp.abs(current - previous)) / scale


def _gauge_invariant_scalar_update(
    current: jnp.ndarray,
    previous: jnp.ndarray,
    volume: jnp.ndarray,
    *,
    scale: float,
) -> jnp.ndarray:
    """Measure an update after removing its volume-weighted constant gauge."""

    delta = current - previous
    delta = delta - jnp.sum(delta * volume) / jnp.sum(volume)
    return jnp.max(jnp.abs(delta)) / max(scale, 1.0e-20)


def _variable_coefficient_residual_3d(
    field: jnp.ndarray,
    rhs: jnp.ndarray,
    conductivity: jnp.ndarray,
    *,
    dx: float,
    dy: float | jnp.ndarray,
    dz: float | jnp.ndarray,
) -> jnp.ndarray:
    x_west, x_east, y_south, y_north, z_bottom, z_top = _neighbor_fields(
        field,
        mode_x="neumann",
        mode_y="neumann",
        mode_z="neumann",
    )
    coef_x_w, coef_x_e, coef_y_s, coef_y_n, coef_z_b, coef_z_t = _variable_diffusion_coefficients_3d(
        conductivity, dx=dx, dy=dy, dz=dz
    )
    operator = (
        coef_x_w * (x_west - field)
        + coef_x_e * (x_east - field)
        + coef_y_s * (y_south - field)
        + coef_y_n * (y_north - field)
        + coef_z_b * (z_bottom - field)
        + coef_z_t * (z_top - field)
    )
    return operator - rhs


def _variable_diffusion_coefficients_3d(
    conductivity: jnp.ndarray,
    *,
    dx: float,
    dy: float | jnp.ndarray,
    dz: float | jnp.ndarray,
    validated_spacing: bool = False,
    thin_wall_fluid_mask: jnp.ndarray | None = None,
    axial_coefficients: tuple[jnp.ndarray, jnp.ndarray] | None = None,
) -> tuple[jnp.ndarray, ...]:
    """Return normalized diffusion coefficients; axial pairs are cell-aligned."""

    nx, ny, nz = conductivity.shape
    spacing = _coerce_spacing_vector if validated_spacing else _spacing_vector
    dy_widths = spacing(dy, ny, dtype=conductivity.dtype)
    dz_widths = spacing(dz, nz, dtype=conductivity.dtype)
    if axial_coefficients is None:
        sigma_x = _harmonic_mean(conductivity[1:], conductivity[:-1])
        scaled = sigma_x / max(dx**2, 1.0e-12)
        coef_x_w = jnp.concatenate([jnp.zeros_like(conductivity[:1]), scaled], axis=0)
        coef_x_e = jnp.concatenate([scaled, jnp.zeros_like(conductivity[-1:])], axis=0)
    else:
        coef_x_w, coef_x_e = axial_coefficients

    sigma_y = _distance_weighted_harmonic_mean(
        conductivity[:, 1:, :],
        conductivity[:, :-1, :],
        dy_widths[None, 1:, None],
        dy_widths[None, :-1, None],
    )
    if thin_wall_fluid_mask is not None:
        sigma_y = _thin_wall_interface_mean(
            conductivity[:, 1:, :],
            conductivity[:, :-1, :],
            dy_widths[None, 1:, None],
            dy_widths[None, :-1, None],
            thin_wall_fluid_mask[:, 1:, :],
            thin_wall_fluid_mask[:, :-1, :],
        )
    y_distance = 0.5 * (dy_widths[:-1] + dy_widths[1:])
    coef_y_s = (
        jnp.zeros_like(conductivity)
        .at[:, 1:, :]
        .set(sigma_y / (dy_widths[None, 1:, None] * y_distance[None, :, None]))
    )
    coef_y_n = (
        jnp.zeros_like(conductivity)
        .at[:, :-1, :]
        .set(sigma_y / (dy_widths[None, :-1, None] * y_distance[None, :, None]))
    )

    sigma_z = _distance_weighted_harmonic_mean(
        conductivity[:, :, 1:],
        conductivity[:, :, :-1],
        dz_widths[None, None, 1:],
        dz_widths[None, None, :-1],
    )
    if thin_wall_fluid_mask is not None:
        sigma_z = _thin_wall_interface_mean(
            conductivity[:, :, 1:],
            conductivity[:, :, :-1],
            dz_widths[None, None, 1:],
            dz_widths[None, None, :-1],
            thin_wall_fluid_mask[:, :, 1:],
            thin_wall_fluid_mask[:, :, :-1],
        )
    z_distance = 0.5 * (dz_widths[:-1] + dz_widths[1:])
    coef_z_b = (
        jnp.zeros_like(conductivity)
        .at[:, :, 1:]
        .set(sigma_z / (dz_widths[None, None, 1:] * z_distance[None, None, :]))
    )
    coef_z_t = (
        jnp.zeros_like(conductivity)
        .at[:, :, :-1]
        .set(sigma_z / (dz_widths[None, None, :-1] * z_distance[None, None, :]))
    )
    return coef_x_w, coef_x_e, coef_y_s, coef_y_n, coef_z_b, coef_z_t


def _projection_pressure_correction_3d(
    rhs: jnp.ndarray,
    *,
    dx: float,
    dy: float | jnp.ndarray,
    dz: float | jnp.ndarray,
    iterations: int,
    tolerance: float,
) -> jnp.ndarray:
    """Apply the collocated projection stencil with SOLVAX fixed-point control."""

    dy_widths = _spacing_vector(dy, rhs.shape[1], dtype=rhs.dtype)
    dz_widths = _spacing_vector(dz, rhs.shape[2], dtype=rhs.dtype)
    volume = jnp.broadcast_to(dy_widths[None, :, None] * dz_widths[None, None, :], rhs.shape)
    compatible_rhs = rhs - jnp.sum(rhs * volume) / jnp.sum(volume)
    coefficients = _variable_diffusion_coefficients_3d(jnp.ones_like(rhs), dx=dx, dy=dy_widths, dz=dz_widths)
    diagonal = jnp.maximum(sum(coefficients), 1.0e-12)

    def update(field):
        neighbors = _neighbor_fields(
            field,
            mode_x="neumann",
            mode_y="neumann",
            mode_z="neumann",
        )
        corrected = (
            sum(coefficient * neighbor for coefficient, neighbor in zip(coefficients, neighbors))
            - compatible_rhs
        ) / diagonal
        return corrected - jnp.sum(corrected * volume) / jnp.sum(volume)

    solution = fixed_point_iteration(
        update,
        jnp.zeros_like(rhs),
        residual_norm=lambda field: jnp.max(
            jnp.abs(
                _variable_coefficient_residual_3d(
                    field,
                    compatible_rhs,
                    jnp.ones_like(rhs),
                    dx=dx,
                    dy=dy_widths,
                    dz=dz_widths,
                )
            )
        ),
        rtol=0.0,
        atol=tolerance,
        max_steps=iterations,
    )
    return solution.x


def _rectangular_fluid_bounds(fluid_mask: jnp.ndarray) -> tuple[int, int, int, int]:
    cross_mask = np.asarray(jnp.any(fluid_mask, axis=0), dtype=bool)
    y_active = np.flatnonzero(np.any(cross_mask, axis=1))
    z_active = np.flatnonzero(np.any(cross_mask, axis=0))
    if not y_active.size or not z_active.size:
        raise ValueError("Face-flux projection requires a nonempty fluid subdomain")
    y0, y1 = int(y_active[0]), int(y_active[-1]) + 1
    z0, z1 = int(z_active[0]), int(z_active[-1]) + 1
    rectangular = np.zeros_like(cross_mask)
    rectangular[y0:y1, z0:z1] = True
    if not np.array_equal(cross_mask, rectangular):
        raise ValueError("Face-flux projection requires a rectangular fluid mask")
    return y0, y1, z0, z1


def _axial_mean_preconditioner_3d(
    volume: jnp.ndarray,
    coef_x_w: jnp.ndarray,
    coef_x_e: jnp.ndarray,
    *,
    gauge: bool = True,
    field_sharding: NamedSharding | None = None,
):
    """Invert the Galerkin operator for cross-section-constant axial modes."""
    _, ny, nz = volume.shape
    normalization = math.sqrt(ny * nz)
    west, east = (
        -jnp.sum(volume * coefficient, axis=(1, 2)) / (ny * nz) for coefficient in (coef_x_w, coef_x_e)
    )
    diagonal, gauge_vector = -(west + east), jnp.sum(volume, axis=(1, 2)) / normalization
    if gauge:
        diagonal, east = diagonal.at[0].set(1.0), east.at[0].set(0.0)

    def solve_coarse(system):
        lower, center, upper, weights, reduced = jnp.moveaxis(system, -1, 0)
        if gauge:
            coefficient = jnp.sum(reduced) / jnp.sum(weights)
            reduced = (reduced - coefficient * weights).at[0].set(0.0)
        correction = tridiagonal_solve(lower, center, upper, reduced)
        if gauge:
            correction += (
                coefficient * normalization * jnp.sum(weights) - jnp.vdot(weights, correction)
            ) / jnp.sum(weights)
        return correction

    replicated = None if field_sharding is None else NamedSharding(field_sharding.mesh, P())
    coarse_solve = jax.jit(solve_coarse, in_shardings=replicated, out_shardings=replicated)

    def apply(residual: jnp.ndarray) -> jnp.ndarray:
        reduced = jnp.sum(residual, axis=(1, 2)) / normalization
        correction = coarse_solve(jnp.stack((west, diagonal, east, gauge_vector, reduced), axis=-1))
        return jnp.broadcast_to(correction[:, None, None] / normalization, residual.shape)

    return apply


def _region_interpolation(
    widths: jnp.ndarray,
    lower: int,
    upper: int,
    stride: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, int]:
    """Build physical-coordinate interpolation without crossing an interface."""

    centers = jnp.cumsum(widths) - 0.5 * widths
    left_parts = []
    right_parts = []
    weight_parts = []
    offset = 0
    for start, stop in ((0, lower), (lower, upper), (upper, widths.size)):
        size = stop - start
        count = max(1, math.ceil(size / stride))
        local_widths = widths[start:stop]
        local_centers = centers[start:stop]
        groups = jnp.minimum(jnp.arange(size) * count // size, count - 1)
        coarse_widths = jnp.zeros(count, dtype=widths.dtype).at[groups].add(local_widths)
        coarse_centers = (
            jnp.zeros(count, dtype=widths.dtype).at[groups].add(local_widths * local_centers) / coarse_widths
        )
        high = jnp.clip(jnp.searchsorted(coarse_centers, local_centers), 1, count - 1)
        high = jnp.where(count == 1, 0, high)
        low = jnp.where(count == 1, 0, high - 1)
        span = jnp.where(count == 1, 1.0, coarse_centers[high] - coarse_centers[low])
        left_parts.append(low + offset)
        right_parts.append(high + offset)
        weight_parts.append(jnp.clip((local_centers - coarse_centers[low]) / span, 0.0, 1.0))
        offset += count
    return (
        jnp.concatenate(left_parts),
        jnp.concatenate(right_parts),
        jnp.concatenate(weight_parts),
        offset,
    )


def _transverse_modal_correction_3d(
    volume: jnp.ndarray,
    coefficient: jnp.ndarray,
    coefficients: tuple[jnp.ndarray, ...],
    *,
    dx: float,
    dy: jnp.ndarray,
    dz: jnp.ndarray,
    fluid_bounds: tuple[int, int, int, int],
    stride: int,
    sharding: NamedSharding | None = None,
):
    """Return a global fast-diagonalization Galerkin correction.

    Only the restricted coarse grid is replicated when the fine field is
    sharded; the solved correction is repartitioned before prolongation.
    """

    nx = volume.shape[0]
    y0, y1, z0, z1 = fluid_bounds
    yl, yr, yw, ncy = _region_interpolation(dy, y0, y1, stride)
    zl, zr, zw, ncz = _region_interpolation(dz, z0, z1, stride)
    coarse_shape = (nx, ncy, ncz)
    coarse_cross_shape = coarse_shape[1:]
    coarse_cross_zero = jnp.zeros(coarse_cross_shape, dtype=volume.dtype)
    yw = yw[:, None]
    zw = zw[None, :]

    def prolong_cross(coarse: jnp.ndarray) -> jnp.ndarray:
        fine_y = (1.0 - yw) * coarse[yl, :] + yw * coarse[yr, :]
        return (1.0 - zw) * fine_y[:, zl] + zw * fine_y[:, zr]

    def restrict_cross(fine: jnp.ndarray) -> jnp.ndarray:
        return jax.linear_transpose(prolong_cross, coarse_cross_zero)(fine)[0]

    volume_cross = volume[0]

    def transverse_matvec(field: jnp.ndarray) -> jnp.ndarray:
        south, north, bottom, top = _neighbor_fields(
            field[None], mode_x="neumann", mode_y="neumann", mode_z="neumann"
        )[2:]
        diffusion = (
            coefficients[2][0] * (south[0] - field)
            + coefficients[3][0] * (north[0] - field)
            + coefficients[4][0] * (bottom[0] - field)
            + coefficients[5][0] * (top[0] - field)
        )
        return -volume_cross * diffusion

    def galerkin(apply: Callable[[jnp.ndarray], jnp.ndarray], coarse: jnp.ndarray):
        return restrict_cross(apply(prolong_cross(coarse)))

    coarse_size = ncy * ncz
    basis = jnp.eye(coarse_size, dtype=volume.dtype).reshape((coarse_size, ncy, ncz))
    stiffness = jax.vmap(lambda column: galerkin(transverse_matvec, column).reshape(-1))(basis).T
    mass = jax.vmap(
        lambda column: galerkin(
            lambda field: volume_cross * coefficient[0] * field / dx**2,
            column,
        ).reshape(-1)
    )(basis).T
    mass_factor = jnp.linalg.cholesky(0.5 * (mass + mass.T))
    whitened_left = solve_triangular(mass_factor, stiffness, lower=True)
    whitened = solve_triangular(mass_factor, whitened_left.T, lower=True).T
    eigenvalues, modes = jnp.linalg.eigh(0.5 * (whitened + whitened.T))
    inverse_modes = solve_triangular(mass_factor.T, modes, lower=False)
    coarse_volume = restrict_cross(volume_cross).reshape(-1)
    whitened_gauge = solve_triangular(mass_factor, coarse_volume, lower=True)
    gauge_eigenvalue = jnp.dot(modes[:, 0], whitened_gauge) ** 2 / jnp.sum(volume_cross)

    axial_eigenvalues = 2.0 - 2.0 * jnp.cos(jnp.pi * jnp.arange(nx, dtype=volume.dtype) / nx)
    denominators = axial_eigenvalues[:, None] + jnp.maximum(eigenvalues[None], 0.0)
    denominators = denominators.at[0, 0].add(gauge_eigenvalue)

    def solve_global(rhs: jnp.ndarray) -> jnp.ndarray:
        transformed = dct(rhs, type=2, axis=0, norm="ortho").reshape(nx, -1)
        spectral = transformed @ inverse_modes
        solved = (spectral / denominators) @ inverse_modes.T
        return idct(solved.reshape(coarse_shape), type=2, axis=0, norm="ortho")

    coarse_solve = solve_global

    def reshard_coarse(value):
        return value

    if sharding is not None:  # pragma: no cover - exercised by hardware gates
        replicated = NamedSharding(sharding.mesh, P())
        coarse_solve = jax.jit(solve_global, in_shardings=replicated, out_shardings=replicated)
        reshard_coarse = jax.jit(
            lambda value: value,
            in_shardings=replicated,
            out_shardings=sharding,
        )
    coarse_zero = jnp.zeros(coarse_shape, dtype=volume.dtype)

    def prolong(coarse: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(prolong_cross)(coarse)

    def restrict(fine: jnp.ndarray) -> jnp.ndarray:
        return jax.linear_transpose(prolong, coarse_zero)(fine)[0]

    def apply(residual: jnp.ndarray) -> jnp.ndarray:
        return prolong(reshard_coarse(coarse_solve(restrict(residual))))

    return apply


def _finalize_local_pressure_solve(
    solution,
    *,
    linear_rhs: jnp.ndarray,
    matvec: Callable[[jnp.ndarray], jnp.ndarray],
    local_residual_fn: Callable[[jnp.ndarray], jnp.ndarray],
    volume: jnp.ndarray,
    precondition: Callable[[jnp.ndarray], jnp.ndarray],
    iterations: int,
    effective_atol: float,
    local_tolerance: float | None,
    single_reduction: bool = False,
    gauge: bool = True,
) -> tuple[jnp.ndarray, ...]:
    """Finalize a pressure solve and optionally apply one refinement correction."""

    volume_sum = jnp.sum(volume)

    def gauge_fixed(field: jnp.ndarray) -> jnp.ndarray:
        if gauge:
            return field - jnp.sum(field * volume) / volume_sum
        return field

    field = gauge_fixed(solution.x)
    local_residual = local_residual_fn(field)
    if local_tolerance is None:
        return (
            field,
            solution.residual_norm,
            solution.converged,
            solution.relative_residual_norm,
            solution.iterations,
            solution.status,
            jnp.max(jnp.abs(local_residual)),
        )

    needs_refinement = jnp.max(jnp.abs(local_residual)) > local_tolerance

    def refine(_):
        correction = pcg_linear_solve(
            matvec,
            linear_rhs - matvec(field),
            x0=jnp.zeros_like(field),
            precond=precondition,
            transpose_precond=precondition,
            rtol=0.0,
            atol=effective_atol,
            max_steps=iterations,
            transpose_rtol=0.0,
            transpose_atol=effective_atol,
            transpose_max_steps=iterations,
            single_reduction=single_reduction,
        )
        return correction.x, correction.iterations, correction.status

    correction, correction_iterations, correction_status = jax.lax.cond(
        needs_refinement,
        refine,
        lambda _: (
            jnp.zeros_like(field),
            jnp.zeros_like(solution.iterations),
            jnp.ones_like(solution.status),
        ),
        operand=None,
    )
    field = gauge_fixed(field + correction)
    local_residual = local_residual_fn(field)
    final_linear_residual = linear_rhs - matvec(field)
    residual_norm = jnp.linalg.norm(final_linear_residual)
    relative_residual_norm = residual_norm / jnp.maximum(jnp.linalg.norm(linear_rhs), jnp.asarray(1.0e-30))
    converged = jnp.max(jnp.abs(local_residual)) <= local_tolerance
    return (
        field,
        residual_norm,
        converged,
        relative_residual_norm,
        solution.iterations + correction_iterations,
        jnp.where(converged, jnp.asarray(1), correction_status),
        jnp.max(jnp.abs(local_residual)),
    )

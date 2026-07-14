"""Three-dimensional inductionless fringing-field solvers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from functools import lru_cache, partial
import hashlib
import math

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.fft import dct, idct
from jax.scipy.linalg import solve_triangular
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from solvax import (
    KrylovSolution,
    additive_preconditioner,
    aitken_relaxation,
    anderson_mixing,
    block_thomas_factor,
    block_thomas_solve,
    cyclic_tridiagonal_solve,
    gmres,
    linear_solve,
    pcg_linear_solve,
    tridiagonal_solve,
)

try:
    from scipy import sparse
    from scipy.sparse.linalg import spsolve as sparse_spsolve
except Exception:  # pragma: no cover - SciPy should be present in shipped environments.
    sparse = None
    sparse_spsolve = None

from .cases import _ha_to_b, make_hunt_case, make_shercliff_case
from .core import Solution
from .field_models import load_tabulated_field, sample_tabulated_field_volume
from .mesh import (
    generate_bent_pipe_mesh,
    generate_layered_duct_mesh,
    generate_pipe_ogrid_mesh,
    generate_rect_duct_mesh,
)
from .physics import build_material_fields
from .specs import (
    BoundaryCondition,
    CaseSpec,
    GeometrySpec,
    MagneticFieldSpec,
    OutputSpec,
    RegionSpec,
    SolverConfig,
    TimeStepperConfig,
)
from .solvers import solve_steady
from .validation import validation_summary
from ._fringing_types import (
    ExtrudedFieldBundle,
    ExtrudedInductionlessProblem,
    ExtrudedInductionlessSolution,
    ExtrudedInductionlessValidation,
    ExtrudedIterationProgress,
    FringingProfile,
)


MAGNETIC_OBSTACLE_LITERATURE_REFERENCES: dict[str, dict[str, object]] = {
    "cuevas_smolentsev_abdou_q2d": {
        "label": "Cuevas, Smolentsev, Abdou quasi-2D magnetic obstacle",
        "url": "https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/on-the-flow-past-a-magnetic-obstacle/F4185BE5315273DBA9D1C53DD49990AA",
        "model_class": "quasi_2d_rectangular_channel",
        "required_observables": [
            "centerline_velocity_deficit",
            "wake_recovery",
            "pressure_drop_or_drag_proxy",
            "current_closure",
            "vorticity_or_recirculation_structure",
        ],
    },
    "votyakov_zienicke_kolesnikov_jfm": {
        "label": "Votyakov, Zienicke, Kolesnikov constrained magnetic obstacle",
        "url": "https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/constrained-flow-around-a-magnetic-obstacle/DFD706B066E0B0C7E8598544E1783BC0",
        "model_class": "3d_rectangular_channel_with_experimental_comparison",
        "required_observables": [
            "centerline_velocity_deficit",
            "wake_recovery",
            "pressure_drop",
            "cross_sectional_distortion",
            "recirculation_topology",
        ],
    },
    "andreev_kolesnikov_thess_experiment": {
        "label": "Andreev, Kolesnikov, Thess nonuniform-field channel experiment",
        "url": "https://doi.org/10.1063/1.2213639",
        "model_class": "rectangular_channel_experiment",
        "required_observables": [
            "ultrasound_velocity_profiles",
            "magnet_position_response",
            "pressure_drop",
            "wake_structure",
        ],
    },
}

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


# Frozen in both ALEX Benchmark B specifications for mass and current closure.
ALEX_BALANCE_TOLERANCE = 1.0e-3
ALEX_B2_STEADY_STEPS = 3
ALEX_B2_CANONICAL_SHELL_THICKNESS = 0.02
# The frozen Aitken ceiling is monotone on the accepted medium B2 restart and
# shortens tight-tolerance continuation without changing unconverged states.
ALEX_B2_SETTLED_RELAXATION = 2.0


def _sustained_convergence(streak: int, passed: bool) -> tuple[int, bool]:
    """Require repeated passing updates before accepting an oscillatory B2 map."""

    streak = streak + 1 if passed else 0
    return streak, streak >= ALEX_B2_STEADY_STEPS


def _canonical_shell_widths(widths: jnp.ndarray, lower: int, upper: int) -> jnp.ndarray:
    """Map explicit wall cells to the frozen B2 mixed-dimensional shell."""

    if lower:
        widths = widths.at[:lower].multiply(
            ALEX_B2_CANONICAL_SHELL_THICKNESS / jnp.sum(widths[:lower])
        )
    if upper < widths.size:
        widths = widths.at[upper:].multiply(
            ALEX_B2_CANONICAL_SHELL_THICKNESS / jnp.sum(widths[upper:])
        )
    return widths


def magnetic_obstacle_literature_reference_cases() -> dict[str, dict[str, object]]:
    return {
        key: {**value} for key, value in MAGNETIC_OBSTACLE_LITERATURE_REFERENCES.items()
    }


def _broadcast_station_profile(values: jnp.ndarray, ny: int, nz: int) -> jnp.ndarray:
    return jnp.broadcast_to(
        jnp.asarray(values, dtype=float)[:, None, None], (values.shape[0], ny, nz)
    )


def _broadcast_cross_section(values: jnp.ndarray, nx: int) -> jnp.ndarray:
    return jnp.broadcast_to(
        jnp.asarray(values, dtype=float)[None, :, :], (nx,) + tuple(values.shape)
    )


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


def _anderson_extruded_state(
    iterates: list[jnp.ndarray],
    residuals: list[jnp.ndarray],
    *,
    history_size: int,
    regularization: float,
    damping: float,
) -> jnp.ndarray:
    """Accelerate an ALEX outer map while retaining its affine constraints."""

    return anderson_mixing(
        jnp.stack(iterates[-history_size:]),
        jnp.stack(residuals[-history_size:]),
        regularization=regularization,
        damping=damping,
    )


def _coerce_spacing_vector(
    spacing: float | jnp.ndarray, size: int, *, dtype
) -> jnp.ndarray:
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


def _nonuniform_axis_gradient(
    field: jnp.ndarray, widths: jnp.ndarray, *, axis: int
) -> jnp.ndarray:
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
        flux = jnp.zeros(
            (field.shape[0], field.shape[1] + 1, field.shape[2]), dtype=field.dtype
        )
        flux = flux.at[:, 1:-1, :].set(
            (field[:, 1:, :] - field[:, :-1, :]) / center_distance[None, :, None]
        )
        if mode == "dirichlet":
            flux = flux.at[:, 0, :].set(field[:, 0, :] / (0.5 * widths[0]))
            flux = flux.at[:, -1, :].set(-field[:, -1, :] / (0.5 * widths[-1]))
        return (flux[:, 1:, :] - flux[:, :-1, :]) / widths[None, :, None]
    flux = jnp.zeros(
        (field.shape[0], field.shape[1], field.shape[2] + 1), dtype=field.dtype
    )
    flux = flux.at[:, :, 1:-1].set(
        (field[:, :, 1:] - field[:, :, :-1]) / center_distance[None, None, :]
    )
    if mode == "dirichlet":
        flux = flux.at[:, :, 0].set(field[:, :, 0] / (0.5 * widths[0]))
        flux = flux.at[:, :, -1].set(-field[:, :, -1] / (0.5 * widths[-1]))
    return (flux[:, :, 1:] - flux[:, :, :-1]) / widths[None, None, :]


def _neighbor_fields(
    field: jnp.ndarray, *, mode_x: str, mode_y: str, mode_z: str
) -> tuple[jnp.ndarray, ...]:
    x_west = (
        jnp.concatenate([field[:1], field[:-1]], axis=0)
        if mode_x == "neumann"
        else jnp.concatenate([jnp.zeros_like(field[:1]), field[:-1]], axis=0)
    )
    x_east = (
        jnp.concatenate([field[1:], field[-1:]], axis=0)
        if mode_x == "neumann"
        else jnp.concatenate([field[1:], jnp.zeros_like(field[-1:])], axis=0)
    )
    y_south = (
        jnp.concatenate([field[:, :1, :], field[:, :-1, :]], axis=1)
        if mode_y == "neumann"
        else jnp.concatenate(
            [jnp.zeros_like(field[:, :1, :]), field[:, :-1, :]], axis=1
        )
    )
    y_north = (
        jnp.concatenate([field[:, 1:, :], field[:, -1:, :]], axis=1)
        if mode_y == "neumann"
        else jnp.concatenate(
            [field[:, 1:, :], jnp.zeros_like(field[:, -1:, :])], axis=1
        )
    )
    z_bottom = (
        jnp.concatenate([field[:, :, :1], field[:, :, :-1]], axis=2)
        if mode_z == "neumann"
        else jnp.concatenate(
            [jnp.zeros_like(field[:, :, :1]), field[:, :, :-1]], axis=2
        )
    )
    z_top = (
        jnp.concatenate([field[:, :, 1:], field[:, :, -1:]], axis=2)
        if mode_z == "neumann"
        else jnp.concatenate(
            [field[:, :, 1:], jnp.zeros_like(field[:, :, -1:])], axis=2
        )
    )
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
        (y_south - 2.0 * field + y_north) / max(float(dy_values) ** 2, 1.0e-12)
        if dy_values.ndim == 0
        else _nonuniform_axis_laplacian(
            field,
            _spacing_vector(dy_values, field.shape[1], dtype=field.dtype),
            axis=1,
            mode=mode_y,
        )
    )
    z_term = (
        (z_bottom - 2.0 * field + z_top) / max(float(dz_values) ** 2, 1.0e-12)
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


def _enforce_velocity_bc_3d(
    field: jnp.ndarray, active_mask: jnp.ndarray | None = None
) -> jnp.ndarray:
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


def _enforce_fluid_mask_3d(field: jnp.ndarray, active_mask: jnp.ndarray) -> jnp.ndarray:
    """Mask solids while retaining cell-centred tangential wall values."""

    bounded = jnp.where(active_mask, field, 0.0)
    if bounded.shape[0] > 1:
        bounded = bounded.at[0].set(bounded[1])
        bounded = bounded.at[-1].set(bounded[-2])
    return bounded


def _enforce_stationwise_flow_rate_3d(
    u: jnp.ndarray,
    *,
    active_mask: jnp.ndarray,
    cell_area: jnp.ndarray,
    target_flow_rate: float | None = None,
    relaxation: float = 1.0,
) -> jnp.ndarray:
    active_area = jnp.maximum(
        jnp.sum(jnp.where(active_mask, cell_area, 0.0), axis=(1, 2)), 1.0e-20
    )
    station_flow_rate = jnp.sum(jnp.where(active_mask, u * cell_area, 0.0), axis=(1, 2))
    if target_flow_rate is None:
        target = jnp.mean(station_flow_rate)
    else:
        target = jnp.asarray(target_flow_rate, dtype=u.dtype)
    correction = (relaxation * (station_flow_rate - target) / active_area)[
        :, None, None
    ]
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
    pressure_loss_gradient = (
        jnp.asarray(base_pressure_loss_gradient, dtype=u.dtype) + multiplier
    )
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
        raise ValueError(
            "magnetic_axis and side_axis must be distinct members of {1, 2}"
        )

    if len(active_mask.addressable_shards) == 1 and not bool(jnp.any(active_mask)):
        raise ValueError("Pressure difference requires active fluid boundary cells")
    return _cross_duct_pressure_difference_kernel(
        p,
        active_mask,
        magnetic_axis=magnetic_axis,
        side_axis=side_axis,
    )


@partial(jax.jit, static_argnames=("magnetic_axis", "side_axis"))
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
            mask = jnp.take_along_axis(active_mask, high[:, None, None], axis=2)[
                :, :, 0
            ]
        else:
            values = jnp.take_along_axis(p, high[:, None, None], axis=1)[:, 0, :]
            mask = jnp.take_along_axis(active_mask, high[:, None, None], axis=1)[
                :, 0, :
            ]
        orthogonal_size = values.shape[1]
        orthogonal_index = jnp.arange(orthogonal_size, dtype=p.dtype)[None, :]
        center_index = 0.5 * (orthogonal_size - 1)
        distance = jnp.where(mask, jnp.abs(orthogonal_index - center_index), jnp.inf)
        closest = jnp.min(distance, axis=1, keepdims=True)
        tap_mask = mask & (distance <= closest + 1.0e-12)
        return jnp.sum(jnp.where(tap_mask, values, 0.0), axis=1) / jnp.maximum(
            jnp.sum(tap_mask, axis=1), 1
        )

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


def _poisson_jacobi_3d(
    rhs: jnp.ndarray,
    *,
    dx: float,
    dy: float | jnp.ndarray,
    dz: float | jnp.ndarray,
    iterations: int,
    tolerance: float,
) -> tuple[jnp.ndarray, float, int, float]:
    if jnp.asarray(dy).ndim or jnp.asarray(dz).ndim:
        return _variable_coefficient_poisson_jacobi_3d(
            rhs,
            jnp.ones_like(rhs),
            dx=dx,
            dy=dy,
            dz=dz,
            iterations=iterations,
            tolerance=tolerance,
        )
    rhs_compatible = rhs - jnp.mean(rhs)
    diagonal = (
        2.0 / max(dx**2, 1.0e-12)
        + 2.0 / max(dy**2, 1.0e-12)
        + 2.0 / max(dz**2, 1.0e-12)
    )
    field = jnp.zeros_like(rhs_compatible)
    initial_residual = float(
        jnp.max(
            jnp.abs(
                _laplacian_3d(
                    field,
                    dx=dx,
                    dy=dy,
                    dz=dz,
                    mode_x="neumann",
                    mode_y="neumann",
                    mode_z="neumann",
                )
                - rhs_compatible
            )
        )
    )
    residual = initial_residual
    iteration_count = 0
    for iteration in range(iterations):
        x_west, x_east, y_south, y_north, z_bottom, z_top = _neighbor_fields(
            field,
            mode_x="neumann",
            mode_y="neumann",
            mode_z="neumann",
        )
        updated = (
            (x_west + x_east) / max(dx**2, 1.0e-12)
            + (y_south + y_north) / max(dy**2, 1.0e-12)
            + (z_bottom + z_top) / max(dz**2, 1.0e-12)
            - rhs_compatible
        ) / diagonal
        field = jnp.nan_to_num(updated - jnp.mean(updated))
        residual = float(
            jnp.max(
                jnp.abs(
                    _laplacian_3d(
                        field,
                        dx=dx,
                        dy=dy,
                        dz=dz,
                        mode_x="neumann",
                        mode_y="neumann",
                        mode_z="neumann",
                    )
                    - rhs_compatible
                )
            )
        )
        iteration_count = iteration + 1
        if residual <= tolerance:
            break
    return field, residual, iteration_count, initial_residual


def _variable_coefficient_poisson_jacobi_3d(
    rhs: jnp.ndarray,
    conductivity: jnp.ndarray,
    *,
    dx: float,
    dy: float | jnp.ndarray,
    dz: float | jnp.ndarray,
    iterations: int,
    tolerance: float,
    initial_field: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, float, int, float]:
    dy_widths = _spacing_vector(dy, rhs.shape[1], dtype=rhs.dtype)
    dz_widths = _spacing_vector(dz, rhs.shape[2], dtype=rhs.dtype)
    volume_weights = jnp.broadcast_to(
        dy_widths[None, :, None] * dz_widths[None, None, :], rhs.shape
    )
    rhs_compatible = rhs - jnp.sum(rhs * volume_weights) / jnp.sum(volume_weights)
    coef_x_w, coef_x_e, coef_y_s, coef_y_n, coef_z_b, coef_z_t = (
        _variable_diffusion_coefficients_3d(
            conductivity, dx=dx, dy=dy_widths, dz=dz_widths
        )
    )
    diagonal = coef_x_w + coef_x_e + coef_y_s + coef_y_n + coef_z_b + coef_z_t
    diagonal = jnp.maximum(diagonal, 1.0e-12)
    if initial_field is None:
        field = jnp.zeros_like(rhs_compatible)
    else:
        field = jnp.nan_to_num(jnp.asarray(initial_field, dtype=rhs_compatible.dtype))
        field = field - jnp.sum(field * volume_weights) / jnp.sum(volume_weights)
    initial_residual = float(
        jnp.max(
            jnp.abs(
                _variable_coefficient_residual_3d(
                    field, rhs_compatible, conductivity, dx=dx, dy=dy, dz=dz
                )
            )
        )
    )
    residual = initial_residual
    iteration_count = 0
    for iteration in range(iterations):
        x_west, x_east, y_south, y_north, z_bottom, z_top = _neighbor_fields(
            field,
            mode_x="neumann",
            mode_y="neumann",
            mode_z="neumann",
        )
        updated = (
            -rhs_compatible
            + coef_x_w * x_west
            + coef_x_e * x_east
            + coef_y_s * y_south
            + coef_y_n * y_north
            + coef_z_b * z_bottom
            + coef_z_t * z_top
        ) / diagonal
        field = jnp.nan_to_num(
            updated - jnp.sum(updated * volume_weights) / jnp.sum(volume_weights)
        )
        residual = float(
            jnp.max(
                jnp.abs(
                    _variable_coefficient_residual_3d(
                        field,
                        rhs_compatible,
                        conductivity,
                        dx=dx,
                        dy=dy,
                        dz=dz,
                    )
                )
            )
        )
        iteration_count = iteration + 1
        if residual <= tolerance:
            break
    return field, residual, iteration_count, initial_residual


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
    coef_x_w, coef_x_e, coef_y_s, coef_y_n, coef_z_b, coef_z_t = (
        _variable_diffusion_coefficients_3d(conductivity, dx=dx, dy=dy, dz=dz)
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
) -> tuple[jnp.ndarray, ...]:
    """Cell-volume-normalized face coefficients for ``div(sigma grad)``."""

    nx, ny, nz = conductivity.shape
    spacing = _coerce_spacing_vector if validated_spacing else _spacing_vector
    dy_widths = spacing(dy, ny, dtype=conductivity.dtype)
    dz_widths = spacing(dz, nz, dtype=conductivity.dtype)
    sigma_x = _harmonic_mean(conductivity[1:], conductivity[:-1])
    coef_x_w = jnp.concatenate(
        [jnp.zeros_like(conductivity[:1]), sigma_x / max(dx**2, 1.0e-12)], axis=0
    )
    coef_x_e = jnp.concatenate(
        [sigma_x / max(dx**2, 1.0e-12), jnp.zeros_like(conductivity[-1:])], axis=0
    )

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


def _variable_coefficient_poisson_sparse_3d(
    rhs: jnp.ndarray,
    conductivity: jnp.ndarray,
    *,
    dx: float,
    dy: float | jnp.ndarray,
    dz: float | jnp.ndarray,
    iterations: int,
    tolerance: float,
    initial_field: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, float, int, float]:
    if sparse is None or sparse_spsolve is None:
        return _variable_coefficient_poisson_jacobi_3d(
            rhs,
            conductivity,
            dx=dx,
            dy=dy,
            dz=dz,
            iterations=iterations,
            tolerance=tolerance,
            initial_field=initial_field,
        )

    conductivity_np = np.asarray(conductivity, dtype=float)
    rhs_np = np.asarray(rhs, dtype=float)
    nx, ny, nz = conductivity_np.shape
    dy_widths_np = np.asarray(_spacing_vector(dy, ny, dtype=rhs.dtype), dtype=float)
    dz_widths_np = np.asarray(_spacing_vector(dz, nz, dtype=rhs.dtype), dtype=float)
    volume_weights = np.broadcast_to(
        dy_widths_np[None, :, None] * dz_widths_np[None, None, :], rhs_np.shape
    )
    rhs_compatible = rhs_np - np.sum(rhs_np * volume_weights) / np.sum(volume_weights)
    coefficients = tuple(
        np.asarray(value, dtype=float)
        for value in _variable_diffusion_coefficients_3d(
            conductivity, dx=dx, dy=dy_widths_np, dz=dz_widths_np
        )
    )
    coef_x_w, coef_x_e, coef_y_s, coef_y_n, coef_z_b, coef_z_t = coefficients
    size = nx * ny * nz

    def flat_index(i: int, j: int, k: int) -> int:
        return (i * ny + j) * nz + k

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    rhs_vector = (-rhs_compatible).reshape(-1)

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                idx = flat_index(i, j, k)
                if idx == 0:
                    rows.append(idx)
                    cols.append(idx)
                    data.append(1.0)
                    rhs_vector[idx] = 0.0
                    continue

                diag = 0.0
                if i > 0:
                    coef = float(coef_x_w[i, j, k])
                    diag += coef
                    rows.append(idx)
                    cols.append(flat_index(i - 1, j, k))
                    data.append(-coef)
                if i + 1 < nx:
                    coef = float(coef_x_e[i, j, k])
                    diag += coef
                    rows.append(idx)
                    cols.append(flat_index(i + 1, j, k))
                    data.append(-coef)
                if j > 0:
                    coef = float(coef_y_s[i, j, k])
                    diag += coef
                    rows.append(idx)
                    cols.append(flat_index(i, j - 1, k))
                    data.append(-coef)
                if j + 1 < ny:
                    coef = float(coef_y_n[i, j, k])
                    diag += coef
                    rows.append(idx)
                    cols.append(flat_index(i, j + 1, k))
                    data.append(-coef)
                if k > 0:
                    coef = float(coef_z_b[i, j, k])
                    diag += coef
                    rows.append(idx)
                    cols.append(flat_index(i, j, k - 1))
                    data.append(-coef)
                if k + 1 < nz:
                    coef = float(coef_z_t[i, j, k])
                    diag += coef
                    rows.append(idx)
                    cols.append(flat_index(i, j, k + 1))
                    data.append(-coef)
                rows.append(idx)
                cols.append(idx)
                data.append(max(diag, 1.0e-12))

    matrix = sparse.csr_matrix((data, (rows, cols)), shape=(size, size))
    x0 = np.zeros(size, dtype=float)
    if initial_field is not None:
        initial_np = np.asarray(initial_field, dtype=float).reshape(-1)
        x0 = initial_np - np.sum(
            initial_np.reshape(rhs_compatible.shape) * volume_weights
        ) / np.sum(volume_weights)
        x0 = np.asarray(x0, dtype=float).reshape(-1)
        x0[0] = 0.0
    initial_residual = float(
        np.max(
            np.abs(
                np.asarray(
                    _variable_coefficient_residual_3d(
                        jnp.asarray(x0.reshape(rhs_compatible.shape)),
                        jnp.asarray(rhs_compatible),
                        conductivity,
                        dx=dx,
                        dy=dy,
                        dz=dz,
                    )
                )
            )
        )
    )
    try:
        solution = sparse_spsolve(matrix, rhs_vector)
    except Exception:
        return _variable_coefficient_poisson_jacobi_3d(
            rhs,
            conductivity,
            dx=dx,
            dy=dy,
            dz=dz,
            iterations=iterations,
            tolerance=tolerance,
            initial_field=initial_field,
        )
    field = solution.reshape(rhs_compatible.shape)
    weights_sum = np.sum(volume_weights)
    field = field - np.sum(field * volume_weights) / weights_sum
    residual = float(
        np.max(
            np.abs(
                np.asarray(
                    _variable_coefficient_residual_3d(
                        jnp.asarray(field),
                        jnp.asarray(rhs_compatible),
                        conductivity,
                        dx=dx,
                        dy=dy,
                        dz=dz,
                    )
                )
            )
        )
    )
    return jnp.asarray(field, dtype=float), residual, 1, initial_residual


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


def _line_solvers_3d(
    diagonal: jnp.ndarray,
    directions: tuple[tuple[int, jnp.ndarray, jnp.ndarray], ...],
):
    solvers = []
    for axis, lower, upper in directions:
        permutation = (axis,) + tuple(index for index in range(3) if index != axis)
        inverse = tuple(np.argsort(permutation))

        def solve_line(
            residual: jnp.ndarray,
            *,
            lower=lower,
            upper=upper,
            permutation=permutation,
            inverse=inverse,
        ) -> jnp.ndarray:
            solved = tridiagonal_solve(
                jnp.transpose(lower, permutation),
                jnp.transpose(diagonal, permutation),
                jnp.transpose(upper, permutation),
                jnp.transpose(residual, permutation),
            )
            return jnp.transpose(solved, inverse)

        solvers.append(solve_line)
    return tuple(solvers)


def _additive_line_preconditioner_3d(
    diagonal: jnp.ndarray,
    directions: tuple[tuple[int, jnp.ndarray, jnp.ndarray], ...],
    *,
    periodic_last_axis: tuple[jnp.ndarray, jnp.ndarray] | None = None,
):
    line_solves = list(_line_solvers_3d(diagonal, directions))
    if periodic_last_axis is not None:
        lower, upper = periodic_last_axis

        def solve_periodic(residual: jnp.ndarray) -> jnp.ndarray:
            solved = cyclic_tridiagonal_solve(
                jnp.moveaxis(lower, -1, 0),
                jnp.moveaxis(diagonal, -1, 0),
                jnp.moveaxis(upper, -1, 0),
                jnp.moveaxis(residual, -1, 0),
            )
            return jnp.moveaxis(solved, 0, -1)

        line_solves.append(solve_periodic)

    return additive_preconditioner(line_solves)


def _axial_mean_preconditioner_3d(
    volume: jnp.ndarray,
    coef_x_w: jnp.ndarray,
    coef_x_e: jnp.ndarray,
    *,
    gauge: bool = True,
):
    """Invert the Galerkin operator for cross-section-constant axial modes."""

    nx, ny, nz = volume.shape
    normalization = math.sqrt(ny * nz)
    west = -jnp.sum(volume * coef_x_w, axis=(1, 2)) / (ny * nz)
    east = -jnp.sum(volume * coef_x_e, axis=(1, 2)) / (ny * nz)
    diagonal = -(west + east)
    coarse = jnp.diag(diagonal)
    coarse = coarse + jnp.diag(west[1:], -1) + jnp.diag(east[:-1], 1)
    if gauge:
        gauge_vector = jnp.sum(volume, axis=(1, 2)) / normalization
        coarse = coarse + jnp.outer(gauge_vector, gauge_vector) / jnp.sum(volume)
    coarse_inverse = jnp.linalg.inv(coarse)

    def apply(residual: jnp.ndarray) -> jnp.ndarray:
        reduced = jnp.sum(residual, axis=(1, 2)) / normalization
        correction = coarse_inverse @ reduced
        return jnp.broadcast_to(
            correction[:, None, None] / normalization, residual.shape
        )

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
        coarse_widths = (
            jnp.zeros(count, dtype=widths.dtype).at[groups].add(local_widths)
        )
        coarse_centers = (
            jnp.zeros(count, dtype=widths.dtype)
            .at[groups]
            .add(local_widths * local_centers)
            / coarse_widths
        )
        high = jnp.clip(jnp.searchsorted(coarse_centers, local_centers), 1, count - 1)
        high = jnp.where(count == 1, 0, high)
        low = jnp.where(count == 1, 0, high - 1)
        span = jnp.where(count == 1, 1.0, coarse_centers[high] - coarse_centers[low])
        left_parts.append(low + offset)
        right_parts.append(high + offset)
        weight_parts.append(
            jnp.clip((local_centers - coarse_centers[low]) / span, 0.0, 1.0)
        )
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
    """Return a shard-local fast-diagonalization Galerkin correction.

    A DCT diagonalizes each local Neumann axial block. One generalized
    transverse eigendecomposition then inverts every axial mode, avoiding a
    dense factorization per mode and all communication inside the correction.
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
    stiffness = jax.vmap(
        lambda column: galerkin(transverse_matvec, column).reshape(-1)
    )(basis).T
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

    partitions = 1 if sharding is None else sharding.mesh.size
    local_nx = nx // partitions
    axial_eigenvalues = 2.0 - 2.0 * jnp.cos(
        jnp.pi * jnp.arange(local_nx, dtype=volume.dtype) / local_nx
    )
    denominators = axial_eigenvalues[:, None] + jnp.maximum(eigenvalues[None], 0.0)
    denominators = denominators.at[0, 0].add(gauge_eigenvalue)

    def solve_local(rhs: jnp.ndarray) -> jnp.ndarray:
        transformed = dct(rhs, type=2, axis=0, norm="ortho").reshape(local_nx, -1)
        spectral = transformed @ inverse_modes
        solved = (spectral / denominators) @ inverse_modes.T
        return idct(solved.reshape((local_nx, ncy, ncz)), type=2, axis=0, norm="ortho")

    coarse_solve = solve_local
    if sharding is not None:  # pragma: no cover - exercised by hardware gates
        coarse_solve = jax.shard_map(
            solve_local,
            mesh=sharding.mesh,
            in_specs=sharding.spec,
            out_specs=sharding.spec,
            check_vma=False,
        )
    coarse_zero = jnp.zeros(coarse_shape, dtype=volume.dtype)

    def prolong(coarse: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(prolong_cross)(coarse)

    def restrict(fine: jnp.ndarray) -> jnp.ndarray:
        return jax.linear_transpose(prolong, coarse_zero)(fine)[0]

    def apply(residual: jnp.ndarray) -> jnp.ndarray:
        return prolong(coarse_solve(restrict(residual)))

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
    relative_residual_norm = residual_norm / jnp.maximum(
        jnp.linalg.norm(linear_rhs), jnp.asarray(1.0e-30)
    )
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
        raise ValueError(
            "Mixed axial pressure does not support the Neumann coarse correction"
        )

    dy_widths = _coerce_spacing_vector(dy, rhs.shape[1], dtype=rhs.dtype)
    dz_widths = _coerce_spacing_vector(dz, rhs.shape[2], dtype=rhs.dtype)
    volume = jnp.broadcast_to(
        dy_widths[None, :, None] * dz_widths[None, None, :], rhs.shape
    )
    volume_sum = jnp.sum(volume)
    solved_rhs = (
        rhs if mixed_axial_pressure else rhs - jnp.sum(rhs * volume) / volume_sum
    )
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
        coef_x_e = coef_x_e.at[-1].set(2.0 * mobility[-1] / max(dx**2, 1.0e-12))
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
        )
        if mixed_axial_pressure:
            x_east = jnp.concatenate([field[1:], jnp.zeros_like(field[-1:])], axis=0)
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
    line_precondition = _additive_line_preconditioner_3d(diagonal, directions)
    axial_precondition = _axial_mean_preconditioner_3d(
        volume, coef_x_w, coef_x_e, gauge=not mixed_axial_pressure
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
            return (
                line_precondition(residual)
                + axial_precondition(residual)
                + coarse_correction(residual)
            )

    linear_rhs = -volume * solved_rhs
    effective_rtol = tolerance
    effective_atol = tolerance
    if local_tolerance is not None:
        volume_min = (
            float(jnp.min(volume)) if local_volume_min is None else local_volume_min
        )
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
        transpose_rtol=tolerance,
        transpose_atol=tolerance,
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
    include_axial_line: bool = True,
    prescribed_inlet: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Solve one frozen, conservative three-component momentum system.

    ``force`` includes explicit deviatoric stresses and body forces.  The inlet
    is prescribed and the outlet is zero-gradient; ``prescribed_inlet=False``
    retains the boundary-neutral diffusion limit.  Affine terms stay outside GMRES.
    """

    shape = velocity.shape
    if shape != (*density.shape, 3) or force.shape != shape:
        raise ValueError("Momentum fields must share one (nx, ny, nz, 3) shape")
    dy_widths = _coerce_spacing_vector(dy, shape[1], dtype=velocity.dtype)
    dz_widths = _coerce_spacing_vector(dz, shape[2], dtype=velocity.dtype)
    dx_widths = jnp.full((shape[0],), dx, dtype=velocity.dtype)
    volume = dx_widths[:, None, None] * dy_widths[None, :, None] * dz_widths[None, None, :]
    dynamic_viscosity = density * viscosity
    coefficients = _variable_diffusion_coefficients_3d(
        dynamic_viscosity, dx=dx, dy=dy_widths, dz=dz_widths, validated_spacing=True
    )
    coef_x_w, coef_x_e, coef_y_s, coef_y_n, coef_z_b, coef_z_t = coefficients
    wall_sink = jnp.zeros_like(density)
    wall_sink = wall_sink.at[:, 0, :].add(dynamic_viscosity[:, 0, :] / (0.5 * dy_widths[0] ** 2))
    wall_sink = wall_sink.at[:, -1, :].add(dynamic_viscosity[:, -1, :] / (0.5 * dy_widths[-1] ** 2))
    wall_sink = wall_sink.at[:, :, 0].add(dynamic_viscosity[:, :, 0] / (0.5 * dz_widths[0] ** 2))
    wall_sink = wall_sink.at[:, :, -1].add(dynamic_viscosity[:, :, -1] / (0.5 * dz_widths[-1] ** 2))
    inlet_sink = jnp.zeros_like(density)
    if prescribed_inlet:
        inlet_sink = inlet_sink.at[0].set(2.0 * dynamic_viscosity[0] / dx**2)
    diffusion_sink = wall_sink + inlet_sink
    widths = (dx_widths, dy_widths, dz_widths)
    weights = _limited_linear_vector_face_weights_duct(
        velocity, rho_phi, boundary_velocity, widths
    )
    zero_patches = tuple(jnp.zeros_like(value) for value in boundary_velocity)
    prescribed_patches = (boundary_velocity[0], zero_patches[1], *boundary_velocity[2:])
    boundary_action = _limited_linear_convection_matrix_action_duct(
        jnp.zeros_like(velocity), rho_phi, weights, prescribed_patches, widths
    )

    def matvec(field: jnp.ndarray) -> jnp.ndarray:
        neighbours = _neighbor_fields(
            field,
            mode_x="neumann",
            mode_y="neumann",
            mode_z="neumann",
        )
        diffusion = sum(c[..., None] * (n - field) for c, n in zip(
            coefficients, neighbours, strict=True)) - diffusion_sink[..., None] * field
        homogeneous_patches = (zero_patches[0], field[-1], *zero_patches[2:])
        convection = _limited_linear_convection_matrix_action_duct(
            field, rho_phi, weights, homogeneous_patches, widths
        )
        return volume[..., None] * (density[..., None] * field + dt * (convection - diffusion))

    diagonal = volume * (density + dt * (sum(coefficients) + diffusion_sink))
    directions = (
        (0, -volume * dt * coef_x_w, -volume * dt * coef_x_e),
        (1, -volume * dt * coef_y_s, -volume * dt * coef_y_n),
        (2, -volume * dt * coef_z_b, -volume * dt * coef_z_t),
    )
    if not include_axial_line:
        directions = directions[1:]
    scalar_precondition = _additive_line_preconditioner_3d(diagonal, directions)

    def precondition(flat: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(scalar_precondition, in_axes=-1, out_axes=-1)(
            flat.reshape(shape)).reshape(-1)

    inlet_velocity = jnp.zeros_like(velocity).at[0].set(boundary_velocity[0])
    source = force - boundary_action + inlet_sink[..., None] * inlet_velocity
    linear_rhs = volume[..., None] * (density[..., None] * velocity + dt * source)
    flat_rhs = linear_rhs.reshape(-1)

    def flat_matvec(flat: jnp.ndarray) -> jnp.ndarray:
        return matvec(flat.reshape(shape)).reshape(-1)

    restart = min(12, flat_rhs.size)

    def krylov(operator, rhs):
        return gmres(
            operator, rhs, x0=jnp.zeros_like(rhs), precond=precondition,
            restart=restart, rtol=tolerance, atol=tolerance,
            max_restarts=max(1, math.ceil(iterations / restart))
        ).x

    solved = linear_solve(flat_matvec, flat_rhs, krylov).reshape(shape)
    residual = jnp.linalg.norm(flat_rhs - flat_matvec(solved.reshape(-1)))
    target = jnp.maximum(tolerance, tolerance * jnp.linalg.norm(flat_rhs))
    return solved, residual, residual <= target


def _cell_limited_least_squares_gradient_duct(
    field: jnp.ndarray,
    boundary_values: tuple[jnp.ndarray, ...],
    widths: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
) -> tuple[jnp.ndarray, ...]:
    """Return ``cellLimited leastSquares 1`` from patch-face values x-, x+, y-, y+, z-, z+."""
    gradients, neighbours = [], []
    for axis, width in enumerate(widths):
        values = jnp.moveaxis(field, axis, 0)
        lo = jnp.concatenate((boundary_values[2 * axis][None], values[:-1]))
        hi = jnp.concatenate((values[1:], boundary_values[2 * axis + 1][None]))
        centers = 0.5 * (width[:-1] + width[1:])
        dm = jnp.concatenate((0.5 * width[:1], centers))[:, None, None]
        dp = jnp.concatenate((centers, 0.5 * width[-1:]))[:, None, None]
        fraction = width[1:] / (width[:-1] + width[1:])
        lo_weight = jnp.concatenate((jnp.ones(1), fraction))[:, None, None]
        hi_weight = jnp.concatenate((1.0 - fraction, jnp.ones(1)))[:, None, None]
        gradient = (lo_weight * (values - lo) / dm + hi_weight * (hi - values) / dp) / (
            lo_weight + hi_weight
        )
        gradients.append(jnp.moveaxis(gradient, 0, axis))
        neighbours.extend((jnp.moveaxis(lo, 0, axis), jnp.moveaxis(hi, 0, axis)))
    local = jnp.stack((field, *neighbours))
    minimum, maximum = jnp.min(local, axis=0), jnp.max(local, axis=0)
    limiter = jnp.ones_like(field)
    for axis, (gradient, width) in enumerate(zip(gradients, widths, strict=True)):
        shape = [1, 1, 1]
        shape[axis] = field.shape[axis]
        half_step = 0.5 * width.reshape(shape) * gradient
        for extrapolate in (-half_step, half_step):
            delta = jnp.where(extrapolate > 0.0, maximum - field, minimum - field)
            ratio = delta / jnp.where(jnp.abs(extrapolate) > 1.0e-15, extrapolate, 1.0)
            bounded = jnp.where(
                jnp.abs(extrapolate) > 1.0e-15, jnp.minimum(ratio, 1.0), 1.0
            )
            limiter = jnp.minimum(limiter, bounded)
    return tuple(limiter * gradient for gradient in gradients)


def _limited_linear_vector_face_weights_duct(
    velocity: jnp.ndarray,
    rho_phi: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    boundary_velocity: tuple[jnp.ndarray, ...],
    widths: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
) -> tuple[jnp.ndarray, ...]:
    """Freeze v2206 vector weights from patch-face values x-, x+, y-, y+, z-, z+."""
    q = jnp.sum(velocity**2, axis=-1)
    gradients = _cell_limited_least_squares_gradient_duct(
        q, tuple(jnp.sum(value**2, axis=-1) for value in boundary_velocity), widths
    )
    weights = []
    for axis, (width, gradient, face_flux) in enumerate(
        zip(widths, gradients, rho_phi, strict=True)
    ):
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
    field, rho_phi, weights, boundary_values, widths
):
    """Return cell-volume divergence from patch-face values x-, x+, y-, y+, z-, z+."""
    dx, dy, dz = widths
    volume = dx[:, None, None] * dy[None, :, None] * dz[None, None, :]
    action = jnp.zeros_like(field)
    for axis, (face_flux, weight) in enumerate(zip(rho_phi, weights, strict=True)):
        values = jnp.moveaxis(field, axis, 0)
        weight = jnp.moveaxis(weight, axis, 0)[..., None]
        interpolated = weight * values[:-1] + (1.0 - weight) * values[1:]
        faces = jnp.concatenate(
            (
                boundary_values[2 * axis][None],
                interpolated,
                boundary_values[2 * axis + 1][None],
            )
        )
        divergence = jnp.diff(
            jnp.moveaxis(face_flux, axis, 0)[..., None] * faces, axis=0
        )
        action += jnp.moveaxis(divergence, 0, axis) / volume[..., None]
    return action


def _pack_duct_mass_flux(rho_phi):
    """Pack coordinate-oriented duct fluxes as positive faces plus the inlet."""
    fx, fy, fz = rho_phi
    return jnp.stack((fx[1:], fy[:, 1:], fz[:, :, 1:])), fx[0]


def _unpack_duct_mass_flux(rho_phi_plus, rho_phi_inlet):
    """Restore full face fluxes; lower transverse walls are implicit zeros."""
    return (
        jnp.concatenate((rho_phi_inlet[None], rho_phi_plus[0]), axis=0),
        jnp.concatenate((jnp.zeros_like(rho_phi_plus[1, :, :1]), rho_phi_plus[1]), axis=1),
        jnp.concatenate((jnp.zeros_like(rho_phi_plus[2, :, :, :1]), rho_phi_plus[2]), axis=2),
    )


def _initialize_duct_mass_flux(velocity, density, inlet_velocity, *, dx, dy, dz):
    """Pack ``linearInterpolate(rho*U)&Sf`` for a no-slip rectangular duct.

    The prescribed inlet uses adjacent-cell density, while the zero-gradient
    outlet uses the adjacent cell. Fluxes are area-integrated and oriented
    along increasing coordinates rather than outward patch normals.
    """
    momentum = density[..., None] * velocity
    area_x = dy[:, None] * dz[None, :]
    inlet = density[0] * inlet_velocity[..., 0] * area_x
    plus_x = jnp.concatenate((
        0.5 * (momentum[:-1, ..., 0] + momentum[1:, ..., 0]),
        momentum[-1:, ..., 0])) * area_x
    wy = (dy[1:] / (dy[:-1] + dy[1:]))[None, :, None]
    plus_y = jnp.concatenate((
        wy * momentum[:, :-1, :, 1] + (1.0 - wy) * momentum[:, 1:, :, 1],
        jnp.zeros_like(momentum[:, :1, :, 1])), axis=1) * (dx * dz[None, None, :])
    wz = (dz[1:] / (dz[:-1] + dz[1:]))[None, None, :]
    plus_z = jnp.concatenate((
        wz * momentum[:, :, :-1, 2] + (1.0 - wz) * momentum[:, :, 1:, 2],
        jnp.zeros_like(momentum[:, :, :1, 2])), axis=2) * (dx * dy[None, :, None])
    return jnp.stack((plus_x, plus_y, plus_z)), inlet


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
    initial_pressure: jnp.ndarray | None = None,
    single_reduction: bool = False,
    include_axial_line: bool = True,
    inlet_flow_rate: float | None = None,
) -> tuple[jnp.ndarray, ...]:
    """Project duct face fluxes; mixed boundaries also return flow diagnostics."""

    mixed_axial_pressure = inlet_flow_rate is not None
    if inlet_flow_rate is not None and inlet_flow_rate <= 0.0:
        raise ValueError("Inlet flow rate must be positive")

    y0, y1, z0, z1 = (
        _rectangular_fluid_bounds(fluid_mask) if fluid_bounds is None else fluid_bounds
    )
    us = u[:, y0:y1, z0:z1]
    vs = v[:, y0:y1, z0:z1]
    ws = w[:, y0:y1, z0:z1]
    rhos = rho[:, y0:y1, z0:z1]
    dys = dy[y0:y1]
    dzs = dz[z0:z1]
    nx, ny, nz = us.shape

    uf = jnp.zeros((nx + 1, ny, nz), dtype=u.dtype)
    uf = uf.at[1:-1].set(0.5 * (us[1:] + us[:-1]))
    uf = uf.at[0].set(us[0])
    uf = uf.at[-1].set(us[-1])
    vf = jnp.zeros((nx, ny + 1, nz), dtype=v.dtype)
    vf = vf.at[:, 1:-1, :].set(0.5 * (vs[:, 1:, :] + vs[:, :-1, :]))
    wf = jnp.zeros((nx, ny, nz + 1), dtype=w.dtype)
    wf = wf.at[:, :, 1:-1].set(0.5 * (ws[:, :, 1:] + ws[:, :, :-1]))

    face_area = dys[:, None] * dzs[None, :]
    if inlet_flow_rate is not None:
        target = jnp.asarray(inlet_flow_rate, dtype=u.dtype)
        inlet_profile = jnp.maximum(us[0], 0.0)
        estimated = jnp.sum(inlet_profile * face_area)
        scaled = inlet_profile * target / jnp.maximum(estimated, 1.0e-20)
        shifted = inlet_profile + (target - estimated) / jnp.sum(face_area)
        inlet_profile = jnp.where(estimated > 0.5 * target, scaled, shifted)
        uf = uf.at[0].set(inlet_profile)

    divergence = (
        (uf[1:] - uf[:-1]) / max(dx, 1.0e-12)
        + (vf[:, 1:, :] - vf[:, :-1, :]) / dys[None, :, None]
        + (wf[:, :, 1:] - wf[:, :, :-1]) / dzs[None, None, :]
    )
    mobility = dt / jnp.maximum(rhos, 1.0e-20)
    pressure, _, _, _, _, _, _ = _solvax_pressure_poisson_duct(
        divergence,
        mobility,
        dx=dx,
        dy=dys,
        dz=dzs,
        iterations=iterations,
        tolerance=tolerance,
        initial_field=(
            None if initial_pressure is None else initial_pressure[:, y0:y1, z0:z1]
        ),
        single_reduction=single_reduction,
        include_axial_line=include_axial_line,
        axial_pressure_mode=(
            _MIXED_AXIAL_PRESSURE_MODE if mixed_axial_pressure else "neumann"
        ),
    )

    mobility_x = _harmonic_mean(mobility[1:], mobility[:-1])
    uf = uf.at[1:-1].add(
        -mobility_x * (pressure[1:] - pressure[:-1]) / max(dx, 1.0e-12)
    )
    if mixed_axial_pressure:
        uf = uf.at[-1].add(
            -mobility[-1] * (0.0 - pressure[-1]) / max(0.5 * dx, 1.0e-12)
        )
    mobility_y = _harmonic_mean(mobility[:, 1:, :], mobility[:, :-1, :])
    y_distance = 0.5 * (dys[:-1] + dys[1:])
    vf = vf.at[:, 1:-1, :].add(
        -mobility_y
        * (pressure[:, 1:, :] - pressure[:, :-1, :])
        / y_distance[None, :, None]
    )
    mobility_z = _harmonic_mean(mobility[:, :, 1:], mobility[:, :, :-1])
    z_distance = 0.5 * (dzs[:-1] + dzs[1:])
    wf = wf.at[:, :, 1:-1].add(
        -mobility_z
        * (pressure[:, :, 1:] - pressure[:, :, :-1])
        / z_distance[None, None, :]
    )
    divergence_after = (
        (uf[1:] - uf[:-1]) / max(dx, 1.0e-12)
        + (vf[:, 1:, :] - vf[:, :-1, :]) / dys[None, :, None]
        + (wf[:, :, 1:] - wf[:, :, :-1]) / dzs[None, None, :]
    )

    projected_u = 0.5 * (uf[:-1] + uf[1:])
    projected_v = 0.5 * (vf[:, :-1, :] + vf[:, 1:, :])
    projected_w = 0.5 * (wf[:, :, :-1] + wf[:, :, 1:])
    full_u = jnp.zeros_like(u).at[:, y0:y1, z0:z1].set(projected_u)
    full_v = jnp.zeros_like(v).at[:, y0:y1, z0:z1].set(projected_v)
    full_w = jnp.zeros_like(w).at[:, y0:y1, z0:z1].set(projected_w)
    full_p = jnp.zeros_like(u).at[:, y0:y1, z0:z1].set(pressure)
    divergence_norm = jnp.max(jnp.abs(divergence_after))
    if not mixed_axial_pressure:
        return full_u, full_v, full_w, full_p, divergence_norm

    inlet_flow = jnp.sum(uf[0] * face_area)
    outlet_flow = jnp.sum(uf[-1] * face_area)
    active_mask = fluid_mask[:, y0:y1, z0:z1]
    area = face_area[None, :, :]
    active_area = jnp.sum(jnp.where(active_mask, area, 0.0), axis=(1, 2))
    mean_pressure = (
        jnp.sum(jnp.where(active_mask, pressure * area, 0.0), axis=(1, 2)) / active_area
    )
    pressure_loss_faces = jnp.zeros((pressure.shape[0] + 1,), dtype=pressure.dtype)
    pressure_loss_faces = pressure_loss_faces.at[1:-1].set(
        -(mean_pressure[1:] - mean_pressure[:-1]) / max(dx, 1.0e-12)
    )
    pressure_loss_faces = pressure_loss_faces.at[-1].set(
        mean_pressure[-1] / max(0.5 * dx, 1.0e-12)
    )
    pressure_loss = 0.5 * (pressure_loss_faces[:-1] + pressure_loss_faces[1:])
    flow_error = jnp.maximum(
        jnp.abs(inlet_flow - inlet_flow_rate), jnp.abs(outlet_flow - inlet_flow_rate)
    )
    return full_u, full_v, full_w, full_p, pressure_loss, divergence_norm, flow_error


def _safe_correlation(x: jnp.ndarray, y: jnp.ndarray) -> float:
    centered_x = x - jnp.mean(x)
    centered_y = y - jnp.mean(y)
    denom = jnp.sqrt(jnp.sum(centered_x**2) * jnp.sum(centered_y**2))
    return float(jnp.where(denom > 0.0, jnp.sum(centered_x * centered_y) / denom, 0.0))


def _mirror_residual(values: jnp.ndarray, *, odd: bool) -> float:
    if values.size == 0:
        return 0.0
    mirrored = values[::-1]
    residual = values + mirrored if odd else values - mirrored
    return float(jnp.max(jnp.abs(residual)))


def _center_station_value(values: jnp.ndarray) -> float:
    if values.size == 0:
        return 0.0
    n = int(values.shape[0])
    if n % 2 == 1:
        return float(values[n // 2])
    return float(0.5 * (values[n // 2 - 1] + values[n // 2]))


def _clip_state(field: jnp.ndarray, limit: float) -> jnp.ndarray:
    return jnp.clip(jnp.nan_to_num(field), -limit, limit)


def _cross_section_mesh(case: CaseSpec):
    geometry = case.geometry
    if geometry.kind == "rect_duct":
        mesh = generate_rect_duct_mesh(
            width=geometry.width,
            height=geometry.height,
            length=geometry.length,
            nx=geometry.nx,
            ny=geometry.ny,
            nz=geometry.nz,
        )
    if geometry.kind == "layered_duct":
        mesh = generate_layered_duct_mesh(
            width=geometry.width,
            height=geometry.height,
            length=geometry.length,
            nx=geometry.nx,
            ny=geometry.ny,
            nz=geometry.nz,
            wall_thickness=geometry.wall_thickness,
            wall_cells=geometry.wall_cells,
            target_ha=geometry.target_ha,
        )
    if geometry.kind == "pipe_ogrid":
        wall_thickness = max(geometry.wall_thickness)
        wall_cells = max(geometry.wall_cells)
        mesh = generate_pipe_ogrid_mesh(
            radius=geometry.radius or (0.5 * geometry.width),
            length=geometry.length,
            nx=geometry.nx,
            nr=geometry.nr or geometry.ny,
            ntheta=geometry.ntheta or geometry.nz,
            wall_thickness=wall_thickness,
            wall_cells=wall_cells,
            target_ha=geometry.target_ha,
            hartmann_layer_cells=geometry.hartmann_layer_cells,
        )
    if geometry.kind == "bent_pipe":
        mesh = generate_bent_pipe_mesh(
            tube_radius=geometry.radius or (0.5 * geometry.width),
            bend_radius=geometry.bend_radius or max(geometry.length, geometry.width),
            bend_angle=geometry.bend_angle or 0.5 * jnp.pi,
            nx=geometry.nx,
            nr=geometry.nr or geometry.ny,
            ntheta=geometry.ntheta or geometry.nz,
        )
    if geometry.kind not in {"rect_duct", "layered_duct", "pipe_ogrid", "bent_pipe"}:
        raise ValueError(f"Unsupported extruded geometry {geometry.kind!r}")
    if geometry.axial_origin != 0.0:
        points = mesh.point_coordinates
        if points is not None:
            points = points.at[..., 0].add(geometry.axial_origin)
        mesh = replace(
            mesh,
            x_faces=mesh.x_faces + geometry.axial_origin,
            point_coordinates=points,
        )
    return mesh


def _sample_volume_field(volume_field, x, y, z):
    """Validate and unpack one analytic magnetic field sampled on a volume."""

    sampled = jnp.asarray(volume_field(x, y, z), dtype=float)
    if sampled.shape != (*x.shape, 3):
        raise ValueError("Fringing volume field must append one three-component axis")
    return sampled[..., 0], sampled[..., 1], sampled[..., 2]


def _sample_station_magnetic_field_duct(
    case: CaseSpec,
    mesh,
    *,
    field_scale: jnp.ndarray,
    volume_field: Callable[..., jnp.ndarray] | None = None,
    nx: int,
    ny: int,
    nz: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    if volume_field is not None:
        xx, yy, zz = jnp.meshgrid(
            mesh.x_centers, mesh.y_centers, mesh.z_centers, indexing="ij"
        )
        return _sample_volume_field(volume_field, xx, yy, zz)
    x_coords = np.asarray(
        case.geometry.length * jnp.linspace(0.0, 1.0, nx), dtype=float
    )
    y_coords = np.asarray(mesh.y_centers, dtype=float)
    z_coords = np.asarray(mesh.z_centers, dtype=float)
    if case.magnetic_field.kind == "constant":
        base_field = case.magnetic_field.value or (0.0, 0.0, 0.0)
        bx = _broadcast_station_profile(field_scale * float(base_field[0]), ny, nz)
        by = _broadcast_station_profile(field_scale * float(base_field[1]), ny, nz)
        bz = _broadcast_station_profile(field_scale * float(base_field[2]), ny, nz)
        return bx, by, bz
    if case.magnetic_field.kind == "analytic":
        if case.magnetic_field.fn is None:
            raise ValueError("Analytic magnetic field requires fn")
        yc, zc = jnp.meshgrid(
            jnp.asarray(mesh.y_centers, dtype=float),
            jnp.asarray(mesh.z_centers, dtype=float),
            indexing="ij",
        )
        sampled = jnp.asarray(case.magnetic_field.fn(yc, zc), dtype=float)
        bx0 = _broadcast_cross_section(sampled[..., 0], nx)
        by0 = _broadcast_cross_section(sampled[..., 1], nx)
        bz0 = _broadcast_cross_section(sampled[..., 2], nx)
        station_scale = field_scale[:, None, None]
        return station_scale * bx0, station_scale * by0, station_scale * bz0
    if case.magnetic_field.kind == "tabulated":
        if case.magnetic_field.table_path is None:
            raise ValueError("Tabulated magnetic field requires table_path")
        table = load_tabulated_field(case.magnetic_field.table_path)
        xx, yy, zz = np.meshgrid(x_coords, y_coords, z_coords, indexing="ij")
        sampled = sample_tabulated_field_volume(
            case.magnetic_field.table_path, x=xx, y=yy, z=zz
        )
        if "x" not in table:
            sampled = sampled * np.asarray(
                field_scale[:, None, None, None], dtype=float
            )
        return (
            jnp.asarray(sampled[..., 0], dtype=float),
            jnp.asarray(sampled[..., 1], dtype=float),
            jnp.asarray(sampled[..., 2], dtype=float),
        )
    raise ValueError(f"Unsupported magnetic-field kind {case.magnetic_field.kind!r}")


def _sample_station_magnetic_field_pipe(
    case: CaseSpec,
    *,
    rr: jnp.ndarray,
    theta_grid: jnp.ndarray,
    field_scale: jnp.ndarray,
    x: jnp.ndarray,
    volume_field: Callable[..., jnp.ndarray] | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    x_coords = np.asarray(
        case.geometry.length * jnp.linspace(0.0, 1.0, rr.shape[0]), dtype=float
    )
    yy = np.asarray(rr[0] * jnp.cos(theta_grid[0]), dtype=float)
    zz = np.asarray(rr[0] * jnp.sin(theta_grid[0]), dtype=float)
    if volume_field is not None:
        xx = jnp.broadcast_to(jnp.asarray(x)[:, None, None], rr.shape)
        yy3 = rr * jnp.cos(theta_grid)
        zz3 = rr * jnp.sin(theta_grid)
        return _sample_volume_field(volume_field, xx, yy3, zz3)
    if case.magnetic_field.kind == "constant":
        base_field = case.magnetic_field.value or (0.0, 0.0, 0.0)
        bx = jnp.broadcast_to(
            field_scale[:, None, None] * float(base_field[0]), rr.shape
        )
        by = jnp.broadcast_to(
            field_scale[:, None, None] * float(base_field[1]), rr.shape
        )
        bz = jnp.broadcast_to(
            field_scale[:, None, None] * float(base_field[2]), rr.shape
        )
        return bx, by, bz
    if case.magnetic_field.kind == "analytic":
        if case.magnetic_field.fn is None:
            raise ValueError("Analytic magnetic field requires fn")
        sampled = jnp.asarray(
            case.magnetic_field.fn(jnp.asarray(yy), jnp.asarray(zz)), dtype=float
        )
        bx0 = jnp.broadcast_to(sampled[..., 0][None, :, :], rr.shape)
        by0 = jnp.broadcast_to(sampled[..., 1][None, :, :], rr.shape)
        bz0 = jnp.broadcast_to(sampled[..., 2][None, :, :], rr.shape)
        station_scale = field_scale[:, None, None]
        return station_scale * bx0, station_scale * by0, station_scale * bz0
    if case.magnetic_field.kind == "tabulated":
        if case.magnetic_field.table_path is None:
            raise ValueError("Tabulated magnetic field requires table_path")
        table = load_tabulated_field(case.magnetic_field.table_path)
        xx = np.broadcast_to(x_coords[:, None, None], rr.shape)
        yy3 = np.broadcast_to(yy[None, :, :], rr.shape)
        zz3 = np.broadcast_to(zz[None, :, :], rr.shape)
        sampled = sample_tabulated_field_volume(
            case.magnetic_field.table_path, x=xx, y=yy3, z=zz3
        )
        if "x" not in table:
            sampled = sampled * np.asarray(
                field_scale[:, None, None, None], dtype=float
            )
        return (
            jnp.asarray(sampled[..., 0], dtype=float),
            jnp.asarray(sampled[..., 1], dtype=float),
            jnp.asarray(sampled[..., 2], dtype=float),
        )
    raise ValueError(f"Unsupported magnetic-field kind {case.magnetic_field.kind!r}")


def _bundle_station_history(
    bundle: ExtrudedFieldBundle,
) -> tuple[dict[str, float], ...]:
    names = (
        "x",
        "field_scale",
        "u_max",
        "mean_velocity",
        "volumetric_flow_rate",
        "axial_current",
        "wall_current_leakage",
        "current_scaled_pressure_proxy",
        "axial_pressure_loss_gradient",
        "transverse_pressure_difference",
        "pressure_span",
        "residual",
        "charge_balance_residual",
        "boundary_current_residual",
    )
    zeros = jnp.zeros_like(bundle.x)
    axial_pressure = getattr(bundle, "axial_pressure_loss_gradient", zeros)
    transverse_pressure = getattr(bundle, "transverse_pressure_difference", zeros)
    axial_pressure = axial_pressure if axial_pressure.size else zeros
    transverse_pressure = transverse_pressure if transverse_pressure.size else zeros
    columns = jnp.stack(
        (
            bundle.x,
            bundle.field_scale,
            jnp.max(jnp.abs(bundle.u), axis=(1, 2)),
            bundle.mean_velocity,
            bundle.volumetric_flow_rate,
            bundle.axial_current,
            bundle.wall_current_leakage,
            bundle.current_scaled_pressure_proxy,
            axial_pressure,
            transverse_pressure,
            jnp.max(bundle.p, axis=(1, 2)) - jnp.min(bundle.p, axis=(1, 2)),
            bundle.residual,
            bundle.charge_balance_residual,
            bundle.boundary_current_residual,
        ),
        axis=1,
    )
    return tuple(
        dict(zip(names, map(float, row), strict=True)) for row in np.asarray(columns)
    )


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
    phi_grad_x = (phi[1:] - phi[:-1]) / max(dx, 1.0e-12)
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


def _station_axial_current_from_fluxes(
    fx: jnp.ndarray, cell_area: jnp.ndarray
) -> jnp.ndarray:
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
        (fx[1:] - fx[:-1]) / max(dx, 1.0e-12)
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
    boundary_residual = jnp.abs(
        jnp.sum(div_j * cross_section_area[None, :, :], axis=(1, 2)) * dx
    )
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
        (fx[1:] - fx[:-1]) / max(dx, 1.0e-12)
        + (fy[:, 1:, :] - fy[:, :-1, :]) / dy_widths[None, :, None]
        + (fz[:, :, 1:] - fz[:, :, :-1]) / dz_widths[None, None, :]
    )


def _pipe_theta_neighbors(field: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    theta_prev = jnp.concatenate([field[:, :, -1:], field[:, :, :-1]], axis=2)
    theta_next = jnp.concatenate([field[:, :, 1:], field[:, :, :1]], axis=2)
    return theta_prev, theta_next


def _pipe_gradient_3d(
    field: jnp.ndarray,
    *,
    dx: float,
    dr: float | jnp.ndarray,
    dtheta: float,
    r: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    dr_values = jnp.asarray(dr)
    if dr_values.ndim:
        widths = _spacing_vector(dr_values, field.shape[1], dtype=field.dtype)
        safe_r = jnp.broadcast_to(jnp.asarray(r, dtype=field.dtype), field.shape)
    else:
        widths = None
        safe_r = jnp.maximum(r, 0.5 * float(dr_values))
    x_west = jnp.concatenate([field[:1], field[:-1]], axis=0)
    x_east = jnp.concatenate([field[1:], field[-1:]], axis=0)
    r_inner = jnp.concatenate([field[:, :1, :], field[:, :-1, :]], axis=1)
    r_outer = jnp.concatenate([field[:, 1:, :], field[:, -1:, :]], axis=1)
    theta_prev, theta_next = _pipe_theta_neighbors(field)
    d_dx = (x_east - x_west) / max(2.0 * dx, 1.0e-12)
    d_dr = (
        (r_outer - r_inner) / max(2.0 * float(dr_values), 1.0e-12)
        if widths is None
        else _nonuniform_axis_gradient(field, widths, axis=1)
    )
    d_dtheta = (theta_next - theta_prev) / jnp.maximum(2.0 * dtheta * safe_r, 1.0e-12)
    d_dr = d_dr.at[:, 0, :].set(0.0)
    d_dtheta = d_dtheta.at[:, 0, :].set(0.0)
    return d_dx, d_dr, d_dtheta


def _pipe_laplacian_3d(
    field: jnp.ndarray,
    *,
    dx: float,
    dr: float | jnp.ndarray,
    dtheta: float,
    r: jnp.ndarray,
    outer_dirichlet: bool = True,
) -> jnp.ndarray:
    dr_values = jnp.asarray(dr)
    if dr_values.ndim:
        widths = _spacing_vector(dr_values, field.shape[1], dtype=field.dtype)
        r_values = jnp.asarray(r, dtype=field.dtype)
        r_centers = r_values[0, :, 0] if r_values.ndim == 3 else r_values.reshape(-1)
        r_faces = jnp.concatenate(
            [jnp.zeros((1,), dtype=field.dtype), jnp.cumsum(widths)]
        )
        safe_r = jnp.broadcast_to(r_centers[None, :, None], field.shape)
    else:
        widths = None
        safe_r = jnp.maximum(r, 0.5 * float(dr_values))
    x_west = jnp.concatenate([field[:1], field[:-1]], axis=0)
    x_east = jnp.concatenate([field[1:], field[-1:]], axis=0)
    r_inner = jnp.concatenate([field[:, :1, :], field[:, :-1, :]], axis=1)
    outer_ghost = (
        jnp.zeros_like(field[:, -1:, :]) if outer_dirichlet else field[:, -1:, :]
    )
    r_outer = jnp.concatenate([field[:, 1:, :], outer_ghost], axis=1)
    theta_prev, theta_next = _pipe_theta_neighbors(field)
    dxx = (x_west - 2.0 * field + x_east) / max(dx**2, 1.0e-12)
    drr = (r_inner - 2.0 * field + r_outer) / max(
        float(dr_values) ** 2 if widths is None else 1.0, 1.0e-12
    )
    d_dr = (r_outer - r_inner) / max(
        2.0 * float(dr_values) if widths is None else 1.0, 1.0e-12
    )
    dtheta2 = (theta_prev - 2.0 * field + theta_next) / jnp.maximum(
        (safe_r**2) * dtheta**2, 1.0e-12
    )
    if widths is None:
        lap = dxx + drr + d_dr / safe_r + dtheta2
        return lap.at[:, 0, :].set(
            dxx[:, 0, :]
            + 2.0
            * (field[:, 1, :] - field[:, 0, :])
            / max(float(dr_values) ** 2, 1.0e-12)
        )
    radial_flux = jnp.zeros(
        (field.shape[0], field.shape[1] + 1, field.shape[2]), dtype=field.dtype
    )
    center_distance = 0.5 * (widths[:-1] + widths[1:])
    radial_flux = radial_flux.at[:, 1:-1, :].set(
        r_faces[None, 1:-1, None]
        * (field[:, 1:, :] - field[:, :-1, :])
        / center_distance[None, :, None]
    )
    if outer_dirichlet:
        radial_flux = radial_flux.at[:, -1, :].set(
            r_faces[-1] * (-field[:, -1, :]) / (0.5 * widths[-1])
        )
    radial_term = (radial_flux[:, 1:, :] - radial_flux[:, :-1, :]) / jnp.maximum(
        r_centers[None, :, None] * widths[None, :, None], 1.0e-20
    )
    return dxx + radial_term + dtheta2


def _pipe_divergence_3d(
    jx: jnp.ndarray,
    jr: jnp.ndarray,
    jtheta: jnp.ndarray,
    *,
    dx: float,
    dr: float | jnp.ndarray,
    dtheta: float,
    r: jnp.ndarray,
) -> jnp.ndarray:
    dr_values = jnp.asarray(dr)
    if dr_values.ndim:
        widths = _spacing_vector(dr_values, jx.shape[1], dtype=jx.dtype)
        r_values = jnp.asarray(r, dtype=jx.dtype)
        r_centers = r_values[0, :, 0] if r_values.ndim == 3 else r_values.reshape(-1)
        r_faces = jnp.concatenate([jnp.zeros((1,), dtype=jx.dtype), jnp.cumsum(widths)])
        safe_r = jnp.broadcast_to(r_centers[None, :, None], jx.shape)
    else:
        widths = None
        safe_r = jnp.maximum(r, 0.5 * float(dr_values))
    djx_dx = _pipe_gradient_3d(jx, dx=dx, dr=dr, dtheta=dtheta, r=r)[0]
    if widths is not None:
        radial_face = jnp.zeros(
            (jr.shape[0], jr.shape[1] + 1, jr.shape[2]), dtype=jr.dtype
        )
        radial_face = radial_face.at[:, 1:-1, :].set(
            0.5 * (jr[:, 1:, :] + jr[:, :-1, :])
        )
        radial_term = (
            r_faces[None, 1:, None] * radial_face[:, 1:, :]
            - r_faces[None, :-1, None] * radial_face[:, :-1, :]
        ) / jnp.maximum(r_centers[None, :, None] * widths[None, :, None], 1.0e-20)
        theta_prev, theta_next = _pipe_theta_neighbors(jtheta)
        theta_term = (theta_next - theta_prev) / jnp.maximum(
            2.0 * dtheta * safe_r, 1.0e-12
        )
        return djx_dx + radial_term + theta_term
    rjr = safe_r * jr
    rjr_inner = jnp.concatenate([rjr[:, :1, :], rjr[:, :-1, :]], axis=1)
    rjr_outer = jnp.concatenate([rjr[:, 1:, :], rjr[:, -1:, :]], axis=1)
    radial_term = (rjr_outer - rjr_inner) / jnp.maximum(
        2.0 * float(dr_values) * safe_r, 1.0e-12
    )
    theta_prev, theta_next = _pipe_theta_neighbors(jtheta)
    theta_term = (theta_next - theta_prev) / jnp.maximum(2.0 * dtheta * safe_r, 1.0e-12)
    divergence = djx_dx + radial_term + theta_term
    return divergence.at[:, 0, :].set(
        djx_dx[:, 0, :] + 2.0 * jr[:, 1, :] / max(float(dr_values), 1.0e-12)
    )


def _pipe_radial_fluid_count(fluid_mask: jnp.ndarray) -> int:
    radial_active = np.asarray(jnp.any(fluid_mask, axis=(0, 2)), dtype=bool)
    active = np.flatnonzero(radial_active)
    if not active.size:
        raise ValueError("Pipe face-flux projection requires a nonempty fluid domain")
    count = int(active[-1]) + 1
    if not np.array_equal(active, np.arange(count)):
        raise ValueError("Pipe fluid cells must be contiguous from the axis")
    expected = np.zeros_like(np.asarray(fluid_mask, dtype=bool))
    expected[:, :count, :] = True
    if not np.array_equal(np.asarray(fluid_mask, dtype=bool), expected):
        raise ValueError("Pipe face-flux projection requires a full annular fluid mask")
    return count


def _pipe_variable_diffusion_coefficients_3d(
    coefficient: jnp.ndarray,
    *,
    dx: float,
    r_faces: jnp.ndarray,
    r_centers: jnp.ndarray,
    dtheta: float,
) -> tuple[jnp.ndarray, ...]:
    """Cylindrical FV coefficients for ``div(coefficient grad)``."""

    radial_widths = jnp.diff(r_faces)
    radial_distance = jnp.diff(r_centers)
    sigma_x = _harmonic_mean(coefficient[1:], coefficient[:-1])
    coef_x_w = jnp.concatenate(
        [jnp.zeros_like(coefficient[:1]), sigma_x / max(dx**2, 1.0e-12)],
        axis=0,
    )
    coef_x_e = jnp.concatenate(
        [sigma_x / max(dx**2, 1.0e-12), jnp.zeros_like(coefficient[-1:])],
        axis=0,
    )

    sigma_r = _harmonic_mean(coefficient[:, 1:, :], coefficient[:, :-1, :])
    radial_face_factor = r_faces[1:-1][None, :, None] / jnp.maximum(
        radial_distance[None, :, None], 1.0e-20
    )
    coef_r_inner = (
        jnp.zeros_like(coefficient)
        .at[:, 1:, :]
        .set(
            sigma_r
            * radial_face_factor
            / jnp.maximum(
                r_centers[None, 1:, None] * radial_widths[None, 1:, None],
                1.0e-20,
            )
        )
    )
    coef_r_outer = (
        jnp.zeros_like(coefficient)
        .at[:, :-1, :]
        .set(
            sigma_r
            * radial_face_factor
            / jnp.maximum(
                r_centers[None, :-1, None] * radial_widths[None, :-1, None],
                1.0e-20,
            )
        )
    )

    sigma_theta_out = _harmonic_mean(coefficient, jnp.roll(coefficient, -1, axis=2))
    coef_theta_out = sigma_theta_out / jnp.maximum(
        r_centers[None, :, None] ** 2 * dtheta**2, 1.0e-20
    )
    coef_theta_in = jnp.roll(coef_theta_out, 1, axis=2)
    return (
        coef_x_w,
        coef_x_e,
        coef_r_inner,
        coef_r_outer,
        coef_theta_in,
        coef_theta_out,
    )


def _apply_pipe_diffusion_coefficients_3d(
    field: jnp.ndarray, coefficients: tuple[jnp.ndarray, ...]
) -> jnp.ndarray:
    coef_x_w, coef_x_e, coef_r_i, coef_r_o, coef_t_i, coef_t_o = coefficients
    x_west = jnp.concatenate([field[:1], field[:-1]], axis=0)
    x_east = jnp.concatenate([field[1:], field[-1:]], axis=0)
    r_inner = jnp.concatenate([field[:, :1, :], field[:, :-1, :]], axis=1)
    r_outer = jnp.concatenate([field[:, 1:, :], field[:, -1:, :]], axis=1)
    theta_in = jnp.roll(field, 1, axis=2)
    theta_out = jnp.roll(field, -1, axis=2)
    return (
        coef_x_w * (x_west - field)
        + coef_x_e * (x_east - field)
        + coef_r_i * (r_inner - field)
        + coef_r_o * (r_outer - field)
        + coef_t_i * (theta_in - field)
        + coef_t_o * (theta_out - field)
    )


def _separable_pressure_poisson_pipe(
    rhs: jnp.ndarray,
    coefficient: jnp.ndarray,
    *,
    dx: float,
    r_faces: jnp.ndarray,
    r_centers: jnp.ndarray,
    dtheta: float,
    tolerance: float,
) -> tuple[jnp.ndarray, ...]:
    """Solve an axisymmetric cylindrical Neumann operator by x/theta modes."""

    nx, nr, ntheta = rhs.shape
    radial_widths = jnp.diff(r_faces)
    volume = jnp.broadcast_to(
        r_centers[None, :, None] * radial_widths[None, :, None] * dtheta,
        rhs.shape,
    )
    volume_sum = jnp.sum(volume)
    rhs_compatible = rhs - jnp.sum(rhs * volume) / volume_sum
    coefficients = _pipe_variable_diffusion_coefficients_3d(
        coefficient,
        dx=dx,
        r_faces=r_faces,
        r_centers=r_centers,
        dtheta=dtheta,
    )
    _, _, radial_inner, radial_outer, _, theta_outer = coefficients
    sigma = coefficient[0, :, 0]
    x_wave = jnp.pi * jnp.arange(nx, dtype=rhs.dtype) / nx
    theta_wave = 2.0 * jnp.pi * jnp.arange(ntheta // 2 + 1, dtype=rhs.dtype) / ntheta
    radial_inner_rate = radial_inner[0, :, 0]
    radial_outer_rate = radial_outer[0, :, 0]
    radial_rate = radial_inner_rate + radial_outer_rate
    modal_rate = (
        radial_rate[:, None, None]
        + 2.0 * sigma[:, None, None] * (1.0 - jnp.cos(x_wave))[None, :, None] / dx**2
        + 2.0
        * theta_outer[0, :, 0][:, None, None]
        * (1.0 - jnp.cos(theta_wave))[None, None, :]
    )
    radial_volume = r_centers * radial_widths * dtheta
    diagonal = radial_volume[:, None, None] * modal_rate
    lower = jnp.broadcast_to(
        -radial_volume[:, None, None] * radial_inner_rate[:, None, None],
        diagonal.shape,
    )
    upper = jnp.broadcast_to(
        -radial_volume[:, None, None] * radial_outer_rate[:, None, None],
        diagonal.shape,
    )
    modal_rhs = jnp.moveaxis(
        jnp.fft.rfft(
            dct(-volume * rhs_compatible, type=2, norm="ortho", axis=0), axis=2
        ),
        1,
        0,
    )
    # The compatible (kx, m) = (0, 0) radial system has one redundant row.
    # Pin its outer cell for the solve, then restore the volume-weighted gauge.
    lower = lower.at[-1, 0, 0].set(0.0)
    upper = upper.at[-1, 0, 0].set(0.0)
    diagonal = diagonal.at[-1, 0, 0].set(1.0)
    modal_rhs = modal_rhs.at[-1, 0, 0].set(0.0)
    modes = tridiagonal_solve(lower, diagonal, upper, modal_rhs.real) + 1j * (
        tridiagonal_solve(lower, diagonal, upper, modal_rhs.imag)
    )
    field = idct(
        jnp.fft.irfft(jnp.moveaxis(modes, 0, 1), n=ntheta, axis=2),
        type=2,
        norm="ortho",
        axis=0,
    )
    field = field - jnp.sum(field * volume) / volume_sum
    local_residual = jnp.max(
        jnp.abs(
            _apply_pipe_diffusion_coefficients_3d(field, coefficients) - rhs_compatible
        )
    )
    linear_rhs = -volume * rhs_compatible
    linear_residual = jnp.linalg.norm(
        -volume * _apply_pipe_diffusion_coefficients_3d(field, coefficients)
        - linear_rhs
    )
    relative_residual = linear_residual / jnp.maximum(
        jnp.linalg.norm(linear_rhs), 1.0e-30
    )
    converged = local_residual <= tolerance
    return (
        field,
        linear_residual,
        converged,
        relative_residual,
        jnp.asarray(1),
        jnp.where(converged, 1, 0),
        local_residual,
    )


def _solvax_pressure_poisson_pipe(
    rhs: jnp.ndarray,
    coefficient: jnp.ndarray,
    *,
    dx: float,
    r_faces: jnp.ndarray,
    r_centers: jnp.ndarray,
    dtheta: float,
    iterations: int,
    tolerance: float,
    initial_field: jnp.ndarray | None = None,
    local_tolerance: float | None = None,
    include_theta_line: bool = False,
) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
]:
    """Implicit SOLVAX solve for a cylindrical Neumann FV operator."""

    radial_widths = jnp.diff(r_faces)
    volume = jnp.broadcast_to(
        r_centers[None, :, None]
        * radial_widths[None, :, None]
        * jnp.asarray(dtheta, dtype=rhs.dtype),
        rhs.shape,
    )
    volume_sum = jnp.sum(volume)
    rhs_compatible = rhs - jnp.sum(rhs * volume) / volume_sum
    coefficients = _pipe_variable_diffusion_coefficients_3d(
        coefficient,
        dx=dx,
        r_faces=r_faces,
        r_centers=r_centers,
        dtheta=dtheta,
    )
    coef_x_w, coef_x_e, coef_r_i, coef_r_o, coef_t_i, coef_t_o = coefficients

    def matvec(field: jnp.ndarray) -> jnp.ndarray:
        diffusion = _apply_pipe_diffusion_coefficients_3d(field, coefficients)
        gauge = volume * jnp.sum(volume * field) / volume_sum
        return -volume * diffusion + gauge

    diagonal = volume * sum(coefficients) + volume**2 / volume_sum

    directions = (
        (0, -volume * coef_x_w, -volume * coef_x_e),
        (1, -volume * coef_r_i, -volume * coef_r_o),
    )
    precondition = _additive_line_preconditioner_3d(
        diagonal,
        directions,
        periodic_last_axis=(
            (-volume * coef_t_i, -volume * coef_t_o) if include_theta_line else None
        ),
    )

    linear_rhs = -volume * rhs_compatible
    effective_rtol = tolerance
    effective_atol = tolerance
    if local_tolerance is not None:
        local_absolute_target = float(jnp.min(volume)) * local_tolerance
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
        transpose_rtol=tolerance,
        transpose_atol=tolerance,
        transpose_max_steps=iterations,
    )
    return _finalize_local_pressure_solve(
        solution,
        linear_rhs=linear_rhs,
        matvec=matvec,
        local_residual_fn=lambda field: (
            _apply_pipe_diffusion_coefficients_3d(field, coefficients) - rhs_compatible
        ),
        volume=volume,
        precondition=precondition,
        iterations=iterations,
        effective_atol=effective_atol,
        local_tolerance=local_tolerance,
    )


def _solve_pipe_diffusion_system(
    linear_rhs: jnp.ndarray,
    volume: jnp.ndarray,
    coefficients: tuple[jnp.ndarray, ...],
    wall_sink: jnp.ndarray,
    initial_field: jnp.ndarray | None,
    *,
    mass_coefficient: float = 1.0,
    diffusion_coefficient: float = 1.0,
    iterations: int,
    tolerance: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Solve one prepared shifted cylindrical diffusion system."""

    coef_x_w, coef_x_e, coef_r_i, coef_r_o, _, _ = coefficients

    def matvec(field: jnp.ndarray) -> jnp.ndarray:
        diffusion = (
            _apply_pipe_diffusion_coefficients_3d(field, coefficients)
            - wall_sink * field
        )
        return volume * (mass_coefficient * field - diffusion_coefficient * diffusion)

    diagonal = volume * (
        mass_coefficient + diffusion_coefficient * (sum(coefficients) + wall_sink)
    )
    directions = (
        (
            0,
            -volume * diffusion_coefficient * coef_x_w,
            -volume * diffusion_coefficient * coef_x_e,
        ),
        (
            1,
            -volume * diffusion_coefficient * coef_r_i,
            -volume * diffusion_coefficient * coef_r_o,
        ),
    )
    precondition = _additive_line_preconditioner_3d(diagonal, directions)
    solution = pcg_linear_solve(
        matvec,
        linear_rhs,
        x0=initial_field,
        precond=precondition,
        transpose_precond=precondition,
        rtol=tolerance,
        atol=tolerance,
        max_steps=iterations,
        transpose_rtol=tolerance,
        transpose_atol=tolerance,
        transpose_max_steps=iterations,
    )
    return solution.x, solution.residual_norm, solution.converged


def _solvax_diffusion_pipe(
    rhs: jnp.ndarray,
    viscosity: jnp.ndarray,
    *,
    dt: float | None,
    dx: float,
    r_faces: jnp.ndarray,
    r_centers: jnp.ndarray,
    dtheta: float,
    iterations: int,
    tolerance: float,
    initial_field: jnp.ndarray | None = None,
    reaction: jnp.ndarray | None = None,
    decouple_axial: bool = False,
    _system_solve: Callable | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Solve steady or implicit cylindrical no-slip diffusion.

    ``dt=None`` solves ``-div(viscosity grad(field)) = rhs``. A positive
    ``dt`` solves the implicit update ``field - dt * div(...) = rhs``.
    ``decouple_axial`` retains the axial diagonal while dropping neighboring
    station couplings, providing a cross-section block-Jacobi inverse.
    """

    radial_widths = jnp.diff(r_faces)
    volume = jnp.broadcast_to(
        r_centers[None, :, None] * radial_widths[None, :, None] * dtheta,
        rhs.shape,
    )
    coefficients = _pipe_variable_diffusion_coefficients_3d(
        viscosity,
        dx=dx,
        r_faces=r_faces,
        r_centers=r_centers,
        dtheta=dtheta,
    )
    axial_sink = jnp.zeros_like(rhs)
    if decouple_axial:
        axial_sink = coefficients[0] + coefficients[1]
        zero = jnp.zeros_like(coefficients[0])
        coefficients = (zero, zero, *coefficients[2:])
    wall_sink = (
        jnp.zeros_like(rhs)
        .at[:, -1, :]
        .set(
            viscosity[:, -1, :]
            * r_faces[-1]
            / jnp.maximum(
                r_centers[-1] * radial_widths[-1] * (0.5 * radial_widths[-1]),
                1.0e-20,
            )
        )
    ) + axial_sink
    if reaction is not None:
        wall_sink = wall_sink + reaction
    system = (volume * rhs, volume, coefficients, wall_sink, initial_field)
    if _system_solve is not None:
        return _system_solve(*system)
    return _solve_pipe_diffusion_system(
        *system,
        mass_coefficient=float(dt is not None),
        diffusion_coefficient=1.0 if dt is None else dt,
        iterations=iterations,
        tolerance=tolerance,
    )


def _masked_laplacian_pipe(
    field: jnp.ndarray,
    fluid_mask: jnp.ndarray,
    *,
    dx: float,
    r_faces: jnp.ndarray,
    r_centers: jnp.ndarray,
    dtheta: float,
    radial_fluid_count: int,
) -> jnp.ndarray:
    """Cylindrical no-slip diffusion with the wall at the fluid radial face."""

    fluid = field[:, :radial_fluid_count, :]
    faces = r_faces[: radial_fluid_count + 1]
    centers = r_centers[:radial_fluid_count]
    ones = jnp.ones_like(fluid)
    coefficients = _pipe_variable_diffusion_coefficients_3d(
        ones,
        dx=dx,
        r_faces=faces,
        r_centers=centers,
        dtheta=dtheta,
    )
    laplacian = _apply_pipe_diffusion_coefficients_3d(fluid, coefficients)
    outer_wall = (
        -faces[-1]
        * fluid[:, -1, :]
        / jnp.maximum(
            centers[-1] * jnp.diff(faces)[-1] * (0.5 * jnp.diff(faces)[-1]),
            1.0e-20,
        )
    )
    laplacian = laplacian.at[:, -1, :].add(outer_wall)
    full = jnp.zeros_like(field).at[:, :radial_fluid_count, :].set(laplacian)
    return jnp.where(fluid_mask, full, 0.0)


def _pipe_velocity_faces(
    u: jnp.ndarray, v: jnp.ndarray, w: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Interpolate cylindrical cell velocities to conservative faces."""

    nx, nr, ntheta = u.shape
    uf = jnp.zeros((nx + 1, nr, ntheta), dtype=u.dtype)
    uf = uf.at[1:-1].set(0.5 * (u[1:] + u[:-1]))
    uf = uf.at[0].set(u[0])
    uf = uf.at[-1].set(u[-1])
    vf = jnp.zeros((nx, nr + 1, ntheta), dtype=v.dtype)
    vf = vf.at[:, 1:-1, :].set(0.5 * (v[:, 1:, :] + v[:, :-1, :]))
    wf = 0.5 * (w + jnp.roll(w, -1, axis=2))
    return uf, vf, wf


def _pipe_face_divergence(
    uf: jnp.ndarray,
    vf: jnp.ndarray,
    wf: jnp.ndarray,
    *,
    dx: float,
    r_faces: jnp.ndarray,
    r_centers: jnp.ndarray,
    dtheta: float,
) -> jnp.ndarray:
    """Return finite-volume divergence from cylindrical face fluxes."""

    widths = jnp.diff(r_faces)
    return (
        (uf[1:] - uf[:-1]) / max(dx, 1.0e-12)
        + (
            r_faces[None, 1:, None] * vf[:, 1:, :]
            - r_faces[None, :-1, None] * vf[:, :-1, :]
        )
        / jnp.maximum(r_centers[None, :, None] * widths[None, :, None], 1.0e-20)
        + (wf - jnp.roll(wf, 1, axis=2))
        / jnp.maximum(r_centers[None, :, None] * dtheta, 1.0e-20)
    )


def _pipe_pressure_face_correction(
    pressure: jnp.ndarray,
    mobility: jnp.ndarray,
    *,
    dx: float,
    r_centers: jnp.ndarray,
    dtheta: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return ``-mobility grad(pressure)`` on cylindrical faces."""

    nx, nr, ntheta = pressure.shape
    correction_x = jnp.zeros((nx + 1, nr, ntheta), dtype=pressure.dtype)
    correction_x = correction_x.at[1:-1].set(
        -_harmonic_mean(mobility[1:], mobility[:-1])
        * (pressure[1:] - pressure[:-1])
        / max(dx, 1.0e-12)
    )
    correction_r = jnp.zeros((nx, nr + 1, ntheta), dtype=pressure.dtype)
    correction_r = correction_r.at[:, 1:-1, :].set(
        -_harmonic_mean(mobility[:, 1:, :], mobility[:, :-1, :])
        * (pressure[:, 1:, :] - pressure[:, :-1, :])
        / jnp.diff(r_centers)[None, :, None]
    )
    correction_theta = (
        -_harmonic_mean(mobility, jnp.roll(mobility, -1, axis=2))
        * (jnp.roll(pressure, -1, axis=2) - pressure)
        / jnp.maximum(r_centers[None, :, None] * dtheta, 1.0e-20)
    )
    return correction_x, correction_r, correction_theta


def _pipe_face_velocity_cells(
    uf: jnp.ndarray, vf: jnp.ndarray, wf: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Reconstruct cylindrical cell velocities from conservative faces."""

    return (
        0.5 * (uf[:-1] + uf[1:]),
        0.5 * (vf[:, :-1, :] + vf[:, 1:, :]),
        0.5 * (wf + jnp.roll(wf, 1, axis=2)),
    )


def _pipe_retained_modal_factors(
    mobility: jnp.ndarray,
    pressure_mobility: jnp.ndarray,
    cell_area: jnp.ndarray,
    momentum_coefficients: tuple[jnp.ndarray, ...],
    momentum_sink: jnp.ndarray,
    *,
    dx: float,
    r_faces: jnp.ndarray,
    r_centers: jnp.ndarray,
    dtheta: float,
    modes: tuple[int, ...] = (1, 2, 3, 4),
) -> tuple[object, tuple[object, ...]]:
    """Factor separated axisymmetric low-mode Schur blocks."""

    nx, nr, ntheta = mobility.shape
    radial_widths = jnp.diff(r_faces)
    mobility = mobility[:, :, 0]
    pressure_mobility = pressure_mobility[:, :, 0]
    cell_area = cell_area[:, :, 0]
    coefficients = tuple(coefficient[:, :, 0] for coefficient in momentum_coefficients)
    momentum_sink = momentum_sink[:, :, 0]
    face_area = jnp.concatenate(
        (cell_area[:1], 0.5 * (cell_area[:-1] + cell_area[1:]), cell_area[-1:])
    )
    radial_weights = cell_area[0]

    def pressure_faces(pressure, coefficient, phase):
        axial = jnp.zeros((nx + 1, nr), dtype=pressure.dtype)
        axial = axial.at[1:-1].set(
            -_harmonic_mean(coefficient[1:], coefficient[:-1])
            * (pressure[1:] - pressure[:-1])
            / max(dx, 1.0e-12)
        )
        radial = jnp.zeros((nx, nr + 1), dtype=pressure.dtype)
        radial = radial.at[:, 1:-1].set(
            -_harmonic_mean(coefficient[:, 1:], coefficient[:, :-1])
            * (pressure[:, 1:] - pressure[:, :-1])
            / jnp.diff(r_centers)[None, :]
        )
        azimuthal = (
            -coefficient
            * (phase - 1.0)
            * pressure
            / jnp.maximum(r_centers[None, :] * dtheta, 1.0e-20)
        )
        return axial, radial, azimuthal

    def face_cells(axial, radial, azimuthal, phase):
        return (
            0.5 * (axial[:-1] + axial[1:]),
            0.5 * (radial[:, :-1] + radial[:, 1:]),
            0.5 * (1.0 + 1.0 / phase) * azimuthal,
        )

    def velocity_faces(axial, radial, azimuthal, phase):
        axial_faces = jnp.zeros((nx + 1, nr), dtype=axial.dtype)
        axial_faces = axial_faces.at[1:-1].set(0.5 * (axial[1:] + axial[:-1]))
        axial_faces = axial_faces.at[0].set(axial[0]).at[-1].set(axial[-1])
        radial_faces = jnp.zeros((nx, nr + 1), dtype=radial.dtype)
        radial_faces = radial_faces.at[:, 1:-1].set(
            0.5 * (radial[:, 1:] + radial[:, :-1])
        )
        return axial_faces, radial_faces, 0.5 * (1.0 + phase) * azimuthal

    def momentum_inverse(rhs, phase):
        coef_x_w, coef_x_e, coef_r_i, coef_r_o, coef_t_i, coef_t_o = coefficients
        diagonal = (
            coef_x_w
            + coef_x_e
            + coef_r_i
            + coef_r_o
            + coef_t_i * (1.0 - 1.0 / phase)
            + coef_t_o * (1.0 - phase)
            + momentum_sink
        ).real
        lower = -coef_r_i
        upper = -coef_r_o
        solve_args = tuple(
            jnp.swapaxes(array, 0, 1) for array in (lower, diagonal, upper, rhs)
        )
        solved = tridiagonal_solve(*solve_args[:3], solve_args[3].real) + 1j * (
            tridiagonal_solve(*solve_args[:3], solve_args[3].imag)
        )
        return jnp.swapaxes(solved, 0, 1)

    def divergence(axial, radial, azimuthal, phase):
        return (
            (axial[1:] - axial[:-1]) / max(dx, 1.0e-12)
            + (r_faces[None, 1:] * radial[:, 1:] - r_faces[None, :-1] * radial[:, :-1])
            / jnp.maximum(r_centers[None, :] * radial_widths[None, :], 1.0e-20)
            + (1.0 - 1.0 / phase)
            * azimuthal
            / jnp.maximum(r_centers[None, :] * dtheta, 1.0e-20)
        )

    def modal_action(radial_pressure, source, phase, zero_mean):
        if zero_mean:
            final = -jnp.sum(radial_pressure * radial_weights[:-1]) / jnp.maximum(
                radial_weights[-1], 1.0e-20
            )
            radial_pressure = jnp.concatenate((radial_pressure, final[None]))
        pressure = jnp.zeros((nx, nr), dtype=jnp.result_type(mobility, 1j))
        pressure = pressure.at[source].set(radial_pressure)
        forcing_faces = pressure_faces(pressure, mobility, phase)
        forcing = face_cells(*forcing_faces, phase)
        response = tuple(momentum_inverse(force, phase) for force in forcing)

        direct_faces = pressure_faces(pressure, pressure_mobility, phase)
        reconstructed = velocity_faces(*face_cells(*direct_faces, phase), phase)
        stabilization = tuple(
            direct - recovered for direct, recovered in zip(direct_faces, reconstructed)
        )
        axial = stabilization[0]
        if zero_mean:
            axial = axial - jnp.sum(axial * face_area, axis=1)[:, None] / jnp.maximum(
                jnp.sum(face_area, axis=1)[:, None], 1.0e-30
            )
        axial = axial.at[0].set(0.0).at[-1].set(0.0)
        response_faces = velocity_faces(*response, phase)
        result = divergence(
            response_faces[0] + axial,
            response_faces[1] + stabilization[1],
            response_faces[2] + stabilization[2],
            phase,
        )
        if zero_mean:
            result = result - jnp.sum(result * cell_area, axis=1)[
                :, None
            ] / jnp.maximum(jnp.sum(cell_area, axis=1)[:, None], 1.0e-20)
            return result[:, :-1]
        return result

    def factor_mode(mode, size, zero_mean):
        phase = 1.0 if zero_mean else jnp.exp(2j * jnp.pi * mode / ntheta)
        jacobian = jax.jit(
            jax.jacfwd(
                lambda pressure, source: modal_action(
                    pressure, source, phase, zero_mean
                )
            )
        )
        actions = tuple(
            jacobian(jnp.zeros((size,), dtype=mobility.dtype), source)
            for source in range(nx)
        )
        if zero_mean:
            actions = tuple(action.real for action in actions)
        empty = jnp.zeros((size, size), dtype=actions[0].dtype)
        blocks = (
            jnp.stack((empty, *(actions[i - 1][i] for i in range(1, nx)))),
            jnp.stack(tuple(actions[i][i] for i in range(nx))),
            jnp.stack((*(actions[i + 1][i] for i in range(nx - 1)), empty)),
        )
        return block_thomas_factor(*blocks)

    return (
        factor_mode(0, nr - 1, True),
        tuple(factor_mode(mode, nr, False) for mode in modes),
    )


def _solve_pipe_retained_modal_factors(factors, residual):
    """Solve real cosine/sine residuals with separated complex factors."""

    axisymmetric, modes = factors
    radial_size = axisymmetric[0].shape[-1]
    radial = block_thomas_solve(axisymmetric, residual[:, :radial_size])
    mode_rhs = residual[:, radial_size:].reshape((residual.shape[0], 2, len(modes), -1))
    solved = jnp.stack(
        tuple(
            block_thomas_solve(factor, mode_rhs[:, 0, i] - 1j * mode_rhs[:, 1, i])
            for i, factor in enumerate(modes)
        ),
        axis=1,
    )
    return jnp.concatenate(
        (
            radial,
            jnp.stack((solved.real, -solved.imag), axis=1).reshape(
                (residual.shape[0], -1)
            ),
        ),
        axis=1,
    )


def _steady_stokes_projection_pipe(
    u: jnp.ndarray,
    v: jnp.ndarray,
    w: jnp.ndarray,
    rho: jnp.ndarray,
    unit_flow_response: jnp.ndarray,
    cell_area: jnp.ndarray,
    apply_momentum_inverse: Callable[[jnp.ndarray], jnp.ndarray],
    *,
    target_flow_rate: float,
    dx: float,
    r_faces: jnp.ndarray,
    r_centers: jnp.ndarray,
    dtheta: float,
    pressure_iterations: int,
    pressure_tolerance: float,
    restart: int = 24,
    max_restarts: int = 8,
    flow_response_matrix: jnp.ndarray | None = None,
    pressure_preconditioner_mobility: jnp.ndarray | None = None,
    apply_momentum_inverse_components: Callable[[jnp.ndarray], jnp.ndarray]
    | None = None,
    apply_modal_momentum_inverse: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    modal_momentum_coefficients: tuple[jnp.ndarray, ...] | None = None,
    modal_momentum_sink: jnp.ndarray | None = None,
    modal_stabilization: bool = False,
    modal_factor_key: tuple[object, ...] | None = None,
    physical_tolerance: float | None = None,
) -> tuple[jnp.ndarray, ...]:
    """Apply the compatible steady ``D A^-1 G`` pipe projection."""

    response_flow = jnp.sum(unit_flow_response * cell_area, axis=(1, 2))
    area_sum = jnp.sum(cell_area, axis=(1, 2), keepdims=True)
    mobility = 1.0 / jnp.maximum(rho, 1.0e-20)
    pressure_mobility = (
        jnp.maximum(unit_flow_response, jnp.mean(unit_flow_response) * 1.0e-8)
        if pressure_preconditioner_mobility is None
        else pressure_preconditioner_mobility
    )
    cross_section_size = u.shape[1] * u.shape[2]
    pressure_size = u.shape[0] * (cross_section_size - 1)
    flat_area = cell_area.reshape((u.shape[0], cross_section_size))
    sqrt_area = jnp.sqrt(jnp.maximum(flat_area, 1.0e-30))
    gauge = sqrt_area / jnp.linalg.norm(sqrt_area, axis=1, keepdims=True)
    householder = gauge.at[:, -1].add(-1.0)
    householder_scale = 2.0 / jnp.maximum(
        jnp.sum(householder**2, axis=1, keepdims=True), 1.0e-30
    )

    def reflect(field):
        return field - householder_scale * householder * jnp.sum(
            householder * field, axis=1, keepdims=True
        )

    def unpack_pressure(reduced):
        reduced = reduced.reshape((u.shape[0], cross_section_size - 1))
        if modal_stabilization:
            transformed = reflect(
                jnp.concatenate(
                    (reduced, jnp.zeros((u.shape[0], 1), dtype=reduced.dtype)), axis=1
                )
            )
            return (transformed / sqrt_area).reshape(u.shape)
        final = -jnp.sum(reduced * flat_area[:, :-1], axis=1) / jnp.maximum(
            flat_area[:, -1], 1.0e-20
        )
        return jnp.concatenate((reduced, final[:, None]), axis=1).reshape(u.shape)

    def reduce_field(field):
        if modal_stabilization:
            transformed = reflect(sqrt_area * field.reshape(flat_area.shape))
            return transformed[:, :-1].reshape(-1)
        return field.reshape((u.shape[0], cross_section_size))[:, :-1].reshape(-1)

    def velocity_response(
        state, *, use_component_inverse=True, use_modal_inverse=False
    ):
        pressure = unpack_pressure(state[:pressure_size])
        pressure_loss = state[pressure_size:]
        face_force = _pipe_pressure_face_correction(
            pressure,
            mobility,
            dx=dx,
            r_centers=r_centers,
            dtheta=dtheta,
        )
        force_u, force_v, force_w = _pipe_face_velocity_cells(*face_force)
        force_u = force_u + pressure_loss[:, None, None] * mobility
        forces = jnp.stack((force_u, force_v, force_w))
        momentum_inverse = (
            apply_modal_momentum_inverse
            if use_modal_inverse and apply_modal_momentum_inverse is not None
            else apply_momentum_inverse
        )
        responses = (
            jnp.stack(tuple(momentum_inverse(force) for force in forces))
            if apply_momentum_inverse_components is None
            or not use_component_inverse
            or use_modal_inverse
            else apply_momentum_inverse_components(forces)
        )
        return tuple(responses)

    def rhie_chow_faces(pressure, response):
        pressure_faces = _pipe_pressure_face_correction(
            pressure,
            pressure_mobility,
            dx=dx,
            r_centers=r_centers,
            dtheta=dtheta,
        )
        reconstructed_faces = _pipe_velocity_faces(
            *_pipe_face_velocity_cells(*pressure_faces)
        )
        stabilization = tuple(
            direct - reconstructed
            for direct, reconstructed in zip(pressure_faces, reconstructed_faces)
        )
        face_area = jnp.concatenate(
            (
                cell_area[:1],
                0.5 * (cell_area[:-1] + cell_area[1:]),
                cell_area[-1:],
            ),
            axis=0,
        )
        axial_mean = jnp.sum(stabilization[0] * face_area, axis=(1, 2)) / jnp.maximum(
            jnp.sum(face_area, axis=(1, 2)), 1.0e-30
        )
        axial = stabilization[0] - axial_mean[:, None, None]
        axial = axial.at[0].set(0.0).at[-1].set(0.0)
        stabilization = (axial, *stabilization[1:])
        return tuple(
            exact + correction
            for exact, correction in zip(_pipe_velocity_faces(*response), stabilization)
        )

    def constraints(state_u, state_v, state_w, *, faces=None):
        divergence = _pipe_face_divergence(
            *(
                _pipe_velocity_faces(state_u, state_v, state_w)
                if faces is None
                else faces
            ),
            dx=dx,
            r_faces=r_faces,
            r_centers=r_centers,
            dtheta=dtheta,
        )
        mean_divergence = jnp.sum(
            divergence * cell_area, axis=(1, 2), keepdims=True
        ) / jnp.maximum(area_sum, 1.0e-20)
        flow = jnp.sum(state_u * cell_area, axis=(1, 2))
        return jnp.concatenate((reduce_field(divergence - mean_divergence), flow))

    base_constraints = constraints(u, v, w)
    rhs = -base_constraints.at[pressure_size:].add(-target_flow_rate)

    def schur(state, *, use_component_inverse=True, use_modal_inverse=False):
        response = velocity_response(
            state,
            use_component_inverse=use_component_inverse,
            use_modal_inverse=use_modal_inverse,
        )
        if not modal_stabilization:
            return constraints(*response)
        pressure = unpack_pressure(state[:pressure_size])
        return constraints(*response, faces=rhie_chow_faces(pressure, response))

    def local_precondition(residual):
        divergence = unpack_pressure(residual[:pressure_size])
        pressure, *_ = _solvax_pressure_poisson_pipe(
            -divergence,
            pressure_mobility,
            dx=dx,
            r_faces=r_faces,
            r_centers=r_centers,
            dtheta=dtheta,
            iterations=pressure_iterations,
            tolerance=pressure_tolerance,
            include_theta_line=True,
        )
        flow_residual = residual[pressure_size:]
        pressure_loss = (
            jnp.linalg.solve(flow_response_matrix, flow_residual)
            if flow_response_matrix is not None
            else flow_residual / jnp.maximum(jnp.mean(response_flow), 1.0e-20)
        )
        return jnp.concatenate((reduce_field(pressure), pressure_loss))

    if modal_stabilization:
        radial_weights = jnp.sum(cell_area, axis=2)
        coarse_pressure_size = u.shape[0] * (u.shape[1] - 1)
        use_direct_modal_factors = (
            modal_momentum_coefficients is not None and modal_momentum_sink is not None
        )
        modal_modes = (1, 2, 3, 4) if use_direct_modal_factors else (2,)
        mode_size = 2 * len(modal_modes) * u.shape[0] * u.shape[1]
        coarse_size = coarse_pressure_size + mode_size + u.shape[0]
        theta = jnp.arange(u.shape[2], dtype=u.dtype) * dtheta
        mode_angles = jnp.asarray(modal_modes, dtype=u.dtype)[:, None] * theta
        mode_cosine = jnp.cos(mode_angles)
        mode_sine = jnp.sin(mode_angles)

        def prolong(coarse):
            radial = coarse[:coarse_pressure_size].reshape((u.shape[0], u.shape[1] - 1))
            final = -jnp.sum(radial * radial_weights[:, :-1], axis=1) / jnp.maximum(
                radial_weights[:, -1], 1.0e-20
            )
            pressure = jnp.broadcast_to(
                jnp.concatenate((radial, final[:, None]), axis=1)[:, :, None],
                u.shape,
            )
            offset = coarse_pressure_size
            modes = coarse[offset : offset + mode_size].reshape(
                (2, len(modal_modes), u.shape[0], u.shape[1])
            )
            pressure = (
                pressure
                + jnp.einsum("mxr,mt->xrt", modes[0], mode_cosine)
                + jnp.einsum("mxr,mt->xrt", modes[1], mode_sine)
            )
            return jnp.concatenate(
                (
                    reduce_field(pressure),
                    coarse[coarse_pressure_size + mode_size :],
                )
            )

        def restrict(residual):
            divergence = unpack_pressure(residual[:pressure_size])
            return jnp.concatenate(
                (
                    jnp.mean(divergence, axis=2)[:, :-1].reshape(-1),
                    (
                        2.0
                        / u.shape[2]
                        * jnp.stack(
                            (
                                jnp.einsum("xrt,mt->mxr", divergence, mode_cosine),
                                jnp.einsum("xrt,mt->mxr", divergence, mode_sine),
                            )
                        )
                    ).reshape(-1),
                    residual[pressure_size:],
                )
            )

        local_size = u.shape[1] - 1 + 2 * len(modal_modes) * u.shape[1]
        local_basis = jnp.eye(local_size, dtype=u.dtype)

        def station_prolong(station, local):
            coarse = jnp.zeros((coarse_size,), dtype=u.dtype)
            radial = coarse[:coarse_pressure_size].reshape((u.shape[0], u.shape[1] - 1))
            radial = radial.at[station].set(local[: u.shape[1] - 1])
            coarse = coarse.at[:coarse_pressure_size].set(radial.reshape(-1))
            offset = coarse_pressure_size
            modes = coarse[offset : offset + mode_size].reshape(
                (2, len(modal_modes), u.shape[0], u.shape[1])
            )
            modes = modes.at[:, :, station].set(
                local[u.shape[1] - 1 :].reshape((2, len(modal_modes), u.shape[1]))
            )
            coarse = coarse.at[offset : offset + mode_size].set(modes.reshape(-1))
            return prolong(coarse)

        def modal_restrict(residual):
            coarse = restrict(residual)
            radial = coarse[:coarse_pressure_size].reshape((u.shape[0], u.shape[1] - 1))
            modes = coarse[
                coarse_pressure_size : coarse_pressure_size + mode_size
            ].reshape((2, len(modal_modes), u.shape[0], u.shape[1]))
            return jnp.concatenate(
                (radial, jnp.transpose(modes, (2, 0, 1, 3)).reshape((u.shape[0], -1))),
                axis=1,
            )

        stations = jnp.arange(u.shape[0])

        def build_modal_factors():
            if use_direct_modal_factors:
                return _pipe_retained_modal_factors(
                    mobility,
                    pressure_mobility,
                    cell_area,
                    modal_momentum_coefficients,
                    modal_momentum_sink,
                    dx=dx,
                    r_faces=r_faces,
                    r_centers=r_centers,
                    dtheta=dtheta,
                    modes=modal_modes,
                )

            def modal_action(source, basis):
                return modal_restrict(
                    schur(
                        station_prolong(source, basis),
                        use_component_inverse=False,
                        use_modal_inverse=True,
                    )
                )

            if apply_modal_momentum_inverse is None:
                modal_actions = jax.vmap(
                    lambda source: jax.vmap(lambda basis: modal_action(source, basis))(
                        local_basis
                    )
                )(stations)
            else:
                modal_actions = jnp.stack(
                    tuple(
                        jnp.stack(
                            tuple(modal_action(source, basis) for basis in local_basis)
                        )
                        for source in stations
                    )
                )
            diagonal = jax.vmap(
                lambda station: modal_actions[station, :, station, :].T
            )(stations)
            lower = (
                jnp.zeros_like(diagonal)
                .at[1:]
                .set(
                    jax.vmap(lambda target: modal_actions[target - 1, :, target, :].T)(
                        stations[1:]
                    )
                )
            )
            upper = (
                jnp.zeros_like(diagonal)
                .at[:-1]
                .set(
                    jax.vmap(lambda target: modal_actions[target + 1, :, target, :].T)(
                        stations[:-1]
                    )
                )
            )
            return block_thomas_factor(lower, diagonal, upper)

        modal_factors = (
            build_modal_factors()
            if modal_factor_key is None
            else _reuse_modal_factors(modal_factor_key, build_modal_factors)
        )

        def modal_prolong(local):
            coarse = jnp.zeros((coarse_size,), dtype=u.dtype)
            coarse = coarse.at[:coarse_pressure_size].set(
                local[:, : u.shape[1] - 1].reshape(-1)
            )
            modes = local[:, u.shape[1] - 1 :].reshape(
                (u.shape[0], 2, len(modal_modes), u.shape[1])
            )
            coarse = coarse.at[
                coarse_pressure_size : coarse_pressure_size + mode_size
            ].set(jnp.transpose(modes, (1, 2, 0, 3)).reshape(-1))
            return prolong(coarse)

        def precondition(residual):
            local = local_precondition(residual)
            modal_residual = modal_restrict(residual - schur(local))
            modal_correction = (
                _solve_pipe_retained_modal_factors(modal_factors, modal_residual)
                if use_direct_modal_factors
                else block_thomas_solve(modal_factors, modal_residual)
            )
            candidate = local + modal_prolong(modal_correction)
            flow_residual = (residual - schur(candidate))[pressure_size:]
            flow_response = (
                jnp.linalg.solve(flow_response_matrix, flow_residual)
                if flow_response_matrix is not None
                else flow_residual / jnp.maximum(jnp.mean(response_flow), 1.0e-20)
            )
            return candidate + jnp.zeros_like(candidate).at[pressure_size:].set(
                flow_response
            )

    else:
        precondition = local_precondition

    if modal_factor_key is not None:
        pressure_kernel_key = (
            jax.default_backend(),
            modal_factor_key,
            pressure_iterations,
            pressure_tolerance,
        )
        schur = _reuse_fringing_jit(
            ("b1_steady_schur", *pressure_kernel_key),
            jax.jit(schur, inline=False),
        )
        precondition = _reuse_fringing_jit(
            ("b1_steady_preconditioner", *pressure_kernel_key),
            jax.jit(precondition, inline=False),
        )

    preconditioned_rhs = precondition(rhs)

    def physical_constraint_residual(state, linear_rhs):
        residual = schur(state) - linear_rhs
        divergence = unpack_pressure(residual[:pressure_size])
        flow = residual[pressure_size:] / jnp.maximum(jnp.mean(area_sum), 1.0e-30)
        return jnp.maximum(jnp.max(jnp.abs(divergence)), jnp.max(jnp.abs(flow)))

    def solve_pressure(linear_rhs, initial):
        if physical_tolerance is None:
            return gmres(
                schur,
                linear_rhs,
                x0=initial,
                precond=precondition,
                restart=restart,
                rtol=pressure_tolerance,
                atol=pressure_tolerance,
                max_restarts=max_restarts,
            )
        pilot_tolerance = max(pressure_tolerance, min(1.0e-6, physical_tolerance))
        pilot = gmres(
            schur,
            linear_rhs,
            x0=initial,
            precond=precondition,
            restart=restart,
            rtol=pilot_tolerance,
            atol=pilot_tolerance,
            max_restarts=1,
        )
        pilot_passes = (
            physical_constraint_residual(pilot.x, linear_rhs) <= physical_tolerance
        )

        def accept_pilot(_):
            return pilot

        def refine_pilot(_):
            refined = gmres(
                schur,
                linear_rhs,
                x0=pilot.x,
                precond=precondition,
                restart=restart,
                rtol=pressure_tolerance,
                atol=pressure_tolerance,
                max_restarts=max_restarts,
            )
            return KrylovSolution(
                refined.x,
                refined.residual_norm,
                pilot.iterations + refined.iterations,
                refined.converged,
                refined.recycle,
            )

        return jax.lax.cond(pilot_passes, accept_pilot, refine_pilot, operand=None)

    if modal_factor_key is not None:
        solve_pressure = _reuse_fringing_jit(
            (
                "b1_steady_pressure",
                jax.default_backend(),
                modal_factor_key,
                pressure_iterations,
                pressure_tolerance,
                restart,
                max_restarts,
                physical_tolerance,
            ),
            jax.jit(solve_pressure),
        )
    pressure_solution = solve_pressure(rhs, preconditioned_rhs)
    pressure = unpack_pressure(pressure_solution.x[:pressure_size])
    pressure_loss = pressure_solution.x[pressure_size:]
    correction = velocity_response(pressure_solution.x)
    projected = tuple(base + delta for base, delta in zip((u, v, w), correction))
    if flow_response_matrix is not None:
        flow_delta = target_flow_rate - jnp.sum(projected[0] * cell_area, axis=(1, 2))
        flow_refinement_state = (
            jnp.zeros_like(pressure_solution.x)
            .at[pressure_size:]
            .set(jnp.linalg.solve(flow_response_matrix, flow_delta))
        )
        flow_refinement = velocity_response(flow_refinement_state)
        correction = tuple(
            base + delta for base, delta in zip(correction, flow_refinement)
        )
        projected = tuple(base + delta for base, delta in zip((u, v, w), correction))
        pressure_loss = pressure_loss + flow_refinement_state[pressure_size:]
        refined_state = pressure_solution.x + flow_refinement_state
        refined_residual = jnp.linalg.norm(schur(refined_state) - rhs)
        pressure_solution = KrylovSolution(
            refined_state,
            refined_residual,
            pressure_solution.iterations,
            refined_residual <= pressure_tolerance,
            pressure_solution.recycle,
        )
    final_flow = jnp.sum(projected[0] * cell_area, axis=(1, 2))
    projected_faces = _pipe_velocity_faces(*projected)
    if modal_stabilization:
        response_faces = rhie_chow_faces(pressure, correction)
        projected_faces = tuple(
            base + response
            for base, response in zip(_pipe_velocity_faces(u, v, w), response_faces)
        )
    final_divergence = _pipe_face_divergence(
        *projected_faces,
        dx=dx,
        r_faces=r_faces,
        r_centers=r_centers,
        dtheta=dtheta,
    )
    final_mean_divergence = jnp.sum(
        final_divergence * cell_area, axis=(1, 2), keepdims=True
    ) / jnp.maximum(area_sum, 1.0e-30)
    final_mean_free_divergence = final_divergence - final_mean_divergence
    divergence_residual = jnp.max(jnp.abs(final_mean_free_divergence))
    flow_residual = jnp.max(jnp.abs(final_flow - target_flow_rate))
    normalized_flow_residual = flow_residual / jnp.maximum(jnp.mean(area_sum), 1.0e-30)
    physical_residual = jnp.maximum(divergence_residual, normalized_flow_residual)
    convergence_tolerance = (
        pressure_tolerance if physical_tolerance is None else physical_tolerance
    )
    pressure_solution = KrylovSolution(
        pressure_solution.x,
        physical_residual,
        pressure_solution.iterations,
        physical_residual <= convergence_tolerance,
        pressure_solution.recycle,
    )
    return (
        *projected,
        pressure,
        pressure_loss,
        divergence_residual,
        flow_residual,
        pressure_solution,
    )


def _face_flux_pressure_projection_pipe(
    u: jnp.ndarray,
    v: jnp.ndarray,
    w: jnp.ndarray,
    rho: jnp.ndarray,
    fluid_mask: jnp.ndarray,
    *,
    dt: float,
    dx: float,
    r_faces: jnp.ndarray,
    r_centers: jnp.ndarray,
    dtheta: float,
    iterations: int,
    tolerance: float,
    radial_fluid_count: int | None = None,
    initial_pressure: jnp.ndarray | None = None,
    include_theta_line: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Project mapped-pipe velocity using the same cylindrical face flux."""

    count = (
        _pipe_radial_fluid_count(fluid_mask)
        if radial_fluid_count is None
        else radial_fluid_count
    )
    us = u[:, :count, :]
    vs = v[:, :count, :]
    ws = w[:, :count, :]
    rhos = rho[:, :count, :]
    faces = r_faces[: count + 1]
    centers = r_centers[:count]
    uf, vf, wf = _pipe_velocity_faces(us, vs, ws)
    divergence = _pipe_face_divergence(
        uf,
        vf,
        wf,
        dx=dx,
        r_faces=faces,
        r_centers=centers,
        dtheta=dtheta,
    )
    mobility = dt / jnp.maximum(rhos, 1.0e-20)
    pressure, _, _, _, _, _, _ = _solvax_pressure_poisson_pipe(
        divergence,
        mobility,
        dx=dx,
        r_faces=faces,
        r_centers=centers,
        dtheta=dtheta,
        iterations=iterations,
        tolerance=tolerance,
        initial_field=(
            None if initial_pressure is None else initial_pressure[:, :count, :]
        ),
        include_theta_line=include_theta_line,
    )

    correction = _pipe_pressure_face_correction(
        pressure,
        mobility,
        dx=dx,
        r_centers=centers,
        dtheta=dtheta,
    )
    uf, vf, wf = (face + delta for face, delta in zip((uf, vf, wf), correction))
    divergence_after = _pipe_face_divergence(
        uf,
        vf,
        wf,
        dx=dx,
        r_faces=faces,
        r_centers=centers,
        dtheta=dtheta,
    )

    projected_u, projected_v, projected_w = _pipe_face_velocity_cells(uf, vf, wf)
    full_u = jnp.zeros_like(u).at[:, :count, :].set(projected_u)
    full_v = jnp.zeros_like(v).at[:, :count, :].set(projected_v)
    full_w = jnp.zeros_like(w).at[:, :count, :].set(projected_w)
    full_p = jnp.zeros_like(u).at[:, :count, :].set(pressure)
    return full_u, full_v, full_w, full_p, jnp.max(jnp.abs(divergence_after))


def _fixed_flow_face_flux_projection_pipe(
    u: jnp.ndarray,
    v: jnp.ndarray,
    w: jnp.ndarray,
    rho: jnp.ndarray,
    fluid_mask: jnp.ndarray,
    unit_pressure_response: jnp.ndarray,
    cell_area: jnp.ndarray,
    *,
    target_flow_rate: float,
    base_pressure_loss_gradient: float | jnp.ndarray,
    dt: float,
    dx: float,
    r_faces: jnp.ndarray,
    r_centers: jnp.ndarray,
    dtheta: float,
    iterations: int,
    tolerance: float,
    radial_fluid_count: int,
    initial_pressure: jnp.ndarray | None = None,
    include_theta_line: bool = False,
) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
]:
    response_delta = unit_pressure_response - unit_pressure_response[:1]
    if float(jnp.max(jnp.abs(response_delta))) > 1.0e-12:
        raise ValueError("Fixed-flow pipe projection requires an x-invariant response")
    constrained_u, pressure_loss_gradient = _apply_fixed_flow_pressure_constraint(
        u,
        unit_pressure_response=unit_pressure_response,
        active_mask=fluid_mask,
        cell_area=cell_area,
        target_flow_rate=target_flow_rate,
        base_pressure_loss_gradient=base_pressure_loss_gradient,
    )
    projected_u, projected_v, projected_w, pressure, divergence = (
        _face_flux_pressure_projection_pipe(
            constrained_u,
            v,
            w,
            rho,
            fluid_mask,
            dt=dt,
            dx=dx,
            r_faces=r_faces,
            r_centers=r_centers,
            dtheta=dtheta,
            iterations=iterations,
            tolerance=tolerance,
            radial_fluid_count=radial_fluid_count,
            initial_pressure=initial_pressure,
            include_theta_line=include_theta_line,
        )
    )
    projected_flow = jnp.sum(
        jnp.where(fluid_mask, projected_u * cell_area, 0.0), axis=(1, 2)
    )
    response_flow = jnp.sum(
        jnp.where(fluid_mask, unit_pressure_response * cell_area, 0.0), axis=(1, 2)
    )
    mean_response = jnp.mean(response_flow)
    if float(jnp.abs(mean_response)) <= 1.0e-20:
        raise ValueError("Fixed-flow pipe pressure response must be nonzero")
    global_multiplier = (
        jnp.asarray(target_flow_rate, dtype=u.dtype) - jnp.mean(projected_flow)
    ) / mean_response
    projected_u = jnp.where(
        fluid_mask,
        projected_u + global_multiplier * unit_pressure_response,
        0.0,
    )
    pressure_loss_gradient = pressure_loss_gradient + global_multiplier
    final_flow = jnp.sum(
        jnp.where(fluid_mask, projected_u * cell_area, 0.0), axis=(1, 2)
    )
    flow_error = jnp.max(
        jnp.abs(final_flow - jnp.asarray(target_flow_rate, dtype=u.dtype))
    )
    return (
        projected_u,
        projected_v,
        projected_w,
        pressure,
        pressure_loss_gradient,
        divergence,
        flow_error,
    )


def _pipe_poisson_jacobi_3d(
    rhs: jnp.ndarray,
    *,
    dx: float,
    dr: float,
    dtheta: float,
    r: jnp.ndarray,
    iterations: int,
    tolerance: float,
) -> tuple[jnp.ndarray, float, int, float]:
    weights = jnp.maximum(r, 0.5 * dr)
    rhs_compatible = rhs - jnp.sum(rhs * weights) / jnp.sum(weights)
    safe_r = jnp.maximum(r, 0.5 * dr)
    field = jnp.zeros_like(rhs_compatible)
    initial_residual = float(
        jnp.max(
            jnp.abs(
                _pipe_laplacian_3d(
                    field, dx=dx, dr=dr, dtheta=dtheta, r=r, outer_dirichlet=False
                )
                - rhs_compatible
            )
        )
    )
    residual = initial_residual
    iteration_count = 0
    diagonal = (
        2.0 / max(dx**2, 1.0e-12)
        + 2.0 / max(dr**2, 1.0e-12)
        + 2.0 / jnp.maximum((safe_r**2) * dtheta**2, 1.0e-12)
    )
    diagonal = jnp.maximum(diagonal, 1.0e-12)
    for iteration in range(iterations):
        x_west = jnp.concatenate([field[:1], field[:-1]], axis=0)
        x_east = jnp.concatenate([field[1:], field[-1:]], axis=0)
        r_inner = jnp.concatenate([field[:, :1, :], field[:, :-1, :]], axis=1)
        r_outer = jnp.concatenate([field[:, 1:, :], field[:, -1:, :]], axis=1)
        theta_prev, theta_next = _pipe_theta_neighbors(field)
        cross = (
            (x_west + x_east) / max(dx**2, 1.0e-12)
            + (r_inner + r_outer) / max(dr**2, 1.0e-12)
            + (theta_prev + theta_next) / jnp.maximum((safe_r**2) * dtheta**2, 1.0e-12)
            + (r_outer - r_inner) / jnp.maximum(2.0 * safe_r * dr, 1.0e-12)
        )
        updated = (cross - rhs_compatible) / diagonal
        updated = updated.at[:, 0, :].set(updated[:, 1, :])
        field = jnp.nan_to_num(updated - jnp.sum(updated * weights) / jnp.sum(weights))
        residual = float(
            jnp.max(
                jnp.abs(
                    _pipe_laplacian_3d(
                        field, dx=dx, dr=dr, dtheta=dtheta, r=r, outer_dirichlet=False
                    )
                    - rhs_compatible
                )
            )
        )
        iteration_count = iteration + 1
        if residual <= tolerance:
            break
    return field, residual, iteration_count, initial_residual


def _pipe_conservative_current_fluxes_3d(
    sigma: jnp.ndarray,
    phi: jnp.ndarray,
    uxb_x: jnp.ndarray,
    uxb_r: jnp.ndarray,
    uxb_theta: jnp.ndarray,
    *,
    dx: float,
    r_faces: jnp.ndarray,
    r_centers: jnp.ndarray,
    dtheta: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    nx, nr, ntheta = phi.shape
    fx = jnp.zeros((nx + 1, nr, ntheta), dtype=phi.dtype)
    fr = jnp.zeros((nx, nr + 1, ntheta), dtype=phi.dtype)
    ftheta = jnp.zeros((nx, nr, ntheta), dtype=phi.dtype)

    sigma_x = _harmonic_mean(sigma[1:], sigma[:-1])
    phi_grad_x = (phi[1:] - phi[:-1]) / max(dx, 1.0e-12)
    uxb_face_x = 0.5 * (uxb_x[1:] + uxb_x[:-1])
    fx = fx.at[1:-1].set(sigma_x * (-phi_grad_x + uxb_face_x))

    dr_centers = jnp.maximum(
        0.5 * (jnp.diff(r_faces)[1:] + jnp.diff(r_faces)[:-1]), 1.0e-12
    )
    sigma_r = _harmonic_mean(sigma[:, 1:, :], sigma[:, :-1, :])
    phi_grad_r = (phi[:, 1:, :] - phi[:, :-1, :]) / dr_centers[None, :, None]
    uxb_face_r = 0.5 * (uxb_r[:, 1:, :] + uxb_r[:, :-1, :])
    fr = fr.at[:, 1:-1, :].set(sigma_r * (-phi_grad_r + uxb_face_r))

    sigma_theta = _harmonic_mean(sigma, jnp.roll(sigma, -1, axis=2))
    phi_grad_theta = (jnp.roll(phi, -1, axis=2) - phi) / jnp.maximum(
        r_centers[None, :, None] * dtheta, 1.0e-12
    )
    uxb_face_theta = 0.5 * (uxb_theta + jnp.roll(uxb_theta, -1, axis=2))
    ftheta = sigma_theta * (-phi_grad_theta + uxb_face_theta)
    return fx, fr, ftheta


def _pipe_conservative_current_diagnostics_3d(
    sigma: jnp.ndarray,
    phi: jnp.ndarray,
    uxb_x: jnp.ndarray,
    uxb_r: jnp.ndarray,
    uxb_theta: jnp.ndarray,
    *,
    dx: float,
    r_faces: jnp.ndarray,
    r_centers: jnp.ndarray,
    dtheta: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    fx, fr, ftheta = _pipe_conservative_current_fluxes_3d(
        sigma,
        phi,
        uxb_x,
        uxb_r,
        uxb_theta,
        dx=dx,
        r_faces=r_faces,
        r_centers=r_centers,
        dtheta=dtheta,
    )
    dr = jnp.diff(r_faces)
    radial_term = (
        r_faces[None, 1:, None] * fr[:, 1:, :]
        - r_faces[None, :-1, None] * fr[:, :-1, :]
    ) / jnp.maximum(r_centers[None, :, None] * dr[None, :, None], 1.0e-12)
    theta_term = (ftheta - jnp.roll(ftheta, 1, axis=2)) / jnp.maximum(
        r_centers[None, :, None] * dtheta, 1.0e-12
    )
    div_j = (fx[1:] - fx[:-1]) / max(dx, 1.0e-12) + radial_term + theta_term
    wall_area = float(r_faces[-1]) * dx * dtheta
    wall_leakage = jnp.sum(jnp.abs(fr[:, -1, :]) * wall_area, axis=1)
    yz_area = r_centers[:, None] * dr[:, None] * dtheta
    boundary_residual = jnp.abs(
        -jnp.sum(fx[0] * yz_area)
        + jnp.sum(fx[-1] * yz_area)
        + jnp.sum(fr[:, -1, :] * wall_area, axis=1)
    )
    return div_j, wall_leakage, boundary_residual


def _pipe_conservative_emf_rhs_3d(
    sigma: jnp.ndarray,
    uxb_x: jnp.ndarray,
    uxb_r: jnp.ndarray,
    uxb_theta: jnp.ndarray,
    *,
    dx: float,
    r_faces: jnp.ndarray,
    r_centers: jnp.ndarray,
    dtheta: float,
) -> jnp.ndarray:
    zeros = jnp.zeros_like(uxb_x)
    fx, fr, ftheta = _pipe_conservative_current_fluxes_3d(
        sigma,
        zeros,
        uxb_x,
        uxb_r,
        uxb_theta,
        dx=dx,
        r_faces=r_faces,
        r_centers=r_centers,
        dtheta=dtheta,
    )
    dr = jnp.diff(r_faces)
    radial_term = (
        r_faces[None, 1:, None] * fr[:, 1:, :]
        - r_faces[None, :-1, None] * fr[:, :-1, :]
    ) / jnp.maximum(r_centers[None, :, None] * dr[None, :, None], 1.0e-12)
    theta_term = (ftheta - jnp.roll(ftheta, 1, axis=2)) / jnp.maximum(
        r_centers[None, :, None] * dtheta, 1.0e-12
    )
    return (fx[1:] - fx[:-1]) / max(dx, 1.0e-12) + radial_term + theta_term


def _pipe_poisson_sparse_3d(
    rhs: jnp.ndarray,
    conductivity: jnp.ndarray,
    *,
    dx: float,
    r_faces: jnp.ndarray,
    r_centers: jnp.ndarray,
    dtheta: float,
    iterations: int,
    tolerance: float,
    initial_field: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, float, int, float]:
    if sparse is None or sparse_spsolve is None:
        field, residual, iteration_count, initial_residual = _pipe_poisson_jacobi_3d(
            rhs,
            dx=dx,
            dr=float(jnp.mean(jnp.diff(r_faces))),
            dtheta=dtheta,
            r=r_centers[None, :, None],
            iterations=iterations,
            tolerance=tolerance,
        )
        return field, residual, iteration_count, initial_residual

    rhs_np = np.asarray(rhs, dtype=float)
    sigma_np = np.asarray(conductivity, dtype=float)
    r_faces_np = np.asarray(r_faces, dtype=float)
    r_centers_np = np.asarray(r_centers, dtype=float)
    dr_np = np.diff(r_faces_np)
    cell_weights = (r_centers_np[None, :, None] * dr_np[None, :, None]) * dtheta
    rhs_np = rhs_np - np.sum(rhs_np * cell_weights) / max(np.sum(cell_weights), 1.0e-20)

    nx, nr, ntheta = rhs_np.shape
    size = nx * nr * ntheta

    def flat(i: int, j: int, k: int) -> int:
        return (i * nr + j) * ntheta + k

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    rhs_vector = rhs_np.reshape(-1).copy()
    anchor = flat(0, 0, 0)

    for i in range(nx):
        for j in range(nr):
            for k in range(ntheta):
                row = flat(i, j, k)
                if row == anchor:
                    rows.append(row)
                    cols.append(row)
                    data.append(1.0)
                    rhs_vector[row] = 0.0
                    continue

                diagonal = 0.0
                sigma_cell = sigma_np[i, j, k]
                if i > 0:
                    sigma_face = _harmonic_mean(
                        jnp.asarray(sigma_cell), jnp.asarray(sigma_np[i - 1, j, k])
                    ).item()
                    coeff = sigma_face / max(dx**2, 1.0e-12)
                    diagonal += coeff
                    rows.append(row)
                    cols.append(flat(i - 1, j, k))
                    data.append(-coeff)
                if i < nx - 1:
                    sigma_face = _harmonic_mean(
                        jnp.asarray(sigma_cell), jnp.asarray(sigma_np[i + 1, j, k])
                    ).item()
                    coeff = sigma_face / max(dx**2, 1.0e-12)
                    diagonal += coeff
                    rows.append(row)
                    cols.append(flat(i + 1, j, k))
                    data.append(-coeff)
                if j > 0:
                    sigma_face = _harmonic_mean(
                        jnp.asarray(sigma_cell), jnp.asarray(sigma_np[i, j - 1, k])
                    ).item()
                    dr_face = max(0.5 * (dr_np[j - 1] + dr_np[j]), 1.0e-12)
                    coeff = (
                        r_faces_np[j]
                        * sigma_face
                        / max(r_centers_np[j] * dr_np[j] * dr_face, 1.0e-12)
                    )
                    diagonal += coeff
                    rows.append(row)
                    cols.append(flat(i, j - 1, k))
                    data.append(-coeff)
                if j < nr - 1:
                    sigma_face = _harmonic_mean(
                        jnp.asarray(sigma_cell), jnp.asarray(sigma_np[i, j + 1, k])
                    ).item()
                    dr_face = max(0.5 * (dr_np[j] + dr_np[j + 1]), 1.0e-12)
                    coeff = (
                        r_faces_np[j + 1]
                        * sigma_face
                        / max(r_centers_np[j] * dr_np[j] * dr_face, 1.0e-12)
                    )
                    diagonal += coeff
                    rows.append(row)
                    cols.append(flat(i, j + 1, k))
                    data.append(-coeff)

                k_prev = (k - 1) % ntheta
                k_next = (k + 1) % ntheta
                sigma_prev = _harmonic_mean(
                    jnp.asarray(sigma_cell), jnp.asarray(sigma_np[i, j, k_prev])
                ).item()
                sigma_next = _harmonic_mean(
                    jnp.asarray(sigma_cell), jnp.asarray(sigma_np[i, j, k_next])
                ).item()
                theta_coeff_prev = sigma_prev / max(
                    r_centers_np[j] ** 2 * dtheta**2, 1.0e-12
                )
                theta_coeff_next = sigma_next / max(
                    r_centers_np[j] ** 2 * dtheta**2, 1.0e-12
                )
                diagonal += theta_coeff_prev + theta_coeff_next
                rows.append(row)
                cols.append(flat(i, j, k_prev))
                data.append(-theta_coeff_prev)
                rows.append(row)
                cols.append(flat(i, j, k_next))
                data.append(-theta_coeff_next)
                rows.append(row)
                cols.append(row)
                data.append(max(diagonal, 1.0e-12))

    matrix = sparse.csr_matrix((data, (rows, cols)), shape=(size, size))
    if initial_field is not None:
        _ = np.asarray(initial_field, dtype=float)
    initial_residual = float(np.max(np.abs(matrix @ np.zeros(size) - rhs_vector)))
    solution = sparse_spsolve(matrix, rhs_vector)
    field = jnp.asarray(solution.reshape((nx, nr, ntheta)), dtype=rhs.dtype)
    field = field - field[0, 0, 0]
    residual_field = (
        _pipe_conservative_current_diagnostics_3d(
            jnp.asarray(conductivity, dtype=field.dtype),
            field,
            jnp.zeros_like(field),
            jnp.zeros_like(field),
            jnp.zeros_like(field),
            dx=dx,
            r_faces=jnp.asarray(r_faces, dtype=field.dtype),
            r_centers=jnp.asarray(r_centers, dtype=field.dtype),
            dtheta=dtheta,
        )[0]
        - rhs
    )
    residual = float(jnp.max(jnp.abs(residual_field)))
    return field, residual, 1, initial_residual


def _enforce_pipe_velocity_bc(
    u: jnp.ndarray,
    v: jnp.ndarray,
    w: jnp.ndarray,
    *,
    r_centers: jnp.ndarray | None = None,
    r_faces: jnp.ndarray | None = None,
    fluid_mask: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    if u.shape[1] > 1:
        u = u.at[:, 0, :].set(u[:, 1, :])
        w = w.at[:, 0, :].set(w[:, 1, :])
    v = v.at[:, 0, :].set(0.0)
    if fluid_mask is not None:
        active = jnp.asarray(fluid_mask, dtype=bool)
        u = jnp.where(active, u, 0.0)
        v = jnp.where(active, v, 0.0)
        w = jnp.where(active, w, 0.0)
        radial_active = jnp.any(active, axis=(0, 2))
        interface_index = int(jnp.sum(radial_active)) - 1
        if interface_index > 0 and r_centers is not None and r_faces is not None:
            fluid_radius = r_faces[interface_index + 1]
            ratio = (
                0.9
                * (fluid_radius - r_centers[interface_index])
                / jnp.maximum(fluid_radius - r_centers[interface_index - 1], 1.0e-12)
            )
            u = u.at[:, interface_index, :].set(ratio * u[:, interface_index - 1, :])
            w = w.at[:, interface_index, :].set(ratio * w[:, interface_index - 1, :])
            v = v.at[:, interface_index, :].set(0.0)
    elif r_centers is not None and r_faces is not None and u.shape[1] > 1:
        outer_ratio = (
            0.9
            * (r_faces[-1] - r_centers[-1])
            / jnp.maximum(r_faces[-1] - r_centers[-2], 1.0e-12)
        )
        u = u.at[:, -1, :].set(outer_ratio * u[:, -2, :])
        w = w.at[:, -1, :].set(outer_ratio * w[:, -2, :])
    else:
        u = u.at[:, -1, :].set(0.0)
        w = w.at[:, -1, :].set(0.0)
    v = v.at[:, -1, :].set(0.0)
    if u.shape[0] > 1:
        u = u.at[0, :, :].set(u[1, :, :])
        u = u.at[-1, :, :].set(u[-2, :, :])
        v = v.at[0, :, :].set(v[1, :, :])
        v = v.at[-1, :, :].set(v[-2, :, :])
        w = w.at[0, :, :].set(w[1, :, :])
        w = w.at[-1, :, :].set(w[-2, :, :])
    return u, v, w


def smooth_fringing_profile(
    *,
    length: float,
    nx: int,
    entry_center: float,
    exit_center: float,
    transition_width: float,
    peak_scale: float = 1.0,
    axis: str = "z",
) -> FringingProfile:
    if axis not in {"x", "y", "z"}:
        raise ValueError(f"Unsupported magnetic axis {axis!r}")
    x = jnp.linspace(0.0, length, nx)
    width = max(float(transition_width), 1.0e-6)
    rise = 0.5 * (1.0 + jnp.tanh((x - entry_center) / width))
    fall = 0.5 * (1.0 - jnp.tanh((x - exit_center) / width))
    return FringingProfile(x=x, field_scale=peak_scale * rise * fall, axis=axis)


def _constant_field_on_axis(axis: str, magnitude: float) -> tuple[float, float, float]:
    if axis == "x":
        return (magnitude, 0.0, 0.0)
    if axis == "y":
        return (0.0, magnitude, 0.0)
    if axis == "z":
        return (0.0, 0.0, magnitude)
    raise ValueError(f"Unsupported magnetic axis {axis!r}")


def clone_case_with_field(
    case: CaseSpec, *, axis: str, magnitude: float, suffix: str | None = None
) -> CaseSpec:
    magnetic_field = replace(
        case.magnetic_field,
        kind="constant",
        value=_constant_field_on_axis(axis, magnitude),
    )
    name = case.name if suffix is None else f"{case.name}_{suffix}"
    return replace(case, name=name, magnetic_field=magnetic_field)


def build_square_duct_fringing_benchmark(
    *,
    ha_peak: float = 20.0,
    width: float = 2.0,
    height: float = 2.0,
    ny: int = 48,
    nz: int = 48,
    length: float = 6.0,
    nx_stations: int = 21,
    entry_center: float = 1.5,
    exit_center: float = 4.5,
    transition_width: float = 0.35,
) -> tuple[CaseSpec, FringingProfile]:
    base_case = make_shercliff_case(
        ha=ha_peak, width=width, height=height, ny=ny, nz=nz
    )
    base_case = replace(
        base_case,
        geometry=replace(base_case.geometry, length=length, nx=nx_stations),
        time_stepper=replace(
            base_case.time_stepper,
            max_steps=min(base_case.time_stepper.max_steps, 80),
            potential_iterations=min(base_case.time_stepper.potential_iterations, 80),
            steady_tolerance=1.0e-6,
        ),
        solver=replace(
            base_case.solver,
            kind="extruded_inductionless",
            coupling_iterations=min(base_case.solver.coupling_iterations, 8),
            coupling_tolerance=1.0e-7,
        ),
    )
    profile = smooth_fringing_profile(
        length=length,
        nx=nx_stations,
        entry_center=entry_center,
        exit_center=exit_center,
        transition_width=transition_width,
        peak_scale=1.0,
        axis="z",
    )
    return base_case, profile


def build_square_duct_extruded_problem(
    *,
    ha_peak: float = 20.0,
    width: float = 2.0,
    height: float = 2.0,
    ny: int = 48,
    nz: int = 48,
    length: float = 6.0,
    nx_stations: int = 21,
    entry_center: float = 1.5,
    exit_center: float = 4.5,
    transition_width: float = 0.35,
) -> ExtrudedInductionlessProblem:
    case, profile = build_square_duct_fringing_benchmark(
        ha_peak=ha_peak,
        width=width,
        height=height,
        ny=ny,
        nz=nz,
        length=length,
        nx_stations=nx_stations,
        entry_center=entry_center,
        exit_center=exit_center,
        transition_width=transition_width,
    )
    return ExtrudedInductionlessProblem(case=case, profile=profile)


def build_variable_field_duct_extruded_problem(
    *,
    width: float = 2.4,
    height: float = 1.6,
    base_bz: float = 12.0,
    perturbation: float = 0.12,
    ny: int = 40,
    nz: int = 40,
    length: float = 6.0,
    nx_stations: int = 21,
    entry_center: float = 1.5,
    exit_center: float = 4.5,
    transition_width: float = 0.35,
) -> ExtrudedInductionlessProblem:
    from .field_models import make_divergence_free_cross_section_field

    field_fn = make_divergence_free_cross_section_field(
        width=width,
        height=height,
        base_bz=base_bz,
        perturbation=perturbation,
    )
    case = make_shercliff_case(ha=1.0, width=width, height=height, ny=ny, nz=nz)
    case = replace(
        case,
        name=f"variable_field_duct_bz{int(base_bz)}",
        geometry=replace(case.geometry, length=length, nx=nx_stations),
        magnetic_field=MagneticFieldSpec(kind="analytic", fn=field_fn),
        time_stepper=replace(
            case.time_stepper,
            max_steps=min(case.time_stepper.max_steps, 80),
            potential_iterations=min(case.time_stepper.potential_iterations, 80),
            steady_tolerance=1.0e-6,
        ),
        solver=replace(
            case.solver,
            kind="extruded_inductionless",
            coupling_iterations=min(case.solver.coupling_iterations, 8),
            coupling_tolerance=1.0e-7,
        ),
        notes=(
            "Rectangular extruded inductionless solve with analytic divergence-free "
            "cross-sectional magnetic field variation."
        ),
    )
    profile = smooth_fringing_profile(
        length=length,
        nx=nx_stations,
        entry_center=entry_center,
        exit_center=exit_center,
        transition_width=transition_width,
        peak_scale=1.0,
        axis="z",
    )
    return ExtrudedInductionlessProblem(case=case, profile=profile)


def build_variable_field_layered_extruded_problem(
    *,
    width: float = 2.0,
    height: float = 2.0,
    base_bz: float = 12.0,
    perturbation: float = 0.12,
    ny: int = 28,
    nz: int = 28,
    wall_cells: int = 4,
    wall_thickness: float = 0.1,
    insulator_cells: int | None = None,
    insulator_thickness: float | None = None,
    length: float = 6.0,
    nx_stations: int = 21,
    entry_center: float = 1.5,
    exit_center: float = 4.5,
    transition_width: float = 0.35,
) -> ExtrudedInductionlessProblem:
    from .field_models import make_divergence_free_cross_section_field

    field_fn = make_divergence_free_cross_section_field(
        width=width,
        height=height,
        base_bz=base_bz,
        perturbation=perturbation,
    )
    case = make_hunt_case(
        ha=1.0,
        width=width,
        height=height,
        ny=ny,
        nz=nz,
        wall_cells=wall_cells,
        wall_thickness=wall_thickness,
        insulator_cells=insulator_cells,
        insulator_thickness=insulator_thickness,
    )
    case = replace(
        case,
        name=f"variable_field_layered_bz{int(base_bz)}",
        geometry=replace(case.geometry, length=length, nx=nx_stations),
        magnetic_field=MagneticFieldSpec(kind="analytic", fn=field_fn),
        time_stepper=replace(
            case.time_stepper,
            max_steps=min(case.time_stepper.max_steps, 80),
            potential_iterations=min(case.time_stepper.potential_iterations, 80),
            steady_tolerance=1.0e-6,
        ),
        solver=replace(
            case.solver,
            kind="extruded_inductionless",
            coupling_iterations=min(case.solver.coupling_iterations, 8),
            coupling_tolerance=1.0e-7,
        ),
        notes=(
            "Layered extruded inductionless solve with analytic divergence-free "
            "cross-sectional magnetic field variation."
        ),
    )
    profile = smooth_fringing_profile(
        length=length,
        nx=nx_stations,
        entry_center=entry_center,
        exit_center=exit_center,
        transition_width=transition_width,
        peak_scale=1.0,
        axis="z",
    )
    return ExtrudedInductionlessProblem(case=case, profile=profile)


def build_variable_field_pipe_ogrid_extruded_problem(
    *,
    radius: float = 0.5,
    base_bz: float = 12.0,
    core_fraction_y: float = 0.5,
    core_fraction_z: float = 0.5,
    nr: int = 18,
    ntheta: int = 40,
    length: float = 6.0,
    nx_stations: int = 21,
    entry_center: float = 1.5,
    exit_center: float = 4.5,
    transition_width: float = 0.35,
) -> ExtrudedInductionlessProblem:
    from .field_models import make_localized_divergence_free_obstacle_field

    field_fn = make_localized_divergence_free_obstacle_field(
        width=2.0 * radius,
        height=2.0 * radius,
        base_bz=base_bz,
        core_fraction_y=core_fraction_y,
        core_fraction_z=core_fraction_z,
    )
    case = build_pipe_ogrid_extruded_problem(
        ha_peak=1.0,
        radius=radius,
        nr=nr,
        ntheta=ntheta,
        length=length,
        nx_stations=nx_stations,
        entry_center=entry_center,
        exit_center=exit_center,
        transition_width=transition_width,
    ).case
    case = replace(
        case,
        name=f"variable_field_pipe_bz{int(base_bz)}",
        magnetic_field=MagneticFieldSpec(kind="analytic", fn=field_fn),
    )
    profile = smooth_fringing_profile(
        length=length,
        nx=nx_stations,
        entry_center=entry_center,
        exit_center=exit_center,
        transition_width=transition_width,
        peak_scale=1.0,
        axis="z",
    )
    return ExtrudedInductionlessProblem(case=case, profile=profile)


def build_magnetic_obstacle_rect_extruded_problem(
    *,
    width: float = 2.0,
    height: float = 2.0,
    base_bz: float = 12.0,
    core_fraction_y: float = 0.35,
    core_fraction_z: float = 0.35,
    ny: int = 36,
    nz: int = 36,
    length: float = 6.0,
    nx_stations: int = 21,
    entry_center: float = 2.2,
    exit_center: float = 3.8,
    transition_width: float = 0.22,
    forcing: float = 1.0,
) -> ExtrudedInductionlessProblem:
    from .field_models import make_localized_divergence_free_obstacle_field

    field_fn = make_localized_divergence_free_obstacle_field(
        width=width,
        height=height,
        base_bz=base_bz,
        core_fraction_y=core_fraction_y,
        core_fraction_z=core_fraction_z,
    )
    case = make_shercliff_case(ha=1.0, width=width, height=height, ny=ny, nz=nz)
    case = replace(
        case,
        name=f"magnetic_obstacle_rect_bz{int(base_bz)}",
        geometry=replace(case.geometry, length=length, nx=nx_stations),
        magnetic_field=MagneticFieldSpec(kind="analytic", fn=field_fn),
        time_stepper=replace(
            case.time_stepper,
            max_steps=min(case.time_stepper.max_steps, 80),
            potential_iterations=min(case.time_stepper.potential_iterations, 80),
            steady_tolerance=1.0e-6,
        ),
        solver=replace(
            case.solver,
            kind="extruded_inductionless",
            coupling_iterations=min(case.solver.coupling_iterations, 8),
            coupling_tolerance=1.0e-7,
        ),
        forcing=forcing,
        notes=(
            "Localized-field magnetic-obstacle baseline on the rectangular "
            "extruded inductionless solver lane."
        ),
    )
    profile = smooth_fringing_profile(
        length=length,
        nx=nx_stations,
        entry_center=entry_center,
        exit_center=exit_center,
        transition_width=transition_width,
        peak_scale=1.0,
        axis="z",
    )
    return ExtrudedInductionlessProblem(case=case, profile=profile)


def build_wham_mirror_pipe_extruded_problem(
    *,
    table_path: str,
    radius: float = 0.25,
    nr: int = 18,
    ntheta: int = 48,
    length: float = 1.4,
    nx_stations: int = 25,
    conductivity: float = 1.0,
    density: float = 1.0,
    viscosity: float = 1.0,
    forcing: float = 1.0,
) -> ExtrudedInductionlessProblem:
    problem = build_pipe_ogrid_extruded_problem(
        ha_peak=1.0,
        radius=radius,
        nr=nr,
        ntheta=ntheta,
        length=length,
        nx_stations=nx_stations,
        entry_center=0.3 * length,
        exit_center=0.7 * length,
        transition_width=0.08 * length,
        conductivity=conductivity,
        density=density,
        viscosity=viscosity,
    )
    return ExtrudedInductionlessProblem(
        case=replace(
            problem.case,
            name="wham_mirror_pipe",
            magnetic_field=MagneticFieldSpec(
                kind="tabulated", table_path=str(table_path)
            ),
            forcing=forcing,
            notes=(
                "Pipe crossing a tabulated WHAM-like mirror field. "
                "This is the current stronger Benchmark D inductionless baseline."
            ),
        ),
        profile=FringingProfile(
            x=problem.profile.x,
            field_scale=jnp.ones_like(problem.profile.field_scale),
            axis="z",
        ),
    )


def build_layered_duct_extruded_problem(
    *,
    ha_peak: float = 20.0,
    width: float = 2.0,
    height: float = 2.0,
    ny: int = 32,
    nz: int = 32,
    wall_cells: int = 4,
    wall_thickness: float = 0.1,
    insulator_cells: int | None = None,
    insulator_thickness: float | None = None,
    length: float = 6.0,
    nx_stations: int = 21,
    entry_center: float = 1.5,
    exit_center: float = 4.5,
    transition_width: float = 0.35,
) -> ExtrudedInductionlessProblem:
    from .cases import make_hunt_case

    case = make_hunt_case(
        ha=ha_peak,
        width=width,
        height=height,
        ny=ny,
        nz=nz,
        wall_cells=wall_cells,
        wall_thickness=wall_thickness,
        insulator_cells=insulator_cells,
        insulator_thickness=insulator_thickness,
    )
    case = replace(
        case,
        geometry=replace(case.geometry, length=length, nx=nx_stations),
        time_stepper=replace(
            case.time_stepper,
            max_steps=min(case.time_stepper.max_steps, 80),
            potential_iterations=min(case.time_stepper.potential_iterations, 80),
            steady_tolerance=1.0e-6,
        ),
        solver=replace(
            case.solver,
            kind="extruded_inductionless",
            coupling_iterations=min(case.solver.coupling_iterations, 8),
            coupling_tolerance=1.0e-7,
        ),
    )
    profile = smooth_fringing_profile(
        length=length,
        nx=nx_stations,
        entry_center=entry_center,
        exit_center=exit_center,
        transition_width=transition_width,
        peak_scale=1.0,
        axis="z",
    )
    return ExtrudedInductionlessProblem(case=case, profile=profile)


def build_pipe_ogrid_extruded_problem(
    *,
    ha_peak: float = 20.0,
    radius: float = 1.0,
    nr: int = 24,
    ntheta: int = 64,
    length: float = 6.0,
    nx_stations: int = 21,
    entry_center: float = 1.5,
    exit_center: float = 4.5,
    transition_width: float = 0.35,
    conductivity: float = 1.0,
    density: float = 1.0,
    viscosity: float = 1.0,
) -> ExtrudedInductionlessProblem:
    bmag = _ha_to_b(ha_peak, radius, conductivity, density, viscosity)
    case = CaseSpec(
        name=f"pipe_fringing_ha{int(ha_peak)}",
        geometry=GeometrySpec(
            kind="pipe_ogrid",
            width=2.0 * radius,
            height=2.0 * radius,
            radius=radius,
            length=length,
            nx=nx_stations,
            nr=nr,
            ntheta=ntheta,
        ),
        regions=(RegionSpec("fluid", "fluid", conductivity, density, viscosity),),
        magnetic_field=MagneticFieldSpec(kind="constant", value=(0.0, 0.0, bmag)),
        boundary_conditions=(
            BoundaryCondition("wall", "no_slip"),
            BoundaryCondition("electric", "insulating"),
        ),
        time_stepper=TimeStepperConfig(
            dt=0.001,
            t_final=1.0,
            max_steps=80,
            potential_iterations=80,
            steady_tolerance=1.0e-6,
        ),
        solver=SolverConfig(
            kind="extruded_inductionless",
            mode="steady",
            linear_solver="auto",
            preconditioner="jacobi",
            time_scheme="implicit_euler",
            coupling_iterations=8,
            coupling_tolerance=1.0e-7,
        ),
        output=OutputSpec(),
        forcing=1.0,
        reference_pressure_gradient=-1.0,
        reference_phi_cell=(max(1, nr // 4), max(1, ntheta // 8)),
        notes="Mapped-pipe fringing research slice with cylindrical metric terms.",
    )
    profile = smooth_fringing_profile(
        length=length,
        nx=nx_stations,
        entry_center=entry_center,
        exit_center=exit_center,
        transition_width=transition_width,
        peak_scale=1.0,
        axis="z",
    )
    return ExtrudedInductionlessProblem(case=case, profile=profile)


def build_bent_pipe_extruded_problem(
    *,
    ha_peak: float = 20.0,
    radius: float = 1.0,
    bend_radius: float = 6.0,
    bend_angle: float = 0.75 * jnp.pi,
    nr: int = 24,
    ntheta: int = 64,
    nx_stations: int = 25,
    entry_center_fraction: float = 0.25,
    exit_center_fraction: float = 0.75,
    transition_width_fraction: float = 0.08,
    conductivity: float = 1.0,
    density: float = 1.0,
    viscosity: float = 1.0,
) -> ExtrudedInductionlessProblem:
    arc_length = float(bend_radius * bend_angle)
    bmag = _ha_to_b(ha_peak, radius, conductivity, density, viscosity)
    case = CaseSpec(
        name=f"bent_pipe_fringing_ha{int(ha_peak)}",
        geometry=GeometrySpec(
            kind="bent_pipe",
            width=2.0 * radius,
            height=2.0 * radius,
            radius=radius,
            bend_radius=bend_radius,
            bend_angle=bend_angle,
            length=arc_length,
            nx=nx_stations,
            nr=nr,
            ntheta=ntheta,
        ),
        regions=(RegionSpec("fluid", "fluid", conductivity, density, viscosity),),
        magnetic_field=MagneticFieldSpec(kind="constant", value=(0.0, 0.0, bmag)),
        boundary_conditions=(
            BoundaryCondition("wall", "no_slip"),
            BoundaryCondition("electric", "insulating"),
        ),
        time_stepper=TimeStepperConfig(
            dt=0.001,
            t_final=1.0,
            max_steps=80,
            potential_iterations=80,
            steady_tolerance=1.0e-6,
        ),
        solver=SolverConfig(
            kind="extruded_inductionless",
            mode="steady",
            linear_solver="auto",
            preconditioner="jacobi",
            time_scheme="implicit_euler",
            coupling_iterations=8,
            coupling_tolerance=1.0e-7,
        ),
        output=OutputSpec(),
        forcing=1.0,
        reference_pressure_gradient=-1.0,
        reference_phi_cell=(max(1, nr // 4), max(1, ntheta // 8)),
        notes=(
            "Curved-centerline inductionless baseline for low-De bent-pipe MHD. "
            "Secondary Dean vortices are not modeled in this lane."
        ),
    )
    profile = smooth_fringing_profile(
        length=arc_length,
        nx=nx_stations,
        entry_center=entry_center_fraction * arc_length,
        exit_center=exit_center_fraction * arc_length,
        transition_width=transition_width_fraction * arc_length,
        peak_scale=1.0,
        axis="z",
    )
    return ExtrudedInductionlessProblem(case=case, profile=profile)


def build_variable_field_bent_pipe_extruded_problem(
    *,
    radius: float = 0.45,
    bend_radius: float = 3.6,
    bend_angle: float = 1.15,
    base_bz: float = 12.0,
    core_fraction_y: float = 0.5,
    core_fraction_z: float = 0.5,
    nr: int = 18,
    ntheta: int = 40,
    nx_stations: int = 15,
    entry_center_fraction: float = 0.25,
    exit_center_fraction: float = 0.75,
    transition_width_fraction: float = 0.08,
) -> ExtrudedInductionlessProblem:
    from .field_models import make_localized_divergence_free_obstacle_field

    field_fn = make_localized_divergence_free_obstacle_field(
        width=2.0 * radius,
        height=2.0 * radius,
        base_bz=base_bz,
        core_fraction_y=core_fraction_y,
        core_fraction_z=core_fraction_z,
    )
    problem = build_bent_pipe_extruded_problem(
        ha_peak=1.0,
        radius=radius,
        bend_radius=bend_radius,
        bend_angle=bend_angle,
        nr=nr,
        ntheta=ntheta,
        nx_stations=nx_stations,
        entry_center_fraction=entry_center_fraction,
        exit_center_fraction=exit_center_fraction,
        transition_width_fraction=transition_width_fraction,
    )
    return ExtrudedInductionlessProblem(
        case=replace(
            problem.case,
            name=f"variable_field_bent_pipe_bz{int(base_bz)}",
            magnetic_field=MagneticFieldSpec(kind="analytic", fn=field_fn),
        ),
        profile=problem.profile,
    )


def _signed_pipe_cut(
    values: jnp.ndarray, r: jnp.ndarray, *, theta_index: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    ntheta = int(values.shape[1])
    opposite = (theta_index + ntheta // 2) % ntheta
    negative = values[:, opposite][::-1]
    positive = values[1:, theta_index]
    positions = jnp.concatenate([-r[::-1], r[1:]])
    cut = jnp.concatenate([negative, positive])
    return positions, cut


def validate_bent_pipe_low_de_baseline(
    bent_solution: ExtrudedInductionlessSolution,
    straight_solution: ExtrudedInductionlessSolution,
) -> dict[str, float | bool]:
    bent_geometry = bent_solution.problem.case.geometry
    if bent_geometry.kind != "bent_pipe":
        raise ValueError("Bent-pipe validation requires a bent_pipe solution")
    if straight_solution.problem.case.geometry.kind != "pipe_ogrid":
        raise ValueError(
            "Bent-pipe validation requires a straight pipe_ogrid comparison solution"
        )
    if bent_solution.bundle.u.shape != straight_solution.bundle.u.shape:
        raise ValueError(
            "Bent and straight comparison bundles must share the same shape"
        )

    bent_bundle = bent_solution.bundle
    straight_bundle = straight_solution.bundle
    mid_index = int(bent_bundle.u.shape[0] // 2)
    bent_mid = bent_bundle.u[mid_index]
    straight_mid = straight_bundle.u[mid_index]
    bent_secondary_mid = jnp.sqrt(
        bent_bundle.v[mid_index] ** 2 + bent_bundle.w[mid_index] ** 2
    )
    reference_norm = jnp.maximum(jnp.linalg.norm(straight_mid), 1.0e-12)
    cross_section_l2_error = float(
        jnp.linalg.norm(bent_mid - straight_mid) / reference_norm
    )

    r = jnp.asarray(bent_bundle.y, dtype=float)
    signed_r, bent_cut = _signed_pipe_cut(bent_mid, r, theta_index=0)
    _, straight_cut = _signed_pipe_cut(straight_mid, r, theta_index=0)
    cut_norm = jnp.maximum(jnp.linalg.norm(straight_cut), 1.0e-12)
    centerline_l2_error = float(jnp.linalg.norm(bent_cut - straight_cut) / cut_norm)
    axial_scale = float(jnp.maximum(jnp.mean(jnp.abs(bent_mid)), 1.0e-12))
    peak_axial_scale = float(jnp.maximum(jnp.max(jnp.abs(bent_mid)), 1.0e-12))
    secondary_flow_rms_ratio = float(
        jnp.sqrt(jnp.mean(bent_secondary_mid**2)) / axial_scale
    )
    secondary_flow_peak_ratio = float(jnp.max(bent_secondary_mid) / peak_axial_scale)
    bent_cut_abs = jnp.abs(bent_cut)
    straight_cut_abs = jnp.abs(straight_cut)
    bent_weight = jnp.maximum(jnp.sum(bent_cut_abs), 1.0e-12)
    straight_weight = jnp.maximum(jnp.sum(straight_cut_abs), 1.0e-12)
    bent_velocity_centroid = float(jnp.sum(signed_r * bent_cut_abs) / bent_weight)
    straight_velocity_centroid = float(
        jnp.sum(signed_r * straight_cut_abs) / straight_weight
    )
    velocity_centroid_shift = bent_velocity_centroid - straight_velocity_centroid
    radius_scale = float(jnp.maximum(jnp.max(jnp.abs(signed_r)), 1.0e-12))
    normalized_velocity_centroid_shift = float(velocity_centroid_shift / radius_scale)
    inner_outer_velocity_ratio = float(
        jnp.max(jnp.abs(bent_cut[signed_r >= 0.0]))
        / jnp.maximum(jnp.max(jnp.abs(bent_cut[signed_r <= 0.0])), 1.0e-12)
    )

    region = bent_solution.problem.case.regions[0]
    mean_velocity = float(jnp.mean(jnp.abs(bent_bundle.mean_velocity)))
    diameter = 2.0 * float(bent_geometry.radius or 0.5 * bent_geometry.width)
    reynolds_number = float(
        (region.density or 1.0)
        * mean_velocity
        * diameter
        / max(region.viscosity or 1.0, 1.0e-12)
    )
    curvature_ratio = float(
        (bent_geometry.radius or 0.5 * bent_geometry.width)
        / max(bent_geometry.bend_radius or 1.0, 1.0e-12)
    )
    dean_number = float(reynolds_number * np.sqrt(max(curvature_ratio, 0.0)))
    throughput_span = float(bent_solution.validation.volumetric_flow_rate_span)
    max_charge_balance_residual = float(
        bent_solution.validation.max_charge_balance_residual
    )
    max_wall_current_leakage = float(bent_solution.validation.max_wall_current_leakage)
    net_boundary_current_residual = float(
        bent_solution.validation.net_boundary_current_residual
    )
    bounded_charge_balance_tolerance = 5.0e-2
    research_grade_charge_balance_tolerance = 1.0e-3
    research_grade_charge_balance_pass = bool(
        max_charge_balance_residual <= research_grade_charge_balance_tolerance
    )
    validation_pass = bool(
        dean_number <= 10.0
        and cross_section_l2_error <= 0.08
        and centerline_l2_error <= 0.08
        and throughput_span <= 1.0e-3
        and max_charge_balance_residual <= bounded_charge_balance_tolerance
        and max_wall_current_leakage <= 1.0e-8
        and net_boundary_current_residual <= 1.0e-8
    )
    return {
        "curvature_ratio": curvature_ratio,
        "reynolds_number": reynolds_number,
        "dean_number": dean_number,
        "dean_vortex_observables_available": True,
        "secondary_flow_rms_ratio": secondary_flow_rms_ratio,
        "secondary_flow_peak_ratio": secondary_flow_peak_ratio,
        "bent_velocity_centroid": bent_velocity_centroid,
        "straight_velocity_centroid": straight_velocity_centroid,
        "velocity_centroid_shift": velocity_centroid_shift,
        "normalized_velocity_centroid_shift": normalized_velocity_centroid_shift,
        "inner_outer_velocity_ratio": inner_outer_velocity_ratio,
        "cross_section_l2_error": cross_section_l2_error,
        "centerline_l2_error": centerline_l2_error,
        "throughput_span": throughput_span,
        "max_charge_balance_residual": max_charge_balance_residual,
        "bounded_charge_balance_tolerance": bounded_charge_balance_tolerance,
        "research_grade_charge_balance_tolerance": research_grade_charge_balance_tolerance,
        "research_grade_charge_balance_pass": research_grade_charge_balance_pass,
        "max_wall_current_leakage": max_wall_current_leakage,
        "net_boundary_current_residual": net_boundary_current_residual,
        "validation_pass": validation_pass,
        "signed_radius": np.asarray(signed_r, dtype=float).tolist(),
        "bent_centerline_cut": np.asarray(bent_cut, dtype=float).tolist(),
        "straight_centerline_cut": np.asarray(straight_cut, dtype=float).tolist(),
        "literature_target": "Dean curved-pipe secondary-flow and curvature-response observables",
        "validation_status": "low_de_straight_pipe_limit_passes_full_dean_vortex_reference_open",
        "research_grade_dean_validation_pass": False,
    }


def validate_variable_field_extruded_solution(
    solution: ExtrudedInductionlessSolution,
    *,
    field_ny: int = 81,
    field_nz: int = 81,
) -> dict[str, float | bool]:
    if solution.problem.case.geometry.kind not in {"rect_duct", "layered_duct"}:
        raise ValueError(
            "Variable-field extruded validation currently supports rectangular and layered ducts only"
        )
    field_metrics = _variable_field_metrics(
        solution, field_ny=field_ny, field_nz=field_nz
    )
    validation = solution.validation
    field_scale = np.asarray(solution.bundle.field_scale, dtype=float)
    mean_velocity = np.asarray(solution.bundle.mean_velocity, dtype=float)
    current_proxy = np.asarray(
        solution.bundle.current_scaled_pressure_proxy, dtype=float
    )
    field_velocity_correlation = float(
        _safe_correlation(jnp.asarray(field_scale), jnp.asarray(mean_velocity))
    )
    velocity_change = (
        float(np.max(mean_velocity) - np.min(mean_velocity))
        if mean_velocity.size
        else 0.0
    )
    current_proxy_change = (
        float(np.max(current_proxy) - np.min(current_proxy))
        if current_proxy.size
        else 0.0
    )
    charge_limit = (
        5.0e-2 if solution.problem.case.geometry.kind == "rect_duct" else 2.0e-1
    )
    validation_pass = bool(
        field_metrics["rms_divergence"] <= 5.0e-2
        and validation.max_charge_balance_residual <= charge_limit
        and validation.net_boundary_current_residual <= 1.0e-8
        and validation.max_wall_current_leakage <= 1.0e-8
        and abs(field_velocity_correlation) >= 0.2
        and velocity_change > 1.0e-8
        and current_proxy_change > 1.0e-8
    )
    return {
        **field_metrics,
        "field_velocity_correlation": field_velocity_correlation,
        "mean_velocity_change": velocity_change,
        "current_proxy_change": current_proxy_change,
        "max_charge_balance_residual": float(validation.max_charge_balance_residual),
        "max_wall_current_leakage": float(validation.max_wall_current_leakage),
        "net_boundary_current_residual": float(
            validation.net_boundary_current_residual
        ),
        "validation_pass": validation_pass,
    }


def validate_variable_field_pipe_solution(
    solution: ExtrudedInductionlessSolution,
    *,
    field_ny: int = 81,
    field_nz: int = 81,
) -> dict[str, float | bool]:
    if solution.problem.case.geometry.kind not in {"pipe_ogrid", "bent_pipe"}:
        raise ValueError(
            "Variable-field pipe validation currently supports pipe_ogrid and bent_pipe only"
        )
    field_metrics = _variable_field_metrics(
        solution, field_ny=field_ny, field_nz=field_nz
    )
    validation = solution.validation
    mean_velocity = np.asarray(solution.bundle.mean_velocity, dtype=float)
    current_proxy = np.asarray(
        solution.bundle.current_scaled_pressure_proxy, dtype=float
    )
    velocity_change = (
        float(np.max(mean_velocity) - np.min(mean_velocity))
        if mean_velocity.size
        else 0.0
    )
    current_proxy_change = (
        float(np.max(current_proxy) - np.min(current_proxy))
        if current_proxy.size
        else 0.0
    )
    divergence_ratio = float(
        field_metrics["rms_divergence"]
        / max(field_metrics["mean_field_magnitude"], 1.0e-12)
    )
    validation_pass = bool(
        divergence_ratio <= 8.0e-2
        and validation.max_charge_balance_residual <= 6.0e-2
        and validation.net_boundary_current_residual <= 1.0e-8
        and validation.max_wall_current_leakage <= 1.0e-8
        and current_proxy_change > 1.0e-6
    )
    return {
        **field_metrics,
        "divergence_to_field_ratio": divergence_ratio,
        "mean_velocity_change": velocity_change,
        "current_proxy_change": current_proxy_change,
        "max_charge_balance_residual": float(validation.max_charge_balance_residual),
        "max_wall_current_leakage": float(validation.max_wall_current_leakage),
        "net_boundary_current_residual": float(
            validation.net_boundary_current_residual
        ),
        "validation_pass": validation_pass,
    }


def _variable_field_metrics(
    solution: ExtrudedInductionlessSolution,
    *,
    field_ny: int = 81,
    field_nz: int = 81,
) -> dict[str, float]:
    field_kind = solution.problem.case.magnetic_field.kind
    geometry = solution.problem.case.geometry
    if field_kind == "analytic" and solution.problem.case.magnetic_field.fn is not None:
        from .field_models import cross_section_divergence_metrics

        return cross_section_divergence_metrics(
            solution.problem.case.magnetic_field.fn,
            width=geometry.width,
            height=geometry.height,
            ny=field_ny,
            nz=field_nz,
        )
    if field_kind == "tabulated":
        # The bundle does not store B explicitly, so resample the tabulated field at the magnet mid-station.
        mesh = _cross_section_mesh(solution.problem.case)
        x_mid = np.full((mesh.ny, mesh.nz), 0.5 * geometry.length, dtype=float)
        y_mid, z_mid = np.meshgrid(
            np.asarray(mesh.y_centers, dtype=float),
            np.asarray(mesh.z_centers, dtype=float),
            indexing="ij",
        )
        sampled = sample_tabulated_field_volume(
            solution.problem.case.magnetic_field.table_path,
            x=x_mid,
            y=y_mid,
            z=z_mid,
        )
        by = np.asarray(sampled[..., 1], dtype=float)
        bz = np.asarray(sampled[..., 2], dtype=float)
        dy = float(mesh.y_centers[1] - mesh.y_centers[0]) if mesh.ny > 1 else 1.0
        dz = float(mesh.z_centers[1] - mesh.z_centers[0]) if mesh.nz > 1 else 1.0
        div = np.gradient(by, dy, axis=0) + np.gradient(bz, dz, axis=1)
        magnitude = np.sqrt(by**2 + bz**2)
        return {
            "max_abs_divergence": float(np.max(np.abs(div))),
            "rms_divergence": float(np.sqrt(np.mean(div**2))),
            "mean_field_magnitude": float(np.mean(magnitude)),
        }
    raise ValueError(
        "Variable-field validation currently supports analytic and tabulated magnetic fields"
    )


def validate_magnetic_obstacle_baseline(
    solution: ExtrudedInductionlessSolution,
    *,
    field_ny: int = 81,
    field_nz: int = 81,
) -> dict[str, float | bool | str]:
    if solution.problem.case.geometry.kind != "rect_duct":
        raise ValueError(
            "Magnetic-obstacle baseline currently supports rectangular ducts only"
        )
    if (
        solution.problem.case.magnetic_field.kind != "analytic"
        or solution.problem.case.magnetic_field.fn is None
    ):
        raise ValueError(
            "Magnetic-obstacle baseline requires an analytic magnetic field"
        )

    from .field_models import cross_section_divergence_metrics

    geometry = solution.problem.case.geometry
    field_metrics = cross_section_divergence_metrics(
        solution.problem.case.magnetic_field.fn,
        width=geometry.width,
        height=geometry.height,
        ny=field_ny,
        nz=field_nz,
    )
    bundle = solution.bundle
    validation = solution.validation
    field_scale = np.asarray(bundle.field_scale, dtype=float)
    mean_velocity = np.asarray(bundle.mean_velocity, dtype=float)
    current_proxy = np.asarray(bundle.current_scaled_pressure_proxy, dtype=float)
    peak_index = int(np.argmax(field_scale)) if field_scale.size else 0
    inlet_reference = float(mean_velocity[0]) if mean_velocity.size else 0.0
    obstacle_velocity_deficit = (
        float(inlet_reference - mean_velocity[peak_index])
        if mean_velocity.size
        else 0.0
    )
    current_proxy_peak = float(np.max(current_proxy)) if current_proxy.size else 0.0
    field_velocity_correlation = float(
        _safe_correlation(jnp.asarray(field_scale), jnp.asarray(mean_velocity))
    )
    divergence_to_field_ratio = float(
        field_metrics["rms_divergence"]
        / max(field_metrics["mean_field_magnitude"], 1.0e-12)
    )
    field_quality_pass = bool(divergence_to_field_ratio <= 2.5e-2)
    conservation_pass = bool(
        validation.max_charge_balance_residual <= 5.0e-2
        and validation.net_boundary_current_residual <= 1.0e-8
        and validation.max_wall_current_leakage <= 1.0e-8
    )
    response_observable_pass = bool(
        obstacle_velocity_deficit > 1.0e-8
        and current_proxy_peak > 1.0e-8
        and field_velocity_correlation < -0.2
    )
    validation_pass = bool(
        field_quality_pass and conservation_pass and response_observable_pass
    )
    return {
        **field_metrics,
        "divergence_to_field_ratio": divergence_to_field_ratio,
        "obstacle_velocity_deficit": obstacle_velocity_deficit,
        "current_proxy_peak": current_proxy_peak,
        "field_velocity_correlation": field_velocity_correlation,
        "max_charge_balance_residual": float(validation.max_charge_balance_residual),
        "max_wall_current_leakage": float(validation.max_wall_current_leakage),
        "net_boundary_current_residual": float(
            validation.net_boundary_current_residual
        ),
        "field_quality_pass": field_quality_pass,
        "conservation_pass": conservation_pass,
        "response_observable_pass": response_observable_pass,
        "reference_kind": "none",
        "external_reference_available": False,
        "research_grade_validation_pass": False,
        "validation_pass": validation_pass,
    }


def validate_magnetic_obstacle_benchmark(
    solution: ExtrudedInductionlessSolution,
    reference_solution: ExtrudedInductionlessSolution,
    *,
    field_ny: int = 81,
    field_nz: int = 81,
) -> dict[str, float | bool | str]:
    if solution.problem.case.geometry.kind != "rect_duct":
        raise ValueError(
            "Magnetic-obstacle benchmark currently supports rectangular ducts only"
        )
    if reference_solution.problem.case.geometry.kind != "rect_duct":
        raise ValueError(
            "Magnetic-obstacle benchmark reference must be a rectangular duct"
        )
    if solution.bundle.u.shape != reference_solution.bundle.u.shape:
        raise ValueError(
            "Benchmark and reference solutions must share the same stacked field shape"
        )

    baseline = validate_magnetic_obstacle_baseline(
        solution, field_ny=field_ny, field_nz=field_nz
    )
    bundle = solution.bundle
    reference_bundle = reference_solution.bundle
    divergence_ratio = float(baseline["divergence_to_field_ratio"])
    field_scale = np.asarray(bundle.field_scale, dtype=float)
    mean_velocity = np.asarray(bundle.mean_velocity, dtype=float)
    ref_mean_velocity = np.asarray(reference_bundle.mean_velocity, dtype=float)
    current_proxy = np.asarray(bundle.current_scaled_pressure_proxy, dtype=float)
    pressure_span = np.max(np.asarray(bundle.p, dtype=float), axis=(1, 2)) - np.min(
        np.asarray(bundle.p, dtype=float), axis=(1, 2)
    )
    reference_pressure_span = np.max(
        np.asarray(reference_bundle.p, dtype=float), axis=(1, 2)
    ) - np.min(np.asarray(reference_bundle.p, dtype=float), axis=(1, 2))
    peak_index = int(np.argmax(field_scale)) if field_scale.size else 0

    denom = np.maximum(np.abs(ref_mean_velocity), 1.0e-12)
    velocity_deficit_ratio = np.maximum(
        (ref_mean_velocity - mean_velocity) / denom, 0.0
    )
    peak_velocity_deficit_ratio = (
        float(np.max(velocity_deficit_ratio)) if velocity_deficit_ratio.size else 0.0
    )
    peak_station_velocity_deficit_ratio = (
        float(velocity_deficit_ratio[peak_index])
        if velocity_deficit_ratio.size
        else 0.0
    )
    wake_recovery_ratio = (
        float(mean_velocity[-1] / max(mean_velocity[0], 1.0e-12))
        if mean_velocity.size
        else 0.0
    )

    pressure_excess = np.maximum(pressure_span - reference_pressure_span, 0.0)
    pressure_excess_proxy = (
        float(
            np.trapezoid(pressure_excess, np.asarray(bundle.x, dtype=float))
            / max(float(bundle.x[-1] - bundle.x[0]), 1.0e-12)
        )
        if pressure_excess.size > 1
        else 0.0
    )
    peak_pressure_excess = (
        float(np.max(pressure_excess)) if pressure_excess.size else 0.0
    )
    current_proxy_peak = (
        float(np.max(np.abs(current_proxy))) if current_proxy.size else 0.0
    )
    integrated_velocity_deficit_ratio = (
        float(
            np.trapezoid(velocity_deficit_ratio, np.asarray(bundle.x, dtype=float))
            / max(float(bundle.x[-1] - bundle.x[0]), 1.0e-12)
        )
        if velocity_deficit_ratio.size > 1
        else 0.0
    )

    mid_y = int(bundle.u.shape[1] // 2)
    mid_z = int(bundle.u.shape[2] // 2)
    y_cut = np.asarray(bundle.u[peak_index, :, mid_z], dtype=float)
    y_cut_ref = np.asarray(reference_bundle.u[peak_index, :, mid_z], dtype=float)
    z_cut = np.asarray(bundle.u[peak_index, mid_y, :], dtype=float)
    z_cut_ref = np.asarray(reference_bundle.u[peak_index, mid_y, :], dtype=float)
    y_l2_distortion = float(
        np.linalg.norm(y_cut - y_cut_ref) / max(np.linalg.norm(y_cut_ref), 1.0e-12)
    )
    z_l2_distortion = float(
        np.linalg.norm(z_cut - z_cut_ref) / max(np.linalg.norm(z_cut_ref), 1.0e-12)
    )
    shared_cut_scale = max(
        float(np.max(np.abs(y_cut_ref))),
        float(np.max(np.abs(z_cut_ref))),
        float(np.max(np.abs(y_cut))),
        float(np.max(np.abs(z_cut))),
        1.0e-12,
    )
    y_peak_cut_abs_error = float(np.max(np.abs(y_cut - y_cut_ref)) / shared_cut_scale)
    z_peak_cut_abs_error = float(np.max(np.abs(z_cut - z_cut_ref)) / shared_cut_scale)
    peak_crosscut_distortion = max(y_l2_distortion, z_l2_distortion)

    center_velocity = np.asarray(bundle.u[:, mid_y, mid_z], dtype=float)
    ref_center_velocity = np.asarray(reference_bundle.u[:, mid_y, mid_z], dtype=float)
    center_velocity_deficit_ratio = np.maximum(
        (ref_center_velocity - center_velocity)
        / np.maximum(np.abs(ref_center_velocity), 1.0e-12),
        0.0,
    )
    peak_centerline_deficit_ratio = (
        float(np.max(center_velocity_deficit_ratio))
        if center_velocity_deficit_ratio.size
        else 0.0
    )
    peak_centerline_station_deficit_ratio = (
        float(center_velocity_deficit_ratio[peak_index])
        if center_velocity_deficit_ratio.size
        else 0.0
    )
    recovery_station = float(bundle.x[-1]) if len(bundle.x) else 0.0
    if center_velocity_deficit_ratio.size:
        threshold = max(0.1 * peak_centerline_deficit_ratio, 1.0e-6)
        tail = np.where(center_velocity_deficit_ratio[peak_index:] <= threshold)[0]
        if tail.size:
            recovery_station = float(bundle.x[peak_index + int(tail[0])])

    validation_pass = bool(
        divergence_ratio <= 2.5e-2
        and baseline["max_charge_balance_residual"] <= 5.0e-2
        and baseline["net_boundary_current_residual"] <= 1.0e-8
        and baseline["max_wall_current_leakage"] <= 1.0e-8
        and peak_velocity_deficit_ratio >= 1.0e-2
        and peak_station_velocity_deficit_ratio >= 5.0e-3
        and peak_pressure_excess >= 1.0e-3
        and pressure_excess_proxy >= 1.0e-3
        and current_proxy_peak >= 1.0e-2
        and y_l2_distortion >= 5.0e-3
        and z_l2_distortion >= 5.0e-3
        and wake_recovery_ratio > 0.8
    )
    return {
        **baseline,
        "divergence_to_field_ratio": divergence_ratio,
        "peak_velocity_deficit_ratio": peak_velocity_deficit_ratio,
        "peak_station_velocity_deficit_ratio": peak_station_velocity_deficit_ratio,
        "integrated_velocity_deficit_ratio": integrated_velocity_deficit_ratio,
        "peak_centerline_deficit_ratio": peak_centerline_deficit_ratio,
        "peak_centerline_station_deficit_ratio": peak_centerline_station_deficit_ratio,
        "recovery_station": recovery_station,
        "wake_recovery_ratio": wake_recovery_ratio,
        "peak_pressure_excess": peak_pressure_excess,
        "pressure_excess_proxy": pressure_excess_proxy,
        "y_l2_distortion": y_l2_distortion,
        "z_l2_distortion": z_l2_distortion,
        "y_peak_cut_abs_error": y_peak_cut_abs_error,
        "z_peak_cut_abs_error": z_peak_cut_abs_error,
        "peak_crosscut_distortion": peak_crosscut_distortion,
        "reference_kind": "matched_no_field_lmx",
        "external_reference_available": False,
        "internal_response_pass": validation_pass,
        "research_grade_validation_pass": False,
        "benchmark_pass": validation_pass,
    }


def validate_magnetic_obstacle_literature_slice(
    solution: ExtrudedInductionlessSolution,
    reference_solution: ExtrudedInductionlessSolution,
    *,
    field_ny: int = 81,
    field_nz: int = 81,
) -> dict[str, float | bool | str]:
    benchmark = validate_magnetic_obstacle_benchmark(
        solution,
        reference_solution,
        field_ny=field_ny,
        field_nz=field_nz,
    )
    x = np.asarray(solution.bundle.x, dtype=float)
    peak_index = (
        int(np.argmax(np.asarray(solution.bundle.field_scale, dtype=float)))
        if len(solution.bundle.x)
        else 0
    )
    peak_station = float(x[peak_index]) if x.size else 0.0
    outlet_station = float(x[-1]) if x.size else 0.0
    recovery_distance = max(float(benchmark["recovery_station"]) - peak_station, 0.0)
    normalized_recovery_distance = recovery_distance / max(
        outlet_station - peak_station, 1.0e-12
    )
    literature_shape_gate = bool(
        benchmark["benchmark_pass"]
        and benchmark["peak_centerline_deficit_ratio"] >= 0.2
        and benchmark["integrated_velocity_deficit_ratio"] >= 1.0e-2
        and benchmark["peak_crosscut_distortion"] >= 1.0e-1
        and benchmark["pressure_excess_proxy"] >= 5.0e-2
        and 0.0 <= normalized_recovery_distance <= 1.0
    )
    return {
        **benchmark,
        "peak_station": peak_station,
        "outlet_station": outlet_station,
        "recovery_distance": recovery_distance,
        "normalized_recovery_distance": normalized_recovery_distance,
        "literature_shape_gate": literature_shape_gate,
        "external_reference_available": False,
        "research_grade_validation_pass": False,
        "literature_status": "internal_lmx_response_only",
        "literature_pass": False,
    }


def validate_magnetic_obstacle_external_readiness(
    solution: ExtrudedInductionlessSolution,
    *,
    reference_case: str = "votyakov_zienicke_kolesnikov_jfm",
    field_ny: int = 81,
    field_nz: int = 81,
) -> dict[str, object]:
    """Report literature-facing magnetic-obstacle observables without claiming parity."""

    references = magnetic_obstacle_literature_reference_cases()
    if reference_case not in references:
        available = ", ".join(sorted(references))
        raise ValueError(
            f"Unknown magnetic-obstacle reference case {reference_case!r}; available: {available}"
        )
    baseline = validate_magnetic_obstacle_baseline(
        solution, field_ny=field_ny, field_nz=field_nz
    )
    bundle = solution.bundle
    x = np.asarray(bundle.x, dtype=float)
    field_scale = np.asarray(bundle.field_scale, dtype=float)
    mean_velocity = np.asarray(bundle.mean_velocity, dtype=float)
    pressure_proxy = np.asarray(bundle.current_scaled_pressure_proxy, dtype=float)
    peak_index = int(np.argmax(field_scale)) if field_scale.size else 0
    inlet_velocity = float(mean_velocity[0]) if mean_velocity.size else 0.0
    minimum_velocity = float(np.min(mean_velocity)) if mean_velocity.size else 0.0
    min_velocity_index = int(np.argmin(mean_velocity)) if mean_velocity.size else 0
    center_velocity_deficit = (
        float(inlet_velocity - mean_velocity[peak_index]) if mean_velocity.size else 0.0
    )
    centerline_deficit_ratio = center_velocity_deficit / max(
        abs(inlet_velocity), 1.0e-20
    )
    minimum_centerline_velocity_ratio = minimum_velocity / max(
        abs(inlet_velocity), 1.0e-20
    )
    recovery_ratio = (
        float(mean_velocity[-1] / max(abs(inlet_velocity), 1.0e-20))
        if mean_velocity.size
        else 0.0
    )
    integrated_pressure_proxy = (
        float(np.trapezoid(pressure_proxy, x) / max(float(x[-1] - x[0]), 1.0e-12))
        if pressure_proxy.size > 1 and x.size > 1
        else 0.0
    )
    recovery_index = peak_index
    if mean_velocity.size and x.size:
        deficit_scale = max(abs(center_velocity_deficit), 1.0e-20)
        recovery_threshold = inlet_velocity - 0.05 * deficit_scale
        downstream = np.flatnonzero(mean_velocity[peak_index:] >= recovery_threshold)
        recovery_index = (
            peak_index + int(downstream[0])
            if downstream.size
            else len(mean_velocity) - 1
        )
    downstream_length = max(float(x[-1] - x[peak_index]), 1.0e-12) if x.size else 1.0
    recovery_distance = (
        max(float(x[recovery_index] - x[peak_index]), 0.0) if x.size else 0.0
    )
    normalized_recovery_distance = recovery_distance / downstream_length
    observable_payload = {
        "centerline_velocity_deficit": center_velocity_deficit,
        "centerline_velocity_deficit_ratio": centerline_deficit_ratio,
        "minimum_centerline_velocity": minimum_velocity,
        "minimum_centerline_velocity_ratio": minimum_centerline_velocity_ratio,
        "minimum_velocity_station": float(x[min_velocity_index]) if x.size else 0.0,
        "peak_field_station": float(x[peak_index]) if x.size else 0.0,
        "recovery_station": float(x[recovery_index]) if x.size else 0.0,
        "wake_recovery_ratio": recovery_ratio,
        "normalized_recovery_distance": normalized_recovery_distance,
        "pressure_drop_proxy": integrated_pressure_proxy,
        "current_proxy_peak": float(baseline["current_proxy_peak"]),
        "field_velocity_correlation": float(baseline["field_velocity_correlation"]),
        "max_charge_balance_residual": float(baseline["max_charge_balance_residual"]),
    }
    measured_observables = sorted(observable_payload)
    required_observables = list(references[reference_case]["required_observables"])
    missing_observables = [
        observable
        for observable in required_observables
        if observable
        not in {
            "centerline_velocity_deficit",
            "wake_recovery",
            "pressure_drop_or_drag_proxy",
            "pressure_drop",
            "current_closure",
        }
    ]
    return {
        "reference_case": reference_case,
        "reference": references[reference_case],
        "measured_observables": measured_observables,
        "required_observables": required_observables,
        "missing_observables": missing_observables,
        "observables": observable_payload,
        "external_reference_available": False,
        "research_grade_validation_pass": False,
        "validation_status": "literature_target_registered_no_digitized_reference",
    }


def validate_wham_mirror_pipe_baseline(
    solution: ExtrudedInductionlessSolution,
) -> dict[str, float | bool]:
    if solution.problem.case.geometry.kind != "pipe_ogrid":
        raise ValueError(
            "WHAM mirror pipe validation currently supports pipe_ogrid only"
        )
    if (
        solution.problem.case.magnetic_field.kind != "tabulated"
        or solution.problem.case.magnetic_field.table_path is None
    ):
        raise ValueError(
            "WHAM mirror pipe validation requires a tabulated magnetic field"
        )

    bundle = solution.bundle
    validation = solution.validation
    x = np.asarray(bundle.x, dtype=float)
    zeros = np.zeros_like(x)
    centerline_field = np.asarray(
        sample_tabulated_field_volume(
            solution.problem.case.magnetic_field.table_path,
            x=x,
            y=zeros,
            z=zeros,
        ),
        dtype=float,
    )
    bz_profile = centerline_field[..., 2]
    field_scale = np.abs(bz_profile) / max(float(np.max(np.abs(bz_profile))), 1.0e-12)
    mean_velocity = np.asarray(bundle.mean_velocity, dtype=float)
    current_proxy = np.asarray(bundle.current_scaled_pressure_proxy, dtype=float)
    pressure_span = np.max(np.asarray(bundle.p, dtype=float), axis=(1, 2)) - np.min(
        np.asarray(bundle.p, dtype=float), axis=(1, 2)
    )
    peak_index = int(np.argmax(field_scale)) if field_scale.size else 0
    obstacle_velocity_deficit = (
        float(mean_velocity[0] - mean_velocity[peak_index])
        if mean_velocity.size
        else 0.0
    )
    field_velocity_correlation = float(
        _safe_correlation(jnp.asarray(field_scale), jnp.asarray(mean_velocity))
    )
    current_proxy_peak = (
        float(np.max(np.abs(current_proxy))) if current_proxy.size else 0.0
    )
    pressure_drop_proxy = (
        float(np.trapezoid(pressure_span, x) / max(x[-1] - x[0], 1.0e-12))
        if pressure_span.size > 1
        else 0.0
    )
    validation_pass = bool(
        validation.max_charge_balance_residual <= 6.0e-2
        and validation.net_boundary_current_residual <= 1.0e-8
        and validation.max_wall_current_leakage <= 1.0e-8
        and obstacle_velocity_deficit > 1.0e-8
        and current_proxy_peak > 1.0e-6
        and pressure_drop_proxy > 1.0e-8
        and field_velocity_correlation < -0.2
    )
    return {
        "obstacle_velocity_deficit": obstacle_velocity_deficit,
        "current_proxy_peak": current_proxy_peak,
        "field_velocity_correlation": field_velocity_correlation,
        "pressure_drop_proxy": pressure_drop_proxy,
        "max_charge_balance_residual": float(validation.max_charge_balance_residual),
        "max_wall_current_leakage": float(validation.max_wall_current_leakage),
        "net_boundary_current_residual": float(
            validation.net_boundary_current_residual
        ),
        "validation_pass": validation_pass,
    }


def build_extruded_problem_from_case(
    case: CaseSpec,
    *,
    entry_center: float,
    exit_center: float,
    transition_width: float,
    axis: str = "z",
) -> ExtrudedInductionlessProblem:
    profile = smooth_fringing_profile(
        length=case.geometry.length,
        nx=case.geometry.nx,
        entry_center=entry_center,
        exit_center=exit_center,
        transition_width=transition_width,
        peak_scale=1.0,
        axis=axis,
    )
    return ExtrudedInductionlessProblem(case=case, profile=profile)


def _station_case(
    base_case: CaseSpec, *, axis: str, magnitude: float, suffix: str
) -> CaseSpec:
    station_case = clone_case_with_field(
        base_case, axis=axis, magnitude=magnitude, suffix=suffix
    )
    return replace(
        station_case,
        solver=replace(station_case.solver, kind="fully_developed_inductionless"),
    )


def run_fringing_station_sweep(
    base_case: CaseSpec,
    profile: FringingProfile,
    *,
    solver=solve_steady,
) -> list[dict[str, float]]:
    if (
        base_case.magnetic_field.kind != "constant"
        or base_case.magnetic_field.value is None
    ):
        raise ValueError("Fringing station sweep requires a constant-field base case")

    base_magnitude = max(
        abs(float(component)) for component in base_case.magnetic_field.value
    )
    history: list[dict[str, float]] = []
    previous_state = None
    for index, (x_value, scale) in enumerate(
        zip(profile.x, profile.field_scale, strict=True)
    ):
        station_case = _station_case(
            base_case,
            axis=profile.axis,
            magnitude=base_magnitude * float(scale),
            suffix=f"station{index:03d}",
        )
        solution: Solution = solver(station_case, initial_state=previous_state)
        metrics = validation_summary(
            solution, station_case.name, ha=base_case.geometry.target_ha
        )
        history.append(
            {
                "x": float(x_value),
                "field_scale": float(scale),
                "u_max": float(metrics["u_max"]),
                "mean_velocity": float(metrics["mean_velocity"]),
                "volumetric_flow_rate": float(metrics["volumetric_flow_rate"]),
                "current_scaled_pressure_proxy": float(
                    metrics["current_scaled_pressure_proxy"]
                ),
                "residual": float(solution.state.residual),
            }
        )
        previous_state = solution.state
    return history


def run_extruded_inductionless_slice(
    base_case: CaseSpec,
    profile: FringingProfile,
    *,
    solver=solve_steady,
) -> ExtrudedFieldBundle:
    if base_case.geometry.kind not in {"rect_duct", "layered_duct"}:
        raise ValueError(
            "The current extruded_inductionless slice is only implemented for rectangular and layered ducts"
        )
    if (
        base_case.magnetic_field.kind != "constant"
        or base_case.magnetic_field.value is None
    ):
        raise ValueError("Extruded fringing slice requires a constant-field base case")

    base_magnitude = max(
        abs(float(component)) for component in base_case.magnetic_field.value
    )
    previous_state = None
    station_solutions: list[Solution] = []
    for index, scale in enumerate(profile.field_scale):
        station_case = _station_case(
            base_case,
            axis=profile.axis,
            magnitude=base_magnitude * float(scale),
            suffix=f"station{index:03d}",
        )
        solution = solver(station_case, initial_state=previous_state)
        station_solutions.append(solution)
        previous_state = solution.state

    first = station_solutions[0]
    u = jnp.stack([solution.state.u for solution in station_solutions], axis=0)
    phi = jnp.stack([solution.state.phi for solution in station_solutions], axis=0)
    jy = jnp.stack([solution.state.jy for solution in station_solutions], axis=0)
    jz = jnp.stack([solution.state.jz for solution in station_solutions], axis=0)
    lorentz_x = jnp.stack(
        [solution.state.lorentz_x for solution in station_solutions], axis=0
    )
    residual = jnp.asarray(
        [solution.state.residual for solution in station_solutions], dtype=float
    )
    volumetric_flow_rate = jnp.asarray(
        [
            float(solution.diagnostics.volumetric_flow_rate_history[-1])
            if solution.diagnostics.volumetric_flow_rate_history.size
            else 0.0
            for solution in station_solutions
        ],
        dtype=float,
    )
    mean_velocity = jnp.asarray(
        [
            float(solution.diagnostics.mean_velocity_history[-1])
            if solution.diagnostics.mean_velocity_history.size
            else float(jnp.mean(solution.state.u))
            for solution in station_solutions
        ],
        dtype=float,
    )
    current_scaled_pressure_proxy = jnp.asarray(
        [
            float(solution.diagnostics.current_scaled_pressure_proxy_history[-1])
            if solution.diagnostics.current_scaled_pressure_proxy_history.size
            else 0.0
            for solution in station_solutions
        ],
        dtype=float,
    )
    charge_balance_residual = jnp.asarray(
        [
            float(solution.diagnostics.charge_balance_residual_history[-1])
            if solution.diagnostics.charge_balance_residual_history.size
            else 0.0
            for solution in station_solutions
        ],
        dtype=float,
    )
    return ExtrudedFieldBundle(
        x=jnp.asarray(profile.x, dtype=float),
        y=first.mesh.y_centers,
        z=first.mesh.z_centers,
        field_scale=jnp.asarray(profile.field_scale, dtype=float),
        u=u,
        v=jnp.zeros_like(u),
        w=jnp.zeros_like(u),
        p=jnp.zeros_like(u),
        phi=phi,
        jx=jnp.zeros_like(u),
        jy=jy,
        jz=jz,
        lorentz_x=lorentz_x,
        lorentz_y=jnp.zeros_like(u),
        lorentz_z=jnp.zeros_like(u),
        residual=residual,
        volumetric_flow_rate=volumetric_flow_rate,
        mean_velocity=mean_velocity,
        axial_current=jnp.zeros_like(mean_velocity),
        wall_current_leakage=jnp.zeros_like(mean_velocity),
        current_scaled_pressure_proxy=current_scaled_pressure_proxy,
        charge_balance_residual=charge_balance_residual,
        boundary_current_residual=jnp.zeros_like(mean_velocity),
        geometry_kind=base_case.geometry.kind,
        solver_kind=base_case.solver.kind,
    )


def _shard_extruded_fields(
    fields: tuple[jnp.ndarray, ...], *, num_devices: int | None
) -> tuple[jnp.ndarray, ...]:
    """Place 3-D extruded fields on an axial JAX device mesh.

    JAX propagates this named sharding through the production operators and
    inserts the required neighbor communication at axial stencil boundaries.
    One device retains the normal single-device placement.
    """

    if num_devices is None or num_devices == 1:
        return fields
    devices = jax.devices()
    if not 1 <= num_devices <= len(devices):
        raise ValueError(
            f"Requested {num_devices} devices, but only {len(devices)} are visible."
        )
    axial_size = fields[0].shape[0]
    if axial_size % num_devices:
        raise ValueError(
            f"Axial cell count {axial_size} must be divisible by {num_devices} devices."
        )
    sharding = _axial_field_sharding(num_devices)
    # JAX 0.6.x CUDA can leave non-primary shards uninitialized when directly
    # resharding a single-GPU array. Stage each global initial field once on the
    # host; all subsequent production iterations remain device-resident.
    return tuple(jax.device_put(np.asarray(field), sharding) for field in fields)


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
        geometry_kind=case.geometry.kind,
        solver_kind=case.solver.kind,
        axial_pressure_loss_gradient=(
            jnp.zeros_like(x)
            if axial_pressure_loss_gradient is None
            else axial_pressure_loss_gradient
        ),
        transverse_pressure_difference=(
            jnp.zeros_like(x)
            if transverse_pressure_difference is None
            else transverse_pressure_difference
        ),
        iteration_residual_history=jnp.asarray(residual_history, dtype=float),
        iteration_component_residual_history=jnp.asarray(
            component_history, dtype=float
        ).reshape((-1, 6)),
        iteration_pressure_residual_history=jnp.asarray(pressure_history, dtype=float),
        iteration_electric_linear_history=jnp.asarray(
            electric_history, dtype=float
        ).reshape((-1, 6)),
        iteration_potential_residual_history=jnp.asarray(
            potential_history, dtype=float
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
        checkpoint_interval
        and (step % checkpoint_interval == 0 or converged or step == total_steps)
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


@lru_cache(maxsize=None)
def _axial_field_sharding(num_devices: int) -> NamedSharding:
    """Return one process-stable axial mesh for compilation and repeat reuse."""

    devices = jax.devices()
    if not 1 <= num_devices <= len(devices):
        raise ValueError(
            f"Requested {num_devices} devices, but only {len(devices)} are visible."
        )
    mesh = Mesh(np.asarray(devices[:num_devices], dtype=object), ("x",))
    return NamedSharding(mesh, P("x", None, None))


def _solve_extruded_projection(
    problem: ExtrudedInductionlessProblem,
    *,
    initial_bundle: ExtrudedFieldBundle | None = None,
    num_devices: int | None = None,
    progress_callback: Callable[[ExtrudedIterationProgress], None] | None = None,
    checkpoint_interval: int | None = None,
) -> ExtrudedFieldBundle:
    case = problem.case
    mesh = _cross_section_mesh(case)
    use_alex_b2_finite_volume = (
        case.name.startswith("alex_b2-fringing-square_")
        and case.geometry.kind == "layered_duct"
    )
    use_alex_b1_finite_volume = (
        case.name.startswith("alex_b1-fringing-pipe_")
        and case.geometry.kind == "pipe_ogrid"
    )
    use_compatible_steady_b1 = use_alex_b1_finite_volume
    if case.name.startswith("alex_") and not (
        use_alex_b1_finite_volume or use_alex_b2_finite_volume
    ):
        raise NotImplementedError(
            "Unsupported ALEX production case; only the frozen B1 pipe and B2 square "
            "finite-volume paths are implemented"
        )
    if num_devices is not None and num_devices > 1 and not use_alex_b2_finite_volume:
        raise NotImplementedError(
            "Production spatial sharding currently supports the ALEX B2 duct path"
        )
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
        fluid_mask = (
            _broadcast_cross_section(materials.fluid_mask.astype(float), nx) > 0.5
        )
        radial_fluid_count = (
            _pipe_radial_fluid_count(fluid_mask) if use_alex_b1_finite_volume else None
        )
        rr = jnp.broadcast_to(jnp.maximum(r[None, :, None], 0.5 * dr), (nx, nr, ntheta))
        theta_grid = jnp.broadcast_to(theta[None, None, :], (nx, nr, ntheta))
        forcing = float(case.forcing)
        field_scale = jnp.asarray(problem.profile.field_scale, dtype=float)
        bx, by, bz = _sample_station_magnetic_field_pipe(
            case,
            rr=rr,
            theta_grid=theta_grid,
            field_scale=field_scale,
            x=problem.profile.x,
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
            * (
                1.0 / max(dx**2, 1.0e-12)
                + 1.0 / max(min_dr**2, 1.0e-12)
                + 1.0 / max(min_arc**2, 1.0e-12)
            )
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
        outer_steps = max(
            2,
            min(
                case.time_stepper.max_steps, max(6, case.solver.coupling_iterations * 2)
            ),
        )
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
        momentum_iterations = max(
            poisson_iterations, 2000 if use_compatible_steady_b1 else 400
        )
        momentum_tolerance = min(poisson_tolerance, 1.0e-10)
        velocity_limit = max(
            5.0, 2.0 * math.sqrt(float(case.geometry.target_ha or 1.0))
        )
        scalar_limit = max(
            20.0,
            2.0 * float(jnp.max(bx**2 + by**2 + bz**2)),
        )
        electric_potential_scale = max(
            1.0, math.sqrt(float(jnp.max(bx**2 + by**2 + bz**2)))
        )
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

            def diffusion_system_solve(
                linear_rhs, volume, coefficients, wall_sink, initial
            ):
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

            diffusion_system_solve = _reuse_fringing_jit(
                kernel_key, jax.jit(diffusion_system_solve)
            )
            steady_reaction = (
                2.0
                * sigma[:, :count, :]
                * (
                    bx[:, :count, :] ** 2
                    + br[:, :count, :] ** 2
                    + btheta[:, :count, :] ** 2
                )
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
                steady_rate_diagonal = (
                    sum(steady_coefficients) + wall_sink + steady_reaction
                )
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
                    _system_solve=(
                        None if use_compatible_steady_b1 else diffusion_system_solve
                    ),
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

                def modal_momentum_solve(rhs):
                    return _solvax_diffusion_pipe(
                        rhs,
                        momentum_viscosity,
                        dt=None,
                        dx=dx,
                        r_faces=faces,
                        r_centers=centers,
                        dtheta=dtheta,
                        iterations=momentum_iterations,
                        tolerance=momentum_tolerance,
                        reaction=steady_reaction,
                        decouple_axial=True,
                    )[0]

                modal_momentum_solve = _reuse_fringing_jit(
                    (
                        "b1_modal_momentum",
                        jax.default_backend(),
                        kernel_key,
                        _array_fingerprint(momentum_viscosity, steady_reaction),
                    ),
                    jax.jit(modal_momentum_solve),
                )
            else:
                modal_momentum_solve = None
            response_rhs = (1.0 if use_compatible_steady_b1 else dt) / rho[:, :count, :]
            response_fluid, _, _ = momentum_solve(
                response_rhs, jnp.zeros_like(response_rhs)
            )
            if not use_compatible_steady_b1:
                response_cross_section = jnp.mean(response_fluid, axis=0, keepdims=True)
                response_fluid = jnp.broadcast_to(
                    response_cross_section, response_fluid.shape
                )
                flow_response_matrix = None
            else:
                basis_rhs = (
                    jnp.eye(nx, dtype=u.dtype)[:, :, None, None]
                    / rho[None, :, :count, :]
                )
                zero = jnp.zeros_like(response_fluid)
                basis_response = jnp.stack(
                    tuple(momentum_solve(rhs, zero)[0] for rhs in basis_rhs)
                )
                flow_response_matrix = jnp.sum(
                    basis_response * fluid_cell_area[None, :, :count, :], axis=(2, 3)
                ).T
            unit_pressure_response = (
                jnp.zeros_like(u).at[:, :count, :].set(response_fluid)
            )
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
                dp_dx, dp_dr, dp_dtheta = _pipe_gradient_3d(
                    p, dx=dx, dr=dr, dtheta=dtheta, r=rr
                )
                laplacian_u = _pipe_laplacian_3d(u, dx=dx, dr=dr, dtheta=dtheta, r=rr)
                laplacian_v = _pipe_laplacian_3d(v, dx=dx, dr=dr, dtheta=dtheta, r=rr)
                laplacian_w = _pipe_laplacian_3d(w, dx=dx, dr=dr, dtheta=dtheta, r=rr)
            if use_alex_b1_finite_volume:
                count = radial_fluid_count
                faces = r_faces[: count + 1]
                centers = r[:count]
                rhs_u = u[:, :count, :] + dt * (
                    forcing / rho[:, :count, :]
                    + lorentz_x[:, :count, :] / rho[:, :count, :]
                )
                rhs_v = v[:, :count, :] + dt * (
                    lorentz_r[:, :count, :] / rho[:, :count, :]
                )
                rhs_w = w[:, :count, :] + dt * (
                    lorentz_theta[:, :count, :] / rho[:, :count, :]
                )
                if use_compatible_steady_b1:
                    rhs_u = (forcing + lorentz_x[:, :count, :]) / rho[
                        :, :count, :
                    ] + steady_reaction * u[:, :count, :]
                    rhs_v = (
                        lorentz_r[:, :count, :] / rho[:, :count, :]
                        + steady_reaction * v[:, :count, :]
                    )
                    rhs_w = (
                        lorentz_theta[:, :count, :] / rho[:, :count, :]
                        + steady_reaction * w[:, :count, :]
                    )
                u_fluid, _, _ = momentum_solve(rhs_u, u[:, :count, :])
                v_fluid, _, _ = momentum_solve(rhs_v, v[:, :count, :])
                w_fluid, _, _ = momentum_solve(rhs_w, w[:, :count, :])
                u_star = jnp.zeros_like(u).at[:, :count, :].set(u_fluid)
                v_star = jnp.zeros_like(v).at[:, :count, :].set(v_fluid)
                w_star = jnp.zeros_like(w).at[:, :count, :].set(w_fluid)
            else:
                u_star = u + dt * (
                    laplacian_u * nu + forcing / rho + lorentz_x / rho - dp_dx / rho
                )
                v_star = v + dt * (laplacian_v * nu + lorentz_r / rho - dp_dr / rho)
                w_star = w + dt * (
                    laplacian_w * nu + lorentz_theta / rho - dp_dtheta / rho
                )
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
                        pressure_preconditioner_mobility=(
                            pressure_preconditioner_mobility
                        ),
                        apply_modal_momentum_inverse=modal_momentum_solve,
                        modal_momentum_coefficients=steady_coefficients,
                        modal_momentum_sink=wall_sink + steady_reaction,
                        modal_stabilization=True,
                        modal_factor_key=modal_factor_key,
                        physical_tolerance=ALEX_BALANCE_TOLERANCE,
                    )
                    u_next = (
                        jnp.zeros_like(u).at[:, :count, :].set(steady_projection[0])
                    )
                    v_next = (
                        jnp.zeros_like(v).at[:, :count, :].set(steady_projection[1])
                    )
                    w_next = (
                        jnp.zeros_like(w).at[:, :count, :].set(steady_projection[2])
                    )
                    p_corr = (
                        jnp.zeros_like(p).at[:, :count, :].set(steady_projection[3])
                    )
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
                divergence = _pipe_divergence_3d(
                    u_star, v_star, w_star, dx=dx, dr=dr, dtheta=dtheta, r=rr
                )
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
                dpc_dx, dpc_dr, dpc_dtheta = _pipe_gradient_3d(
                    p_corr, dx=dx, dr=dr, dtheta=dtheta, r=rr
                )
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
                    u_next, axial_pressure_loss_gradient = (
                        _apply_fixed_flow_pressure_constraint(
                            u_next,
                            unit_pressure_response=unit_pressure_response,
                            active_mask=fluid_mask,
                            cell_area=fluid_cell_area,
                            target_flow_rate=target_flow_rate,
                            base_pressure_loss_gradient=forcing,
                        )
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
            p = _clip_state(
                p_corr if use_alex_b1_finite_volume else p + p_corr, scalar_limit
            )

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
            jtheta = _clip_state(
                0.5 * (ftheta + jnp.roll(ftheta, 1, axis=2)), scalar_limit
            )
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
                mapped_state = (
                    jnp.stack((u_next, v_next, w_next, phi)) / fixed_point_scale
                )
                fixed_point_residual = mapped_state - current_state
                if case.solver.coupling_acceleration == "anderson":
                    fixed_point_iterates.append(current_state)
                    fixed_point_residuals.append(fixed_point_residual)
                    del fixed_point_iterates[: -case.solver.coupling_history_depth]
                    del fixed_point_residuals[: -case.solver.coupling_history_depth]
                    accelerated = _anderson_extruded_state(
                        fixed_point_iterates,
                        fixed_point_residuals,
                        history_size=case.solver.coupling_history_depth,
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
                    accelerated = (
                        current_state + fixed_point_relaxation * fixed_point_residual
                    )
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
            u=jnp.nan_to_num(u),
            v=jnp.nan_to_num(v),
            w=jnp.nan_to_num(w),
            p=jnp.nan_to_num(p),
            phi=jnp.nan_to_num(phi),
            jx=jnp.nan_to_num(jx),
            jy=jnp.nan_to_num(jr),
            jz=jnp.nan_to_num(jtheta),
            lorentz_x=jnp.nan_to_num(lorentz_x),
            lorentz_y=jnp.nan_to_num(lorentz_r),
            lorentz_z=jnp.nan_to_num(lorentz_theta),
            residual=jnp.nan_to_num(residual),
            volumetric_flow_rate=jnp.nan_to_num(volumetric_flow_rate),
            mean_velocity=jnp.nan_to_num(mean_velocity),
            axial_current=jnp.nan_to_num(axial_current),
            wall_current_leakage=jnp.nan_to_num(wall_current_leakage),
            current_scaled_pressure_proxy=jnp.nan_to_num(current_scaled_pressure_proxy),
            charge_balance_residual=jnp.nan_to_num(charge_balance_residual),
            boundary_current_residual=jnp.nan_to_num(boundary_current_residual),
            geometry_kind=case.geometry.kind,
            solver_kind=case.solver.kind,
            axial_pressure_loss_gradient=jnp.nan_to_num(axial_pressure_loss_gradient),
            transverse_pressure_difference=jnp.zeros((nx,), dtype=float),
            iteration_residual_history=jnp.asarray(residual_by_step, dtype=float),
            iteration_component_residual_history=jnp.asarray(
                component_residual_by_step, dtype=float
            ).reshape((-1, 6)),
            iteration_pressure_residual_history=jnp.asarray(
                pressure_residual_by_step, dtype=float
            ),
            iteration_electric_linear_history=jnp.asarray(
                electric_linear_by_step, dtype=float
            ).reshape((-1, 6)),
            iteration_potential_residual_history=jnp.asarray(
                potential_residual_by_step, dtype=float
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
    fluid_bounds = (
        _rectangular_fluid_bounds(fluid_mask) if use_alex_b2_finite_volume else None
    )
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
    bx, by, bz = _sample_station_magnetic_field_duct(
        case,
        mesh,
        field_scale=field_scale,
        volume_field=problem.profile.volume_field,
        nx=nx,
        ny=ny,
        nz=nz,
    )

    if initial_bundle is not None:
        if initial_bundle.u.shape != (nx, ny, nz):
            raise ValueError(
                "Extruded restart bundle shape does not match the current duct problem"
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
        0.001
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
            raise ValueError(
                "ALEX B2 requires one inlet flow rate and zero outlet pressure"
            )
        target_flow_rate = float(inlet[0].value)
    else:
        target_flow_rate = (
            float(
                jnp.mean(
                    jnp.sum(jnp.where(fluid_mask, u * cell_area, 0.0), axis=(1, 2))
                )
            )
            if initial_bundle is not None or case.initial_velocity != 0.0
            else None
        )
    outer_steps = max(
        2, min(case.time_stepper.max_steps, max(6, case.solver.coupling_iterations * 2))
    )
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
    electric_potential_scale = max(
        1.0, math.sqrt(float(jnp.max(bx**2 + by**2 + bz**2)))
    )
    residual_by_step: list[float] = []
    component_residual_by_step: list[tuple[float, ...]] = []
    pressure_residual_by_step: list[float] = []
    electric_linear_by_step: list[tuple[float, ...]] = []
    potential_residual_by_step: list[float] = []
    fixed_point_iterates: list[jnp.ndarray] = []
    fixed_point_residuals: list[jnp.ndarray] = []
    previous_fixed_point_residual: jnp.ndarray | None = None
    steady_streak = 0
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
    axial_pressure_loss_gradient = jnp.full((nx,), forcing, dtype=float)
    if use_alex_b2_finite_volume:
        y0, y1, z0, z1 = fluid_bounds
        local_dy = dy[y0:y1]
        local_dz = dz[z0:z1]
        field_sharding = (
            u.sharding if num_devices is not None and num_devices > 1 else None
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
            scalar_limit,
        )

        fluid_shape = (nx, y1 - y0, z1 - z0)
        zero_rho_phi = (
            jnp.zeros((nx + 1, *fluid_shape[1:]), dtype=u.dtype),
            jnp.zeros((nx, fluid_shape[1] + 1, fluid_shape[2]), dtype=u.dtype),
            jnp.zeros((*fluid_shape[:2], fluid_shape[2] + 1), dtype=u.dtype),
        )
        zero_boundary_velocity = (
            jnp.zeros((*fluid_shape[1:], 3), dtype=u.dtype),
            jnp.zeros((*fluid_shape[1:], 3), dtype=u.dtype),
            jnp.zeros((nx, fluid_shape[2], 3), dtype=u.dtype),
            jnp.zeros((nx, fluid_shape[2], 3), dtype=u.dtype),
            jnp.zeros((nx, fluid_shape[1], 3), dtype=u.dtype),
            jnp.zeros((nx, fluid_shape[1], 3), dtype=u.dtype),
        )

        def momentum_solve(velocity, force, density, viscosity):
            return _solvax_implicit_momentum_duct(
                velocity[:, y0:y1, z0:z1],
                force[:, y0:y1, z0:z1],
                density[:, y0:y1, z0:z1],
                viscosity[:, y0:y1, z0:z1],
                zero_rho_phi,
                zero_boundary_velocity,
                dt=dt,
                dx=dx,
                dy=local_dy,
                dz=local_dz,
                iterations=momentum_iterations,
                tolerance=momentum_tolerance,
                include_axial_line=False,
                prescribed_inlet=False,
            )

        # The multi-device branches require a real device mesh and are covered
        # by the GPU scaling gate documented in ``docs/performance.md``.
        if field_sharding is not None:  # pragma: no cover - hardware gate
            replicated_sharding = NamedSharding(field_sharding.mesh, P())
            momentum_solve = jax.jit(
                momentum_solve,
                in_shardings=(field_sharding,) * 4,
                out_shardings=(
                    field_sharding,
                    replicated_sharding,
                    replicated_sharding,
                ),
            )
            momentum_solve = _reuse_fringing_jit(
                ("momentum", *kernel_key), momentum_solve
            )

        def embed_velocity(velocity, mask):
            full = (
                jnp.zeros((*mask.shape, 3), dtype=velocity.dtype)
                .at[:, y0:y1, z0:z1]
                .set(velocity)
            )
            full = jnp.where(mask[..., None], full, 0.0)
            return full[..., 0], full[..., 1], full[..., 2]

        if field_sharding is not None:  # pragma: no cover - hardware gate
            embed_velocity = jax.jit(
                embed_velocity,
                in_shardings=(field_sharding,) * 2,
                out_shardings=(field_sharding,) * 3,
            )
            embed_velocity = _reuse_fringing_jit(
                ("embed_velocity", *kernel_key), embed_velocity
            )

    else:
        unit_pressure_response = _enforce_velocity_bc_3d(
            jnp.where(fluid_mask, dt / rho, 0.0), fluid_mask
        )

    if use_alex_b2_finite_volume:
        # The strict corner-cell bound oversolves the global PCG system; the
        # returned local residual is still checked against the physical gate.
        electric_volume_min = 4.0 * float(jnp.min(dy) * jnp.min(dz))
        # Transverse lines capture the dominant wall-normal coupling; an axial
        # line crosses shards, dilutes those blocks, and regresses PCG scaling.
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
            current_x = _clip_state(conductivity * (-dphi_dx + emf_x), scalar_limit)
            current_y = _clip_state(conductivity * (-dphi_dy + emf_y), scalar_limit)
            current_z = _clip_state(conductivity * (-dphi_dz + emf_z), scalar_limit)
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

        if field_sharding is not None:  # pragma: no cover - hardware gate
            axial_sharding = NamedSharding(field_sharding.mesh, P("x"))
            state_sharding = NamedSharding(
                field_sharding.mesh, P(None, "x", None, None)
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
            (
                mixed_boundary_projection,
                electric_solve,
                emf_operator,
                reconstruct_electric,
                lorentz_operator,
                scaled_state,
                state_difference,
                unscaled_state,
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
                )
            )

    for step in range(outer_steps):
        phi_previous = phi
        pressure_observable_previous = (
            _cross_duct_pressure_difference(
                p, active_mask=fluid_mask, magnetic_axis=1, side_axis=2
            )
            if use_alex_b2_finite_volume
            else jnp.zeros((nx,), dtype=p.dtype)
        )
        uxb_x = v * bz - w * by
        uxb_y = w * bx - u * bz
        uxb_z = u * by - v * bx
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

        if use_alex_b2_finite_volume:
            dp_dx = jnp.zeros_like(p)
            dp_dy = jnp.zeros_like(p)
            dp_dz = jnp.zeros_like(p)
        else:
            dp_dx, dp_dy, dp_dz = _gradient_3d(p, dx=dx, dy=dy_momentum, dz=dz_momentum)
            laplacian_u = _laplacian_3d(u, dx=dx, dy=dy_momentum, dz=dz_momentum)
            laplacian_v = _laplacian_3d(v, dx=dx, dy=dy_momentum, dz=dz_momentum)
            laplacian_w = _laplacian_3d(w, dx=dx, dy=dy_momentum, dz=dz_momentum)
        if use_alex_b2_finite_volume:
            velocity = jnp.stack((u, v, w), axis=-1)
            momentum_force = jnp.stack(
                (lorentz_x + forcing, lorentz_y, lorentz_z), axis=-1
            )
            velocity_fluid, _, _ = momentum_solve(velocity, momentum_force, rho, nu)
            u_star, v_star, w_star = embed_velocity(velocity_fluid, fluid_mask)
        else:
            u_star = u + dt * (
                nu * laplacian_u + forcing / rho + lorentz_x / rho - dp_dx / rho
            )
            v_star = v + dt * (nu * laplacian_v + lorentz_y / rho - dp_dy / rho)
            w_star = w + dt * (nu * laplacian_w + lorentz_z / rho - dp_dz / rho)
        u_star = _clip_state(u_star, velocity_limit)
        v_star = _clip_state(v_star, velocity_limit)
        w_star = _clip_state(w_star, velocity_limit)
        if not use_alex_b2_finite_volume:
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
            ) = mixed_boundary_projection(
                u_star,
                v_star,
                w_star,
                p,
                rho,
                fluid_mask,
            )
            p_corr = _clip_state(p_corr, scalar_limit)
            u_next = _clip_state(u_next, velocity_limit)
            v_next = _clip_state(v_next, velocity_limit)
            w_next = _clip_state(w_next, velocity_limit)
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
            dpc_dx, dpc_dy, dpc_dz = _gradient_3d(
                p_corr, dx=dx, dy=dy_momentum, dz=dz_momentum
            )
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
                u_next, axial_pressure_loss_gradient = (
                    _apply_fixed_flow_pressure_constraint(
                        u_next,
                        unit_pressure_response=unit_pressure_response,
                        active_mask=fluid_mask,
                        cell_area=cell_area,
                        target_flow_rate=target_flow_rate,
                        base_pressure_loss_gradient=forcing,
                    )
                )
            u_next = _enforce_velocity_bc_3d(u_next, fluid_mask)
            projected_divergence_norm = float("nan")
            fixed_flow_error = 0.0
        p = _clip_state(
            jnp.where(
                fluid_mask,
                p_corr if use_alex_b2_finite_volume else p + p_corr,
                0.0,
            ),
            scalar_limit,
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
        phi = _clip_state(phi, scalar_limit)
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
                    pressure_update,
                    potential_update,
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
            pressure_update,
            potential_update,
            *electric_diagnostics,
        ) = map(float, diagnostics)
        electric_linear_by_step.append(tuple(electric_diagnostics))
        update_residual = max(
            u_update,
            v_update,
            w_update,
            pressure_update,
            potential_update,
        )
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
        instantaneous_convergence = (
            update_residual <= case.solver.coupling_tolerance
            and projected_divergence_max <= ALEX_BALANCE_TOLERANCE
            and flow_error_value <= ALEX_BALANCE_TOLERANCE
            and charge_balance <= ALEX_BALANCE_TOLERANCE
        )
        accepted_state_converged = (
            max(u_update, v_update, w_update, potential_update)
            <= case.time_stepper.steady_tolerance
        )
        if use_alex_b2_finite_volume:
            steady_streak, converged = _sustained_convergence(
                steady_streak, instantaneous_convergence
            )
        else:
            converged = instantaneous_convergence
        if use_alex_b2_finite_volume and not converged and step + 1 < outer_steps:
            current_state = scaled_state(u, v, w, phi_previous)
            mapped_state = scaled_state(u_next, v_next, w_next, phi)
            fixed_point_residual = state_difference(mapped_state, current_state)
            if case.solver.coupling_acceleration == "anderson":
                fixed_point_iterates.append(current_state)
                fixed_point_residuals.append(fixed_point_residual)
                del fixed_point_iterates[: -case.solver.coupling_history_depth]
                del fixed_point_residuals[: -case.solver.coupling_history_depth]
                if field_sharding is None:
                    accelerated = _anderson_extruded_state(
                        fixed_point_iterates,
                        fixed_point_residuals,
                        history_size=case.solver.coupling_history_depth,
                        regularization=case.solver.coupling_regularization,
                        damping=case.solver.coupling_damping,
                    )
                elif len(fixed_point_iterates) == 1:
                    accelerated = mapped_state
                else:
                    iterates = tuple(
                        fixed_point_iterates[-case.solver.coupling_history_depth :]
                    )
                    residuals = tuple(
                        fixed_point_residuals[-case.solver.coupling_history_depth :]
                    )
                    mix_history = jax.jit(
                        lambda xs, rs: anderson_mixing(
                            jnp.stack(xs),
                            jnp.stack(rs),
                            regularization=case.solver.coupling_regularization,
                            damping=case.solver.coupling_damping,
                        ),
                        in_shardings=(
                            (state_sharding,) * len(iterates),
                            (state_sharding,) * len(residuals),
                        ),
                        out_shardings=state_sharding,
                    )
                    mix_history = _reuse_fringing_jit(
                        ("anderson", len(iterates), *kernel_key), mix_history
                    )
                    accelerated = mix_history(iterates, residuals)
            elif case.solver.coupling_acceleration == "aitken":
                if accepted_state_converged:
                    # Avoid reduction noise after settling while retaining a
                    # conservative, empirically monotone coupled acceleration.
                    accelerated = (
                        current_state
                        + ALEX_B2_SETTLED_RELAXATION * fixed_point_residual
                    )
                    previous_fixed_point_residual = None
                    fixed_point_relaxation = jnp.asarray(1.0, dtype=u.dtype)
                elif previous_fixed_point_residual is not None:
                    fixed_point_relaxation = aitken_relaxation(
                        previous_fixed_point_residual,
                        fixed_point_residual,
                        fixed_point_relaxation,
                        min_relaxation=case.solver.coupling_min_relaxation,
                        max_relaxation=case.solver.coupling_max_relaxation,
                    )
                    accelerated = (
                        current_state + fixed_point_relaxation * fixed_point_residual
                    )
                else:
                    accelerated = mapped_state
                if not accepted_state_converged:
                    previous_fixed_point_residual = fixed_point_residual
            else:
                accelerated = mapped_state
            u, v, w, phi = unscaled_state(accelerated)
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
            ),
        )
        if converged:
            break

    final_step_residual = residual_by_step[-1] if residual_by_step else 0.0
    residual = jnp.full((nx,), final_step_residual, dtype=float)
    fluid_area = jnp.maximum(
        jnp.sum(jnp.where(fluid_mask, cell_area, 0.0), axis=(1, 2)), 1.0e-20
    )
    volumetric_flow_rate = jnp.sum(
        jnp.where(fluid_mask, u * cell_area, 0.0), axis=(1, 2)
    )
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
        boundary_current_residual = jnp.abs(
            jnp.sum(div_j * cell_area, axis=(1, 2)) * dx
        )
    else:
        div_j, wall_current_leakage, boundary_current_residual = (
            _conservative_current_diagnostics_3d(
                sigma,
                phi,
                uxb_x,
                uxb_y,
                uxb_z,
                dx=dx,
                dy=dy,
                dz=dz,
            )
        )
    current_scaled_pressure_proxy = jnp.max(jnp.abs(jy), axis=(1, 2)) * jnp.maximum(
        jnp.max(jnp.abs(bx) + jnp.abs(by) + jnp.abs(bz), axis=(1, 2)), 1.0e-12
    )
    charge_balance_residual = jnp.max(jnp.abs(div_j), axis=(1, 2))
    residual = jnp.nan_to_num(
        residual, nan=scalar_limit, posinf=scalar_limit, neginf=scalar_limit
    )
    volumetric_flow_rate = jnp.nan_to_num(volumetric_flow_rate)
    mean_velocity = jnp.nan_to_num(mean_velocity)
    axial_current = jnp.nan_to_num(
        axial_current, nan=scalar_limit, posinf=scalar_limit, neginf=scalar_limit
    )
    wall_current_leakage = jnp.nan_to_num(
        wall_current_leakage, nan=scalar_limit, posinf=scalar_limit, neginf=scalar_limit
    )
    current_scaled_pressure_proxy = jnp.nan_to_num(
        current_scaled_pressure_proxy,
        nan=scalar_limit,
        posinf=scalar_limit,
        neginf=scalar_limit,
    )
    charge_balance_residual = jnp.nan_to_num(
        charge_balance_residual,
        nan=scalar_limit,
        posinf=scalar_limit,
        neginf=scalar_limit,
    )
    boundary_current_residual = jnp.nan_to_num(
        boundary_current_residual,
        nan=scalar_limit,
        posinf=scalar_limit,
        neginf=scalar_limit,
    )
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
        current_scaled_pressure_proxy=jnp.asarray(
            current_scaled_pressure_proxy, dtype=float
        ),
        charge_balance_residual=jnp.asarray(charge_balance_residual, dtype=float),
        boundary_current_residual=jnp.asarray(boundary_current_residual, dtype=float),
        geometry_kind=case.geometry.kind,
        solver_kind=case.solver.kind,
        axial_pressure_loss_gradient=jnp.asarray(
            axial_pressure_loss_gradient, dtype=float
        ),
        transverse_pressure_difference=jnp.asarray(
            transverse_pressure_difference, dtype=float
        ),
        iteration_residual_history=jnp.asarray(residual_by_step, dtype=float),
        iteration_component_residual_history=jnp.asarray(
            component_residual_by_step, dtype=float
        ).reshape((-1, 6)),
        iteration_pressure_residual_history=jnp.asarray(
            pressure_residual_by_step, dtype=float
        ),
        iteration_electric_linear_history=jnp.asarray(
            electric_linear_by_step, dtype=float
        ).reshape((-1, 6)),
        iteration_potential_residual_history=jnp.asarray(
            potential_residual_by_step, dtype=float
        ),
    )


def validate_extruded_inductionless_solution(
    bundle: ExtrudedFieldBundle,
    *,
    station_history: list[dict[str, float]]
    | tuple[dict[str, float], ...]
    | None = None,
) -> ExtrudedInductionlessValidation:
    field_scale = jnp.asarray(bundle.field_scale, dtype=float)
    mean_velocity = jnp.asarray(bundle.mean_velocity, dtype=float)
    volumetric_flow_rate = jnp.asarray(bundle.volumetric_flow_rate, dtype=float)
    axial_current = jnp.asarray(bundle.axial_current, dtype=float)
    wall_current_leakage = jnp.asarray(bundle.wall_current_leakage, dtype=float)
    residual = jnp.asarray(bundle.residual, dtype=float)
    charge_balance_residual = jnp.asarray(bundle.charge_balance_residual, dtype=float)
    boundary_current_residual = jnp.asarray(
        bundle.boundary_current_residual, dtype=float
    )
    peak_velocity = jnp.max(jnp.abs(bundle.u), axis=(1, 2))
    pressure_span = jnp.max(bundle.p, axis=(1, 2)) - jnp.min(bundle.p, axis=(1, 2))
    correlation = _safe_correlation(field_scale, mean_velocity)
    axial_current_mirror_residual = _mirror_residual(axial_current, odd=True)
    pressure_span_mirror_residual = _mirror_residual(pressure_span, odd=False)
    center_axial_current = _center_station_value(axial_current)
    center_pressure_span = _center_station_value(pressure_span)
    component_history = jnp.asarray(
        getattr(bundle, "iteration_component_residual_history", jnp.zeros((0, 6)))
    )
    max_divergence_residual = (
        float(component_history[-1, 3])
        if component_history.ndim == 2 and component_history.shape[0]
        else 0.0
    )
    return ExtrudedInductionlessValidation(
        station_count=int(bundle.x.shape[0]),
        max_residual=float(jnp.max(jnp.abs(residual))) if residual.size else 0.0,
        max_charge_balance_residual=float(jnp.max(jnp.abs(charge_balance_residual)))
        if charge_balance_residual.size
        else 0.0,
        mean_velocity_span=float(jnp.max(mean_velocity) - jnp.min(mean_velocity))
        if mean_velocity.size
        else 0.0,
        volumetric_flow_rate_span=float(
            jnp.max(volumetric_flow_rate) - jnp.min(volumetric_flow_rate)
        )
        if volumetric_flow_rate.size
        else 0.0,
        axial_current_span=float(jnp.max(axial_current) - jnp.min(axial_current))
        if axial_current.size
        else 0.0,
        axial_current_mirror_residual=axial_current_mirror_residual,
        max_wall_current_leakage=float(jnp.max(jnp.abs(wall_current_leakage)))
        if wall_current_leakage.size
        else 0.0,
        net_boundary_current_residual=float(jnp.max(jnp.abs(boundary_current_residual)))
        if boundary_current_residual.size
        else 0.0,
        field_mean_velocity_correlation=correlation,
        peak_velocity_span=float(jnp.max(peak_velocity) - jnp.min(peak_velocity))
        if peak_velocity.size
        else 0.0,
        pressure_span_range=float(jnp.max(pressure_span) - jnp.min(pressure_span))
        if pressure_span.size
        else 0.0,
        pressure_span_mirror_residual=pressure_span_mirror_residual,
        center_axial_current=center_axial_current,
        center_pressure_span=center_pressure_span,
        max_divergence_residual=max_divergence_residual,
    )


def solve_extruded_inductionless(
    problem: ExtrudedInductionlessProblem,
    *,
    solver=solve_steady,
    initial_bundle: ExtrudedFieldBundle | None = None,
    num_devices: int | None = None,
    progress_callback: Callable[[ExtrudedIterationProgress], None] | None = None,
    checkpoint_interval: int | None = None,
) -> ExtrudedInductionlessSolution:
    """Solve an extruded problem with optional sharding and progress checkpoints.

    ``progress_callback`` is called after every outer iteration. Its progress
    object contains a restart-capable bundle at ``checkpoint_interval`` steps
    and on convergence; no checkpoint arrays are materialized otherwise.
    """

    if checkpoint_interval is not None and checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be positive")

    if problem.case.geometry.kind in {
        "rect_duct",
        "layered_duct",
        "pipe_ogrid",
        "bent_pipe",
    }:
        projection_kwargs = {
            "initial_bundle": initial_bundle,
            "progress_callback": progress_callback,
            "checkpoint_interval": checkpoint_interval,
        }
        if num_devices is not None:
            projection_kwargs["num_devices"] = num_devices
        bundle = _solve_extruded_projection(problem, **projection_kwargs)
        station_history = _bundle_station_history(bundle)
    else:
        station_history = run_fringing_station_sweep(
            problem.case, problem.profile, solver=solver
        )
        bundle = run_extruded_inductionless_slice(
            problem.case, problem.profile, solver=solver
        )
    validation = validate_extruded_inductionless_solution(
        bundle, station_history=station_history
    )
    return ExtrudedInductionlessSolution(
        problem=problem,
        bundle=bundle,
        station_history=tuple(station_history),
        validation=validation,
    )

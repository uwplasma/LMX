from __future__ import annotations

from dataclasses import dataclass, replace

import jax.numpy as jnp
import numpy as np

try:
    from scipy import sparse
    from scipy.sparse.linalg import spsolve as sparse_spsolve
except Exception:  # pragma: no cover - SciPy should be present in shipped environments.
    sparse = None
    sparse_spsolve = None

from .cases import _ha_to_b, make_shercliff_case
from .core import Solution
from .mesh import generate_layered_duct_mesh, generate_pipe_ogrid_mesh, generate_rect_duct_mesh
from .physics import build_material_fields
from .specs import BoundaryCondition, CaseSpec, GeometrySpec, MagneticFieldSpec, OutputSpec, RegionSpec, SolverConfig, TimeStepperConfig
from .solvers import solve_steady
from .validation import validation_summary


@dataclass(frozen=True)
class FringingProfile:
    x: jnp.ndarray
    field_scale: jnp.ndarray
    axis: str


@dataclass(frozen=True)
class ExtrudedFieldBundle:
    x: jnp.ndarray
    y: jnp.ndarray
    z: jnp.ndarray
    field_scale: jnp.ndarray
    u: jnp.ndarray
    v: jnp.ndarray
    w: jnp.ndarray
    p: jnp.ndarray
    phi: jnp.ndarray
    jx: jnp.ndarray
    jy: jnp.ndarray
    jz: jnp.ndarray
    lorentz_x: jnp.ndarray
    lorentz_y: jnp.ndarray
    lorentz_z: jnp.ndarray
    residual: jnp.ndarray
    volumetric_flow_rate: jnp.ndarray
    mean_velocity: jnp.ndarray
    axial_current: jnp.ndarray
    wall_current_leakage: jnp.ndarray
    current_scaled_pressure_proxy: jnp.ndarray
    charge_balance_residual: jnp.ndarray
    boundary_current_residual: jnp.ndarray
    geometry_kind: str
    solver_kind: str


@dataclass(frozen=True)
class ExtrudedInductionlessProblem:
    case: CaseSpec
    profile: FringingProfile


@dataclass(frozen=True)
class ExtrudedInductionlessValidation:
    station_count: int
    max_residual: float
    max_charge_balance_residual: float
    mean_velocity_span: float
    volumetric_flow_rate_span: float
    axial_current_span: float
    max_wall_current_leakage: float
    net_boundary_current_residual: float
    field_mean_velocity_correlation: float
    peak_velocity_span: float = 0.0
    pressure_span_range: float = 0.0


@dataclass(frozen=True)
class ExtrudedInductionlessSolution:
    problem: ExtrudedInductionlessProblem
    bundle: ExtrudedFieldBundle
    station_history: tuple[dict[str, float], ...]
    validation: ExtrudedInductionlessValidation


def _broadcast_station_profile(values: jnp.ndarray, ny: int, nz: int) -> jnp.ndarray:
    return jnp.broadcast_to(jnp.asarray(values, dtype=float)[:, None, None], (values.shape[0], ny, nz))


def _broadcast_cross_section(values: jnp.ndarray, nx: int) -> jnp.ndarray:
    return jnp.broadcast_to(jnp.asarray(values, dtype=float)[None, :, :], (nx,) + tuple(values.shape))


def _harmonic_mean(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    denom = jnp.maximum(a + b, 1.0e-20)
    return 2.0 * a * b / denom


def _neighbor_fields(field: jnp.ndarray, *, mode_x: str, mode_y: str, mode_z: str) -> tuple[jnp.ndarray, ...]:
    x_west = jnp.concatenate([field[:1], field[:-1]], axis=0) if mode_x == "neumann" else jnp.concatenate(
        [jnp.zeros_like(field[:1]), field[:-1]], axis=0
    )
    x_east = jnp.concatenate([field[1:], field[-1:]], axis=0) if mode_x == "neumann" else jnp.concatenate(
        [field[1:], jnp.zeros_like(field[-1:])], axis=0
    )
    y_south = jnp.concatenate([field[:, :1, :], field[:, :-1, :]], axis=1) if mode_y == "neumann" else jnp.concatenate(
        [jnp.zeros_like(field[:, :1, :]), field[:, :-1, :]], axis=1
    )
    y_north = jnp.concatenate([field[:, 1:, :], field[:, -1:, :]], axis=1) if mode_y == "neumann" else jnp.concatenate(
        [field[:, 1:, :], jnp.zeros_like(field[:, -1:, :])], axis=1
    )
    z_bottom = jnp.concatenate([field[:, :, :1], field[:, :, :-1]], axis=2) if mode_z == "neumann" else jnp.concatenate(
        [jnp.zeros_like(field[:, :, :1]), field[:, :, :-1]], axis=2
    )
    z_top = jnp.concatenate([field[:, :, 1:], field[:, :, -1:]], axis=2) if mode_z == "neumann" else jnp.concatenate(
        [field[:, :, 1:], jnp.zeros_like(field[:, :, -1:])], axis=2
    )
    return x_west, x_east, y_south, y_north, z_bottom, z_top


def _laplacian_3d(
    field: jnp.ndarray,
    *,
    dx: float,
    dy: float,
    dz: float,
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
    return (
        (x_west - 2.0 * field + x_east) / max(dx**2, 1.0e-12)
        + (y_south - 2.0 * field + y_north) / max(dy**2, 1.0e-12)
        + (z_bottom - 2.0 * field + z_top) / max(dz**2, 1.0e-12)
    )


def _gradient_3d(field: jnp.ndarray, *, dx: float, dy: float, dz: float) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    x_west, x_east, y_south, y_north, z_bottom, z_top = _neighbor_fields(
        field,
        mode_x="neumann",
        mode_y="neumann",
        mode_z="neumann",
    )
    d_dx = (x_east - x_west) / max(2.0 * dx, 1.0e-12)
    d_dy = (y_north - y_south) / max(2.0 * dy, 1.0e-12)
    d_dz = (z_top - z_bottom) / max(2.0 * dz, 1.0e-12)
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


def _poisson_jacobi_3d(
    rhs: jnp.ndarray,
    *,
    dx: float,
    dy: float,
    dz: float,
    iterations: int,
    tolerance: float,
) -> tuple[jnp.ndarray, float, int, float]:
    rhs_compatible = rhs - jnp.mean(rhs)
    diagonal = 2.0 / max(dx**2, 1.0e-12) + 2.0 / max(dy**2, 1.0e-12) + 2.0 / max(dz**2, 1.0e-12)
    field = jnp.zeros_like(rhs_compatible)
    initial_residual = float(jnp.max(jnp.abs(_laplacian_3d(field, dx=dx, dy=dy, dz=dz, mode_x="neumann", mode_y="neumann", mode_z="neumann") - rhs_compatible)))
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
    dy: float,
    dz: float,
    iterations: int,
    tolerance: float,
    initial_field: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, float, int, float]:
    weights = jnp.maximum(conductivity, 1.0e-20)
    rhs_compatible = rhs - jnp.sum(rhs * weights) / jnp.sum(weights)
    sigma_x_w = jnp.concatenate([conductivity[:1], _harmonic_mean(conductivity[1:], conductivity[:-1])], axis=0)
    sigma_x_e = jnp.concatenate([_harmonic_mean(conductivity[1:], conductivity[:-1]), conductivity[-1:]], axis=0)
    sigma_y_s = jnp.concatenate([conductivity[:, :1, :], _harmonic_mean(conductivity[:, 1:, :], conductivity[:, :-1, :])], axis=1)
    sigma_y_n = jnp.concatenate([_harmonic_mean(conductivity[:, 1:, :], conductivity[:, :-1, :]), conductivity[:, -1:, :]], axis=1)
    sigma_z_b = jnp.concatenate([conductivity[:, :, :1], _harmonic_mean(conductivity[:, :, 1:], conductivity[:, :, :-1])], axis=2)
    sigma_z_t = jnp.concatenate([_harmonic_mean(conductivity[:, :, 1:], conductivity[:, :, :-1]), conductivity[:, :, -1:]], axis=2)
    coef_x_w = sigma_x_w / max(dx**2, 1.0e-12)
    coef_x_e = sigma_x_e / max(dx**2, 1.0e-12)
    coef_y_s = sigma_y_s / max(dy**2, 1.0e-12)
    coef_y_n = sigma_y_n / max(dy**2, 1.0e-12)
    coef_z_b = sigma_z_b / max(dz**2, 1.0e-12)
    coef_z_t = sigma_z_t / max(dz**2, 1.0e-12)
    diagonal = coef_x_w + coef_x_e + coef_y_s + coef_y_n + coef_z_b + coef_z_t
    diagonal = jnp.maximum(diagonal, 1.0e-12)
    if initial_field is None:
        field = jnp.zeros_like(rhs_compatible)
    else:
        field = jnp.nan_to_num(jnp.asarray(initial_field, dtype=rhs_compatible.dtype))
        field = field - jnp.sum(field * weights) / jnp.sum(weights)
    initial_residual = float(jnp.max(jnp.abs(_variable_coefficient_residual_3d(field, rhs_compatible, conductivity, dx=dx, dy=dy, dz=dz))))
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
            rhs_compatible
            + coef_x_w * x_west
            + coef_x_e * x_east
            + coef_y_s * y_south
            + coef_y_n * y_north
            + coef_z_b * z_bottom
            + coef_z_t * z_top
        ) / diagonal
        field = jnp.nan_to_num(updated - jnp.sum(updated * weights) / jnp.sum(weights))
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
    dy: float,
    dz: float,
) -> jnp.ndarray:
    x_west, x_east, y_south, y_north, z_bottom, z_top = _neighbor_fields(
        field,
        mode_x="neumann",
        mode_y="neumann",
        mode_z="neumann",
    )
    sigma_x_w = jnp.concatenate([conductivity[:1], _harmonic_mean(conductivity[1:], conductivity[:-1])], axis=0)
    sigma_x_e = jnp.concatenate([_harmonic_mean(conductivity[1:], conductivity[:-1]), conductivity[-1:]], axis=0)
    sigma_y_s = jnp.concatenate([conductivity[:, :1, :], _harmonic_mean(conductivity[:, 1:, :], conductivity[:, :-1, :])], axis=1)
    sigma_y_n = jnp.concatenate([_harmonic_mean(conductivity[:, 1:, :], conductivity[:, :-1, :]), conductivity[:, -1:, :]], axis=1)
    sigma_z_b = jnp.concatenate([conductivity[:, :, :1], _harmonic_mean(conductivity[:, :, 1:], conductivity[:, :, :-1])], axis=2)
    sigma_z_t = jnp.concatenate([_harmonic_mean(conductivity[:, :, 1:], conductivity[:, :, :-1]), conductivity[:, :, -1:]], axis=2)
    operator = (
        sigma_x_w * (x_west - field) / max(dx**2, 1.0e-12)
        + sigma_x_e * (x_east - field) / max(dx**2, 1.0e-12)
        + sigma_y_s * (y_south - field) / max(dy**2, 1.0e-12)
        + sigma_y_n * (y_north - field) / max(dy**2, 1.0e-12)
        + sigma_z_b * (z_bottom - field) / max(dz**2, 1.0e-12)
        + sigma_z_t * (z_top - field) / max(dz**2, 1.0e-12)
    )
    return operator - rhs


def _variable_coefficient_poisson_sparse_3d(
    rhs: jnp.ndarray,
    conductivity: jnp.ndarray,
    *,
    dx: float,
    dy: float,
    dz: float,
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
    weights = np.maximum(conductivity_np, 1.0e-20)
    rhs_compatible = rhs_np - np.sum(rhs_np * weights) / np.sum(weights)
    nx, ny, nz = conductivity_np.shape
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
                    coef = float(_harmonic_mean(conductivity_np[i, j, k], conductivity_np[i - 1, j, k]) / max(dx**2, 1.0e-12))
                    diag += coef
                    rows.append(idx)
                    cols.append(flat_index(i - 1, j, k))
                    data.append(-coef)
                if i + 1 < nx:
                    coef = float(_harmonic_mean(conductivity_np[i, j, k], conductivity_np[i + 1, j, k]) / max(dx**2, 1.0e-12))
                    diag += coef
                    rows.append(idx)
                    cols.append(flat_index(i + 1, j, k))
                    data.append(-coef)
                if j > 0:
                    coef = float(_harmonic_mean(conductivity_np[i, j, k], conductivity_np[i, j - 1, k]) / max(dy**2, 1.0e-12))
                    diag += coef
                    rows.append(idx)
                    cols.append(flat_index(i, j - 1, k))
                    data.append(-coef)
                if j + 1 < ny:
                    coef = float(_harmonic_mean(conductivity_np[i, j, k], conductivity_np[i, j + 1, k]) / max(dy**2, 1.0e-12))
                    diag += coef
                    rows.append(idx)
                    cols.append(flat_index(i, j + 1, k))
                    data.append(-coef)
                if k > 0:
                    coef = float(_harmonic_mean(conductivity_np[i, j, k], conductivity_np[i, j, k - 1]) / max(dz**2, 1.0e-12))
                    diag += coef
                    rows.append(idx)
                    cols.append(flat_index(i, j, k - 1))
                    data.append(-coef)
                if k + 1 < nz:
                    coef = float(_harmonic_mean(conductivity_np[i, j, k], conductivity_np[i, j, k + 1]) / max(dz**2, 1.0e-12))
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
        x0 = initial_np - np.sum(initial_np.reshape(rhs_compatible.shape) * weights) / np.sum(weights)
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
    weights_sum = np.sum(weights)
    field = field - np.sum(field * weights) / weights_sum
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


def _safe_correlation(x: jnp.ndarray, y: jnp.ndarray) -> float:
    centered_x = x - jnp.mean(x)
    centered_y = y - jnp.mean(y)
    denom = jnp.sqrt(jnp.sum(centered_x**2) * jnp.sum(centered_y**2))
    return float(jnp.where(denom > 0.0, jnp.sum(centered_x * centered_y) / denom, 0.0))


def _clip_state(field: jnp.ndarray, limit: float) -> jnp.ndarray:
    return jnp.clip(jnp.nan_to_num(field), -limit, limit)


def _cross_section_mesh(case: CaseSpec):
    geometry = case.geometry
    if geometry.kind == "rect_duct":
        return generate_rect_duct_mesh(
            width=geometry.width,
            height=geometry.height,
            length=geometry.length,
            nx=geometry.nx,
            ny=geometry.ny,
            nz=geometry.nz,
        )
    if geometry.kind == "layered_duct":
        return generate_layered_duct_mesh(
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
        return generate_pipe_ogrid_mesh(
            radius=geometry.radius or (0.5 * geometry.width),
            length=geometry.length,
            nx=geometry.nx,
            nr=geometry.nr or geometry.ny,
            ntheta=geometry.ntheta or geometry.nz,
        )
    raise ValueError(f"Unsupported extruded geometry {geometry.kind!r}")


def _bundle_station_history(bundle: ExtrudedFieldBundle) -> tuple[dict[str, float], ...]:
    return tuple(
        {
            "x": float(bundle.x[index]),
            "field_scale": float(bundle.field_scale[index]),
            "u_max": float(jnp.max(jnp.abs(bundle.u[index]))),
            "mean_velocity": float(bundle.mean_velocity[index]),
            "volumetric_flow_rate": float(bundle.volumetric_flow_rate[index]),
            "axial_current": float(bundle.axial_current[index]),
            "wall_current_leakage": float(bundle.wall_current_leakage[index]),
            "current_scaled_pressure_proxy": float(bundle.current_scaled_pressure_proxy[index]),
            "pressure_span": float(jnp.max(bundle.p[index]) - jnp.min(bundle.p[index])),
            "residual": float(bundle.residual[index]),
            "charge_balance_residual": float(bundle.charge_balance_residual[index]),
            "boundary_current_residual": float(bundle.boundary_current_residual[index]),
        }
        for index in range(bundle.x.shape[0])
    )


def _net_boundary_current_residual(
    jx: jnp.ndarray,
    jy: jnp.ndarray,
    jz: jnp.ndarray,
    *,
    dx: float,
    dy: float,
    dz: float,
) -> float:
    yz_area = dy * dz
    xz_area = dx * dz
    xy_area = dx * dy
    inlet_flux = -jnp.sum(jx[0, :, :]) * yz_area
    outlet_flux = jnp.sum(jx[-1, :, :]) * yz_area
    south_flux = -jnp.sum(jy[:, 0, :]) * xz_area
    north_flux = jnp.sum(jy[:, -1, :]) * xz_area
    bottom_flux = -jnp.sum(jz[:, :, 0]) * xy_area
    top_flux = jnp.sum(jz[:, :, -1]) * xy_area
    return float(jnp.abs(inlet_flux + outlet_flux + south_flux + north_flux + bottom_flux + top_flux))


def _conservative_current_fluxes_3d(
    sigma: jnp.ndarray,
    phi: jnp.ndarray,
    uxb_x: jnp.ndarray,
    uxb_y: jnp.ndarray,
    uxb_z: jnp.ndarray,
    *,
    dx: float,
    dy: float,
    dz: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    nx, ny, nz = phi.shape
    fx = jnp.zeros((nx + 1, ny, nz), dtype=phi.dtype)
    fy = jnp.zeros((nx, ny + 1, nz), dtype=phi.dtype)
    fz = jnp.zeros((nx, ny, nz + 1), dtype=phi.dtype)

    sigma_x = _harmonic_mean(sigma[1:], sigma[:-1])
    phi_grad_x = (phi[1:] - phi[:-1]) / max(dx, 1.0e-12)
    uxb_face_x = 0.5 * (uxb_x[1:] + uxb_x[:-1])
    fx = fx.at[1:-1].set(sigma_x * (-phi_grad_x + uxb_face_x))

    sigma_y = _harmonic_mean(sigma[:, 1:, :], sigma[:, :-1, :])
    phi_grad_y = (phi[:, 1:, :] - phi[:, :-1, :]) / max(dy, 1.0e-12)
    uxb_face_y = 0.5 * (uxb_y[:, 1:, :] + uxb_y[:, :-1, :])
    fy = fy.at[:, 1:-1, :].set(sigma_y * (-phi_grad_y + uxb_face_y))

    sigma_z = _harmonic_mean(sigma[:, :, 1:], sigma[:, :, :-1])
    phi_grad_z = (phi[:, :, 1:] - phi[:, :, :-1]) / max(dz, 1.0e-12)
    uxb_face_z = 0.5 * (uxb_z[:, :, 1:] + uxb_z[:, :, :-1])
    fz = fz.at[:, :, 1:-1].set(sigma_z * (-phi_grad_z + uxb_face_z))
    return fx, fy, fz


def _conservative_current_diagnostics_3d(
    sigma: jnp.ndarray,
    phi: jnp.ndarray,
    uxb_x: jnp.ndarray,
    uxb_y: jnp.ndarray,
    uxb_z: jnp.ndarray,
    *,
    dx: float,
    dy: float,
    dz: float,
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
    )
    div_j = (
        (fx[1:] - fx[:-1]) / max(dx, 1.0e-12)
        + (fy[:, 1:, :] - fy[:, :-1, :]) / max(dy, 1.0e-12)
        + (fz[:, :, 1:] - fz[:, :, :-1]) / max(dz, 1.0e-12)
    )
    wall_leakage = (
        jnp.sum(jnp.abs(fy[:, 0, :]), axis=1) * dx * dz
        + jnp.sum(jnp.abs(fy[:, -1, :]), axis=1) * dx * dz
        + jnp.sum(jnp.abs(fz[:, :, 0]), axis=1) * dx * dy
        + jnp.sum(jnp.abs(fz[:, :, -1]), axis=1) * dx * dy
    )
    boundary_residual = jnp.abs(
        -jnp.sum(fx[0], axis=(0, 1)) * dy * dz
        + jnp.sum(fx[-1], axis=(0, 1)) * dy * dz
        - jnp.sum(fy[:, 0, :], axis=1) * dx * dz
        + jnp.sum(fy[:, -1, :], axis=1) * dx * dz
        - jnp.sum(fz[:, :, 0], axis=1) * dx * dy
        + jnp.sum(fz[:, :, -1], axis=1) * dx * dy
    )
    return div_j, wall_leakage, boundary_residual


def _conservative_emf_rhs_3d(
    sigma: jnp.ndarray,
    uxb_x: jnp.ndarray,
    uxb_y: jnp.ndarray,
    uxb_z: jnp.ndarray,
    *,
    dx: float,
    dy: float,
    dz: float,
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
    )
    return (
        (fx[1:] - fx[:-1]) / max(dx, 1.0e-12)
        + (fy[:, 1:, :] - fy[:, :-1, :]) / max(dy, 1.0e-12)
        + (fz[:, :, 1:] - fz[:, :, :-1]) / max(dz, 1.0e-12)
    )


def _pipe_theta_neighbors(field: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    theta_prev = jnp.concatenate([field[:, :, -1:], field[:, :, :-1]], axis=2)
    theta_next = jnp.concatenate([field[:, :, 1:], field[:, :, :1]], axis=2)
    return theta_prev, theta_next


def _pipe_gradient_3d(
    field: jnp.ndarray,
    *,
    dx: float,
    dr: float,
    dtheta: float,
    r: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    safe_r = jnp.maximum(r, 0.5 * dr)
    x_west = jnp.concatenate([field[:1], field[:-1]], axis=0)
    x_east = jnp.concatenate([field[1:], field[-1:]], axis=0)
    r_inner = jnp.concatenate([field[:, :1, :], field[:, :-1, :]], axis=1)
    r_outer = jnp.concatenate([field[:, 1:, :], field[:, -1:, :]], axis=1)
    theta_prev, theta_next = _pipe_theta_neighbors(field)
    d_dx = (x_east - x_west) / max(2.0 * dx, 1.0e-12)
    d_dr = (r_outer - r_inner) / max(2.0 * dr, 1.0e-12)
    d_dtheta = (theta_next - theta_prev) / jnp.maximum(2.0 * dtheta * safe_r, 1.0e-12)
    d_dr = d_dr.at[:, 0, :].set(0.0)
    d_dtheta = d_dtheta.at[:, 0, :].set(0.0)
    return d_dx, d_dr, d_dtheta


def _pipe_laplacian_3d(
    field: jnp.ndarray,
    *,
    dx: float,
    dr: float,
    dtheta: float,
    r: jnp.ndarray,
    outer_dirichlet: bool = True,
) -> jnp.ndarray:
    safe_r = jnp.maximum(r, 0.5 * dr)
    x_west = jnp.concatenate([field[:1], field[:-1]], axis=0)
    x_east = jnp.concatenate([field[1:], field[-1:]], axis=0)
    r_inner = jnp.concatenate([field[:, :1, :], field[:, :-1, :]], axis=1)
    outer_ghost = jnp.zeros_like(field[:, -1:, :]) if outer_dirichlet else field[:, -1:, :]
    r_outer = jnp.concatenate([field[:, 1:, :], outer_ghost], axis=1)
    theta_prev, theta_next = _pipe_theta_neighbors(field)
    dxx = (x_west - 2.0 * field + x_east) / max(dx**2, 1.0e-12)
    drr = (r_inner - 2.0 * field + r_outer) / max(dr**2, 1.0e-12)
    d_dr = (r_outer - r_inner) / max(2.0 * dr, 1.0e-12)
    dtheta2 = (theta_prev - 2.0 * field + theta_next) / jnp.maximum((safe_r**2) * dtheta**2, 1.0e-12)
    lap = dxx + drr + d_dr / safe_r + dtheta2
    return lap.at[:, 0, :].set(dxx[:, 0, :] + 2.0 * (field[:, 1, :] - field[:, 0, :]) / max(dr**2, 1.0e-12))


def _pipe_divergence_3d(
    jx: jnp.ndarray,
    jr: jnp.ndarray,
    jtheta: jnp.ndarray,
    *,
    dx: float,
    dr: float,
    dtheta: float,
    r: jnp.ndarray,
) -> jnp.ndarray:
    safe_r = jnp.maximum(r, 0.5 * dr)
    djx_dx = _pipe_gradient_3d(jx, dx=dx, dr=dr, dtheta=dtheta, r=r)[0]
    rjr = safe_r * jr
    rjr_inner = jnp.concatenate([rjr[:, :1, :], rjr[:, :-1, :]], axis=1)
    rjr_outer = jnp.concatenate([rjr[:, 1:, :], rjr[:, -1:, :]], axis=1)
    radial_term = (rjr_outer - rjr_inner) / jnp.maximum(2.0 * dr * safe_r, 1.0e-12)
    theta_prev, theta_next = _pipe_theta_neighbors(jtheta)
    theta_term = (theta_next - theta_prev) / jnp.maximum(2.0 * dtheta * safe_r, 1.0e-12)
    divergence = djx_dx + radial_term + theta_term
    return divergence.at[:, 0, :].set(djx_dx[:, 0, :] + 2.0 * jr[:, 1, :] / max(dr, 1.0e-12))


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
    initial_residual = float(jnp.max(jnp.abs(_pipe_laplacian_3d(field, dx=dx, dr=dr, dtheta=dtheta, r=r, outer_dirichlet=False) - rhs_compatible)))
    residual = initial_residual
    iteration_count = 0
    diagonal = 2.0 / max(dx**2, 1.0e-12) + 2.0 / max(dr**2, 1.0e-12) + 2.0 / jnp.maximum((safe_r**2) * dtheta**2, 1.0e-12)
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
        residual = float(jnp.max(jnp.abs(_pipe_laplacian_3d(field, dx=dx, dr=dr, dtheta=dtheta, r=r, outer_dirichlet=False) - rhs_compatible)))
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

    dr_centers = jnp.maximum(0.5 * (jnp.diff(r_faces)[1:] + jnp.diff(r_faces)[:-1]), 1.0e-12)
    sigma_r = _harmonic_mean(sigma[:, 1:, :], sigma[:, :-1, :])
    phi_grad_r = (phi[:, 1:, :] - phi[:, :-1, :]) / dr_centers[None, :, None]
    uxb_face_r = 0.5 * (uxb_r[:, 1:, :] + uxb_r[:, :-1, :])
    fr = fr.at[:, 1:-1, :].set(sigma_r * (-phi_grad_r + uxb_face_r))

    sigma_theta = _harmonic_mean(sigma, jnp.roll(sigma, -1, axis=2))
    phi_grad_theta = (jnp.roll(phi, -1, axis=2) - phi) / jnp.maximum(r_centers[None, :, None] * dtheta, 1.0e-12)
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
        r_faces[None, 1:, None] * fr[:, 1:, :] - r_faces[None, :-1, None] * fr[:, :-1, :]
    ) / jnp.maximum(r_centers[None, :, None] * dr[None, :, None], 1.0e-12)
    theta_term = (ftheta - jnp.roll(ftheta, 1, axis=2)) / jnp.maximum(r_centers[None, :, None] * dtheta, 1.0e-12)
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
        r_faces[None, 1:, None] * fr[:, 1:, :] - r_faces[None, :-1, None] * fr[:, :-1, :]
    ) / jnp.maximum(r_centers[None, :, None] * dr[None, :, None], 1.0e-12)
    theta_term = (ftheta - jnp.roll(ftheta, 1, axis=2)) / jnp.maximum(r_centers[None, :, None] * dtheta, 1.0e-12)
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
                    sigma_face = _harmonic_mean(jnp.asarray(sigma_cell), jnp.asarray(sigma_np[i - 1, j, k])).item()
                    coeff = sigma_face / max(dx**2, 1.0e-12)
                    diagonal += coeff
                    rows.append(row)
                    cols.append(flat(i - 1, j, k))
                    data.append(-coeff)
                if i < nx - 1:
                    sigma_face = _harmonic_mean(jnp.asarray(sigma_cell), jnp.asarray(sigma_np[i + 1, j, k])).item()
                    coeff = sigma_face / max(dx**2, 1.0e-12)
                    diagonal += coeff
                    rows.append(row)
                    cols.append(flat(i + 1, j, k))
                    data.append(-coeff)
                if j > 0:
                    sigma_face = _harmonic_mean(jnp.asarray(sigma_cell), jnp.asarray(sigma_np[i, j - 1, k])).item()
                    dr_face = max(0.5 * (dr_np[j - 1] + dr_np[j]), 1.0e-12)
                    coeff = r_faces_np[j] * sigma_face / max(r_centers_np[j] * dr_np[j] * dr_face, 1.0e-12)
                    diagonal += coeff
                    rows.append(row)
                    cols.append(flat(i, j - 1, k))
                    data.append(-coeff)
                if j < nr - 1:
                    sigma_face = _harmonic_mean(jnp.asarray(sigma_cell), jnp.asarray(sigma_np[i, j + 1, k])).item()
                    dr_face = max(0.5 * (dr_np[j] + dr_np[j + 1]), 1.0e-12)
                    coeff = r_faces_np[j + 1] * sigma_face / max(r_centers_np[j] * dr_np[j] * dr_face, 1.0e-12)
                    diagonal += coeff
                    rows.append(row)
                    cols.append(flat(i, j + 1, k))
                    data.append(-coeff)

                k_prev = (k - 1) % ntheta
                k_next = (k + 1) % ntheta
                sigma_prev = _harmonic_mean(jnp.asarray(sigma_cell), jnp.asarray(sigma_np[i, j, k_prev])).item()
                sigma_next = _harmonic_mean(jnp.asarray(sigma_cell), jnp.asarray(sigma_np[i, j, k_next])).item()
                theta_coeff_prev = sigma_prev / max(r_centers_np[j] ** 2 * dtheta**2, 1.0e-12)
                theta_coeff_next = sigma_next / max(r_centers_np[j] ** 2 * dtheta**2, 1.0e-12)
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
    residual_field = _pipe_conservative_current_diagnostics_3d(
        jnp.asarray(conductivity, dtype=field.dtype),
        field,
        jnp.zeros_like(field),
        jnp.zeros_like(field),
        jnp.zeros_like(field),
        dx=dx,
        r_faces=jnp.asarray(r_faces, dtype=field.dtype),
        r_centers=jnp.asarray(r_centers, dtype=field.dtype),
        dtheta=dtheta,
    )[0] - rhs
    residual = float(jnp.max(jnp.abs(residual_field)))
    return field, residual, 1, initial_residual


def _enforce_pipe_velocity_bc(u: jnp.ndarray, v: jnp.ndarray, w: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    if u.shape[1] > 1:
        u = u.at[:, 0, :].set(u[:, 1, :])
        w = w.at[:, 0, :].set(w[:, 1, :])
    v = v.at[:, 0, :].set(0.0)
    u = u.at[:, -1, :].set(0.0)
    v = v.at[:, -1, :].set(0.0)
    w = w.at[:, -1, :].set(0.0)
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


def clone_case_with_field(case: CaseSpec, *, axis: str, magnitude: float, suffix: str | None = None) -> CaseSpec:
    magnetic_field = replace(case.magnetic_field, kind="constant", value=_constant_field_on_axis(axis, magnitude))
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
    base_case = make_shercliff_case(ha=ha_peak, width=width, height=height, ny=ny, nz=nz)
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


def _station_case(base_case: CaseSpec, *, axis: str, magnitude: float, suffix: str) -> CaseSpec:
    station_case = clone_case_with_field(base_case, axis=axis, magnitude=magnitude, suffix=suffix)
    return replace(station_case, solver=replace(station_case.solver, kind="fully_developed_inductionless"))


def run_fringing_station_sweep(
    base_case: CaseSpec,
    profile: FringingProfile,
    *,
    solver=solve_steady,
) -> list[dict[str, float]]:
    if base_case.magnetic_field.kind != "constant" or base_case.magnetic_field.value is None:
        raise ValueError("Fringing station sweep requires a constant-field base case")

    base_magnitude = max(abs(float(component)) for component in base_case.magnetic_field.value)
    history: list[dict[str, float]] = []
    previous_state = None
    for index, (x_value, scale) in enumerate(zip(profile.x, profile.field_scale, strict=True)):
        station_case = _station_case(
            base_case,
            axis=profile.axis,
            magnitude=base_magnitude * float(scale),
            suffix=f"station{index:03d}",
        )
        solution: Solution = solver(station_case, initial_state=previous_state)
        metrics = validation_summary(solution, station_case.name, ha=base_case.geometry.target_ha)
        history.append(
            {
                "x": float(x_value),
                "field_scale": float(scale),
                "u_max": float(metrics["u_max"]),
                "mean_velocity": float(metrics["mean_velocity"]),
                "volumetric_flow_rate": float(metrics["volumetric_flow_rate"]),
                "current_scaled_pressure_proxy": float(metrics["current_scaled_pressure_proxy"]),
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
    if base_case.magnetic_field.kind != "constant" or base_case.magnetic_field.value is None:
        raise ValueError("Extruded fringing slice requires a constant-field base case")

    base_magnitude = max(abs(float(component)) for component in base_case.magnetic_field.value)
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
    lorentz_x = jnp.stack([solution.state.lorentz_x for solution in station_solutions], axis=0)
    residual = jnp.asarray([solution.state.residual for solution in station_solutions], dtype=float)
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


def _solve_extruded_projection(
    problem: ExtrudedInductionlessProblem,
    *,
    initial_bundle: ExtrudedFieldBundle | None = None,
) -> ExtrudedFieldBundle:
    case = problem.case
    mesh = _cross_section_mesh(case)
    if case.geometry.kind == "pipe_ogrid":
        x = jnp.asarray(mesh.x_centers, dtype=float)
        r_faces = jnp.asarray(mesh.y_faces, dtype=float)
        r = jnp.asarray(mesh.y_centers, dtype=float)
        theta = jnp.asarray(mesh.z_centers, dtype=float)
        nx, nr, ntheta = len(x), len(r), len(theta)
        dx = float(jnp.mean(mesh.dx))
        dr = float(jnp.mean(mesh.dy))
        dtheta = float(jnp.mean(mesh.dz))
        region = case.regions[0]
        sigma = jnp.full((nx, nr, ntheta), region.conductivity, dtype=float)
        rho = jnp.full((nx, nr, ntheta), region.density or 1.0, dtype=float)
        nu = jnp.full((nx, nr, ntheta), region.viscosity or 1.0, dtype=float)
        rr = jnp.broadcast_to(jnp.maximum(r[None, :, None], 0.5 * dr), (nx, nr, ntheta))
        theta_grid = jnp.broadcast_to(theta[None, None, :], (nx, nr, ntheta))
        forcing = float(case.forcing)
        field_scale = jnp.asarray(problem.profile.field_scale, dtype=float)
        base_field = case.magnetic_field.value or (0.0, 0.0, 0.0)
        bx = jnp.broadcast_to(field_scale[:, None, None] * float(base_field[0]), (nx, nr, ntheta))
        by = jnp.broadcast_to(field_scale[:, None, None] * float(base_field[1]), (nx, nr, ntheta))
        bz = jnp.broadcast_to(field_scale[:, None, None] * float(base_field[2]), (nx, nr, ntheta))
        br = by * jnp.cos(theta_grid) + bz * jnp.sin(theta_grid)
        btheta = -by * jnp.sin(theta_grid) + bz * jnp.cos(theta_grid)

        if initial_bundle is not None:
            if initial_bundle.u.shape != (nx, nr, ntheta):
                raise ValueError("Extruded restart bundle shape does not match the current mapped-pipe problem")
            u = jnp.asarray(initial_bundle.u, dtype=float)
            v = jnp.asarray(initial_bundle.v, dtype=float)
            w = jnp.asarray(initial_bundle.w, dtype=float)
            p = jnp.asarray(initial_bundle.p, dtype=float)
            phi = jnp.asarray(initial_bundle.phi, dtype=float)
        else:
            u = jnp.zeros((nx, nr, ntheta), dtype=float)
            v = jnp.zeros_like(u)
            w = jnp.zeros_like(u)
            p = jnp.zeros_like(u)
            phi = jnp.zeros_like(u)

        min_dr = float(jnp.min(mesh.dy))
        min_arc = float(jnp.min(jnp.maximum(r[1:], 0.5 * min_dr))) * dtheta if nr > 1 else max(float(r[0]) * dtheta, 0.5 * min_dr * dtheta)
        inverse_diffusive_scale = float(
            jnp.max(nu)
            * (
                1.0 / max(dx**2, 1.0e-12)
                + 1.0 / max(min_dr**2, 1.0e-12)
                + 1.0 / max(min_arc**2, 1.0e-12)
            )
        )
        stable_dt = 0.1 / max(inverse_diffusive_scale, 1.0e-12)
        dt = min(float(case.time_stepper.dt), stable_dt)
        outer_steps = max(2, min(case.time_stepper.max_steps, max(6, case.solver.coupling_iterations * 2)))
        poisson_iterations = min(case.time_stepper.potential_iterations, 80)
        poisson_tolerance = case.solver.coupling_tolerance
        velocity_limit = 5.0
        scalar_limit = 20.0
        residual_by_step: list[float] = []

        for _ in range(outer_steps):
            dphi_dx, dphi_dr, dphi_dtheta = _pipe_gradient_3d(phi, dx=dx, dr=dr, dtheta=dtheta, r=rr)
            uxb_x = v * btheta - w * br
            uxb_r = w * bx - u * btheta
            uxb_theta = u * br - v * bx
            jx = sigma * (-dphi_dx + uxb_x)
            jr = sigma * (-dphi_dr + uxb_r)
            jtheta = sigma * (-dphi_dtheta + uxb_theta)
            lorentz_x = jr * btheta - jtheta * br
            lorentz_r = jtheta * bx - jx * btheta
            lorentz_theta = jx * br - jr * bx

            dp_dx, dp_dr, dp_dtheta = _pipe_gradient_3d(p, dx=dx, dr=dr, dtheta=dtheta, r=rr)
            u_star = u + dt * (_pipe_laplacian_3d(u, dx=dx, dr=dr, dtheta=dtheta, r=rr) * nu + forcing / rho + lorentz_x / rho - dp_dx / rho)
            v_star = v + dt * (_pipe_laplacian_3d(v, dx=dx, dr=dr, dtheta=dtheta, r=rr) * nu + lorentz_r / rho - dp_dr / rho)
            w_star = w + dt * (_pipe_laplacian_3d(w, dx=dx, dr=dr, dtheta=dtheta, r=rr) * nu + lorentz_theta / rho - dp_dtheta / rho)
            u_star = _clip_state(u_star, velocity_limit)
            v_star = _clip_state(v_star, velocity_limit)
            w_star = _clip_state(w_star, velocity_limit)
            u_star, v_star, w_star = _enforce_pipe_velocity_bc(u_star, v_star, w_star)

            divergence = _pipe_divergence_3d(u_star, v_star, w_star, dx=dx, dr=dr, dtheta=dtheta, r=rr)
            p_corr, _, _, _ = _pipe_poisson_jacobi_3d(
                (rho / max(dt, 1.0e-12)) * divergence,
                dx=dx,
                dr=dr,
                dtheta=dtheta,
                r=rr,
                iterations=poisson_iterations,
                tolerance=poisson_tolerance,
            )
            p_corr = _clip_state(p_corr, scalar_limit)
            dpc_dx, dpc_dr, dpc_dtheta = _pipe_gradient_3d(p_corr, dx=dx, dr=dr, dtheta=dtheta, r=rr)
            u_next = _clip_state(u_star - (dt / rho) * dpc_dx, velocity_limit)
            v_next = _clip_state(v_star - (dt / rho) * dpc_dr, velocity_limit)
            w_next = _clip_state(w_star - (dt / rho) * dpc_dtheta, velocity_limit)
            u_next, v_next, w_next = _enforce_pipe_velocity_bc(u_next, v_next, w_next)
            p = _clip_state(p + p_corr, scalar_limit)

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
            phi, _, _, _ = _pipe_poisson_sparse_3d(
                emf_rhs,
                sigma,
                dx=dx,
                r_faces=r_faces,
                r_centers=r,
                dtheta=dtheta,
                iterations=poisson_iterations,
                tolerance=poisson_tolerance,
                initial_field=phi,
            )
            phi = _clip_state(phi, scalar_limit)

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
            update_residual = max(
                float(jnp.max(jnp.abs(u_next - u))),
                float(jnp.max(jnp.abs(v_next - v))),
                float(jnp.max(jnp.abs(w_next - w))),
                float(jnp.max(jnp.abs(divergence))),
            )
            charge_balance = float(jnp.max(jnp.abs(div_j)))
            residual_by_step.append(update_residual)
            u, v, w = u_next, v_next, w_next
            if update_residual <= case.solver.coupling_tolerance and charge_balance <= max(1.0e-6, case.solver.coupling_tolerance):
                break

        final_step_residual = residual_by_step[-1] if residual_by_step else 0.0
        residual = jnp.full((nx,), final_step_residual, dtype=float)
        cell_area = rr * dr * dtheta
        cross_section_area = jnp.maximum(jnp.sum(cell_area, axis=(1, 2)), 1.0e-20)
        volumetric_flow_rate = jnp.sum(u * cell_area, axis=(1, 2))
        mean_velocity = volumetric_flow_rate / cross_section_area
        axial_current = jnp.sum(jx * cell_area, axis=(1, 2))
        _, wall_current_leakage, boundary_current_residual = _pipe_conservative_current_diagnostics_3d(
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
        current_scaled_pressure_proxy = jnp.max(jnp.abs(jr), axis=(1, 2)) * jnp.maximum(
            jnp.max(jnp.abs(bx) + jnp.abs(br) + jnp.abs(btheta), axis=(1, 2)),
            1.0e-12,
        )
        charge_balance_residual = jnp.max(
            jnp.abs(
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
                )[0]
            ),
            axis=(1, 2),
        )
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
        )
    materials = build_material_fields(case, mesh)
    x = jnp.asarray(mesh.x_centers, dtype=float)
    y = jnp.asarray(mesh.y_centers, dtype=float)
    z = jnp.asarray(mesh.z_centers, dtype=float)
    nx, ny, nz = len(x), len(y), len(z)
    dx = float(jnp.mean(mesh.dx))
    dy = float(jnp.mean(mesh.dy))
    dz = float(jnp.mean(mesh.dz))
    sigma = _broadcast_cross_section(materials.conductivity, nx)
    rho = _broadcast_cross_section(materials.density, nx)
    nu = _broadcast_cross_section(materials.viscosity, nx)
    fluid_mask = _broadcast_cross_section(materials.fluid_mask.astype(float), nx) > 0.5
    forcing = float(case.forcing)
    field_scale = jnp.asarray(problem.profile.field_scale, dtype=float)
    base_field = case.magnetic_field.value or (0.0, 0.0, 0.0)
    bx = _broadcast_station_profile(field_scale * float(base_field[0]), ny, nz)
    by = _broadcast_station_profile(field_scale * float(base_field[1]), ny, nz)
    bz = _broadcast_station_profile(field_scale * float(base_field[2]), ny, nz)

    if initial_bundle is not None:
        if initial_bundle.u.shape != (nx, ny, nz):
            raise ValueError("Extruded restart bundle shape does not match the current duct problem")
        u = jnp.asarray(initial_bundle.u, dtype=float)
        v = jnp.asarray(initial_bundle.v, dtype=float)
        w = jnp.asarray(initial_bundle.w, dtype=float)
        p = jnp.asarray(initial_bundle.p, dtype=float)
        phi = jnp.asarray(initial_bundle.phi, dtype=float)
    else:
        u = jnp.zeros((nx, ny, nz), dtype=float)
        v = jnp.zeros_like(u)
        w = jnp.zeros_like(u)
        p = jnp.zeros_like(u)
        phi = jnp.zeros_like(u)

    inverse_diffusive_scale = float(jnp.max(nu) * (1.0 / max(dx**2, 1.0e-12) + 1.0 / max(dy**2, 1.0e-12) + 1.0 / max(dz**2, 1.0e-12)))
    stable_dt = 0.2 / max(inverse_diffusive_scale, 1.0e-12)
    dt = min(float(case.time_stepper.dt), stable_dt)
    outer_steps = max(2, min(case.time_stepper.max_steps, max(6, case.solver.coupling_iterations * 2)))
    poisson_iterations = min(case.time_stepper.potential_iterations, 80)
    poisson_tolerance = case.solver.coupling_tolerance
    velocity_limit = 5.0
    scalar_limit = 20.0
    residual_by_step: list[float] = []

    for _ in range(outer_steps):
        dphi_dx, dphi_dy, dphi_dz = _gradient_3d(phi, dx=dx, dy=dy, dz=dz)
        uxb_x = v * bz - w * by
        uxb_y = w * bx - u * bz
        uxb_z = u * by - v * bx
        jx = sigma * (-dphi_dx + uxb_x)
        jy = sigma * (-dphi_dy + uxb_y)
        jz = sigma * (-dphi_dz + uxb_z)
        lorentz_x = jy * bz - jz * by
        lorentz_y = jz * bx - jx * bz
        lorentz_z = jx * by - jy * bx

        dp_dx, dp_dy, dp_dz = _gradient_3d(p, dx=dx, dy=dy, dz=dz)
        u_star = u + dt * (nu * _laplacian_3d(u, dx=dx, dy=dy, dz=dz) + forcing / rho + lorentz_x / rho - dp_dx / rho)
        v_star = v + dt * (nu * _laplacian_3d(v, dx=dx, dy=dy, dz=dz) + lorentz_y / rho - dp_dy / rho)
        w_star = w + dt * (nu * _laplacian_3d(w, dx=dx, dy=dy, dz=dz) + lorentz_z / rho - dp_dz / rho)
        u_star = _clip_state(u_star, velocity_limit)
        v_star = _clip_state(v_star, velocity_limit)
        w_star = _clip_state(w_star, velocity_limit)
        u_star = _enforce_velocity_bc_3d(u_star, fluid_mask)
        v_star = _enforce_velocity_bc_3d(v_star, fluid_mask)
        w_star = _enforce_velocity_bc_3d(w_star, fluid_mask)

        du_dx, du_dy, du_dz = _gradient_3d(u_star, dx=dx, dy=dy, dz=dz)
        dv_dx, dv_dy, dv_dz = _gradient_3d(v_star, dx=dx, dy=dy, dz=dz)
        dw_dx, dw_dy, dw_dz = _gradient_3d(w_star, dx=dx, dy=dy, dz=dz)
        divergence = jnp.where(fluid_mask, du_dx + dv_dy + dw_dz, 0.0)
        p_corr, _, _, _ = _poisson_jacobi_3d(
            (rho / max(dt, 1.0e-12)) * divergence,
            dx=dx,
            dy=dy,
            dz=dz,
            iterations=poisson_iterations,
            tolerance=poisson_tolerance,
        )
        p_corr = _clip_state(jnp.where(fluid_mask, p_corr, 0.0), scalar_limit)
        dpc_dx, dpc_dy, dpc_dz = _gradient_3d(p_corr, dx=dx, dy=dy, dz=dz)
        u_next = _enforce_velocity_bc_3d(u_star - (dt / rho) * dpc_dx, fluid_mask)
        v_next = _enforce_velocity_bc_3d(v_star - (dt / rho) * dpc_dy, fluid_mask)
        w_next = _enforce_velocity_bc_3d(w_star - (dt / rho) * dpc_dz, fluid_mask)
        u_next = _clip_state(u_next, velocity_limit)
        v_next = _clip_state(v_next, velocity_limit)
        w_next = _clip_state(w_next, velocity_limit)
        p = _clip_state(jnp.where(fluid_mask, p + p_corr, 0.0), scalar_limit)

        uxb_x = v_next * bz - w_next * by
        uxb_y = w_next * bx - u_next * bz
        uxb_z = u_next * by - v_next * bx
        emf_rhs = _conservative_emf_rhs_3d(
            sigma,
            uxb_x,
            uxb_y,
            uxb_z,
            dx=dx,
            dy=dy,
            dz=dz,
        )
        electric_solver = _variable_coefficient_poisson_sparse_3d if case.geometry.kind == "layered_duct" else _variable_coefficient_poisson_jacobi_3d
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
        phi = _clip_state(phi, scalar_limit)

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

        update_residual = max(
            float(jnp.max(jnp.abs(u_next - u))),
            float(jnp.max(jnp.abs(v_next - v))),
            float(jnp.max(jnp.abs(w_next - w))),
            float(jnp.max(jnp.abs(divergence))),
        )
        charge_balance = float(jnp.max(jnp.abs(div_j)))
        residual_by_step.append(update_residual)
        u, v, w = u_next, v_next, w_next
        if update_residual <= case.solver.coupling_tolerance and charge_balance <= max(1.0e-6, case.solver.coupling_tolerance):
            break

    final_step_residual = residual_by_step[-1] if residual_by_step else 0.0
    residual = jnp.full((nx,), final_step_residual, dtype=float)
    cell_area = _broadcast_cross_section(mesh.dy[:, None] * mesh.dz[None, :], nx)
    fluid_area = jnp.maximum(jnp.sum(jnp.where(fluid_mask, cell_area, 0.0), axis=(1, 2)), 1.0e-20)
    volumetric_flow_rate = jnp.sum(jnp.where(fluid_mask, u * cell_area, 0.0), axis=(1, 2))
    mean_velocity = volumetric_flow_rate / fluid_area
    axial_current = jnp.sum(jx * cell_area, axis=(1, 2))
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
    current_scaled_pressure_proxy = jnp.max(jnp.abs(jy), axis=(1, 2)) * jnp.maximum(jnp.max(jnp.abs(bx) + jnp.abs(by) + jnp.abs(bz), axis=(1, 2)), 1.0e-12)
    charge_balance_residual = jnp.max(jnp.abs(div_j), axis=(1, 2))
    residual = jnp.nan_to_num(residual, nan=scalar_limit, posinf=scalar_limit, neginf=scalar_limit)
    volumetric_flow_rate = jnp.nan_to_num(volumetric_flow_rate)
    mean_velocity = jnp.nan_to_num(mean_velocity)
    axial_current = jnp.nan_to_num(axial_current, nan=scalar_limit, posinf=scalar_limit, neginf=scalar_limit)
    wall_current_leakage = jnp.nan_to_num(wall_current_leakage, nan=scalar_limit, posinf=scalar_limit, neginf=scalar_limit)
    current_scaled_pressure_proxy = jnp.nan_to_num(current_scaled_pressure_proxy, nan=scalar_limit, posinf=scalar_limit, neginf=scalar_limit)
    charge_balance_residual = jnp.nan_to_num(charge_balance_residual, nan=scalar_limit, posinf=scalar_limit, neginf=scalar_limit)
    boundary_current_residual = jnp.nan_to_num(boundary_current_residual, nan=scalar_limit, posinf=scalar_limit, neginf=scalar_limit)
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
        current_scaled_pressure_proxy=jnp.asarray(current_scaled_pressure_proxy, dtype=float),
        charge_balance_residual=jnp.asarray(charge_balance_residual, dtype=float),
        boundary_current_residual=jnp.asarray(boundary_current_residual, dtype=float),
        geometry_kind=case.geometry.kind,
        solver_kind=case.solver.kind,
    )


def validate_extruded_inductionless_solution(
    bundle: ExtrudedFieldBundle,
    *,
    station_history: list[dict[str, float]] | tuple[dict[str, float], ...] | None = None,
) -> ExtrudedInductionlessValidation:
    field_scale = jnp.asarray(bundle.field_scale, dtype=float)
    mean_velocity = jnp.asarray(bundle.mean_velocity, dtype=float)
    volumetric_flow_rate = jnp.asarray(bundle.volumetric_flow_rate, dtype=float)
    axial_current = jnp.asarray(bundle.axial_current, dtype=float)
    wall_current_leakage = jnp.asarray(bundle.wall_current_leakage, dtype=float)
    residual = jnp.asarray(bundle.residual, dtype=float)
    charge_balance_residual = jnp.asarray(bundle.charge_balance_residual, dtype=float)
    boundary_current_residual = jnp.asarray(bundle.boundary_current_residual, dtype=float)
    peak_velocity = jnp.max(jnp.abs(bundle.u), axis=(1, 2))
    pressure_span = jnp.max(bundle.p, axis=(1, 2)) - jnp.min(bundle.p, axis=(1, 2))
    correlation = _safe_correlation(field_scale, mean_velocity)
    return ExtrudedInductionlessValidation(
        station_count=int(bundle.x.shape[0]),
        max_residual=float(jnp.max(jnp.abs(residual))) if residual.size else 0.0,
        max_charge_balance_residual=float(jnp.max(jnp.abs(charge_balance_residual)))
        if charge_balance_residual.size
        else 0.0,
        mean_velocity_span=float(jnp.max(mean_velocity) - jnp.min(mean_velocity)) if mean_velocity.size else 0.0,
        volumetric_flow_rate_span=float(jnp.max(volumetric_flow_rate) - jnp.min(volumetric_flow_rate))
        if volumetric_flow_rate.size
        else 0.0,
        axial_current_span=float(jnp.max(axial_current) - jnp.min(axial_current)) if axial_current.size else 0.0,
        max_wall_current_leakage=float(jnp.max(jnp.abs(wall_current_leakage))) if wall_current_leakage.size else 0.0,
        net_boundary_current_residual=float(jnp.max(jnp.abs(boundary_current_residual))) if boundary_current_residual.size else 0.0,
        field_mean_velocity_correlation=correlation,
        peak_velocity_span=float(jnp.max(peak_velocity) - jnp.min(peak_velocity)) if peak_velocity.size else 0.0,
        pressure_span_range=float(jnp.max(pressure_span) - jnp.min(pressure_span)) if pressure_span.size else 0.0,
    )


def solve_extruded_inductionless(
    problem: ExtrudedInductionlessProblem,
    *,
    solver=solve_steady,
    initial_bundle: ExtrudedFieldBundle | None = None,
) -> ExtrudedInductionlessSolution:
    if problem.case.geometry.kind in {"rect_duct", "layered_duct", "pipe_ogrid"}:
        bundle = _solve_extruded_projection(problem, initial_bundle=initial_bundle)
        station_history = _bundle_station_history(bundle)
    else:
        station_history = run_fringing_station_sweep(problem.case, problem.profile, solver=solver)
        bundle = run_extruded_inductionless_slice(problem.case, problem.profile, solver=solver)
    validation = validate_extruded_inductionless_solution(bundle, station_history=station_history)
    return ExtrudedInductionlessSolution(
        problem=problem,
        bundle=bundle,
        station_history=tuple(station_history),
        validation=validation,
    )

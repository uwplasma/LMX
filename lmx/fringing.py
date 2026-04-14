from __future__ import annotations

from dataclasses import dataclass, replace

import jax.numpy as jnp

from .cases import make_shercliff_case
from .core import Solution
from .mesh import generate_layered_duct_mesh, generate_rect_duct_mesh
from .physics import build_material_fields
from .specs import CaseSpec, MagneticFieldSpec
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
    field = jnp.zeros_like(rhs_compatible)
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
            "residual": float(bundle.residual[index]),
            "charge_balance_residual": float(bundle.charge_balance_residual[index]),
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
        geometry_kind=base_case.geometry.kind,
        solver_kind=base_case.solver.kind,
    )


def _solve_extruded_projection(problem: ExtrudedInductionlessProblem) -> ExtrudedFieldBundle:
    case = problem.case
    mesh = _cross_section_mesh(case)
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
        source_x = sigma * uxb_x
        source_y = sigma * uxb_y
        source_z = sigma * uxb_z
        emf_rhs = (
            _gradient_3d(source_x, dx=dx, dy=dy, dz=dz)[0]
            + _gradient_3d(source_y, dx=dx, dy=dy, dz=dz)[1]
            + _gradient_3d(source_z, dx=dx, dy=dy, dz=dz)[2]
        )
        phi, _, _, _ = _variable_coefficient_poisson_jacobi_3d(
            emf_rhs,
            sigma,
            dx=dx,
            dy=dy,
            dz=dz,
            iterations=poisson_iterations,
            tolerance=poisson_tolerance,
        )
        phi = _clip_state(phi, scalar_limit)

        dphi_dx, dphi_dy, dphi_dz = _gradient_3d(phi, dx=dx, dy=dy, dz=dz)
        jx = _clip_state(sigma * (-dphi_dx + uxb_x), scalar_limit)
        jy = _clip_state(sigma * (-dphi_dy + uxb_y), scalar_limit)
        jz = _clip_state(sigma * (-dphi_dz + uxb_z), scalar_limit)
        div_j = _gradient_3d(jx, dx=dx, dy=dy, dz=dz)[0] + _gradient_3d(jy, dx=dx, dy=dy, dz=dz)[1] + _gradient_3d(jz, dx=dx, dy=dy, dz=dz)[2]
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
    wall_current_leakage = (
        jnp.sum(jnp.abs(jy[:, 0, :]), axis=1) * dx * dz
        + jnp.sum(jnp.abs(jy[:, -1, :]), axis=1) * dx * dz
        + jnp.sum(jnp.abs(jz[:, :, 0]), axis=1) * dx * dy
        + jnp.sum(jnp.abs(jz[:, :, -1]), axis=1) * dx * dy
    )
    current_scaled_pressure_proxy = jnp.max(jnp.abs(jy), axis=(1, 2)) * jnp.maximum(jnp.max(jnp.abs(bx) + jnp.abs(by) + jnp.abs(bz), axis=(1, 2)), 1.0e-12)
    charge_balance_residual = jnp.max(jnp.abs(_gradient_3d(jx, dx=dx, dy=dy, dz=dz)[0] + _gradient_3d(jy, dx=dx, dy=dy, dz=dz)[1] + _gradient_3d(jz, dx=dx, dy=dy, dz=dz)[2]), axis=(1, 2))
    residual = jnp.nan_to_num(residual, nan=scalar_limit, posinf=scalar_limit, neginf=scalar_limit)
    volumetric_flow_rate = jnp.nan_to_num(volumetric_flow_rate)
    mean_velocity = jnp.nan_to_num(mean_velocity)
    axial_current = jnp.nan_to_num(axial_current, nan=scalar_limit, posinf=scalar_limit, neginf=scalar_limit)
    wall_current_leakage = jnp.nan_to_num(wall_current_leakage, nan=scalar_limit, posinf=scalar_limit, neginf=scalar_limit)
    current_scaled_pressure_proxy = jnp.nan_to_num(current_scaled_pressure_proxy, nan=scalar_limit, posinf=scalar_limit, neginf=scalar_limit)
    charge_balance_residual = jnp.nan_to_num(charge_balance_residual, nan=scalar_limit, posinf=scalar_limit, neginf=scalar_limit)
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
    correlation = _safe_correlation(field_scale, mean_velocity)
    if bundle.x.size > 1:
        dx = float(jnp.mean(jnp.diff(bundle.x)))
    else:
        dx = 1.0
    if bundle.y.size > 1:
        dy = float(jnp.mean(jnp.diff(bundle.y)))
    else:
        dy = 1.0
    if bundle.z.size > 1:
        dz = float(jnp.mean(jnp.diff(bundle.z)))
    else:
        dz = 1.0
    boundary_current_residual = _net_boundary_current_residual(
        jnp.asarray(bundle.jx, dtype=float),
        jnp.asarray(bundle.jy, dtype=float),
        jnp.asarray(bundle.jz, dtype=float),
        dx=dx,
        dy=dy,
        dz=dz,
    )
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
        net_boundary_current_residual=boundary_current_residual,
        field_mean_velocity_correlation=correlation,
    )


def solve_extruded_inductionless(
    problem: ExtrudedInductionlessProblem,
    *,
    solver=solve_steady,
) -> ExtrudedInductionlessSolution:
    if problem.case.geometry.kind in {"rect_duct", "layered_duct"}:
        bundle = _solve_extruded_projection(problem)
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

"""Public 3-D fringing problems, applications, solves, and validation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from ._fringing_common import (
    _EXTRUDED_NUMERICAL_RESULTS,
    _coordinate_scale,
)
from ._fringing_solver import (
    _solve_extruded_projection,
)
from .cases import _ha_to_b, make_shercliff_case
from .mesh import (
    _cross_section_mesh,
    sample_tabulated_field_volume,
)
from .physics import build_material_fields
from .specs import (
    BoundaryCondition,
    CaseSpec,
    ExtrudedFieldBundle,
    ExtrudedInductionlessProblem,
    ExtrudedInductionlessSolution,
    ExtrudedInductionlessValidation,
    ExtrudedIterationProgress,
    FringingProfile,
    GeometrySpec,
    MagneticFieldSpec,
    OutputSpec,
    RegionSpec,
    SolverConfig,
    TimeStepperConfig,
    require_finite,
)


def _safe_correlation(x: jnp.ndarray, y: jnp.ndarray) -> float:
    x, y = x - jnp.mean(x), y - jnp.mean(y)
    scale = jnp.sqrt(jnp.sum(x**2) * jnp.sum(y**2))
    return float(jnp.where(scale > 0.0, jnp.sum(x * y) / scale, 0.0))


def _bundle_station_history(bundle: ExtrudedFieldBundle) -> tuple[dict[str, float], ...]:
    zeros = jnp.zeros_like(bundle.x)
    axial_pressure = getattr(bundle, "axial_pressure_loss_gradient", zeros)
    transverse_pressure = getattr(bundle, "transverse_pressure_difference", zeros)
    columns = {
        "x": bundle.x,
        "field_scale": bundle.field_scale,
        "u_max": jnp.max(jnp.abs(bundle.u), axis=(1, 2)),
        "mean_velocity": bundle.mean_velocity,
        "volumetric_flow_rate": bundle.volumetric_flow_rate,
        "axial_current": bundle.axial_current,
        "wall_current_leakage": bundle.wall_current_leakage,
        "current_scaled_pressure_proxy": bundle.current_scaled_pressure_proxy,
        "axial_pressure_loss_gradient": axial_pressure if axial_pressure.size else zeros,
        "transverse_pressure_difference": transverse_pressure if transverse_pressure.size else zeros,
        "pressure_span": jnp.max(bundle.p, axis=(1, 2)) - jnp.min(bundle.p, axis=(1, 2)),
        "residual": bundle.residual,
        "charge_balance_residual": bundle.charge_balance_residual,
        "boundary_current_residual": bundle.boundary_current_residual,
    }
    rows = np.asarray(jnp.stack(tuple(columns.values()), axis=1))
    return tuple(dict(zip(columns, map(float, row), strict=True)) for row in rows)


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
    case = make_shercliff_case(ha=ha_peak, width=width, height=height, ny=ny, nz=nz)
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
    from .mesh import make_localized_divergence_free_obstacle_field

    field_fn = make_localized_divergence_free_obstacle_field(
        width=width,
        height=height,
        base_bz=base_bz,
        core_fraction_y=core_fraction_y,
        core_fraction_z=core_fraction_z,
    )
    problem = build_square_duct_extruded_problem(
        ha_peak=1.0,
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
    case = replace(
        problem.case,
        name=f"magnetic_obstacle_rect_bz{int(base_bz)}",
        magnetic_field=MagneticFieldSpec(kind="analytic", fn=field_fn),
        forcing=forcing,
        notes=(
            "Localized-field magnetic-obstacle baseline on the rectangular "
            "extruded inductionless solver lane."
        ),
    )
    return replace(problem, case=case)


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
    field_metrics = _variable_field_metrics(solution, field_ny=field_ny, field_nz=field_nz)
    validation = solution.validation
    finite_velocity = bool(np.isfinite(np.asarray(solution.bundle.u, dtype=float)).all())
    field_scale = np.asarray(solution.bundle.field_scale, dtype=float)
    mean_velocity = np.asarray(solution.bundle.mean_velocity, dtype=float)
    current_proxy = np.asarray(solution.bundle.current_scaled_pressure_proxy, dtype=float)
    field_velocity_correlation = float(
        _safe_correlation(jnp.asarray(field_scale), jnp.asarray(mean_velocity))
    )
    velocity_change = float(np.max(mean_velocity) - np.min(mean_velocity)) if mean_velocity.size else 0.0
    current_proxy_change = float(np.max(current_proxy) - np.min(current_proxy)) if current_proxy.size else 0.0
    charge_limit = 5.0e-2 if solution.problem.case.geometry.kind == "rect_duct" else 2.0e-1
    validation_pass = bool(
        finite_velocity
        and field_metrics["rms_divergence"] <= 5.0e-2
        and validation.max_charge_balance_residual <= charge_limit
        and validation.net_boundary_current_residual <= 1.0e-8
        and validation.max_wall_current_leakage <= 1.0e-8
        and abs(field_velocity_correlation) >= 0.2
        and velocity_change > 1.0e-8
        and current_proxy_change > 1.0e-8
    )
    return {
        **field_metrics,
        "finite_velocity": finite_velocity,
        "field_velocity_correlation": field_velocity_correlation,
        "mean_velocity_change": velocity_change,
        "current_proxy_change": current_proxy_change,
        "max_charge_balance_residual": float(validation.max_charge_balance_residual),
        "max_wall_current_leakage": float(validation.max_wall_current_leakage),
        "net_boundary_current_residual": float(validation.net_boundary_current_residual),
        "validation_pass": validation_pass,
    }


def validate_variable_field_pipe_solution(
    solution: ExtrudedInductionlessSolution,
    *,
    field_ny: int = 81,
    field_nz: int = 81,
) -> dict[str, float | bool]:
    if solution.problem.case.geometry.kind != "pipe_ogrid":
        raise ValueError("Variable-field pipe validation requires pipe_ogrid geometry")
    field_metrics = _variable_field_metrics(solution, field_ny=field_ny, field_nz=field_nz)
    validation = solution.validation
    mean_velocity = np.asarray(solution.bundle.mean_velocity, dtype=float)
    current_proxy = np.asarray(solution.bundle.current_scaled_pressure_proxy, dtype=float)
    velocity_change = float(np.max(mean_velocity) - np.min(mean_velocity)) if mean_velocity.size else 0.0
    current_proxy_change = float(np.max(current_proxy) - np.min(current_proxy)) if current_proxy.size else 0.0
    divergence_ratio = float(
        field_metrics["rms_divergence"] / max(field_metrics["mean_field_magnitude"], 1.0e-12)
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
        "net_boundary_current_residual": float(validation.net_boundary_current_residual),
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
        from .mesh import cross_section_divergence_metrics

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
    raise ValueError("Variable-field validation currently supports analytic and tabulated magnetic fields")


def validate_magnetic_obstacle_baseline(
    solution: ExtrudedInductionlessSolution,
    *,
    field_ny: int = 81,
    field_nz: int = 81,
) -> dict[str, float | bool | str]:
    if solution.problem.case.geometry.kind != "rect_duct":
        raise ValueError("Magnetic-obstacle baseline currently supports rectangular ducts only")
    field_metrics = _variable_field_metrics(solution, field_ny=field_ny, field_nz=field_nz)
    bundle = solution.bundle
    validation = solution.validation
    field_scale = np.asarray(bundle.field_scale, dtype=float)
    mean_velocity = np.asarray(bundle.mean_velocity, dtype=float)
    current_proxy = np.asarray(bundle.current_scaled_pressure_proxy, dtype=float)
    peak_index = int(np.argmax(field_scale)) if field_scale.size else 0
    inlet_reference = float(mean_velocity[0]) if mean_velocity.size else 0.0
    obstacle_velocity_deficit = (
        float(inlet_reference - mean_velocity[peak_index]) if mean_velocity.size else 0.0
    )
    current_proxy_peak = float(np.max(current_proxy)) if current_proxy.size else 0.0
    field_velocity_correlation = float(
        _safe_correlation(jnp.asarray(field_scale), jnp.asarray(mean_velocity))
    )
    divergence_to_field_ratio = float(
        field_metrics["rms_divergence"] / max(field_metrics["mean_field_magnitude"], 1.0e-12)
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
    validation_pass = bool(field_quality_pass and conservation_pass and response_observable_pass)
    return {
        **field_metrics,
        "divergence_to_field_ratio": divergence_to_field_ratio,
        "obstacle_velocity_deficit": obstacle_velocity_deficit,
        "current_proxy_peak": current_proxy_peak,
        "field_velocity_correlation": field_velocity_correlation,
        "max_charge_balance_residual": float(validation.max_charge_balance_residual),
        "max_wall_current_leakage": float(validation.max_wall_current_leakage),
        "net_boundary_current_residual": float(validation.net_boundary_current_residual),
        "field_quality_pass": field_quality_pass,
        "conservation_pass": conservation_pass,
        "response_observable_pass": response_observable_pass,
        "reference_kind": "none",
        "external_reference_available": False,
        "research_grade_validation_pass": False,
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


def validate_extruded_inductionless_solution(
    bundle: ExtrudedFieldBundle,
    *,
    station_history: list[dict[str, float]] | tuple[dict[str, float], ...] | None = None,
) -> ExtrudedInductionlessValidation:
    required = (
        "field_scale",
        "u_max",
        "mean_velocity",
        "volumetric_flow_rate",
        "axial_current",
        "wall_current_leakage",
        "residual",
        "charge_balance_residual",
        "boundary_current_residual",
        "pressure_span",
    )
    history = tuple(station_history or ())
    if len(history) != bundle.x.shape[0] or any(any(name not in row for name in required) for row in history):
        history = _bundle_station_history(bundle)
    values = {name: np.asarray([row[name] for row in history], dtype=float) for name in required}
    field_scale, mean_velocity = values["field_scale"], values["mean_velocity"]
    centered_field = field_scale - np.mean(field_scale)
    centered_velocity = mean_velocity - np.mean(mean_velocity)
    correlation_scale = np.sqrt(np.sum(centered_field**2) * np.sum(centered_velocity**2))
    correlation = (
        float(np.sum(centered_field * centered_velocity) / correlation_scale)
        if correlation_scale > 0.0
        else 0.0
    )
    axial_current, pressure_span = values["axial_current"], values["pressure_span"]
    component_history = np.asarray(getattr(bundle, "iteration_component_residual_history", jnp.zeros((0, 6))))

    def center(array):
        return float(np.mean(array[(array.size - 1) // 2 : array.size // 2 + 1]))

    return ExtrudedInductionlessValidation(
        station_count=int(bundle.x.shape[0]),
        max_residual=float(np.max(np.abs(values["residual"]))),
        max_charge_balance_residual=float(np.max(np.abs(values["charge_balance_residual"]))),
        mean_velocity_span=float(np.ptp(mean_velocity)),
        volumetric_flow_rate_span=float(np.ptp(values["volumetric_flow_rate"])),
        axial_current_span=float(np.ptp(axial_current)),
        axial_current_mirror_residual=float(np.max(np.abs(axial_current + axial_current[::-1]))),
        max_wall_current_leakage=float(np.max(np.abs(values["wall_current_leakage"]))),
        net_boundary_current_residual=float(np.max(np.abs(values["boundary_current_residual"]))),
        field_mean_velocity_correlation=correlation,
        peak_velocity_span=float(np.ptp(values["u_max"])),
        pressure_span_range=float(np.ptp(pressure_span)),
        pressure_span_mirror_residual=float(np.max(np.abs(pressure_span - pressure_span[::-1]))),
        center_axial_current=center(axial_current),
        center_pressure_span=center(pressure_span),
        max_divergence_residual=(
            float(component_history[-1, 3])
            if component_history.ndim == 2 and component_history.shape[0]
            else 0.0
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

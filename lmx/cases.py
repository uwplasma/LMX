"""Fully developed steady and transient inductionless solvers."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from solvax import (
    aitken_relaxation as _solvax_aitken_relaxation,
)
from solvax import (
    anderson_mixing as _solvax_anderson_mixing,
)
from solvax import fixed_point_iteration

from .config import RestartLogInfo
from .mesh import (
    StructuredMesh,
    generate_rect_duct_mesh,
    gradient_scalar,
)
from .physics import build_material_fields, magnetic_field_components, magnetic_field_from_hartmann
from .solvers import (
    _LINEAR_RESIDUAL_FLOOR,
    _MIN_STRICT_POTENTIAL_COUPLING_SOLVES,
    _POTENTIAL_COUPLING_NORMALIZED_GATE,
    _bounded_time_step_count,
    _build_mesh,
    _cell_metric,
    _compute_current_and_lorentz,
    _concat_history,
    _coupling_potential_tolerance,
    _emit_solver_header,
    _emit_solver_step,
    _enforce_target_mean_velocity,
    _enforce_velocity_bc,
    _face_current_emf_and_lorentz_max,
    _face_emf,
    _fully_developed_rhs,
    _has_uniform_spacing,
    _initial_solver_state,
    _integral_diagnostics,
    _limited_velocity_update,
    _nested_velocity_tolerance,
    _potential_coefficients,
    _potential_preconditioner_for_materials,
    _pressure_proxy_reference_current,
    _reference_mean_velocity,
    _resolve_potential_solver,
    _scaled_pressure_proxy_value,
    _solve_potential,
    _solve_velocity_system,
    _target_mean_velocity,
    _velocity_system_coefficients,
    _volume_scaled_potential_system,
    solve_poisson_jacobi_state,
)
from .specs import (
    BoundaryCondition,
    CaseSpec,
    Diagnostics,
    GeometrySpec,
    MagneticFieldSpec,
    MHDState,
    OutputSpec,
    RegionSpec,
    Solution,
    SolverConfig,
    TimeStepperConfig,
    require_finite,
)


def _ha_to_b(ha: float, length_scale: float, conductivity: float, density: float, viscosity: float) -> float:
    """Return ``B`` for target ``Ha`` with ``viscosity`` interpreted as ``nu``."""

    return magnetic_field_from_hartmann(
        hartmann=ha,
        length_scale=length_scale,
        conductivity=conductivity,
        density=density,
        kinematic_viscosity=viscosity,
    )


def _wall_conductivity_from_conductance_ratio(
    *,
    wall_conductance_ratio: float,
    fluid_conductivity: float,
    wall_thickness: float,
    hartmann_half_spacing: float,
) -> float:
    if wall_thickness <= 0.0:
        raise ValueError(
            "wall_thickness must be positive when deriving wall conductivity from conductance ratio"
        )
    if hartmann_half_spacing <= 0.0:
        raise ValueError(
            "hartmann_half_spacing must be positive when deriving wall conductivity from conductance ratio"
        )
    return wall_conductance_ratio * fluid_conductivity * hartmann_half_spacing / wall_thickness


def _hunt_short_transient_controls(ha: float) -> TimeStepperConfig:
    if ha <= 20.0:
        return TimeStepperConfig(
            dt=0.002,
            t_final=1.0,
            max_steps=500,
            outer_iterations=6,
            potential_iterations=400,
            relaxation=0.08,
            velocity_update_limit=2e-3,
            current_reconstruction="cell_centered",
        )
    if ha <= 100.0:
        return TimeStepperConfig(
            dt=0.002,
            t_final=1.0,
            max_steps=500,
            outer_iterations=4,
            potential_iterations=400,
            relaxation=0.1,
            velocity_update_limit=1e-3,
            current_reconstruction="cell_centered",
        )
    return TimeStepperConfig(
        dt=0.002,
        t_final=1.0,
        max_steps=500,
        outer_iterations=3,
        potential_iterations=400,
        relaxation=0.1,
        velocity_update_limit=1e-3,
        current_reconstruction="cell_centered",
    )


def _fully_developed_solver(mode: str = "steady") -> SolverConfig:
    return SolverConfig(
        kind="fully_developed_inductionless",
        mode=mode,
        preconditioner="jacobi",
        time_scheme="implicit_euler",
        coupling_iterations=16,
        coupling_tolerance=1e-8,
    )


def make_hartmann_case(
    ha: float = 20.0,
    width: float = 2.0,
    height: float = 2.0,
    ny: int = 96,
    nz: int = 96,
    conductivity: float = 1.0,
    density: float = 1.0,
    viscosity: float = 1.0,
    output_dir: str | None = None,
) -> CaseSpec:
    """Build an insulating rectangular Hartmann-duct reference case."""

    bmag = _ha_to_b(ha, 0.5 * height, conductivity, density, viscosity)
    anchor = (ny // 2, nz // 2)
    return CaseSpec(
        name=f"hartmann_ha{int(ha)}",
        geometry=GeometrySpec(kind="rect_duct", width=width, height=height, ny=ny, nz=nz, target_ha=ha),
        regions=(RegionSpec("fluid", "fluid", conductivity, density, viscosity),),
        magnetic_field=MagneticFieldSpec(kind="constant", value=(0.0, 1.0 * bmag, 0.0)),
        boundary_conditions=(
            BoundaryCondition("walls", "no_slip"),
            BoundaryCondition("electric", "insulating"),
        ),
        time_stepper=TimeStepperConfig(
            dt=0.001, t_final=1.0, max_steps=400, potential_iterations=200, relaxation=0.1
        ),
        solver=_fully_developed_solver(),
        output=OutputSpec(directory=output_dir),
        forcing=1.0,
        reference_pressure_gradient=-1.0,
        reference_phi_cell=anchor,
        notes="Planar Hartmann-like reference configuration for solver smoke tests.",
    )


def make_shercliff_case(
    ha: float = 20.0,
    width: float = 2.0,
    height: float = 2.0,
    ny: int = 96,
    nz: int = 96,
    conductivity: float = 1.0,
    density: float = 1.0,
    viscosity: float = 1.0,
    output_dir: str | None = None,
) -> CaseSpec:
    """Build an all-insulating rectangular Shercliff-duct case."""

    bmag = _ha_to_b(ha, 0.5 * width, conductivity, density, viscosity)
    anchor = (ny // 2, nz // 2)
    return CaseSpec(
        name=f"shercliff_ha{int(ha)}",
        geometry=GeometrySpec(kind="rect_duct", width=width, height=height, ny=ny, nz=nz, target_ha=ha),
        regions=(RegionSpec("fluid", "fluid", conductivity, density, viscosity),),
        magnetic_field=MagneticFieldSpec(kind="constant", value=(0.0, 1.0 * bmag, 0.0)),
        boundary_conditions=(
            BoundaryCondition("walls", "no_slip"),
            BoundaryCondition("electric", "insulating"),
        ),
        time_stepper=TimeStepperConfig(
            dt=0.001, t_final=1.5, max_steps=400, potential_iterations=225, relaxation=0.1
        ),
        solver=_fully_developed_solver(),
        output=OutputSpec(directory=output_dir),
        forcing=1.0,
        reference_pressure_gradient=-1.0,
        reference_phi_cell=anchor,
        notes="All-insulating Shercliff-style duct. Analytical validation hooks are staged through the benchmark and validation utilities.",
    )


def make_hunt_case(
    ha: float = 20.0,
    width: float = 2.0,
    height: float = 2.0,
    ny: int = 72,
    nz: int = 72,
    wall_cells: int = 8,
    wall_thickness: float = 0.1,
    insulator_cells: int | None = None,
    insulator_thickness: float | None = None,
    fluid_conductivity: float = 1.0,
    wall_conductance_ratio: float = 0.05,
    wall_conductivity: float | None = None,
    insulator_conductivity: float | None = None,
    insulator_conductivity_ratio: float = 1e-12,
    density: float = 1.0,
    viscosity: float = 1.0,
    output_dir: str | None = None,
) -> CaseSpec:
    """Build a Hunt duct with conducting Hartmann and insulating side walls."""

    bmag = _ha_to_b(ha, 0.5 * width, fluid_conductivity, density, viscosity)
    if wall_conductivity is None:
        wall_conductivity = _wall_conductivity_from_conductance_ratio(
            wall_conductance_ratio=wall_conductance_ratio,
            fluid_conductivity=fluid_conductivity,
            wall_thickness=wall_thickness,
            hartmann_half_spacing=0.5 * height,
        )
    if insulator_cells is None:
        insulator_cells = wall_cells
    if insulator_thickness is None:
        insulator_thickness = wall_thickness
    if insulator_conductivity is None:
        insulator_conductivity = fluid_conductivity * insulator_conductivity_ratio
    anchor = ((ny + 2 * insulator_cells) // 2, (nz + 2 * wall_cells) // 2)
    controls = _hunt_short_transient_controls(ha)
    return CaseSpec(
        name=f"hunt_ha{int(ha)}",
        geometry=GeometrySpec(
            kind="layered_duct",
            width=width,
            height=height,
            ny=ny,
            nz=nz,
            wall_thickness=(insulator_thickness, insulator_thickness, wall_thickness, wall_thickness),
            wall_cells=(insulator_cells, insulator_cells, wall_cells, wall_cells),
            target_ha=ha,
        ),
        regions=(
            RegionSpec("fluid", "fluid", fluid_conductivity, density, viscosity),
            RegionSpec("conducting_wall", "solid", wall_conductivity, density, viscosity, wall_thickness),
            RegionSpec(
                "insulating_wall", "solid", insulator_conductivity, density, viscosity, insulator_thickness
            ),
        ),
        magnetic_field=MagneticFieldSpec(kind="constant", value=(0.0, 1.0 * bmag, 0.0)),
        boundary_conditions=(
            BoundaryCondition("walls", "no_slip"),
            BoundaryCondition(
                "conducting_hartmann_walls", "conducting_wall", region="conducting_wall", side="left_right"
            ),
            BoundaryCondition(
                "insulating_side_walls", "insulating", region="insulating_wall", side="top_bottom"
            ),
        ),
        time_stepper=controls,
        solver=_fully_developed_solver(),
        output=OutputSpec(directory=output_dir),
        forcing=1.0,
        reference_pressure_gradient=-1.0,
        reference_phi_cell=anchor,
        notes=(
            "Hunt-style duct with explicit conducting Hartmann-wall layers and insulating side-wall layers. "
            f"Default wall conductance ratio c={wall_conductance_ratio:g}."
        ),
    )


def _fully_developed_case_step(
    *,
    case: CaseSpec,
    mesh: StructuredMesh,
    materials,
    u_previous: jnp.ndarray,
    step_time: float,
    potential_solver: str,
    target_mean_velocity: float | None,
    preconditioner: str,
    coupling_iterations: int,
    coupling_tolerance: float,
    phi_previous: jnp.ndarray | None = None,
) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
]:
    _, by, bz = magnetic_field_components(case.magnetic_field, mesh, time=step_time)
    forcing = jnp.asarray(case.forcing, dtype=by.dtype)
    fluid_mask = materials.fluid_mask
    active_mask = fluid_mask
    u_iter = u_previous
    phi_iter = jnp.zeros_like(u_previous) if phi_previous is None else phi_previous
    dt = case.time_stepper.dt
    fluid_weight = jnp.where(fluid_mask, _cell_metric(mesh).astype(u_previous.dtype), 0.0)
    fluid_total_weight = jnp.maximum(jnp.sum(fluid_weight), 1e-20)
    velocity_residual = jnp.asarray(jnp.inf, dtype=u_previous.dtype)
    potential_residual = jnp.asarray(jnp.inf, dtype=u_previous.dtype)
    potential_iteration_count = jnp.asarray(0, dtype=jnp.int32)
    potential_initial_residual = jnp.asarray(0.0, dtype=u_previous.dtype)
    linear_iteration_count = jnp.asarray(0, dtype=jnp.int32)
    linear_residual = jnp.asarray(jnp.inf, dtype=u_previous.dtype)
    linear_initial_residual = jnp.asarray(0.0, dtype=u_previous.dtype)
    applied_forcing = forcing
    steady_mode = case.solver.mode == "steady"
    potential_preconditioner = None
    potential_flexible = False
    if potential_solver == "cg_volume":
        coefficients = _potential_coefficients(mesh, materials.conductivity)
        scaled = _volume_scaled_potential_system(mesh, *coefficients, jnp.zeros_like(u_previous))
        potential_preconditioner, potential_flexible = _potential_preconditioner_for_materials(
            mesh,
            materials.conductivity,
            *scaled[:5],
            case.reference_phi_cell,
        )
    acceleration = case.solver.coupling_acceleration
    if acceleration not in {"none", "aitken", "anderson"}:
        raise ValueError(f"Unsupported coupling acceleration {acceleration!r}")
    if (
        case.solver.coupling_min_relaxation <= 0.0
        or case.solver.coupling_max_relaxation < case.solver.coupling_min_relaxation
    ):
        raise ValueError("Coupling relaxation bounds must satisfy 0 < min <= max")
    if case.solver.coupling_history_depth < 1:
        raise ValueError("Anderson coupling history depth must be positive")
    if case.solver.coupling_regularization < 0.0:
        raise ValueError("Anderson coupling regularization must be non-negative")
    if not 0.0 <= case.solver.coupling_damping <= 1.0:
        raise ValueError("Anderson coupling damping must lie in [0, 1]")
    previous_fixed_point_residual: jnp.ndarray | None = None
    coupling_relaxation = jnp.asarray(1.0, dtype=u_previous.dtype)
    anderson_iterates: list[jnp.ndarray] = []
    anderson_residuals: list[jnp.ndarray] = []
    strict_potential_solves = 0
    velocity_linear_tolerance = _nested_velocity_tolerance(coupling_tolerance, u_previous.dtype)

    if case.solver.time_scheme != "implicit_euler" and not steady_mode:
        raise NotImplementedError("fully_developed_inductionless currently supports implicit_euler only")

    for _ in range(max(1, coupling_iterations)):
        potential_iteration_tolerance = _coupling_potential_tolerance(
            case.time_stepper.potential_tolerance,
            velocity_residual=float(velocity_residual),
            coupling_tolerance=float(coupling_tolerance),
            flexible=potential_flexible,
        )
        potential_result = _solve_potential(
            mesh,
            materials.conductivity,
            fluid_mask,
            u_iter,
            by,
            bz,
            case.reference_phi_cell,
            case.time_stepper.potential_iterations,
            tolerance=potential_iteration_tolerance,
            relaxation=case.time_stepper.potential_relaxation,
            solver=potential_solver,
            initial_phi=phi_iter,
            potential_preconditioner=potential_preconditioner,
            potential_flexible=potential_flexible,
            return_solver_residual=True,
        )
        requested_potential_tolerance = case.time_stepper.potential_tolerance
        if (
            requested_potential_tolerance is not None
            and potential_iteration_tolerance is not None
            and potential_iteration_tolerance <= requested_potential_tolerance
        ):
            strict_potential_solves += 1
        (
            phi,
            potential_residual,
            potential_iteration_count,
            potential_initial_residual,
            _potential_solver_residual,
        ) = potential_result
        require_finite(
            "potential solve",
            potential=phi,
            residual=potential_residual,
        )
        phi_iter = phi
        magnetic_reaction = jnp.where(
            active_mask,
            materials.conductivity * (by**2 + bz**2) / materials.density,
            0.0,
        )
        reaction = magnetic_reaction
        if not steady_mode:
            reaction = reaction + jnp.where(active_mask, 1.0 / dt, 0.0)
        rhs_base, _ = _fully_developed_rhs(
            mesh=mesh,
            sigma=materials.conductivity,
            rho=materials.density,
            fluid_mask=fluid_mask,
            u=u_iter,
            phi=phi,
            by=by,
            bz=bz,
            forcing=jnp.asarray(0.0, dtype=u_previous.dtype),
            current_reconstruction=case.time_stepper.current_reconstruction,
        )
        rhs_base = rhs_base + magnetic_reaction * jnp.where(active_mask, u_iter, 0.0)
        if not steady_mode:
            rhs_base = rhs_base + jnp.where(active_mask, u_previous / dt, 0.0)
        if target_mean_velocity is None:
            rhs = rhs_base + jnp.where(active_mask, forcing / materials.density, 0.0)
            u_next, velocity_linear_residual, linear_iteration_count, linear_initial_residual = (
                _solve_velocity_system(
                    mesh=mesh,
                    diffusivity=materials.viscosity,
                    reaction=reaction,
                    rhs=rhs,
                    active_mask=active_mask,
                    preconditioner=preconditioner,
                    max_steps=max(case.time_stepper.max_steps, case.solver.coupling_iterations * 25),
                    tolerance=velocity_linear_tolerance,
                )
            )
            applied_forcing = forcing
        else:
            unit_rhs = jnp.where(active_mask, 1.0 / materials.density, 0.0)
            u_base, velocity_linear_residual, linear_iteration_count, linear_initial_residual = (
                _solve_velocity_system(
                    mesh=mesh,
                    diffusivity=materials.viscosity,
                    reaction=reaction,
                    rhs=rhs_base,
                    active_mask=active_mask,
                    preconditioner=preconditioner,
                    max_steps=max(case.time_stepper.max_steps, case.solver.coupling_iterations * 25),
                    tolerance=velocity_linear_tolerance,
                )
            )
            u_sensitivity, _, _, sensitivity_initial_residual = _solve_velocity_system(
                mesh=mesh,
                diffusivity=materials.viscosity,
                reaction=reaction,
                rhs=unit_rhs,
                active_mask=active_mask,
                preconditioner=preconditioner,
                max_steps=max(case.time_stepper.max_steps, case.solver.coupling_iterations * 25),
                tolerance=velocity_linear_tolerance,
            )
            linear_initial_residual = jnp.maximum(linear_initial_residual, sensitivity_initial_residual)
            mean_base = jnp.sum(fluid_weight * u_base) / fluid_total_weight
            mean_sensitivity = jnp.sum(fluid_weight * u_sensitivity) / fluid_total_weight
            applied_forcing = jnp.where(
                mean_sensitivity > 1e-20,
                (jnp.asarray(target_mean_velocity, dtype=u_previous.dtype) - mean_base) / mean_sensitivity,
                jnp.asarray(0.0, dtype=u_previous.dtype),
            )
            u_next = u_base + applied_forcing * u_sensitivity
        linear_residual = velocity_linear_residual
        u_next = _limited_velocity_update(
            u_iter,
            u_next,
            fluid_mask,
            max_delta=case.time_stepper.velocity_update_limit,
            limiter=case.time_stepper.velocity_update_limiter,
        )
        u_next = _enforce_velocity_bc(
            jnp.where(fluid_mask, u_next, 0.0),
            mesh,
            fluid_mask,
            interpolate_direct_fluid_walls=case.geometry.kind == "rect_duct",
        )
        u_next = _enforce_target_mean_velocity(u_next, mesh, fluid_mask, target_mean_velocity)
        fixed_point_residual = u_next - u_iter
        if acceleration == "aitken" and previous_fixed_point_residual is not None:
            coupling_relaxation = _solvax_aitken_relaxation(
                previous_fixed_point_residual,
                fixed_point_residual,
                coupling_relaxation,
                min_relaxation=case.solver.coupling_min_relaxation,
                max_relaxation=case.solver.coupling_max_relaxation,
            )
            u_next = u_iter + coupling_relaxation * fixed_point_residual
            u_next = _enforce_velocity_bc(
                jnp.where(fluid_mask, u_next, 0.0),
                mesh,
                fluid_mask,
                interpolate_direct_fluid_walls=case.geometry.kind == "rect_duct",
            )
            u_next = _enforce_target_mean_velocity(u_next, mesh, fluid_mask, target_mean_velocity)
        elif acceleration == "anderson":
            anderson_iterates.append(u_iter)
            anderson_residuals.append(fixed_point_residual)
            depth = case.solver.coupling_history_depth
            anderson_iterates = anderson_iterates[-depth:]
            anderson_residuals = anderson_residuals[-depth:]
            u_next = _solvax_anderson_mixing(
                jnp.stack(anderson_iterates),
                jnp.stack(anderson_residuals),
                regularization=case.solver.coupling_regularization,
                damping=case.solver.coupling_damping,
            )
            u_next = _enforce_velocity_bc(
                jnp.where(fluid_mask, u_next, 0.0),
                mesh,
                fluid_mask,
                interpolate_direct_fluid_walls=case.geometry.kind == "rect_duct",
            )
            u_next = _enforce_target_mean_velocity(u_next, mesh, fluid_mask, target_mean_velocity)
        # Convergence is defined by the unrelaxed fixed-point residual at the
        # current iterate.  If it passes, retain that certified iterate rather
        # than returning the subsequently extrapolated Anderson/Aitken point,
        # whose residual has not been evaluated.  Acceleration is used only to
        # choose the next iterate when another map evaluation is required.
        velocity_residual = jnp.max(jnp.abs(fixed_point_residual))
        velocity_converged = float(velocity_residual) <= float(coupling_tolerance)
        auxiliary_converged = (
            strict_potential_solves >= _MIN_STRICT_POTENTIAL_COUPLING_SOLVES
            and float(potential_residual) <= _POTENTIAL_COUPLING_NORMALIZED_GATE
            and float(linear_residual) <= max(float(coupling_tolerance), _LINEAR_RESIDUAL_FLOOR)
        )
        if velocity_converged and auxiliary_converged:
            break
        if velocity_converged:
            # Re-evaluate the auxiliary solves at the certified velocity until
            # their own gates pass; do not perturb it merely to accumulate the
            # required strict-solve evidence.
            previous_fixed_point_residual = None
            continue
        previous_fixed_point_residual = fixed_point_residual
        u_iter = u_next

    phi, potential_residual, potential_iteration_count, potential_initial_residual = _solve_potential(
        mesh,
        materials.conductivity,
        fluid_mask,
        u_iter,
        by,
        bz,
        case.reference_phi_cell,
        case.time_stepper.potential_iterations,
        tolerance=case.time_stepper.potential_tolerance,
        relaxation=case.time_stepper.potential_relaxation,
        solver=potential_solver,
        initial_phi=phi_iter,
        potential_preconditioner=potential_preconditioner,
        potential_flexible=potential_flexible,
    )
    jy, jz, lorentz = _compute_current_and_lorentz(
        mesh,
        materials.conductivity,
        fluid_mask,
        u_iter,
        phi,
        by,
        bz,
        reconstruction=case.time_stepper.current_reconstruction,
    )
    face_current_max, emf_max, face_lorentz_max = _face_current_emf_and_lorentz_max(
        mesh,
        materials.conductivity,
        fluid_mask,
        u_iter,
        phi,
        by,
        bz,
    )
    mean_velocity = jnp.sum(fluid_weight * u_iter) / fluid_total_weight
    return (
        u_iter,
        phi,
        jy,
        jz,
        lorentz,
        velocity_residual,
        potential_residual,
        potential_iteration_count,
        linear_residual,
        linear_iteration_count,
        face_current_max,
        emf_max,
        face_lorentz_max,
        mean_velocity,
        applied_forcing,
        potential_initial_residual,
        linear_initial_residual,
    )


def _fully_developed_converged(
    case: CaseSpec,
    *,
    velocity_residual: float,
    linear_residual: float,
    potential_residual: float,
) -> bool:
    """Apply the fully developed velocity, linear, and potential stop gate."""

    potential_gate = case.time_stepper.steady_potential_tolerance
    if potential_gate is None:
        potential_gate = case.time_stepper.potential_tolerance
    if potential_gate is None:
        potential_gate = case.time_stepper.steady_tolerance
    return bool(
        velocity_residual <= float(case.time_stepper.steady_tolerance)
        and linear_residual <= max(float(case.time_stepper.steady_tolerance), _LINEAR_RESIDUAL_FLOOR)
        and potential_residual <= float(potential_gate)
    )


def _solve_fully_developed(
    case: CaseSpec,
    logger=None,
    *,
    mesh: StructuredMesh | None = None,
    initial_state: MHDState | None = None,
    initial_diagnostics: Diagnostics | None = None,
    append_diagnostics: bool = False,
    restart_info: RestartLogInfo | None = None,
) -> Solution:
    mesh = _build_mesh(case) if mesh is None else mesh
    materials = build_material_fields(case, mesh)
    target_mean_velocity = _target_mean_velocity(case)
    reference_mean_velocity = _reference_mean_velocity(case)
    potential_solver = _resolve_potential_solver(case.time_stepper.potential_solver, materials.fluid_mask)
    if potential_solver == "cg" and not _has_uniform_spacing(mesh):
        potential_solver = "cg_volume"
    if case.geometry.kind not in {"rect_duct", "layered_duct"}:
        raise NotImplementedError(
            f"Solver {case.solver.kind!r} does not yet support geometry {case.geometry.kind!r}"
        )
    interpolate_direct_fluid_walls = case.geometry.kind == "rect_duct"
    initial_u, initial_phi, initial_jy, initial_jz, initial_lorentz, start_time = _initial_solver_state(
        case=case,
        mesh=mesh,
        fluid_mask=materials.fluid_mask,
        interpolate_direct_fluid_walls=interpolate_direct_fluid_walls,
        initial_state=initial_state,
    )
    dt = case.time_stepper.dt
    steady_mode = case.solver.mode == "steady"
    steps = _bounded_time_step_count(
        start_time=start_time,
        dt=dt,
        t_final=case.time_stepper.t_final,
        max_steps=case.time_stepper.max_steps,
    )
    if steady_mode:
        step_coupling_iterations = case.solver.coupling_iterations
        step_coupling_tolerance = float(case.time_stepper.steady_tolerance)
    else:
        step_coupling_iterations = case.solver.coupling_iterations
        step_coupling_tolerance = case.solver.coupling_tolerance
    _emit_solver_header(
        logger,
        case=case,
        mesh=mesh,
        materials=materials,
        mode=case.solver.mode,
        potential_solver=f"{potential_solver} / solvax_pcg",
        target_mean_velocity=target_mean_velocity,
        reference_mean_velocity=reference_mean_velocity,
        restart=restart_info,
    )

    u = initial_u
    phi = initial_phi
    jy = initial_jy
    jz = initial_jz
    lorentz = initial_lorentz
    time_history: list[float] = []
    u_max_history: list[float] = []
    mean_velocity_history: list[float] = []
    applied_forcing_history: list[float] = []
    pressure_proxy_history: list[float] = []
    current_scaled_pressure_proxy_history: list[float] = []
    residual_history: list[float] = []
    courant_history: list[float] = []
    ohmic_history: list[float] = []
    current_max_history: list[float] = []
    face_current_max_history: list[float] = []
    emf_max_history: list[float] = []
    lorentz_max_history: list[float] = []
    face_lorentz_max_history: list[float] = []
    potential_history: list[float] = []
    potential_iteration_history: list[float] = []
    linear_residual_history: list[float] = []
    linear_iteration_history: list[float] = []
    volumetric_flow_rate_history: list[float] = []
    mean_current_magnitude_history: list[float] = []
    lorentz_power_history: list[float] = []
    div_current_max_history: list[float] = []
    charge_balance_residual_history: list[float] = []
    gauge_residual_history: list[float] = []
    interface_current_residual_history: list[float] = []
    raw_update_max_history: list[float] = []
    limiter_scale_history: list[float] = []
    limited_fraction_history: list[float] = []
    pressure_proxy_reference_current = _pressure_proxy_reference_current(
        initial_diagnostics if append_diagnostics else None
    )
    residual_value = float(initial_state.residual if initial_state is not None else 0.0)
    step_count = 0

    for step_index in range(steps):
        step_time = float(start_time + (step_index + 1) * dt)
        u_before_step = u
        (
            u,
            phi,
            jy,
            jz,
            lorentz,
            residual,
            potential_residual,
            potential_iteration_count,
            linear_residual,
            _linear_iteration_count,
            face_current_max,
            emf_max,
            face_lorentz_max,
            mean_velocity,
            applied_forcing,
            potential_initial_residual,
            linear_initial_residual,
        ) = _fully_developed_case_step(
            case=case,
            mesh=mesh,
            materials=materials,
            u_previous=u,
            step_time=step_time,
            potential_solver=potential_solver,
            target_mean_velocity=target_mean_velocity,
            preconditioner=case.solver.preconditioner,
            coupling_iterations=step_coupling_iterations,
            coupling_tolerance=step_coupling_tolerance,
            phi_previous=phi,
        )
        outer_update = float(jnp.max(jnp.abs(u - u_before_step)))
        residual_value = max(float(residual), outer_update)
        u_max_value = float(jnp.max(jnp.abs(u)))
        courant_like = float(u_max_value * dt / jnp.min(mesh.dy))
        ohmic = float(jnp.mean(jy**2 + jz**2))
        max_current = float(jnp.max(jnp.sqrt(jy**2 + jz**2)))
        max_lorentz = float(jnp.max(jnp.abs(lorentz)))
        _, by_step, bz_step = magnetic_field_components(case.magnetic_field, mesh, time=step_time)
        (
            volumetric_flow_rate,
            mean_current_magnitude,
            lorentz_power,
            div_current_max,
            charge_balance_residual,
            gauge_residual,
            interface_current_residual,
        ) = _integral_diagnostics(
            mesh=mesh,
            sigma=materials.conductivity,
            fluid_mask=materials.fluid_mask,
            u=u,
            phi=phi,
            jy=jy,
            jz=jz,
            lorentz=lorentz,
            by=by_step,
            bz=bz_step,
            anchor=case.reference_phi_cell,
        )
        time_history.append(step_time)
        u_max_history.append(u_max_value)
        mean_velocity_history.append(float(mean_velocity))
        applied_forcing_history.append(float(applied_forcing))
        pressure_proxy_history.append(float(applied_forcing))
        residual_history.append(residual_value)
        courant_history.append(courant_like)
        ohmic_history.append(ohmic)
        current_max_history.append(max_current)
        face_current_max_history.append(float(face_current_max))
        emf_max_history.append(float(emf_max))
        lorentz_max_history.append(max_lorentz)
        face_lorentz_max_history.append(float(face_lorentz_max))
        potential_history.append(float(potential_residual))
        potential_iteration_history.append(float(potential_iteration_count))
        linear_residual_history.append(float(linear_residual))
        linear_iteration_history.append(float(_linear_iteration_count))
        volumetric_flow_rate_history.append(float(volumetric_flow_rate))
        mean_current_magnitude_history.append(float(mean_current_magnitude))
        lorentz_power_history.append(float(lorentz_power))
        div_current_max_history.append(float(div_current_max))
        charge_balance_residual_history.append(float(charge_balance_residual))
        gauge_residual_history.append(float(gauge_residual))
        interface_current_residual_history.append(float(interface_current_residual))
        raw_update_max_history.append(residual_value)
        limiter_scale_history.append(1.0)
        limited_fraction_history.append(0.0)
        current_scaled_pressure_proxy, pressure_proxy_reference_current = _scaled_pressure_proxy_value(
            float(applied_forcing),
            max_current,
            float(face_current_max),
            pressure_proxy_reference_current,
        )
        current_scaled_pressure_proxy_history.append(float(current_scaled_pressure_proxy))
        _emit_solver_step(
            logger,
            step_index=step_index + 1,
            step_time=step_time,
            dt=dt,
            u_max_value=u_max_value,
            mean_velocity=float(mean_velocity),
            max_current=max_current,
            face_current_max=float(face_current_max),
            emf_max=float(emf_max),
            max_lorentz=max_lorentz,
            face_lorentz_max=float(face_lorentz_max),
            residual_value=residual_value,
            potential_residual=float(potential_residual),
            potential_iteration_count=float(potential_iteration_count),
            linear_residual=float(linear_residual),
            linear_iteration_count=float(_linear_iteration_count),
            applied_forcing=float(applied_forcing),
            pressure_proxy=float(applied_forcing),
            current_scaled_pressure_proxy=float(current_scaled_pressure_proxy),
            raw_update_max=residual_value,
            limiter_scale=1.0,
            limited_fraction=0.0,
            courant_like=courant_like,
            ohmic=ohmic,
            volumetric_flow_rate=float(volumetric_flow_rate),
            mean_current_magnitude=float(mean_current_magnitude),
            lorentz_power=float(lorentz_power),
            div_current_max=float(div_current_max),
            charge_balance_residual=float(charge_balance_residual),
            gauge_residual=float(gauge_residual),
            interface_current_residual=float(interface_current_residual),
            potential_initial_residual=float(potential_initial_residual),
            linear_initial_residual=float(linear_initial_residual),
        )
        step_count = step_index + 1
        if steady_mode and _fully_developed_converged(
            case,
            velocity_residual=residual_value,
            linear_residual=float(linear_residual),
            potential_residual=float(potential_residual),
        ):
            break

    require_finite(
        "fully developed solve",
        velocity=u,
        potential=phi,
        current_y=jy,
        current_z=jz,
        lorentz_force=lorentz,
        residual=residual_value,
        residual_history=residual_history,
        potential_residual_history=potential_history,
        linear_residual_history=linear_residual_history,
    )
    steady_converged = bool(
        steady_mode
        and residual_history
        and _fully_developed_converged(
            case,
            velocity_residual=residual_history[-1],
            linear_residual=linear_residual_history[-1],
            potential_residual=potential_history[-1],
        )
    )
    state = MHDState(
        u=u,
        phi=phi,
        jy=jy,
        jz=jz,
        lorentz_x=lorentz,
        time=float(start_time + step_count * dt),
        residual=residual_value,
    )
    diagnostics = Diagnostics(
        time_history=_concat_history(
            initial_diagnostics.time_history if initial_diagnostics is not None else None,
            jnp.asarray(time_history, dtype=float),
            append=append_diagnostics,
        ),
        u_max_history=_concat_history(
            initial_diagnostics.u_max_history if initial_diagnostics is not None else None,
            jnp.asarray(u_max_history, dtype=float),
            append=append_diagnostics,
        ),
        mean_velocity_history=_concat_history(
            initial_diagnostics.mean_velocity_history if initial_diagnostics is not None else None,
            jnp.asarray(mean_velocity_history, dtype=float),
            append=append_diagnostics,
        ),
        applied_forcing_history=_concat_history(
            initial_diagnostics.applied_forcing_history if initial_diagnostics is not None else None,
            jnp.asarray(applied_forcing_history, dtype=float),
            append=append_diagnostics,
        ),
        pressure_proxy_history=_concat_history(
            initial_diagnostics.pressure_proxy_history if initial_diagnostics is not None else None,
            jnp.asarray(pressure_proxy_history, dtype=float),
            append=append_diagnostics,
        ),
        current_scaled_pressure_proxy_history=_concat_history(
            initial_diagnostics.current_scaled_pressure_proxy_history
            if initial_diagnostics is not None
            else None,
            jnp.asarray(current_scaled_pressure_proxy_history, dtype=float),
            append=append_diagnostics,
        ),
        raw_update_max_history=_concat_history(
            initial_diagnostics.raw_update_max_history if initial_diagnostics is not None else None,
            jnp.asarray(raw_update_max_history, dtype=float),
            append=append_diagnostics,
        ),
        limiter_scale_history=_concat_history(
            initial_diagnostics.limiter_scale_history if initial_diagnostics is not None else None,
            jnp.asarray(limiter_scale_history, dtype=float),
            append=append_diagnostics,
        ),
        limited_fraction_history=_concat_history(
            initial_diagnostics.limited_fraction_history if initial_diagnostics is not None else None,
            jnp.asarray(limited_fraction_history, dtype=float),
            append=append_diagnostics,
        ),
        residual_history=_concat_history(
            initial_diagnostics.residual_history if initial_diagnostics is not None else None,
            jnp.asarray(residual_history, dtype=float),
            append=append_diagnostics,
        ),
        courant_like=_concat_history(
            initial_diagnostics.courant_like if initial_diagnostics is not None else None,
            jnp.asarray(courant_history, dtype=float),
            append=append_diagnostics,
        ),
        ohmic_power=_concat_history(
            initial_diagnostics.ohmic_power if initial_diagnostics is not None else None,
            jnp.asarray(ohmic_history, dtype=float),
            append=append_diagnostics,
        ),
        current_max_history=_concat_history(
            initial_diagnostics.current_max_history if initial_diagnostics is not None else None,
            jnp.asarray(current_max_history, dtype=float),
            append=append_diagnostics,
        ),
        face_current_max_history=_concat_history(
            initial_diagnostics.face_current_max_history if initial_diagnostics is not None else None,
            jnp.asarray(face_current_max_history, dtype=float),
            append=append_diagnostics,
        ),
        emf_max_history=_concat_history(
            initial_diagnostics.emf_max_history if initial_diagnostics is not None else None,
            jnp.asarray(emf_max_history, dtype=float),
            append=append_diagnostics,
        ),
        lorentz_max_history=_concat_history(
            initial_diagnostics.lorentz_max_history if initial_diagnostics is not None else None,
            jnp.asarray(lorentz_max_history, dtype=float),
            append=append_diagnostics,
        ),
        face_lorentz_max_history=_concat_history(
            initial_diagnostics.face_lorentz_max_history if initial_diagnostics is not None else None,
            jnp.asarray(face_lorentz_max_history, dtype=float),
            append=append_diagnostics,
        ),
        potential_residual_history=_concat_history(
            initial_diagnostics.potential_residual_history if initial_diagnostics is not None else None,
            jnp.asarray(potential_history, dtype=float),
            append=append_diagnostics,
        ),
        potential_iterations_history=_concat_history(
            initial_diagnostics.potential_iterations_history if initial_diagnostics is not None else None,
            jnp.asarray(potential_iteration_history, dtype=float),
            append=append_diagnostics,
        ),
        linear_residual_history=_concat_history(
            initial_diagnostics.linear_residual_history if initial_diagnostics is not None else None,
            jnp.asarray(linear_residual_history, dtype=float),
            append=append_diagnostics,
        ),
        linear_iterations_history=_concat_history(
            initial_diagnostics.linear_iterations_history if initial_diagnostics is not None else None,
            jnp.asarray(linear_iteration_history, dtype=float),
            append=append_diagnostics,
        ),
        volumetric_flow_rate_history=_concat_history(
            initial_diagnostics.volumetric_flow_rate_history if initial_diagnostics is not None else None,
            jnp.asarray(volumetric_flow_rate_history, dtype=float),
            append=append_diagnostics,
        ),
        mean_current_magnitude_history=_concat_history(
            initial_diagnostics.mean_current_magnitude_history if initial_diagnostics is not None else None,
            jnp.asarray(mean_current_magnitude_history, dtype=float),
            append=append_diagnostics,
        ),
        lorentz_power_history=_concat_history(
            initial_diagnostics.lorentz_power_history if initial_diagnostics is not None else None,
            jnp.asarray(lorentz_power_history, dtype=float),
            append=append_diagnostics,
        ),
        div_current_max_history=_concat_history(
            initial_diagnostics.div_current_max_history if initial_diagnostics is not None else None,
            jnp.asarray(div_current_max_history, dtype=float),
            append=append_diagnostics,
        ),
        charge_balance_residual_history=_concat_history(
            initial_diagnostics.charge_balance_residual_history if initial_diagnostics is not None else None,
            jnp.asarray(charge_balance_residual_history, dtype=float),
            append=append_diagnostics,
        ),
        gauge_residual_history=_concat_history(
            initial_diagnostics.gauge_residual_history if initial_diagnostics is not None else None,
            jnp.asarray(gauge_residual_history, dtype=float),
            append=append_diagnostics,
        ),
        interface_current_residual_history=_concat_history(
            initial_diagnostics.interface_current_residual_history
            if initial_diagnostics is not None
            else None,
            jnp.asarray(interface_current_residual_history, dtype=float),
            append=append_diagnostics,
        ),
    )
    solution = Solution(
        mesh=mesh,
        state=state,
        diagnostics=diagnostics,
        case_name=case.name,
        converged=steady_converged if steady_mode else None,
        status=("converged" if steady_converged else "step_limit" if steady_mode else "completed"),
        steps=step_count,
    )
    if logger is not None:
        logger.emit_footer(solution)
    return solution


def solve_transient(
    case: CaseSpec,
    logger=None,
    *,
    mesh: StructuredMesh | None = None,
    initial_state: MHDState | None = None,
    initial_diagnostics: Diagnostics | None = None,
    append_diagnostics: bool = False,
    restart_info: RestartLogInfo | None = None,
) -> Solution:
    """Advance a supported case in transient mode, optionally from a restart."""

    solver_kind = getattr(getattr(case, "solver", None), "kind", "fully_developed_inductionless")
    if solver_kind == "fully_developed_inductionless":
        transient_case = (
            case
            if case.solver.mode == "transient"
            else case.__class__(
                **{
                    **case.__dict__,
                    "solver": case.solver.__class__(**{**case.solver.__dict__, "mode": "transient"}),
                }
            )
        )
        return _solve_fully_developed(
            transient_case,
            logger=logger,
            mesh=mesh,
            initial_state=initial_state,
            initial_diagnostics=initial_diagnostics,
            append_diagnostics=append_diagnostics,
            restart_info=restart_info,
        )
    raise NotImplementedError(f"Solver kind {solver_kind!r} is not implemented for transient runs")


def solve_steady(
    case: CaseSpec,
    logger=None,
    *,
    mesh: StructuredMesh | None = None,
    initial_state: MHDState | None = None,
    initial_diagnostics: Diagnostics | None = None,
    append_diagnostics: bool = False,
    restart_info: RestartLogInfo | None = None,
) -> Solution:
    """Solve a supported case to its configured steady-state stopping gate."""

    solver_kind = getattr(getattr(case, "solver", None), "kind", "fully_developed_inductionless")
    if solver_kind == "fully_developed_inductionless":
        steady_case = (
            case
            if case.solver.mode == "steady"
            else case.__class__(
                **{
                    **case.__dict__,
                    "solver": case.solver.__class__(**{**case.solver.__dict__, "mode": "steady"}),
                }
            )
        )
        return _solve_fully_developed(
            steady_case,
            logger=logger,
            mesh=mesh,
            initial_state=initial_state,
            initial_diagnostics=initial_diagnostics,
            append_diagnostics=append_diagnostics,
            restart_info=restart_info,
        )
    raise NotImplementedError(f"Solver kind {solver_kind!r} is not implemented for steady runs")


@dataclass(frozen=True)
class HartmannAutodiffProblem:
    mesh: StructuredMesh
    sigma: jnp.ndarray
    rho: jnp.ndarray
    nu: jnp.ndarray
    fluid_mask: jnp.ndarray
    anchor: tuple[int, int] = (0, 0)
    potential_iterations: int = 80
    velocity_iterations: int = 120
    macro_iterations: int = 8
    relaxation: float = 0.9


def build_hartmann_autodiff_problem(
    *,
    ny: int = 48,
    nz: int = 48,
    width: float = 2.0,
    height: float = 2.0,
    conductivity: float = 1.0,
    density: float = 1.0,
    viscosity: float = 0.01,
    potential_iterations: int = 80,
    velocity_iterations: int = 120,
    macro_iterations: int = 8,
    relaxation: float = 0.9,
) -> HartmannAutodiffProblem:
    mesh = generate_rect_duct_mesh(width=width, height=height, ny=ny, nz=nz)
    yz_shape = mesh.yz_shape
    return HartmannAutodiffProblem(
        mesh=mesh,
        sigma=jnp.full(yz_shape, conductivity),
        rho=jnp.full(yz_shape, density),
        nu=jnp.full(yz_shape, viscosity),
        fluid_mask=jnp.ones(yz_shape, dtype=bool),
        potential_iterations=potential_iterations,
        velocity_iterations=velocity_iterations,
        macro_iterations=macro_iterations,
        relaxation=relaxation,
    )


def _solve_velocity_jacobi_state(
    *,
    mesh: StructuredMesh,
    diffusivity: jnp.ndarray,
    reaction: jnp.ndarray,
    rhs: jnp.ndarray,
    active_mask: jnp.ndarray,
    iterations: int,
    relaxation: float,
) -> jnp.ndarray:
    diagonal, west, east, south, north = _velocity_system_coefficients(
        mesh, diffusivity, reaction, active_mask
    )
    diagonal = jnp.maximum(diagonal, 1.0e-12)
    field0 = jnp.zeros_like(rhs)

    def update(field):
        west_field = jnp.pad(field[:-1, :], ((1, 0), (0, 0)))
        east_field = jnp.pad(field[1:, :], ((0, 1), (0, 0)))
        south_field = jnp.pad(field[:, :-1], ((0, 0), (1, 0)))
        north_field = jnp.pad(field[:, 1:], ((0, 0), (0, 1)))
        updated = (
            rhs + west * west_field + east * east_field + south * south_field + north * north_field
        ) / diagonal
        return jnp.where(active_mask, updated, 0.0)

    return fixed_point_iteration(
        update,
        field0,
        relaxation=relaxation,
        rtol=0.0,
        atol=0.0,
        max_steps=iterations,
        fixed_steps=True,
    ).x


def solve_differentiable_hartmann(
    problem: HartmannAutodiffProblem,
    *,
    forcing: float | jnp.ndarray,
    hartmann_number: float | jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    mesh = problem.mesh
    sigma = problem.sigma
    rho = problem.rho
    nu = problem.nu
    fluid_mask = problem.fluid_mask
    by = jnp.zeros(mesh.yz_shape, dtype=sigma.dtype)
    bz = jnp.full(mesh.yz_shape, hartmann_number, dtype=sigma.dtype)
    forcing_value = jnp.asarray(forcing, dtype=sigma.dtype)

    def macro_body(_, u_iter):
        uxb_y = jnp.where(fluid_mask, -u_iter * bz, 0.0)
        uxb_z = jnp.where(fluid_mask, u_iter * by, 0.0)
        conv_y = _face_emf(mesh, sigma, uxb_y, axis=0)
        conv_z = _face_emf(mesh, sigma, uxb_z, axis=1)
        face_conv_y = jnp.pad(conv_y, ((1, 1), (0, 0)))
        face_conv_z = jnp.pad(conv_z, ((0, 0), (1, 1)))
        rhs_phi = -(
            (face_conv_y[1:, :] - face_conv_y[:-1, :]) / mesh.dy[:, None]
            + (face_conv_z[:, 1:] - face_conv_z[:, :-1]) / mesh.dz[None, :]
        )
        diagonal, west, east, south, north = _potential_coefficients(mesh, sigma)
        phi, _, _ = solve_poisson_jacobi_state(
            diagonal,
            west,
            east,
            south,
            north,
            rhs_phi,
            problem.anchor,
            problem.potential_iterations,
            tolerance=None,
            relaxation=problem.relaxation,
        )
        reaction = sigma * (bz**2 + by**2) / rho
        rhs_u, _ = _fully_developed_rhs(
            mesh=mesh,
            sigma=sigma,
            rho=rho,
            fluid_mask=fluid_mask,
            u=u_iter,
            phi=phi,
            by=by,
            bz=bz,
            forcing=forcing_value,
        )
        u_next = _solve_velocity_jacobi_state(
            mesh=mesh,
            diffusivity=nu,
            reaction=reaction,
            rhs=rhs_u,
            active_mask=fluid_mask,
            iterations=problem.velocity_iterations,
            relaxation=problem.relaxation,
        )
        return _enforce_velocity_bc(u_next, mesh, fluid_mask, interpolate_direct_fluid_walls=False)

    u = jax.lax.fori_loop(
        0,
        problem.macro_iterations,
        macro_body,
        jnp.zeros(mesh.yz_shape, dtype=sigma.dtype),
    )

    uxb_y = jnp.where(fluid_mask, -u * bz, 0.0)
    uxb_z = jnp.where(fluid_mask, u * by, 0.0)
    conv_y = _face_emf(mesh, sigma, uxb_y, axis=0)
    conv_z = _face_emf(mesh, sigma, uxb_z, axis=1)
    face_conv_y = jnp.pad(conv_y, ((1, 1), (0, 0)))
    face_conv_z = jnp.pad(conv_z, ((0, 0), (1, 1)))
    rhs_phi = -(
        (face_conv_y[1:, :] - face_conv_y[:-1, :]) / mesh.dy[:, None]
        + (face_conv_z[:, 1:] - face_conv_z[:, :-1]) / mesh.dz[None, :]
    )
    diagonal, west, east, south, north = _potential_coefficients(mesh, sigma)
    phi, _, _ = solve_poisson_jacobi_state(
        diagonal,
        west,
        east,
        south,
        north,
        rhs_phi,
        problem.anchor,
        problem.potential_iterations,
        tolerance=None,
        relaxation=problem.relaxation,
    )
    return u, phi


def hartmann_current_proxy(
    problem: HartmannAutodiffProblem,
    *,
    forcing: float | jnp.ndarray,
    hartmann_number: float | jnp.ndarray,
) -> jnp.ndarray:
    u, phi = solve_differentiable_hartmann(problem, forcing=forcing, hartmann_number=hartmann_number)
    dphi_dy, _ = gradient_scalar(phi, problem.mesh)
    jy = -dphi_dy - u * jnp.asarray(hartmann_number, dtype=u.dtype)
    return jnp.mean(jnp.abs(jy))


def hartmann_mean_velocity(
    problem: HartmannAutodiffProblem,
    *,
    forcing: float | jnp.ndarray,
    hartmann_number: float | jnp.ndarray,
) -> jnp.ndarray:
    u, _ = solve_differentiable_hartmann(problem, forcing=forcing, hartmann_number=hartmann_number)
    return jnp.mean(u)


def hartmann_mean_velocity_gradients(
    problem: HartmannAutodiffProblem,
    *,
    forcing: float | jnp.ndarray,
    hartmann_number: float | jnp.ndarray,
) -> dict[str, jnp.ndarray]:
    def objective(force_value, ha_value):
        return hartmann_mean_velocity(
            problem,
            forcing=force_value,
            hartmann_number=ha_value,
        )

    mean_velocity = objective(forcing, hartmann_number)
    d_mean_velocity_d_forcing, d_mean_velocity_d_ha = jax.grad(objective, argnums=(0, 1))(
        forcing, hartmann_number
    )
    return {
        "mean_velocity": mean_velocity,
        "d_mean_velocity_d_forcing": d_mean_velocity_d_forcing,
        "d_mean_velocity_d_ha": d_mean_velocity_d_ha,
    }


def hartmann_mean_velocity_finite_difference_gradients(
    problem: HartmannAutodiffProblem,
    *,
    forcing: float | jnp.ndarray,
    hartmann_number: float | jnp.ndarray,
    delta_forcing: float = 1.0e-3,
    delta_ha: float = 1.0e-3,
) -> dict[str, jnp.ndarray]:
    mean_velocity = hartmann_mean_velocity(problem, forcing=forcing, hartmann_number=hartmann_number)
    plus_forcing = hartmann_mean_velocity(
        problem,
        forcing=jnp.asarray(forcing) + delta_forcing,
        hartmann_number=hartmann_number,
    )
    minus_forcing = hartmann_mean_velocity(
        problem,
        forcing=jnp.asarray(forcing) - delta_forcing,
        hartmann_number=hartmann_number,
    )
    plus_ha = hartmann_mean_velocity(
        problem,
        forcing=forcing,
        hartmann_number=jnp.asarray(hartmann_number) + delta_ha,
    )
    minus_ha = hartmann_mean_velocity(
        problem,
        forcing=forcing,
        hartmann_number=jnp.asarray(hartmann_number) - delta_ha,
    )
    return {
        "mean_velocity": mean_velocity,
        "d_mean_velocity_d_forcing": (plus_forcing - minus_forcing) / (2.0 * delta_forcing),
        "d_mean_velocity_d_ha": (plus_ha - minus_ha) / (2.0 * delta_ha),
    }


def hartmann_profile_loss(
    problem: HartmannAutodiffProblem,
    *,
    forcing: float | jnp.ndarray,
    hartmann_number: float | jnp.ndarray,
    target_profile: jnp.ndarray,
) -> jnp.ndarray:
    u, _ = solve_differentiable_hartmann(problem, forcing=forcing, hartmann_number=hartmann_number)
    centerline = u[:, u.shape[1] // 2]
    centerline_scale = jnp.maximum(jnp.max(jnp.abs(centerline)), 1.0e-12)
    target_scale = jnp.maximum(jnp.max(jnp.abs(target_profile)), 1.0e-12)
    return jnp.mean((centerline / centerline_scale - target_profile / target_scale) ** 2)


def hartmann_profile_loss_gradients(
    problem: HartmannAutodiffProblem,
    *,
    forcing: float | jnp.ndarray,
    hartmann_number: float | jnp.ndarray,
    target_profile: jnp.ndarray,
) -> dict[str, jnp.ndarray]:
    def objective(force_value, ha_value):
        return hartmann_profile_loss(
            problem,
            forcing=force_value,
            hartmann_number=ha_value,
            target_profile=target_profile,
        )

    loss, (d_loss_d_forcing, d_loss_d_ha) = jax.value_and_grad(objective, argnums=(0, 1))(
        forcing, hartmann_number
    )
    return {
        "loss": loss,
        "d_loss_d_forcing": d_loss_d_forcing,
        "d_loss_d_ha": d_loss_d_ha,
    }


def run_hartmann_profile_inverse_design(
    problem: HartmannAutodiffProblem,
    *,
    target_profile: jnp.ndarray,
    forcing_init: float,
    hartmann_init: float,
    learning_rate_forcing: float = 20.0,
    learning_rate_ha: float = 5.0,
    steps: int = 24,
) -> dict[str, object]:
    forcing = jnp.asarray(forcing_init, dtype=jnp.float32)
    hartmann_number = jnp.asarray(hartmann_init, dtype=jnp.float32)
    history: list[dict[str, float]] = []
    for step in range(steps):
        gradients = hartmann_profile_loss_gradients(
            problem,
            forcing=forcing,
            hartmann_number=hartmann_number,
            target_profile=target_profile,
        )
        history.append(
            {
                "iteration": float(step),
                "forcing": float(forcing),
                "hartmann_number": float(hartmann_number),
                "loss": float(gradients["loss"]),
                "d_loss_d_forcing": float(gradients["d_loss_d_forcing"]),
                "d_loss_d_ha": float(gradients["d_loss_d_ha"]),
            }
        )
        forcing = jnp.clip(forcing - learning_rate_forcing * gradients["d_loss_d_forcing"], 0.05, 5.0)
        hartmann_number = jnp.clip(hartmann_number - learning_rate_ha * gradients["d_loss_d_ha"], 0.5, 40.0)

    recovered_u, recovered_phi = solve_differentiable_hartmann(
        problem,
        forcing=forcing,
        hartmann_number=hartmann_number,
    )
    recovered_profile = recovered_u[:, recovered_u.shape[1] // 2]
    return {
        "forcing": float(forcing),
        "hartmann_number": float(hartmann_number),
        "history": history,
        "recovered_profile": recovered_profile,
        "recovered_phi_max": float(jnp.max(jnp.abs(recovered_phi))),
    }

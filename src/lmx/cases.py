"""Fully developed steady and transient inductionless solvers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from solvax import affine_fixed_point_gmres
from solvax import (
    aitken_relaxation as _solvax_aitken_relaxation,
)
from solvax import (
    anderson_mixing as _solvax_anderson_mixing,
)

from .mesh import StructuredMesh
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
    _fully_developed_rhs,
    _has_uniform_spacing,
    _initial_solver_state,
    _integral_diagnostics,
    _limited_velocity_update,
    _nested_velocity_tolerance,
    _PotentialSystem,
    _prepare_potential_system,
    _reference_mean_velocity,
    _resolve_potential_solver,
    _solve_potential,
    _solve_velocity_system,
    _target_mean_velocity,
    _velocity_system_coefficients,
)
from .specs import (
    BoundaryCondition,
    CaseSpec,
    Diagnostics,
    ExtrudedInductionlessProblem,
    ExtrudedInductionlessSolution,
    GeometrySpec,
    MagneticFieldSpec,
    MHDState,
    OutputSpec,
    RegionSpec,
    RestartLogInfo,
    Solution,
    SolverConfig,
    StreamingSolverLogger,
    TimeStepperConfig,
    require_finite,
)

if TYPE_CHECKING:
    from .q2d import Q2DProblem, Q2DResult

_STEP_DIAGNOSTIC_NAMES = (
    "u_max_history",
    "mean_velocity_history",
    "applied_forcing_history",
    "residual_history",
    "courant_like",
    "ohmic_power",
    "current_max_history",
    "face_current_max_history",
    "emf_max_history",
    "lorentz_max_history",
    "face_lorentz_max_history",
    "potential_residual_history",
    "potential_iterations_history",
    "linear_residual_history",
    "linear_iterations_history",
    "volumetric_flow_rate_history",
    "mean_current_magnitude_history",
    "lorentz_power_history",
    "div_current_max_history",
    "charge_balance_residual_history",
    "gauge_residual_history",
    "interface_current_residual_history",
)
_VelocitySystem = tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    tuple[jnp.ndarray, ...],
    jnp.ndarray,
]


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
            potential_iterations=400,
            relaxation=0.08,
            velocity_update_limit=2e-3,
        )
    if ha <= 100.0:
        return TimeStepperConfig(
            dt=0.002,
            t_final=1.0,
            max_steps=500,
            potential_iterations=400,
            relaxation=0.1,
            velocity_update_limit=1e-3,
        )
    return TimeStepperConfig(
        dt=0.002,
        t_final=1.0,
        max_steps=500,
        potential_iterations=400,
        relaxation=0.1,
        velocity_update_limit=1e-3,
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
    velocity_system: _VelocitySystem | None = None,
    potential_system: _PotentialSystem | None = None,
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
    if velocity_system is None:
        velocity_system = _prepare_fully_developed_velocity_system(case, mesh, materials, step_time)
    by, bz, magnetic_reaction, velocity_coefficients, cell_metric = velocity_system
    forcing = jnp.asarray(case.forcing, dtype=by.dtype)
    fluid_mask = materials.fluid_mask
    active_mask = fluid_mask
    u_iter = u_previous
    phi_iter = jnp.zeros_like(u_previous) if phi_previous is None else phi_previous
    dt = case.time_stepper.dt
    fluid_weight = jnp.where(fluid_mask, cell_metric, 0.0)
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
    if potential_system is None:
        potential_system = _prepare_potential_system(
            mesh, materials.conductivity, case.reference_phi_cell, potential_solver
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
            flexible=potential_system.flexible,
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
            system=potential_system,
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
        )
        rhs_base = rhs_base + magnetic_reaction * jnp.where(active_mask, u_iter, 0.0)
        if not steady_mode:
            rhs_base = rhs_base + jnp.where(active_mask, u_previous / dt, 0.0)
        if target_mean_velocity is None:
            rhs = rhs_base + jnp.where(active_mask, forcing / materials.density, 0.0)
            u_next, velocity_linear_residual, linear_iteration_count, linear_initial_residual = (
                _solve_velocity_system(
                    coefficients=velocity_coefficients,
                    cell_metric=cell_metric,
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
                    coefficients=velocity_coefficients,
                    cell_metric=cell_metric,
                    rhs=rhs_base,
                    active_mask=active_mask,
                    preconditioner=preconditioner,
                    max_steps=max(case.time_stepper.max_steps, case.solver.coupling_iterations * 25),
                    tolerance=velocity_linear_tolerance,
                )
            )
            u_sensitivity, _, _, sensitivity_initial_residual = _solve_velocity_system(
                coefficients=velocity_coefficients,
                cell_metric=cell_metric,
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
        system=potential_system,
    )
    jy, jz, lorentz = _compute_current_and_lorentz(
        mesh,
        materials.conductivity,
        fluid_mask,
        u_iter,
        phi,
        by,
        bz,
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


def _prepare_fully_developed_velocity_system(
    case: CaseSpec,
    mesh: StructuredMesh,
    materials,
    step_time: float,
) -> _VelocitySystem:
    """Assemble the velocity system once for a fixed material and field state."""

    _, by, bz = magnetic_field_components(case.magnetic_field, mesh, time=step_time)
    active_mask = materials.fluid_mask
    magnetic_reaction = jnp.where(
        active_mask,
        materials.conductivity * (by**2 + bz**2) / materials.density,
        0.0,
    )
    reaction = magnetic_reaction
    if case.solver.mode != "steady":
        reaction = reaction + jnp.where(active_mask, 1.0 / case.time_stepper.dt, 0.0)
    cell_metric = _cell_metric(mesh).astype(materials.conductivity.dtype)
    coefficients = tuple(
        coefficient * cell_metric
        for coefficient in _velocity_system_coefficients(mesh, materials.viscosity, reaction, active_mask)
    )
    return by, bz, magnetic_reaction, coefficients, cell_metric


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


def _prepare_fully_developed_case(case: CaseSpec, mesh: StructuredMesh | None = None):
    """Prepare shared mesh, materials, and potential algebra for field solves."""

    mesh = _build_mesh(case) if mesh is None else mesh
    materials = build_material_fields(case, mesh)
    potential_solver = _resolve_potential_solver(case.time_stepper.potential_solver, materials.fluid_mask)
    if potential_solver == "cg" and not _has_uniform_spacing(mesh):
        potential_solver = "cg_volume"
    potential_system = _prepare_potential_system(
        mesh, materials.conductivity, case.reference_phi_cell, potential_solver
    )
    return mesh, materials, potential_solver, potential_system


def _solve_fully_developed(
    case: CaseSpec,
    logger: StreamingSolverLogger | None = None,
    *,
    mesh: StructuredMesh | None = None,
    initial_state: MHDState | None = None,
    initial_diagnostics: Diagnostics | None = None,
    append_diagnostics: bool = False,
    restart_info: RestartLogInfo | None = None,
) -> Solution:
    if case.output.history_stride < 0:
        raise ValueError("history_stride must be non-negative")
    mesh, materials, potential_solver, potential_system = _prepare_fully_developed_case(case, mesh)
    target_mean_velocity = _target_mean_velocity(case)
    reference_mean_velocity = _reference_mean_velocity(case)
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
    history_values: dict[str, list[float]] = {name: [] for name in ("time_history", *_STEP_DIAGNOSTIC_NAMES)}
    last_step_diagnostics: dict[str, float] = {}
    residual_value = float(initial_state.residual if initial_state is not None else 0.0)
    step_count = 0
    fixed_velocity_system = (
        _prepare_fully_developed_velocity_system(case, mesh, materials, start_time)
        if case.magnetic_field.ramp_duration <= 0.0
        else None
    )

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
            velocity_system=fixed_velocity_system,
            potential_system=potential_system,
        )
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
        u_max = jnp.max(jnp.abs(u))
        device_diagnostics = jnp.stack(
            (
                u_max,
                mean_velocity,
                applied_forcing,
                jnp.maximum(residual, jnp.max(jnp.abs(u - u_before_step))),
                u_max * dt / jnp.min(mesh.dy),
                jnp.mean(jy**2 + jz**2),
                jnp.max(jnp.sqrt(jy**2 + jz**2)),
                face_current_max,
                emf_max,
                jnp.max(jnp.abs(lorentz)),
                face_lorentz_max,
                potential_residual,
                potential_iteration_count,
                linear_residual,
                _linear_iteration_count,
                volumetric_flow_rate,
                mean_current_magnitude,
                lorentz_power,
                div_current_max,
                charge_balance_residual,
                gauge_residual,
                interface_current_residual,
            )
        )
        last_step_diagnostics = {
            "time_history": step_time,
            **dict(zip(_STEP_DIAGNOSTIC_NAMES, map(float, jax.device_get(device_diagnostics)), strict=True)),
        }
        residual_value = last_step_diagnostics["residual_history"]
        retain_step = case.output.history_stride == 0 or step_index % max(case.output.history_stride, 1) == 0
        if retain_step:
            for name, value in last_step_diagnostics.items():
                if case.output.history_stride == 0:
                    history_values[name][:] = (value,)
                else:
                    history_values[name].append(value)
        _emit_solver_step(
            logger,
            step_index=step_index + 1,
            step_time=step_time,
            u_max_value=last_step_diagnostics["u_max_history"],
            mean_velocity=last_step_diagnostics["mean_velocity_history"],
            max_current=last_step_diagnostics["current_max_history"],
            max_lorentz=last_step_diagnostics["lorentz_max_history"],
            residual_value=residual_value,
            potential_residual=last_step_diagnostics["potential_residual_history"],
            potential_iteration_count=last_step_diagnostics["potential_iterations_history"],
            linear_residual=last_step_diagnostics["linear_residual_history"],
            linear_iteration_count=last_step_diagnostics["linear_iterations_history"],
            applied_forcing=last_step_diagnostics["applied_forcing_history"],
            courant_like=last_step_diagnostics["courant_like"],
            ohmic=last_step_diagnostics["ohmic_power"],
            volumetric_flow_rate=last_step_diagnostics["volumetric_flow_rate_history"],
            div_current_max=last_step_diagnostics["div_current_max_history"],
            charge_balance_residual=last_step_diagnostics["charge_balance_residual_history"],
            gauge_residual=last_step_diagnostics["gauge_residual_history"],
            interface_current_residual=last_step_diagnostics["interface_current_residual_history"],
            potential_initial_residual=float(potential_initial_residual),
            linear_initial_residual=float(linear_initial_residual),
        )
        step_count = step_index + 1
        if steady_mode and _fully_developed_converged(
            case,
            velocity_residual=residual_value,
            linear_residual=last_step_diagnostics["linear_residual_history"],
            potential_residual=last_step_diagnostics["potential_residual_history"],
        ):
            break

    if case.output.history_stride > 1 and step_count and (step_count - 1) % case.output.history_stride != 0:
        for name, value in last_step_diagnostics.items():
            history_values[name].append(value)

    require_finite(
        "fully developed solve",
        velocity=u,
        potential=phi,
        current_y=jy,
        current_z=jz,
        lorentz_force=lorentz,
        residual=residual_value,
        residual_history=history_values["residual_history"],
        potential_residual_history=history_values["potential_residual_history"],
        linear_residual_history=history_values["linear_residual_history"],
    )
    steady_converged = bool(
        steady_mode
        and step_count
        and _fully_developed_converged(
            case,
            velocity_residual=last_step_diagnostics["residual_history"],
            linear_residual=last_step_diagnostics["linear_residual_history"],
            potential_residual=last_step_diagnostics["potential_residual_history"],
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

    def retained_history(name, values):
        array = jnp.asarray(values, dtype=float)
        stride = case.output.history_stride
        initial = (
            getattr(initial_diagnostics, name) if initial_diagnostics is not None and stride != 0 else None
        )
        return _concat_history(initial, array, append=append_diagnostics)

    diagnostics = Diagnostics(
        **{name: retained_history(name, values) for name, values in history_values.items()}
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
    logger: StreamingSolverLogger | None = None,
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
    logger: StreamingSolverLogger | None = None,
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


def solve(
    model: CaseSpec | ExtrudedInductionlessProblem | Q2DProblem,
) -> Solution | ExtrudedInductionlessSolution | Q2DResult:
    """Solve a fully developed, three-dimensional fringing, or Q2D problem.

    The configured mode selects steady or transient execution for ``CaseSpec``.
    Advanced restart, mesh, logging, progress, and timing hooks remain on the
    specialized functions in :mod:`lmx.cases` and :mod:`lmx.fringing`.
    """

    if isinstance(model, CaseSpec):
        return solve_transient(model) if model.solver.mode == "transient" else solve_steady(model)
    if isinstance(model, ExtrudedInductionlessProblem):
        from .fringing import solve_extruded_inductionless

        return solve_extruded_inductionless(model)
    from .q2d import Q2DProblem, solve_q2d

    if isinstance(model, Q2DProblem):
        return solve_q2d(model)
    raise TypeError(
        f"solve expects CaseSpec, ExtrudedInductionlessProblem, or Q2DProblem, got {type(model).__name__}"
    )


def solve_fully_developed_fields(
    case: CaseSpec,
    *,
    forcing: float | jax.Array | None = None,
    magnetic_field_scale: float | jax.Array = 1.0,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return steady duct fields through the production discretization.

    ``forcing`` and ``magnetic_field_scale`` are continuous design inputs.
    The coupled affine state uses a SOLVAX implicit tangent/transpose solve,
    so reverse mode does not retain potential, momentum, or coupling
    iterations. Meshes, material regions, boundary kinds, and solver controls
    are static; construct or close over ``case`` outside ``jit``.
    """

    if case.solver.kind != "fully_developed_inductionless":
        raise ValueError("case must select the fully developed inductionless solver")
    if case.geometry.kind not in {"rect_duct", "layered_duct"}:
        raise NotImplementedError(f"differentiable fully developed fields do not support {case.geometry.kind!r}")
    if case.magnetic_field.ramp_duration > 0.0:
        raise ValueError("steady differentiable fields require an unramped magnetic field")
    if _target_mean_velocity(case) is not None:
        raise NotImplementedError("fixed-flow differentiation is not yet supported")
    with jax.ensure_compile_time_eval():
        mesh, materials, potential_solver, potential_system = _prepare_fully_developed_case(case)
        _, by, bz = magnetic_field_components(case.magnetic_field, mesh)
    field_scale = jnp.asarray(magnetic_field_scale, dtype=by.dtype)
    by, bz = field_scale * by, field_scale * bz
    fluid_mask = materials.fluid_mask
    reaction = jnp.where(
        fluid_mask,
        materials.conductivity * (by**2 + bz**2) / materials.density,
        0.0,
    )
    cell_metric = _cell_metric(mesh).astype(by.dtype)
    coefficients = tuple(
        coefficient * cell_metric
        for coefficient in _velocity_system_coefficients(mesh, materials.viscosity, reaction, fluid_mask)
    )
    tolerance = _nested_velocity_tolerance(case.solver.coupling_tolerance, by.dtype)
    source = jnp.asarray(case.forcing if forcing is None else forcing, dtype=by.dtype)
    max_steps = max(case.time_stepper.max_steps, case.solver.coupling_iterations * 25)

    def potential(velocity):
        return _solve_potential(
            mesh,
            materials.conductivity,
            fluid_mask,
            velocity,
            by,
            bz,
            case.reference_phi_cell,
            case.time_stepper.potential_iterations,
            tolerance=tolerance,
            solver=potential_solver,
            system=potential_system,
        )[0]

    def mapping(velocity):
        phi = potential(velocity)
        rhs, _ = _fully_developed_rhs(
            mesh=mesh,
            sigma=materials.conductivity,
            rho=materials.density,
            fluid_mask=fluid_mask,
            u=velocity,
            phi=phi,
            by=by,
            bz=bz,
            forcing=source,
        )
        velocity, _, _, _ = _solve_velocity_system(
            coefficients=coefficients,
            cell_metric=cell_metric,
            rhs=rhs + reaction * jnp.where(fluid_mask, velocity, 0.0),
            active_mask=fluid_mask,
            preconditioner=case.solver.preconditioner,
            max_steps=max_steps,
            tolerance=tolerance,
        )
        return _enforce_velocity_bc(
            velocity,
            mesh,
            fluid_mask,
            interpolate_direct_fluid_walls=case.geometry.kind == "rect_duct",
        )

    zero = jnp.zeros(mesh.yz_shape, dtype=by.dtype)
    restart = min(30, max(2, mesh.ny * mesh.nz))
    max_restarts = 50
    coupled = affine_fixed_point_gmres(
        mapping,
        zero,
        restart=restart,
        rtol=case.solver.coupling_tolerance,
        max_restarts=max_restarts,
        transpose_rtol=case.solver.coupling_tolerance,
        transpose_max_restarts=max_restarts,
    )
    phi = potential(coupled.x)
    jy, jz, lorentz = _compute_current_and_lorentz(
        mesh, materials.conductivity, fluid_mask, coupled.x, phi, by, bz
    )
    return coupled.x, phi, jy, jz, lorentz

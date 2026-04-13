from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

import jax.numpy as jnp


RegionKind = Literal["fluid", "solid"]
GeometryKind = Literal["rect_duct", "layered_duct", "pipe_ogrid"]
SolverKind = Literal["fully_developed_inductionless", "reduced_inductionless", "extruded_inductionless"]
SolveMode = Literal["steady", "transient"]
BoundaryKind = Literal[
    "no_slip",
    "insulating",
    "conducting_wall",
    "inlet_velocity",
    "inlet_flow_rate",
    "outlet_pressure",
    "imposed_current_density",
]
MagneticFieldKind = Literal["constant", "analytic", "tabulated"]
PotentialSolverKind = Literal["auto", "jacobi", "cg", "cg_volume", "lineax_cg"]
CurrentReconstructionKind = Literal["cell_centered", "face_averaged", "hybrid_face_lorentz"]
VelocityUpdateLimiterKind = Literal["global_scale", "local_clip"]
LinearSolverKind = Literal["auto", "cg", "gmres", "bicgstab"]
PreconditionerKind = Literal["none", "jacobi", "block_jacobi"]
TimeSchemeKind = Literal["implicit_euler", "crank_nicolson"]


@dataclass(frozen=True)
class RegionSpec:
    name: str
    kind: RegionKind
    conductivity: float
    density: float | None = None
    viscosity: float | None = None
    wall_thickness: float | None = None


@dataclass(frozen=True)
class BoundaryCondition:
    name: str
    kind: BoundaryKind
    value: float | tuple[float, float, float] | None = None
    region: str | None = None
    axis: str | None = None
    side: str | None = None


@dataclass(frozen=True)
class MagneticFieldSpec:
    kind: MagneticFieldKind
    value: tuple[float, float, float] | None = None
    fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] | None = None
    table_path: str | None = None
    ramp_start: float = 0.0
    ramp_duration: float = 0.0


@dataclass(frozen=True)
class SolverConfig:
    kind: SolverKind = "fully_developed_inductionless"
    mode: SolveMode = "steady"
    linear_solver: LinearSolverKind = "auto"
    preconditioner: PreconditionerKind = "jacobi"
    time_scheme: TimeSchemeKind = "implicit_euler"
    coupling_iterations: int = 12
    coupling_tolerance: float = 1e-8


@dataclass(frozen=True)
class TimeStepperConfig:
    dt: float
    t_final: float
    max_steps: int
    # Legacy reduced-solver controls kept for regression and fallback only.
    outer_iterations: int = 2
    potential_iterations: int = 400
    potential_tolerance: float | None = None
    potential_relaxation: float = 1.0
    potential_solver: PotentialSolverKind = "auto"
    current_reconstruction: CurrentReconstructionKind = "cell_centered"
    post_update_potential_refresh: bool = False
    steady_tolerance: float = 1e-8
    steady_potential_tolerance: float | None = None
    relaxation: float = 0.35
    velocity_update_limit: float = 1e-3
    velocity_update_limiter: VelocityUpdateLimiterKind = "global_scale"
    checkpoint_stride: int = 1


@dataclass(frozen=True)
class OutputSpec:
    directory: str | None = None
    write_paraview: bool = True
    write_csv_profiles: bool = True
    write_npz: bool = True
    write_json_summary: bool = True
    write_plots: bool = False
    copy_input_file: bool = True
    write_stride: int = 1


@dataclass(frozen=True)
class GeometrySpec:
    kind: GeometryKind
    width: float
    height: float
    length: float = 1.0
    nx: int = 1
    ny: int = 64
    nz: int = 64
    radius: float | None = None
    nr: int | None = None
    ntheta: int | None = None
    wall_thickness: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    wall_cells: tuple[int, int, int, int] = (0, 0, 0, 0)
    target_ha: float | None = None
    target_side_layer: float | None = None


@dataclass(frozen=True)
class CaseSpec:
    name: str
    geometry: GeometrySpec
    regions: tuple[RegionSpec, ...]
    magnetic_field: MagneticFieldSpec
    boundary_conditions: tuple[BoundaryCondition, ...]
    time_stepper: TimeStepperConfig
    solver: SolverConfig = field(default_factory=SolverConfig)
    output: OutputSpec = field(default_factory=OutputSpec)
    forcing: float = 1.0
    initial_velocity: float = 0.0
    reference_pressure_gradient: float = -1.0
    reference_phi_cell: tuple[int, int] = (0, 0)
    notes: str = ""

    @property
    def output_dir(self) -> Path | None:
        if self.output.directory is None:
            return None
        return Path(self.output.directory)

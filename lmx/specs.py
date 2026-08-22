from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Literal

import jax.numpy as jnp

if TYPE_CHECKING:
    from .mesh import StructuredMesh

RegionKind = Literal["fluid", "solid"]
GeometryKind = Literal["rect_duct", "layered_duct", "pipe_ogrid", "bent_pipe"]
SolverKind = Literal["fully_developed_inductionless", "extruded_inductionless"]
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
PotentialSolverKind = Literal["auto", "jacobi", "cg", "cg_volume"]
CurrentReconstructionKind = Literal["cell_centered", "face_averaged", "hybrid_face_lorentz"]
VelocityUpdateLimiterKind = Literal["global_scale", "local_clip"]
PreconditionerKind = Literal["none", "jacobi"]
TimeSchemeKind = Literal["implicit_euler", "crank_nicolson"]
CouplingAccelerationKind = Literal["none", "aitken", "anderson"]
MagneticAxisKind = Literal["x", "y", "z"]


@dataclass(frozen=True)
class RegionSpec:
    """Material-region specification.

    ``viscosity`` is kinematic viscosity ``nu`` in ``m^2/s``. Dynamic
    viscosity ``mu`` should be converted before constructing a case.
    """

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
    preconditioner: PreconditionerKind = "jacobi"
    time_scheme: TimeSchemeKind = "implicit_euler"
    coupling_iterations: int = 12
    coupling_tolerance: float = 1e-8
    coupling_acceleration: CouplingAccelerationKind = "none"
    coupling_min_relaxation: float = 0.05
    coupling_max_relaxation: float = 100.0
    coupling_history_depth: int = 6
    coupling_regularization: float = 1.0e-8
    coupling_damping: float = 1.0


@dataclass(frozen=True)
class TimeStepperConfig:
    dt: float
    t_final: float
    max_steps: int
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
class FringingSpec:
    enabled: bool = False
    entry_center: float = 1.5
    exit_center: float = 4.5
    transition_width: float = 0.35
    axis: MagneticAxisKind = "z"


@dataclass(frozen=True)
class GeometrySpec:
    kind: GeometryKind
    width: float
    height: float
    length: float = 1.0
    axial_origin: float = 0.0
    nx: int = 1
    ny: int = 64
    nz: int = 64
    radius: float | None = None
    bend_radius: float | None = None
    bend_angle: float | None = None
    nr: int | None = None
    ntheta: int | None = None
    wall_thickness: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    wall_cells: tuple[int, int, int, int] = (0, 0, 0, 0)
    target_ha: float | None = None
    target_side_layer: float | None = None
    hartmann_layer_cells: int | None = None


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


EXTRUDED_HISTORY_WIDTHS = (
    ("iteration_residual_history", 0),
    ("iteration_momentum_defect_history", 0),
    ("iteration_component_residual_history", 6),
    ("iteration_pressure_residual_history", 0),
    ("iteration_pressure_linear_history", 5),
    ("iteration_electric_linear_history", 6),
    ("iteration_potential_residual_history", 0),
    ("iteration_courant_history", 3),
)


@dataclass(frozen=True)
class FringingProfile:
    """Axial field scale and optional full vector field for extruded problems."""

    x: jnp.ndarray
    field_scale: jnp.ndarray
    axis: str
    volume_field: Callable[..., jnp.ndarray] | None = None


@dataclass(frozen=True)
class ExtrudedFieldBundle:
    """Stationwise field bundle returned by the extruded inductionless solver."""

    x: jnp.ndarray
    y: jnp.ndarray
    z: jnp.ndarray
    field_scale: jnp.ndarray
    u: jnp.ndarray
    v: jnp.ndarray
    w: jnp.ndarray
    p: jnp.ndarray
    phi: jnp.ndarray
    geometry_kind: str
    solver_kind: str
    rho_phi_plus: jnp.ndarray | None = None
    rho_phi_inlet: jnp.ndarray | None = None
    aitken_state: tuple[jnp.ndarray | None, float, int] | None = None
    anderson_state: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray] | None = None
    stopping_state: tuple[int, int, str] = (0, 0, "not_recorded")
    jx: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    jy: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    jz: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    lorentz_x: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    lorentz_y: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    lorentz_z: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    residual: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    volumetric_flow_rate: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    mean_velocity: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    axial_current: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    wall_current_leakage: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    current_scaled_pressure_proxy: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    charge_balance_residual: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    boundary_current_residual: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    axial_pressure_loss_gradient: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    transverse_pressure_difference: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    iteration_residual_history: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    iteration_momentum_defect_history: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    iteration_component_residual_history: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 6)))
    iteration_pressure_residual_history: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    iteration_pressure_linear_history: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 5)))
    iteration_electric_linear_history: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 6)))
    iteration_potential_residual_history: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    iteration_courant_history: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 3)))


@dataclass(frozen=True)
class ExtrudedIterationProgress:
    """One completed outer iteration and an optional resumable state."""

    step: int
    total_steps: int
    residual: float
    component_residuals: tuple[float, ...]
    pressure_residual: float
    potential_residual: float
    checkpoint: ExtrudedFieldBundle | None = None


@dataclass(frozen=True)
class ExtrudedInductionlessProblem:
    """Case specification plus imposed fringing-field profile."""

    case: CaseSpec
    profile: FringingProfile


@dataclass(frozen=True)
class ExtrudedInductionlessValidation:
    """Compact validation metrics for an extruded inductionless solution."""

    station_count: int
    max_residual: float
    max_charge_balance_residual: float
    mean_velocity_span: float
    volumetric_flow_rate_span: float
    axial_current_span: float
    max_wall_current_leakage: float
    net_boundary_current_residual: float
    field_mean_velocity_correlation: float
    axial_current_mirror_residual: float = 0.0
    peak_velocity_span: float = 0.0
    pressure_span_range: float = 0.0
    pressure_span_mirror_residual: float = 0.0
    center_axial_current: float = 0.0
    center_pressure_span: float = 0.0
    max_divergence_residual: float = 0.0


@dataclass(frozen=True)
class ExtrudedInductionlessSolution:
    """Solved fringing problem with fields, station history, and validation."""

    problem: ExtrudedInductionlessProblem
    bundle: ExtrudedFieldBundle
    station_history: tuple[dict[str, float], ...]
    validation: ExtrudedInductionlessValidation

    @property
    def steps(self) -> int:
        """Number of completed outer iterations."""

        return int(self.bundle.stopping_state[0])

    @property
    def status(self) -> str:
        """Terminal solver status."""

        return str(self.bundle.stopping_state[2])

    @property
    def converged(self) -> bool:
        """Whether the configured steady gates passed."""

        return self.status == "converged"


class NumericalFailure(RuntimeError):
    """Raised when a solver produces nonfinite numerical state."""


def require_finite(stage: str, **values) -> None:
    """Raise with field names when numerical output is nonfinite."""

    failed = [name for name, value in values.items() if not bool(jnp.all(jnp.isfinite(jnp.asarray(value))))]
    if failed:
        raise NumericalFailure(f"{stage} produced nonfinite {', '.join(failed)}")


@dataclass(frozen=True)
class MHDState:
    u: jnp.ndarray
    phi: jnp.ndarray
    jy: jnp.ndarray
    jz: jnp.ndarray
    lorentz_x: jnp.ndarray
    time: float
    residual: float


@dataclass(frozen=True)
class Diagnostics:
    residual_history: jnp.ndarray
    courant_like: jnp.ndarray
    ohmic_power: jnp.ndarray
    time_history: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    u_max_history: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    mean_velocity_history: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    applied_forcing_history: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    pressure_proxy_history: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    current_scaled_pressure_proxy_history: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    raw_update_max_history: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    limiter_scale_history: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    limited_fraction_history: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    current_max_history: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    face_current_max_history: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    emf_max_history: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    lorentz_max_history: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    face_lorentz_max_history: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    potential_residual_history: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    potential_iterations_history: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    linear_residual_history: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    linear_iterations_history: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    volumetric_flow_rate_history: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    mean_current_magnitude_history: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    lorentz_power_history: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    div_current_max_history: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    charge_balance_residual_history: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    gauge_residual_history: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    interface_current_residual_history: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))


@dataclass(frozen=True)
class Solution:
    mesh: StructuredMesh
    state: MHDState
    diagnostics: Diagnostics
    case_name: str
    converged: bool | None = None
    status: str = "not_recorded"
    steps: int = 0


def zeros_state(mesh: StructuredMesh) -> MHDState:
    zeros = jnp.zeros(mesh.yz_shape)
    return MHDState(
        u=zeros,
        phi=zeros,
        jy=zeros,
        jz=zeros,
        lorentz_x=zeros,
        time=0.0,
        residual=0.0,
    )

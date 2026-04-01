from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

import jax.numpy as jnp


RegionKind = Literal["fluid", "solid"]
GeometryKind = Literal["rect_duct", "layered_duct", "pipe_ogrid"]
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


@dataclass(frozen=True)
class TimeStepperConfig:
    dt: float
    t_final: float
    max_steps: int
    outer_iterations: int = 2
    potential_iterations: int = 400
    potential_tolerance: float | None = None
    steady_tolerance: float = 1e-8
    relaxation: float = 0.35
    velocity_update_limit: float = 1e-3
    checkpoint_stride: int = 1


@dataclass(frozen=True)
class OutputSpec:
    directory: str | None = None
    write_paraview: bool = True
    write_csv_profiles: bool = True
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

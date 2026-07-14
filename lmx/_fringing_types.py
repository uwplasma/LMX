"""Data containers for the private 3D inductionless fringing implementation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import jax.numpy as jnp

from .specs import CaseSpec


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
    iteration_component_residual_history: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 6)))
    iteration_pressure_residual_history: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
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

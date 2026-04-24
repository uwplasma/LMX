"""Data containers for the private 3D inductionless fringing implementation."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from .specs import CaseSpec


@dataclass(frozen=True)
class FringingProfile:
    """Axial magnetic-field profile used by extruded fringing problems."""

    x: jnp.ndarray
    field_scale: jnp.ndarray
    axis: str


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


@dataclass(frozen=True)
class ExtrudedInductionlessSolution:
    """Solved fringing problem with fields, station history, and validation."""

    problem: ExtrudedInductionlessProblem
    bundle: ExtrudedFieldBundle
    station_history: tuple[dict[str, float], ...]
    validation: ExtrudedInductionlessValidation

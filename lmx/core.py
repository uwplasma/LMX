from __future__ import annotations

from dataclasses import dataclass, field

import jax.numpy as jnp

from .mesh import StructuredMesh


class NumericalFailure(RuntimeError):
    """Raised when a solver produces nonfinite numerical state."""


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
    shape = mesh.yz_shape
    zeros = jnp.zeros(shape)
    return MHDState(
        u=zeros,
        phi=zeros,
        jy=zeros,
        jz=zeros,
        lorentz_x=zeros,
        time=0.0,
        residual=0.0,
    )

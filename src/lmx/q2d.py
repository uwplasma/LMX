"""Quasi-two-dimensional inductionless MHD on periodic planes."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from math import isfinite

import jax
import jax.numpy as jnp
from solvax import (
    checkpointed_fori_loop,
    periodic_poisson_eigenvalues,
    solve_periodic_poisson_spectral,
)

from .specs import require_finite

__all__ = ["Q2DDiagnostics", "Q2DProblem", "Q2DResult", "evolve_q2d", "make_q2d_case", "solve_q2d"]


def _state_arrays(initial_vorticity, forcing, *coefficients):
    """Choose one real working dtype before constructing a loop carry."""
    vorticity = jnp.asarray(initial_vorticity)
    forcing = jnp.zeros_like(vorticity) if forcing is None else jnp.asarray(forcing)
    dtype = jnp.result_type(vorticity, forcing, *coefficients, jnp.float32)
    if not jnp.issubdtype(dtype, jnp.floating):
        raise ValueError("Q2D state, forcing, and coefficients must be real")
    return vorticity.astype(dtype), forcing.astype(dtype)


@dataclass(frozen=True)
class Q2DProblem:
    """Periodic Sommeria--Moreau vorticity problem.

    ``forcing`` is a vorticity source. The linear Hartmann-layer closure is
    ``-hartmann_friction * vorticity``; lengths, viscosity, time, and forcing
    must use one consistent unit system. All inputs determine a common real
    working dtype through JAX promotion, with at least float32 precision.
    ``energy_budget_tolerance`` bounds the normalized trapezoidal energy
    defect for host acceptance; it does not change the numerical trajectory.
    """

    initial_vorticity: jax.Array
    forcing: jax.Array | None = None
    length: tuple[float, float] = (2.0 * jnp.pi, 2.0 * jnp.pi)
    viscosity: float = 1.0e-2
    hartmann_friction: float = 0.1
    dt: float = 1.0e-2
    steps: int = 100
    history_stride: int = 0
    adjoint_checkpoint_size: int | None = None
    energy_budget_tolerance: float = 1.0e-3

    def __post_init__(self) -> None:
        vorticity, forcing = _state_arrays(
            self.initial_vorticity,
            self.forcing,
            *self.length,
            self.viscosity,
            self.hartmann_friction,
            self.dt,
        )
        if vorticity.ndim != 2 or min(vorticity.shape) < 4:
            raise ValueError("initial_vorticity must be a 2-D array with at least four points per axis")
        if forcing.shape != vorticity.shape:
            raise ValueError("forcing must match initial_vorticity")
        if any(value <= 0.0 for value in self.length):
            raise ValueError("length values must be positive")
        if self.viscosity < 0.0 or self.hartmann_friction < 0.0:
            raise ValueError("viscosity and hartmann_friction must be non-negative")
        if self.dt <= 0.0 or self.steps < 1 or self.history_stride < 0:
            raise ValueError("dt and steps must be positive and history_stride non-negative")
        if self.adjoint_checkpoint_size is not None and self.adjoint_checkpoint_size < 1:
            raise ValueError("adjoint_checkpoint_size must be positive")
        if not isfinite(self.energy_budget_tolerance) or self.energy_budget_tolerance <= 0:
            raise ValueError("energy_budget_tolerance must be finite and positive")
        object.__setattr__(self, "initial_vorticity", vorticity)
        object.__setattr__(self, "forcing", forcing)


@dataclass(frozen=True)
class Q2DDiagnostics:
    """Compact physical and numerical gates for a Q2D solve."""

    kinetic_energy_initial: float
    kinetic_energy_final: float
    enstrophy_final: float
    energy_budget_residual: float
    max_divergence: float
    max_courant: float


@dataclass(frozen=True)
class Q2DResult:
    """Final Q2D fields, optional frames, and acceptance diagnostics."""

    problem: Q2DProblem
    x: jax.Array
    y: jax.Array
    vorticity: jax.Array
    velocity_x: jax.Array
    velocity_y: jax.Array
    frame_times: jax.Array
    vorticity_history: jax.Array
    diagnostics: Q2DDiagnostics
    status: str

    @property
    def fields(self) -> Q2DResult:
        """Return the field-bearing result, matching the common solve contract."""

        return self

    @property
    def steps(self) -> int:
        return self.problem.steps

    @property
    def converged(self) -> bool:
        """Whether the finite trajectory passed acceptance, not steady convergence."""
        return self.status == "completed"

    @property
    def residual(self) -> float:
        return max(self.diagnostics.energy_budget_residual, self.diagnostics.max_divergence)


def make_q2d_case(
    *,
    shape: tuple[int, int] = (64, 64),
    length: tuple[float, float] = (2.0 * jnp.pi, 2.0 * jnp.pi),
    mode: tuple[int, int] = (1, 1),
    amplitude: float = 1.0,
    viscosity: float = 1.0e-2,
    hartmann_friction: float = 0.1,
    dt: float = 1.0e-2,
    steps: int = 100,
    history_stride: int = 0,
    energy_budget_tolerance: float = 1.0e-3,
) -> Q2DProblem:
    """Build a Taylor--Green decay case with an analytical exponential rate."""

    if len(shape) != 2 or len(length) != 2 or len(mode) != 2 or any(value < 1 for value in mode):
        raise ValueError("shape, length, and positive mode must each describe two axes")
    x = jnp.arange(shape[0]) * length[0] / shape[0]
    y = jnp.arange(shape[1]) * length[1] / shape[1]
    kx, ky = 2.0 * jnp.pi * mode[0] / length[0], 2.0 * jnp.pi * mode[1] / length[1]
    vorticity = amplitude * jnp.sin(kx * x[:, None]) * jnp.sin(ky * y[None, :])
    return Q2DProblem(
        vorticity,
        length=length,
        viscosity=viscosity,
        hartmann_friction=hartmann_friction,
        dt=dt,
        steps=steps,
        history_stride=history_stride,
        energy_budget_tolerance=energy_budget_tolerance,
    )


def _flow(omega_hat, eigenvalues, kx, ky):
    psi_hat = solve_periodic_poisson_spectral(omega_hat, eigenvalues=eigenvalues)
    ux = jnp.fft.ifftn(1j * ky * psi_hat).real
    uy = jnp.fft.ifftn(-1j * kx * psi_hat).real
    return psi_hat, ux, uy


def _nonlinear(omega_hat, forcing_hat, eigenvalues, kx, ky, dealias):
    _, ux, uy = _flow(omega_hat, eigenvalues, kx, ky)
    omega_x = jnp.fft.ifftn(1j * kx * omega_hat).real
    omega_y = jnp.fft.ifftn(1j * ky * omega_hat).real
    return (forcing_hat - jnp.fft.fftn(ux * omega_x + uy * omega_y)) * dealias


def _step(omega_hat, forcing_hat, eigenvalues, kx, ky, dealias, dt, decay, half_decay):
    first = _nonlinear(omega_hat, forcing_hat, eigenvalues, kx, ky, dealias)
    a = half_decay * (omega_hat + 0.5 * dt * first)
    second = _nonlinear(a, forcing_hat, eigenvalues, kx, ky, dealias)
    b = half_decay * omega_hat + 0.5 * dt * second
    third = _nonlinear(b, forcing_hat, eigenvalues, kx, ky, dealias)
    c = decay * omega_hat + dt * half_decay * third
    fourth = _nonlinear(c, forcing_hat, eigenvalues, kx, ky, dealias)
    updated = decay * omega_hat + (dt / 6.0) * (decay * first + 2.0 * half_decay * (second + third) + fourth)
    return (updated * dealias).at[0, 0].set(0.0)


def _measures(omega_hat, forcing, eigenvalues, kx, ky, dt, spacing, viscosity, friction):
    psi_hat, ux, uy = _flow(omega_hat, eigenvalues, kx, ky)
    omega = jnp.fft.ifftn(omega_hat).real
    psi = jnp.fft.ifftn(psi_hat).real
    energy = 0.5 * jnp.mean(ux**2 + uy**2)
    enstrophy = 0.5 * jnp.mean(omega**2)
    divergence = jnp.fft.ifftn(1j * kx * jnp.fft.fftn(ux) + 1j * ky * jnp.fft.fftn(uy)).real
    courant = dt * jnp.max(jnp.abs(ux) / spacing[0] + jnp.abs(uy) / spacing[1])
    energy_rate = -2.0 * viscosity * enstrophy - 2.0 * friction * energy + jnp.mean(psi * forcing)
    return energy, enstrophy, jnp.max(jnp.abs(divergence)), courant, energy_rate


@partial(jax.jit, static_argnames=("steps", "checkpoint_size"))
def _integrate(
    omega_hat,
    forcing,
    forcing_hat,
    eigenvalues,
    kx,
    ky,
    dealias,
    decay,
    half_decay,
    spacing,
    viscosity,
    friction,
    dt,
    measures,
    budget,
    max_courant,
    *,
    steps,
    checkpoint_size,
):
    def advance(_index, carry):
        omega, before, budget, max_courant = carry
        updated = _step(omega, forcing_hat, eigenvalues, kx, ky, dealias, dt, decay, half_decay)
        after = _measures(updated, forcing, eigenvalues, kx, ky, dt, spacing, viscosity, friction)
        budget += 0.5 * dt * (before[-1] + after[-1])
        return updated, after, budget, jnp.maximum(max_courant, after[-2])

    initial = (omega_hat, measures, budget, max_courant)
    return checkpointed_fori_loop(0, steps, advance, initial, checkpoint_size=checkpoint_size)


def _setup(initial_vorticity, forcing, length, viscosity, friction, dt):
    shape, dtype = initial_vorticity.shape, initial_vorticity.dtype
    spacing = (length[0] / shape[0], length[1] / shape[1])
    eigenvalues = periodic_poisson_eigenvalues(shape, spacing).astype(dtype)
    kx = (2.0 * jnp.pi * jnp.fft.fftfreq(shape[0], d=spacing[0])).astype(dtype)[:, None]
    ky = (2.0 * jnp.pi * jnp.fft.fftfreq(shape[1], d=spacing[1])).astype(dtype)[None, :]
    ix, iy = jnp.fft.fftfreq(shape[0]) * shape[0], jnp.fft.fftfreq(shape[1]) * shape[1]
    dealias = (jnp.abs(ix[:, None]) <= shape[0] / 3.0) & (jnp.abs(iy[None, :]) <= shape[1] / 3.0)
    omega_hat = (jnp.fft.fftn(initial_vorticity) * dealias).at[0, 0].set(0.0)
    forcing_hat = (jnp.fft.fftn(forcing) * dealias).at[0, 0].set(0.0)
    decay = jnp.exp(-dt * (viscosity * eigenvalues + friction))
    return omega_hat, forcing_hat, eigenvalues, kx, ky, dealias, decay, spacing


def _evolve(initial_vorticity, forcing, length, viscosity, friction, dt, steps, checkpoint_size):
    omega_hat, forcing_hat, eigenvalues, kx, ky, dealias, decay, _ = _setup(
        initial_vorticity, forcing, length, viscosity, friction, dt
    )

    def advance(_index, current):
        return _step(current, forcing_hat, eigenvalues, kx, ky, dealias, dt, decay, jnp.sqrt(decay))

    omega_hat = checkpointed_fori_loop(0, steps, advance, omega_hat, checkpoint_size=checkpoint_size)
    _, ux, uy = _flow(omega_hat, eigenvalues, kx, ky)
    return jnp.fft.ifftn(omega_hat).real, ux, uy


def evolve_q2d(
    initial_vorticity: jax.Array,
    *,
    forcing: jax.Array | None = None,
    length: tuple[float, float] = (2.0 * jnp.pi, 2.0 * jnp.pi),
    viscosity: float | jax.Array = 1.0e-2,
    hartmann_friction: float | jax.Array = 0.1,
    dt: float | jax.Array = 1.0e-2,
    steps: int = 100,
    adjoint_checkpoint_size: int | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return final Q2D fields through a JIT- and autodiff-safe numerical core.

    Array state, forcing, domain lengths, viscosity, Hartmann friction, and
    timestep are differentiable. ``steps`` and ``adjoint_checkpoint_size`` are
    static controls; the default checkpoint schedule retains
    ``O(sqrt(steps))`` trajectory states in reverse mode. State, forcing and
    coefficients use JAX's common real dtype, with at least float32 precision;
    float64 requires enabling JAX x64 before constructing inputs.
    """
    initial_vorticity, forcing = _state_arrays(
        initial_vorticity, forcing, *length, viscosity, hartmann_friction, dt
    )
    return _evolve(
        initial_vorticity,
        forcing,
        length,
        viscosity,
        hartmann_friction,
        dt,
        steps,
        adjoint_checkpoint_size,
    )


def solve_q2d(problem: Q2DProblem) -> Q2DResult:
    """Evolve IFRK4 fields and enforce Courant and normalized energy-budget gates."""

    shape, dtype = problem.initial_vorticity.shape, problem.initial_vorticity.dtype
    omega_hat, forcing_hat, eigenvalues, kx, ky, dealias, decay, spacing = _setup(
        problem.initial_vorticity,
        problem.forcing,
        problem.length,
        problem.viscosity,
        problem.hartmann_friction,
        problem.dt,
    )
    initial = _measures(
        omega_hat,
        problem.forcing,
        eigenvalues,
        kx,
        ky,
        problem.dt,
        spacing,
        problem.viscosity,
        problem.hartmann_friction,
    )
    stride = problem.history_stride or problem.steps
    frames, frame_steps = [], []
    budget, max_courant, completed = jnp.asarray(0.0, dtype=dtype), initial[-2], 0
    final = initial
    if problem.history_stride:
        frames.append(jnp.fft.ifftn(omega_hat).real)
        frame_steps.append(0)
    while completed < problem.steps:
        segment = min(stride, problem.steps - completed)
        omega_hat, final, budget, max_courant = _integrate(
            omega_hat,
            problem.forcing,
            forcing_hat,
            eigenvalues,
            kx,
            ky,
            dealias,
            decay,
            jnp.sqrt(decay),
            spacing,
            problem.viscosity,
            problem.hartmann_friction,
            problem.dt,
            final,
            budget,
            max_courant,
            steps=segment,
            checkpoint_size=problem.adjoint_checkpoint_size,
        )
        completed += segment
        if problem.history_stride:
            frames.append(jnp.fft.ifftn(omega_hat).real)
            frame_steps.append(completed)
    psi_hat, ux, uy = _flow(omega_hat, eigenvalues, kx, ky)
    vorticity = jnp.fft.ifftn(omega_hat).real
    budget_residual = jnp.abs(final[0] - initial[0] - budget) / jnp.maximum(
        jnp.maximum(initial[0], jnp.abs(budget)), jnp.finfo(vorticity.dtype).tiny
    )
    require_finite(
        "Q2D solve",
        vorticity=vorticity,
        ux=ux,
        uy=uy,
        psi=psi_hat,
        diagnostics=jnp.asarray((*initial, *final, budget_residual, max_courant)),
    )
    diagnostics = Q2DDiagnostics(
        *(float(value) for value in (initial[0], final[0], final[1], budget_residual, final[2], max_courant))
    )
    status = (
        "courant_limit_exceeded"
        if diagnostics.max_courant > 1.0
        else "energy_budget_exceeded"
        if diagnostics.energy_budget_residual > problem.energy_budget_tolerance
        else "completed"
    )
    return Q2DResult(
        problem=problem,
        x=jnp.arange(shape[0]) * spacing[0],
        y=jnp.arange(shape[1]) * spacing[1],
        vorticity=vorticity,
        velocity_x=ux,
        velocity_y=uy,
        frame_times=jnp.asarray(frame_steps) * problem.dt,
        vorticity_history=(jnp.stack(frames) if frames else jnp.zeros((0, *shape), dtype=vorticity.dtype)),
        diagnostics=diagnostics,
        status=status,
    )

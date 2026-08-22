"""Quasi-two-dimensional inductionless MHD on periodic planes."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from solvax import periodic_poisson_eigenvalues, solve_periodic_poisson_spectral

from .specs import require_finite

__all__ = ["Q2DDiagnostics", "Q2DProblem", "Q2DResult", "make_q2d_case", "solve_q2d"]


@dataclass(frozen=True)
class Q2DProblem:
    """Periodic Sommeria--Moreau vorticity problem.

    ``forcing`` is a vorticity source. The linear Hartmann-layer closure is
    ``-hartmann_friction * vorticity``; lengths, viscosity, time, and forcing
    must use one consistent unit system.
    """

    initial_vorticity: jax.Array
    forcing: jax.Array | None = None
    length: tuple[float, float] = (2.0 * jnp.pi, 2.0 * jnp.pi)
    viscosity: float = 1.0e-2
    hartmann_friction: float = 0.1
    dt: float = 1.0e-2
    steps: int = 100
    history_stride: int = 0

    def __post_init__(self) -> None:
        vorticity = jnp.asarray(self.initial_vorticity)
        vorticity = vorticity.astype(jnp.result_type(vorticity, jnp.float32))
        forcing = (
            jnp.zeros_like(vorticity)
            if self.forcing is None
            else jnp.asarray(self.forcing, dtype=vorticity.dtype)
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


@partial(jax.jit, static_argnames="steps")
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
):
    def advance(carry, _):
        omega, before, budget, max_courant = carry
        updated = _step(omega, forcing_hat, eigenvalues, kx, ky, dealias, dt, decay, half_decay)
        after = _measures(updated, forcing, eigenvalues, kx, ky, dt, spacing, viscosity, friction)
        budget += 0.5 * dt * (before[-1] + after[-1])
        return (updated, after, budget, jnp.maximum(max_courant, after[-2])), None

    initial = (omega_hat, measures, budget, max_courant)
    return jax.lax.scan(advance, initial, None, length=steps)[0]


def solve_q2d(problem: Q2DProblem) -> Q2DResult:
    """Solve a periodic Q2D problem with dealiased Fourier IFRK4 evolution."""

    shape = problem.initial_vorticity.shape
    dtype = problem.initial_vorticity.dtype
    spacing = (problem.length[0] / shape[0], problem.length[1] / shape[1])
    eigenvalues = periodic_poisson_eigenvalues(shape, spacing).astype(dtype)
    kx = (2.0 * jnp.pi * jnp.fft.fftfreq(shape[0], d=spacing[0])).astype(dtype)[:, None]
    ky = (2.0 * jnp.pi * jnp.fft.fftfreq(shape[1], d=spacing[1])).astype(dtype)[None, :]
    ix, iy = jnp.fft.fftfreq(shape[0]) * shape[0], jnp.fft.fftfreq(shape[1]) * shape[1]
    dealias = (jnp.abs(ix[:, None]) <= shape[0] / 3.0) & (jnp.abs(iy[None, :]) <= shape[1] / 3.0)
    omega_hat = (
        (jnp.fft.fftn(problem.initial_vorticity.astype(eigenvalues.dtype)) * dealias).at[0, 0].set(0.0)
    )
    forcing_hat = (jnp.fft.fftn(problem.forcing.astype(eigenvalues.dtype)) * dealias).at[0, 0].set(0.0)
    decay = jnp.exp(-problem.dt * (problem.viscosity * eigenvalues + problem.hartmann_friction))
    half_decay = jnp.sqrt(decay)
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
            half_decay,
            spacing,
            problem.viscosity,
            problem.hartmann_friction,
            problem.dt,
            final,
            budget,
            max_courant,
            steps=segment,
        )
        completed += segment
        if problem.history_stride:
            frames.append(jnp.fft.ifftn(omega_hat).real)
            frame_steps.append(completed)
    psi_hat, ux, uy = _flow(omega_hat, eigenvalues, kx, ky)
    vorticity = jnp.fft.ifftn(omega_hat).real
    budget_residual = jnp.abs(final[0] - initial[0] - budget) / jnp.maximum(
        jnp.maximum(initial[0], jnp.abs(budget)), jnp.finfo(vorticity.dtype).eps
    )
    require_finite("Q2D solve", vorticity=vorticity, ux=ux, uy=uy, psi=psi_hat)
    diagnostics = Q2DDiagnostics(
        *(float(value) for value in (initial[0], final[0], final[1], budget_residual, final[2], max_courant))
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
        status="completed" if diagnostics.max_courant <= 1.0 else "courant_limit_exceeded",
    )

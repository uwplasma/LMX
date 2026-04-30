from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class Q2DDecayCase:
    nx: int = 96
    ny: int = 96
    lx: float = 2.0
    ly: float = 2.0
    viscosity: float = 0.01
    hartmann_friction: float = 2.0
    mode_x: int = 1
    mode_y: int = 1
    amplitude: float = 1.0
    dt: float = 5.0e-4
    t_final: float = 0.08


@dataclass(frozen=True)
class Q2DDecaySolution:
    x: np.ndarray
    y: np.ndarray
    initial: np.ndarray
    final: np.ndarray
    analytic_final: np.ndarray
    time: np.ndarray
    amplitude_numeric: np.ndarray
    amplitude_analytic: np.ndarray
    decay_rate: float


@dataclass(frozen=True)
class Q2DForcedCase:
    nx: int = 96
    ny: int = 96
    lx: float = 2.0
    ly: float = 2.0
    viscosity: float = 0.01
    hartmann_friction: float = 2.0
    mode_x: int = 1
    mode_y: int = 1
    forcing_amplitude: float = 1.0
    dt: float = 5.0e-4
    t_final: float = 0.2


@dataclass(frozen=True)
class Q2DForcedSolution:
    x: np.ndarray
    y: np.ndarray
    steady_numeric: np.ndarray
    steady_analytic: np.ndarray
    time: np.ndarray
    amplitude_numeric: np.ndarray
    amplitude_analytic: np.ndarray
    decay_rate: float
    steady_amplitude: float


@dataclass(frozen=True)
class Q2DWallBoundedForcedCase:
    nx: int = 96
    ny: int = 96
    lx: float = 2.0
    ly: float = 2.0
    viscosity: float = 0.01
    hartmann_friction: float = 2.0
    mode_x: int = 1
    mode_y: int = 1
    forcing_amplitude: float = 1.0
    dt: float = 5.0e-4
    t_final: float = 0.2


@dataclass(frozen=True)
class Q2DWallBoundedForcedSolution:
    x: np.ndarray
    y: np.ndarray
    field: np.ndarray
    analytic_final: np.ndarray
    time: np.ndarray
    amplitude_numeric: np.ndarray
    amplitude_analytic: np.ndarray
    decay_rate: float
    steady_amplitude: float


@dataclass(frozen=True)
class Q2DWallDrivenCavityCase:
    nx: int = 201
    ny: int = 101
    lx: float = 0.04
    ly: float = 0.04
    viscosity: float = 2.27e-7
    hartmann_friction: float = 1.7025e-2
    right_wall_velocity: float = 0.1
    dt: float = 5.0e-4
    t_final: float = 1.0
    frame_count: int = 48


@dataclass(frozen=True)
class Q2DWallDrivenCavitySolution:
    x: np.ndarray
    y: np.ndarray
    time: np.ndarray
    streamfunction_frames: np.ndarray
    vorticity_frames: np.ndarray
    ux_frames: np.ndarray
    uy_frames: np.ndarray
    kinetic_energy: np.ndarray
    enstrophy: np.ndarray
    max_courant: np.ndarray
    divergence_linf: np.ndarray


@dataclass(frozen=True)
class Q2DTurbulenceDecayCase:
    nx: int = 96
    ny: int = 96
    lx: float = 2.0
    ly: float = 2.0
    viscosity: float = 8.0e-4
    hartmann_friction: float = 0.08
    amplitude: float = 6.0
    forcing_amplitude: float = 0.08
    forcing_wavenumber: int = 4
    dt: float = 2.0e-3
    t_final: float = 3.0
    frame_count: int = 72


@dataclass(frozen=True)
class Q2DTurbulenceDecaySolution:
    x: np.ndarray
    y: np.ndarray
    time: np.ndarray
    frames: np.ndarray
    kinetic_energy: np.ndarray
    enstrophy_proxy: np.ndarray
    velocity_rms: np.ndarray
    max_courant: np.ndarray
    divergence_linf: np.ndarray
    turnover_count: float
    initial_spectrum: dict[str, list[float]]
    final_spectrum: dict[str, list[float]]


def build_q2d_decay_case(
    *,
    nx: int = 96,
    ny: int = 96,
    lx: float = 2.0,
    ly: float = 2.0,
    viscosity: float = 0.01,
    hartmann_friction: float = 2.0,
    mode_x: int = 1,
    mode_y: int = 1,
    amplitude: float = 1.0,
    dt: float = 5.0e-4,
    t_final: float = 0.08,
) -> Q2DDecayCase:
    return Q2DDecayCase(
        nx=nx,
        ny=ny,
        lx=lx,
        ly=ly,
        viscosity=viscosity,
        hartmann_friction=hartmann_friction,
        mode_x=mode_x,
        mode_y=mode_y,
        amplitude=amplitude,
        dt=dt,
        t_final=t_final,
    )


def build_q2d_turbulence_decay_case(
    *,
    nx: int = 96,
    ny: int = 96,
    lx: float = 2.0,
    ly: float = 2.0,
    viscosity: float = 8.0e-4,
    hartmann_friction: float = 0.08,
    amplitude: float = 6.0,
    forcing_amplitude: float = 0.08,
    forcing_wavenumber: int = 4,
    dt: float = 2.0e-3,
    t_final: float = 3.0,
    frame_count: int = 72,
) -> Q2DTurbulenceDecayCase:
    return Q2DTurbulenceDecayCase(
        nx=nx,
        ny=ny,
        lx=lx,
        ly=ly,
        viscosity=viscosity,
        hartmann_friction=hartmann_friction,
        amplitude=amplitude,
        forcing_amplitude=forcing_amplitude,
        forcing_wavenumber=forcing_wavenumber,
        dt=dt,
        t_final=t_final,
        frame_count=frame_count,
    )


def build_q2d_forced_case(
    *,
    nx: int = 96,
    ny: int = 96,
    lx: float = 2.0,
    ly: float = 2.0,
    viscosity: float = 0.01,
    hartmann_friction: float = 2.0,
    mode_x: int = 1,
    mode_y: int = 1,
    forcing_amplitude: float = 1.0,
    dt: float = 5.0e-4,
    t_final: float = 0.2,
) -> Q2DForcedCase:
    return Q2DForcedCase(
        nx=nx,
        ny=ny,
        lx=lx,
        ly=ly,
        viscosity=viscosity,
        hartmann_friction=hartmann_friction,
        mode_x=mode_x,
        mode_y=mode_y,
        forcing_amplitude=forcing_amplitude,
        dt=dt,
        t_final=t_final,
    )


def build_q2d_wall_bounded_forced_case(
    *,
    nx: int = 96,
    ny: int = 96,
    lx: float = 2.0,
    ly: float = 2.0,
    viscosity: float = 0.01,
    hartmann_friction: float = 2.0,
    mode_x: int = 1,
    mode_y: int = 1,
    forcing_amplitude: float = 1.0,
    dt: float = 5.0e-4,
    t_final: float = 0.2,
) -> Q2DWallBoundedForcedCase:
    return Q2DWallBoundedForcedCase(
        nx=nx,
        ny=ny,
        lx=lx,
        ly=ly,
        viscosity=viscosity,
        hartmann_friction=hartmann_friction,
        mode_x=mode_x,
        mode_y=mode_y,
        forcing_amplitude=forcing_amplitude,
        dt=dt,
        t_final=t_final,
    )


def build_q2d_wall_driven_cavity_case(
    *,
    nx: int = 201,
    ny: int = 101,
    lx: float = 0.04,
    ly: float = 0.04,
    viscosity: float = 2.27e-7,
    hartmann_friction: float = 1.7025e-2,
    right_wall_velocity: float = 0.1,
    dt: float = 5.0e-4,
    t_final: float = 1.0,
    frame_count: int = 48,
) -> Q2DWallDrivenCavityCase:
    return Q2DWallDrivenCavityCase(
        nx=nx,
        ny=ny,
        lx=lx,
        ly=ly,
        viscosity=viscosity,
        hartmann_friction=hartmann_friction,
        right_wall_velocity=right_wall_velocity,
        dt=dt,
        t_final=t_final,
        frame_count=frame_count,
    )


def _periodic_laplacian(field: np.ndarray, *, dx: float, dy: float) -> np.ndarray:
    return (
        (np.roll(field, -1, axis=0) - 2.0 * field + np.roll(field, 1, axis=0)) / dx**2
        + (np.roll(field, -1, axis=1) - 2.0 * field + np.roll(field, 1, axis=1)) / dy**2
    )


def _dirichlet_laplacian(field: np.ndarray, *, dx: float, dy: float) -> np.ndarray:
    padded = np.pad(field, ((1, 1), (1, 1)), mode="constant", constant_values=0.0)
    return (
        (padded[2:, 1:-1] - 2.0 * padded[1:-1, 1:-1] + padded[:-2, 1:-1]) / dx**2
        + (padded[1:-1, 2:] - 2.0 * padded[1:-1, 1:-1] + padded[1:-1, :-2]) / dy**2
    )


def _mode_shape(case: Q2DDecayCase, xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
    return case.amplitude * np.sin(2.0 * np.pi * case.mode_x * xx / case.lx) * np.sin(2.0 * np.pi * case.mode_y * yy / case.ly)


def _wall_mode_shape(*, amplitude: float, lx: float, ly: float, mode_x: int, mode_y: int, xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
    return amplitude * np.sin(mode_x * np.pi * xx / lx) * np.sin(mode_y * np.pi * yy / ly)


def _q2d_multimode_initial_condition(case: Q2DTurbulenceDecayCase, xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
    modes = (
        (1, 2, 1.00, 0.15),
        (2, 1, -0.75, 1.20),
        (3, 2, 0.45, 2.10),
        (2, 4, -0.32, 0.70),
        (5, 3, 0.20, 2.70),
        (4, 5, -0.14, 1.80),
        (8, 5, 0.10, 0.40),
        (10, 7, -0.07, 2.40),
    )
    field = np.zeros_like(xx, dtype=float)
    for mode_x, mode_y, weight, phase in modes:
        field += weight * np.sin(2.0 * np.pi * mode_x * xx / case.lx + phase) * np.cos(
            2.0 * np.pi * mode_y * yy / case.ly - 0.5 * phase
        )
    field -= float(np.mean(field))
    field /= max(float(np.max(np.abs(field))), 1.0e-12)
    return case.amplitude * field


def _q2d_vorticity_forcing(case: Q2DTurbulenceDecayCase, xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
    kf = max(1, int(case.forcing_wavenumber))
    forcing = (
        np.sin(2.0 * np.pi * kf * xx / case.lx)
        + 0.7 * np.cos(2.0 * np.pi * kf * yy / case.ly + 0.35)
        + 0.45 * np.sin(2.0 * np.pi * (kf + 1) * (xx + yy) / max(case.lx + case.ly, 1.0e-12))
    )
    forcing -= float(np.mean(forcing))
    forcing /= max(float(np.max(np.abs(forcing))), 1.0e-12)
    return float(case.forcing_amplitude) * forcing


def _q2d_dirichlet_poisson(rhs: np.ndarray, *, dx: float, dy: float) -> np.ndarray:
    from scipy.fft import dstn, idstn

    values = np.asarray(rhs, dtype=float)
    nx, ny = values.shape
    if nx < 1 or ny < 1:
        return np.zeros_like(values)
    i = np.arange(1, nx + 1, dtype=float)[:, None]
    j = np.arange(1, ny + 1, dtype=float)[None, :]
    eigenvalues = (
        -4.0 * np.sin(np.pi * i / (2.0 * (nx + 1))) ** 2 / dx**2
        - 4.0 * np.sin(np.pi * j / (2.0 * (ny + 1))) ** 2 / dy**2
    )
    return idstn(dstn(values, type=1, norm="ortho") / eigenvalues, type=1, norm="ortho")


def _q2d_wall_driven_apply_vorticity_boundary(
    omega: np.ndarray,
    psi: np.ndarray,
    *,
    dx: float,
    dy: float,
    right_wall_velocity: float,
) -> None:
    omega[:, 0] = -2.0 * psi[:, 1] / dy**2
    omega[:, -1] = -2.0 * psi[:, -2] / dy**2
    omega[0, :] = -2.0 * psi[1, :] / dx**2
    omega[-1, :] = -2.0 * psi[-2, :] / dx**2 + 2.0 * right_wall_velocity / dx
    omega[0, 0] = 0.5 * (omega[1, 0] + omega[0, 1])
    omega[0, -1] = 0.5 * (omega[1, -1] + omega[0, -2])
    omega[-1, 0] = 0.5 * (omega[-2, 0] + omega[-1, 1])
    omega[-1, -1] = 0.5 * (omega[-2, -1] + omega[-1, -2])


def _q2d_wall_driven_streamfunction(omega: np.ndarray, *, dx: float, dy: float) -> np.ndarray:
    psi = np.zeros_like(omega, dtype=float)
    psi[1:-1, 1:-1] = _q2d_dirichlet_poisson(-omega[1:-1, 1:-1], dx=dx, dy=dy)
    return psi


def _q2d_wall_driven_velocity(
    psi: np.ndarray,
    *,
    dx: float,
    dy: float,
    right_wall_velocity: float,
) -> tuple[np.ndarray, np.ndarray]:
    ux = np.zeros_like(psi, dtype=float)
    uy = np.zeros_like(psi, dtype=float)
    ux[1:-1, 1:-1] = (psi[1:-1, 2:] - psi[1:-1, :-2]) / (2.0 * dy)
    uy[1:-1, 1:-1] = -(psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2.0 * dx)
    uy[-1, :] = float(right_wall_velocity)
    return ux, uy


def _q2d_wall_driven_divergence(ux: np.ndarray, uy: np.ndarray, *, dx: float, dy: float) -> float:
    divergence = (ux[2:, 1:-1] - ux[:-2, 1:-1]) / (2.0 * dx) + (uy[1:-1, 2:] - uy[1:-1, :-2]) / (2.0 * dy)
    return float(np.max(np.abs(divergence))) if divergence.size else 0.0


def _q2d_spectral_operators(nx: int, ny: int, lx: float, ly: float) -> tuple[np.ndarray, ...]:
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=lx / max(nx, 1))
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=ly / max(ny, 1))
    kkx, kky = np.meshgrid(kx, ky, indexing="ij")
    k2 = kkx**2 + kky**2
    inv_k2 = np.zeros_like(k2)
    nonzero = k2 > 0.0
    inv_k2[nonzero] = 1.0 / k2[nonzero]
    kx_index = np.fft.fftfreq(nx) * nx
    ky_index = np.fft.fftfreq(ny) * ny
    ix, iy = np.meshgrid(kx_index, ky_index, indexing="ij")
    dealias = (np.abs(ix) <= nx / 3.0) & (np.abs(iy) <= ny / 3.0)
    return kkx, kky, k2, inv_k2, dealias


def _q2d_velocity_from_vorticity(
    omega: np.ndarray,
    *,
    kkx: np.ndarray,
    kky: np.ndarray,
    inv_k2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    omega_hat = np.fft.fft2(omega)
    psi_hat = omega_hat * inv_k2
    u = np.fft.ifft2(1j * kky * psi_hat).real
    v = np.fft.ifft2(-1j * kkx * psi_hat).real
    return u, v, omega_hat


def _q2d_nonlinear_rhs(
    omega: np.ndarray,
    *,
    forcing_hat: np.ndarray,
    kkx: np.ndarray,
    kky: np.ndarray,
    k2: np.ndarray,
    inv_k2: np.ndarray,
    dealias: np.ndarray,
    viscosity: float,
    hartmann_friction: float,
) -> np.ndarray:
    u, v, omega_hat = _q2d_velocity_from_vorticity(omega, kkx=kkx, kky=kky, inv_k2=inv_k2)
    omega_x = np.fft.ifft2(1j * kkx * omega_hat).real
    omega_y = np.fft.ifft2(1j * kky * omega_hat).real
    advective_hat = np.fft.fft2(u * omega_x + v * omega_y)
    advective_hat = np.where(dealias, advective_hat, 0.0)
    rhs_hat = -advective_hat - float(viscosity) * k2 * omega_hat - float(hartmann_friction) * omega_hat + forcing_hat
    rhs_hat[0, 0] = 0.0
    return np.fft.ifft2(rhs_hat).real


def _q2d_rk4_step(
    omega: np.ndarray,
    *,
    dt: float,
    forcing_hat: np.ndarray,
    kkx: np.ndarray,
    kky: np.ndarray,
    k2: np.ndarray,
    inv_k2: np.ndarray,
    dealias: np.ndarray,
    viscosity: float,
    hartmann_friction: float,
) -> np.ndarray:
    rhs_kwargs = {
        "forcing_hat": forcing_hat,
        "kkx": kkx,
        "kky": kky,
        "k2": k2,
        "inv_k2": inv_k2,
        "dealias": dealias,
        "viscosity": viscosity,
        "hartmann_friction": hartmann_friction,
    }
    k1 = _q2d_nonlinear_rhs(omega, **rhs_kwargs)
    k2_rhs = _q2d_nonlinear_rhs(omega + 0.5 * dt * k1, **rhs_kwargs)
    k3 = _q2d_nonlinear_rhs(omega + 0.5 * dt * k2_rhs, **rhs_kwargs)
    k4 = _q2d_nonlinear_rhs(omega + dt * k3, **rhs_kwargs)
    updated = omega + (dt / 6.0) * (k1 + 2.0 * k2_rhs + 2.0 * k3 + k4)
    updated -= float(np.mean(updated))
    return updated


def _q2d_record_vorticity_state(
    omega: np.ndarray,
    *,
    dt: float,
    dx: float,
    dy: float,
    kkx: np.ndarray,
    kky: np.ndarray,
    inv_k2: np.ndarray,
) -> tuple[float, float, float, float, float]:
    u, v, _ = _q2d_velocity_from_vorticity(omega, kkx=kkx, kky=kky, inv_k2=inv_k2)
    speed_squared = u**2 + v**2
    kinetic_energy = 0.5 * float(np.mean(speed_squared))
    enstrophy = 0.5 * float(np.mean(omega**2))
    velocity_rms = float(np.sqrt(max(float(np.mean(speed_squared)), 0.0)))
    max_speed = float(np.max(np.sqrt(speed_squared))) if speed_squared.size else 0.0
    max_courant = max_speed * float(dt) / max(min(float(dx), float(dy)), 1.0e-12)
    u_hat = np.fft.fft2(u)
    v_hat = np.fft.fft2(v)
    divergence = np.fft.ifft2(1j * kkx * u_hat + 1j * kky * v_hat).real
    divergence_linf = float(np.max(np.abs(divergence))) if divergence.size else 0.0
    return kinetic_energy, enstrophy, velocity_rms, max_courant, divergence_linf


def solve_q2d_decay(case: Q2DDecayCase) -> Q2DDecaySolution:
    x = np.linspace(0.0, case.lx, case.nx, endpoint=False)
    y = np.linspace(0.0, case.ly, case.ny, endpoint=False)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    dx = float(case.lx / case.nx)
    dy = float(case.ly / case.ny)
    initial = _mode_shape(case, xx, yy)
    field = initial.copy()

    kx = 2.0 * np.pi * case.mode_x / case.lx
    ky = 2.0 * np.pi * case.mode_y / case.ly
    decay_rate = case.viscosity * (kx**2 + ky**2) + case.hartmann_friction

    steps = max(1, int(round(case.t_final / case.dt)))
    time = np.linspace(0.0, steps * case.dt, steps + 1)
    amplitude_numeric = np.empty(steps + 1, dtype=float)
    amplitude_numeric[0] = float(np.max(np.abs(field)))
    for index in range(steps):
        field = field + case.dt * (case.viscosity * _periodic_laplacian(field, dx=dx, dy=dy) - case.hartmann_friction * field)
        amplitude_numeric[index + 1] = float(np.max(np.abs(field)))
    amplitude_analytic = case.amplitude * np.exp(-decay_rate * time)
    analytic_final = _mode_shape(case, xx, yy) * np.exp(-decay_rate * time[-1])
    return Q2DDecaySolution(
        x=x,
        y=y,
        initial=initial,
        final=field,
        analytic_final=analytic_final,
        time=time,
        amplitude_numeric=amplitude_numeric,
        amplitude_analytic=amplitude_analytic,
        decay_rate=float(decay_rate),
    )


def solve_q2d_turbulence_decay(case: Q2DTurbulenceDecayCase) -> Q2DTurbulenceDecaySolution:
    """Evolve a deterministic nonlinear Q2D vorticity field with Hartmann friction.

    The equation is the periodic vorticity form of the Sommeria-Moreau shallow
    core model with viscosity, linear Hartmann drag, and weak deterministic
    large-scale forcing. It is intentionally compact so that CI can exercise the
    nonlinear branch while the README example can run long enough to show
    vortex interaction rather than single-mode diffusion.
    """

    x = np.linspace(0.0, case.lx, case.nx, endpoint=False)
    y = np.linspace(0.0, case.ly, case.ny, endpoint=False)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    dx = float(case.lx / case.nx)
    dy = float(case.ly / case.ny)
    field = _q2d_multimode_initial_condition(case, xx, yy)
    forcing = _q2d_vorticity_forcing(case, xx, yy)
    kkx, kky, k2, inv_k2, dealias = _q2d_spectral_operators(case.nx, case.ny, case.lx, case.ly)
    forcing_hat = np.where(dealias, np.fft.fft2(forcing), 0.0)
    forcing_hat[0, 0] = 0.0
    steps = max(1, int(round(case.t_final / case.dt)))
    frame_count = max(2, min(int(case.frame_count), steps + 1))
    frame_indices = np.unique(np.linspace(0, steps, frame_count, dtype=int))
    frames: list[np.ndarray] = []
    frame_times: list[float] = []
    kinetic_energy: list[float] = []
    enstrophy_proxy: list[float] = []
    velocity_rms: list[float] = []
    max_courant: list[float] = []
    divergence_linf: list[float] = []
    turnover_count = 0.0
    previous_rms = 0.0

    def _record(step: int, values: np.ndarray) -> None:
        frames.append(values.copy())
        frame_times.append(step * case.dt)
        ke, enstrophy, rms, cfl, div_linf = _q2d_record_vorticity_state(
            values,
            dt=case.dt,
            dx=dx,
            dy=dy,
            kkx=kkx,
            kky=kky,
            inv_k2=inv_k2,
        )
        kinetic_energy.append(ke)
        enstrophy_proxy.append(enstrophy)
        velocity_rms.append(rms)
        max_courant.append(cfl)
        divergence_linf.append(div_linf)

    frame_index_set = set(int(index) for index in frame_indices.tolist())
    _record(0, field)
    previous_rms = velocity_rms[-1]
    for step in range(1, steps + 1):
        field = _q2d_rk4_step(
            field,
            dt=case.dt,
            forcing_hat=forcing_hat,
            kkx=kkx,
            kky=kky,
            k2=k2,
            inv_k2=inv_k2,
            dealias=dealias,
            viscosity=case.viscosity,
            hartmann_friction=case.hartmann_friction,
        )
        ke, _, rms, _, _ = _q2d_record_vorticity_state(
            field,
            dt=case.dt,
            dx=dx,
            dy=dy,
            kkx=kkx,
            kky=kky,
            inv_k2=inv_k2,
        )
        _ = ke
        turnover_count += 0.5 * (previous_rms + rms) * case.dt / max(min(case.lx, case.ly), 1.0e-12)
        previous_rms = rms
        if step in frame_index_set:
            _record(step, field)

    return Q2DTurbulenceDecaySolution(
        x=x,
        y=y,
        time=np.asarray(frame_times, dtype=float),
        frames=np.asarray(frames, dtype=float),
        kinetic_energy=np.asarray(kinetic_energy, dtype=float),
        enstrophy_proxy=np.asarray(enstrophy_proxy, dtype=float),
        velocity_rms=np.asarray(velocity_rms, dtype=float),
        max_courant=np.asarray(max_courant, dtype=float),
        divergence_linf=np.asarray(divergence_linf, dtype=float),
        turnover_count=float(turnover_count),
        initial_spectrum=q2d_energy_spectrum(frames[0] - float(np.mean(frames[0])), lx=case.lx, ly=case.ly),
        final_spectrum=q2d_energy_spectrum(frames[-1] - float(np.mean(frames[-1])), lx=case.lx, ly=case.ly),
    )


def solve_q2d_forced(case: Q2DForcedCase) -> Q2DForcedSolution:
    x = np.linspace(0.0, case.lx, case.nx, endpoint=False)
    y = np.linspace(0.0, case.ly, case.ny, endpoint=False)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    dx = float(case.lx / case.nx)
    dy = float(case.ly / case.ny)
    mode = case.forcing_amplitude * np.sin(2.0 * np.pi * case.mode_x * xx / case.lx) * np.sin(2.0 * np.pi * case.mode_y * yy / case.ly)
    field = np.zeros_like(mode)

    kx = 2.0 * np.pi * case.mode_x / case.lx
    ky = 2.0 * np.pi * case.mode_y / case.ly
    decay_rate = case.viscosity * (kx**2 + ky**2) + case.hartmann_friction
    steady_amplitude = case.forcing_amplitude / decay_rate
    steady_analytic = steady_amplitude * np.sin(2.0 * np.pi * case.mode_x * xx / case.lx) * np.sin(2.0 * np.pi * case.mode_y * yy / case.ly)

    steps = max(1, int(round(case.t_final / case.dt)))
    time = np.linspace(0.0, steps * case.dt, steps + 1)
    amplitude_numeric = np.empty(steps + 1, dtype=float)
    amplitude_numeric[0] = 0.0
    for index in range(steps):
        field = field + case.dt * (case.viscosity * _periodic_laplacian(field, dx=dx, dy=dy) - case.hartmann_friction * field + mode)
        amplitude_numeric[index + 1] = float(np.max(np.abs(field)))
    amplitude_analytic = steady_amplitude * (1.0 - np.exp(-decay_rate * time))
    return Q2DForcedSolution(
        x=x,
        y=y,
        steady_numeric=field,
        steady_analytic=steady_analytic,
        time=time,
        amplitude_numeric=amplitude_numeric,
        amplitude_analytic=amplitude_analytic,
        decay_rate=float(decay_rate),
        steady_amplitude=float(steady_amplitude),
    )


def solve_q2d_wall_bounded_forced(case: Q2DWallBoundedForcedCase) -> Q2DWallBoundedForcedSolution:
    x = np.linspace(0.0, case.lx, case.nx)
    y = np.linspace(0.0, case.ly, case.ny)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    dx = float(case.lx / max(case.nx - 1, 1))
    dy = float(case.ly / max(case.ny - 1, 1))
    mode = _wall_mode_shape(
        amplitude=case.forcing_amplitude,
        lx=case.lx,
        ly=case.ly,
        mode_x=case.mode_x,
        mode_y=case.mode_y,
        xx=xx,
        yy=yy,
    )
    field = np.zeros_like(mode)
    kx = case.mode_x * np.pi / case.lx
    ky = case.mode_y * np.pi / case.ly
    decay_rate = case.viscosity * (kx**2 + ky**2) + case.hartmann_friction
    steady_amplitude = case.forcing_amplitude / decay_rate
    steady_field = _wall_mode_shape(
        amplitude=steady_amplitude,
        lx=case.lx,
        ly=case.ly,
        mode_x=case.mode_x,
        mode_y=case.mode_y,
        xx=xx,
        yy=yy,
    )
    steps = max(1, int(round(case.t_final / case.dt)))
    time = np.linspace(0.0, steps * case.dt, steps + 1)
    amplitude_numeric = np.empty(steps + 1, dtype=float)
    amplitude_numeric[0] = 0.0
    for index in range(steps):
        field = field + case.dt * (case.viscosity * _dirichlet_laplacian(field, dx=dx, dy=dy) - case.hartmann_friction * field + mode)
        field[0, :] = 0.0
        field[-1, :] = 0.0
        field[:, 0] = 0.0
        field[:, -1] = 0.0
        amplitude_numeric[index + 1] = float(np.max(np.abs(field)))
    amplitude_analytic = steady_amplitude * (1.0 - np.exp(-decay_rate * time))
    analytic_final = steady_field * (amplitude_analytic[-1] / max(steady_amplitude, 1.0e-12))
    return Q2DWallBoundedForcedSolution(
        x=x,
        y=y,
        field=field,
        analytic_final=analytic_final,
        time=time,
        amplitude_numeric=amplitude_numeric,
        amplitude_analytic=amplitude_analytic,
        decay_rate=float(decay_rate),
        steady_amplitude=float(steady_amplitude),
    )


def solve_q2d_wall_driven_cavity(case: Q2DWallDrivenCavityCase) -> Q2DWallDrivenCavitySolution:
    """Solve a bounded Q2D side-wall-driven cavity with linear Hartmann drag.

    This is the LMX counterpart to the Q2DmhdFoam `run/lidDriven` smoke case:
    zero buoyancy, no pressure-driven mean flow, no-slip walls, and an imposed
    vertical velocity on the right boundary. The vorticity-streamfunction form
    keeps the velocity divergence-free while preserving a compact CI-scale
    implementation for external-code parity studies.
    """

    x = np.linspace(0.0, case.lx, case.nx)
    y = np.linspace(-0.5 * case.ly, 0.5 * case.ly, case.ny)
    dx = float(case.lx / max(case.nx - 1, 1))
    dy = float(case.ly / max(case.ny - 1, 1))
    omega = np.zeros((case.nx, case.ny), dtype=float)
    psi = np.zeros_like(omega)
    steps = max(1, int(round(case.t_final / case.dt)))
    frame_count = max(2, min(int(case.frame_count), steps + 1))
    frame_indices = set(int(index) for index in np.unique(np.linspace(0, steps, frame_count, dtype=int)).tolist())

    frame_times: list[float] = []
    psi_frames: list[np.ndarray] = []
    omega_frames: list[np.ndarray] = []
    ux_frames: list[np.ndarray] = []
    uy_frames: list[np.ndarray] = []
    kinetic_energy: list[float] = []
    enstrophy: list[float] = []
    max_courant: list[float] = []
    divergence_linf: list[float] = []

    def _record(step: int) -> tuple[np.ndarray, np.ndarray]:
        nonlocal psi
        _q2d_wall_driven_apply_vorticity_boundary(
            omega,
            psi,
            dx=dx,
            dy=dy,
            right_wall_velocity=case.right_wall_velocity,
        )
        psi = _q2d_wall_driven_streamfunction(omega, dx=dx, dy=dy)
        ux, uy = _q2d_wall_driven_velocity(
            psi,
            dx=dx,
            dy=dy,
            right_wall_velocity=case.right_wall_velocity,
        )
        speed_squared = ux**2 + uy**2
        frame_times.append(step * case.dt)
        psi_frames.append(psi.copy())
        omega_frames.append(omega.copy())
        ux_frames.append(ux.copy())
        uy_frames.append(uy.copy())
        kinetic_energy.append(0.5 * float(np.mean(speed_squared)))
        enstrophy.append(0.5 * float(np.mean(omega**2)))
        max_speed = float(np.max(np.sqrt(speed_squared))) if speed_squared.size else 0.0
        max_courant.append(max_speed * case.dt / max(min(dx, dy), 1.0e-12))
        divergence_linf.append(_q2d_wall_driven_divergence(ux, uy, dx=dx, dy=dy))
        return ux, uy

    _record(0)
    for step in range(1, steps + 1):
        _q2d_wall_driven_apply_vorticity_boundary(
            omega,
            psi,
            dx=dx,
            dy=dy,
            right_wall_velocity=case.right_wall_velocity,
        )
        psi = _q2d_wall_driven_streamfunction(omega, dx=dx, dy=dy)
        ux, uy = _q2d_wall_driven_velocity(
            psi,
            dx=dx,
            dy=dy,
            right_wall_velocity=case.right_wall_velocity,
        )
        interior = omega[1:-1, 1:-1]
        omega_x = np.where(
            ux[1:-1, 1:-1] >= 0.0,
            (interior - omega[:-2, 1:-1]) / dx,
            (omega[2:, 1:-1] - interior) / dx,
        )
        omega_y = np.where(
            uy[1:-1, 1:-1] >= 0.0,
            (interior - omega[1:-1, :-2]) / dy,
            (omega[1:-1, 2:] - interior) / dy,
        )
        laplacian = (
            (omega[2:, 1:-1] - 2.0 * interior + omega[:-2, 1:-1]) / dx**2
            + (omega[1:-1, 2:] - 2.0 * interior + omega[1:-1, :-2]) / dy**2
        )
        omega[1:-1, 1:-1] = interior + case.dt * (
            -ux[1:-1, 1:-1] * omega_x
            - uy[1:-1, 1:-1] * omega_y
            + case.viscosity * laplacian
            - case.hartmann_friction * interior
        )
        if step in frame_indices:
            _record(step)

    if steps not in frame_indices:
        _record(steps)
    return Q2DWallDrivenCavitySolution(
        x=x,
        y=y,
        time=np.asarray(frame_times, dtype=float),
        streamfunction_frames=np.asarray(psi_frames, dtype=float),
        vorticity_frames=np.asarray(omega_frames, dtype=float),
        ux_frames=np.asarray(ux_frames, dtype=float),
        uy_frames=np.asarray(uy_frames, dtype=float),
        kinetic_energy=np.asarray(kinetic_energy, dtype=float),
        enstrophy=np.asarray(enstrophy, dtype=float),
        max_courant=np.asarray(max_courant, dtype=float),
        divergence_linf=np.asarray(divergence_linf, dtype=float),
    )


def validate_q2d_decay_solution(case: Q2DDecayCase, solution: Q2DDecaySolution) -> dict[str, float | bool]:
    analytic_norm = max(float(np.linalg.norm(solution.analytic_final)), 1.0e-12)
    l2_error = float(np.linalg.norm(solution.final - solution.analytic_final) / analytic_norm)
    linf_error = float(np.max(np.abs(solution.final - solution.analytic_final)) / max(float(np.max(np.abs(solution.analytic_final))), 1.0e-12))
    amplitude_rel_error = float(
        abs(solution.amplitude_numeric[-1] - solution.amplitude_analytic[-1]) / max(abs(solution.amplitude_analytic[-1]), 1.0e-12)
    )
    validation_pass = bool(l2_error <= 5.0e-2 and linf_error <= 8.0e-2 and amplitude_rel_error <= 5.0e-2)
    return {
        "decay_rate": float(solution.decay_rate),
        "l2_error": l2_error,
        "linf_error": linf_error,
        "amplitude_rel_error": amplitude_rel_error,
        "validation_pass": validation_pass,
    }


def validate_q2d_forced_solution(case: Q2DForcedCase, solution: Q2DForcedSolution) -> dict[str, float | bool]:
    transient_factor = solution.amplitude_analytic[-1] / max(solution.steady_amplitude, 1.0e-12)
    analytic_final = solution.steady_analytic * transient_factor
    analytic_norm = max(float(np.linalg.norm(analytic_final)), 1.0e-12)
    l2_error = float(np.linalg.norm(solution.steady_numeric - analytic_final) / analytic_norm)
    linf_error = float(
        np.max(np.abs(solution.steady_numeric - analytic_final))
        / max(float(np.max(np.abs(analytic_final))), 1.0e-12)
    )
    steady_amplitude_rel_error = float(
        abs(solution.amplitude_numeric[-1] - solution.amplitude_analytic[-1]) / max(abs(solution.amplitude_analytic[-1]), 1.0e-12)
    )
    validation_pass = bool(l2_error <= 7.0e-2 and linf_error <= 1.0e-1 and steady_amplitude_rel_error <= 7.0e-2)
    return {
        "decay_rate": float(solution.decay_rate),
        "steady_amplitude": float(solution.steady_amplitude),
        "l2_error": l2_error,
        "linf_error": linf_error,
        "steady_amplitude_rel_error": steady_amplitude_rel_error,
        "validation_pass": validation_pass,
    }


def validate_q2d_wall_bounded_forced_solution(
    case: Q2DWallBoundedForcedCase,
    solution: Q2DWallBoundedForcedSolution,
) -> dict[str, float | bool]:
    analytic_norm = max(float(np.linalg.norm(solution.analytic_final)), 1.0e-12)
    l2_error = float(np.linalg.norm(solution.field - solution.analytic_final) / analytic_norm)
    linf_error = float(
        np.max(np.abs(solution.field - solution.analytic_final))
        / max(float(np.max(np.abs(solution.analytic_final))), 1.0e-12)
    )
    amplitude_rel_error = float(
        abs(solution.amplitude_numeric[-1] - solution.amplitude_analytic[-1]) / max(abs(solution.amplitude_analytic[-1]), 1.0e-12)
    )
    validation_pass = bool(l2_error <= 8.0e-2 and linf_error <= 1.1e-1 and amplitude_rel_error <= 8.0e-2)
    return {
        "decay_rate": float(solution.decay_rate),
        "steady_amplitude": float(solution.steady_amplitude),
        "l2_error": l2_error,
        "linf_error": linf_error,
        "amplitude_rel_error": amplitude_rel_error,
        "validation_pass": validation_pass,
    }


def q2d_modal_energy_budget(
    *,
    time: np.ndarray,
    amplitude: np.ndarray,
    decay_rate: float,
    mode_mean_square: float,
    mode_peak: float = 1.0,
    forcing_amplitude: float = 0.0,
    relative_tolerance: float = 6.0e-2,
) -> dict[str, float | bool]:
    """Check the modal Q2D energy budget.

    For a single Q2D mode, ``dE/dt = P - 2 lambda E`` where ``lambda`` is the
    viscous-plus-Hartmann decay rate and ``P`` is the modal forcing production.
    This is the compact Sommeria-Moreau-facing closure used by the validation
    examples before adding turbulent reference spectra.
    """

    time_values = np.asarray(time, dtype=float)
    amplitude_values = np.asarray(amplitude, dtype=float)
    if time_values.ndim != 1 or amplitude_values.ndim != 1 or time_values.size != amplitude_values.size:
        raise ValueError("Q2D modal energy budget expects matching 1D time and amplitude arrays")
    if time_values.size < 3:
        raise ValueError("Q2D modal energy budget requires at least three samples")
    if not np.all(np.diff(time_values) > 0.0):
        raise ValueError("Q2D modal energy budget requires strictly increasing time")
    coefficient = amplitude_values / max(abs(float(mode_peak)), 1.0e-12)
    mean_square = max(float(mode_mean_square), 1.0e-20)
    energy = 0.5 * mean_square * coefficient**2
    production = float(forcing_amplitude) * mean_square * coefficient
    dissipation = 2.0 * float(decay_rate) * energy
    derivative = np.gradient(energy, time_values, edge_order=2)
    residual = derivative - (production - dissipation)
    interior = slice(1, -1)
    scale = max(float(np.linalg.norm((production - dissipation)[interior])), 1.0e-20)
    relative_l2 = float(np.linalg.norm(residual[interior]) / scale)
    return {
        "initial_energy": float(energy[0]),
        "final_energy": float(energy[-1]),
        "peak_production": float(np.max(np.abs(production))),
        "peak_dissipation": float(np.max(np.abs(dissipation))),
        "relative_budget_l2": relative_l2,
        "max_abs_budget_residual": float(np.max(np.abs(residual[interior]))),
        "validation_pass": bool(relative_l2 <= relative_tolerance and np.all(energy >= -1.0e-14)),
        "literature_target": "Sommeria-Moreau Q2D modal energy balance",
    }


def validate_q2d_decay_energy_budget(
    case: Q2DDecayCase,
    solution: Q2DDecaySolution,
) -> dict[str, float | bool]:
    mode_mean_square, mode_peak = _periodic_mode_statistics(
        nx=case.nx,
        ny=case.ny,
        lx=case.lx,
        ly=case.ly,
        mode_x=case.mode_x,
        mode_y=case.mode_y,
    )
    return q2d_modal_energy_budget(
        time=solution.time,
        amplitude=solution.amplitude_numeric,
        decay_rate=solution.decay_rate,
        mode_mean_square=mode_mean_square,
        mode_peak=mode_peak,
    )


def validate_q2d_forced_energy_budget(
    case: Q2DForcedCase,
    solution: Q2DForcedSolution,
) -> dict[str, float | bool]:
    mode_mean_square, mode_peak = _periodic_mode_statistics(
        nx=case.nx,
        ny=case.ny,
        lx=case.lx,
        ly=case.ly,
        mode_x=case.mode_x,
        mode_y=case.mode_y,
    )
    return q2d_modal_energy_budget(
        time=solution.time,
        amplitude=solution.amplitude_numeric,
        decay_rate=solution.decay_rate,
        mode_mean_square=mode_mean_square,
        mode_peak=mode_peak,
        forcing_amplitude=case.forcing_amplitude,
    )


def validate_q2d_wall_bounded_energy_budget(
    case: Q2DWallBoundedForcedCase,
    solution: Q2DWallBoundedForcedSolution,
) -> dict[str, float | bool]:
    mode_mean_square, mode_peak = _wall_mode_statistics(
        nx=case.nx,
        ny=case.ny,
        lx=case.lx,
        ly=case.ly,
        mode_x=case.mode_x,
        mode_y=case.mode_y,
    )
    return q2d_modal_energy_budget(
        time=solution.time,
        amplitude=solution.amplitude_numeric,
        decay_rate=solution.decay_rate,
        mode_mean_square=mode_mean_square,
        mode_peak=mode_peak,
        forcing_amplitude=case.forcing_amplitude,
    )


def q2d_energy_spectrum(
    field: np.ndarray,
    *,
    lx: float,
    ly: float,
    bins: int = 16,
) -> dict[str, list[float]]:
    """Return an isotropic shell spectrum for a scalar Q2D field.

    The returned shell energies are integrated over Fourier modes in each
    radial wavenumber bin. This is intentionally lightweight: it provides the
    spectral observables needed for validation summaries without committing the
    solver to one turbulent-spectrum normalization convention.
    """

    values = np.asarray(field, dtype=float)
    if values.ndim != 2:
        raise ValueError("Q2D energy spectrum expects a 2D field")
    if bins < 1:
        raise ValueError("Q2D energy spectrum requires at least one bin")
    nx, ny = values.shape
    spectrum_density = 0.5 * np.abs(np.fft.fft2(values)) ** 2 / max(nx * ny, 1) ** 2
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=lx / max(nx, 1))
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=ly / max(ny, 1))
    kkx, kky = np.meshgrid(kx, ky, indexing="ij")
    k_mag = np.sqrt(kkx**2 + kky**2)
    max_k = float(np.max(k_mag)) if k_mag.size else 0.0
    edges = np.linspace(0.0, max_k, max(2, bins + 1))
    shell_energy = np.zeros(edges.size - 1, dtype=float)
    shell_counts = np.zeros(edges.size - 1, dtype=int)
    for index in range(shell_energy.size):
        if index == shell_energy.size - 1:
            mask = (k_mag >= edges[index]) & (k_mag <= edges[index + 1])
        else:
            mask = (k_mag >= edges[index]) & (k_mag < edges[index + 1])
        shell_energy[index] = float(np.sum(spectrum_density[mask]))
        shell_counts[index] = int(np.count_nonzero(mask))
    centers = 0.5 * (edges[:-1] + edges[1:])
    return {"wavenumber": centers.tolist(), "energy": shell_energy.tolist(), "counts": shell_counts.tolist()}


def q2d_turbulence_readiness_metrics(
    field: np.ndarray,
    *,
    lx: float,
    ly: float,
    viscosity: float,
    hartmann_friction: float,
) -> dict[str, object]:
    """Compute Sommeria-Moreau-facing observables for future Q2D tests."""

    values = np.asarray(field, dtype=float)
    if values.ndim != 2:
        raise ValueError("Q2D turbulence readiness metrics expect a 2D field")
    dx = lx / max(values.shape[0] - 1, 1)
    dy = ly / max(values.shape[1] - 1, 1)
    grad_x, grad_y = np.gradient(values, dx, dy, edge_order=1)
    fluctuation = values - float(np.mean(values))
    kinetic_energy = 0.5 * float(np.mean(values**2))
    fluctuation_kinetic_energy = 0.5 * float(np.mean(fluctuation**2))
    enstrophy_proxy = 0.5 * float(np.mean(grad_x**2 + grad_y**2))
    hartmann_dissipation_proxy = float(hartmann_friction) * float(np.mean(values**2))
    viscous_dissipation_proxy = 2.0 * float(viscosity) * enstrophy_proxy
    spectrum = q2d_energy_spectrum(fluctuation, lx=lx, ly=ly)
    spectrum_energy = np.asarray(spectrum["energy"], dtype=float)
    spectrum_wavenumber = np.asarray(spectrum["wavenumber"], dtype=float)
    total_spectral_energy = float(np.sum(spectrum_energy))
    peak_index = int(np.argmax(spectrum_energy)) if spectrum_energy.size else 0
    high_k_cutoff = float(np.percentile(spectrum_wavenumber, 75.0)) if spectrum_wavenumber.size else 0.0
    high_k_mask = spectrum_wavenumber >= high_k_cutoff
    high_k_energy_fraction = (
        float(np.sum(spectrum_energy[high_k_mask]) / total_spectral_energy)
        if total_spectral_energy > 0.0
        else 0.0
    )
    positive = (spectrum_wavenumber > 0.0) & (spectrum_energy > 0.0)
    spectrum_log_slope = 0.0
    if int(np.count_nonzero(positive)) >= 2:
        log_k = np.log(spectrum_wavenumber[positive])
        log_energy = np.log(spectrum_energy[positive])
        spectrum_log_slope = float(np.polyfit(log_k, log_energy, 1)[0])
    return {
        "kinetic_energy": kinetic_energy,
        "fluctuation_kinetic_energy": fluctuation_kinetic_energy,
        "enstrophy_proxy": enstrophy_proxy,
        "hartmann_dissipation_proxy": hartmann_dissipation_proxy,
        "viscous_dissipation_proxy": viscous_dissipation_proxy,
        "spectrum_peak_wavenumber": float(spectrum_wavenumber[peak_index]) if spectrum_wavenumber.size else 0.0,
        "spectrum_log_slope": spectrum_log_slope,
        "high_wavenumber_energy_fraction": high_k_energy_fraction,
        "spectrum": spectrum,
        "literature_target": "Sommeria-Moreau quasi-2D turbulence observables",
        "required_next_observables": [
            "energy_decay_exponent_or_rate",
            "energy_spectrum_slope",
            "inverse_cascade_or_large-scale-condensate trend",
            "Hartmann-friction damping trend",
        ],
        "validation_status": "spectral_observables_available_no_turbulent_reference",
        "research_grade_turbulence_validation_pass": False,
    }


def _periodic_mode_statistics(
    *,
    nx: int,
    ny: int,
    lx: float,
    ly: float,
    mode_x: int,
    mode_y: int,
) -> tuple[float, float]:
    x = np.linspace(0.0, lx, nx, endpoint=False)
    y = np.linspace(0.0, ly, ny, endpoint=False)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    mode = np.sin(2.0 * np.pi * mode_x * xx / lx) * np.sin(2.0 * np.pi * mode_y * yy / ly)
    return float(np.mean(mode**2)), max(float(np.max(np.abs(mode))), 1.0e-12)


def _wall_mode_statistics(
    *,
    nx: int,
    ny: int,
    lx: float,
    ly: float,
    mode_x: int,
    mode_y: int,
) -> tuple[float, float]:
    x = np.linspace(0.0, lx, nx)
    y = np.linspace(0.0, ly, ny)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    mode = _wall_mode_shape(
        amplitude=1.0,
        lx=lx,
        ly=ly,
        mode_x=mode_x,
        mode_y=mode_y,
        xx=xx,
        yy=yy,
    )
    return float(np.mean(mode**2)), max(float(np.max(np.abs(mode))), 1.0e-12)


def q2d_turbulence_observables(
    field: np.ndarray,
    *,
    lx: float,
    ly: float,
    viscosity: float,
    hartmann_friction: float,
) -> dict[str, object]:
    """Compatibility alias for Q2D turbulence-readiness observables."""

    return q2d_turbulence_readiness_metrics(
        field,
        lx=lx,
        ly=ly,
        viscosity=viscosity,
        hartmann_friction=hartmann_friction,
    )


def validate_q2d_turbulence_decay_observables(
    case: Q2DTurbulenceDecayCase,
    solution: Q2DTurbulenceDecaySolution,
) -> dict[str, float | bool | str]:
    """Validate bounded nonlinear Q2D spectral observables for the movie lane."""

    energy = np.asarray(solution.kinetic_energy, dtype=float)
    enstrophy = np.asarray(solution.enstrophy_proxy, dtype=float)
    if energy.size < 2 or enstrophy.size < 2:
        raise ValueError("Q2D turbulence-decay validation requires at least two frames")
    finite_energy = bool(np.all(np.isfinite(energy)) and np.all(energy >= -1.0e-14))
    finite_enstrophy = bool(np.all(np.isfinite(enstrophy)) and np.all(enstrophy >= -1.0e-14))
    energy_decay_ratio = float(energy[-1] / max(energy[0], 1.0e-20))
    enstrophy_decay_ratio = float(enstrophy[-1] / max(enstrophy[0], 1.0e-20))
    energy_variation_ratio = float((np.max(energy) - np.min(energy)) / max(np.mean(energy), 1.0e-20))
    enstrophy_variation_ratio = float((np.max(enstrophy) - np.min(enstrophy)) / max(np.mean(enstrophy), 1.0e-20))

    initial_energy = np.asarray(solution.initial_spectrum["energy"], dtype=float)
    final_energy = np.asarray(solution.final_spectrum["energy"], dtype=float)
    wavenumber = np.asarray(solution.initial_spectrum["wavenumber"], dtype=float)
    active = initial_energy > max(float(np.max(initial_energy)) * 1.0e-12, 1.0e-30) if initial_energy.size else np.asarray([], dtype=bool)
    active_wavenumber = wavenumber[active] if wavenumber.size and active.size else np.asarray([], dtype=float)
    cutoff = 0.5 * float(np.max(active_wavenumber)) if active_wavenumber.size else 0.0
    high_k = wavenumber >= cutoff
    initial_high_k_fraction = (
        float(np.sum(initial_energy[high_k]) / max(np.sum(initial_energy), 1.0e-20))
        if initial_energy.size
        else 0.0
    )
    final_high_k_fraction = (
        float(np.sum(final_energy[high_k]) / max(np.sum(final_energy), 1.0e-20))
        if final_energy.size
        else 0.0
    )
    initial_spectral_centroid = float(np.sum(wavenumber * initial_energy) / max(np.sum(initial_energy), 1.0e-20)) if initial_energy.size else 0.0
    final_spectral_centroid = float(np.sum(wavenumber * final_energy) / max(np.sum(final_energy), 1.0e-20)) if final_energy.size else 0.0
    spectral_centroid_shift = float(abs(final_spectral_centroid - initial_spectral_centroid))
    max_courant = float(np.max(solution.max_courant)) if solution.max_courant.size else 0.0
    max_divergence_linf = float(np.max(solution.divergence_linf)) if solution.divergence_linf.size else 0.0
    nonlinear_activity_pass = bool(float(solution.turnover_count) >= 0.12 and spectral_centroid_shift > 1.0e-3)
    validation_pass = bool(
        finite_energy
        and finite_enstrophy
        and max_courant < 0.45
        and max_divergence_linf < 1.0e-9
        and nonlinear_activity_pass
        and solution.frames.shape[0] >= 8
    )
    return {
        "energy_decay_ratio": energy_decay_ratio,
        "enstrophy_decay_ratio": enstrophy_decay_ratio,
        "energy_variation_ratio": energy_variation_ratio,
        "enstrophy_variation_ratio": enstrophy_variation_ratio,
        "initial_high_k_energy_fraction": initial_high_k_fraction,
        "final_high_k_energy_fraction": final_high_k_fraction,
        "initial_spectral_centroid": initial_spectral_centroid,
        "final_spectral_centroid": final_spectral_centroid,
        "spectral_centroid_shift": spectral_centroid_shift,
        "max_courant": max_courant,
        "max_divergence_linf": max_divergence_linf,
        "turnover_count": float(solution.turnover_count),
        "nonlinear_activity_pass": nonlinear_activity_pass,
        "frame_count": int(solution.frames.shape[0]),
        "validation_pass": validation_pass,
        "literature_target": "Sommeria-Moreau quasi-2D turbulence with Hartmann-friction damping",
        "validation_status": "nonlinear_q2d_movie_available_external_turbulent_reference_open",
        "research_grade_turbulence_validation_pass": False,
    }


def q2d_wall_driven_cavity_observables(
    case: Q2DWallDrivenCavityCase,
    solution: Q2DWallDrivenCavitySolution,
) -> dict[str, float | int | str | bool]:
    """Return field observables for a side-wall-driven Q2D cavity run."""

    ux = np.asarray(solution.ux_frames[-1], dtype=float)
    uy = np.asarray(solution.uy_frames[-1], dtype=float)
    omega = np.asarray(solution.vorticity_frames[-1], dtype=float)
    speed = np.sqrt(ux**2 + uy**2)
    return {
        "sample_count": int(speed.size),
        "speed_mean": float(np.mean(speed)),
        "speed_max": float(np.max(speed)),
        "speed_rms": float(np.sqrt(np.mean(speed**2))),
        "ux_mean": float(np.mean(ux)),
        "uy_mean": float(np.mean(uy)),
        "vorticity_peak": float(np.max(np.abs(omega))),
        "kinetic_energy_final": float(solution.kinetic_energy[-1]),
        "enstrophy_final": float(solution.enstrophy[-1]),
        "max_courant": float(np.max(solution.max_courant)) if solution.max_courant.size else 0.0,
        "max_divergence_linf": float(np.max(solution.divergence_linf)) if solution.divergence_linf.size else 0.0,
        "frame_count": int(solution.time.size),
        "final_time": float(solution.time[-1]),
        "right_wall_velocity": float(case.right_wall_velocity),
        "hartmann_friction": float(case.hartmann_friction),
        "viscosity": float(case.viscosity),
        "validation_pass": bool(
            np.all(np.isfinite(speed))
            and np.all(np.isfinite(omega))
            and float(np.max(solution.max_courant)) < 0.4
            and float(np.max(solution.divergence_linf)) < 1.0e-9
            and abs(float(np.max(speed)) - float(case.right_wall_velocity)) <= 1.0e-10
        ),
        "literature_target": "Sommeria-Moreau/Q2DmhdFoam side-wall-driven Hartmann-friction cavity",
    }


def compare_q2d_wall_driven_observables(
    lmx_observables: Mapping[str, float | int | str | bool],
    reference_observables: Mapping[str, float | int | str | bool],
    *,
    relative_tolerance: float = 0.20,
) -> dict[str, object]:
    """Compare compact LMX and Q2DmhdFoam side-wall-driven observables."""

    keys = ("speed_mean", "speed_rms", "uy_mean", "vorticity_peak")
    rows: list[dict[str, float | str | bool]] = []
    passed = 0
    for key in keys:
        lmx_value = float(lmx_observables[key])
        reference_value = float(reference_observables[key])
        absolute_error = abs(lmx_value - reference_value)
        relative_error = absolute_error / max(abs(reference_value), 1.0e-20)
        validation_pass = bool(relative_error <= relative_tolerance)
        passed += int(validation_pass)
        rows.append(
            {
                "observable": key,
                "lmx_value": lmx_value,
                "reference_value": reference_value,
                "absolute_error": absolute_error,
                "relative_error": relative_error,
                "relative_tolerance": relative_tolerance,
                "validation_pass": validation_pass,
            }
        )
    strict_pass = bool(passed == len(rows) and bool(lmx_observables.get("validation_pass", False)))
    return {
        "rows": rows,
        "compared_observable_count": len(rows),
        "passed_observable_count": passed,
        "relative_tolerance": relative_tolerance,
        "validation_pass": strict_pass,
        "matched_parity": strict_pass,
        "status": "matched_side_wall_observable_comparison" if strict_pass else "matched_side_wall_observable_offenders",
    }


def write_q2d_wall_driven_comparison_plots(
    case: Q2DWallDrivenCavityCase,
    solution: Q2DWallDrivenCavitySolution,
    comparison: Mapping[str, object],
    output_dir: str | Path,
    *,
    output_stem: str = "q2d_lmx_q2dmhdfoam_lid_driven_parity",
    reference_speed_grid: np.ndarray | None = None,
    reference_x: np.ndarray | None = None,
    reference_y: np.ndarray | None = None,
) -> list[Path]:
    """Write a publication-facing matched side-wall Q2D comparison panel."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ux = np.asarray(solution.ux_frames[-1], dtype=float)
    uy = np.asarray(solution.uy_frames[-1], dtype=float)
    speed = np.sqrt(ux**2 + uy**2)
    vorticity = np.asarray(solution.vorticity_frames[-1], dtype=float)
    x_edges = _centers_to_edges_1d(solution.x)
    y_edges = _centers_to_edges_1d(solution.y)
    vmax_speed = max(float(np.max(speed)), 1.0e-12)
    vmax_vort = max(float(np.max(np.abs(vorticity))), 1.0e-12)

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.0), constrained_layout=True)
    im0 = axes[0, 0].pcolormesh(
        solution.x,
        solution.y,
        speed.T,
        shading="auto",
        cmap="magma",
        vmin=0.0,
        vmax=vmax_speed,
    )
    axes[0, 0].set_title("LMX wall-driven speed")
    axes[0, 0].set_xlabel("x [m]")
    axes[0, 0].set_ylabel("y [m]")
    axes[0, 0].set_aspect("equal")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04, label="|U| [m/s]")

    if reference_speed_grid is not None and reference_x is not None and reference_y is not None:
        im1 = axes[0, 1].pcolormesh(
            np.asarray(reference_x, dtype=float),
            np.asarray(reference_y, dtype=float),
            np.asarray(reference_speed_grid, dtype=float),
            shading="auto",
            cmap="magma",
            vmin=0.0,
            vmax=vmax_speed,
        )
        axes[0, 1].set_title("Q2DmhdFoam VTK speed")
        fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04, label="|U| [m/s]")
    else:
        im1 = axes[0, 1].pcolormesh(x_edges, y_edges, vorticity.T, shading="auto", cmap="RdBu_r", vmin=-vmax_vort, vmax=vmax_vort)
        axes[0, 1].set_title("LMX vorticity")
        fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04, label="omega [1/s]")
    axes[0, 1].set_xlabel("x [m]")
    axes[0, 1].set_ylabel("y [m]")
    axes[0, 1].set_aspect("equal")

    kinetic_normalized = solution.kinetic_energy / max(float(np.max(solution.kinetic_energy)), 1.0e-20)
    enstrophy_normalized = solution.enstrophy / max(float(np.max(solution.enstrophy)), 1.0e-20)
    axes[1, 0].plot(solution.time, kinetic_normalized, color="#0f766e", linewidth=2.0, label="kinetic energy / max")
    axes[1, 0].plot(solution.time, enstrophy_normalized, color="#b45309", linewidth=2.0, label="enstrophy / max")
    axes[1, 0].set_title("LMX transient diagnostics")
    axes[1, 0].set_xlabel("time [s]")
    axes[1, 0].set_ylabel("diagnostic")
    axes[1, 0].grid(True, alpha=0.25)
    axes[1, 0].legend(frameon=False)

    rows = list(comparison.get("rows", []))
    labels = [str(row["observable"]).replace("_", "\n") for row in rows]
    ratios = np.asarray([float(row["relative_error"]) / max(float(row["relative_tolerance"]), 1.0e-20) for row in rows], dtype=float)
    colors = ["#2a9d8f" if bool(row["validation_pass"]) else "#c2410c" for row in rows]
    x = np.arange(len(rows), dtype=float)
    axes[1, 1].bar(x, ratios, color=colors)
    axes[1, 1].axhline(1.0, color="black", linestyle="--", linewidth=1.0, label="tolerance")
    axes[1, 1].set_title("LMX vs Q2DmhdFoam observable errors")
    axes[1, 1].set_xticks(x, labels)
    axes[1, 1].set_ylabel("relative error / tolerance")
    axes[1, 1].grid(True, axis="y", alpha=0.25)
    axes[1, 1].legend(frameon=False)
    axes[1, 1].text(
        0.98,
        0.95,
        f"matched parity: {bool(comparison.get('matched_parity', False))}",
        ha="right",
        va="top",
        transform=axes[1, 1].transAxes,
        bbox={"facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.9},
    )

    fig.suptitle("LMX/Q2DmhdFoam matched side-wall Q2D comparison", fontsize=15.0, fontweight="bold")
    paths = [out_dir / f"{output_stem}.png", out_dir / f"{output_stem}.pdf"]
    for path in paths:
        fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return paths


def write_q2d_turbulence_decay_movie(
    solution: Q2DTurbulenceDecaySolution,
    out_dir: str | Path,
    *,
    title: str = "Q2D nonlinear Hartmann-friction turbulence slice",
    fps: int = 10,
) -> list[Path]:
    """Write a GIF movie and poster for a Q2D multi-mode decay solution."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = np.asarray(solution.frames, dtype=float)
    if frames.ndim != 3:
        raise ValueError("Q2D turbulence-decay movie expects frames with shape (time, nx, ny)")
    x_edges = _centers_to_edges_1d(solution.x)
    y_edges = _centers_to_edges_1d(solution.y)
    vmax = max(float(np.max(np.abs(frames))), 1.0e-12)

    fig, ax = plt.subplots(figsize=(6.0, 5.2), constrained_layout=True)
    image = ax.pcolormesh(y_edges, x_edges, frames[0], shading="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_title(f"{title}\nt = {solution.time[0]:.3f}")
    ax.set_xlabel("y")
    ax.set_ylabel("x")
    ax.set_aspect("equal")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="vorticity")

    def _update(index: int):
        image.set_array(frames[index].ravel())
        ax.set_title(f"{title}\nt = {solution.time[index]:.3f}")
        return (image,)

    movie = animation.FuncAnimation(fig, _update, frames=frames.shape[0], interval=1000.0 / max(fps, 1), blit=False)
    gif_path = out_dir / "q2d_turbulence_decay.gif"
    poster_path = out_dir / "q2d_turbulence_decay_poster.png"
    writer = animation.PillowWriter(fps=max(int(fps), 1))
    movie.save(gif_path, writer=writer)
    _update(frames.shape[0] - 1)
    fig.savefig(poster_path, bbox_inches="tight")
    plt.close(fig)
    return [gif_path, poster_path]


def _centers_to_edges_1d(values: np.ndarray) -> np.ndarray:
    data = np.asarray(values, dtype=float)
    if data.size <= 1:
        center = float(data[0]) if data.size else 0.0
        return np.asarray([center - 0.5, center + 0.5], dtype=float)
    midpoints = 0.5 * (data[1:] + data[:-1])
    first = data[0] - 0.5 * (data[1] - data[0])
    last = data[-1] + 0.5 * (data[-1] - data[-2])
    return np.concatenate([[first], midpoints, [last]])


def write_q2d_turbulence_observable_plots(
    field: np.ndarray,
    out_dir: str | Path,
    *,
    lx: float,
    ly: float,
    viscosity: float,
    hartmann_friction: float,
    title: str = "Q2D turbulence-observable readiness gate",
) -> list[Path]:
    """Write a publication-facing panel for Q2D spectral observables."""

    values = np.asarray(field, dtype=float)
    if values.ndim != 2:
        raise ValueError("Q2D turbulence observable plots expect a 2D field")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = q2d_turbulence_observables(
        values,
        lx=lx,
        ly=ly,
        viscosity=viscosity,
        hartmann_friction=hartmann_friction,
    )
    spectrum = metrics["spectrum"]
    wavenumber = np.asarray(spectrum["wavenumber"], dtype=float)
    energy = np.asarray(spectrum["energy"], dtype=float)
    x_edges = np.linspace(0.0, lx, values.shape[0] + 1)
    y_edges = np.linspace(0.0, ly, values.shape[1] + 1)
    vmax = max(float(np.max(np.abs(values))), 1.0e-12)

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), constrained_layout=True)
    fig.suptitle(title, fontsize=16)

    image = axes[0].pcolormesh(y_edges, x_edges, values, shading="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[0].set_title("Wall-bounded Q2D field")
    axes[0].set_xlabel("y")
    axes[0].set_ylabel("x")
    fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)

    positive = (wavenumber > 0.0) & (energy > 0.0)
    axes[1].loglog(wavenumber[positive], energy[positive], marker="o", color="#1d4ed8", linewidth=1.8)
    axes[1].axvline(float(metrics["spectrum_peak_wavenumber"]), color="#b91c1c", linestyle="--", linewidth=1.0)
    axes[1].set_title("Shell energy spectrum")
    axes[1].set_xlabel("|k|")
    axes[1].set_ylabel("E(k)")
    axes[1].grid(True, which="both", alpha=0.25)
    axes[1].text(
        0.04,
        0.06,
        f"log-slope = {float(metrics['spectrum_log_slope']):.2f}",
        transform=axes[1].transAxes,
        ha="left",
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.9},
    )

    labels = ["K", "K'", "Z", "Ha diss.", "ν diss."]
    values_bar = [
        float(metrics["kinetic_energy"]),
        float(metrics["fluctuation_kinetic_energy"]),
        float(metrics["enstrophy_proxy"]),
        float(metrics["hartmann_dissipation_proxy"]),
        float(metrics["viscous_dissipation_proxy"]),
    ]
    axes[2].bar(labels, values_bar, color=["#0f766e", "#14b8a6", "#f59e0b", "#7c3aed", "#475569"])
    axes[2].set_yscale("log")
    axes[2].set_title("Energy and dissipation proxies")
    axes[2].set_ylabel("proxy magnitude")
    axes[2].tick_params(axis="x", rotation=25)
    axes[2].text(
        0.04,
        0.96,
        "not a turbulence parity claim",
        transform=axes[2].transAxes,
        ha="left",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.9},
    )

    png = out_dir / "q2d_turbulence_observables.png"
    pdf = out_dir / "q2d_turbulence_observables.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def write_q2d_decay_plots(
    case: Q2DDecayCase,
    solution: Q2DDecaySolution,
    out_dir: str | Path,
    *,
    title: str = "Q2D Hartmann-friction decay baseline",
) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    x_edges = np.linspace(0.0, case.lx, case.nx + 1)
    y_edges = np.linspace(0.0, case.ly, case.ny + 1)
    vmax = max(float(np.max(np.abs(solution.initial))), float(np.max(np.abs(solution.final))), float(np.max(np.abs(solution.analytic_final))), 1.0e-12)

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.5), constrained_layout=True)
    fig.suptitle(title, fontsize=16)

    im0 = axes[0, 0].pcolormesh(y_edges, x_edges, solution.initial, shading="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[0, 0].set_title("Initial mode")
    axes[0, 0].set_xlabel("y")
    axes[0, 0].set_ylabel("x")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].pcolormesh(y_edges, x_edges, solution.final, shading="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[0, 1].set_title("Numerical final state")
    axes[0, 1].set_xlabel("y")
    axes[0, 1].set_ylabel("x")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im2 = axes[1, 0].pcolormesh(y_edges, x_edges, solution.analytic_final, shading="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[1, 0].set_title("Analytic final state")
    axes[1, 0].set_xlabel("y")
    axes[1, 0].set_ylabel("x")
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

    axes[1, 1].plot(solution.time, solution.amplitude_numeric, color="#0f766e", label="Numerical amplitude")
    axes[1, 1].plot(solution.time, solution.amplitude_analytic, color="#b45309", linestyle="--", label="Analytic amplitude")
    axes[1, 1].set_title("Mode decay history")
    axes[1, 1].set_xlabel("t")
    axes[1, 1].set_ylabel("|u| max")
    axes[1, 1].legend(loc="upper right")

    png = out_dir / "q2d_decay_overview.png"
    pdf = out_dir / "q2d_decay_overview.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def write_q2d_forced_plots(
    case: Q2DForcedCase,
    solution: Q2DForcedSolution,
    out_dir: str | Path,
    *,
    title: str = "Q2D forced Hartmann-friction duct baseline",
) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    x_edges = np.linspace(0.0, case.lx, case.nx + 1)
    y_edges = np.linspace(0.0, case.ly, case.ny + 1)
    transient_factor = solution.amplitude_analytic[-1] / max(solution.steady_amplitude, 1.0e-12)
    analytic_final = solution.steady_analytic * transient_factor
    vmax = max(float(np.max(np.abs(solution.steady_numeric))), float(np.max(np.abs(analytic_final))), 1.0e-12)

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.6), constrained_layout=True)
    fig.suptitle(title, fontsize=16)
    im0 = axes[0].pcolormesh(y_edges, x_edges, solution.steady_numeric, shading="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[0].set_title("Numerical steady state")
    axes[0].set_xlabel("y")
    axes[0].set_ylabel("x")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].pcolormesh(y_edges, x_edges, analytic_final, shading="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[1].set_title("Analytic state at t = t_final")
    axes[1].set_xlabel("y")
    axes[1].set_ylabel("x")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].plot(solution.time, solution.amplitude_numeric, color="#0f766e", label="Numerical amplitude")
    axes[2].plot(solution.time, solution.amplitude_analytic, color="#b45309", linestyle="--", label="Analytic amplitude")
    axes[2].set_title("Forced-mode approach to steady state")
    axes[2].set_xlabel("t")
    axes[2].set_ylabel("|u| max")
    axes[2].legend(loc="lower right")

    png = out_dir / "q2d_forced_overview.png"
    pdf = out_dir / "q2d_forced_overview.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def write_q2d_wall_bounded_forced_plots(
    case: Q2DWallBoundedForcedCase,
    solution: Q2DWallBoundedForcedSolution,
    out_dir: str | Path,
    *,
    title: str = "Wall-bounded Q2D duct forced baseline",
) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    x_edges = np.linspace(0.0, case.lx, case.nx + 1)
    y_edges = np.linspace(0.0, case.ly, case.ny + 1)
    vmax = max(float(np.max(np.abs(solution.field))), float(np.max(np.abs(solution.analytic_final))), 1.0e-12)
    err_max = max(float(np.max(np.abs(solution.field - solution.analytic_final))), 1.0e-12)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), constrained_layout=True)
    fig.suptitle(title, fontsize=16)

    im0 = axes[0].pcolormesh(y_edges, x_edges, solution.field, shading="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[0].set_title("Numerical field at $t_f$")
    axes[0].set_xlabel("y")
    axes[0].set_ylabel("x")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].pcolormesh(y_edges, x_edges, solution.analytic_final, shading="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[1].set_title("Analytic field at $t_f$")
    axes[1].set_xlabel("y")
    axes[1].set_ylabel("x")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].plot(solution.time, solution.amplitude_numeric, color="#0f766e", linewidth=2.0, label="LMX")
    axes[2].plot(solution.time, solution.amplitude_analytic, color="#b45309", linestyle="--", linewidth=2.0, label="Analytic")
    axes[2].set_title("Amplitude history")
    axes[2].set_xlabel("t")
    axes[2].set_ylabel("|u| max")
    axes[2].legend(loc="lower right")
    axes[2].text(
        0.04,
        0.96,
        f"L∞ error = {err_max / vmax:.2e}",
        transform=axes[2].transAxes,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.9},
    )

    png = out_dir / "q2d_wall_bounded_overview.png"
    pdf = out_dir / "q2d_wall_bounded_overview.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]

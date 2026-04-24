from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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

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

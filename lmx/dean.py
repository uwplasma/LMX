"""Dean-flow literature correlations and reduced secondary-flow fields."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class DeanVelocityPoint:
    """One Dean-velocity correlation point."""

    dean_number: float
    axial_velocity: float
    kinematic_viscosity: float
    largest_channel_dimension: float


def bayat_rezai_dean_velocity(
    dean_number: float | np.ndarray,
    *,
    kinematic_viscosity: float,
    largest_channel_dimension: float,
) -> np.ndarray:
    """Return the Bayat-Rezai average Dean velocity correlation.

    Bayat and Rezai report ``V_De = 0.031 * (nu / s) * De**1.63`` for
    curved microchannels, where ``s`` is the largest cross-section dimension.
    The correlation was fitted to numerical data and validated experimentally
    through the low-Dean-number range used here, up to about ``De = 30``.
    """

    dean = np.asarray(dean_number, dtype=float)
    if float(kinematic_viscosity) <= 0.0:
        raise ValueError("kinematic_viscosity must be positive")
    if float(largest_channel_dimension) <= 0.0:
        raise ValueError("largest_channel_dimension must be positive")
    return 0.031 * (float(kinematic_viscosity) / float(largest_channel_dimension)) * np.maximum(dean, 0.0) ** 1.63


def bayat_rezai_lateral_reynolds(dean_number: float | np.ndarray) -> np.ndarray:
    """Return the dimensionless lateral Reynolds correlation ``Re_s,VDe``."""

    dean = np.asarray(dean_number, dtype=float)
    return 0.031 * np.maximum(dean, 0.0) ** 1.63


def dean_number_from_reynolds(reynolds_number: float | np.ndarray, curvature_ratio: float) -> np.ndarray:
    """Return ``De = Re * sqrt(radius / bend_radius)`` for curved pipes."""

    if float(curvature_ratio) < 0.0:
        raise ValueError("curvature_ratio must be nonnegative")
    return np.asarray(reynolds_number, dtype=float) * np.sqrt(float(curvature_ratio))


def dean_velocity_reference_rows(
    dean_numbers: Sequence[float] = (2.73, 6.82, 10.0, 20.0, 30.0),
    *,
    kinematic_viscosity: float = 1.0e-6,
    largest_channel_dimension: float = 150.0e-6,
    relative_tolerance: float = 0.08,
) -> list[dict[str, float | str]]:
    """Return scalar reference rows for the Bayat-Rezai Dean correlation."""

    rows: list[dict[str, float | str]] = []
    for dean in dean_numbers:
        velocity = float(
            bayat_rezai_dean_velocity(
                dean,
                kinematic_viscosity=kinematic_viscosity,
                largest_channel_dimension=largest_channel_dimension,
            )
        )
        rows.append(
            {
                "observable": f"bayat_rezai_vde_de_{dean:g}",
                "value": velocity,
                "tolerance": abs(velocity) * relative_tolerance,
                "relative_tolerance": relative_tolerance,
                "units": "m/s",
                "source": "Bayat & Rezai, Sci. Rep. 2017, Eq. 8",
                "note": "Average Dean velocity correlation for De up to about 30.",
            }
        )
    return rows


def compare_dean_velocity_points(
    points: Sequence[DeanVelocityPoint],
    *,
    relative_tolerance: float = 0.08,
) -> dict[str, object]:
    """Compare Dean point values against the Bayat-Rezai correlation."""

    rows: list[dict[str, float | bool | str]] = []
    for point in points:
        reference = float(
            bayat_rezai_dean_velocity(
                point.dean_number,
                kinematic_viscosity=point.kinematic_viscosity,
                largest_channel_dimension=point.largest_channel_dimension,
            )
        )
        lateral_re = float(bayat_rezai_lateral_reynolds(point.dean_number))
        predicted_ratio = reference / max(abs(float(point.axial_velocity)), 1.0e-30)
        tolerance = max(abs(reference) * float(relative_tolerance), 1.0e-30)
        rows.append(
            {
                "dean_number": float(point.dean_number),
                "reference_dean_velocity": reference,
                "axial_velocity": float(point.axial_velocity),
                "dean_to_axial_velocity_ratio": predicted_ratio,
                "lateral_reynolds": lateral_re,
                "tolerance": tolerance,
                "validation_pass": bool(point.dean_number <= 30.0 and reference >= 0.0),
            }
        )
    return {
        "case": "bayat_rezai_dean_velocity_correlation",
        "point_count": len(rows),
        "rows": rows,
        "max_dean_number": max((float(row["dean_number"]) for row in rows), default=0.0),
        "validation_pass": bool(rows and all(bool(row["validation_pass"]) for row in rows)),
        "literature_reference": "Bayat & Rezai, Scientific Reports 7, 13655 (2017), Eq. 8 and Eq. 9",
    }


def dean_secondary_flow_field(
    y: np.ndarray,
    z: np.ndarray,
    *,
    tube_radius: float,
    target_rms_velocity: float,
) -> dict[str, np.ndarray | float]:
    """Return a reduced two-cell Dean secondary-flow field on a cross-section.

    The shape is a divergence-free streamfunction model intended for QA,
    plotting, and reduced design studies. It is not a replacement for a
    resolved Navier-Stokes curved-pipe solve.
    """

    yy, zz = np.meshgrid(np.asarray(y, dtype=float), np.asarray(z, dtype=float), indexing="ij")
    if float(tube_radius) <= 0.0:
        raise ValueError("tube_radius must be positive")
    radius = float(tube_radius)
    eta2 = (yy / radius) ** 2 + (zz / radius) ** 2
    mask = eta2 <= 1.0
    core = np.where(mask, (1.0 - eta2) ** 2, 0.0)
    psi = (zz / radius) * core
    dy = float(np.mean(np.diff(np.asarray(y, dtype=float)))) if len(y) > 1 else 1.0
    dz = float(np.mean(np.diff(np.asarray(z, dtype=float)))) if len(z) > 1 else 1.0
    v = np.gradient(psi, dz, axis=1)
    w = -np.gradient(psi, dy, axis=0)
    v = np.where(mask, v, 0.0)
    w = np.where(mask, w, 0.0)
    rms = float(np.sqrt(np.mean((v[mask] ** 2 + w[mask] ** 2)))) if np.any(mask) else 0.0
    scale = float(target_rms_velocity) / max(rms, 1.0e-30)
    v *= scale
    w *= scale
    speed = np.sqrt(v**2 + w**2)
    return {
        "y": yy,
        "z": zz,
        "v": v,
        "w": w,
        "speed": speed,
        "mask": mask.astype(float),
        "rms_velocity": float(np.sqrt(np.mean(speed[mask] ** 2))) if np.any(mask) else 0.0,
        "peak_velocity": float(np.max(speed)) if speed.size else 0.0,
    }


def write_dean_literature_validation_plots(
    comparison: Mapping[str, object],
    output_dir: str | Path,
    *,
    output_stem: str = "dean_literature_validation",
) -> list[Path]:
    """Write a publication-facing Dean correlation and secondary-flow panel."""

    import matplotlib.pyplot as plt
    from .plotting import _save_figure_pair

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [dict(row) for row in comparison.get("rows", [])]
    dean_curve = np.linspace(0.0, 30.0, 240)
    lateral_re = bayat_rezai_lateral_reynolds(dean_curve)
    dean_points = np.asarray([float(row["dean_number"]) for row in rows], dtype=float)
    lateral_points = np.asarray([float(row["lateral_reynolds"]) for row in rows], dtype=float)
    velocity_ratios = np.asarray([float(row["dean_to_axial_velocity_ratio"]) for row in rows], dtype=float)

    y = np.linspace(-1.0, 1.0, 81)
    z = np.linspace(-1.0, 1.0, 81)
    target = float(np.max([row["reference_dean_velocity"] for row in rows])) if rows else 1.0
    field = dean_secondary_flow_field(y, z, tube_radius=1.0, target_rms_velocity=target)

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6), constrained_layout=True)
    axes[0].plot(dean_curve, lateral_re, color="#1f4e79", linewidth=2.2, label="Bayat-Rezai Eq. 9")
    axes[0].scatter(dean_points, lateral_points, color="#c2410c", zorder=3, label="validation points")
    axes[0].set_title("Dean velocity correlation")
    axes[0].set_xlabel("Dean number")
    axes[0].set_ylabel(r"$Re_{s,V_{De}}$")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].plot(dean_points, velocity_ratios, marker="o", color="#2a9d8f")
    axes[1].set_title("Secondary velocity scale")
    axes[1].set_xlabel("Dean number")
    axes[1].set_ylabel(r"$V_{De}/U$")
    axes[1].grid(True, alpha=0.25)

    speed = np.asarray(field["speed"], dtype=float)
    mask = np.asarray(field["mask"], dtype=float)
    speed = np.where(mask > 0.5, speed, np.nan)
    contour = axes[2].contourf(field["z"], field["y"], speed, levels=24, cmap="magma")
    axes[2].streamplot(
        np.asarray(z, dtype=float),
        np.asarray(y, dtype=float),
        np.asarray(field["w"], dtype=float),
        np.asarray(field["v"], dtype=float),
        color="white",
        density=1.2,
        linewidth=0.7,
        arrowsize=0.7,
    )
    axes[2].set_title("Reduced two-cell secondary flow")
    axes[2].set_xlabel("z / R")
    axes[2].set_ylabel("y / R")
    axes[2].set_aspect("equal")
    fig.colorbar(contour, ax=axes[2], fraction=0.046, pad=0.04, label="secondary speed")
    fig.suptitle("Dean-vortex literature gate for curved-pipe validation", fontsize=15, fontweight="bold")

    return _save_figure_pair(fig, out_dir, output_stem, dpi=185)

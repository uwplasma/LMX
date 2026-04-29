"""Reduced WHAM blanket liquid-metal flow previews.

This module provides a fast, geometry-following engineering model for the
approved WHAM blanket pipe route. It is intentionally explicit about its
assumptions: fixed flow rate, local transverse-field MHD drag, pipe-friction
losses, and a distributed bend-loss estimate. It is a pre-solver design lane,
not a replacement for the future full curved-pipe MHD solve.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import colors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
from PIL import Image

from .blanket_geometry import WhamBlanketLoop, build_wham_blanket_centerline, tube_surface_from_centerline
from .field_models import sample_wham_mirror_field


@dataclass(frozen=True)
class LiquidMetalProperties:
    """Liquid-metal material properties used by the reduced blanket model."""

    density: float = 9300.0
    dynamic_viscosity: float = 1.8e-3
    electrical_conductivity: float = 7.9e5
    label: str = "PbLi-like liquid metal"


@dataclass(frozen=True)
class BlanketFlowSettings:
    """Operating and modeling choices for the reduced blanket-flow preview."""

    mean_velocity: float = 0.20
    field_scale: float = 8.0
    mhd_drag_factor: float = 0.35
    bend_loss_coefficient: float = 0.35
    radial_loops: int = 12
    axial_loops: int = 4
    cross_section_points: int = 61


def solve_wham_blanket_reduced_flow(
    centerline: dict[str, np.ndarray] | None = None,
    *,
    geometry: WhamBlanketLoop | None = None,
    properties: LiquidMetalProperties | None = None,
    settings: BlanketFlowSettings | None = None,
    coil_parameters: dict[str, float | int] | None = None,
    field_sampler: Callable[..., np.ndarray] | None = None,
) -> dict[str, object]:
    """Evaluate a reduced liquid-metal flow and pressure budget on a blanket route."""

    spec = geometry or WhamBlanketLoop()
    props = properties or LiquidMetalProperties()
    opts = settings or BlanketFlowSettings()
    route = centerline or build_wham_blanket_centerline(spec)
    points = _centerline_points(route)
    station = np.asarray(route["station"], dtype=float)
    tangent = _centerline_tangent(points)
    curvature = _centerline_curvature(points, station)
    field = _sample_blanket_field(
        points,
        spec,
        opts,
        coil_parameters=coil_parameters,
        field_sampler=field_sampler,
    )
    field_parallel = np.sum(field * tangent, axis=1)
    field_perp = field - field_parallel[:, None] * tangent
    b_perp = np.linalg.norm(field_perp, axis=1)
    b_mag = np.linalg.norm(field, axis=1)

    radius = float(spec.pipe_radius)
    diameter = 2.0 * radius
    mean_velocity = float(opts.mean_velocity)
    reynolds = props.density * mean_velocity * diameter / max(props.dynamic_viscosity, 1.0e-30)
    hartmann = b_perp * radius * np.sqrt(props.electrical_conductivity / max(props.dynamic_viscosity, 1.0e-30))
    interaction = props.electrical_conductivity * b_perp**2 * radius / max(props.density * mean_velocity, 1.0e-30)

    friction_factor = _darcy_friction_factor(reynolds)
    hydraulic_gradient = friction_factor * props.density * mean_velocity**2 / max(2.0 * diameter, 1.0e-30)
    hydraulic_gradient_profile = np.full_like(station, hydraulic_gradient)
    mhd_gradient = opts.mhd_drag_factor * props.electrical_conductivity * mean_velocity * b_perp**2
    curvature_integral = _trapz(curvature, station)
    if curvature_integral > 1.0e-14:
        curvature_gradient = (
            opts.bend_loss_coefficient
            * 0.5
            * props.density
            * mean_velocity**2
            * curvature
            / curvature_integral
        )
    else:
        curvature_gradient = np.zeros_like(station)
    total_gradient = hydraulic_gradient_profile + mhd_gradient + curvature_gradient
    cumulative_pressure = _cumulative_trapezoid(total_gradient, station)

    a, b, mask = _cross_section_grid(radius, opts.cross_section_points)
    velocity_sections = np.stack(
        [
            _local_velocity_profile(
                a,
                b,
                mask,
                radius=radius,
                hartmann_number=float(ha),
                mean_velocity=mean_velocity,
            )
            for ha in hartmann
        ],
        axis=0,
    )

    return {
        "case": "wham_blanket_reduced_flow",
        "geometry": spec,
        "properties": props,
        "settings": opts,
        "centerline": route,
        "field": field,
        "station": station,
        "tangent": tangent,
        "curvature": curvature,
        "b_magnitude": b_mag,
        "b_perp": b_perp,
        "hartmann": hartmann,
        "interaction_parameter": interaction,
        "reynolds": float(reynolds),
        "friction_factor": float(friction_factor),
        "pressure_gradient_hydraulic": hydraulic_gradient_profile,
        "pressure_gradient_mhd": mhd_gradient,
        "pressure_gradient_curvature": curvature_gradient,
        "pressure_gradient_total": total_gradient,
        "cumulative_pressure_drop": cumulative_pressure,
        "pressure_drop": float(cumulative_pressure[-1]),
        "cross_section_a": a,
        "cross_section_b": b,
        "cross_section_mask": mask,
        "velocity_sections": velocity_sections,
        "metrics": _blanket_flow_metrics(
            station=station,
            b_perp=b_perp,
            hartmann=hartmann,
            interaction=interaction,
            pressure=cumulative_pressure,
            hydraulic_gradient=hydraulic_gradient_profile,
            mhd_gradient=mhd_gradient,
            curvature_gradient=curvature_gradient,
            reynolds=float(reynolds),
            friction_factor=float(friction_factor),
            mean_velocity=mean_velocity,
            radius=radius,
        ),
    }


def write_wham_blanket_flow_plots(
    flow: dict[str, object],
    out_dir: str | Path,
    *,
    filename_stem: str = "wham_blanket_flow",
) -> list[Path]:
    """Write a steady reduced-flow pressure and velocity panel."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _set_flow_plot_style()

    station = np.asarray(flow["station"], dtype=float)
    centerline = flow["centerline"]
    geometry = flow["geometry"]
    metrics = flow["metrics"]
    pressure = np.asarray(flow["cumulative_pressure_drop"], dtype=float)
    b_perp = np.asarray(flow["b_perp"], dtype=float)
    hartmann = np.asarray(flow["hartmann"], dtype=float)
    total_gradient = np.asarray(flow["pressure_gradient_total"], dtype=float)
    hydraulic_gradient = np.asarray(flow["pressure_gradient_hydraulic"], dtype=float)
    mhd_gradient = np.asarray(flow["pressure_gradient_mhd"], dtype=float)
    curvature_gradient = np.asarray(flow["pressure_gradient_curvature"], dtype=float)

    section_indices = _representative_station_indices(b_perp)
    fig = plt.figure(figsize=(13.2, 8.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.02, 1.0])
    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax_field = fig.add_subplot(gs[0, 1])
    ax_pressure = fig.add_subplot(gs[0, 2])
    section_axes = [fig.add_subplot(gs[1, i]) for i in range(3)]

    _plot_flow_route(ax3d, flow, color_values=b_perp, label=r"$B_\perp$ [T]")
    ax_field.plot(station, b_perp, color="#0f766e", label=r"$B_\perp$")
    ax_field.set_xlabel("station s [m]")
    ax_field.set_ylabel(r"$B_\perp$ [T]")
    ax_field_ha = ax_field.twinx()
    ax_field_ha.plot(station, hartmann, color="#b91c1c", linestyle="--", label="Ha")
    ax_field_ha.set_ylabel("Hartmann number")
    ax_field.set_title("WHAM field sampled along blanket route")
    lines = ax_field.get_lines() + ax_field_ha.get_lines()
    ax_field.legend(lines, [line.get_label() for line in lines], loc="upper right", fontsize=9)

    ax_pressure.plot(station, pressure / 1000.0, color="#111827", linewidth=2.2, label="total")
    ax_pressure.fill_between(station, 0.0, _cumulative_trapezoid(mhd_gradient, station) / 1000.0, color="#0f766e", alpha=0.18, label="MHD contribution")
    ax_pressure.plot(station, _cumulative_trapezoid(hydraulic_gradient, station) / 1000.0, color="#2563eb", linestyle=":", label="pipe friction")
    ax_pressure.plot(station, _cumulative_trapezoid(curvature_gradient, station) / 1000.0, color="#f97316", linestyle="--", label="bend loss")
    ax_pressure.set_xlabel("station s [m]")
    ax_pressure.set_ylabel(r"cumulative $\Delta p$ [kPa]")
    ax_pressure.set_title(f"Pressure drop = {metrics['pressure_drop_kpa']:.2f} kPa")
    ax_pressure.legend(loc="upper left", fontsize=8)
    ax_pressure_twin = ax_pressure.twinx()
    ax_pressure_twin.plot(station, total_gradient / 1000.0, color="#64748b", alpha=0.55, linewidth=1.2)
    ax_pressure_twin.set_ylabel(r"$dp/ds$ [kPa/m]")

    for axis, index, title in zip(section_axes, section_indices, ("inlet", "peak field", "outlet"), strict=True):
        _plot_velocity_section(axis, flow, index, title=title)

    fig.suptitle("WHAM blanket reduced liquid-metal flow preview", fontsize=17)
    png = out / f"{filename_stem}.png"
    pdf = out / f"{filename_stem}.pdf"
    summary = out / f"{filename_stem}_summary.json"
    csv = out / f"{filename_stem}_station_data.csv"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    summary.write_text(json.dumps(_json_summary(flow, [png.name, pdf.name]), indent=2) + "\n", encoding="utf-8")
    _write_station_csv(
        csv,
        station=station,
        b_perp=b_perp,
        hartmann=hartmann,
        pressure=pressure,
        total_gradient=total_gradient,
    )
    return [png, pdf, summary, csv]


def write_wham_blanket_flow_movie(
    flow: dict[str, object],
    out_dir: str | Path,
    *,
    filename_stem: str = "wham_blanket_flow",
    frame_count: int = 36,
    fps: int = 12,
) -> list[Path]:
    """Write a compact GIF of the reduced flow filling the blanket route."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frames = []
    count = max(int(frame_count), 3)
    station = np.asarray(flow["station"], dtype=float)
    path_length = float(station[-1])
    mean_velocity = float(flow["settings"].mean_velocity)
    transit_time = path_length / max(mean_velocity, 1.0e-12)
    time_values = np.linspace(0.0, 1.18 * transit_time, count)
    for frame_index, time_value in enumerate(time_values):
        fill = _flow_front(station, mean_velocity * time_value, width=0.055 * path_length)
        selected = int(np.clip(np.searchsorted(station, min(mean_velocity * time_value, path_length)), 0, len(station) - 1))
        image = _render_movie_frame(
            flow,
            fill_fraction=fill,
            selected_station_index=selected,
            time_value=float(time_value),
            frame_index=frame_index,
            frame_count=count,
        )
        frames.append(image)

    gif = out / f"{filename_stem}.gif"
    poster = out / f"{filename_stem}_poster.png"
    frames[-1].save(poster)
    duration_ms = max(int(1000 / max(int(fps), 1)), 20)
    frames[0].save(
        gif,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    return [gif, poster]


def write_wham_blanket_flow_summary(
    flow: dict[str, object],
    path: str | Path,
    *,
    artifacts: list[str] | None = None,
) -> Path:
    """Write a machine-readable summary for docs and regression checks."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_json_summary(flow, artifacts or []), indent=2) + "\n", encoding="utf-8")
    return out


def _sample_blanket_field(
    points: np.ndarray,
    geometry: WhamBlanketLoop,
    settings: BlanketFlowSettings,
    *,
    coil_parameters: dict[str, float | int] | None,
    field_sampler: Callable[..., np.ndarray] | None,
) -> np.ndarray:
    if field_sampler is not None:
        field = np.asarray(field_sampler(points[:, 0], points[:, 1], points[:, 2]), dtype=float)
    else:
        kwargs: dict[str, float | int] = {
            "coil_separation": geometry.coil_separation,
            "inner_radius": geometry.coil_inner_radius,
            "outer_radius": geometry.coil_outer_radius,
            "coil_axial_thickness": geometry.coil_axial_thickness,
            "radial_loops": settings.radial_loops,
            "axial_loops": settings.axial_loops,
        }
        if coil_parameters:
            kwargs.update(coil_parameters)
        field = np.asarray(
            sample_wham_mirror_field(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                **kwargs,
            ),
            dtype=float,
        )
    if field.shape != points.shape:
        raise ValueError("field_sampler must return an array with shape (station_count, 3)")
    return settings.field_scale * field


def _centerline_points(centerline: dict[str, np.ndarray]) -> np.ndarray:
    points = np.column_stack(
        [
            np.asarray(centerline["x"], dtype=float),
            np.asarray(centerline["y"], dtype=float),
            np.asarray(centerline["z"], dtype=float),
        ]
    )
    if points.shape[0] < 3:
        raise ValueError("centerline must contain at least three points")
    return points


def _centerline_tangent(points: np.ndarray) -> np.ndarray:
    tangent = np.gradient(points, axis=0)
    return tangent / np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1.0e-14)


def _centerline_curvature(points: np.ndarray, station: np.ndarray) -> np.ndarray:
    tangent = _centerline_tangent(points)
    ds = np.gradient(station)
    dt_ds = np.gradient(tangent, axis=0) / np.maximum(ds[:, None], 1.0e-14)
    curvature = np.linalg.norm(dt_ds, axis=1)
    return np.clip(curvature, 0.0, None)


def _darcy_friction_factor(reynolds: float) -> float:
    reynolds = max(float(reynolds), 1.0e-12)
    if reynolds < 2300.0:
        return 64.0 / reynolds
    return 0.3164 / reynolds**0.25


def _cross_section_grid(radius: float, points: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = max(int(points), 17)
    coordinate = np.linspace(-radius, radius, count)
    a, b = np.meshgrid(coordinate, coordinate, indexing="ij")
    mask = a**2 + b**2 <= radius**2
    return a, b, mask


def _local_velocity_profile(
    a: np.ndarray,
    b: np.ndarray,
    mask: np.ndarray,
    *,
    radius: float,
    hartmann_number: float,
    mean_velocity: float,
) -> np.ndarray:
    r = np.sqrt(a**2 + b**2)
    poiseuille = np.clip(2.0 * (1.0 - (r / radius) ** 2), 0.0, None)
    ha = max(float(hartmann_number), 1.0e-6)
    blend = ha / (ha + 60.0)
    span_b = np.sqrt(np.clip(radius**2 - a**2, 0.0, None))
    hartmann_distance = np.clip(span_b - np.abs(b), 0.0, None)
    side_distance = np.clip(radius - np.abs(a), 0.0, None)
    delta_h = radius / max(ha, 1.0)
    delta_s = radius / max(np.sqrt(ha), 1.0)
    hartmann_layer = 1.0 - np.exp(-hartmann_distance / max(delta_h, 1.0e-12))
    side_boost = 1.0 + 0.18 * blend * np.exp(-side_distance / max(delta_s, 1.0e-12)) * np.clip(
        1.0 - (b / np.maximum(span_b, 1.0e-12)) ** 2,
        0.0,
        1.0,
    )
    mhd_shape = np.clip(hartmann_layer * side_boost, 0.0, None)
    shape = (1.0 - blend) * poiseuille + blend * mhd_shape
    shape = np.where(mask, shape, np.nan)
    mean_shape = float(np.nanmean(shape))
    if not np.isfinite(mean_shape) or mean_shape <= 0.0:
        return np.where(mask, mean_velocity, np.nan)
    return mean_velocity * shape / mean_shape


def _blanket_flow_metrics(
    *,
    station: np.ndarray,
    b_perp: np.ndarray,
    hartmann: np.ndarray,
    interaction: np.ndarray,
    pressure: np.ndarray,
    hydraulic_gradient: np.ndarray,
    mhd_gradient: np.ndarray,
    curvature_gradient: np.ndarray,
    reynolds: float,
    friction_factor: float,
    mean_velocity: float,
    radius: float,
) -> dict[str, float | str]:
    hydraulic_drop = float(_trapz(hydraulic_gradient, station))
    mhd_drop = float(_trapz(mhd_gradient, station))
    curvature_drop = float(_trapz(curvature_gradient, station))
    flow_rate = mean_velocity * np.pi * radius**2
    return {
        "model_status": "reduced_fixed_flow_rate_pressure_budget",
        "mean_velocity_m_per_s": float(mean_velocity),
        "flow_rate_m3_per_s": float(flow_rate),
        "reynolds_number": float(reynolds),
        "darcy_friction_factor": float(friction_factor),
        "peak_b_perp_t": float(np.max(b_perp)),
        "mean_b_perp_t": float(np.mean(b_perp)),
        "peak_hartmann_number": float(np.max(hartmann)),
        "mean_hartmann_number": float(np.mean(hartmann)),
        "peak_interaction_parameter": float(np.max(interaction)),
        "pressure_drop_pa": float(pressure[-1]),
        "pressure_drop_kpa": float(pressure[-1] / 1000.0),
        "hydraulic_pressure_drop_pa": hydraulic_drop,
        "mhd_pressure_drop_pa": mhd_drop,
        "curvature_pressure_drop_pa": curvature_drop,
        "mhd_pressure_fraction": float(mhd_drop / max(float(pressure[-1]), 1.0e-12)),
    }


def _representative_station_indices(b_perp: np.ndarray) -> tuple[int, int, int]:
    return 0, int(np.argmax(b_perp)), int(len(b_perp) - 1)


def _plot_flow_route(ax, flow: dict[str, object], *, color_values: np.ndarray, label: str) -> None:
    centerline = flow["centerline"]
    geometry = flow["geometry"]
    x = np.asarray(centerline["x"], dtype=float)
    y = np.asarray(centerline["y"], dtype=float)
    z = np.asarray(centerline["z"], dtype=float)
    points = np.column_stack([x, y, z])
    segments = np.stack([points[:-1], points[1:]], axis=1)
    norm = colors.Normalize(vmin=float(np.min(color_values)), vmax=float(np.max(color_values)))
    collection = Line3DCollection(segments, cmap="viridis", norm=norm, linewidth=4.0)
    collection.set_array(0.5 * (color_values[:-1] + color_values[1:]))
    ax.add_collection(collection)
    tube = tube_surface_from_centerline(centerline, radius=geometry.pipe_radius)
    ax.plot_surface(
        tube["x"],
        tube["y"],
        tube["z"],
        rstride=8,
        cstride=8,
        color="#94a3b8",
        alpha=0.12,
        linewidth=0.0,
        shade=False,
    )
    _draw_simple_coils(ax, geometry)
    ax.scatter([x[0]], [y[0]], [z[0]], color="#2563eb", s=34)
    ax.scatter([x[-1]], [y[-1]], [z[-1]], color="#dc2626", s=34)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("")
    ax.set_title("Route colored by transverse field")
    ax.view_init(elev=24, azim=-42)
    _set_route_limits(ax, x, y, z, geometry)
    plt.colorbar(collection, ax=ax, shrink=0.60, pad=0.15, label=label)


def _plot_velocity_section(ax, flow: dict[str, object], index: int, *, title: str) -> None:
    a = np.asarray(flow["cross_section_a"], dtype=float)
    b = np.asarray(flow["cross_section_b"], dtype=float)
    velocity = np.asarray(flow["velocity_sections"], dtype=float)[index]
    b_perp = np.asarray(flow["b_perp"], dtype=float)[index]
    hartmann = np.asarray(flow["hartmann"], dtype=float)[index]
    masked = np.ma.masked_invalid(velocity)
    vmax = max(float(np.nanmax(np.asarray(flow["velocity_sections"], dtype=float))), 1.0e-12)
    image = ax.pcolormesh(a, b, masked, shading="auto", cmap="RdYlBu_r", vmin=0.0, vmax=vmax)
    ax.contour(a, b, np.asarray(np.isfinite(velocity), dtype=float), levels=[0.5], colors="#111827", linewidths=0.7)
    ax.set_aspect("equal")
    ax.set_xlabel("local cross-section a [m]")
    ax.set_ylabel("local cross-section b [m]")
    ax.set_title(f"{title}: B⊥={b_perp:.2f} T, Ha={hartmann:.0f}")
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="streamwise velocity [m/s]")


def _render_movie_frame(
    flow: dict[str, object],
    *,
    fill_fraction: np.ndarray,
    selected_station_index: int,
    time_value: float,
    frame_index: int,
    frame_count: int,
) -> Image.Image:
    _set_flow_plot_style()
    fig = plt.figure(figsize=(8.2, 4.8), dpi=115, constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.28, 0.95, 0.92])
    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax_section = fig.add_subplot(gs[0, 1])
    ax_pressure = fig.add_subplot(gs[0, 2])

    centerline = flow["centerline"]
    geometry = flow["geometry"]
    station = np.asarray(flow["station"], dtype=float)
    x = np.asarray(centerline["x"], dtype=float)
    y = np.asarray(centerline["y"], dtype=float)
    z = np.asarray(centerline["z"], dtype=float)
    speed = np.asarray(flow["settings"].mean_velocity * fill_fraction, dtype=float)
    points = np.column_stack([x, y, z])
    segments = np.stack([points[:-1], points[1:]], axis=1)
    collection = Line3DCollection(segments, cmap="turbo", norm=colors.Normalize(vmin=0.0, vmax=float(flow["settings"].mean_velocity)), linewidth=5.0)
    collection.set_array(0.5 * (speed[:-1] + speed[1:]))
    ax3d.add_collection(collection)
    ax3d.scatter([x[selected_station_index]], [y[selected_station_index]], [z[selected_station_index]], color="#111827", s=32)
    _draw_simple_coils(ax3d, geometry)
    ax3d.set_title("blanket flow filling route")
    ax3d.set_xlabel("x [m]")
    ax3d.set_ylabel("y [m]")
    ax3d.set_zlabel("")
    ax3d.view_init(elev=25, azim=-44)
    _set_route_limits(ax3d, x, y, z, geometry)

    section_flow = {**flow, "velocity_sections": np.asarray(flow["velocity_sections"], dtype=float) * fill_fraction[:, None, None]}
    _plot_velocity_section(ax_section, section_flow, selected_station_index, title=f"s={station[selected_station_index]:.2f} m")

    pressure = np.asarray(flow["cumulative_pressure_drop"], dtype=float)
    ax_pressure.plot(station, pressure / 1000.0, color="#111827", linewidth=2.0)
    ax_pressure.axvline(station[selected_station_index], color="#dc2626", linewidth=1.2)
    ax_pressure.fill_between(
        station,
        0.0,
        pressure / 1000.0,
        where=station <= station[selected_station_index],
        color="#0f766e",
        alpha=0.20,
    )
    ax_pressure.set_title("cumulative pressure drop")
    ax_pressure.set_xlabel("station s [m]")
    ax_pressure.set_ylabel(r"$\Delta p$ [kPa]")
    ax_pressure.text(
        0.04,
        0.95,
        f"t = {time_value:.1f} s\nframe {frame_index + 1}/{frame_count}",
        transform=ax_pressure.transAxes,
        va="top",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#cbd5e1"},
        fontsize=9,
    )

    fig.suptitle("LMX WHAM blanket reduced-flow startup preview", fontsize=13)
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
    image = Image.fromarray(rgba).convert("P", palette=Image.ADAPTIVE, colors=128)
    plt.close(fig)
    return image


def _flow_front(station: np.ndarray, front_station: float, *, width: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp((station - front_station) / max(width, 1.0e-12)))


def _json_summary(flow: dict[str, object], artifacts: list[str]) -> dict[str, object]:
    return {
        "case": "wham_blanket_reduced_flow",
        "geometry": asdict(flow["geometry"]),
        "properties": asdict(flow["properties"]),
        "settings": asdict(flow["settings"]),
        "metrics": flow["metrics"],
        "artifacts": artifacts,
        "model_equation": (
            "Delta p = integral [ f_D*rho*U^2/(2D) + C_m*sigma*U*B_perp^2 "
            "+ distributed K_b*rho*U^2/2 bend loss ] ds"
        ),
        "model_limitations": (
            "Fixed-flow-rate reduced pipe model; velocity profiles are local "
            "Hartmann-layer approximations and do not yet resolve full curved-pipe "
            "secondary flow, turbulence, heat transfer, or induced magnetic field."
        ),
    }


def _write_station_csv(
    path: Path,
    *,
    station: np.ndarray,
    b_perp: np.ndarray,
    hartmann: np.ndarray,
    pressure: np.ndarray,
    total_gradient: np.ndarray,
) -> None:
    rows = ["station_m,b_perp_t,hartmann,pressure_drop_pa,total_pressure_gradient_pa_per_m"]
    for values in zip(station, b_perp, hartmann, pressure, total_gradient, strict=True):
        rows.append(",".join(f"{float(value):.12e}" for value in values))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _draw_simple_coils(ax, geometry: WhamBlanketLoop) -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 96)
    for zc in (-0.5 * geometry.coil_separation, 0.5 * geometry.coil_separation):
        ax.plot(
            geometry.coil_outer_radius * np.cos(theta),
            geometry.coil_outer_radius * np.sin(theta),
            zc * np.ones_like(theta),
            color="#7f1d1d",
            linewidth=1.1,
            alpha=0.85,
        )
        ax.plot(
            geometry.coil_inner_radius * np.cos(theta),
            geometry.coil_inner_radius * np.sin(theta),
            zc * np.ones_like(theta),
            color="#7f1d1d",
            linewidth=0.8,
            alpha=0.75,
        )


def _set_route_limits(ax, x: np.ndarray, y: np.ndarray, z: np.ndarray, geometry: WhamBlanketLoop) -> None:
    xlim = (min(float(np.min(x)), -geometry.central_cell_radius) - 0.25, max(float(np.max(x)), geometry.central_cell_radius) + 0.25)
    ylim = (min(float(np.min(y)), -geometry.bend_radius) - 0.35, max(float(np.max(y)), geometry.bend_radius) + 0.35)
    zlim = (-0.65 * geometry.coil_separation, 0.65 * geometry.coil_separation)
    ranges = np.array([xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]])
    centers = np.array([sum(xlim) / 2.0, sum(ylim) / 2.0, sum(zlim) / 2.0])
    radius = 0.5 * float(np.max(ranges))
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)
    try:
        ax.set_box_aspect((1.0, 1.0, 0.72))
    except Exception:  # pragma: no cover - older Matplotlib.
        pass


def _cumulative_trapezoid(values: np.ndarray, coordinate: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    coordinate = np.asarray(coordinate, dtype=float)
    if values.size == 0:
        return values.copy()
    increments = 0.5 * (values[:-1] + values[1:]) * np.diff(coordinate)
    return np.concatenate([[0.0], np.cumsum(increments)])


def _trapz(values: np.ndarray, coordinate: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    return float(np.trapezoid(values, coordinate))


def _set_flow_plot_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 260,
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.linewidth": 0.9,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.5,
            "legend.frameon": True,
            "legend.framealpha": 0.94,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )

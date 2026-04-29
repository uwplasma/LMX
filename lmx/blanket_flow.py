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

import jax
import jax.numpy as jnp
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


@dataclass(frozen=True)
class BlanketTransientFlowSettings:
    """Unsteady centerline-flow settings for the WHAM blanket preview.

    This is a stationwise pressure/velocity model with turbulent pipe-friction
    closure, local MHD drag, and distributed bend losses. It is intended as the
    next solver-facing WHAM blanket gate beyond static pressure budgeting.
    """

    pressure_drive_factor: float = 1.0
    initial_velocity: float = 0.0
    time_step: float = 0.20
    final_time: float = 90.0
    axial_diffusivity: float = 5.0e-4
    incompressibility_projection: float = 0.92
    frame_count: int = 72
    steady_window: int = 18
    steady_relative_tolerance: float = 2.0e-3


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


def solve_wham_blanket_transient_flow(
    flow: dict[str, object],
    *,
    settings: BlanketTransientFlowSettings | None = None,
) -> dict[str, object]:
    """Run a stationwise transient pressure/velocity model on a WHAM blanket route.

    The model advances the local streamwise mean velocity along the curved
    centerline under a prescribed pump pressure. Losses are reconstructed from
    turbulent/laminar Darcy friction, local inductionless MHD drag, and bend
    curvature. It is still a reduced centerline solver, but it exercises an
    actual time-dependent pressure/velocity closure rather than only a static
    pressure-budget postprocess.
    """

    opts = settings or BlanketTransientFlowSettings()
    station = np.asarray(flow["station"], dtype=float)
    b_perp = np.asarray(flow["b_perp"], dtype=float)
    curvature = np.asarray(flow["curvature"], dtype=float)
    geometry = flow["geometry"]
    properties = flow["properties"]
    flow_settings = flow["settings"]
    radius = float(geometry.pipe_radius)
    diameter = 2.0 * radius
    rho = float(properties.density)
    mu = float(properties.dynamic_viscosity)
    sigma = float(properties.electrical_conductivity)
    length = max(float(station[-1] - station[0]), 1.0e-12)
    pressure_drive = float(flow["pressure_drop"]) * float(opts.pressure_drive_factor)
    drive_gradient = pressure_drive / length
    dt = max(float(opts.time_step), 1.0e-9)
    step_count = max(1, int(np.ceil(float(opts.final_time) / dt)))
    time = np.linspace(0.0, step_count * dt, step_count + 1)
    velocity = np.full_like(station, max(float(opts.initial_velocity), 0.0), dtype=float)

    mean_history = np.empty(step_count + 1, dtype=float)
    pressure_history = np.empty(step_count + 1, dtype=float)
    residual_history = np.empty(step_count + 1, dtype=float)
    courant_history = np.empty(step_count + 1, dtype=float)
    mean_history[0] = float(np.mean(velocity))
    pressure_history[0] = 0.0
    residual_history[0] = 1.0
    courant_history[0] = 0.0
    frame_indices = np.unique(np.linspace(0, step_count, max(int(opts.frame_count), 3), dtype=int))
    velocity_frames: list[np.ndarray] = []
    pressure_frames: list[np.ndarray] = []
    time_frames: list[float] = []
    if 0 in frame_indices:
        losses0 = _blanket_loss_gradients_for_velocity(
            velocity,
            station=station,
            b_perp=b_perp,
            curvature=curvature,
            radius=radius,
            density=rho,
            dynamic_viscosity=mu,
            electrical_conductivity=sigma,
            mhd_drag_factor=float(flow_settings.mhd_drag_factor),
            bend_loss_coefficient=float(flow_settings.bend_loss_coefficient),
        )
        velocity_frames.append(velocity.copy())
        pressure_frames.append(_cumulative_trapezoid(losses0["total"], station))
        time_frames.append(0.0)

    for step in range(1, step_count + 1):
        losses = _blanket_loss_gradients_for_velocity(
            velocity,
            station=station,
            b_perp=b_perp,
            curvature=curvature,
            radius=radius,
            density=rho,
            dynamic_viscosity=mu,
            electrical_conductivity=sigma,
            mhd_drag_factor=float(flow_settings.mhd_drag_factor),
            bend_loss_coefficient=float(flow_settings.bend_loss_coefficient),
        )
        d2u_ds2 = _centerline_second_derivative(velocity, station)
        mhd_linear = float(flow_settings.mhd_drag_factor) * sigma * b_perp**2 / max(rho, 1.0e-30)
        nonlinear_loss = losses["hydraulic"] + losses["curvature"]
        nonlinear_coeff = np.divide(
            nonlinear_loss,
            max(rho, 1.0e-30) * np.maximum(np.abs(velocity), 1.0e-12),
            out=np.zeros_like(velocity),
            where=np.abs(velocity) > 1.0e-12,
        )
        source = drive_gradient / max(rho, 1.0e-30) + float(opts.axial_diffusivity) * d2u_ds2
        denominator = 1.0 + dt * mhd_linear + dt * np.maximum(nonlinear_coeff, 0.0)
        next_velocity = np.maximum((velocity + dt * source) / np.maximum(denominator, 1.0e-12), 0.0)
        projection = float(np.clip(opts.incompressibility_projection, 0.0, 1.0))
        next_velocity = (1.0 - projection) * next_velocity + projection * float(np.mean(next_velocity))
        delta = float(np.max(np.abs(next_velocity - velocity)))
        velocity = next_velocity
        updated_losses = _blanket_loss_gradients_for_velocity(
            velocity,
            station=station,
            b_perp=b_perp,
            curvature=curvature,
            radius=radius,
            density=rho,
            dynamic_viscosity=mu,
            electrical_conductivity=sigma,
            mhd_drag_factor=float(flow_settings.mhd_drag_factor),
            bend_loss_coefficient=float(flow_settings.bend_loss_coefficient),
        )
        pressure_curve = _cumulative_trapezoid(updated_losses["total"], station)
        mean_velocity = float(np.mean(velocity))
        mean_history[step] = mean_velocity
        pressure_history[step] = float(pressure_curve[-1])
        residual_history[step] = delta / max(abs(mean_velocity), 1.0e-12)
        courant_history[step] = float(np.max(velocity) * dt / max(float(np.min(np.diff(station))), 1.0e-12))
        if step in frame_indices:
            velocity_frames.append(velocity.copy())
            pressure_frames.append(pressure_curve)
            time_frames.append(float(time[step]))

    final_losses = _blanket_loss_gradients_for_velocity(
        velocity,
        station=station,
        b_perp=b_perp,
        curvature=curvature,
        radius=radius,
        density=rho,
        dynamic_viscosity=mu,
        electrical_conductivity=sigma,
        mhd_drag_factor=float(flow_settings.mhd_drag_factor),
        bend_loss_coefficient=float(flow_settings.bend_loss_coefficient),
    )
    window = min(max(int(opts.steady_window), 2), residual_history.size)
    steady_residual = float(np.max(residual_history[-window:]))
    final_pressure = float(pressure_history[-1])
    return {
        "case": "wham_blanket_centerline_transient_pressure_velocity",
        "base_flow": flow,
        "settings": opts,
        "time": time,
        "station": station,
        "velocity_mean_history": mean_history,
        "pressure_drop_history": pressure_history,
        "relative_update_history": residual_history,
        "courant_history": courant_history,
        "frame_time": np.asarray(time_frames, dtype=float),
        "velocity_frames": np.asarray(velocity_frames, dtype=float),
        "pressure_frames": np.asarray(pressure_frames, dtype=float),
        "final_velocity": velocity,
        "final_pressure_curve": _cumulative_trapezoid(final_losses["total"], station),
        "final_loss_gradients": final_losses,
        "pressure_drive_pa": pressure_drive,
        "drive_gradient_pa_per_m": drive_gradient,
        "metrics": {
            "model_status": "centerline_transient_pressure_velocity_turbulent_closure",
            "final_mean_velocity_m_per_s": float(np.mean(velocity)),
            "target_static_mean_velocity_m_per_s": float(flow_settings.mean_velocity),
            "final_peak_velocity_m_per_s": float(np.max(velocity)),
            "final_min_velocity_m_per_s": float(np.min(velocity)),
            "final_pressure_drop_kpa": final_pressure / 1000.0,
            "pressure_drive_kpa": pressure_drive / 1000.0,
            "steady_relative_update": steady_residual,
            "steady_state_reached": bool(steady_residual <= float(opts.steady_relative_tolerance)),
            "max_courant": float(np.max(courant_history)),
            "max_pseudo_courant": float(np.max(courant_history)),
            "simulated_time_s": float(time[-1]),
            "frame_count": int(len(time_frames)),
            "turbulent_closure": "Darcy friction uses laminar 64/Re below Re=2300 and Blasius 0.3164/Re^0.25 above.",
        },
    }


def blanket_pressure_budget_from_transverse_field(
    station: jnp.ndarray,
    b_perp: jnp.ndarray,
    curvature: jnp.ndarray,
    *,
    pipe_radius: float | jnp.ndarray,
    mean_velocity: float | jnp.ndarray,
    density: float | jnp.ndarray,
    dynamic_viscosity: float | jnp.ndarray,
    electrical_conductivity: float | jnp.ndarray,
    mhd_drag_factor: float | jnp.ndarray = 0.35,
    bend_loss_coefficient: float | jnp.ndarray = 0.35,
) -> dict[str, jnp.ndarray]:
    """Differentiable fixed-flow pressure budget from a transverse-field trace."""

    station = jnp.asarray(station, dtype=jnp.float32)
    b_perp = jnp.asarray(b_perp, dtype=jnp.float32)
    curvature = jnp.asarray(curvature, dtype=jnp.float32)
    radius = jnp.asarray(pipe_radius, dtype=jnp.float32)
    velocity = jnp.asarray(mean_velocity, dtype=jnp.float32)
    rho = jnp.asarray(density, dtype=jnp.float32)
    mu = jnp.asarray(dynamic_viscosity, dtype=jnp.float32)
    sigma = jnp.asarray(electrical_conductivity, dtype=jnp.float32)
    diameter = 2.0 * radius
    reynolds = rho * velocity * diameter / jnp.maximum(mu, 1.0e-30)
    friction_factor = _darcy_friction_factor_jax(reynolds)
    hydraulic_gradient = friction_factor * rho * velocity**2 / jnp.maximum(2.0 * diameter, 1.0e-30)
    hydraulic_gradient_profile = jnp.full_like(station, hydraulic_gradient)
    mhd_gradient = jnp.asarray(mhd_drag_factor, dtype=jnp.float32) * sigma * velocity * b_perp**2
    curvature_integral = _trapz_jax(curvature, station)
    curvature_gradient = jnp.where(
        curvature_integral > 1.0e-14,
        jnp.asarray(bend_loss_coefficient, dtype=jnp.float32)
        * 0.5
        * rho
        * velocity**2
        * curvature
        / jnp.maximum(curvature_integral, 1.0e-30),
        jnp.zeros_like(station),
    )
    total_gradient = hydraulic_gradient_profile + mhd_gradient + curvature_gradient
    cumulative_pressure = _cumulative_trapezoid_jax(total_gradient, station)
    hartmann = b_perp * radius * jnp.sqrt(sigma / jnp.maximum(mu, 1.0e-30))
    interaction = sigma * b_perp**2 * radius / jnp.maximum(rho * velocity, 1.0e-30)
    return {
        "station": station,
        "b_perp": b_perp,
        "hartmann": hartmann,
        "interaction_parameter": interaction,
        "pressure_gradient_hydraulic": hydraulic_gradient_profile,
        "pressure_gradient_mhd": mhd_gradient,
        "pressure_gradient_curvature": curvature_gradient,
        "pressure_gradient_total": total_gradient,
        "cumulative_pressure_drop": cumulative_pressure,
        "pressure_drop": cumulative_pressure[-1],
        "hydraulic_pressure_drop": _trapz_jax(hydraulic_gradient_profile, station),
        "mhd_pressure_drop": _trapz_jax(mhd_gradient, station),
        "curvature_pressure_drop": _trapz_jax(curvature_gradient, station),
        "reynolds_number": reynolds,
        "friction_factor": friction_factor,
    }


def wham_blanket_pressure_drop_history(
    centerline: dict[str, np.ndarray] | None = None,
    *,
    geometry: WhamBlanketLoop | None = None,
    properties: LiquidMetalProperties | None = None,
    settings: BlanketFlowSettings | None = None,
    coil_parameters: dict[str, float | int] | None = None,
    coil_separation: float | jnp.ndarray | None = None,
    field_scale: float | jnp.ndarray | None = None,
    mean_velocity: float | jnp.ndarray | None = None,
) -> dict[str, jnp.ndarray]:
    """Differentiable WHAM blanket pressure history for sensitivity studies."""

    spec = geometry or WhamBlanketLoop()
    props = properties or LiquidMetalProperties()
    opts = settings or BlanketFlowSettings()
    params = dict(coil_parameters or {})
    route = centerline or build_wham_blanket_centerline(spec)
    points_np = _centerline_points(route)
    station = jnp.asarray(route["station"], dtype=jnp.float32)
    points = jnp.asarray(points_np, dtype=jnp.float32)
    tangent = jnp.asarray(_centerline_tangent(points_np), dtype=jnp.float32)
    curvature = jnp.asarray(_centerline_curvature(points_np, np.asarray(route["station"], dtype=float)), dtype=jnp.float32)
    separation = jnp.asarray(
        spec.coil_separation if coil_separation is None else coil_separation,
        dtype=jnp.float32,
    )
    field_multiplier = jnp.asarray(opts.field_scale if field_scale is None else field_scale, dtype=jnp.float32)
    velocity = jnp.asarray(opts.mean_velocity if mean_velocity is None else mean_velocity, dtype=jnp.float32)
    field = field_multiplier * sample_wham_mirror_field(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        coil_separation=separation,
        current_scale=float(params.get("current_scale", 100323.62459546926)),
        inner_radius=float(params.get("inner_radius", spec.coil_inner_radius)),
        outer_radius=float(params.get("outer_radius", spec.coil_outer_radius)),
        coil_axial_thickness=float(params.get("coil_axial_thickness", spec.coil_axial_thickness)),
        radial_loops=int(params.get("radial_loops", opts.radial_loops)),
        axial_loops=int(params.get("axial_loops", opts.axial_loops)),
    )
    field_parallel = jnp.sum(field * tangent, axis=1)
    field_perp = field - field_parallel[:, None] * tangent
    b_perp = jnp.linalg.norm(field_perp, axis=1)
    budget = blanket_pressure_budget_from_transverse_field(
        station,
        b_perp,
        curvature,
        pipe_radius=spec.pipe_radius,
        mean_velocity=velocity,
        density=props.density,
        dynamic_viscosity=props.dynamic_viscosity,
        electrical_conductivity=props.electrical_conductivity,
        mhd_drag_factor=opts.mhd_drag_factor,
        bend_loss_coefficient=opts.bend_loss_coefficient,
    )
    return {
        **budget,
        "field": field,
        "b_magnitude": jnp.linalg.norm(field, axis=1),
        "tangent": tangent,
        "curvature": curvature,
        "coil_separation": separation,
        "field_scale": field_multiplier,
        "mean_velocity": velocity,
    }


def wham_blanket_pressure_drop_sensitivity(
    centerline: dict[str, np.ndarray] | None = None,
    *,
    geometry: WhamBlanketLoop | None = None,
    properties: LiquidMetalProperties | None = None,
    settings: BlanketFlowSettings | None = None,
    coil_parameters: dict[str, float | int] | None = None,
    coil_separation: float | None = None,
    field_scale: float | None = None,
    mean_velocity: float | None = None,
) -> dict[str, jnp.ndarray]:
    """Return autodiff sensitivities of WHAM blanket pressure drop."""

    spec = geometry or WhamBlanketLoop()
    opts = settings or BlanketFlowSettings()
    reference_separation = spec.coil_separation if coil_separation is None else float(coil_separation)
    reference_field_scale = opts.field_scale if field_scale is None else float(field_scale)
    reference_velocity = opts.mean_velocity if mean_velocity is None else float(mean_velocity)

    def objective(separation_value, field_scale_value, velocity_value):
        return wham_blanket_pressure_drop_history(
            centerline,
            geometry=geometry,
            properties=properties,
            settings=opts,
            coil_parameters=coil_parameters,
            coil_separation=separation_value,
            field_scale=field_scale_value,
            mean_velocity=velocity_value,
        )["pressure_drop"]

    pressure_drop = objective(reference_separation, reference_field_scale, reference_velocity)
    gradients = jax.grad(objective, argnums=(0, 1, 2))(reference_separation, reference_field_scale, reference_velocity)
    history = wham_blanket_pressure_drop_history(
        centerline,
        geometry=geometry,
        properties=properties,
        settings=opts,
        coil_parameters=coil_parameters,
        coil_separation=reference_separation,
        field_scale=reference_field_scale,
        mean_velocity=reference_velocity,
    )
    safe_pressure = jnp.maximum(jnp.abs(pressure_drop), 1.0e-30)
    return {
        **history,
        "pressure_drop": pressure_drop,
        "pressure_drop_kpa": pressure_drop / 1000.0,
        "d_pressure_drop_d_coil_separation": gradients[0],
        "d_pressure_drop_d_field_scale": gradients[1],
        "d_pressure_drop_d_mean_velocity": gradients[2],
        "elasticity_coil_separation": gradients[0] * jnp.asarray(reference_separation, dtype=jnp.float32) / safe_pressure,
        "elasticity_field_scale": gradients[1] * jnp.asarray(reference_field_scale, dtype=jnp.float32) / safe_pressure,
        "elasticity_mean_velocity": gradients[2] * jnp.asarray(reference_velocity, dtype=jnp.float32) / safe_pressure,
    }


def write_wham_blanket_autodiff_research_plots(
    study: dict[str, object],
    out_dir: str | Path,
    *,
    filename_stem: str = "wham_blanket_autodiff_research",
) -> list[Path]:
    """Write pressure-drop sensitivity and inverse-design panels."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _set_flow_plot_style()
    reference = study["reference"]
    separation_sweep = np.asarray(study["separation_sweep"], dtype=float)
    separation_pressure = np.asarray(study["separation_pressure_drop_kpa"], dtype=float)
    field_history = list(study["field_scale_design_history"])
    station = np.asarray(reference["station"], dtype=float)
    pressure = np.asarray(reference["cumulative_pressure_drop"], dtype=float) / 1000.0
    hydraulic = _cumulative_trapezoid(np.asarray(reference["pressure_gradient_hydraulic"], dtype=float), station) / 1000.0
    mhd = _cumulative_trapezoid(np.asarray(reference["pressure_gradient_mhd"], dtype=float), station) / 1000.0
    curvature = _cumulative_trapezoid(np.asarray(reference["pressure_gradient_curvature"], dtype=float), station) / 1000.0
    elasticities = [
        float(reference["elasticity_coil_separation"]),
        float(reference["elasticity_field_scale"]),
        float(reference["elasticity_mean_velocity"]),
    ]
    labels = ["coil separation", "field scale", "mean velocity"]

    fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.4), constrained_layout=True)
    axes[0, 0].plot(station, pressure, color="#111827", linewidth=2.2, label="total")
    axes[0, 0].plot(station, hydraulic, color="#2563eb", linestyle=":", label="pipe friction")
    axes[0, 0].plot(station, mhd, color="#0f766e", linestyle="-.", label="MHD")
    axes[0, 0].plot(station, curvature, color="#f97316", linestyle="--", label="bend")
    axes[0, 0].set_title("Differentiable pressure budget")
    axes[0, 0].set_xlabel("station s [m]")
    axes[0, 0].set_ylabel(r"cumulative $\Delta p$ [kPa]")
    axes[0, 0].legend(loc="upper left", fontsize=8)

    axes[0, 1].plot(separation_sweep, separation_pressure, marker="o", color="#0f766e", label="evaluated")
    reference_separation = float(reference["coil_separation"])
    reference_pressure = float(reference["pressure_drop"] / 1000.0)
    slope = float(reference["d_pressure_drop_d_coil_separation"] / 1000.0)
    tangent = reference_pressure + slope * (separation_sweep - reference_separation)
    axes[0, 1].plot(separation_sweep, tangent, color="#7c3aed", linestyle="--", label="autodiff tangent")
    axes[0, 1].axvline(reference_separation, color="#111827", linestyle=":", linewidth=1.0)
    axes[0, 1].set_title("Local coil-spacing pressure sensitivity")
    axes[0, 1].set_xlabel("coil separation [m]")
    axes[0, 1].set_ylabel(r"$\Delta p$ [kPa]")
    axes[0, 1].legend(loc="best", fontsize=8)

    colors_bar = ["#2563eb" if value < 0 else "#b91c1c" for value in elasticities]
    axes[1, 0].bar(labels, elasticities, color=colors_bar, alpha=0.88)
    axes[1, 0].axhline(0.0, color="#111827", linewidth=0.9)
    axes[1, 0].set_title("Autodiff pressure-drop elasticities")
    axes[1, 0].set_ylabel(r"$d\log(\Delta p)/d\log(q)$")
    axes[1, 0].tick_params(axis="x", rotation=18)

    steps = np.asarray([item["step"] for item in field_history], dtype=float)
    field_scale = np.asarray([item["field_scale"] for item in field_history], dtype=float)
    pressure_history = np.asarray([item["pressure_drop_kpa"] for item in field_history], dtype=float)
    target = float(study["target_pressure_drop_kpa"])
    axes[1, 1].plot(steps, pressure_history, marker="o", color="#0f766e", label=r"$\Delta p$")
    axes[1, 1].axhline(target, color="#b91c1c", linestyle="--", label="target")
    ax_scale = axes[1, 1].twinx()
    ax_scale.plot(steps, field_scale, marker="s", color="#7c3aed", label="field scale")
    axes[1, 1].set_title("Inverse design: field multiplier for target pressure")
    axes[1, 1].set_xlabel("Newton update")
    axes[1, 1].set_ylabel(r"$\Delta p$ [kPa]", color="#0f766e")
    ax_scale.set_ylabel("field multiplier", color="#7c3aed")
    axes[1, 1].tick_params(axis="y", labelcolor="#0f766e")
    ax_scale.tick_params(axis="y", labelcolor="#7c3aed")
    handles = axes[1, 1].get_lines() + ax_scale.get_lines()
    axes[1, 1].legend(handles, [line.get_label() for line in handles], loc="best", fontsize=8)

    fig.suptitle("LMX WHAM blanket differentiable pressure-drop research study", fontsize=17)
    png = out / f"{filename_stem}.png"
    pdf = out / f"{filename_stem}.pdf"
    summary = out / f"{filename_stem}_summary.json"
    station_csv = out / f"{filename_stem}_station_data.csv"
    design_csv = out / f"{filename_stem}_design_data.csv"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    summary.write_text(json.dumps(_json_ready_study(study, [png.name]), indent=2) + "\n", encoding="utf-8")
    _write_autodiff_station_csv(
        station_csv,
        station=station,
        total_pressure=pressure,
        hydraulic_pressure=hydraulic,
        mhd_pressure=mhd,
        curvature_pressure=curvature,
    )
    _write_autodiff_design_csv(
        design_csv,
        separation_sweep=separation_sweep,
        separation_pressure=separation_pressure,
        field_history=field_history,
    )
    return [png, pdf, summary, station_csv, design_csv]


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


def write_wham_blanket_transient_flow_plots(
    transient: dict[str, object],
    out_dir: str | Path,
    *,
    filename_stem: str = "wham_blanket_transient_flow",
) -> list[Path]:
    """Write transient pressure/velocity diagnostics for the blanket route."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _set_flow_plot_style()
    base_flow = transient["base_flow"]
    station = np.asarray(transient["station"], dtype=float)
    time = np.asarray(transient["time"], dtype=float)
    mean_velocity = np.asarray(transient["velocity_mean_history"], dtype=float)
    pressure_drop = np.asarray(transient["pressure_drop_history"], dtype=float) / 1000.0
    residual = np.asarray(transient["relative_update_history"], dtype=float)
    final_velocity = np.asarray(transient["final_velocity"], dtype=float)
    final_pressure = np.asarray(transient["final_pressure_curve"], dtype=float) / 1000.0
    target_velocity = float(base_flow["settings"].mean_velocity)
    pressure_reference = float(base_flow["pressure_drop"]) / 1000.0

    fig, axes = plt.subplots(2, 2, figsize=(13.4, 8.2), constrained_layout=True)
    axes[0, 0].plot(time, mean_velocity, color="#0f766e", linewidth=2.2, label="mean")
    axes[0, 0].axhline(target_velocity, color="#111827", linestyle="--", label="static target")
    axes[0, 0].set_title("Centerline transient velocity")
    axes[0, 0].set_xlabel("time [s]")
    axes[0, 0].set_ylabel("mean velocity [m/s]")
    axes[0, 0].legend(loc="best")

    axes[0, 1].plot(time, pressure_drop, color="#b91c1c", linewidth=2.2, label="loss")
    axes[0, 1].axhline(pressure_reference, color="#111827", linestyle="--", label="steady budget")
    axes[0, 1].set_title("Pressure drop approaches steady budget")
    axes[0, 1].set_xlabel("time [s]")
    axes[0, 1].set_ylabel(r"$\Delta p$ [kPa]")
    axes[0, 1].legend(loc="best")

    axes[1, 0].plot(station, final_velocity, color="#0f766e", linewidth=2.2)
    axes[1, 0].axhline(float(np.mean(final_velocity)), color="#111827", linestyle=":")
    axes[1, 0].set_title("Steady stationwise velocity")
    axes[1, 0].set_xlabel("station s [m]")
    axes[1, 0].set_ylabel("velocity [m/s]")

    axes[1, 1].semilogy(time[1:], np.maximum(residual[1:], 1.0e-16), color="#7c3aed", linewidth=2.2)
    axes[1, 1].axhline(float(transient["settings"].steady_relative_tolerance), color="#111827", linestyle="--")
    axes[1, 1].set_title("Steady-state update residual")
    axes[1, 1].set_xlabel("time [s]")
    axes[1, 1].set_ylabel("relative update")
    ax_pressure = axes[1, 1].twinx()
    ax_pressure.plot(station, final_pressure, color="#64748b", alpha=0.45, linewidth=1.2)
    ax_pressure.set_ylabel("final cumulative pressure [kPa]", color="#64748b")
    ax_pressure.tick_params(axis="y", labelcolor="#64748b")

    metrics = transient["metrics"]
    fig.suptitle(
        "WHAM blanket centerline pressure-velocity transient "
        f"(steady={metrics['steady_state_reached']}, t={metrics['simulated_time_s']:.0f} s)",
        fontsize=16,
    )
    png = out / f"{filename_stem}.png"
    pdf = out / f"{filename_stem}.pdf"
    summary = out / f"{filename_stem}_summary.json"
    csv = out / f"{filename_stem}_history.csv"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    summary.write_text(json.dumps(_json_transient_summary(transient, [png.name, pdf.name, csv.name]), indent=2) + "\n", encoding="utf-8")
    _write_transient_history_csv(csv, transient)
    return [png, pdf, summary, csv]


def write_wham_blanket_transient_flow_movie(
    transient: dict[str, object],
    out_dir: str | Path,
    *,
    filename_stem: str = "wham_blanket_flow",
    fps: int = 12,
) -> list[Path]:
    """Write a longer GIF showing filling, acceleration, and steady-state flow."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frames = []
    frame_time = np.asarray(transient["frame_time"], dtype=float)
    velocity_frames = np.asarray(transient["velocity_frames"], dtype=float)
    pressure_frames = np.asarray(transient["pressure_frames"], dtype=float)
    for frame_index, time_value in enumerate(frame_time):
        image = _render_transient_movie_frame(
            transient,
            velocity=velocity_frames[frame_index],
            pressure=pressure_frames[frame_index],
            time_value=float(time_value),
            frame_index=frame_index,
            frame_count=len(frame_time),
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


def _darcy_friction_factor_array(reynolds: np.ndarray) -> np.ndarray:
    re = np.maximum(np.asarray(reynolds, dtype=float), 1.0e-12)
    return np.where(re < 2300.0, 64.0 / re, 0.3164 / re**0.25)


def _blanket_loss_gradients_for_velocity(
    velocity: np.ndarray,
    *,
    station: np.ndarray,
    b_perp: np.ndarray,
    curvature: np.ndarray,
    radius: float,
    density: float,
    dynamic_viscosity: float,
    electrical_conductivity: float,
    mhd_drag_factor: float,
    bend_loss_coefficient: float,
) -> dict[str, np.ndarray]:
    u = np.asarray(velocity, dtype=float)
    diameter = 2.0 * float(radius)
    abs_u = np.abs(u)
    reynolds = density * abs_u * diameter / max(dynamic_viscosity, 1.0e-30)
    friction = _darcy_friction_factor_array(reynolds)
    hydraulic = friction * density * u * abs_u / max(2.0 * diameter, 1.0e-30)
    mhd = mhd_drag_factor * electrical_conductivity * u * np.asarray(b_perp, dtype=float) ** 2
    curvature_integral = _trapz(np.asarray(curvature, dtype=float), np.asarray(station, dtype=float))
    if curvature_integral > 1.0e-14:
        bend = (
            bend_loss_coefficient
            * 0.5
            * density
            * u
            * abs_u
            * np.asarray(curvature, dtype=float)
            / curvature_integral
        )
    else:
        bend = np.zeros_like(u)
    return {
        "hydraulic": hydraulic,
        "mhd": mhd,
        "curvature": bend,
        "total": hydraulic + mhd + bend,
        "reynolds": reynolds,
        "friction_factor": friction,
    }


def _centerline_second_derivative(values: np.ndarray, coordinate: np.ndarray) -> np.ndarray:
    if values.size < 3:
        return np.zeros_like(values)
    first = np.gradient(values, coordinate, edge_order=1)
    return np.gradient(first, coordinate, edge_order=1)


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


def _render_transient_movie_frame(
    transient: dict[str, object],
    *,
    velocity: np.ndarray,
    pressure: np.ndarray,
    time_value: float,
    frame_index: int,
    frame_count: int,
) -> Image.Image:
    _set_flow_plot_style()
    base_flow = transient["base_flow"]
    fig = plt.figure(figsize=(8.4, 5.0), dpi=115, constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.24, 0.94, 0.94])
    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax_section = fig.add_subplot(gs[0, 1])
    ax_history = fig.add_subplot(gs[0, 2])

    centerline = base_flow["centerline"]
    geometry = base_flow["geometry"]
    station = np.asarray(transient["station"], dtype=float)
    x = np.asarray(centerline["x"], dtype=float)
    y = np.asarray(centerline["y"], dtype=float)
    z = np.asarray(centerline["z"], dtype=float)
    points = np.column_stack([x, y, z])
    segments = np.stack([points[:-1], points[1:]], axis=1)
    vmax = max(float(np.nanmax(np.asarray(transient["velocity_frames"], dtype=float))), 1.0e-12)
    collection = Line3DCollection(segments, cmap="turbo", norm=colors.Normalize(vmin=0.0, vmax=vmax), linewidth=5.0)
    collection.set_array(0.5 * (velocity[:-1] + velocity[1:]))
    ax3d.add_collection(collection)
    _draw_simple_coils(ax3d, geometry)
    selected = _transient_selected_station_index(velocity, station)
    ax3d.scatter([x[selected]], [y[selected]], [z[selected]], color="#111827", s=34)
    ax3d.set_title("curved-pipe velocity")
    ax3d.set_xlabel("x [m]")
    ax3d.set_ylabel("y [m]")
    ax3d.set_zlabel("")
    ax3d.view_init(elev=25, azim=-44)
    _set_route_limits(ax3d, x, y, z, geometry)

    scale = velocity / max(float(base_flow["settings"].mean_velocity), 1.0e-12)
    section_flow = {**base_flow, "velocity_sections": np.asarray(base_flow["velocity_sections"], dtype=float) * scale[:, None, None]}
    _plot_velocity_section(ax_section, section_flow, selected, title=f"s={station[selected]:.2f} m")

    time = np.asarray(transient["time"], dtype=float)
    mean_history = np.asarray(transient["velocity_mean_history"], dtype=float)
    pressure_history = np.asarray(transient["pressure_drop_history"], dtype=float) / 1000.0
    history_index = int(np.clip(np.searchsorted(time, time_value), 0, len(time) - 1))
    ax_history.plot(time, mean_history, color="#0f766e", linewidth=2.0, label="mean U")
    ax_history.axvline(time_value, color="#111827", linewidth=1.0)
    ax_history.set_xlabel("time [s]")
    ax_history.set_ylabel("mean U [m/s]", color="#0f766e")
    ax_history.tick_params(axis="y", labelcolor="#0f766e")
    ax_pressure = ax_history.twinx()
    ax_pressure.plot(time, pressure_history, color="#b91c1c", linewidth=1.7, label=r"$\Delta p$")
    ax_pressure.set_ylabel(r"$\Delta p$ [kPa]", color="#b91c1c")
    ax_pressure.tick_params(axis="y", labelcolor="#b91c1c")
    ax_history.set_title("approach to steady state")
    ax_history.text(
        0.04,
        0.94,
        f"t = {time_value:.1f} s\n"
        f"Umean = {mean_history[history_index]:.3f} m/s\n"
        f"dp = {pressure_history[history_index]:.2f} kPa\n"
        f"frame {frame_index + 1}/{frame_count}",
        transform=ax_history.transAxes,
        va="top",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#cbd5e1"},
        fontsize=8,
    )

    fig.suptitle("LMX WHAM blanket centerline pressure-velocity transient", fontsize=13)
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
    image = Image.fromarray(rgba).convert("P", palette=Image.ADAPTIVE, colors=128)
    plt.close(fig)
    return image


def _transient_selected_station_index(velocity: np.ndarray, station: np.ndarray) -> int:
    active = np.where(velocity >= 0.5 * max(float(np.max(velocity)), 1.0e-12))[0]
    if active.size == 0:
        return 0
    if active[-1] < len(station) - 2:
        return int(active[-1])
    return int(np.argmax(velocity))


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


def _json_transient_summary(transient: dict[str, object], artifacts: list[str]) -> dict[str, object]:
    base_flow = transient["base_flow"]
    return {
        "case": "wham_blanket_centerline_transient_pressure_velocity",
        "base_case": base_flow["case"],
        "geometry": asdict(base_flow["geometry"]),
        "properties": asdict(base_flow["properties"]),
        "steady_settings": asdict(base_flow["settings"]),
        "transient_settings": asdict(transient["settings"]),
        "metrics": transient["metrics"],
        "pressure_drive_pa": float(transient["pressure_drive_pa"]),
        "drive_gradient_pa_per_m": float(transient["drive_gradient_pa_per_m"]),
        "artifacts": artifacts,
        "model_equation": (
            "rho dU/dt = dp_drive/ds - [f_D*rho*U|U|/(2D) + "
            "C_m*sigma*U*B_perp^2 + distributed K_b*rho*U|U|/2 bend loss] "
            "+ rho*nu_s*d2U/ds2"
        ),
        "model_limitations": (
            "Unsteady centerline pressure/velocity closure with turbulent pipe "
            "friction and local MHD drag. It advances velocity and pressure in "
            "time on the curved route, but it does not resolve 3D secondary "
            "flows, cross-section turbulence, heat transfer, or induced magnetic field."
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


def _write_transient_history_csv(path: Path, transient: dict[str, object]) -> None:
    rows = ["time_s,mean_velocity_m_per_s,pressure_drop_pa,relative_update,courant"]
    for values in zip(
        np.asarray(transient["time"], dtype=float),
        np.asarray(transient["velocity_mean_history"], dtype=float),
        np.asarray(transient["pressure_drop_history"], dtype=float),
        np.asarray(transient["relative_update_history"], dtype=float),
        np.asarray(transient["courant_history"], dtype=float),
        strict=True,
    ):
        rows.append(",".join(f"{float(value):.12e}" for value in values))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_autodiff_station_csv(
    path: Path,
    *,
    station: np.ndarray,
    total_pressure: np.ndarray,
    hydraulic_pressure: np.ndarray,
    mhd_pressure: np.ndarray,
    curvature_pressure: np.ndarray,
) -> None:
    rows = ["station_m,total_pressure_kpa,hydraulic_pressure_kpa,mhd_pressure_kpa,curvature_pressure_kpa"]
    for values in zip(station, total_pressure, hydraulic_pressure, mhd_pressure, curvature_pressure, strict=True):
        rows.append(",".join(f"{float(value):.12e}" for value in values))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_autodiff_design_csv(
    path: Path,
    *,
    separation_sweep: np.ndarray,
    separation_pressure: np.ndarray,
    field_history: list[dict[str, object]],
) -> None:
    rows = ["kind,index,input_value,pressure_drop_kpa,gradient_kpa"]
    for index, (separation, pressure) in enumerate(zip(separation_sweep, separation_pressure, strict=True)):
        rows.append(f"separation,{index},{float(separation):.12e},{float(pressure):.12e},nan")
    for item in field_history:
        rows.append(
            "field_scale,"
            f"{int(item['step'])},"
            f"{float(item['field_scale']):.12e},"
            f"{float(item['pressure_drop_kpa']):.12e},"
            f"{float(item.get('d_pressure_drop_d_field_scale_kpa', np.nan)):.12e}"
        )
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


def _cumulative_trapezoid_jax(values: jnp.ndarray, coordinate: jnp.ndarray) -> jnp.ndarray:
    values = jnp.asarray(values)
    coordinate = jnp.asarray(coordinate)
    increments = 0.5 * (values[:-1] + values[1:]) * jnp.diff(coordinate)
    return jnp.concatenate([jnp.zeros((1,), dtype=values.dtype), jnp.cumsum(increments)])


def _trapz_jax(values: jnp.ndarray, coordinate: jnp.ndarray) -> jnp.ndarray:
    values = jnp.asarray(values)
    coordinate = jnp.asarray(coordinate)
    return jnp.where(
        values.size < 2,
        jnp.asarray(0.0, dtype=values.dtype),
        jnp.sum(0.5 * (values[:-1] + values[1:]) * jnp.diff(coordinate)),
    )


def _darcy_friction_factor_jax(reynolds: jnp.ndarray) -> jnp.ndarray:
    re = jnp.maximum(jnp.asarray(reynolds), 1.0e-12)
    return jnp.where(re < 2300.0, 64.0 / re, 0.3164 / re**0.25)


def _json_ready_study(study: dict[str, object], artifacts: list[str]) -> dict[str, object]:
    reference = study["reference"]
    return {
        "case": study.get("case", "wham_blanket_autodiff_research"),
        "research_questions": _json_ready(study.get("research_questions", [])),
        "reference": {
            "coil_separation_m": float(reference["coil_separation"]),
            "field_scale": float(reference["field_scale"]),
            "mean_velocity_m_per_s": float(reference["mean_velocity"]),
            "pressure_drop_kpa": float(reference["pressure_drop"] / 1000.0),
            "hydraulic_pressure_drop_kpa": float(reference["hydraulic_pressure_drop"] / 1000.0),
            "mhd_pressure_drop_kpa": float(reference["mhd_pressure_drop"] / 1000.0),
            "curvature_pressure_drop_kpa": float(reference["curvature_pressure_drop"] / 1000.0),
            "d_pressure_drop_d_coil_separation_kpa_per_m": float(reference["d_pressure_drop_d_coil_separation"] / 1000.0),
            "d_pressure_drop_d_field_scale_kpa": float(reference["d_pressure_drop_d_field_scale"] / 1000.0),
            "d_pressure_drop_d_mean_velocity_kpa_per_m_per_s": float(reference["d_pressure_drop_d_mean_velocity"] / 1000.0),
            "elasticity_coil_separation": float(reference["elasticity_coil_separation"]),
            "elasticity_field_scale": float(reference["elasticity_field_scale"]),
            "elasticity_mean_velocity": float(reference["elasticity_mean_velocity"]),
        },
        "separation_sweep_m": _json_ready(study["separation_sweep"]),
        "separation_pressure_drop_kpa": _json_ready(study["separation_pressure_drop_kpa"]),
        "target_pressure_drop_kpa": float(study["target_pressure_drop_kpa"]),
        "field_scale_design_history": _json_ready(study["field_scale_design_history"]),
        "artifacts": artifacts,
        "source_data_artifacts": [
            "wham_blanket_autodiff_research_station_data.csv",
            "wham_blanket_autodiff_research_design_data.csv",
        ],
        "model_equation": (
            "Delta p = integral [ f_D*rho*U^2/(2D) + C_m*sigma*U*B_perp^2 "
            "+ distributed K_b*rho*U^2/2 bend loss ] ds"
        ),
        "model_status": "differentiable_reduced_fixed_flow_rate_pressure_budget",
    }


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "shape") and hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


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

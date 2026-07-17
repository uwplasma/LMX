from pathlib import Path
import importlib.util

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lmx.blanket_flow import (
    BlanketFlowSettings,
    BlanketTransientFlowSettings,
    LiquidMetalProperties,
    blanket_pressure_budget_from_transverse_field,
    solve_wham_blanket_reduced_flow,
    solve_wham_blanket_transient_flow,
    wham_blanket_pressure_drop_sensitivity,
    write_wham_blanket_autodiff_research_plots,
    write_wham_blanket_flow_movie,
    write_wham_blanket_flow_plots,
    write_wham_blanket_pressure_sweep_plots,
    write_wham_blanket_transient_flow_movie,
    write_wham_blanket_transient_flow_plots,
)
from lmx.blanket_geometry import (
    WhamBlanketLoop,
    build_wham_blanket_centerline,
    tube_surface_from_centerline,
    wham_blanket_clearance_metrics,
    write_centerline_pipe_mesh_preview,
    write_wham_blanket_geometry_preview,
)
from lmx.mesh import generate_centerline_pipe_mesh


pytestmark = pytest.mark.unit

_NO_MAGPY = importlib.util.find_spec("magpylib_jax") is None
_needs_magpy = pytest.mark.skipif(_NO_MAGPY, reason="magpylib_jax unavailable")


def _uniform_field(x, y, z):
    return np.column_stack([np.zeros_like(x), np.zeros_like(y), 0.2 * np.ones_like(z)])


def test_wham_blanket_centerline_has_clearance_and_monotone_station():
    geometry = WhamBlanketLoop(
        pipe_radius=0.08, bend_radius=0.7, entry_length=0.9, central_cell_radius=0.35
    )
    centerline = build_wham_blanket_centerline(
        geometry, straight_points=12, bend_points=24
    )
    metrics = wham_blanket_clearance_metrics(centerline, geometry)

    assert centerline["x"].shape == centerline["y"].shape == centerline["z"].shape
    assert np.all(np.diff(centerline["station"]) > 0.0)
    assert centerline["y"][0] < 0.0
    assert centerline["y"][-1] > 0.0
    assert centerline["x"][0] == pytest.approx(centerline["x"][-1])
    assert metrics["tube_to_cell_clearance"] > 0.0
    assert metrics["path_length"] > 2.0 * geometry.entry_length


def test_wham_blanket_centerline_rejects_overlapping_cell_envelope():
    geometry = WhamBlanketLoop(
        pipe_radius=0.2, bend_radius=0.5, central_cell_radius=0.35
    )

    with pytest.raises(ValueError, match="bend_radius"):
        build_wham_blanket_centerline(geometry)


def test_tube_surface_from_centerline_shape_and_radius():
    geometry = WhamBlanketLoop(
        pipe_radius=0.05, bend_radius=0.7, entry_length=0.9, central_cell_radius=0.35
    )
    centerline = build_wham_blanket_centerline(
        geometry, straight_points=10, bend_points=16
    )
    tube = tube_surface_from_centerline(
        centerline, radius=geometry.pipe_radius, circumferential_points=12
    )

    assert tube["x"].shape == (centerline["x"].size, 12)
    centers = np.column_stack([centerline["x"], centerline["y"], centerline["z"]])
    surface = np.stack([tube["x"], tube["y"], tube["z"]], axis=-1)
    distance = np.linalg.norm(surface - centers[:, None, :], axis=-1)
    assert np.max(np.abs(distance - geometry.pipe_radius)) < 1.0e-12


def test_write_wham_blanket_geometry_preview_outputs_artifacts(tmp_path: Path):
    geometry = WhamBlanketLoop(
        pipe_radius=0.06, bend_radius=0.75, entry_length=0.8, central_cell_radius=0.35
    )
    centerline = build_wham_blanket_centerline(
        geometry, straight_points=10, bend_points=18
    )
    outputs = write_wham_blanket_geometry_preview(
        centerline, tmp_path, geometry=geometry
    )

    names = {path.name for path in outputs}
    assert {
        "wham_blanket_geometry_preview.png",
        "wham_blanket_geometry_preview.pdf",
        "wham_blanket_geometry_preview_summary.json",
    } <= names
    assert all(path.exists() for path in outputs)


def test_write_centerline_pipe_mesh_preview_outputs_artifacts(tmp_path: Path):
    geometry = WhamBlanketLoop(
        pipe_radius=0.06, bend_radius=0.75, entry_length=0.8, central_cell_radius=0.35
    )
    centerline = build_wham_blanket_centerline(
        geometry, straight_points=10, bend_points=18
    )
    mesh = generate_centerline_pipe_mesh(
        centerline, tube_radius=geometry.pipe_radius, nx=8, nr=4, ntheta=12
    )
    outputs = write_centerline_pipe_mesh_preview(
        mesh, tmp_path, filename_stem="mesh_preview"
    )

    names = {path.name for path in outputs}
    assert {"mesh_preview.png", "mesh_preview.pdf", "mesh_preview_summary.json"} <= names
    assert all(path.exists() for path in outputs)


def test_solve_wham_blanket_reduced_flow_pressure_budget_and_profiles():
    geometry = WhamBlanketLoop(
        pipe_radius=0.08, bend_radius=0.6, entry_length=0.5, central_cell_radius=0.32
    )
    centerline = build_wham_blanket_centerline(
        geometry, straight_points=8, bend_points=12
    )
    settings = BlanketFlowSettings(
        mean_velocity=0.1, field_scale=1.0, cross_section_points=21
    )
    properties = LiquidMetalProperties(
        density=9000.0, dynamic_viscosity=2.0e-3, electrical_conductivity=8.0e5
    )

    flow = solve_wham_blanket_reduced_flow(
        centerline,
        geometry=geometry,
        properties=properties,
        settings=settings,
        field_sampler=_uniform_field,
    )

    assert flow["pressure_drop"] > 0.0
    assert flow["metrics"]["peak_hartmann_number"] > 0.0
    assert flow["metrics"]["mhd_pressure_fraction"] > 0.0
    assert flow["metrics"]["peak_dean_number"] > 0.0
    assert flow["metrics"]["peak_dean_skew_strength"] > 0.0
    assert flow["velocity_sections"].shape[0] == flow["station"].size
    profile = flow["velocity_sections"][0]
    assert np.nanmean(profile) == pytest.approx(settings.mean_velocity, rel=2.0e-2)


def test_write_wham_blanket_flow_artifacts(tmp_path: Path):
    geometry = WhamBlanketLoop(
        pipe_radius=0.06, bend_radius=0.55, entry_length=0.4, central_cell_radius=0.28
    )
    centerline = build_wham_blanket_centerline(
        geometry, straight_points=6, bend_points=10
    )
    flow = solve_wham_blanket_reduced_flow(
        centerline,
        geometry=geometry,
        settings=BlanketFlowSettings(mean_velocity=0.08, cross_section_points=17),
        field_sampler=_uniform_field,
    )

    plot_outputs = write_wham_blanket_flow_plots(flow, tmp_path)
    movie_outputs = write_wham_blanket_flow_movie(flow, tmp_path, frame_count=3, fps=4)

    assert (tmp_path / "wham_blanket_flow.png").exists()
    assert (tmp_path / "wham_blanket_flow_summary.json").exists()
    assert (tmp_path / "wham_blanket_flow_station_data.csv").exists()
    assert (tmp_path / "wham_blanket_flow.gif").exists()
    assert (tmp_path / "wham_blanket_flow_poster.png").exists()
    assert all(path.exists() for path in [*plot_outputs, *movie_outputs])


def test_write_wham_blanket_pressure_sweep_artifacts(tmp_path: Path):
    geometry = WhamBlanketLoop(
        pipe_radius=0.06, bend_radius=0.55, entry_length=0.4, central_cell_radius=0.28
    )
    centerline = build_wham_blanket_centerline(
        geometry, straight_points=6, bend_points=10
    )
    flows = [
        solve_wham_blanket_reduced_flow(
            centerline,
            geometry=geometry,
            settings=BlanketFlowSettings(
                mean_velocity=0.08, field_scale=scale, cross_section_points=17
            ),
            field_sampler=_uniform_field,
        )
        for scale in (0.5, 1.0, 1.5)
    ]

    outputs = write_wham_blanket_pressure_sweep_plots(flows, tmp_path)

    assert (tmp_path / "wham_blanket_pressure_sweep.png").exists()
    assert (tmp_path / "wham_blanket_pressure_sweep_summary.json").exists()
    assert (tmp_path / "wham_blanket_pressure_sweep_data.csv").exists()
    assert (
        flows[-1]["metrics"]["pressure_drop_kpa"]
        > flows[0]["metrics"]["pressure_drop_kpa"]
    )
    assert all(path.exists() for path in outputs)


def test_solve_wham_blanket_transient_flow_reaches_bounded_steady_state(tmp_path: Path):
    geometry = WhamBlanketLoop(
        pipe_radius=0.06, bend_radius=0.55, entry_length=0.4, central_cell_radius=0.28
    )
    centerline = build_wham_blanket_centerline(
        geometry, straight_points=6, bend_points=10
    )
    flow = solve_wham_blanket_reduced_flow(
        centerline,
        geometry=geometry,
        settings=BlanketFlowSettings(mean_velocity=0.08, cross_section_points=17),
        field_sampler=_uniform_field,
    )

    transient = solve_wham_blanket_transient_flow(
        flow,
        settings=BlanketTransientFlowSettings(
            time_step=0.05, final_time=3.0, frame_count=4
        ),
    )
    plot_outputs = write_wham_blanket_transient_flow_plots(transient, tmp_path)
    movie_outputs = write_wham_blanket_transient_flow_movie(transient, tmp_path, fps=4)

    assert transient["metrics"]["final_mean_velocity_m_per_s"] > 0.0
    assert transient["metrics"]["final_pressure_drop_kpa"] > 0.0
    assert (
        transient["metrics"]["final_bend_outboard_velocity_m_per_s"]
        > transient["metrics"]["final_bend_inboard_velocity_m_per_s"]
    )
    assert transient["velocity_frames"].shape[0] >= 3
    assert transient["bend_inboard_velocity_history"].shape == transient["time"].shape
    assert transient["bend_outboard_velocity_history"].shape == transient["time"].shape
    assert (tmp_path / "wham_blanket_transient_flow.png").exists()
    assert (tmp_path / "wham_blanket_flow.gif").exists()
    assert all(path.exists() for path in [*plot_outputs, *movie_outputs])


def test_blanket_pressure_budget_is_differentiable():
    station = jnp.linspace(0.0, 1.0, 8)
    curvature = jnp.zeros_like(station)

    def pressure_for_scale(field_scale):
        budget = blanket_pressure_budget_from_transverse_field(
            station,
            field_scale * jnp.ones_like(station),
            curvature,
            pipe_radius=0.05,
            mean_velocity=0.15,
            density=9300.0,
            dynamic_viscosity=1.8e-3,
            electrical_conductivity=7.9e5,
        )
        return budget["pressure_drop"]

    pressure, gradient = jax.value_and_grad(pressure_for_scale)(jnp.asarray(0.2))

    assert float(pressure) > 0.0
    assert float(gradient) > 0.0


@_needs_magpy
def test_wham_blanket_pressure_sensitivity_works_with_reduced_coils():
    geometry = WhamBlanketLoop(
        pipe_radius=0.06, bend_radius=0.55, entry_length=0.4, central_cell_radius=0.28
    )
    centerline = build_wham_blanket_centerline(
        geometry, straight_points=6, bend_points=8
    )
    settings = BlanketFlowSettings(
        mean_velocity=0.08, field_scale=2.0, radial_loops=2, axial_loops=1
    )
    sensitivity = wham_blanket_pressure_drop_sensitivity(
        centerline,
        geometry=geometry,
        settings=settings,
        coil_parameters={"radial_loops": 2, "axial_loops": 1, "current_scale": 2000.0},
    )

    assert float(sensitivity["pressure_drop"]) > 0.0
    assert np.isfinite(float(sensitivity["d_pressure_drop_d_field_scale"]))
    assert np.isfinite(float(sensitivity["elasticity_mean_velocity"]))


def test_write_wham_blanket_autodiff_research_plot(tmp_path: Path):
    station = jnp.linspace(0.0, 1.0, 6)
    reference = blanket_pressure_budget_from_transverse_field(
        station,
        0.2 * jnp.ones_like(station),
        jnp.zeros_like(station),
        pipe_radius=0.05,
        mean_velocity=0.15,
        density=9300.0,
        dynamic_viscosity=1.8e-3,
        electrical_conductivity=7.9e5,
    )
    reference = {
        **reference,
        "coil_separation": jnp.asarray(1.96),
        "field_scale": jnp.asarray(2.0),
        "mean_velocity": jnp.asarray(0.15),
        "d_pressure_drop_d_coil_separation": jnp.asarray(-10.0),
        "d_pressure_drop_d_field_scale": jnp.asarray(20.0),
        "d_pressure_drop_d_mean_velocity": jnp.asarray(30.0),
        "elasticity_coil_separation": jnp.asarray(-0.1),
        "elasticity_field_scale": jnp.asarray(0.2),
        "elasticity_mean_velocity": jnp.asarray(0.3),
    }
    study = {
        "reference": reference,
        "separation_sweep": [1.8, 1.96, 2.1],
        "separation_pressure_drop_kpa": [0.13, 0.12, 0.11],
        "target_pressure_drop_kpa": 0.10,
        "field_scale_design_history": [
            {"step": 0, "field_scale": 2.0, "pressure_drop_kpa": 0.12},
            {"step": 1, "field_scale": 1.8, "pressure_drop_kpa": 0.10},
        ],
    }

    outputs = write_wham_blanket_autodiff_research_plots(study, tmp_path)

    assert (tmp_path / "wham_blanket_autodiff_research.png").exists()
    assert (tmp_path / "wham_blanket_autodiff_research_summary.json").exists()
    assert all(path.exists() for path in outputs)

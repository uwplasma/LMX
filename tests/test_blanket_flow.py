from pathlib import Path

import numpy as np
import pytest

from lmx.blanket_flow import (
    BlanketFlowSettings,
    LiquidMetalProperties,
    solve_wham_blanket_reduced_flow,
    write_wham_blanket_flow_movie,
    write_wham_blanket_flow_plots,
)
from lmx.blanket_geometry import WhamBlanketLoop, build_wham_blanket_centerline


pytestmark = pytest.mark.unit


def _uniform_field(x, y, z):
    return np.column_stack([np.zeros_like(x), np.zeros_like(y), 0.2 * np.ones_like(z)])


def test_solve_wham_blanket_reduced_flow_pressure_budget_and_profiles():
    geometry = WhamBlanketLoop(pipe_radius=0.08, bend_radius=0.6, entry_length=0.5, central_cell_radius=0.32)
    centerline = build_wham_blanket_centerline(geometry, straight_points=8, bend_points=12)
    settings = BlanketFlowSettings(mean_velocity=0.1, field_scale=1.0, cross_section_points=21)
    properties = LiquidMetalProperties(density=9000.0, dynamic_viscosity=2.0e-3, electrical_conductivity=8.0e5)

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
    assert flow["velocity_sections"].shape[0] == flow["station"].size
    profile = flow["velocity_sections"][0]
    assert np.nanmean(profile) == pytest.approx(settings.mean_velocity, rel=2.0e-2)


def test_write_wham_blanket_flow_artifacts(tmp_path: Path):
    geometry = WhamBlanketLoop(pipe_radius=0.06, bend_radius=0.55, entry_length=0.4, central_cell_radius=0.28)
    centerline = build_wham_blanket_centerline(geometry, straight_points=6, bend_points=10)
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

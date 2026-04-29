from pathlib import Path

import numpy as np
import pytest

from lmx.blanket_geometry import (
    WhamBlanketLoop,
    build_wham_blanket_centerline,
    tube_surface_from_centerline,
    wham_blanket_clearance_metrics,
    write_wham_blanket_geometry_preview,
)


pytestmark = pytest.mark.unit


def test_wham_blanket_centerline_has_clearance_and_monotone_station():
    geometry = WhamBlanketLoop(pipe_radius=0.08, bend_radius=0.7, entry_length=0.9, central_cell_radius=0.35)

    centerline = build_wham_blanket_centerline(geometry, straight_points=12, bend_points=24)
    metrics = wham_blanket_clearance_metrics(centerline, geometry)

    assert centerline["x"].shape == centerline["y"].shape == centerline["z"].shape
    assert np.all(np.diff(centerline["station"]) > 0.0)
    assert centerline["y"][0] < 0.0
    assert centerline["y"][-1] > 0.0
    assert centerline["x"][0] == pytest.approx(centerline["x"][-1])
    assert metrics["tube_to_cell_clearance"] > 0.0
    assert metrics["path_length"] > 2.0 * geometry.entry_length


def test_wham_blanket_centerline_rejects_overlapping_cell_envelope():
    geometry = WhamBlanketLoop(pipe_radius=0.2, bend_radius=0.5, central_cell_radius=0.35)

    with pytest.raises(ValueError, match="bend_radius"):
        build_wham_blanket_centerline(geometry)


def test_tube_surface_from_centerline_shape_and_radius():
    geometry = WhamBlanketLoop(pipe_radius=0.05, bend_radius=0.7, entry_length=0.9, central_cell_radius=0.35)
    centerline = build_wham_blanket_centerline(geometry, straight_points=10, bend_points=16)

    tube = tube_surface_from_centerline(centerline, radius=geometry.pipe_radius, circumferential_points=12)

    assert tube["x"].shape == (centerline["x"].size, 12)
    centers = np.column_stack([centerline["x"], centerline["y"], centerline["z"]])
    surface = np.stack([tube["x"], tube["y"], tube["z"]], axis=-1)
    distance = np.linalg.norm(surface - centers[:, None, :], axis=-1)
    assert np.max(np.abs(distance - geometry.pipe_radius)) < 1.0e-12


def test_write_wham_blanket_geometry_preview_outputs_artifacts(tmp_path: Path):
    geometry = WhamBlanketLoop(pipe_radius=0.06, bend_radius=0.75, entry_length=0.8, central_cell_radius=0.35)
    centerline = build_wham_blanket_centerline(geometry, straight_points=10, bend_points=18)

    outputs = write_wham_blanket_geometry_preview(centerline, tmp_path, geometry=geometry)

    names = {path.name for path in outputs}
    assert "wham_blanket_geometry_preview.png" in names
    assert "wham_blanket_geometry_preview.pdf" in names
    assert "wham_blanket_geometry_preview_summary.json" in names
    assert all(path.exists() for path in outputs)

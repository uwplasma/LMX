from pathlib import Path

import numpy as np

from lmx.centerline_fields import (
    centerline_field_quality_metrics,
    centerline_current_closure_metrics,
    centerline_current_pressure_metrics,
    centerline_pipe_frames,
    sample_field_on_centerline_pipe_mesh,
    solve_centerline_pipe_current_closure,
    write_centerline_field_preview,
    write_centerline_current_closure_preview,
)
from lmx.mesh import generate_centerline_pipe_mesh


def _straight_centerline() -> dict[str, np.ndarray]:
    x = np.linspace(0.0, 1.0, 5)
    return {
        "x": x,
        "y": np.zeros_like(x),
        "z": np.zeros_like(x),
    }


def test_centerline_pipe_frames_recover_straight_pipe_basis():
    mesh = generate_centerline_pipe_mesh(_straight_centerline(), tube_radius=0.1, nx=4, nr=4, ntheta=12)
    frames = centerline_pipe_frames(mesh)

    np.testing.assert_allclose(frames["tangent"], np.array([[1.0, 0.0, 0.0]] * 5), atol=1.0e-12)
    np.testing.assert_allclose(frames["normal"], np.array([[0.0, 1.0, 0.0]] * 5), atol=1.0e-12)
    np.testing.assert_allclose(frames["binormal"], np.array([[0.0, 0.0, 1.0]] * 5), atol=1.0e-12)


def test_field_sampling_projects_uniform_transverse_field():
    mesh = generate_centerline_pipe_mesh(_straight_centerline(), tube_radius=0.1, nx=4, nr=4, ntheta=12)

    def uniform_z_field(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        return np.column_stack([np.zeros_like(x), np.zeros_like(y), 2.0 * np.ones_like(z)])

    sample = sample_field_on_centerline_pipe_mesh(mesh, uniform_z_field)
    np.testing.assert_allclose(sample["B_s"], 0.0, atol=1.0e-12)
    np.testing.assert_allclose(sample["B_n"], 0.0, atol=1.0e-12)
    np.testing.assert_allclose(sample["B_b"], 2.0, atol=1.0e-12)
    np.testing.assert_allclose(sample["B_perp"], 2.0, atol=1.0e-12)
    metrics = centerline_field_quality_metrics(sample)
    assert metrics["validation_pass"] is True
    assert metrics["peak_centerline_b_perp"] == 2.0


def test_write_centerline_field_preview_outputs_artifacts(tmp_path: Path):
    mesh = generate_centerline_pipe_mesh(_straight_centerline(), tube_radius=0.1, nx=4, nr=4, ntheta=12)

    def varying_field(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        return np.column_stack([0.1 * x, y, 2.0 + z])

    sample = sample_field_on_centerline_pipe_mesh(mesh, varying_field, max_points_per_call=11)
    outputs = write_centerline_field_preview(sample, tmp_path, filename_stem="field_preview")

    output_names = {path.name for path in outputs}
    assert {"field_preview.png", "field_preview.pdf", "field_preview_summary.json", "field_preview_centerline.csv"} <= output_names
    assert all(path.exists() for path in outputs)


def test_centerline_current_closure_cancels_emf_divergence(tmp_path: Path):
    mesh = generate_centerline_pipe_mesh(_straight_centerline(), tube_radius=0.1, nx=4, nr=4, ntheta=12)

    def uniform_z_field(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        return np.column_stack([np.zeros_like(x), np.zeros_like(y), 2.0 * np.ones_like(z)])

    sample = sample_field_on_centerline_pipe_mesh(mesh, uniform_z_field)
    closure = solve_centerline_pipe_current_closure(
        sample,
        mean_velocity=0.2,
        conductivity=10.0,
        potential_iterations=80,
        potential_tolerance=1.0e-11,
    )
    metrics = centerline_current_closure_metrics(closure)
    pressure_metrics = centerline_current_pressure_metrics(closure)

    assert metrics["validation_pass"] is True
    assert metrics["max_charge_balance_residual"] < 1.0e-8
    assert metrics["net_boundary_current_residual"] < 1.0e-8
    assert pressure_metrics["validation_pass"] is True
    assert pressure_metrics["mhd_pressure_drop_proxy_pa"] >= 0.0

    outputs = write_centerline_current_closure_preview(closure, tmp_path, filename_stem="current_closure")
    output_names = {path.name for path in outputs}
    assert {
        "current_closure.png",
        "current_closure.pdf",
        "current_closure_summary.json",
        "current_closure_station_data.csv",
    } <= output_names

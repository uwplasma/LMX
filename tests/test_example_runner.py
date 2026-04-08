from pathlib import Path
import importlib.util

import numpy as np
import pytest

from lmx.example_runner import run_case_example, run_theory_meeting_demo


pytestmark = pytest.mark.unit


def test_run_case_example_writes_hartmann_outputs(tmp_path: Path):
    report = run_case_example(
        case_kind="hartmann",
        ha=5.0,
        ny=12,
        nz=12,
        out_dir=tmp_path,
        reference_root=None,
    )

    assert report["case"] == "hartmann_ha5"
    assert (tmp_path / "overview.png").exists()
    assert (tmp_path / "overview.pdf").exists()
    assert (tmp_path / "diagnostics.png").exists()
    assert (tmp_path / "diagnostics.pdf").exists()
    assert (tmp_path / "example_report.json").exists()
    assert (tmp_path / "hartmann_ha5.vtr").exists()
    assert (tmp_path / "hartmann_ha5_centerline.csv").exists()


def test_run_theory_meeting_demo_writes_movies_and_reports(tmp_path: Path):
    report = run_theory_meeting_demo(
        out_dir=tmp_path,
        hartmann_ha=5.0,
        shercliff_ha=5.0,
        hunt_ha=5.0,
        resolution=12,
        movie_case="shercliff",
        movie_resolution=10,
        movie_dt=1e-5,
        movie_t_final=3e-5,
        movie_frames=3,
        reference_root=None,
    )

    assert "hartmann" in report
    assert "shercliff" in report
    assert "hunt" in report
    assert report["movie_case"] == "shercliff"
    assert report["movie_mode"] == "raw"
    assert (tmp_path / "meeting_demo_report.json").exists()
    assert (tmp_path / "shercliff" / "shercliff_startup_2d.gif").exists()
    assert (tmp_path / "shercliff" / "shercliff_startup_3d.gif").exists()
    assert (tmp_path / "shercliff" / "shercliff_startup_2d_poster.png").exists()
    assert (tmp_path / "shercliff" / "shercliff_startup_2d_poster.pdf").exists()
    assert (tmp_path / "shercliff" / "shercliff_startup_3d_poster.png").exists()
    assert (tmp_path / "shercliff" / "shercliff_startup_3d_poster.pdf").exists()


def _load_example_module(filename: str):
    module_path = Path(__file__).resolve().parents[1] / "examples" / filename
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plot_npz_results_reads_solution_and_movie_npz(tmp_path: Path):
    plot_module = _load_example_module("plot_npz_results.py")

    y_faces = np.linspace(-1.0, 1.0, 4)
    z_faces = np.linspace(-1.0, 1.0, 4)
    y_centers = 0.5 * (y_faces[:-1] + y_faces[1:])
    z_centers = 0.5 * (z_faces[:-1] + z_faces[1:])
    yy, zz = np.meshgrid(y_centers, z_centers, indexing="ij")
    u = 1.0 - yy**2 - 0.2 * zz**2
    phi = yy - zz
    solution_npz = tmp_path / "solution.npz"
    np.savez_compressed(
        solution_npz,
        metadata_json='{"case": "test_case"}',
        y_centers=y_centers,
        z_centers=z_centers,
        y_faces=y_faces,
        z_faces=z_faces,
        u=u,
        phi=phi,
        time_history=np.array([0.0, 1.0]),
        u_max_history=np.array([1.0, 2.0]),
        current_max_history=np.array([0.5, 0.6]),
        lorentz_max_history=np.array([0.7, 0.8]),
        residual_history=np.array([1e-2, 1e-3]),
        potential_residual_history=np.array([2e-2, 2e-3]),
    )

    plot_outputs = plot_module.plot_solution_npz(solution_npz, tmp_path / "plots")
    assert (tmp_path / "plots" / "overview_from_npz.png").exists()
    assert (tmp_path / "plots" / "diagnostics_from_npz.png").exists()
    assert len(plot_outputs) == 4

    movie_npz = tmp_path / "movie.npz"
    np.savez_compressed(
        movie_npz,
        metadata_json='{"case": "test_movie", "title": "Test movie"}',
        y_centers=y_centers,
        z_centers=z_centers,
        y_faces=y_faces,
        z_faces=z_faces,
        time=np.array([0.0, 1.0]),
        u_stack=np.stack([u, 1.1 * u]),
    )
    movie_outputs = plot_module.plot_movie_npz(movie_npz, tmp_path / "movie", stem="test_movie", fps=2)
    assert (tmp_path / "movie" / "test_movie_2d.gif").exists()
    assert (tmp_path / "movie" / "test_movie_3d.gif").exists()
    assert len(movie_outputs) == 4

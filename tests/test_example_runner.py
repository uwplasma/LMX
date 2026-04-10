from pathlib import Path
import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest

import lmx.example_runner as example_runner
from lmx.example_runner import run_case_example, run_theory_meeting_demo


pytestmark = pytest.mark.unit


def test_run_case_example_writes_hartmann_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fake_solution = SimpleNamespace(
        state=SimpleNamespace(u=np.array([[1.0]]), phi=np.array([[0.0]]), time=0.0, residual=0.0),
        mesh=SimpleNamespace(),
        case_name="hartmann_ha5",
    )

    monkeypatch.setattr(example_runner, "solve_steady", lambda case: fake_solution)
    monkeypatch.setattr(example_runner, "write_paraview", lambda solution, out_dir: (out_dir / "hartmann_ha5.vtr").write_text("vtk"))
    monkeypatch.setattr(example_runner, "write_profile_csv", lambda path, profile: path.write_text("coord,u\n0,1\n"))
    monkeypatch.setattr(example_runner, "extract_centerline", lambda solution: {"y": [0.0], "u": [1.0]})
    monkeypatch.setattr(example_runner, "extract_midplane_profile", lambda solution, axis, fluid_only=True: {"y" if axis == "y" else "z": [0.0], "u": [1.0]})
    monkeypatch.setattr(example_runner, "validation_summary", lambda solution, case_name, ha: {"u_max": 1.0})
    monkeypatch.setattr(
        example_runner,
        "hartmann_validation",
        lambda solution, ha: SimpleNamespace(coordinate=np.array([0.0]), reference=np.array([1.0]), l2_error=0.0, linf_error=0.0),
    )

    def fake_write_case_overview_plots(solution, out_dir: Path, **kwargs):
        outputs = [
            out_dir / "overview.png",
            out_dir / "overview.pdf",
            out_dir / "diagnostics.png",
            out_dir / "diagnostics.pdf",
        ]
        for path in outputs:
            path.write_bytes(b"plot")
        return outputs

    monkeypatch.setattr(example_runner, "write_case_overview_plots", fake_write_case_overview_plots)
    monkeypatch.setattr(example_runner, "write_metrics_json", lambda payload, path: path.write_text("{}"))

    report = run_case_example(
        case_kind="hartmann",
        ha=5.0,
        ny=8,
        nz=8,
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


def test_run_theory_meeting_demo_writes_movies_and_reports(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fake_solution = SimpleNamespace(
        state=SimpleNamespace(u=np.array([[1.0]]), phi=np.array([[0.0]]), time=0.0, residual=0.0),
        mesh=SimpleNamespace(),
        case_name="hunt_ha5",
    )

    def fake_run_case_example(*, case_kind: str, ha: float, ny: int, nz: int, out_dir: Path, reference_root):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_paths = [
            out_dir / "overview.png",
            out_dir / "overview.pdf",
            out_dir / "diagnostics.png",
            out_dir / "diagnostics.pdf",
        ]
        for path in plot_paths:
            path.write_bytes(b"plot")
        (out_dir / "example_report.json").write_text("{}")
        (out_dir / f"{case_kind}_ha{int(ha)}.vtr").write_text("vtk")
        (out_dir / f"{case_kind}_ha{int(ha)}_centerline.csv").write_text("y,u\n0,1\n")
        npz_path = out_dir / f"{case_kind}_ha{int(ha)}_results.npz"
        np.savez_compressed(npz_path, metadata_json='{"case": "demo"}', y_centers=np.array([0.0]), z_centers=np.array([0.0]), y_faces=np.array([-1.0, 1.0]), z_faces=np.array([-1.0, 1.0]), u=np.array([[1.0]]), phi=np.array([[0.0]]))
        return {
            "case": f"{case_kind}_ha{int(ha)}",
            "ha": ha,
            "output_dir": str(out_dir),
            "plots": [str(path) for path in plot_paths],
            "reference": {"available": False},
            "metrics": {"u_max": 1.0},
            "npz": str(npz_path),
        }

    def fake_solve_case_snapshots(case, *, frame_count: int = 12):
        mesh = type("Mesh", (), {"y_centers": np.array([0.0]), "z_centers": np.array([0.0]), "y_faces": np.array([-1.0, 1.0]), "z_faces": np.array([-1.0, 1.0])})()
        return [
            {
                "time": 0.0,
                "u": np.array([[1.0]]),
                "phi": np.array([[0.0]]),
                "jy": np.array([[0.0]]),
                "jz": np.array([[0.0]]),
                "lorentz_x": np.array([[0.0]]),
                "fluid_mask": np.array([[True]]),
                "residual": 0.0,
                "potential_residual": 0.0,
                "potential_iterations": 1.0,
                "face_current_max": 0.0,
                "emf_max": 0.0,
                "face_lorentz_max": 0.0,
                "mean_velocity": 1.0,
                "applied_forcing": 0.0,
                "pressure_proxy": 0.0,
                "mesh": mesh,
            }
            for _ in range(frame_count)
        ]

    def fake_write_transient_movies(frames_payload, movie_dir: Path, *, case_title: str, field_mode: str, output_stem: str):
        movie_dir.mkdir(parents=True, exist_ok=True)
        outputs = [
            movie_dir / f"{output_stem}_2d.gif",
            movie_dir / f"{output_stem}_3d.gif",
            movie_dir / f"{output_stem}_2d_poster.png",
            movie_dir / f"{output_stem}_2d_poster.pdf",
            movie_dir / f"{output_stem}_3d_poster.png",
            movie_dir / f"{output_stem}_3d_poster.pdf",
        ]
        for path in outputs:
            path.write_bytes(b"movie")
        return outputs

    def fake_make_hunt_case(*, ha: float, ny: int, nz: int):
        return SimpleNamespace(name=f"hunt_ha{int(ha)}")

    def fake_write_paraview(solution, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{solution.case_name}.vtr").write_text("vtk")
        return [out_dir / f"{solution.case_name}.vtr"]

    def fake_write_profile_csv(path: Path, profile):
        path.write_text("coord,u\n0,1\n")
        return path

    def fake_validation_summary(solution, case_name: str, ha: float):
        return {"u_max": 1.0, "case": case_name, "ha": ha}

    def fake_write_case_overview_plots(solution, out_dir: Path, **kwargs):
        out_dir.mkdir(parents=True, exist_ok=True)
        outputs = [
            out_dir / "overview.png",
            out_dir / "overview.pdf",
            out_dir / "diagnostics.png",
            out_dir / "diagnostics.pdf",
        ]
        for path in outputs:
            path.write_bytes(b"plot")
        return outputs

    def fake_write_metrics_json(payload, path: Path):
        path.write_text("{}")
        return path

    monkeypatch.setattr(example_runner, "run_case_example", fake_run_case_example)
    monkeypatch.setattr(example_runner, "solve_case_snapshots", fake_solve_case_snapshots)
    monkeypatch.setattr(example_runner, "write_transient_movies", fake_write_transient_movies)
    monkeypatch.setattr(example_runner, "make_hunt_case", fake_make_hunt_case)
    monkeypatch.setattr(example_runner, "solve_steady", lambda case: fake_solution)
    monkeypatch.setattr(example_runner, "write_paraview", fake_write_paraview)
    monkeypatch.setattr(example_runner, "write_profile_csv", fake_write_profile_csv)
    monkeypatch.setattr(example_runner, "extract_centerline", lambda solution: {"y": [0.0], "u": [1.0]})
    monkeypatch.setattr(example_runner, "extract_midplane_profile", lambda solution, axis, fluid_only=True: {"y" if axis == "y" else "z": [0.0], "u": [1.0]})
    monkeypatch.setattr(example_runner, "validation_summary", fake_validation_summary)
    monkeypatch.setattr(example_runner, "write_case_overview_plots", fake_write_case_overview_plots)
    monkeypatch.setattr(example_runner, "write_metrics_json", fake_write_metrics_json)

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
    save_calls: list[Path] = []

    def fake_save(self, filename, *args, **kwargs):
        path = Path(filename)
        path.write_bytes(b"gif")
        save_calls.append(path)

    plot_module.animation.FuncAnimation.save = fake_save

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
    assert len(save_calls) == 2
    assert len(movie_outputs) == 4

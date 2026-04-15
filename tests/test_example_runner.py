from pathlib import Path
import importlib.util
from types import SimpleNamespace
import json

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


def test_run_theory_meeting_demo_records_hunt_reference_when_available(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fake_solution = SimpleNamespace(
        state=SimpleNamespace(u=np.array([[1.0]]), phi=np.array([[0.0]]), time=0.0, residual=0.0),
        mesh=SimpleNamespace(),
        case_name="hunt_ha5",
    )

    monkeypatch.setattr(
        example_runner,
        "run_case_example",
        lambda **kwargs: {"case": "demo", "ha": kwargs["ha"], "output_dir": str(kwargs["out_dir"]), "plots": [], "reference": {"available": False}, "metrics": {}},
    )
    monkeypatch.setattr(example_runner, "solve_case_snapshots", lambda case, frame_count=12: [])
    monkeypatch.setattr(example_runner, "write_transient_movies", lambda *args, **kwargs: [])
    monkeypatch.setattr(example_runner, "solve_steady", lambda case: fake_solution)
    monkeypatch.setattr(example_runner, "write_paraview", lambda solution, out_dir: [out_dir / "hunt_ha5.vtr"])
    def fake_write_profile_csv(path: Path, profile):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("coord,u\n0,1\n")
        return path

    monkeypatch.setattr(example_runner, "write_profile_csv", fake_write_profile_csv)
    monkeypatch.setattr(example_runner, "extract_centerline", lambda solution: {"y": [0.0], "u": [1.0]})
    monkeypatch.setattr(example_runner, "extract_midplane_profile", lambda solution, axis, fluid_only=True: {"y" if axis == "y" else "z": [0.0], "u": [1.0]})
    monkeypatch.setattr(example_runner, "validation_summary", lambda solution, case_name, ha: {"u_max": 1.0})
    monkeypatch.setattr(example_runner, "write_case_overview_plots", lambda *args, **kwargs: [])
    monkeypatch.setattr(example_runner, "write_metrics_json", lambda payload, path: path.write_text("{}"))
    monkeypatch.setattr(
        example_runner,
        "closed_channel_validation",
        lambda solution, case_kind, ha, reference_root: SimpleNamespace(
            y_profile=SimpleNamespace(coordinate=np.array([0.0]), reference=np.array([1.0]), l2_error=0.1),
            z_profile=SimpleNamespace(coordinate=np.array([0.0]), reference=np.array([1.0]), l2_error=0.2),
            reference_path="reference/hunt.json",
        ),
    )

    report = run_theory_meeting_demo(
        out_dir=tmp_path,
        hartmann_ha=5.0,
        shercliff_ha=5.0,
        hunt_ha=5.0,
        resolution=8,
        movie_case="hunt",
        movie_resolution=8,
        movie_dt=1e-5,
        movie_t_final=2e-5,
        movie_frames=2,
        reference_root=tmp_path,
    )

    assert report["hunt"]["reference"]["available"] is True
    assert report["hunt"]["reference"]["kind"] == "closed_channel_analytical"
    assert report["hunt"]["reference"]["path"] == "reference/hunt.json"


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


def test_geometry_preview_demo_writes_preview_and_optional_post_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module = _load_example_module("geometry_preview_demo.py")

    def fake_write_geometry_preview_plots(mesh, out_dir: Path, case_title: str):
        out_dir.mkdir(parents=True, exist_ok=True)
        outputs = [out_dir / "geometry_preview.png", out_dir / "geometry_preview.pdf"]
        for path in outputs:
            path.write_bytes(b"preview")
        return outputs

    def fake_solve_steady(case):
        return SimpleNamespace(case_name=case.name)

    def fake_write_case_overview_plots(solution, out_dir: Path, *, case_title: str):
        out_dir.mkdir(parents=True, exist_ok=True)
        outputs = [out_dir / "overview.png", out_dir / "overview.pdf"]
        for path in outputs:
            path.write_bytes(b"post")
        return outputs

    monkeypatch.setattr(module, "write_geometry_preview_plots", fake_write_geometry_preview_plots)
    monkeypatch.setattr(module, "solve_steady", fake_solve_steady)
    monkeypatch.setattr(module, "write_case_overview_plots", fake_write_case_overview_plots)

    summary = module.write_preview_bundle(out_dir=tmp_path, with_post_run=True, post_case_kind="hartmann")
    assert summary["with_post_run"] is True
    assert summary["post_case_kind"] == "hartmann"
    assert (tmp_path / "hartmann_geometry" / "geometry_preview.png").exists()
    assert (tmp_path / "hunt_geometry" / "geometry_preview.png").exists()
    assert (tmp_path / "pipe_geometry" / "geometry_preview.png").exists()
    assert (tmp_path / "hartmann_post" / "overview.png").exists()
    assert (tmp_path / "geometry_preview_summary.json").exists()


def test_readme_showcase_demo_writes_media_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module = _load_example_module("readme_showcase_demo.py")

    monkeypatch.setattr(
        module,
        "run_geometry_panel_demo",
        lambda *, out_dir: {"plots": ["geometry_gallery.png"], "geometries": {"rect_duct": {}}},
    )
    monkeypatch.setattr(module, "solve_case_snapshots", lambda *args, **kwargs: [{"time": 0.0, "u": np.ones((2, 2)), "mesh": None}])

    def fake_write_transient_movies(frames, out_dir: Path, *, case_title: str, fps: int, field_mode: str, output_stem: str):
        outputs = [
            out_dir / f"{output_stem}_2d.gif",
            out_dir / f"{output_stem}_3d.gif",
        ]
        for path in outputs:
            path.write_bytes(b"gif")
        return outputs

    monkeypatch.setattr(module, "write_transient_movies", fake_write_transient_movies)

    summary = module.run_readme_showcase_demo(out_dir=tmp_path, movie_frames=2)

    assert summary["case"] == "readme_showcase_demo"
    assert (tmp_path / "readme_hunt_startup_2d.gif").exists()
    assert (tmp_path / "readme_hunt_startup_3d.gif").exists()
    assert (tmp_path / "readme_showcase_summary.json").exists()


def test_fringing_benchmark_demo_writes_extruded_bundle_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module = _load_example_module("fringing_benchmark_demo.py")

    monkeypatch.setattr(
        module,
        "build_square_duct_extruded_problem",
        lambda **kwargs: SimpleNamespace(
            case=SimpleNamespace(name="fringing_case", solver=SimpleNamespace(kind="extruded_inductionless")),
            profile=SimpleNamespace(x=np.array([0.0, 1.0]), field_scale=np.array([0.0, 1.0]), axis="z"),
        ),
    )
    monkeypatch.setattr(
        module,
        "build_layered_duct_extruded_problem",
        lambda **kwargs: SimpleNamespace(
            case=SimpleNamespace(name="fringing_case_layered", solver=SimpleNamespace(kind="extruded_inductionless")),
            profile=SimpleNamespace(x=np.array([0.0, 1.0]), field_scale=np.array([0.0, 1.0]), axis="z"),
        ),
    )
    monkeypatch.setattr(
        module,
        "build_pipe_ogrid_extruded_problem",
        lambda **kwargs: SimpleNamespace(
            case=SimpleNamespace(name="fringing_case_pipe", solver=SimpleNamespace(kind="extruded_inductionless")),
            profile=SimpleNamespace(x=np.array([0.0, 1.0]), field_scale=np.array([0.0, 1.0]), axis="z"),
        ),
    )
    monkeypatch.setattr(
        module,
        "solve_extruded_inductionless",
            lambda *args, **kwargs: SimpleNamespace(
                station_history=(
                    {"x": 0.0, "field_scale": 0.0, "mean_velocity": 1.0, "u_max": 1.1, "pressure_span": 0.3, "axial_current": 0.01, "current_scaled_pressure_proxy": 0.2},
                    {"x": 1.0, "field_scale": 1.0, "mean_velocity": 0.8, "u_max": 0.9, "pressure_span": 0.2, "axial_current": 0.015, "current_scaled_pressure_proxy": 0.25},
                ),
            bundle=SimpleNamespace(
                x=np.array([0.0, 1.0]),
                y=np.array([-1.0, 1.0]),
                z=np.array([-1.0, 1.0]),
                field_scale=np.array([0.0, 1.0]),
                u=np.ones((2, 2, 2)),
                charge_balance_residual=np.array([1.0e-7, 2.0e-7]),
                wall_current_leakage=np.array([1.0e-8, 2.0e-8]),
                axial_current=np.array([0.01, 0.015]),
            ),
            validation=SimpleNamespace(
                station_count=2,
                max_residual=1.0e-6,
                max_charge_balance_residual=2.0e-7,
                mean_velocity_span=0.2,
                volumetric_flow_rate_span=0.1,
                axial_current_span=0.01,
                peak_velocity_span=0.03,
                pressure_span_range=0.04,
                max_wall_current_leakage=2.0e-8,
                net_boundary_current_residual=3.0e-8,
                field_mean_velocity_correlation=0.9,
            ),
        ),
    )

    summary = module.run_fringing_benchmark_demo(out_dir=tmp_path, nx_stations=2, ny=4, nz=4)
    assert summary["case"] == "fringing_case"
    assert "extruded_bundle" in summary
    assert "validation" in summary
    assert "pressure_span" in summary["extruded_bundle"]
    assert "u_peak" in summary["extruded_bundle"]
    assert (tmp_path / "fringing_benchmark.png").exists()
    assert (tmp_path / "fringing_benchmark_summary.json").exists()

    summary_layered = module.run_fringing_benchmark_demo(
        out_dir=tmp_path / "layered",
        geometry_kind="layered_duct",
        nx_stations=2,
        ny=4,
        nz=4,
    )
    assert summary_layered["case"] == "fringing_case_layered"
    assert summary_layered["geometry_kind"] == "layered_duct"

    summary_pipe = module.run_fringing_benchmark_demo(
        out_dir=tmp_path / "pipe",
        geometry_kind="pipe_ogrid",
        nx_stations=2,
        ny=4,
        nz=4,
    )
    assert summary_pipe["case"] == "fringing_case_pipe"
    assert summary_pipe["geometry_kind"] == "pipe_ogrid"


def test_autodiff_profile_design_demo_writes_summary(tmp_path: Path):
    module = _load_example_module("autodiff_profile_design_demo.py")
    summary = module.run_autodiff_profile_design_demo(
        out_dir=tmp_path,
        target_forcing=1.0,
        target_hartmann_number=10.0,
        forcing_init=0.5,
        hartmann_init=5.0,
        learning_rate_forcing=5.0,
        learning_rate_ha=1.0,
        steps=4,
    )
    assert "recovered_forcing" in summary
    assert "recovered_hartmann_number" in summary
    assert (tmp_path / "autodiff_profile_design.png").exists()
    assert (tmp_path / "autodiff_profile_design_summary.json").exists()


def test_autodiff_extruded_trajectory_demo_writes_summary(tmp_path: Path):
    module = _load_example_module("autodiff_extruded_trajectory_demo.py")
    summary = module.run_autodiff_extruded_trajectory_demo(
        out_dir=tmp_path,
        nx_stations=5,
        ny=4,
        nz=4,
        steps=2,
    )
    assert summary["case"] == "extruded_rect_projection_trajectory_design"
    assert (tmp_path / "autodiff_extruded_trajectory.png").exists()
    assert (tmp_path / "autodiff_extruded_trajectory_summary.json").exists()


def test_extruded_summary_figures_writes_summary(tmp_path: Path):
    module = _load_example_module("extruded_summary_figures.py")
    summary = module.run_extruded_summary_figures(
        out_dir=tmp_path,
        ha_peak=8.0,
        ny=4,
        nz=4,
        nx_stations=5,
    )
    assert summary["case"] == "extruded_summary_figures"
    assert (tmp_path / "fringing_rect_3d.png").exists()
    assert (tmp_path / "fringing_layered_3d.png").exists()
    assert (tmp_path / "fringing_summary_panel.png").exists()
    assert (tmp_path / "extruded_summary_figures_summary.json").exists()


def test_geometry_panel_demo_writes_summary(tmp_path: Path):
    module = _load_example_module("geometry_panel_demo.py")
    summary = module.run_geometry_panel_demo(out_dir=tmp_path)
    assert summary["case"] == "geometry_panel_demo"
    assert (tmp_path / "geometry_gallery.png").exists()
    assert (tmp_path / "geometry_panel_summary.json").exists()


def test_pipe_reference_comparison_demo_writes_summary(tmp_path: Path):
    module = _load_example_module("pipe_reference_comparison_demo.py")
    summary = module.run_pipe_reference_comparison_demo(
        out_dir=tmp_path,
        nr=4,
        ntheta=16,
        nx_stations=4,
        max_steps=4,
        coupling_iterations=4,
        potential_iterations=16,
    )
    assert summary["geometry_kind"] == "pipe_ogrid"
    assert summary["qualitative_only"] is True
    assert (tmp_path / "pipe_reference_comparison.png").exists()
    assert (tmp_path / "pipe_reference_comparison_summary.json").exists()


def test_autodiff_fringing_design_demo_writes_summary(tmp_path: Path):
    module = _load_example_module("autodiff_fringing_design_demo.py")
    summary = module.run_autodiff_fringing_design_demo(out_dir=tmp_path, steps=4)
    assert "recovered" in summary
    assert (tmp_path / "autodiff_fringing_design.png").exists()
    assert (tmp_path / "autodiff_fringing_design_summary.json").exists()


def test_autodiff_fringing_response_demo_writes_summary(tmp_path: Path):
    module = _load_example_module("autodiff_fringing_response_demo.py")
    summary = module.run_autodiff_fringing_response_demo(out_dir=tmp_path, steps=4)
    assert "recovered" in summary
    assert (tmp_path / "autodiff_fringing_response.png").exists()
    assert (tmp_path / "autodiff_fringing_response_summary.json").exists()


def test_autodiff_extruded_target_demo_writes_summary(tmp_path: Path):
    module = _load_example_module("autodiff_extruded_target_demo.py")
    summary = module.run_autodiff_extruded_target_demo(out_dir=tmp_path, ny=4, nz=4, nx_stations=4, steps=4)
    assert summary["solver_kind"] == "extruded_inductionless"
    assert (tmp_path / "autodiff_extruded_target.png").exists()
    assert (tmp_path / "autodiff_extruded_target_summary.json").exists()


def test_variable_field_geometry_demo_writes_preview_and_run_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module = _load_example_module("variable_field_geometry_demo.py")

    monkeypatch.setattr(module, "_build_mesh", lambda case: SimpleNamespace(geometry=case.geometry.kind))
    def fake_write_geometry_preview_plots(mesh, out_dir: Path, case_title: str):
        out_dir.mkdir(parents=True, exist_ok=True)
        outputs = [out_dir / "geometry_preview.png", out_dir / "geometry_preview.pdf"]
        for path in outputs:
            path.write_bytes(b"preview")
        return outputs

    monkeypatch.setattr(module, "write_geometry_preview_plots", fake_write_geometry_preview_plots)
    monkeypatch.setattr(module, "solve_steady", lambda case: SimpleNamespace(case_name=case.name))
    def fake_write_case_overview_plots(solution, out_dir: Path, case_title: str):
        out_dir.mkdir(parents=True, exist_ok=True)
        outputs = [out_dir / "overview.png", out_dir / "overview.pdf"]
        for path in outputs:
            path.write_bytes(b"plot")
        return outputs

    monkeypatch.setattr(module, "write_case_overview_plots", fake_write_case_overview_plots)
    monkeypatch.setattr(module, "validation_summary", lambda solution, case_name, ha=None: {"u_max": 1.0})

    summary = module.run_variable_field_geometry_demo(out_dir=tmp_path)
    assert summary["rectangular_case"]["magnetic_field_kind"] == "analytic"
    assert summary["layered_case"]["geometry_kind"] == "layered_duct"
    assert (tmp_path / "variable_field_geometry_summary.json").exists()


def test_build_case_rejects_unknown_kind():
    with pytest.raises(ValueError, match="Unsupported case kind"):
        example_runner._build_case("bad", 5.0, 8, 8)


def test_portable_path_prefers_relative_and_falls_back_to_name(tmp_path: Path):
    nested = tmp_path / "nested" / "file.json"
    nested.parent.mkdir()
    nested.write_text("{}")

    assert example_runner._portable_path(nested, relative_to=tmp_path) == "nested/file.json"
    assert example_runner._portable_path("/outside/path/file.json", relative_to=tmp_path) == "file.json"


def test_solve_case_snapshots_records_fully_developed_frames(monkeypatch: pytest.MonkeyPatch):
    case = example_runner._build_case("hartmann", 5.0, 6, 6)

    def fake_step(**kwargs):
        u_prev = kwargs["u_previous"]
        updated = np.asarray(u_prev) + 0.1
        zeros = np.zeros_like(updated)
        return (
            updated,
            zeros,
            zeros,
            zeros,
            zeros,
            1.0e-6,
            1.0e-6,
            2,
            1.0e-6,
            3,
            0.2,
            0.1,
            0.05,
            float(np.mean(updated)),
            0.3,
            1.0e-4,
            1.0e-4,
        )

    monkeypatch.setattr(example_runner.solvers, "_fully_developed_case_step", fake_step)
    frames = example_runner.solve_case_snapshots(case, frame_count=3)
    assert len(frames) >= 3
    assert frames[-1]["time"] > frames[0]["time"]
    assert "pressure_proxy" in frames[0]
    assert "face_lorentz_max" in frames[0]


def test_run_case_example_cli_prints_report(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path):
    monkeypatch.setattr(
        example_runner,
        "run_case_example",
        lambda **kwargs: {"case": "hartmann_ha5", "plots": [], "metrics": {"u_max": 1.0}, "output_dir": str(tmp_path)},
    )
    exit_code = example_runner.run_case_example_cli(
        case_kind="hartmann",
        ha=5.0,
        ny=8,
        nz=8,
        out_dir=tmp_path,
        reference_root=None,
    )
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["case"] == "hartmann_ha5"


def test_run_case_example_uses_reference_data_when_available(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fake_solution = SimpleNamespace(
        state=SimpleNamespace(u=np.array([[1.0]]), phi=np.array([[0.0]]), time=0.0, residual=0.0),
        mesh=SimpleNamespace(),
        case_name="shercliff_ha5",
    )
    reference_root = tmp_path / "refs"
    reference_root.mkdir()

    monkeypatch.setattr(example_runner, "solve_steady", lambda case: fake_solution)
    monkeypatch.setattr(example_runner, "write_paraview", lambda solution, out_dir: [])
    monkeypatch.setattr(example_runner, "write_profile_csv", lambda path, profile: path)
    monkeypatch.setattr(example_runner, "extract_centerline", lambda solution: {"y": [0.0], "u": [1.0]})
    monkeypatch.setattr(example_runner, "extract_midplane_profile", lambda solution, axis, fluid_only=True: {"coord": [0.0], "u": [1.0]})
    monkeypatch.setattr(example_runner, "validation_summary", lambda solution, case_name, ha: {"u_max": 1.0})
    monkeypatch.setattr(
        example_runner,
        "closed_channel_validation",
        lambda solution, case_name, ha, reference_root: SimpleNamespace(
            y_profile=SimpleNamespace(coordinate=np.array([0.0]), reference=np.array([1.0]), l2_error=0.1),
            z_profile=SimpleNamespace(coordinate=np.array([0.0]), reference=np.array([0.9]), l2_error=0.2),
            reference_path=Path("analytical.csv"),
        ),
    )
    monkeypatch.setattr(example_runner, "write_case_overview_plots", lambda solution, out_dir, **kwargs: [out_dir / "overview.png"])
    monkeypatch.setattr(example_runner, "write_metrics_json", lambda payload, path: path.write_text("{}"))

    report = run_case_example(
        case_kind="shercliff",
        ha=5.0,
        ny=8,
        nz=8,
        out_dir=tmp_path,
        reference_root=reference_root,
    )

    assert report["reference"]["available"] is True
    assert report["reference"]["kind"] == "closed_channel_analytical"


def test_run_case_example_handles_missing_reference_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fake_solution = SimpleNamespace(
        state=SimpleNamespace(u=np.array([[1.0]]), phi=np.array([[0.0]]), time=0.0, residual=0.0),
        mesh=SimpleNamespace(),
        case_name="hunt_ha5",
    )
    reference_root = tmp_path / "refs"
    reference_root.mkdir()

    monkeypatch.setattr(example_runner, "solve_steady", lambda case: fake_solution)
    monkeypatch.setattr(example_runner, "write_paraview", lambda solution, out_dir: [])
    monkeypatch.setattr(example_runner, "write_profile_csv", lambda path, profile: path)
    monkeypatch.setattr(example_runner, "extract_centerline", lambda solution: {"y": [0.0], "u": [1.0]})
    monkeypatch.setattr(example_runner, "extract_midplane_profile", lambda solution, axis, fluid_only=True: {"coord": [0.0], "u": [1.0]})
    monkeypatch.setattr(example_runner, "validation_summary", lambda solution, case_name, ha: {"u_max": 1.0})
    monkeypatch.setattr(
        example_runner,
        "closed_channel_validation",
        lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError("missing")),
    )
    monkeypatch.setattr(example_runner, "write_case_overview_plots", lambda solution, out_dir, **kwargs: [out_dir / "overview.png"])
    monkeypatch.setattr(example_runner, "write_metrics_json", lambda payload, path: path.write_text("{}"))

    report = run_case_example(
        case_kind="hunt",
        ha=5.0,
        ny=8,
        nz=8,
        out_dir=tmp_path,
        reference_root=reference_root,
    )

    assert report["reference"]["available"] is False


def test_run_theory_meeting_demo_rejects_unknown_movie_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(example_runner, "run_case_example", lambda **kwargs: {"case": "demo", "plots": [], "metrics": {}})
    monkeypatch.setattr(example_runner, "make_hunt_case", lambda **kwargs: SimpleNamespace(name="hunt_ha5"))
    monkeypatch.setattr(
        example_runner,
        "solve_steady",
        lambda case: SimpleNamespace(
            state=SimpleNamespace(u=np.array([[1.0]]), phi=np.array([[0.0]]), time=0.0, residual=0.0),
            mesh=SimpleNamespace(),
            case_name="hunt_ha5",
        ),
    )
    monkeypatch.setattr(example_runner, "write_paraview", lambda solution, out_dir: [])
    monkeypatch.setattr(example_runner, "write_profile_csv", lambda path, profile: path)
    monkeypatch.setattr(example_runner, "extract_centerline", lambda solution: {"y": [0.0], "u": [1.0]})
    monkeypatch.setattr(
        example_runner,
        "extract_midplane_profile",
        lambda solution, axis, fluid_only=True: {"coord": [0.0], "u": [1.0]},
    )
    monkeypatch.setattr(example_runner, "validation_summary", lambda solution, case_name, ha: {"u_max": 1.0})
    monkeypatch.setattr(example_runner, "write_case_overview_plots", lambda solution, out_dir, **kwargs: [])
    monkeypatch.setattr(example_runner, "write_metrics_json", lambda payload, path: path)

    with pytest.raises(ValueError, match="Unsupported movie_case"):
        run_theory_meeting_demo(out_dir=tmp_path, movie_case="bad")

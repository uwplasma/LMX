from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import lmx.showcase as showcase


pytestmark = pytest.mark.unit


def _fake_mesh():
    y_faces = np.linspace(-1.0, 1.0, 7)
    z_faces = np.linspace(-1.0, 1.0, 7)
    y_centers = 0.5 * (y_faces[:-1] + y_faces[1:])
    z_centers = 0.5 * (z_faces[:-1] + z_faces[1:])
    fluid_mask = np.ones((y_centers.size, z_centers.size), dtype=bool)
    return SimpleNamespace(
        y_faces=y_faces,
        z_faces=z_faces,
        y_centers=y_centers,
        z_centers=z_centers,
        ny=y_centers.size,
        nz=z_centers.size,
        fluid_mask=fluid_mask,
    )


def _fake_solution():
    mesh = _fake_mesh()
    yy, zz = np.meshgrid(mesh.y_centers, mesh.z_centers, indexing="ij")
    u = 1.0 - 0.25 * yy**2 - 0.15 * zz**2
    return SimpleNamespace(mesh=mesh, state=SimpleNamespace(u=u))


def test_showcase_geometry_and_mesh_writers_emit_outputs(tmp_path: Path):
    outputs = showcase.write_lm_duct_geometry_setup_figure(tmp_path)
    assert (tmp_path / "lm_duct_geometry_setup.png") in outputs
    assert (tmp_path / "lm_duct_geometry_setup.png").exists()
    assert (tmp_path / "lm_duct_geometry_setup.pdf").exists()

    mesh_outputs = showcase.write_structured_mesh_figure(_fake_mesh(), tmp_path, nx=12, length=1.0)
    assert (tmp_path / "structured_mesh_ha20.png") in mesh_outputs
    assert (tmp_path / "structured_mesh_ha20.png").exists()
    assert (tmp_path / "structured_mesh_ha20.pdf").exists()


def test_showcase_solution_plot_writers_emit_outputs(tmp_path: Path):
    solution = _fake_solution()

    boundary_outputs = showcase.write_boundary_layer_figure(solution, tmp_path, title="Boundary Layer Development - Shercliff")
    annotated_outputs = showcase.write_annotated_layer_figure(
        solution,
        tmp_path,
        title="U Magnitude - Shercliff flow",
        case_kind="shercliff",
        ha=20.0,
        half_width=0.1,
    )
    volume_outputs = showcase.write_velocity_profile_volume_figure(
        solution,
        tmp_path,
        title="Liquid Metal Velocity Profile - Shercliff",
        case_kind="shercliff",
    )

    assert (tmp_path / "boundary_layer_development.png") in boundary_outputs
    assert (tmp_path / "annotated_layers.png") in annotated_outputs
    assert (tmp_path / "velocity_profile_volume.png") in volume_outputs
    assert (tmp_path / "boundary_layer_development.png").exists()
    assert (tmp_path / "annotated_layers.png").exists()
    assert (tmp_path / "velocity_profile_volume.png").exists()


def test_showcase_profile_comparison_writer_uses_reference_helpers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    analytical = SimpleNamespace(
        coordinate=np.linspace(-0.1, 0.1, 5),
        midplane_z=np.linspace(0.0, 1.0, 5),
    )
    monkeypatch.setattr(showcase, "load_shercliff_analytical", lambda ha: analytical)
    monkeypatch.setattr(showcase, "load_hunt_analytical", lambda ha: analytical)
    monkeypatch.setattr(
        showcase,
        "extract_midplane_profile",
        lambda solution, axis="z", fluid_only=True: {"z": np.linspace(-0.1, 0.1, 5), "u": np.linspace(0.0, 1.0, 5)},
    )

    outputs = showcase.write_closed_channel_profile_comparison_figure(
        tmp_path,
        shercliff_solution=_fake_solution(),
        hunt_solution=_fake_solution(),
        ha=20.0,
    )

    assert (tmp_path / "analytic_velocity_profiles.png") in outputs
    assert (tmp_path / "analytic_velocity_profiles.png").exists()
    assert (tmp_path / "analytic_velocity_profiles.pdf").exists()


def test_write_closed_channel_startup_movies_dispatches_to_movie_writer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(showcase, "solve_case_snapshots", lambda case, frame_count=1: [{"time": 0.0, "u": np.ones((2, 2)), "mesh": _fake_mesh(), "fluid_mask": np.ones((2, 2), dtype=bool)}])

    def fake_write_transient_movies(frames, out_dir, *, case_title: str, output_stem: str, fps: int, symmetry_average_axes=()):
        outputs = [out_dir / f"{output_stem}_2d.gif", out_dir / f"{output_stem}_3d.gif"]
        for path in outputs:
            path.write_bytes(b"gif")
        return outputs

    monkeypatch.setattr(showcase, "write_transient_movies", fake_write_transient_movies)

    outputs = showcase.write_closed_channel_startup_movies("shercliff", tmp_path, ny=8, nz=8, dt=1.0e-4, t_final=2.0e-4)

    assert (tmp_path / "shercliff_startup_2d.gif") in outputs
    assert (tmp_path / "shercliff_startup_2d.gif").exists()
    assert (tmp_path / "shercliff_startup_3d.gif").exists()


def test_write_closed_channel_startup_movies_rejects_unsupported_case(tmp_path: Path):
    with pytest.raises(ValueError):
        showcase.write_closed_channel_startup_movies("bad_case", tmp_path)

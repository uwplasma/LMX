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


def _comparison(size=5, *, l2_error=1.0e-3, linf_error=2.0e-3):
    return SimpleNamespace(
        coordinate=np.linspace(-1.0, 1.0, size),
        simulated=np.linspace(0.0, 1.0, size),
        reference=np.linspace(0.0, 1.0, size),
        l2_error=l2_error, linf_error=linf_error,
    )


@pytest.fixture
def closed_channel_stubs(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    def solve(case, mesh=None, initial_state=None):
        captured.update(case=case, mesh=mesh, initial_state=initial_state)
        return "solution"

    monkeypatch.setattr(showcase, "solve_steady", solve)
    monkeypatch.setattr(
        showcase, "closed_channel_validation", lambda *args, **kwargs: {"ok": True}
    )
    return captured


def test_showcase_profile_comparison_writer_uses_reference_helpers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    comparison = _comparison()
    validation = SimpleNamespace(
        y_profile=comparison,
        z_profile=comparison,
        reference_path="/tmp/reference.csv",
    )
    monkeypatch.setattr(
        showcase, "hartmann_validation", lambda solution, ha: comparison
    )
    monkeypatch.setattr(
        showcase,
        "closed_channel_validation",
        lambda solution, case_kind, ha: validation,
    )

    outputs = showcase.write_closed_channel_profile_comparison_figure(
        tmp_path,
        hartmann_solution=_fake_solution(),
        shercliff_solution=_fake_solution(),
        hunt_solution=_fake_solution(),
        ha=20.0,
    )

    assert (tmp_path / "analytic_velocity_profiles.png") in outputs
    assert (tmp_path / "analytic_velocity_profiles.png").exists()
    assert (tmp_path / "analytic_velocity_profiles.pdf").exists()


def test_showcase_validation_ladder_writer_emits_outputs(tmp_path: Path):
    comparison = _comparison()
    record = {
        "ha": 20.0,
        "y_profile": comparison,
        "z_profile": comparison,
        "reference_path": "/tmp/ref.csv",
    }

    outputs = showcase.write_closed_channel_validation_ladder_figure(
        tmp_path,
        shercliff_records=[record],
        hunt_records=[record],
    )

    assert (tmp_path / "closed_channel_validation_ladder.png") in outputs
    assert (tmp_path / "closed_channel_validation_ladder.png").exists()
    assert (tmp_path / "closed_channel_validation_ladder.pdf").exists()


def test_showcase_hartmann_validation_ladder_writer_emits_outputs(tmp_path: Path):
    comparison = _comparison(7, l2_error=8.0e-3, linf_error=1.4e-2)
    records = [
        {"ha": 20.0, "comparison": comparison},
        {"ha": 100.0, "comparison": comparison},
    ]

    outputs = showcase.write_hartmann_validation_ladder_figure(
        tmp_path,
        hartmann_records=records,
    )

    assert (tmp_path / "hartmann_validation_ladder.png") in outputs
    assert (tmp_path / "hartmann_validation_ladder.png").exists()
    assert (tmp_path / "hartmann_validation_ladder.pdf").exists()


def test_write_closed_channel_startup_movies_dispatches_to_movie_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    captured = {}

    def fake_solve_case_snapshots(case, frame_count=1):
        captured["solver_mode"] = case.solver.mode
        captured["current_reconstruction"] = case.time_stepper.current_reconstruction
        captured["frame_count"] = frame_count
        return [
            {
                "time": 0.0,
                "u": np.ones((2, 2)),
                "mesh": _fake_mesh(),
                "fluid_mask": np.ones((2, 2), dtype=bool),
            }
        ]

    monkeypatch.setattr(showcase, "solve_case_snapshots", fake_solve_case_snapshots)

    def fake_write_transient_movies(
        frames,
        out_dir,
        *,
        case_title: str,
        output_stem: str,
        fps: int,
        include_3d: bool,
        symmetry_average_axes=(),
    ):
        captured["include_3d"] = include_3d
        outputs = [out_dir / f"{output_stem}_2d.gif"]
        if include_3d:
            outputs.append(out_dir / f"{output_stem}_3d.gif")
        for path in outputs:
            path.write_bytes(b"gif")
        return outputs

    monkeypatch.setattr(showcase, "write_transient_movies", fake_write_transient_movies)

    outputs = showcase.write_closed_channel_startup_movies(
        "shercliff", tmp_path, ny=8, nz=8, dt=1.0e-4, t_final=2.0e-4,
        include_3d=False,
    )

    assert captured["solver_mode"] == "transient"
    assert captured["current_reconstruction"] == "face_averaged"
    assert (captured["frame_count"], captured["include_3d"]) == (2, False)
    assert (tmp_path / "shercliff_startup_2d.gif") in outputs
    assert (tmp_path / "shercliff_startup_2d.gif").exists()
    assert not (tmp_path / "shercliff_startup_3d.gif").exists()


def test_write_closed_channel_startup_movies_rejects_unsupported_case(tmp_path: Path):
    with pytest.raises(ValueError):
        showcase.write_closed_channel_startup_movies("bad_case", tmp_path)


def test_solve_closed_channel_benchmark_supports_flow_rate_and_zero_initial_profile(
    monkeypatch: pytest.MonkeyPatch, closed_channel_stubs,
):
    fake_case = showcase.make_shercliff_case(ha=20.0, ny=8, nz=8, width=0.3, height=0.4)
    monkeypatch.setattr(showcase, "make_shercliff_case", lambda **kwargs: fake_case)

    case, solution, comparison = showcase.solve_closed_channel_benchmark(
        "shercliff",
        drive_mode="flow_rate",
        target_mean_velocity=0.2,
        initial_profile="zero",
        width=0.3,
        height=0.4,
        linear_solver="solvax_pcg",
    )

    assert solution == "solution"
    assert comparison == {"ok": True}
    assert closed_channel_stubs["initial_state"] is None
    assert case.forcing == pytest.approx(0.0)
    assert case.initial_velocity == pytest.approx(0.2)
    assert case.solver.linear_solver == "solvax_pcg"
    inlet = case.boundary_conditions[-1]
    assert inlet.kind == "inlet_flow_rate"
    assert inlet.value == pytest.approx(0.2 * 0.3 * 0.4)


def test_solve_closed_channel_benchmark_forwards_reference_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, closed_channel_stubs,
):
    fake_case = showcase.make_shercliff_case(ha=20.0, ny=8, nz=8)
    monkeypatch.setattr(showcase, "make_shercliff_case", lambda **kwargs: fake_case)

    def fake_validation(solution, case_kind, ha, **kwargs):
        closed_channel_stubs["validation_kwargs"] = kwargs
        return {"ok": True}

    monkeypatch.setattr(showcase, "closed_channel_validation", fake_validation)
    showcase.solve_closed_channel_benchmark(
        "shercliff",
        drive_mode="flow_rate",
        initial_profile="zero",
        reference_root=tmp_path,
    )
    assert closed_channel_stubs["validation_kwargs"] == {"reference_root": tmp_path}


def test_solve_closed_channel_benchmark_supports_reference_pressure_gradient(
    monkeypatch: pytest.MonkeyPatch, closed_channel_stubs,
):
    fake_case = showcase.make_shercliff_case(ha=20.0, ny=8, nz=8, width=0.3, height=0.4)
    reference = SimpleNamespace(
        coordinate=np.linspace(-1.0, 1.0, 5),
        midplane_y=np.linspace(0.2, 1.0, 5),
        midplane_z=np.linspace(1.0, 0.2, 5),
        pressure_drop=123.0,
    )
    monkeypatch.setattr(showcase, "make_shercliff_case", lambda **kwargs: fake_case)
    monkeypatch.setattr(showcase, "load_shercliff_analytical", lambda ha: reference)
    monkeypatch.setattr(showcase, "_build_mesh", lambda case: _fake_mesh())

    case, solution, comparison = showcase.solve_closed_channel_benchmark(
        "shercliff", drive_mode="pressure_gradient"
    )

    assert solution == "solution"
    assert comparison == {"ok": True}
    assert case.forcing == pytest.approx(123.0)
    assert closed_channel_stubs["initial_state"] is not None


def test_solve_closed_channel_benchmark_builds_hunt_analytic_initial_state(
    monkeypatch: pytest.MonkeyPatch, closed_channel_stubs,
):
    fake_case = showcase.make_hunt_case(ha=20.0, ny=8, nz=8)
    mesh = _fake_mesh()
    mesh = SimpleNamespace(
        **{**vars(mesh), "fluid_mask": np.tri(mesh.ny, mesh.nz, dtype=bool)}
    )
    reference = SimpleNamespace(
        coordinate=np.linspace(-1.0, 1.0, 5),
        midplane_y=np.linspace(0.2, 1.0, 5),
        midplane_z=np.linspace(1.0, 0.2, 5),
    )
    monkeypatch.setattr(showcase, "make_hunt_case", lambda **kwargs: fake_case)
    monkeypatch.setattr(showcase, "load_hunt_analytical", lambda ha: reference)
    monkeypatch.setattr(showcase, "_build_mesh", lambda case: mesh)

    showcase.solve_closed_channel_benchmark("hunt", initial_profile="analytic")

    initial_state = closed_channel_stubs["initial_state"]
    assert initial_state is not None
    assert np.isfinite(np.asarray(initial_state.u)).all()
    assert np.asarray(initial_state.u).shape == mesh.fluid_mask.shape
    assert np.all(np.asarray(initial_state.u)[~mesh.fluid_mask] == 0.0)


def test_solve_closed_channel_benchmark_passes_custom_mesh_to_solver(
    monkeypatch: pytest.MonkeyPatch, closed_channel_stubs,
):
    fake_case = showcase.make_hunt_case(ha=20.0, ny=8, nz=8)
    custom_mesh = _fake_mesh()
    reference = SimpleNamespace(
        coordinate=np.linspace(-1.0, 1.0, 5),
        midplane_y=np.linspace(0.2, 1.0, 5),
        midplane_z=np.linspace(1.0, 0.2, 5),
    )
    monkeypatch.setattr(showcase, "make_hunt_case", lambda **kwargs: fake_case)
    monkeypatch.setattr(showcase, "load_hunt_analytical", lambda ha: reference)

    case, solution, comparison = showcase.solve_closed_channel_benchmark(
        "hunt", mesh=custom_mesh
    )

    assert solution == "solution"
    assert comparison == {"ok": True}
    assert case.reference_phi_cell == (custom_mesh.ny // 2, custom_mesh.nz // 2)
    assert closed_channel_stubs["mesh"] is custom_mesh
    assert closed_channel_stubs["initial_state"].u.shape == (custom_mesh.ny, custom_mesh.nz)


def test_solve_closed_channel_benchmark_rejects_invalid_modes_and_profiles():
    with pytest.raises(ValueError, match="Unsupported case kind"):
        showcase.solve_closed_channel_benchmark("bad_case")
    with pytest.raises(
        ValueError, match="Unsupported closed-channel benchmark drive mode"
    ):
        showcase.solve_closed_channel_benchmark("shercliff", drive_mode="bad_drive")
    with pytest.raises(
        ValueError, match="Unsupported closed-channel benchmark initial profile"
    ):
        showcase.solve_closed_channel_benchmark(
            "shercliff", initial_profile="bad_profile"
        )

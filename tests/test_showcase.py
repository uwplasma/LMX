from dataclasses import replace
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
        captured["potential_solver"] = case.time_stepper.potential_solver
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
    assert captured["potential_solver"] == "cg_volume"
    assert (captured["frame_count"], captured["include_3d"]) == (2, False)
    assert (tmp_path / "shercliff_startup_2d.gif") in outputs
    assert (tmp_path / "shercliff_startup_2d.gif").exists()
    assert not (tmp_path / "shercliff_startup_3d.gif").exists()


def test_solve_case_snapshots_records_fully_developed_frames(
    monkeypatch: pytest.MonkeyPatch,
):
    case = showcase.make_shercliff_case(ha=5.0, ny=6, nz=6)
    case = replace(
        case,
        time_stepper=replace(case.time_stepper, t_final=0.003, max_steps=3),
    )

    calls = []

    def fake_step(**kwargs):
        calls.append(None)
        update = 0.1 if len(calls) == 1 else 1.0e-10
        updated = np.asarray(kwargs["u_previous"]) + update
        zeros = np.zeros_like(updated)
        residual = 1.0e-6 if len(calls) == 1 else 1.0e-10
        return (
            updated,
            zeros,
            zeros,
            zeros,
            zeros,
            residual,
            residual,
            2,
            residual,
            3,
            0.2,
            0.1,
            0.05,
            float(np.mean(updated)),
            0.3,
            1.0e-4,
            1.0e-4,
        )

    monkeypatch.setattr(showcase.solvers, "_fully_developed_case_step", fake_step)
    frames = showcase.solve_case_snapshots(case, frame_count=3)
    assert len(frames) == 3
    assert frames[-1]["time"] > frames[0]["time"]
    assert {
        "pressure_proxy",
        "face_lorentz_max",
        "linear_residual",
        "coupling_residual",
    } <= frames[0].keys()
    assert (
        frames[-1]["step"],
        frames[-1]["steady_streak"],
        frames[-1]["converged"],
    ) == (2, 1, True)

    calls.clear()
    capped = replace(case, time_stepper=replace(case.time_stepper, max_steps=1))
    with pytest.raises(RuntimeError, match="steady-state ceiling"):
        showcase.solve_case_snapshots(capped, frame_count=1)


def test_write_closed_channel_startup_movies_rejects_unsupported_case(tmp_path: Path):
    with pytest.raises(ValueError):
        showcase.write_closed_channel_startup_movies("bad_case", tmp_path)

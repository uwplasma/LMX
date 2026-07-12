from pathlib import Path
from types import SimpleNamespace

import json
from dataclasses import dataclass

import numpy as np
import pytest

from scripts import run_benchmark_b_quantitative as suite


pytestmark = pytest.mark.unit


def test_run_benchmark_b_quantitative_writes_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    def fake_build_problem(
        geometry_kind: str,
        *,
        ha_peak: float,
        ny: int,
        nz: int,
        nx_stations: int,
        max_steps: int,
        coupling_iterations: int,
        potential_iterations: int,
    ):
        return SimpleNamespace(
            geometry_kind=geometry_kind,
            ha_peak=ha_peak,
            ny=ny,
            nz=nz,
            nx_stations=nx_stations,
            max_steps=max_steps,
            coupling_iterations=coupling_iterations,
            potential_iterations=potential_iterations,
        )

    def fake_solve(problem):
        validation = SimpleNamespace(
            station_count=problem.nx_stations,
            max_residual=1.0e-4,
            max_charge_balance_residual=1.0e-3,
            volumetric_flow_rate_span=2.0e-3,
            axial_current_span=3.0e-3,
            peak_velocity_span=4.0e-3,
            pressure_span_range=5.0e-3,
            max_wall_current_leakage=0.0,
            net_boundary_current_residual=0.0,
            field_mean_velocity_correlation=-0.8,
        )
        bundle = SimpleNamespace()
        return SimpleNamespace(validation=validation, bundle=bundle)

    monkeypatch.setattr(suite, "_build_problem", fake_build_problem)
    monkeypatch.setattr(suite, "solve_extruded_inductionless", fake_solve)
    monkeypatch.setattr(
        suite,
        "_pipe_profile_errors",
        lambda bundle, reference_dir: {
            "center_velocity_l2_error": 0.1,
            "negative_potential_l2_error": 0.2,
            "positive_potential_l2_error": 0.3,
        },
    )

    exit_code = suite.main(
        ["--output", str(tmp_path / "benchmark_b"), "--include-pipe-reference"]
    )

    payload = json.loads(
        (tmp_path / "benchmark_b" / "benchmark_b_quantitative_summary.json").read_text()
    )
    markdown = (
        tmp_path / "benchmark_b" / "benchmark_b_quantitative_summary.md"
    ).read_text()
    csv_text = (
        tmp_path / "benchmark_b" / "benchmark_b_quantitative_summary.csv"
    ).read_text()

    assert exit_code == 0
    assert len(payload) == 3
    assert "pipe_ogrid" in markdown
    assert "center_velocity_l2_error" in csv_text
    assert "negative_potential_l2_error" in csv_text
    assert (tmp_path / "benchmark_b" / "benchmark_b_quantitative_summary.png").exists()


def test_run_benchmark_b_quantitative_can_limit_geometries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    built: list[str] = []

    def fake_build_problem(
        geometry_kind: str,
        *,
        ha_peak: float,
        ny: int,
        nz: int,
        nx_stations: int,
        max_steps: int,
        coupling_iterations: int,
        potential_iterations: int,
    ):
        built.append(geometry_kind)
        return SimpleNamespace(
            geometry_kind=geometry_kind,
            nx_stations=nx_stations,
        )

    def fake_solve(problem):
        validation = SimpleNamespace(
            station_count=problem.nx_stations,
            max_residual=1.0e-4,
            max_charge_balance_residual=1.0e-3,
            volumetric_flow_rate_span=2.0e-3,
            axial_current_span=3.0e-3,
            peak_velocity_span=4.0e-3,
            pressure_span_range=5.0e-3,
            max_wall_current_leakage=0.0,
            net_boundary_current_residual=0.0,
            field_mean_velocity_correlation=-0.8,
        )
        return SimpleNamespace(validation=validation, bundle=SimpleNamespace())

    monkeypatch.setattr(suite, "_build_problem", fake_build_problem)
    monkeypatch.setattr(suite, "solve_extruded_inductionless", fake_solve)

    exit_code = suite.main(
        [
            "--output",
            str(tmp_path / "benchmark_b_rect"),
            "--geometries",
            "rect_duct",
            "layered_duct",
        ]
    )

    payload = json.loads(
        (
            tmp_path / "benchmark_b_rect" / "benchmark_b_quantitative_summary.json"
        ).read_text()
    )
    assert exit_code == 0
    assert built == ["rect_duct", "layered_duct"]
    assert [row["geometry_kind"] for row in payload] == ["layered_duct", "rect_duct"]


def test_pipe_reference_root_and_replace_fields_cover_dataclass_and_object_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    assert suite._pipe_reference_root(tmp_path) == tmp_path
    monkeypatch.setattr(
        suite, "default_fringing_pipe_reference_root", lambda: tmp_path / "default"
    )
    assert suite._pipe_reference_root(None) == tmp_path / "default"

    @dataclass
    class Demo:
        value: int

    replaced = suite._replace_fields(Demo(1), value=2)
    assert replaced.value == 2

    obj = SimpleNamespace(value=1)
    assert suite._replace_fields(obj, value=3).value == 3


def test_pipe_profile_errors_cover_velocity_and_potential_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    pot_path = tmp_path / "potential.csv"
    rows = np.zeros((5, 14))
    rows[:, 13] = np.linspace(-1.0, 1.0, 5)
    np.savetxt(
        pot_path,
        rows,
        delimiter=",",
        header=",".join(f"c{i}" for i in range(14)),
        comments="",
    )

    references = {
        "center": SimpleNamespace(
            x_offset_fraction=0.0,
            coordinate=np.linspace(-1.0, 1.0, 5),
            velocity=np.linspace(0.0, 1.0, 5),
            path=pot_path,
        ),
        "negative": SimpleNamespace(
            x_offset_fraction=-0.5,
            coordinate=np.linspace(-1.0, 1.0, 5),
            velocity=np.zeros(5),
            path=pot_path,
        ),
        "positive": SimpleNamespace(
            x_offset_fraction=0.5,
            coordinate=np.linspace(-1.0, 1.0, 5),
            velocity=np.zeros(5),
            path=pot_path,
        ),
    }

    monkeypatch.setattr(
        suite, "load_fringing_pipe_profile", lambda name, root: references[name]
    )

    def fake_extract(bundle, *, x_offset_fraction, field_name):
        coord = np.linspace(-1.0, 1.0, 5)
        if field_name == "u":
            return coord, np.linspace(0.0, 2.0, 5)
        return coord, np.linspace(-2.0, 2.0, 5)

    monkeypatch.setattr(suite, "_extract_pipe_profile", fake_extract)

    errors = suite._pipe_profile_errors(SimpleNamespace(), tmp_path)

    assert "center_velocity_l2_error" in errors
    assert "negative_potential_l2_error" in errors
    assert "positive_potential_linf_error" in errors


def test_build_problem_rejects_unknown_geometry():
    with pytest.raises(ValueError, match="Unsupported geometry"):
        suite._build_problem(
            "bad_geometry",
            ha_peak=20.0,
            ny=8,
            nz=8,
            nx_stations=5,
            max_steps=4,
            coupling_iterations=2,
            potential_iterations=8,
        )


def test_build_problem_covers_rect_layered_and_pipe_branches(
    monkeypatch: pytest.MonkeyPatch,
):
    base_problem = lambda: SimpleNamespace(
        case=SimpleNamespace(
            solver=SimpleNamespace(coupling_iterations=1),
            time_stepper=SimpleNamespace(max_steps=2, potential_iterations=3),
        )
    )
    monkeypatch.setattr(
        suite, "build_square_duct_extruded_problem", lambda **kwargs: base_problem()
    )
    monkeypatch.setattr(
        suite, "build_layered_duct_extruded_problem", lambda **kwargs: base_problem()
    )
    monkeypatch.setattr(
        suite, "build_pipe_ogrid_extruded_problem", lambda **kwargs: base_problem()
    )

    rect = suite._build_problem(
        "rect_duct",
        ha_peak=20.0,
        ny=8,
        nz=10,
        nx_stations=5,
        max_steps=7,
        coupling_iterations=4,
        potential_iterations=9,
    )
    layered = suite._build_problem(
        "layered_duct",
        ha_peak=20.0,
        ny=8,
        nz=10,
        nx_stations=5,
        max_steps=7,
        coupling_iterations=4,
        potential_iterations=9,
    )
    pipe = suite._build_problem(
        "pipe_ogrid",
        ha_peak=20.0,
        ny=8,
        nz=10,
        nx_stations=5,
        max_steps=7,
        coupling_iterations=4,
        potential_iterations=9,
    )

    assert rect.case.solver.coupling_iterations == 4
    assert layered.case.time_stepper.max_steps == 7
    assert pipe.case.time_stepper.potential_iterations == 9


def test_row_for_solution_uses_optional_validation_fields():
    validation = SimpleNamespace(
        station_count=5,
        max_residual=1.0e-3,
        max_charge_balance_residual=2.0e-3,
        volumetric_flow_rate_span=3.0e-3,
        axial_current_span=4.0e-3,
        peak_velocity_span=5.0e-3,
        pressure_span_range=6.0e-3,
        max_wall_current_leakage=0.0,
        net_boundary_current_residual=0.0,
        field_mean_velocity_correlation=-0.9,
    )
    row = suite._row_for_solution(
        "rect_duct",
        SimpleNamespace(validation=validation),
        ha_peak=20.0,
        ny=8,
        nz=8,
        nx_stations=5,
    )
    assert row["axial_current_mirror_residual"] == pytest.approx(0.0)
    assert row["pressure_span_mirror_residual"] == pytest.approx(0.0)


def test_benchmark_b_writers_emit_expected_artifacts(tmp_path: Path):
    rows = [
        {
            "geometry_kind": "rect_duct",
            "ha_peak": 20.0,
            "cross_section_y": 8.0,
            "cross_section_z": 8.0,
            "nx_stations": 5.0,
            "max_charge_balance_residual": 1.0e-3,
            "volumetric_flow_rate_span": 2.0e-3,
            "axial_current_span": 3.0e-3,
            "axial_current_mirror_residual": 4.0e-3,
            "pressure_span_range": 5.0e-3,
            "pressure_span_mirror_residual": 6.0e-3,
            "center_axial_current": 0.0,
            "center_pressure_span": 0.0,
            "field_mean_velocity_correlation": -0.8,
        }
    ]
    csv_path = suite._write_csv(rows, tmp_path / "summary.csv")
    md_path = suite._write_markdown(rows, tmp_path / "summary.md")
    plots = suite._write_plot(rows, tmp_path / "summary.png")

    assert csv_path.exists()
    assert md_path.exists()
    assert "Axial mirror" in md_path.read_text()
    assert len(plots) == 2
    assert all(path.exists() for path in plots)

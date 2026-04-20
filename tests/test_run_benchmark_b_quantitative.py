from pathlib import Path
from types import SimpleNamespace

import json
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

    exit_code = suite.main(["--output", str(tmp_path / "benchmark_b"), "--include-pipe-reference"])

    payload = json.loads((tmp_path / "benchmark_b" / "benchmark_b_quantitative_summary.json").read_text())
    markdown = (tmp_path / "benchmark_b" / "benchmark_b_quantitative_summary.md").read_text()
    csv_text = (tmp_path / "benchmark_b" / "benchmark_b_quantitative_summary.csv").read_text()

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

    payload = json.loads((tmp_path / "benchmark_b_rect" / "benchmark_b_quantitative_summary.json").read_text())
    assert exit_code == 0
    assert built == ["rect_duct", "layered_duct"]
    assert [row["geometry_kind"] for row in payload] == ["layered_duct", "rect_duct"]

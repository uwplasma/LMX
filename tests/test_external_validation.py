from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from lmx.external_validation import (
    compare_magnetic_obstacle_reference_observables,
    compare_scalar_reference_observables,
    dean_vortex_reference_template_rows,
    external_validation_readiness_rows,
    load_q2dmhdfoam_force_coefficients,
    load_q2dmhdfoam_docker_reference_profile,
    load_q2dmhdfoam_lid_driven_cell_field,
    load_q2dmhdfoam_lid_driven_observables,
    load_q2dmhdfoam_line_profile,
    load_q2dmhdfoam_probe_velocity_history,
    load_q2dmhdfoam_vtk_vector_field,
    load_magnetic_obstacle_reference_observables,
    load_scalar_reference_observables,
    magnetic_obstacle_reference_template_rows,
    q2dmhdfoam_docker_reference_observables,
    q2dmhdfoam_cell_velocity_observables,
    q2dmhdfoam_profile_observables,
    q2dmhdfoam_vtk_velocity_observables,
    q2d_turbulence_reference_template_rows,
    summarize_external_validation_readiness,
    write_dean_vortex_reference_template,
    write_external_validation_readiness_panel,
    write_magnetic_obstacle_reference_comparison_plots,
    write_magnetic_obstacle_reference_comparison_table,
    write_magnetic_obstacle_reference_template,
    write_q2dmhdfoam_docker_reference_panel,
    write_q2dmhdfoam_external_reference_panel,
    write_q2dmhdfoam_profile_observable_table,
    write_q2dmhdfoam_timeseries_observable_table,
    write_q2dmhdfoam_vtk_velocity_panel,
    write_q2d_turbulence_reference_template,
    write_scalar_reference_comparison_plots,
    write_scalar_reference_comparison_table,
)


pytestmark = pytest.mark.unit


def test_magnetic_obstacle_reference_observable_csv_round_trip(tmp_path: Path):
    path = tmp_path / "reference.csv"
    path.write_text(
        "observable,value,tolerance,relative_tolerance,units,source\n"
        "centerline_velocity_deficit_ratio,0.25,0.02,0.10,dimensionless,Votyakov digitized figure\n"
        "pressure_drop_proxy,1.20,0.05,,dimensionless,Cuevas digitized figure\n",
        encoding="utf-8",
    )

    reference = load_magnetic_obstacle_reference_observables(path)

    assert reference["centerline_velocity_deficit_ratio"]["value"] == pytest.approx(0.25)
    assert reference["centerline_velocity_deficit_ratio"]["relative_tolerance"] == pytest.approx(0.10)
    assert reference["pressure_drop_proxy"]["source"] == "Cuevas digitized figure"


def test_compare_magnetic_obstacle_reference_observables_writes_publication_table(tmp_path: Path):
    reference = {
        "centerline_velocity_deficit_ratio": {"value": 0.25, "tolerance": 0.02, "relative_tolerance": 0.10},
        "wake_recovery_ratio": {"value": 0.92, "tolerance": 0.02},
        "pressure_drop_proxy": {"value": 1.2, "tolerance": 0.05, "units": "dimensionless"},
    }
    lmx_observables = {
        "centerline_velocity_deficit_ratio": 0.26,
        "wake_recovery_ratio": 0.91,
        "pressure_drop_proxy": 1.31,
        "current_proxy_peak": 4.0,
    }

    comparison = compare_magnetic_obstacle_reference_observables(lmx_observables, reference)
    table_path = write_magnetic_obstacle_reference_comparison_table(comparison, tmp_path / "comparison.csv")

    assert comparison["compared_observable_count"] == 3
    assert comparison["passed_observable_count"] == 2
    assert comparison["validation_pass"] is False
    assert "current_proxy_peak" in comparison["extra_lmx_observables"]
    assert table_path.read_text(encoding="utf-8").startswith("observable,lmx_value,reference_value")


def test_write_magnetic_obstacle_reference_comparison_plots(tmp_path: Path):
    comparison = compare_magnetic_obstacle_reference_observables(
        {
            "centerline_velocity_deficit_ratio": 0.26,
            "wake_recovery_ratio": 0.91,
        },
        {
            "centerline_velocity_deficit_ratio": {"value": 0.25, "tolerance": 0.02},
            "wake_recovery_ratio": {"value": 0.95, "tolerance": 0.02},
        },
    )

    paths = write_magnetic_obstacle_reference_comparison_plots(comparison, tmp_path)

    assert [path.suffix for path in paths] == [".png", ".pdf"]
    assert all(path.exists() for path in paths)
    assert all(path.stat().st_size > 0 for path in paths)


def test_compare_magnetic_obstacle_reference_observables_reports_missing_lmx_observable():
    comparison = compare_magnetic_obstacle_reference_observables(
        {"centerline_velocity_deficit_ratio": 0.2},
        {
            "centerline_velocity_deficit_ratio": {"value": 0.2, "tolerance": 0.01},
            "wake_recovery_ratio": {"value": 0.9, "tolerance": 0.02},
        },
    )

    assert comparison["missing_lmx_observable_count"] == 1
    assert comparison["validation_pass"] is False
    assert comparison["rows"][1]["status"] == "missing_lmx_observable"


def test_magnetic_obstacle_reference_template_has_required_columns(tmp_path: Path):
    template_rows = magnetic_obstacle_reference_template_rows()
    template_path = write_magnetic_obstacle_reference_template(tmp_path / "template.csv")
    text = template_path.read_text(encoding="utf-8")

    assert {"observable", "value", "tolerance", "relative_tolerance"} <= set(template_rows[0])
    assert "centerline_velocity_deficit_ratio" in text
    assert "pressure_drop_proxy" in text


def test_magnetic_obstacle_reference_csv_rejects_missing_columns(tmp_path: Path):
    path = tmp_path / "bad.csv"
    path.write_text("observable,value\npressure_drop_proxy,1.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required columns"):
        load_magnetic_obstacle_reference_observables(path)


def test_scalar_reference_helpers_cover_q2d_and_dean_templates(tmp_path: Path):
    q2d_path = write_q2d_turbulence_reference_template(tmp_path / "q2d.csv")
    dean_path = write_dean_vortex_reference_template(tmp_path / "dean.csv")

    assert "final_spectral_centroid" in q2d_path.read_text(encoding="utf-8")
    assert "secondary_flow_rms_ratio" in dean_path.read_text(encoding="utf-8")
    assert any(row["observable"] == "turnover_count" for row in q2d_turbulence_reference_template_rows())
    assert any(row["observable"] == "inner_outer_velocity_ratio" for row in dean_vortex_reference_template_rows())


def test_q2dmhdfoam_line_profile_observables_and_panel(tmp_path: Path):
    profile_path = tmp_path / "lineSampled_theta_Ux_250_500_1e6"
    x = np.linspace(0.0, 0.15, 21)
    theta = np.zeros_like(x)
    velocity = 1.0 - 0.35 * ((x - 0.075) / 0.075) ** 2
    np.savetxt(profile_path, np.column_stack([x, theta, velocity]))

    profile = load_q2dmhdfoam_line_profile(profile_path)
    observables = q2dmhdfoam_profile_observables(profile)
    table = write_q2dmhdfoam_profile_observable_table([observables], tmp_path / "profiles.csv")
    paths = write_q2dmhdfoam_external_reference_panel(
        [profile],
        [observables],
        tmp_path,
        turbulence_observables={"weak_mode_count": 2, "weak_peak_over_max_max": 0.2},
    )

    assert profile["hartmann"] == pytest.approx(250.0)
    assert profile["reynolds"] == pytest.approx(500.0)
    assert profile["grashof"] == pytest.approx(1.0e6)
    assert observables["symmetry_l2"] < 1.0e-12
    assert observables["peak_to_mean_velocity"] > 1.0
    assert "peak_to_mean_velocity" in table.read_text(encoding="utf-8")
    assert [path.suffix for path in paths] == [".png", ".pdf"]
    assert all(path.exists() for path in paths)


def test_q2dmhdfoam_lid_driven_summary_parser(tmp_path: Path):
    summary_path = tmp_path / "IDM_output_U.txt"
    summary_path.write_text(
        "-1.4\n"
        "Weak turbulence:[[1.0, 0.15], [2.0, 0.10]]\n"
        "Strong turbulence:[[0.5, 0.12, 0.04]].\n",
        encoding="utf-8",
    )

    observables = load_q2dmhdfoam_lid_driven_observables(summary_path)

    assert observables["weak_mode_count"] == 2
    assert observables["weak_dominant_wavenumber"] == pytest.approx(1.0)
    assert observables["strong_mode_count"] == 1
    assert observables["strong_avg_over_max_max"] == pytest.approx(0.04)


def test_q2dmhdfoam_force_and_probe_history_parsers(tmp_path: Path):
    force_path = tmp_path / "forceCoeffs.dat"
    force_path.write_text(
        "# Time Cd Cl Cm\n"
        "0 2.0 0.4 -1.0\n"
        "1 2.2 0.2 -1.1\n"
        "2 2.4 0.0 -1.2\n"
        "3 2.6 -0.2 -1.3\n",
        encoding="utf-8",
    )
    probe_path = tmp_path / "U"
    probe_path.write_text(
        "# Probe 0 (0 0 0)\n"
        "0 (1 0 0) (0 1 0)\n"
        "1 (2 0 0) (0 2 0)\n"
        "2 (3 0 0) (0 3 0)\n",
        encoding="utf-8",
    )

    force = load_q2dmhdfoam_force_coefficients(force_path, tail_fraction=0.5)
    probe = load_q2dmhdfoam_probe_velocity_history(probe_path)
    table = write_q2dmhdfoam_timeseries_observable_table([force, probe], tmp_path / "timeseries.csv")

    assert force["cd_tail_mean"] == pytest.approx(2.5)
    assert force["cl_tail_mean"] == pytest.approx(-0.1)
    assert probe["probe_count"] == 2
    assert probe["speed_peak"] == pytest.approx(3.0)
    assert "cd_tail_mean" in table.read_text(encoding="utf-8")


def test_q2dmhdfoam_docker_reference_profile_and_panel(tmp_path: Path):
    profile_path = tmp_path / "profile.csv"
    profile_path.write_text(
        "y,y_over_b,ux,ux_over_mean,ux_over_peak\n"
        "-1.0,-1.0,0.0,0.0,0.0\n"
        "-0.5,-0.5,1.0,1.0,0.5\n"
        "0.0,0.0,2.0,2.0,1.0\n"
        "0.5,0.5,1.0,1.0,0.5\n"
        "1.0,1.0,0.0,0.0,0.0\n",
        encoding="utf-8",
    )
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "case": "Q2DmhdFoam/Q2DfullyDeveloped",
                "status": "external_reference_case_complete",
                "hartmann": 50.0,
                "rank_count": 2,
                "cell_count": 20,
                "flow_rate_relative_error": 1.0e-8,
            }
        ),
        encoding="utf-8",
    )

    profile = load_q2dmhdfoam_docker_reference_profile(profile_path, summary_path)
    observables = q2dmhdfoam_docker_reference_observables(profile)
    paths = write_q2dmhdfoam_docker_reference_panel(profile, observables, tmp_path)

    assert profile["hartmann"] == pytest.approx(50.0)
    assert observables["steady_state_reached"] is True
    assert observables["flow_rate_relative_error"] == pytest.approx(1.0e-8)
    assert [path.suffix for path in paths] == [".png", ".pdf"]
    assert all(path.exists() for path in paths)


def test_q2dmhdfoam_vtk_vector_field_and_panel(tmp_path: Path):
    vtk_path = tmp_path / "sample.vtk"
    vtk_path.write_text(
        "# vtk DataFile Version 2.0\n"
        "sample\n"
        "ASCII\n"
        "DATASET UNSTRUCTURED_GRID\n"
        "POINTS 4 float\n"
        "0 0 0 1 0 0 0 1 0 1 1 0\n"
        "POINT_DATA 4\n"
        "FIELD attributes 2\n"
        "vorticity 3 4 float\n"
        "0 0 1 0 0 2 0 0 3 0 0 4\n"
        "U 3 4 float\n"
        "0 0 0 1 0 0 0 1 0 1 1 0\n",
        encoding="utf-8",
    )

    field = load_q2dmhdfoam_vtk_vector_field(vtk_path)
    observables = q2dmhdfoam_vtk_velocity_observables(field)
    paths = write_q2dmhdfoam_vtk_velocity_panel(field, observables, tmp_path)

    assert field["point_count"] == 4
    assert observables["speed_max"] == pytest.approx(np.sqrt(2.0))
    assert observables["vorticity_peak"] == pytest.approx(4.0)
    assert [path.suffix for path in paths] == [".png", ".pdf"]
    assert all(path.exists() for path in paths)


def test_q2dmhdfoam_lid_driven_cell_field_observables(tmp_path: Path):
    case_dir = tmp_path / "case"
    poly_mesh = case_dir / "constant" / "polyMesh"
    time_dir = case_dir / "1.0"
    poly_mesh.mkdir(parents=True)
    time_dir.mkdir(parents=True)
    (poly_mesh / "blockMeshDict").write_text(
        "\n".join(
            [
                "x 2.0;",
                "y 1.0;",
                "yNeg -1.0;",
                "yBL 0.5;",
                "yNegBL -0.5;",
                "Nx 2;",
                "Ny 1;",
                "NyBL 1;",
                "Gy 1;",
                "GyInv 1;",
                "GyBL 1;",
                "GyBLinv 1;",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (time_dir / "U").write_text(
        "internalField nonuniform List<vector>\n"
        "8\n"
        "(\n"
        "(1 0 0)\n"
        "(2 0 0)\n"
        "(3 0 0)\n"
        "(4 0 0)\n"
        "(0 1 0)\n"
        "(0 2 0)\n"
        "(0 3 0)\n"
        "(0 4 0)\n"
        ")\n"
        ";\n",
        encoding="utf-8",
    )
    (time_dir / "vorticity").write_text(
        "internalField nonuniform List<vector>\n"
        "8\n"
        "(\n"
        "(0 0 1)\n"
        "(0 0 2)\n"
        "(0 0 3)\n"
        "(0 0 4)\n"
        "(0 0 5)\n"
        "(0 0 6)\n"
        "(0 0 7)\n"
        "(0 0 8)\n"
        ")\n"
        ";\n",
        encoding="utf-8",
    )

    field = load_q2dmhdfoam_lid_driven_cell_field(case_dir)
    observables = q2dmhdfoam_cell_velocity_observables(field)

    assert field["vectors"].shape == (4, 2, 3)
    assert field["arrays"]["vorticity"].shape == (4, 2, 3)
    assert field["x_width"] == pytest.approx(1.0)
    assert np.asarray(field["y_widths"]).sum() == pytest.approx(2.0)
    assert observables["sample_count"] == 8
    assert observables["speed_mean"] == pytest.approx(2.5)
    assert observables["speed_rms"] == pytest.approx(np.sqrt(7.5))
    assert observables["ux_mean"] == pytest.approx(1.25)
    assert observables["uy_mean"] == pytest.approx(1.25)
    assert observables["vorticity_peak"] == pytest.approx(8.0)
    assert observables["reference_gate"] == "q2dmhdfoam_cell_field_observables"


def test_generic_scalar_reference_comparison_writes_table_and_plots(tmp_path: Path):
    reference_path = tmp_path / "reference.csv"
    reference_path.write_text(
        "observable,value,tolerance,relative_tolerance,units,source\n"
        "energy_decay_ratio,0.75,0.01,0.05,dimensionless,Sommeria-Moreau digitized figure\n"
        "final_spectral_centroid,7.2,0.1,,1/m,Reference spectrum\n",
        encoding="utf-8",
    )
    reference = load_scalar_reference_observables(reference_path, context="Q2D turbulence reference CSV")
    comparison = compare_scalar_reference_observables(
        {
            "energy_decay_ratio": 0.76,
            "final_spectral_centroid": 7.6,
            "turnover_count": 0.4,
        },
        reference,
    )
    table_path = write_scalar_reference_comparison_table(comparison, tmp_path / "comparison.csv")
    plot_paths = write_scalar_reference_comparison_plots(
        comparison,
        tmp_path,
        output_stem="comparison",
        title="Scalar external-reference observables",
        no_data_label="No compared observables",
    )

    assert comparison["compared_observable_count"] == 2
    assert comparison["passed_observable_count"] == 1
    assert comparison["validation_pass"] is False
    assert "turnover_count" in comparison["extra_lmx_observables"]
    assert table_path.exists()
    assert [path.suffix for path in plot_paths] == [".png", ".pdf"]


def test_external_validation_readiness_rows_cover_open_lanes():
    rows = external_validation_readiness_rows()
    summary = summarize_external_validation_readiness(rows)
    lane_names = {str(row["lane"]) for row in rows}

    assert "Q2D turbulence parity" in lane_names
    assert "Magnetic-obstacle validation" in lane_names
    assert "Dean-vortex bent-pipe parity" in lane_names
    assert summary["lane_count"] == len(rows)
    assert summary["research_grade_validation_pass"] is False
    assert summary["runnable_or_data_lane_count"] >= 4


def test_write_external_validation_readiness_panel(tmp_path: Path):
    paths = write_external_validation_readiness_panel(external_validation_readiness_rows(), tmp_path)

    assert [path.suffix for path in paths] == [".png"]
    assert all(path.exists() for path in paths)
    assert all(path.stat().st_size > 0 for path in paths)

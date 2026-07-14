from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import lmx.external_validation as ev
from lmx.external_validation import (
    compare_magnetic_obstacle_reference_observables,
    compare_scalar_reference_observables,
    dean_vortex_reference_template_rows,
    external_validation_readiness_rows,
    audit_q2dmhdfoam_lmx_turbulence_match,
    load_q2dmhdfoam_force_coefficients,
    load_q2dmhdfoam_docker_reference_profile,
    load_q2dmhdfoam_lid_driven_cell_field,
    load_q2dmhdfoam_lid_driven_observables,
    load_q2dmhdfoam_line_profile,
    load_q2dmhdfoam_probe_velocity_history,
    load_q2dmhdfoam_vtk_vector_field,
    load_magnetic_obstacle_reference_observables,
    load_magnetic_obstacle_votyakov_digitized_curve,
    load_scalar_reference_observables,
    magnetic_obstacle_reference_template_rows,
    magnetic_obstacle_votyakov_curve_observables,
    q2dmhdfoam_case_manifest,
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
    write_magnetic_obstacle_votyakov_curve_comparison,
    write_q2dmhdfoam_docker_reference_panel,
    write_q2dmhdfoam_external_reference_panel,
    write_q2dmhdfoam_lmx_turbulence_match_audit,
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

    assert reference["centerline_velocity_deficit_ratio"]["value"] == pytest.approx(
        0.25
    )
    assert reference["centerline_velocity_deficit_ratio"][
        "relative_tolerance"
    ] == pytest.approx(0.10)
    assert reference["pressure_drop_proxy"]["source"] == "Cuevas digitized figure"


def test_compare_magnetic_obstacle_reference_observables_writes_publication_table(
    tmp_path: Path,
):
    reference = {
        "centerline_velocity_deficit_ratio": {
            "value": 0.25,
            "tolerance": 0.02,
            "relative_tolerance": 0.10,
        },
        "wake_recovery_ratio": {"value": 0.92, "tolerance": 0.02},
        "pressure_drop_proxy": {
            "value": 1.2,
            "tolerance": 0.05,
            "units": "dimensionless",
        },
    }
    lmx_observables = {
        "centerline_velocity_deficit_ratio": 0.26,
        "wake_recovery_ratio": 0.91,
        "pressure_drop_proxy": 1.31,
        "current_proxy_peak": 4.0,
    }

    comparison = compare_magnetic_obstacle_reference_observables(
        lmx_observables, reference
    )
    table_path = write_magnetic_obstacle_reference_comparison_table(
        comparison, tmp_path / "comparison.csv"
    )

    assert comparison["compared_observable_count"] == 3
    assert comparison["passed_observable_count"] == 2
    assert comparison["validation_pass"] is False
    assert "current_proxy_peak" in comparison["extra_lmx_observables"]
    assert table_path.read_text(encoding="utf-8").startswith(
        "observable,lmx_value,reference_value"
    )


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
    template_path = write_magnetic_obstacle_reference_template(
        tmp_path / "template.csv"
    )
    text = template_path.read_text(encoding="utf-8")

    assert {"observable", "value", "tolerance", "relative_tolerance"} <= set(
        template_rows[0]
    )
    assert "centerline_velocity_deficit_ratio" in text
    assert "pressure_drop_proxy" in text


def test_magnetic_obstacle_votyakov_curve_loader_and_panel(tmp_path: Path):
    curve_path = tmp_path / "votyakov.csv"
    curve_path.write_text(
        "series,N,ux_min\n"
        "experiment_Ha140,4.0,0.08\n"
        "experiment_Ha140,5.0,0.0\n"
        "experiment_Ha140,16.0,-0.12\n"
        "experiment_Ha140,25.0,-0.14\n"
        "simulation_Re100,4.0,0.1\n"
        "simulation_Re100,6.0,-0.03\n"
        "simulation_Re100,16.0,-0.11\n",
        encoding="utf-8",
    )

    records = load_magnetic_obstacle_votyakov_digitized_curve(curve_path)
    observables = magnetic_obstacle_votyakov_curve_observables(records)
    paths = write_magnetic_obstacle_votyakov_curve_comparison(
        records,
        {
            "minimum_centerline_velocity_ratio": 0.99,
            "pressure_drop_proxy": 0.4,
            "max_charge_balance_residual": 1.0e-12,
        },
        tmp_path,
    )

    experiment = next(row for row in observables if row["series"] == "experiment_Ha140")
    assert len(records) == 7
    assert experiment["reverse_flow_onset_interaction_parameter"] == pytest.approx(5.0)
    assert experiment["plateau_minimum_centerline_velocity_ratio"] == pytest.approx(
        -0.13
    )
    assert [path.suffix for path in paths] == [".csv", ".png", ".pdf"]
    assert all(path.exists() and path.stat().st_size > 0 for path in paths)


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
    assert any(
        row["observable"] == "turnover_count"
        for row in q2d_turbulence_reference_template_rows()
    )
    assert any(
        row["observable"] == "inner_outer_velocity_ratio"
        for row in dean_vortex_reference_template_rows()
    )


def test_q2dmhdfoam_line_profile_observables_and_panel(tmp_path: Path):
    profile_path = tmp_path / "lineSampled_theta_Ux_250_500_1e6"
    x = np.linspace(0.0, 0.15, 21)
    theta = np.zeros_like(x)
    velocity = 1.0 - 0.35 * ((x - 0.075) / 0.075) ** 2
    np.savetxt(profile_path, np.column_stack([x, theta, velocity]))

    profile = load_q2dmhdfoam_line_profile(profile_path)
    observables = q2dmhdfoam_profile_observables(profile)
    table = write_q2dmhdfoam_profile_observable_table(
        [observables], tmp_path / "profiles.csv"
    )
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
        "# Probe 0 (0 0 0)\n0 (1 0 0) (0 1 0)\n1 (2 0 0) (0 2 0)\n2 (3 0 0) (0 3 0)\n",
        encoding="utf-8",
    )

    force = load_q2dmhdfoam_force_coefficients(force_path, tail_fraction=0.5)
    probe = load_q2dmhdfoam_probe_velocity_history(probe_path)
    table = write_q2dmhdfoam_timeseries_observable_table(
        [force, probe], tmp_path / "timeseries.csv"
    )

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


def test_q2dmhdfoam_case_manifest_and_match_audit(tmp_path: Path):
    case_dir = _write_fake_q2dmhdfoam_case(tmp_path / "muck_q2d_FFT")

    manifest = q2dmhdfoam_case_manifest(case_dir)
    audit = audit_q2dmhdfoam_lmx_turbulence_match(case_dir)
    paths = write_q2dmhdfoam_lmx_turbulence_match_audit([audit], tmp_path / "audit")

    assert manifest["application"] == "Q2DmhdFoam"
    assert manifest["has_cylinder_obstacle"] is True
    assert manifest["has_inlet_outlet"] is True
    assert manifest["has_empty_hartmann_walls"] is True
    assert manifest["hartmann_friction_nonzero"] is False
    assert manifest["probe_count"] == 2
    assert manifest["total_cell_count"] == 12
    assert audit["strict_admissible"] is False
    assert audit["decision"] == "not_admissible_for_strict_csv"
    assert {"topology", "hartmann_friction", "forcing", "observables"} <= set(
        audit["blockers"]
    )
    assert [path.suffix for path in paths] == [".json", ".csv", ".png", ".pdf"]
    assert all(path.exists() and path.stat().st_size > 0 for path in paths)
    assert "not_admissible_for_strict_csv" in (
        tmp_path / "audit" / "q2dmhdfoam_lmx_turbulence_match_audit.csv"
    ).read_text(encoding="utf-8")


def test_generic_scalar_reference_comparison_writes_table_and_plots(tmp_path: Path):
    reference_path = tmp_path / "reference.csv"
    reference_path.write_text(
        "observable,value,tolerance,relative_tolerance,units,source\n"
        "energy_decay_ratio,0.75,0.01,0.05,dimensionless,Sommeria-Moreau digitized figure\n"
        "final_spectral_centroid,7.2,0.1,,1/m,Reference spectrum\n",
        encoding="utf-8",
    )
    reference = load_scalar_reference_observables(
        reference_path, context="Q2D turbulence reference CSV"
    )
    comparison = compare_scalar_reference_observables(
        {
            "energy_decay_ratio": 0.76,
            "final_spectral_centroid": 7.6,
            "turnover_count": 0.4,
        },
        reference,
    )
    table_path = write_scalar_reference_comparison_table(
        comparison, tmp_path / "comparison.csv"
    )
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
    paths = write_external_validation_readiness_panel(
        external_validation_readiness_rows(), tmp_path
    )

    assert [path.suffix for path in paths] == [".png"]
    assert all(path.exists() for path in paths)
    assert all(path.stat().st_size > 0 for path in paths)


def test_scalar_reference_loader_rejects_malformed_rows(tmp_path: Path):
    missing = tmp_path / "missing.csv"
    missing.write_text("observable,value\na,1\n")
    with pytest.raises(ValueError, match="missing required columns"):
        ev.load_scalar_reference_observables(missing)

    cases = [
        ("observable,value,tolerance\n,1,1\n", "empty observable"),
        ("observable,value,tolerance\na,1,-1\n", "negative tolerance"),
        (
            "observable,value,tolerance,relative_tolerance\na,1,0,-1\n",
            "negative relative_tolerance",
        ),
        ("observable,value,tolerance\na,,1\n", "empty value"),
        ("observable,value,tolerance\na,nan,1\n", "non-finite value"),
    ]
    for index, (text, message) in enumerate(cases):
        path = tmp_path / f"bad-{index}.csv"
        path.write_text(text)
        with pytest.raises(ValueError, match=message):
            ev.load_scalar_reference_observables(path)


def test_scalar_reference_metadata_missing_and_extra_observables(tmp_path: Path):
    path = tmp_path / "reference.csv"
    path.write_text(
        "observable,value,tolerance,relative_tolerance,units,source\n"
        "matched,0,0,0.1,m/s,paper\n"
        "missing,2,0.1,,m/s,paper\n"
    )
    reference = ev.load_scalar_reference_observables(path)
    comparison = ev.compare_scalar_reference_observables(
        {"matched": 1.0e-22, "extra": 3.0}, reference
    )
    assert comparison["missing_lmx_observable_count"] == 1
    assert comparison["extra_lmx_observables"] == ["extra"]
    assert comparison["validation_pass"] is False
    assert reference["matched"]["units"] == "m/s"

    table = ev.write_scalar_reference_comparison_table(
        comparison, tmp_path / "table.csv"
    )
    assert table.exists()


def test_scalar_reference_plots_cover_empty_infinite_and_missing_annotations(
    tmp_path: Path,
):
    empty = {"rows": [], "missing_lmx_observable_count": 2}
    paths = ev.write_scalar_reference_comparison_plots(
        empty,
        tmp_path / "empty",
        output_stem="empty",
        title="Empty",
        no_data_label="No comparison data",
    )
    assert all(path.exists() for path in paths)

    comparison = {
        "missing_lmx_observable_count": 1,
        "rows": [
            {
                "observable": "a_very_long_observable_name_for_wrapping",
                "status": "compared",
                "lmx_value": 2.0,
                "reference_value": 1.0,
                "absolute_error": 1.0,
                "effective_tolerance": 0.0,
                "validation_pass": False,
            }
        ],
    }
    paths = ev.write_scalar_reference_comparison_plots(
        comparison,
        tmp_path / "infinite",
        output_stem="comparison",
        title="Comparison",
        no_data_label="unused",
    )
    assert all(path.exists() for path in paths)


def test_readiness_and_openfoam_label_helpers_cover_all_outcomes(tmp_path: Path):
    with pytest.raises(ValueError, match="At least one"):
        ev.write_external_validation_readiness_panel([], tmp_path)

    flags = {
        "has_cylinder_obstacle": True,
        "has_inlet_outlet": True,
        "has_side_walls": True,
        "has_cyclic_patch": True,
        "has_empty_hartmann_walls": True,
    }
    assert (
        ev._topology_label(flags)
        == "cylinder, inlet/outlet, sideWalls, cyclic, empty_hartmannWalls"
    )
    assert ev._topology_label({}) == "no recognized topology flags"
    assert "thermal" in ev._forcing_label({"thermal_forcing_nonzero": True, "q0": 2.0})
    assert "mean-flow" in ev._forcing_label(
        {"forced_mean_flow": True, "ubar_magnitude": 3.0}
    )
    assert ev._forcing_label({}).startswith("no q0")
    assert ev._close_or_missing(None, 1.0) is False
    assert ev._close_or_missing("bad", 1.0) is False
    assert ev._close_or_missing(1.0, 1.0) is True
    assert ev._read_text_if_exists(tmp_path / "missing") == ""

    commented = "/* endTime 9; */ application solver; // deltaT 2;\nendTime 1;"
    assert ev._openfoam_word(commented, "application") == "solver"
    assert ev._openfoam_word(commented, "missing") == ""
    assert ev._openfoam_scalar_assignment(commented, "endTime") == 1.0
    assert ev._openfoam_scalar_assignment(commented, "deltaT") is None


def test_votyakov_curve_loader_error_contracts(tmp_path: Path):
    missing = tmp_path / "missing.csv"
    missing.write_text("series,N\na,1\n")
    with pytest.raises(ValueError, match="missing columns"):
        ev.load_magnetic_obstacle_votyakov_digitized_curve(missing)

    empty_series = tmp_path / "empty-series.csv"
    empty_series.write_text("series,N,ux_min\n,1,0\n")
    with pytest.raises(ValueError, match="empty series"):
        ev.load_magnetic_obstacle_votyakov_digitized_curve(empty_series)

    no_rows = tmp_path / "no-rows.csv"
    no_rows.write_text("series,N,ux_min\n")
    with pytest.raises(ValueError, match="no data rows"):
        ev.load_magnetic_obstacle_votyakov_digitized_curve(no_rows)


def test_small_external_adapter_helpers_cover_validation_branches(tmp_path: Path):
    assert ev._zero_crossing(np.array([0.0, 1.0]), np.array([0.0, 1.0])) == 0.0
    assert ev._zero_crossing(np.array([0.0, 2.0]), np.array([-1.0, 1.0])) == 1.0
    assert np.isnan(ev._zero_crossing(np.array([0.0, 1.0]), np.array([1.0, 2.0])))

    with pytest.raises(ValueError, match="does not contain token"):
        ev._require_token(["POINTS"], "VECTORS", tmp_path / "field.vtk")
    assert (
        ev._require_token(("POINTS", "VECTORS"), "VECTORS", tmp_path / "field.vtk")
        == 1
    )

    with pytest.raises(ValueError, match="point coordinates"):
        ev._point_vector_grid(np.zeros(3), np.zeros((1, 3)))
    with pytest.raises(ValueError, match="point-vector"):
        ev._point_vector_grid(np.zeros((2, 3)), np.zeros((1, 3)))
    points = np.array([[0, 0, 0], [1, 0, 0]])
    with pytest.raises(ValueError, match="2D section"):
        ev._point_vector_grid(points, np.zeros((2, 3)))

    assert ev._q2dmhdfoam_conditions_from_name("unmatched") == {}
    assert ev._q2dmhdfoam_profile_label("plain_name.csv", {}) == "plain name"
    assert ev._parse_q2dmhdfoam_list_payload("nothing", "values") == []
    assert ev._readiness_color(3.0) == "#2a9d8f"
    assert ev._readiness_color(2.0) == "#d97706"
    assert ev._readiness_color(1.0) == "#c2410c"
    assert "\n" in ev._compact_observable_label(
        "a_very_long_observable_name_that_wraps"
    )
    assert "\n" in ev._wrap_text("one two three four", 7)
    assert ev._compact_observable_label("") == ""
    assert ev._wrap_text("", 7) == ""


def test_q2dmhdfoam_profile_loader_and_observable_validation(tmp_path: Path):
    one_column = tmp_path / "one-column.txt"
    one_column.write_text("1\n")
    with pytest.raises(ValueError, match="at least two columns"):
        ev.load_q2dmhdfoam_line_profile(one_column)

    too_short = tmp_path / "too-short.txt"
    too_short.write_text("0 1\n1 nan\n2 3\n")
    with pytest.raises(ValueError, match="fewer than three finite"):
        ev.load_q2dmhdfoam_line_profile(too_short)

    zero_span = tmp_path / "zero-span.txt"
    zero_span.write_text("0 1\n0 2\n0 3\n")
    with pytest.raises(ValueError, match="zero coordinate span"):
        ev.load_q2dmhdfoam_line_profile(zero_span)

    valid = tmp_path / "lineSampled_theta_Ux_10_20_30.txt"
    valid.write_text("-1 0\n0 0\n1 0\n")
    with pytest.raises(ValueError, match="half_width"):
        ev.load_q2dmhdfoam_line_profile(valid, coordinate_half_width=0.0)
    profile = ev.load_q2dmhdfoam_line_profile(valid)
    assert profile["label"] == "Ha=10, Re=20, Gr=30"

    with pytest.raises(ValueError, match="matching 1D"):
        ev.q2dmhdfoam_profile_observables(
            {"position": [[0.0]], "velocity": [0.0]}
        )
    with pytest.raises(ValueError, match="at least three"):
        ev.q2dmhdfoam_profile_observables(
            {"position": [0.0, 1.0], "velocity": [0.0, 1.0]}
        )
    with pytest.raises(ValueError, match="span must be positive"):
        ev.q2dmhdfoam_profile_observables(
            {"position": [0.0, 0.0, 0.0], "velocity": [0.0, 1.0, 2.0]}
        )

    observables = ev.q2dmhdfoam_profile_observables(profile)
    assert observables["mean_velocity"] == 0.0
    assert np.isnan(observables["peak_to_mean_velocity"])
    assert observables["wall_gradient_proxy"] == 0.0
    assert observables["hartmann"] == 10.0


def _write_fake_q2dmhdfoam_case(case_dir: Path) -> Path:
    (case_dir / "system").mkdir(parents=True)
    (case_dir / "constant" / "polyMesh").mkdir(parents=True)
    (case_dir / "0").mkdir(parents=True)
    (case_dir / "system" / "controlDict").write_text(
        "application Q2DmhdFoam;\n"
        "endTime 100;\n"
        "deltaT 10;\n"
        "writeInterval 20;\n"
        "functions { probes { probeLocations ((0 0 0) (1 0 0)); } }\n",
        encoding="utf-8",
    )
    (case_dir / "constant" / "transportProperties").write_text(
        "nu nu [0 2 -1 0 0 0 0] 2.0e-7;\n"
        "rho0 rho0 [1 -3 0 0 0 0 0] 9800;\n"
        "sigma sigma [-1 -3 3 0 0 2 0] 7.8e5;\n"
        "q0 q0 [1 -1 -3 0 0 0 0] 0;\n"
        "a a [0 1 0 0 0 0 0] 1;\n"
        "b b [0 1 0 0 0 0 0] 1;\n"
        "Ubar Ubar [0 1 -1 0 0 0 0] (0 0 0);\n",
        encoding="utf-8",
    )
    (case_dir / "0" / "B").write_text("internalField uniform 0;\n", encoding="utf-8")
    (case_dir / "constant" / "polyMesh" / "blockMeshDict").write_text(
        "vertices ((0 0 0) (1 0 0) (1 1 0) (0 1 0) (0 0 0.1) (1 0 0.1) (1 1 0.1) (0 1 0.1));\n"
        "blocks (hex (0 1 2 3 4 5 6 7) (3 4 1) simpleGrading (1 1 1));\n"
        "patches (\n"
        "patch xinlet ((0 4 7 3))\n"
        "patch xoutlet ((1 2 6 5))\n"
        "patch sideWalls ((0 1 5 4) (3 7 6 2))\n"
        "empty hartmannWalls ((0 3 2 1) (4 5 6 7))\n"
        "patch cylinder ((0 1 2 3))\n"
        ");\n",
        encoding="utf-8",
    )
    return case_dir

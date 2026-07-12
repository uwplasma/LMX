from pathlib import Path

import numpy as np
import pytest

import lmx.external_validation as ev


pytestmark = pytest.mark.unit


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
        ev._require_token(("POINTS", "VECTORS"), "VECTORS", tmp_path / "field.vtk") == 1
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
        ev.q2dmhdfoam_profile_observables({"position": [[0.0]], "velocity": [0.0]})
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

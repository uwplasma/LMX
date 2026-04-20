import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import pytest

import lmx.solvers as solvers
import lmx.validation as validation
from lmx.cases import make_hartmann_case, make_hunt_case, make_shercliff_case
from lmx.core import Diagnostics, MHDState, Solution
from lmx.mesh import generate_rect_duct_mesh
from lmx.solvers import _build_mesh
from lmx.validation import (
    closed_channel_validation,
    compare_normalized_profiles,
    compare_with_reference_outputs,
    combined_profile_error,
    duct_layer_resolution_metrics,
    duct_profile_metrics,
    estimate_observed_order,
    infer_mesh_axis_coordinates,
    infer_mesh_bounds,
    infer_region_conductivity,
    infer_sampling_geometry,
    has_conducting_wall_region,
    interior_sample_coordinate,
    extract_midplane_profile,
    latest_reference_sampled_profiles,
    negative_fraction,
    hartmann_analytic_profile,
    hartmann_validation,
    inspect_reference_case,
    latest_field_minmax_record,
    normalize_sample_distance,
    profile_sign_changes,
    processed_slice_validation,
    read_reference_xy_sample,
    read_field_minmax,
    write_analytic_comparison,
    write_closed_channel_validation,
    write_metrics_json,
    write_processed_slice_validation,
    write_validation_report,
    validation_summary,
)


pytestmark = pytest.mark.validation


@pytest.fixture(autouse=True)
def disable_jit(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(solvers.jax, "jit", lambda fn: fn)


def _synthetic_solution(case, *, oscillatory: bool = False) -> Solution:
    mesh = _build_mesh(case)
    y, z = jnp.meshgrid(mesh.y_centers, mesh.z_centers, indexing="ij")
    profile = 1.0 - 0.25 * y**2 - 0.15 * z**2
    if oscillatory:
        profile = profile * jnp.cos(6.0 * y)
    if mesh.fluid_mask is not None:
        profile = jnp.where(mesh.fluid_mask, profile, 0.0)
    zeros = jnp.zeros_like(profile)
    diagnostics = Diagnostics(
        residual_history=jnp.asarray([1.0e-2, 1.0e-4, 1.0e-6]),
        courant_like=jnp.asarray([0.1, 0.08, 0.06]),
        ohmic_power=jnp.asarray([0.2, 0.15, 0.1]),
        time_history=jnp.asarray([0.0, 0.5, 1.0]),
        u_max_history=jnp.asarray([float(jnp.max(profile))] * 3),
        mean_velocity_history=jnp.asarray([0.5, 0.55, 0.6]),
        applied_forcing_history=jnp.asarray([1.0, 1.0, 1.0]),
        pressure_proxy_history=jnp.asarray([0.2, 0.18, 0.16]),
        current_scaled_pressure_proxy_history=jnp.asarray([0.15, 0.14, 0.13]),
        raw_update_max_history=jnp.asarray([0.05, 0.02, 0.01]),
        limiter_scale_history=jnp.asarray([1.0, 1.0, 1.0]),
        limited_fraction_history=jnp.asarray([0.0, 0.0, 0.0]),
        current_max_history=jnp.asarray([0.3, 0.25, 0.2]),
        face_current_max_history=jnp.asarray([0.28, 0.24, 0.19]),
        emf_max_history=jnp.asarray([0.2, 0.18, 0.15]),
        lorentz_max_history=jnp.asarray([0.12, 0.10, 0.08]),
        potential_residual_history=jnp.asarray([1.0e-3, 1.0e-4, 1.0e-5]),
        potential_iterations_history=jnp.asarray([12.0, 10.0, 8.0]),
        linear_residual_history=jnp.asarray([1.0e-2, 1.0e-4, 1.0e-6]),
        linear_iterations_history=jnp.asarray([8.0, 6.0, 4.0]),
        volumetric_flow_rate_history=jnp.asarray([0.7, 0.75, 0.8]),
        mean_current_magnitude_history=jnp.asarray([0.1, 0.09, 0.08]),
        lorentz_power_history=jnp.asarray([0.05, 0.045, 0.04]),
        div_current_max_history=jnp.asarray([1.0e-6, 8.0e-7, 5.0e-7]),
        charge_balance_residual_history=jnp.asarray([4.0e-8, 3.0e-8, 2.0e-8]),
        gauge_residual_history=jnp.asarray([1.0e-8, 7.0e-9, 5.0e-9]),
        interface_current_residual_history=jnp.asarray([1.0e-6, 8.0e-7, 6.0e-7]),
    )
    return Solution(
        mesh=mesh,
        state=MHDState(
            u=profile,
            phi=0.1 * profile,
            jy=0.05 * profile,
            jz=0.02 * profile,
            lorentz_x=0.03 * profile,
            time=1.0,
            residual=1.0e-6,
        ),
        diagnostics=diagnostics,
        case_name=case.name,
    )


def test_hartmann_profile_center_is_maximum():
    y = jnp.linspace(-1.0, 1.0, 101)
    profile = hartmann_analytic_profile(y, ha=10.0)
    assert float(profile[50]) >= float(profile[0])


def test_combined_profile_error_uses_root_mean_square():
    assert combined_profile_error(3.0, 4.0) == pytest.approx((12.5) ** 0.5)


def test_compare_normalized_profiles_handles_cell_centered_simulation_against_wall_sample():
    simulated_coordinate = jnp.linspace(-0.99, 0.99, 65)
    simulated = 1.0 - simulated_coordinate**2
    reference_coordinate = jnp.linspace(-1.0, 1.0, 201)
    reference = 1.0 - reference_coordinate**2

    comparison = compare_normalized_profiles(
        simulated_coordinate,
        simulated,
        reference_coordinate,
        reference,
    )

    assert comparison.l2_error < 0.02
    assert comparison.linf_error < 0.05


def test_profile_sign_changes_and_negative_fraction_handle_oscillatory_profiles():
    profile = jnp.asarray([0.1, 0.05, -0.02, -0.01, 0.03, 0.02])

    assert profile_sign_changes(profile) == 2
    assert negative_fraction(profile) == pytest.approx(2.0 / 6.0)


def test_duct_layer_resolution_metrics_reports_cells_for_supported_ducts():
    case = make_hunt_case(ha=20.0, ny=16, nz=16, wall_cells=2)
    solution = _synthetic_solution(case)

    metrics = duct_layer_resolution_metrics(case, solution.mesh)

    assert metrics["hartmann_layer_thickness"] > 0.0
    assert metrics["side_layer_thickness"] > 0.0
    assert metrics["hartmann_layer_cells"] > 0.0
    assert metrics["side_layer_cells"] > 0.0


def test_validation_summary_includes_latest_potential_residual():
    case = make_hartmann_case(ha=5.0, ny=12, nz=12)
    solution = _synthetic_solution(case)

    metrics = validation_summary(solution, case.name, ha=5.0)

    assert "potential_residual" in metrics
    assert metrics["potential_residual"] >= 0.0
    assert "potential_iterations_used" in metrics
    assert metrics["potential_iterations_used"] >= 0.0
    assert "current_scaled_pressure_proxy" in metrics
    assert metrics["current_scaled_pressure_proxy"] >= 0.0
    assert "linear_residual" in metrics
    assert metrics["linear_residual"] >= 0.0
    assert "linear_iterations_used" in metrics
    assert metrics["linear_iterations_used"] >= 0.0
    assert "volumetric_flow_rate" in metrics
    assert "mean_current_magnitude" in metrics
    assert "lorentz_power" in metrics
    assert "div_current_max" in metrics
    assert metrics["div_current_max"] >= 0.0
    assert "charge_balance_residual" in metrics
    assert metrics["charge_balance_residual"] >= 0.0
    assert "gauge_residual" in metrics
    assert metrics["gauge_residual"] >= 0.0
    assert "interface_current_residual" in metrics
    assert metrics["interface_current_residual"] >= 0.0
    assert "raw_update_max" in metrics
    assert metrics["raw_update_max"] >= 0.0
    assert "limiter_scale" in metrics
    assert 0.0 <= metrics["limiter_scale"] <= 1.0
    assert "limited_fraction" in metrics
    assert 0.0 <= metrics["limited_fraction"] <= 1.0


def test_compare_with_reference_outputs_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    case = make_hartmann_case()
    monkeypatch.setattr(validation, "solve_transient", lambda case_spec: _synthetic_solution(case_spec))
    (tmp_path / "system").mkdir()
    (tmp_path / "constant").mkdir()
    (tmp_path / "0").mkdir()
    (tmp_path / "0" / "fluid").mkdir()
    (tmp_path / "postProcessing" / "liquid" / "minMax" / "0").mkdir(parents=True)
    (tmp_path / "postProcessing" / "sampleDict" / "liquid" / "0.1").mkdir(parents=True)
    (tmp_path / "system" / "controlDict").write_text("application epotMultiRegionFoam;")
    (tmp_path / "0" / "fluid" / "U").write_text("internalField uniform (0 0 0);")
    (tmp_path / "0" / "fluid" / "potE").write_text("internalField uniform 0;")
    (tmp_path / "postProcessing" / "liquid" / "minMax" / "0" / "fieldMinMax.dat").write_text(
        "# header\n0.1 mag(U) 0.0 (0 0 0) 0 0.25 (0 0 0) 0\n"
    )
    sample_lines = "0.0 0.0 0.0 0.0 0.0\n1.0 0.0 1.0 0.0 0.0\n2.0 0.0 0.0 0.0 0.0\n"
    (tmp_path / "postProcessing" / "sampleDict" / "liquid" / "0.1" / "centerlineY_potE_U.xy").write_text(sample_lines)
    (tmp_path / "postProcessing" / "sampleDict" / "liquid" / "0.1" / "centerlineZ_potE_U.xy").write_text(sample_lines)
    report = compare_with_reference_outputs(case, tmp_path)
    path = write_validation_report(report, tmp_path / "report.json")
    assert path.exists()
    assert report.metrics["control_dict_count"] == pytest.approx(1.0)
    assert report.metrics["region_zero_dir_count"] == pytest.approx(1.0)
    assert report.metrics["has_potE_zero_field"] == pytest.approx(1.0)
    assert report.metrics["field_minmax_file_count"] == pytest.approx(1.0)
    assert report.metrics["reference_u_max_latest"] == pytest.approx(0.25)
    assert report.metrics["sampled_profile_pair_available"] == pytest.approx(1.0)
    assert "reference_sample_y_l2_error" in report.metrics


def test_duct_profile_metrics_reports_sign_pathology():
    case = make_hartmann_case(ha=20.0, ny=32, nz=32)
    solution = _synthetic_solution(case, oscillatory=True)
    metrics = duct_profile_metrics(solution)

    assert "centerline_y_sign_changes" in metrics
    assert "centerline_z_sign_changes" in metrics
    assert "centerline_y_negative_fraction" in metrics
    assert metrics["centerline_y_sign_changes"] > 0.0 or metrics["centerline_y_negative_fraction"] > 0.0


def test_inspect_reference_case_collects_case_structure(tmp_path: Path):
    (tmp_path / "system").mkdir()
    (tmp_path / "constant").mkdir()
    (tmp_path / "0").mkdir()
    (tmp_path / "0" / "liquid").mkdir()
    (tmp_path / "processors8").mkdir()
    (tmp_path / "processors8" / "0.001").mkdir(parents=True)
    (tmp_path / "1.5").mkdir()
    (tmp_path / "system" / "controlDict").write_text("application epotMultiRegionFoam;")
    (tmp_path / "system" / "fvSchemes").write_text("ddtSchemes {}")
    (tmp_path / "system" / "fvSolution").write_text("solvers {}")
    (tmp_path / "system" / "blockMeshDict").write_text("blocks ()")
    (tmp_path / "constant" / "regionProperties").write_text("regions ()")
    (tmp_path / "0" / "liquid" / "U").write_text("internalField uniform (0 0 0);")
    inspection = inspect_reference_case(tmp_path)
    assert inspection.control_dicts == ("system/controlDict",)
    assert inspection.region_properties == ("constant/regionProperties",)
    assert inspection.block_mesh_dicts == ("system/blockMeshDict",)
    assert inspection.latest_time_dirs == ("1.5",)
    assert inspection.region_zero_dirs == ("0/liquid",)
    assert inspection.zero_field_files == ("0/liquid/U",)
    assert inspection.processor_layout_dirs == ("processors8",)
    assert inspection.parallel_time_dirs == ("processors8/0.001",)


def test_field_minmax_reader_extracts_latest_mag_u(tmp_path: Path):
    path = tmp_path / "postProcessing" / "liquid" / "minMax" / "0" / "fieldMinMax.dat"
    path.parent.mkdir(parents=True)
    path.write_text(
        "# header\n"
        "1.0e-05 mag(U) 0.0 (0 0 0) 0 0.8 (0 0 0) 0\n"
        "2.0e-05 mag(U) 0.0 (0 0 0) 0 0.9 (0 0 0) 0\n"
    )
    records = read_field_minmax(path)
    latest = latest_field_minmax_record(tmp_path, field="mag(U)")
    assert len(records) == 2
    assert latest is not None
    assert latest.time == pytest.approx(2.0e-05)
    assert latest.max_value == pytest.approx(0.9)
    assert latest.min_location == pytest.approx((0.0, 0.0, 0.0))
    assert latest.max_location == pytest.approx((0.0, 0.0, 0.0))


def test_infer_sampling_geometry_uses_latest_field_minmax_locations(tmp_path: Path):
    path = tmp_path / "postProcessing" / "liquid" / "minMax" / "0" / "fieldMinMax.dat"
    path.parent.mkdir(parents=True)
    path.write_text(
        "# header\n"
        "1.0e-04 mag(U) 0.0 (0.005 -0.1 -0.099995) 0 0.9734 (0.015 0.0987 0.0987) 0\n"
    )
    geometry = infer_sampling_geometry(tmp_path)
    assert geometry.x_position == pytest.approx(0.015)
    assert geometry.y_min == pytest.approx(-0.1)
    assert geometry.y_max == pytest.approx(0.1)
    assert geometry.z_min == pytest.approx(-0.099995)
    assert geometry.z_max == pytest.approx(0.099995)


def test_infer_sampling_geometry_prefers_mesh_bounds_when_points_exist(tmp_path: Path):
    points_path = tmp_path / "constant" / "liquid" / "polyMesh" / "points"
    points_path.parent.mkdir(parents=True)
    points_path.write_text(
        "FoamFile\n{\n}\n4\n(\n"
        "(0 -0.1 -0.2)\n"
        "(1.2 -0.1 0.2)\n"
        "(0 0.1 -0.2)\n"
        "(1.2 0.1 0.2)\n"
        ")\n"
    )
    minmax_path = tmp_path / "postProcessing" / "liquid" / "minMax" / "0" / "fieldMinMax.dat"
    minmax_path.parent.mkdir(parents=True)
    minmax_path.write_text(
        "# header\n"
        "1.0e-04 mag(U) 0.0 (0.005 -0.1 -0.099995) 0 0.9734 (0.015 0.0987 0.0987) 0\n"
    )

    bounds = infer_mesh_bounds(tmp_path)
    geometry = infer_sampling_geometry(tmp_path)

    assert bounds is not None
    assert bounds[0] == pytest.approx((0.0, 1.2))
    assert bounds[1] == pytest.approx((-0.1, 0.1))
    assert bounds[2] == pytest.approx((-0.2, 0.2))
    assert geometry.x_position == pytest.approx(0.6)
    assert geometry.y_min == pytest.approx(-0.1)
    assert geometry.y_max == pytest.approx(0.1)
    assert geometry.z_min == pytest.approx(-0.2)
    assert geometry.z_max == pytest.approx(0.2)


def test_infer_sampling_geometry_keeps_field_based_x_for_conducting_wall_cases(tmp_path: Path):
    points_path = tmp_path / "constant" / "liquid" / "polyMesh" / "points"
    points_path.parent.mkdir(parents=True)
    points_path.write_text(
        "FoamFile\n{\n}\n4\n(\n"
        "(0 -0.1 -0.1)\n"
        "(1.0 -0.1 0.1)\n"
        "(0 0.1 -0.1)\n"
        "(1.0 0.1 0.1)\n"
        ")\n"
    )
    liquid_props = tmp_path / "constant" / "liquid" / "thermophysicalProperties.liquidMetal"
    liquid_props.parent.mkdir(parents=True, exist_ok=True)
    liquid_props.write_text("elcond [-1 -3  3 0 0 2 0]1e6;\n")
    wall_props = tmp_path / "constant" / "solidWalls" / "thermophysicalProperties"
    wall_props.parent.mkdir(parents=True, exist_ok=True)
    wall_props.write_text("elcond 5e6;\n")
    minmax_path = tmp_path / "postProcessing" / "liquid" / "minMax" / "0" / "fieldMinMax.dat"
    minmax_path.parent.mkdir(parents=True)
    minmax_path.write_text(
        "# header\n"
        "1.0e-04 mag(U) 0.0 (0.005 -0.1 -0.099995) 0 0.9734 (0.015 0.0987 0.0987) 0\n"
    )

    assert infer_region_conductivity(tmp_path, "liquid") == pytest.approx(1e6)
    assert infer_region_conductivity(tmp_path, "solidWalls") == pytest.approx(5e6)
    assert has_conducting_wall_region(tmp_path) is True
    geometry = infer_sampling_geometry(tmp_path)
    assert geometry.x_position == pytest.approx(0.015)


def test_infer_mesh_axis_coordinates_and_boundary_interior_selection(tmp_path: Path):
    points_path = tmp_path / "constant" / "liquid" / "polyMesh" / "points"
    points_path.parent.mkdir(parents=True)
    points_path.write_text(
        "FoamFile\n{\n}\n6\n(\n"
        "(0 -0.1 -0.1)\n"
        "(0.015 -0.1 -0.1)\n"
        "(1.0 -0.1 0.1)\n"
        "(0 0.1 -0.1)\n"
        "(0.015 0.1 -0.1)\n"
        "(1.0 0.1 0.1)\n"
        ")\n"
    )
    coordinates = infer_mesh_axis_coordinates(tmp_path, axis="x")
    assert coordinates == pytest.approx((0.0, 0.015, 1.0))
    assert interior_sample_coordinate(coordinates, 0.0) == pytest.approx(0.015)
    assert interior_sample_coordinate(coordinates, 1.0) == pytest.approx(0.015)


def test_infer_sampling_geometry_uses_first_interior_streamwise_plane_for_boundary_cut(tmp_path: Path):
    points_path = tmp_path / "constant" / "liquid" / "polyMesh" / "points"
    points_path.parent.mkdir(parents=True)
    points_path.write_text(
        "FoamFile\n{\n}\n6\n(\n"
        "(0 -0.1 -0.1)\n"
        "(0.015 -0.1 0.1)\n"
        "(1.0 -0.1 0.1)\n"
        "(0 0.1 -0.1)\n"
        "(0.015 0.1 0.1)\n"
        "(1.0 0.1 0.1)\n"
        ")\n"
    )
    liquid_props = tmp_path / "constant" / "liquid" / "thermophysicalProperties.liquidMetal"
    liquid_props.parent.mkdir(parents=True, exist_ok=True)
    liquid_props.write_text("elcond [-1 -3  3 0 0 2 0]1e6;\n")
    wall_props = tmp_path / "constant" / "solidWalls" / "thermophysicalProperties"
    wall_props.parent.mkdir(parents=True, exist_ok=True)
    wall_props.write_text("elcond 5e6;\n")
    minmax_path = tmp_path / "postProcessing" / "liquid" / "minMax" / "0" / "fieldMinMax.dat"
    minmax_path.parent.mkdir(parents=True)
    minmax_path.write_text(
        "# header\n"
        "1.0e-04 mag(U) 0.0 (0.0 -0.1 -0.09975) 0 0.1237 (0.0 0.0 0.09775) 0\n"
    )

    geometry = infer_sampling_geometry(tmp_path)
    assert geometry.x_position == pytest.approx(0.015)


def test_interior_sample_coordinate_uses_midpoint_without_interior_vertices():
    assert interior_sample_coordinate((0.0, 1.0), 0.0) == pytest.approx(0.5)
    assert interior_sample_coordinate((0.0, 1.0), 1.0) == pytest.approx(0.5)


def test_sample_reader_and_latest_profile_detection(tmp_path: Path):
    sample_root = tmp_path / "postProcessing" / "lmxSampleDict" / "liquid" / "0.0001"
    sample_root.mkdir(parents=True)
    rows = "0.0 0.1 1.0 2.0 3.0\n1.0 0.2 4.0 5.0 6.0\n"
    y_path = sample_root / "centerlineY_potE_U.xy"
    z_path = sample_root / "centerlineZ_potE_U.xy"
    y_path.write_text(rows)
    z_path.write_text(rows)
    sample = read_reference_xy_sample(y_path)
    latest = latest_reference_sampled_profiles(tmp_path)
    assert sample.distance.shape[0] == 2
    assert float(sample.u_x[1]) == pytest.approx(4.0)
    assert latest is not None
    assert latest[0].path.endswith("centerlineY_potE_U.xy")
    normalized = normalize_sample_distance(sample.distance)
    assert float(normalized[0]) == pytest.approx(-1.0)
    assert float(normalized[-1]) == pytest.approx(1.0)


def test_extract_midplane_profile_fluid_only_excludes_layer_walls():
    case = make_hunt_case(ha=20.0, ny=12, nz=12, wall_cells=2)
    solution = _synthetic_solution(case)
    full_profile = extract_midplane_profile(solution, axis="z", fluid_only=False)
    fluid_profile = extract_midplane_profile(solution, axis="z", fluid_only=True)

    assert fluid_profile["z"].shape[0] < full_profile["z"].shape[0]
    assert float(full_profile["u"][0]) == pytest.approx(0.0)
    assert float(full_profile["u"][-1]) == pytest.approx(0.0)
    assert float(fluid_profile["z"][0]) > -0.5 * case.geometry.height
    assert float(fluid_profile["z"][-1]) < 0.5 * case.geometry.height


def test_extract_midplane_profile_averages_even_grid_centerlines():
    u = jnp.asarray(
        [
            [0.0, 1.0, 3.0, 0.0],
            [0.0, 2.0, 4.0, 0.0],
            [0.0, 4.0, 6.0, 0.0],
            [0.0, 5.0, 7.0, 0.0],
        ],
        dtype=float,
    )
    phi = 10.0 * u
    centers = jnp.asarray([-0.75, -0.25, 0.25, 0.75], dtype=float)
    solution = SimpleNamespace(
        mesh=SimpleNamespace(y_centers=centers, z_centers=centers, fluid_mask=None),
        state=SimpleNamespace(u=u, phi=phi),
    )

    y_profile = extract_midplane_profile(solution, axis="y", fluid_only=False)
    z_profile = extract_midplane_profile(solution, axis="z", fluid_only=False)

    assert jnp.allclose(y_profile["u"], 0.5 * (u[:, 1] + u[:, 2]))
    assert jnp.allclose(z_profile["u"], 0.5 * (u[1, :] + u[2, :]))
    assert jnp.allclose(y_profile["phi"], 0.5 * (phi[:, 1] + phi[:, 2]))
    assert jnp.allclose(z_profile["phi"], 0.5 * (phi[1, :] + phi[2, :]))


def test_latest_reference_sampled_profiles_prefers_newest_file_when_times_match(tmp_path: Path):
    older_root = tmp_path / "postProcessing" / "lmxAutoSampleDict" / "liquid" / "0.0001"
    newer_root = tmp_path / "postProcessing" / "lmxCiSampleDict" / "liquid" / "0.0001"
    older_root.mkdir(parents=True)
    newer_root.mkdir(parents=True)
    older_rows = "0.0 0.0 1.0 0.0 0.0\n1.0 0.0 2.0 0.0 0.0\n"
    newer_rows = "0.0 0.0 3.0 0.0 0.0\n1.0 0.0 4.0 0.0 0.0\n"
    (older_root / "centerlineY_potE_U.xy").write_text(older_rows)
    (older_root / "centerlineZ_potE_U.xy").write_text(older_rows)
    (newer_root / "centerlineY_potE_U.xy").write_text(newer_rows)
    (newer_root / "centerlineZ_potE_U.xy").write_text(newer_rows)
    os.utime(older_root / "centerlineY_potE_U.xy", ns=(1_000_000_000, 1_000_000_000))
    os.utime(older_root / "centerlineZ_potE_U.xy", ns=(1_000_000_000, 1_000_000_000))
    os.utime(newer_root / "centerlineY_potE_U.xy", ns=(2_000_000_000, 2_000_000_000))
    os.utime(newer_root / "centerlineZ_potE_U.xy", ns=(2_000_000_000, 2_000_000_000))

    latest = latest_reference_sampled_profiles(tmp_path)

    assert latest is not None
    assert "lmxCiSampleDict" in latest[0].path
    assert float(latest[0].u_x[0]) == pytest.approx(3.0)


def test_hartmann_validation_writer(tmp_path: Path):
    case = make_hartmann_case(ha=5.0, ny=16, nz=16)
    solution = _synthetic_solution(case)
    comparison = hartmann_validation(solution, ha=5.0)
    path = write_analytic_comparison(comparison, tmp_path / "analytic.json", axis_name="y")
    assert path.exists()


def test_estimate_observed_order_returns_second_order_for_quadratic_drop():
    order = estimate_observed_order(4.0e-2, 1.0e-2, 0.25, 0.125)
    assert order == pytest.approx(2.0)


def test_estimate_observed_order_returns_none_for_invalid_inputs():
    assert estimate_observed_order(0.0, 1.0e-2, 0.25, 0.125) is None
    assert estimate_observed_order(1.0e-2, 1.0e-3, 0.125, 0.25) is None


def test_duct_profile_metrics_writer(tmp_path: Path):
    case = make_hartmann_case(ha=5.0, ny=16, nz=16)
    solution = _synthetic_solution(case, oscillatory=True)
    metrics = duct_profile_metrics(solution)
    path = write_metrics_json(metrics, tmp_path / "metrics.json")
    assert path.exists()


def test_closed_channel_validation_writer(tmp_path: Path):
    analytical_root = tmp_path / "ClosedChannel" / "AnalyticalSolutions"
    analytical_root.mkdir(parents=True)
    (analytical_root / "Shercliff_Analytical_Ha2_PresDrop1.0.txt").write_text(
        "r\tu1\tu2\n-1.0\t0.0\t0.0\n0.0\t1.0\t1.0\n1.0\t0.0\t0.0\n"
    )
    case = make_shercliff_case(ha=2.0, ny=12, nz=12)
    solution = _synthetic_solution(case)
    comparison = closed_channel_validation(solution, "shercliff", 2, reference_root=tmp_path / "ClosedChannel")
    path = write_closed_channel_validation(comparison, tmp_path / "closed_channel_validation.json")
    assert path.exists()


def test_processed_slice_validation_writer(tmp_path: Path):
    mesh = generate_rect_duct_mesh(width=2.0, height=2.0, ny=5, nz=5)
    y, z = jnp.meshgrid(mesh.y_centers, mesh.z_centers, indexing="ij")
    u = 1.0 - 0.2 * y**2 - 0.3 * z**2
    zeros = jnp.zeros_like(u)
    solution = Solution(
        mesh=mesh,
        state=MHDState(
            u=u,
            phi=zeros,
            jy=zeros,
            jz=zeros,
            lorentz_x=zeros,
            time=0.0,
            residual=0.0,
        ),
        diagnostics=Diagnostics(
            residual_history=jnp.asarray([0.0]),
            courant_like=jnp.asarray([0.0]),
            ohmic_power=jnp.asarray([0.0]),
        ),
        case_name="shercliff_ha2",
    )
    closed_channel_root = tmp_path / "ClosedChannel"
    closed_channel_root.mkdir(parents=True)
    center_y = solution.state.u[:, solution.state.u.shape[1] // 2]
    center_z = solution.state.u[solution.state.u.shape[0] // 2, :]
    rows = ["Points:1,Points:2,U:0,potE"]
    for y_coord, value in zip(mesh.y_centers.tolist(), center_y.tolist()):
        rows.append(f"{y_coord},0.0,{value},0.0")
    for z_coord, value in zip(mesh.z_centers.tolist(), center_z.tolist()):
        if abs(z_coord) < 1e-12:
            continue
        rows.append(f"0.0,{z_coord},{value},0.0")
    (closed_channel_root / "shercliff_Ha2_XSlice1m_4s.csv").write_text("\n".join(rows))
    report = processed_slice_validation(solution, "shercliff", 2, reference_root=closed_channel_root)
    path = write_processed_slice_validation(report, tmp_path / "processed_slice_validation.json")
    assert report.y_profile.l2_error < 0.02
    assert report.z_profile.l2_error < 0.03
    assert path.exists()

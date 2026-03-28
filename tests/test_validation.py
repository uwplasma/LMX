from pathlib import Path

import jax.numpy as jnp
import pytest

from lmx.cases import make_hartmann_case, make_shercliff_case
from lmx.core import Diagnostics, MHDState, Solution
from lmx.mesh import generate_rect_duct_mesh
from lmx.solvers import solve_steady
from lmx.validation import (
    closed_channel_validation,
    compare_with_freemhd,
    duct_profile_metrics,
    latest_sampled_profiles,
    hartmann_analytic_profile,
    hartmann_validation,
    inspect_freemhd_case,
    latest_field_minmax_record,
    normalize_sample_distance,
    processed_slice_validation,
    read_freemhd_xy_sample,
    read_field_minmax,
    write_analytic_comparison,
    write_closed_channel_validation,
    write_metrics_json,
    write_processed_slice_validation,
    write_validation_report,
)


pytestmark = pytest.mark.validation


def test_hartmann_profile_center_is_maximum():
    y = jnp.linspace(-1.0, 1.0, 101)
    profile = hartmann_analytic_profile(y, ha=10.0)
    assert float(profile[50]) >= float(profile[0])


def test_compare_with_freemhd_report(tmp_path: Path):
    case = make_hartmann_case()
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
    report = compare_with_freemhd(case, tmp_path)
    path = write_validation_report(report, tmp_path / "report.json")
    assert path.exists()
    assert report.metrics["control_dict_count"] == pytest.approx(1.0)
    assert report.metrics["region_zero_dir_count"] == pytest.approx(1.0)
    assert report.metrics["has_potE_zero_field"] == pytest.approx(1.0)
    assert report.metrics["field_minmax_file_count"] == pytest.approx(1.0)
    assert report.metrics["freemhd_u_max_latest"] == pytest.approx(0.25)
    assert report.metrics["sampled_profile_pair_available"] == pytest.approx(1.0)
    assert "freemhd_sample_y_l2_error" in report.metrics


def test_inspect_freemhd_case_collects_case_structure(tmp_path: Path):
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
    inspection = inspect_freemhd_case(tmp_path)
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


def test_sample_reader_and_latest_profile_detection(tmp_path: Path):
    sample_root = tmp_path / "postProcessing" / "lmxSampleDict" / "liquid" / "0.0001"
    sample_root.mkdir(parents=True)
    rows = "0.0 0.1 1.0 2.0 3.0\n1.0 0.2 4.0 5.0 6.0\n"
    y_path = sample_root / "centerlineY_potE_U.xy"
    z_path = sample_root / "centerlineZ_potE_U.xy"
    y_path.write_text(rows)
    z_path.write_text(rows)
    sample = read_freemhd_xy_sample(y_path)
    latest = latest_sampled_profiles(tmp_path)
    assert sample.distance.shape[0] == 2
    assert float(sample.u_x[1]) == pytest.approx(4.0)
    assert latest is not None
    assert latest[0].path.endswith("centerlineY_potE_U.xy")
    normalized = normalize_sample_distance(sample.distance)
    assert float(normalized[0]) == pytest.approx(-1.0)
    assert float(normalized[-1]) == pytest.approx(1.0)


def test_hartmann_validation_writer(tmp_path: Path):
    case = make_hartmann_case(ha=5.0, ny=16, nz=16)
    solution = solve_steady(case)
    comparison = hartmann_validation(solution, ha=5.0)
    path = write_analytic_comparison(comparison, tmp_path / "analytic.json", axis_name="y")
    assert path.exists()


def test_hartmann_ha20_validation_error_is_bounded():
    case = make_hartmann_case(ha=20.0, ny=16, nz=16)
    solution = solve_steady(case)
    comparison = hartmann_validation(solution, ha=20.0)
    assert comparison.l2_error < 0.05


def test_duct_profile_metrics_writer(tmp_path: Path):
    case = make_hartmann_case(ha=5.0, ny=16, nz=16)
    solution = solve_steady(case)
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
    solution = solve_steady(case)
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
    assert report.y_profile.l2_error < 1e-12
    assert report.z_profile.l2_error < 1e-12
    assert path.exists()

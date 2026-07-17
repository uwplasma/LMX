from pathlib import Path

import numpy as np
import pytest

from lmx.reference_data import (
    ProcessedSliceReference,
    _fill_missing_structured_values,
    _interpolated_centerline_profile,
    _match_single,
    default_closed_channel_reference_root,
    extract_processed_profile,
    extract_processed_midplane_profile,
    load_closed_channel_analytical,
    load_hunt_analytical,
    load_processed_slice,
    load_shercliff_analytical,
    processed_slice_area_mean,
    processed_slice_field_grid,
    processed_slice_point_mesh,
)


pytestmark = pytest.mark.unit


def _closed_channel_root_or_skip() -> Path:
    root = default_closed_channel_reference_root()
    if not root.exists():
        pytest.skip("optional FreeMHD closed-channel reference data are not available")
    return root


def _processed_slice(tmp_path: Path, rows: list[str] | str) -> ProcessedSliceReference:
    root = tmp_path / "ClosedChannel"
    root.mkdir(parents=True)
    text = rows if isinstance(rows, str) else "\n".join(rows)
    (root / "hunt_exactBL_Ha20_XSlice1m_4s.csv").write_text(text)
    return load_processed_slice("hunt", 20, reference_root=root)


def test_load_closed_channel_analytical_parses_axes_and_pressure_drop(tmp_path: Path):
    analytical_root = tmp_path / "ClosedChannel" / "AnalyticalSolutions"
    analytical_root.mkdir(parents=True)
    path = analytical_root / "Shercliff_Analytical_Ha20_PresDrop2512.1961.txt"
    path.write_text("r\tu1\tu2\n-0.1\t0.0\t0.1\n0.1\t1.0\t0.9\n")
    reference = load_closed_channel_analytical(
        "shercliff", 20, tmp_path / "ClosedChannel"
    )
    assert reference.pressure_drop == pytest.approx(2512.1961)
    assert reference.midplane_z.tolist() == pytest.approx([0.0, 1.0])
    assert reference.midplane_y.tolist() == pytest.approx([0.1, 0.9])


def test_extract_processed_midplane_profile_returns_sorted_cut(tmp_path: Path):
    reference = _processed_slice(
        tmp_path,
        [
            "Points:1,Points:2,U:0,potE",
            "0.0,-1.0,1.0,0.1",
            "0.0,0.0,2.0,0.2",
            "0.0,1.0,3.0,0.3",
            "-1.0,0.0,4.0,0.4",
            "1.0,0.0,5.0,0.5",
        ],
    )
    y_profile = extract_processed_midplane_profile(reference, axis="y")
    z_profile = extract_processed_midplane_profile(reference, axis="z")
    assert y_profile["y"].tolist() == pytest.approx([-1.0, 0.0, 1.0])
    assert y_profile["u"].tolist() == pytest.approx([4.0, 2.0, 5.0])
    assert z_profile["z"].tolist() == pytest.approx([-1.0, 0.0, 1.0])
    assert z_profile["u"].tolist() == pytest.approx([1.0, 2.0, 3.0])


def test_extract_processed_profile_returns_requested_field_component(tmp_path: Path):
    reference = _processed_slice(
        tmp_path,
        [
            "Points:1,Points:2,U:0,J:1,J:2,potE",
            "0.0,-1.0,1.0,10.0,11.0,0.1",
            "0.0,0.0,2.0,20.0,21.0,0.2",
            "0.0,1.0,3.0,30.0,31.0,0.3",
            "-1.0,0.0,4.0,40.0,41.0,0.4",
            "1.0,0.0,5.0,50.0,51.0,0.5",
        ],
    )
    y_profile = extract_processed_profile(
        reference, axis="y", field_name="J", component=1
    )
    z_profile = extract_processed_profile(reference, axis="z", field_name="potE")
    assert y_profile["coordinate"].tolist() == pytest.approx([-1.0, 0.0, 1.0])
    assert y_profile["value"].tolist() == pytest.approx([40.0, 20.0, 50.0])
    assert z_profile["coordinate"].tolist() == pytest.approx([-1.0, 0.0, 1.0])
    assert z_profile["value"].tolist() == pytest.approx([0.1, 0.2, 0.3])


def test_processed_slice_field_grid_averages_duplicate_points(tmp_path: Path):
    reference = _processed_slice(
        tmp_path,
        [
            "Points:1,Points:2,U:0",
            "0.0,0.0,1.0",
            "0.0,1.0,2.0",
            "1.0,0.0,3.0",
            "1.0,1.0,5.0",
            "1.0,1.0,7.0",
        ],
    )
    grid = processed_slice_field_grid(reference, field_name="U", component=0)

    assert grid["y"].tolist() == pytest.approx([0.0, 1.0])
    assert grid["z"].tolist() == pytest.approx([0.0, 1.0])
    assert grid["value"].ravel().tolist() == pytest.approx([1.0, 2.0, 3.0, 6.0])
    assert processed_slice_area_mean(reference) == pytest.approx(3.0)


def test_processed_slice_area_mean_uses_nonuniform_quadrature(tmp_path: Path):
    y_values = [-1.0, -0.25, 1.0]
    z_values = [-2.0, 0.0, 2.0]
    rows = ["Points:1,Points:2,U:0"]
    for point_y in y_values:
        for point_z in z_values:
            rows.append(f"{point_y},{point_z},{2.0 + point_y + 0.5 * point_z}")
    rows.append("-0.25,0.0,1.75")
    reference = _processed_slice(tmp_path, rows)

    assert processed_slice_area_mean(reference) == pytest.approx(2.0)


def test_processed_slice_grid_fills_missing_values_and_reports_bad_field(
    tmp_path: Path,
):
    reference = _processed_slice(
        tmp_path,
        [
            "Points:1,Points:2,U:0",
            "0.0,0.0,1.0",
            "0.0,1.0,3.0",
            "1.0,0.0,5.0",
        ],
    )

    grid = processed_slice_field_grid(reference, field_name="U", component=0)

    assert grid["value"].ravel().tolist() == pytest.approx([1.0, 3.0, 5.0, 5.0])
    with pytest.raises(KeyError, match="available columns"):
        processed_slice_field_grid(reference, field_name="missing")


def test_processed_slice_area_mean_handles_single_sample(tmp_path: Path):
    reference = _processed_slice(
        tmp_path, "Points:1,Points:2,U:0\n0.0,0.0,4.0\n"
    )

    assert processed_slice_area_mean(reference) == pytest.approx(4.0)


def test_processed_slice_point_mesh_uses_unique_slice_coordinates(tmp_path: Path):
    reference = _processed_slice(
        tmp_path,
        [
            "Points:1,Points:2,U:0",
            "-0.1,-0.1,0.0",
            "-0.1,0.0,1.0",
            "-0.1,0.1,0.0",
            "0.0,-0.1,1.0",
            "0.0,0.0,2.0",
            "0.1,0.1,0.0",
        ],
    )

    mesh = processed_slice_point_mesh(reference, length=2.0, nx=2)

    assert mesh.geometry == "rect_duct"
    assert mesh.nx == 2
    assert mesh.y_faces.tolist() == pytest.approx([-0.1, 0.0, 0.1])
    assert mesh.z_faces.tolist() == pytest.approx([-0.1, 0.0, 0.1])


def test_fill_missing_structured_values_covers_column_and_fallback_paths():
    filled = _fill_missing_structured_values(
        grid=np.asarray([[1.0, np.nan], [np.nan, np.nan], [5.0, np.nan]]),
        y=np.asarray([0.0, 1.0, 2.0]),
        z=np.asarray([0.0, 1.0]),
    )
    fallback = _fill_missing_structured_values(
        grid=np.asarray([[np.nan]]),
        y=np.asarray([0.0]),
        z=np.asarray([0.0]),
    )

    assert filled.ravel().tolist() == pytest.approx([1.0, 1.0, 3.0, 3.0, 5.0, 5.0])
    assert fallback.ravel().tolist() == pytest.approx([0.0])


def test_extract_processed_profile_interpolates_symmetric_near_center_planes(
    tmp_path: Path,
):
    reference = _processed_slice(
        tmp_path,
        [
            "Points:1,Points:2,U:0,J:1,potE",
            "-1.0,-0.1,1.0,10.0,-2.0",
            "0.0,-0.1,2.0,20.0,-4.0",
            "1.0,-0.1,3.0,30.0,-6.0",
            "-1.0,0.1,5.0,50.0,2.0",
            "0.0,0.1,6.0,60.0,4.0",
            "1.0,0.1,7.0,70.0,6.0",
            "0.0,-1.0,11.0,110.0,-8.0",
            "0.0,1.0,13.0,130.0,8.0",
        ],
    )

    y_profile = extract_processed_profile(
        reference, axis="y", field_name="J", component=1
    )
    y_midplane = extract_processed_midplane_profile(reference, axis="y")
    z_profile = extract_processed_profile(reference, axis="z", field_name="potE")

    assert y_profile["coordinate"].tolist() == pytest.approx([-1.0, 0.0, 1.0])
    assert y_profile["value"].tolist() == pytest.approx([30.0, 40.0, 50.0])
    assert y_midplane["u"].tolist() == pytest.approx([3.0, 4.0, 5.0])
    assert z_profile["coordinate"].tolist() == pytest.approx([-1.0, -0.1, 0.1, 1.0])
    assert z_profile["value"].tolist() == pytest.approx([-8.0, -4.0, 4.0, 8.0])


@pytest.mark.external
def test_default_closed_channel_reference_root_resolves_bundled_dataset():
    root = _closed_channel_root_or_skip()
    assert root.exists()
    assert (root / "AnalyticalSolutions").exists()
    assert any(root.glob("hunt_*Ha20*XSlice1m_*.csv"))


@pytest.mark.external
def test_load_bundled_reference_data_uses_repo_dataset():
    root = _closed_channel_root_or_skip()
    analytical = load_closed_channel_analytical("hunt", 20, root)
    processed = load_processed_slice("hunt", 20, reference_root=root)

    assert analytical.pressure_drop is not None
    assert analytical.coordinate.shape[0] > 10
    assert analytical.midplane_y.shape == analytical.coordinate.shape
    assert processed.columns["U:0"].shape[0] > 10
    assert "potE" in processed.columns


@pytest.mark.external
def test_case_specific_closed_channel_reference_helpers_forward_to_dataset():
    root = _closed_channel_root_or_skip()
    shercliff = load_shercliff_analytical(20, root)
    hunt = load_hunt_analytical(20, root)

    assert shercliff.case_kind == "shercliff"
    assert shercliff.coordinate.shape[0] > 10
    assert hunt.case_kind == "hunt"
    assert hunt.coordinate.shape[0] > 10


def test_reference_root_overrides_and_missing_match(tmp_path: Path):
    assert default_closed_channel_reference_root(tmp_path) == tmp_path
    with pytest.raises(FileNotFoundError, match="No reference files"):
        _match_single(["*.missing"], tmp_path)


def test_pressure_drop_optional_and_case_helpers(tmp_path: Path):
    analytical = tmp_path / "AnalyticalSolutions"
    analytical.mkdir()
    (analytical / "Shercliff_Analytical_Ha5_profile.txt").write_text("r u1 u2\n0 1 1\n")
    reference = load_shercliff_analytical(5, tmp_path)
    assert reference.pressure_drop is None


def test_centerline_profile_empty_one_sided_and_invalid_axes():
    empty_coord, empty_values = _interpolated_centerline_profile(
        np.asarray([]), np.asarray([]), np.asarray([])
    )
    assert empty_coord.size == empty_values.size == 0

    coord, values = _interpolated_centerline_profile(
        np.asarray([0.0, 1.0]), np.asarray([0.2, 0.2]), np.asarray([2.0, 3.0])
    )
    assert coord.tolist() == pytest.approx([0.0, 1.0])
    assert values.tolist() == pytest.approx([2.0, 3.0])

    reference = ProcessedSliceReference(
        case_kind="hunt",
        ha=1,
        columns={
            "Points:1": np.asarray([0.0]),
            "Points:2": np.asarray([0.0]),
            "U:0": np.asarray([1.0]),
        },
        path="memory",
    )
    with pytest.raises(ValueError, match="Unsupported axis"):
        extract_processed_midplane_profile(reference, axis="x")
    with pytest.raises(ValueError, match="Unsupported axis"):
        extract_processed_profile(reference, axis="x", field_name="U", component=0)
def test_area_mean_empty_and_fill_noop():
    reference = ProcessedSliceReference(
        case_kind="empty",
        ha=0,
        columns={
            "Points:1": np.asarray([]),
            "Points:2": np.asarray([]),
            "U:0": np.asarray([]),
        },
        path="memory",
    )
    assert processed_slice_area_mean(reference) == 0.0
    grid = np.asarray([[1.0, 2.0]])
    assert (
        _fill_missing_structured_values(grid, np.asarray([0.0]), np.asarray([0.0, 1.0]))
        is grid
    )

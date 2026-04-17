from pathlib import Path

import pytest

from lmx.reference_data import (
    default_closed_channel_reference_root,
    default_fringing_pipe_reference_root,
    extract_processed_midplane_profile,
    fringing_pipe_profile_reference_path,
    load_closed_channel_analytical,
    load_fringing_pipe_profile,
    load_processed_slice,
)


pytestmark = pytest.mark.unit


def test_load_closed_channel_analytical_parses_axes_and_pressure_drop(tmp_path: Path):
    analytical_root = tmp_path / "ClosedChannel" / "AnalyticalSolutions"
    analytical_root.mkdir(parents=True)
    path = analytical_root / "Shercliff_Analytical_Ha20_PresDrop2512.1961.txt"
    path.write_text("r\tu1\tu2\n-0.1\t0.0\t0.1\n0.1\t1.0\t0.9\n")
    reference = load_closed_channel_analytical("shercliff", 20, tmp_path / "ClosedChannel")
    assert reference.pressure_drop == pytest.approx(2512.1961)
    assert reference.midplane_z.tolist() == pytest.approx([0.0, 1.0])
    assert reference.midplane_y.tolist() == pytest.approx([0.1, 0.9])


def test_extract_processed_midplane_profile_returns_sorted_cut(tmp_path: Path):
    closed_channel_root = tmp_path / "ClosedChannel"
    closed_channel_root.mkdir(parents=True)
    path = closed_channel_root / "hunt_exactBL_Ha20_XSlice1m_4s.csv"
    path.write_text(
        "\n".join(
            [
                "Points:1,Points:2,U:0,potE",
                "0.0,-1.0,1.0,0.1",
                "0.0,0.0,2.0,0.2",
                "0.0,1.0,3.0,0.3",
                "-1.0,0.0,4.0,0.4",
                "1.0,0.0,5.0,0.5",
            ]
        )
    )
    reference = load_processed_slice("hunt", 20, reference_root=closed_channel_root)
    y_profile = extract_processed_midplane_profile(reference, axis="y")
    z_profile = extract_processed_midplane_profile(reference, axis="z")
    assert y_profile["y"].tolist() == pytest.approx([-1.0, 0.0, 1.0])
    assert y_profile["u"].tolist() == pytest.approx([4.0, 2.0, 5.0])
    assert z_profile["z"].tolist() == pytest.approx([-1.0, 0.0, 1.0])
    assert z_profile["u"].tolist() == pytest.approx([1.0, 2.0, 3.0])


def test_default_closed_channel_reference_root_resolves_bundled_dataset():
    root = default_closed_channel_reference_root()
    assert root.exists()
    assert (root / "AnalyticalSolutions").exists()
    assert any(root.glob("hunt_*Ha20*XSlice1m_*.csv"))


def test_load_bundled_reference_data_uses_repo_dataset():
    root = default_closed_channel_reference_root()
    analytical = load_closed_channel_analytical("hunt", 20, root)
    processed = load_processed_slice("hunt", 20, reference_root=root)

    assert analytical.pressure_drop is not None
    assert analytical.coordinate.shape[0] > 10
    assert analytical.midplane_y.shape == analytical.coordinate.shape
    assert processed.columns["U:0"].shape[0] > 10
    assert "potE" in processed.columns


def test_default_fringing_pipe_reference_root_resolves_bundled_dataset():
    root = default_fringing_pipe_reference_root()
    assert root.exists()
    assert (root / "Buhler2020PaperProperties_Ha2k_Re20k_coarserZMesh5x_CenterLine_5.89s.csv").exists()


def test_load_bundled_fringing_pipe_profile_uses_repo_dataset():
    root = default_fringing_pipe_reference_root()
    center_path = fringing_pipe_profile_reference_path("center", root)
    reference = load_fringing_pipe_profile("center", root)

    assert center_path.exists()
    assert reference.profile_kind == "center"
    assert reference.coordinate.shape[0] > 10
    assert reference.velocity.shape == reference.coordinate.shape
    assert abs(reference.x_offset_fraction) < 1.0e-12

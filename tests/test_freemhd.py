from pathlib import Path

import pytest

from lmx.freemhd import (
    build_case_from_freemhd_reference,
    infer_initial_velocity_x,
    infer_inlet_drive_mode,
    parse_freemhd_execution_seconds,
)


pytestmark = pytest.mark.unit


def test_inference_helpers_read_case_zero_and_latesttime_fallbacks(tmp_path: Path):
    case_zero = tmp_path / "case" / "0" / "liquid"
    case_zero.mkdir(parents=True)
    (case_zero / "U").write_text(
        """internalField   uniform ( 0.9725 0 0 );

boundaryField
{
    inlet
    {
        type            flowRateInletVelocity;
        value           uniform ( 0.9725 0 0 );
        volumetricFlowRate 0.0389;
    }
}
"""
    )
    assert infer_initial_velocity_x(tmp_path) == pytest.approx(0.9725)
    assert infer_inlet_drive_mode(tmp_path) == "inlet_flow_rate"

    fallback_root = tmp_path / "fallback"
    latest = fallback_root / "latestTime" / "liquid"
    latest.mkdir(parents=True)
    (latest / "U").write_text("internalField   uniform ( 0.123 0 0 );\n")
    assert infer_initial_velocity_x(fallback_root) == pytest.approx(0.123)


def test_build_case_from_freemhd_reference_preserves_default_forcing_for_shercliff(tmp_path: Path):
    u_path = tmp_path / "case" / "0" / "liquid"
    u_path.mkdir(parents=True)
    (u_path / "U").write_text("internalField   uniform ( 0.9725 0 0 );\n")

    case = build_case_from_freemhd_reference(
        case_kind="shercliff",
        ha=20.0,
        ny=12,
        nz=12,
        dt=1.0e-5,
        t_final=1.0e-4,
        max_steps=10,
        reference_run_dir=tmp_path,
        forcing=None,
    )

    assert case.forcing == pytest.approx(1.0)
    assert case.initial_velocity == pytest.approx(0.9725)


def test_parse_freemhd_execution_seconds_returns_latest_value(tmp_path: Path):
    path = tmp_path / "run.log"
    path.write_text(
        "\n".join(
            [
                "ExecutionTime = 7.32 s  ClockTime = 19 s",
                "ExecutionTime = 35.28 s  ClockTime = 91 s",
            ]
        )
    )
    assert parse_freemhd_execution_seconds(path) == pytest.approx(35.28)

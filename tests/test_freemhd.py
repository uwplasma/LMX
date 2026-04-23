from pathlib import Path

import pytest

from lmx.freemhd import (
    build_case_from_freemhd_reference,
    infer_initial_velocity_x,
    infer_inlet_drive_mode,
    infer_liquid_properties,
    infer_rectangular_geometry,
    infer_solid_conductivities,
    infer_uniform_b0,
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


def test_build_case_from_freemhd_reference_switches_to_inlet_flow_rate_when_reference_uses_it(tmp_path: Path):
    u_path = tmp_path / "case" / "0" / "liquid"
    u_path.mkdir(parents=True)
    (u_path / "U").write_text(
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

    inlet_boundaries = [bc for bc in case.boundary_conditions if bc.name == "inlet"]
    assert case.forcing == pytest.approx(0.0)
    assert inlet_boundaries
    assert inlet_boundaries[-1].kind == "inlet_flow_rate"
    assert inlet_boundaries[-1].value == pytest.approx(3.89)


def test_freemhd_inference_helpers_recover_geometry_materials_and_b0(tmp_path: Path):
    liquid = tmp_path / "constant" / "liquid"
    liquid.mkdir(parents=True)
    (liquid / "thermophysicalProperties.liquidMetal").write_text(
        """mixture
{
    equationOfState
    {
        rho 1000;
    }
    transport
    {
        mu 0.001;
    }
}

elcond [-1 -3  3 0 0 2 0]1e6;
"""
    )
    solid = tmp_path / "constant" / "solidWalls"
    solid.mkdir(parents=True)
    (solid / "thermophysicalProperties").write_text("elcond 5e6;\n")
    insulator = tmp_path / "constant" / "insulator"
    insulator.mkdir(parents=True)
    (insulator / "thermophysicalProperties").write_text("elcond 1e-6;\n")
    initial = tmp_path / "0" / "liquid"
    initial.mkdir(parents=True)
    (initial / "B0").write_text("internalField uniform ( 0 0.2 0 );\n")
    system = tmp_path / "system"
    system.mkdir()
    (system / "blockMeshDict").write_text("Ly 0.1;\nLy_wall 0.101;\nN_wall 2;\n")

    assert infer_liquid_properties(tmp_path) == pytest.approx((1.0e6, 1000.0, 1.0e-3))
    assert infer_uniform_b0(tmp_path) == pytest.approx((0.0, 0.2, 0.0))
    width, height, wall_thickness, wall_cells = infer_rectangular_geometry(tmp_path)
    assert width == pytest.approx(0.2)
    assert height == pytest.approx(0.2)
    assert wall_thickness == pytest.approx(0.001)
    assert wall_cells == 2
    assert infer_solid_conductivities(tmp_path) == pytest.approx((5.0e6, 1.0e-6))


def test_build_case_from_freemhd_reference_adopts_reference_geometry_materials_and_b0(tmp_path: Path):
    liquid = tmp_path / "constant" / "liquid"
    liquid.mkdir(parents=True)
    (liquid / "thermophysicalProperties.liquidMetal").write_text(
        """mixture
{
    equationOfState
    {
        rho 1000;
    }
    transport
    {
        mu 0.001;
    }
}

elcond [-1 -3  3 0 0 2 0]1e6;
"""
    )
    solid = tmp_path / "constant" / "solidWalls"
    solid.mkdir(parents=True)
    (solid / "thermophysicalProperties").write_text("elcond 5e6;\n")
    insulator = tmp_path / "constant" / "insulator"
    insulator.mkdir(parents=True)
    (insulator / "thermophysicalProperties").write_text("elcond 1e-6;\n")
    initial = tmp_path / "0" / "liquid"
    initial.mkdir(parents=True)
    (initial / "B0").write_text(
        """internalField   uniform ( 0 0.2 0 );
boundaryField {}
"""
    )
    (initial / "U").write_text(
        """internalField   uniform ( 0.9725 0 0 );

boundaryField
{
    inlet
    {
        type flowRateInletVelocity;
        volumetricFlowRate 0.0389;
    }
}
"""
    )
    system = tmp_path / "system"
    system.mkdir()
    (system / "blockMeshDict").write_text("Ly 0.1;\nLy_wall 0.101;\nN_wall 2;\n")

    case = build_case_from_freemhd_reference(
        case_kind="hunt",
        ha=20.0,
        ny=12,
        nz=12,
        dt=1.0e-5,
        t_final=1.0e-4,
        max_steps=10,
        reference_run_dir=tmp_path,
        forcing=None,
    )

    assert case.geometry.width == pytest.approx(0.2)
    assert case.geometry.height == pytest.approx(0.2)
    assert case.geometry.wall_thickness == pytest.approx((0.001, 0.001, 0.001, 0.001))
    assert case.geometry.wall_cells == (2, 2, 2, 2)
    assert case.regions[0].conductivity == pytest.approx(1.0e6)
    assert case.regions[0].density == pytest.approx(1000.0)
    assert case.regions[0].viscosity == pytest.approx(1.0e-3)
    assert case.magnetic_field.value == pytest.approx((0.0, 0.2, 0.0))
    inlet_boundaries = [bc for bc in case.boundary_conditions if bc.name == "inlet"]
    assert inlet_boundaries
    assert inlet_boundaries[-1].kind == "inlet_flow_rate"
    assert inlet_boundaries[-1].value == pytest.approx(0.0389)


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

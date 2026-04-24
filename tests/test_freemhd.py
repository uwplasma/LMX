from pathlib import Path

import pytest

from lmx.freemhd import (
    build_case_from_freemhd_reference,
    candidate_u_paths,
    compare_side_jet_profiles,
    infer_initial_velocity_x,
    infer_inlet_drive_mode,
    infer_inlet_flow_rate,
    infer_liquid_properties,
    infer_magnetic_ramp,
    infer_rectangular_geometry,
    infer_reduced_inlet_flow_rate,
    infer_solid_conductivities,
    infer_uniform_b0,
    parse_freemhd_execution_seconds,
    run_freemhd_demo,
    side_jet_profile_metrics,
    summarize_observable_gate,
    summarize_observable_offenders,
    summarize_profile_error_offenders,
    summarize_runtime_offenders,
)


pytestmark = pytest.mark.unit


def test_side_jet_profile_metrics_and_comparison_capture_peak_locations():
    coordinate = [-1.0, -0.7, 0.0, 0.7, 1.0]
    reference = [0.0, 1.4, 1.0, 1.4, 0.0]
    simulated = [0.0, 1.2, 1.0, 1.3, 0.0]

    metrics = side_jet_profile_metrics(coordinate, reference)
    assert metrics["negative_location"] == pytest.approx(-0.7)
    assert metrics["positive_location"] == pytest.approx(0.7)
    assert metrics["peak_to_center_ratio"] == pytest.approx(1.4)

    comparison = compare_side_jet_profiles(coordinate, simulated, coordinate, reference)
    assert comparison["normalized_location_error"] == pytest.approx(0.0)
    assert comparison["peak_value_relative_error"] == pytest.approx((1.4 - 1.3) / 1.4)


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


def test_freemhd_inference_helpers_cover_missing_and_fallback_paths(tmp_path: Path):
    assert len(candidate_u_paths(tmp_path)) >= 6
    assert infer_liquid_properties(tmp_path) is None
    assert infer_uniform_b0(tmp_path) is None
    assert infer_rectangular_geometry(tmp_path) is None
    assert infer_solid_conductivities(tmp_path) == (None, None)
    assert parse_freemhd_execution_seconds(tmp_path / "missing.log") is None

    incomplete_liquid = tmp_path / "case" / "constant" / "liquid"
    incomplete_liquid.mkdir(parents=True)
    (incomplete_liquid / "thermophysicalProperties").write_text("sigma 3.0;\nrho 1000;\n")
    assert infer_liquid_properties(tmp_path) is None

    (incomplete_liquid / "thermophysicalProperties").write_text("sigma 3.0;\nrho 1000;\nmu 0.002;\n")
    assert infer_liquid_properties(tmp_path) == pytest.approx((3.0, 1000.0, 0.002))

    b0_dir = tmp_path / "case" / "0" / "liquid"
    b0_dir.mkdir(parents=True, exist_ok=True)
    (b0_dir / "B0").write_text("internalField nonuniform List<vector> 0();\n")
    assert infer_uniform_b0(tmp_path) is None

    system = tmp_path / "case" / "system"
    system.mkdir(parents=True)
    (system / "blockMeshDict").write_text("Ly_wall 0.1;\n")
    assert infer_rectangular_geometry(tmp_path) is None
    (system / "blockMeshDict").write_text("Ly 0.1;\nLy_wall 0.09;\n")
    width, height, wall_thickness, wall_cells = infer_rectangular_geometry(tmp_path)
    assert width == pytest.approx(0.2)
    assert height == pytest.approx(0.2)
    assert wall_thickness is None
    assert wall_cells is None

    control = tmp_path / "system"
    control.mkdir(exist_ok=True)
    (control / "controlDict").write_text("application epotMultiRegionFoam;\n")
    assert infer_magnetic_ramp(tmp_path) == pytest.approx((0.0, 0.0))


def test_inlet_flow_rate_helpers_cover_malformed_and_fallback_cases(tmp_path: Path):
    u_dir = tmp_path / "0"
    u_dir.mkdir()
    (u_dir / "U").write_text(
        """internalField uniform ( 0.2 0 0 );
boundaryField
{
    outlet
    {
        type zeroGradient;
    }
}
"""
    )
    assert infer_inlet_drive_mode(tmp_path) is None
    assert infer_inlet_flow_rate(tmp_path) is None

    (u_dir / "U").write_text(
        """internalField uniform ( 0.2 0 0 );
boundaryField
{
    inlet
    {
        value uniform (0.2 0 0);
    }
}
"""
    )
    assert infer_inlet_drive_mode(tmp_path) is None
    assert infer_inlet_flow_rate(tmp_path) is None

    (u_dir / "U").write_text(
        """internalField uniform ( 0.2 0 0 );
boundaryField
{
    inlet
    {
        type flowRateInletVelocity;
        volumetricFlowRate 0.0;
    }
}
"""
    )
    assert infer_reduced_inlet_flow_rate(tmp_path, reduced_area=1.0, initial_velocity=0.2) is None

    (u_dir / "U").write_text(
        """internalField uniform ( 0.2 0 0 );
boundaryField
{
    inlet
    {
        type flowRateInletVelocity;
        volumetricFlowRate constant 0.125;
    }
}
"""
    )
    assert infer_inlet_flow_rate(tmp_path) == pytest.approx(0.125)


def test_freemhd_offender_summaries_rank_accuracy_and_runtime():
    observable_records = [
        {
            "case_kind": "shercliff",
            "drive_mode": "forcing",
            "observables": {
                "velocity": {
                    "y": {"l2_error": 2.0e-2, "linf_error": 5.0e-2},
                    "z": {"l2_error": 4.0e-3, "linf_error": 1.0e-2},
                    "peak_ratio": 0.95,
                },
                "current": {
                    "y": {"l2_error": 8.0e-2, "linf_error": 2.0e-1, "peak_ratio": 1.4},
                    "z": {"l2_error": 1.0e-2, "linf_error": 2.0e-2},
                    "peak_ratio": 1.1,
                },
                "potential": {
                    "y": {"l2_error": 1.0, "linf_error": 1.0, "reference_peak_abs": 1.0e-8},
                    "z": {"l2_error": 2.0e-2, "linf_error": 5.0e-2, "reference_peak_abs": 1.0},
                },
            },
        }
    ]
    observable_offenders = summarize_observable_offenders(observable_records, l2_target=1.0e-2)
    assert observable_offenders[0]["observable"] == "current"
    assert observable_offenders[0]["axis"] == "y"
    assert observable_offenders[0]["status"] == "offender"
    assert observable_offenders[-1]["status"] == "low_signal"

    observable_gate = summarize_observable_gate(observable_records, l2_target=1.0e-2)
    assert observable_gate["research_grade_validation_pass"] is False
    assert observable_gate["observable_offender_count"] == 3
    assert observable_gate["low_signal_count"] == 1
    assert observable_gate["missing_observable_count"] == 1
    assert observable_gate["missing_observables"] == [
        {"case_kind": "shercliff", "observable": "lorentz", "axis": "*"}
    ]

    profile_records = [
        {
            "case_kind": "shercliff",
            "freemhd_execution_seconds": 10.0,
            "lmx_execution_seconds": 8.0,
            "y_l2_error": 2.0e-2,
            "z_l2_error": 5.0e-3,
        },
        {
            "case_kind": "hunt",
            "freemhd_execution_seconds": 10.0,
            "lmx_execution_seconds": 14.0,
            "y_l2_error": 1.0e-1,
            "z_l2_error": 7.0e-2,
        },
    ]
    profile_offenders = summarize_profile_error_offenders(profile_records, l2_target=1.0e-2, top_n=2)
    assert [item["case_kind"] for item in profile_offenders] == ["hunt", "hunt"]
    runtime_offenders = summarize_runtime_offenders(profile_records)
    assert runtime_offenders[0]["case_kind"] == "hunt"
    assert runtime_offenders[0]["status"] == "offender"
    assert runtime_offenders[1]["status"] == "pass"


def test_build_case_from_freemhd_reference_covers_hartmann_velocity_mode_and_errors(tmp_path: Path):
    u_dir = tmp_path / "0"
    u_dir.mkdir()
    (u_dir / "U").write_text(
        """internalField uniform ( 0.3 0 0 );
boundaryField
{
    inlet
    {
        type fixedValue;
        value uniform (0.3 0 0);
    }
}
"""
    )
    case = build_case_from_freemhd_reference(
        case_kind="hartmann",
        ha=10.0,
        ny=8,
        nz=8,
        dt=1.0e-4,
        t_final=1.0e-3,
        max_steps=10,
        reference_run_dir=tmp_path,
        forcing=None,
    )
    inlet = [bc for bc in case.boundary_conditions if bc.name == "inlet"][-1]
    assert case.name == "hartmann_ha10"
    assert case.forcing == pytest.approx(0.0)
    assert inlet.kind == "inlet_velocity"
    assert inlet.value == pytest.approx((0.3, 0.0, 0.0))

    forced = build_case_from_freemhd_reference(
        case_kind="hartmann",
        ha=10.0,
        ny=8,
        nz=8,
        dt=1.0e-4,
        t_final=1.0e-3,
        max_steps=10,
        reference_run_dir=tmp_path,
        forcing=2.5,
    )
    assert forced.forcing == pytest.approx(2.5)
    assert not [bc for bc in forced.boundary_conditions if bc.name == "inlet"]

    with pytest.raises(ValueError, match="Unsupported FreeMHD reference case kind"):
        build_case_from_freemhd_reference(
            case_kind="unknown",
            ha=10.0,
            ny=8,
            nz=8,
            dt=1.0e-4,
            t_final=1.0e-3,
            max_steps=10,
            reference_run_dir=tmp_path,
        )


def test_run_freemhd_demo_invokes_script_and_returns_output_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    calls: list[dict[str, object]] = []

    def fake_run(args, *, cwd, env, check):
        calls.append({"args": args, "cwd": cwd, "env": env, "check": check})

    monkeypatch.setattr("lmx.freemhd.subprocess.run", fake_run)

    output = run_freemhd_demo(tmp_path, demo_kind="hunt", nproc=4, extra_env={"A": "B"})

    assert output == tmp_path / "freemhd_output" / "hunt"
    assert calls[0]["args"] == [str(tmp_path / "run_hunt.sh"), "4"]
    assert calls[0]["cwd"] == tmp_path
    assert calls[0]["check"] is True
    assert calls[0]["env"]["A"] == "B"

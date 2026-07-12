from pathlib import Path

import pytest

from lmx.freemhd import (
    audit_freemhd_case_against_spec,
    build_case_from_freemhd_reference,
    candidate_u_paths,
    compare_side_jet_profiles,
    infer_initial_velocity_x,
    infer_inlet_drive_mode,
    infer_inlet_flow_rate,
    infer_liquid_material_properties,
    infer_liquid_properties,
    infer_magnetic_ramp,
    infer_rectangular_geometry,
    infer_reduced_inlet_flow_rate,
    infer_solid_conductivities,
    infer_uniform_b0,
    load_benchmark_a_spec,
    load_samper_table_i,
    parse_freemhd_execution_seconds,
    run_freemhd_demo,
    side_jet_profile_metrics,
    summarize_observable_ladder_levels,
    summarize_observable_gate,
    summarize_observable_offenders,
    summarize_profile_error_offenders,
    summarize_runtime_offenders,
    write_observable_ladder_table,
)
from scripts.materialize_freemhd_benchmark_a import materialize_matched_freemhd_case


pytestmark = pytest.mark.unit


def _write_matched_freemhd_case(root: Path, case_kind: str) -> None:
    spec = load_benchmark_a_spec(case_kind)
    liquid = root / "case" / "constant" / "liquid"
    liquid.mkdir(parents=True)
    fluid = spec["fluid"]
    (liquid / "thermophysicalProperties.liquidMetal").write_text(
        f"rho {fluid['density']};\nmu {fluid['dynamic_viscosity']};\nelcond {fluid['conductivity']};\n"
    )
    for region, conductivity in (
        ("solidWalls", spec["wall"]["conducting_wall_conductivity"]),
        ("insulator", spec["wall"]["insulating_wall_conductivity"]),
    ):
        directory = root / "case" / "constant" / region
        directory.mkdir(parents=True)
        (directory / "thermophysicalProperties").write_text(f"elcond {conductivity};\n")
    initial = root / "case" / "0" / "liquid"
    initial.mkdir(parents=True)
    field = " ".join(str(value) for value in spec["magnetic_field"]["vector"])
    (initial / "B0").write_text(f"internalField uniform ( {field} );\n")
    (initial / "U").write_text(
        "boundaryField\n{\n"
        "  inlet\n  {\n    type flowRateInletVelocity;\n"
        f"    volumetricFlowRate {spec['drive']['target_flow_rate']};\n  }}\n}}\n"
    )
    system = root / "case" / "system"
    system.mkdir(parents=True)
    geometry = spec["geometry"]
    outer = float(geometry["length_scale"]) + float(geometry["wall_thickness"])
    (system / "blockMeshDict").write_text(
        f"Ly {geometry['length_scale']};\nLy_wall {outer};\n"
        f"N_wall {geometry['wall_cells']};\nHa {spec['magnetic_field']['hartmann_number']};\n"
    )


def _write_demo_template(root: Path) -> None:
    liquid_zero = root / "0" / "liquid"
    liquid_zero.mkdir(parents=True)
    (liquid_zero / "B0").write_text("internalField uniform ( 0 10 0 );\n")
    (liquid_zero / "U").write_text(
        "internalField uniform ( 0.9725 0 0 );\n"
        "boundaryField { inlet { type flowRateInletVelocity; volumetricFlowRate 0.0389; "
        "value uniform ( 0.9725 0 0 ); } }\n"
    )
    for region in ("solidWalls", "insulator"):
        zero = root / "0" / region
        zero.mkdir(parents=True)
        (zero / "B0").write_text("internalField uniform ( 0 10 0 );\n")
        constant = root / "constant" / region
        constant.mkdir(parents=True)
        (constant / "thermophysicalProperties").write_text("elcond 1e-6;\n")
        system = root / "system" / region
        system.mkdir(parents=True)
        (system / "changeDictionaryDict").write_text(
            "B0 { internalField uniform (0 10 0); value uniform (0 10 0); }\n"
        )
    liquid_constant = root / "constant" / "liquid"
    liquid_constant.mkdir(parents=True)
    (liquid_constant / "thermophysicalProperties.liquidMetal").write_text(
        "rho 1000;\nmu 1;\nelcond [-1 -3 3 0 0 2 0] 1e6;\n"
    )
    liquid_system = root / "system" / "liquid"
    liquid_system.mkdir(parents=True)
    (liquid_system / "changeDictionaryDict").write_text(
        "U { internalField uniform ( 0.9725 0 0 ); volumetricFlowRate 0.0389; }\n"
        "B0 { internalField uniform (0 10 0); value uniform (0 10 0); }\n"
    )
    (root / "system" / "blockMeshDict").write_text(
        "Ly 0.1;\nLy_wall 0.101;\nHa 20;\nN_wall 2;\n"
    )


@pytest.mark.parametrize("case_kind", ["shercliff", "hunt"])
def test_matched_benchmark_a_specs_are_dimensionally_consistent(case_kind: str):
    spec = load_benchmark_a_spec(case_kind)

    assert spec["magnetic_field"]["hartmann_number"] == pytest.approx(20.0)
    assert spec["fluid"]["kinematic_viscosity"] == pytest.approx(1.0e-3)
    assert spec["normalization"]["per_profile_peak_fitting"] is False
    assert len(spec["mesh"]["levels"]) == 4
    assert len(spec["sha256"]) == 64


@pytest.mark.parametrize("case_kind", ["shercliff", "hunt"])
def test_freemhd_case_audit_accepts_only_mechanically_matched_inputs(
    tmp_path: Path, case_kind: str
):
    _write_matched_freemhd_case(tmp_path, case_kind)
    report = audit_freemhd_case_against_spec(tmp_path, case_kind=case_kind)

    assert report["matched"] is True
    assert report["failed_check_count"] == 0
    assert report["physical_hartmann_number"] == pytest.approx(20.0)


def test_freemhd_case_audit_exposes_mislabeled_ha_and_hunt_wall(tmp_path: Path):
    _write_matched_freemhd_case(tmp_path, "hunt")
    (tmp_path / "case" / "0" / "liquid" / "B0").write_text(
        "internalField uniform ( 0 10 0 );\n"
    )
    (
        tmp_path / "case" / "constant" / "solidWalls" / "thermophysicalProperties"
    ).write_text("elcond 1e-6;\n")
    report = audit_freemhd_case_against_spec(tmp_path, case_kind="hunt")
    failed_names = {check["name"] for check in report["checks"] if not check["pass"]}

    assert report["matched"] is False
    assert report["physical_hartmann_number"] == pytest.approx(1000.0)
    assert "magnetic_field.vector" in failed_names
    assert "physics.hartmann" in failed_names
    assert "wall.conducting_wall_conductivity" in failed_names


@pytest.mark.parametrize("case_kind", ["shercliff", "hunt"])
def test_materialize_matched_freemhd_case_is_audited_and_refuses_overwrite(
    tmp_path: Path, case_kind: str
):
    template = tmp_path / "template"
    output = tmp_path / "output"
    second_output = tmp_path / "second-output"
    _write_demo_template(template)

    manifest = materialize_matched_freemhd_case(template, output, case_kind=case_kind)

    assert manifest["run_profile"] == "docker_smoke_only"
    assert manifest["audit"]["matched"] is True
    assert len(manifest["source_template_sha256"]) == 64
    assert (output / "lmx-benchmark-manifest.json").is_file()
    assert infer_uniform_b0(output) == pytest.approx((0.0, 0.2, 0.0))
    assert infer_inlet_flow_rate(output) == pytest.approx(
        load_benchmark_a_spec(case_kind)["drive"]["target_flow_rate"]
    )
    assert (
        materialize_matched_freemhd_case(template, second_output, case_kind=case_kind)
        == manifest
    )
    assert (second_output / "lmx-benchmark-manifest.json").read_bytes() == (
        output / "lmx-benchmark-manifest.json"
    ).read_bytes()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        materialize_matched_freemhd_case(template, output, case_kind=case_kind)


def test_benchmark_a_spec_loader_rejects_inconsistent_inputs(tmp_path: Path):
    with pytest.raises(ValueError, match="Unsupported matched Benchmark-A"):
        load_benchmark_a_spec("hartmann")

    source_dir = Path("benchmarks/specs")

    def rejected(case_kind: str, old: str, new: str, message: str) -> None:
        case_dir = tmp_path / message.replace(" ", "_")
        case_dir.mkdir()
        source = source_dir / f"{case_kind}-ha20.toml"
        text = source.read_text().replace(old, new, 1)
        (case_dir / source.name).write_text(text)
        with pytest.raises(ValueError, match=message):
            load_benchmark_a_spec(case_kind, case_dir)

    rejected(
        "shercliff",
        "schema_version = 1",
        "schema_version = 2",
        "Invalid matched benchmark identity",
    )
    rejected(
        "shercliff",
        "kinematic_viscosity = 1.0e-3",
        "kinematic_viscosity = 2.0e-3",
        "Inconsistent dynamic",
    )
    rejected(
        "shercliff",
        "vector = [0.0, 0.2, 0.0]",
        "vector = [0.0, 0.3, 0.0]",
        "do not reproduce Ha",
    )
    rejected("shercliff", "[65, 49]", "[57, 43]", "refinement ratios are too uneven")
    rejected(
        "hunt",
        "conductance_ratio = 0.05",
        "conductance_ratio = 0.06",
        "do not reproduce the conductance ratio",
    )


def test_samper_table_i_reference_is_complete_and_exact(tmp_path: Path):
    table = load_samper_table_i()
    rows = {(row["case_kind"], row["hartmann_number"]): row for row in table["cases"]}

    assert len(table["sha256"]) == 64
    assert rows[("shercliff", 500)]["analytical_flow_rate"] == pytest.approx(7.680e-3)
    assert rows[("shercliff", 15000)]["published_numerical_flow_rate"] == pytest.approx(
        2.648e-4
    )
    assert rows[("hunt", 500)]["hartmann_wall_conductance"] == pytest.approx(0.01)
    assert rows[("hunt", 15000)]["analytical_flow_rate"] == pytest.approx(2.425e-6)

    source = Path("benchmarks/references/samper-table-i.toml")
    invalid = tmp_path / "invalid.toml"
    invalid.write_text(
        source.read_text().replace('case_kind = "hunt"', 'case_kind = "other"', 1)
    )
    with pytest.raises(ValueError, match="Incomplete hunt Hartmann ladder"):
        load_samper_table_i(invalid)


def test_freemhd_audit_reports_missing_inputs_and_explicit_nu(tmp_path: Path):
    report = audit_freemhd_case_against_spec(tmp_path, case_kind="shercliff")
    failed_names = {check["name"] for check in report["checks"] if not check["pass"]}
    assert "geometry.available" in failed_names
    assert "fluid.available" in failed_names
    assert report["physical_hartmann_number"] is None

    liquid = tmp_path / "constant" / "liquid"
    liquid.mkdir(parents=True)
    (liquid / "thermophysicalProperties").write_text("sigma 3;\nrho 1000;\nnu 2e-6;\n")
    properties = infer_liquid_material_properties(tmp_path)
    assert properties == pytest.approx(
        {
            "conductivity": 3.0,
            "density": 1000.0,
            "dynamic_viscosity": 2.0e-3,
            "kinematic_viscosity": 2.0e-6,
        }
    )


@pytest.mark.external
@pytest.mark.parametrize("case_kind", ["shercliff", "hunt"])
def test_local_freemhd_demo_is_audited_before_parity_claim(case_kind: str):
    case_dir = (
        Path("/Users/rogerio/local/tests/freemhd_install/freemhd_output") / case_kind
    )
    if not case_dir.exists():
        pytest.skip("fresh local FreeMHD Docker output is unavailable")

    report = audit_freemhd_case_against_spec(case_dir, case_kind=case_kind)
    failed_names = {check["name"] for check in report["checks"] if not check["pass"]}
    assert report["matched"] is False
    assert report["physical_hartmann_number"] == pytest.approx(1000.0)
    assert "magnetic_field.vector" in failed_names
    if case_kind == "hunt":
        assert "wall.conducting_wall_conductivity" in failed_names


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


def test_build_case_from_freemhd_reference_preserves_default_forcing_for_shercliff(
    tmp_path: Path,
):
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


def test_build_case_from_freemhd_reference_switches_to_inlet_flow_rate_when_reference_uses_it(
    tmp_path: Path,
):
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

    assert infer_liquid_properties(tmp_path) == pytest.approx((1.0e6, 1000.0, 1.0e-6))
    assert infer_liquid_material_properties(tmp_path) == pytest.approx(
        {
            "conductivity": 1.0e6,
            "density": 1000.0,
            "dynamic_viscosity": 1.0e-3,
            "kinematic_viscosity": 1.0e-6,
        }
    )
    assert infer_uniform_b0(tmp_path) == pytest.approx((0.0, 0.2, 0.0))
    width, height, wall_thickness, wall_cells = infer_rectangular_geometry(tmp_path)
    assert width == pytest.approx(0.2)
    assert height == pytest.approx(0.2)
    assert wall_thickness == pytest.approx(0.001)
    assert wall_cells == 2
    assert infer_solid_conductivities(tmp_path) == pytest.approx((5.0e6, 1.0e-6))


def test_build_case_from_freemhd_reference_adopts_reference_geometry_materials_and_b0(
    tmp_path: Path,
):
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
    assert case.regions[0].viscosity == pytest.approx(1.0e-6)
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
    (incomplete_liquid / "thermophysicalProperties").write_text(
        "sigma 3.0;\nrho 1000;\n"
    )
    assert infer_liquid_properties(tmp_path) is None

    (incomplete_liquid / "thermophysicalProperties").write_text(
        "sigma 3.0;\nrho 1000;\nmu 0.002;\n"
    )
    assert infer_liquid_properties(tmp_path) == pytest.approx((3.0, 1000.0, 2.0e-6))

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
    assert (
        infer_reduced_inlet_flow_rate(tmp_path, reduced_area=1.0, initial_velocity=0.2)
        is None
    )

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
                    "y": {
                        "l2_error": 1.0,
                        "linf_error": 1.0,
                        "reference_peak_abs": 1.0e-8,
                    },
                    "z": {
                        "l2_error": 2.0e-2,
                        "linf_error": 5.0e-2,
                        "reference_peak_abs": 1.0,
                    },
                },
            },
        }
    ]
    observable_offenders = summarize_observable_offenders(
        observable_records, l2_target=1.0e-2
    )
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
    profile_offenders = summarize_profile_error_offenders(
        profile_records, l2_target=1.0e-2, top_n=2
    )
    assert [item["case_kind"] for item in profile_offenders] == ["hunt", "hunt"]
    runtime_offenders = summarize_runtime_offenders(profile_records)
    assert runtime_offenders[0]["case_kind"] == "hunt"
    assert runtime_offenders[0]["status"] == "offender"
    assert runtime_offenders[1]["status"] == "pass"


def test_observable_ladder_summary_ranks_mesh_and_offender_progress(tmp_path: Path):
    good_record = {
        "case_kind": "hunt",
        "observables": {
            name: {
                "y": {
                    "l2_error": 8.0e-3,
                    "linf_error": 1.0e-2,
                    "reference_peak_abs": 1.0,
                },
                "z": {
                    "l2_error": 7.0e-3,
                    "linf_error": 9.0e-3,
                    "reference_peak_abs": 1.0,
                },
            }
            for name in ("velocity", "potential", "current", "lorentz")
        },
        "layer_resolution": {
            "hartmann_layer_cells": 10.0,
            "side_layer_cells": 7.0,
            "hartmann_layer_cell_ratio": 1.25,
            "side_layer_cell_ratio": 1.1,
            "minimum_mesh_refinement_factor": 0.9,
        },
    }
    bad_record = {
        **good_record,
        "observables": {
            name: {
                "y": {
                    "l2_error": 2.0e-2,
                    "linf_error": 3.0e-2,
                    "reference_peak_abs": 1.0,
                },
                "z": {
                    "l2_error": 9.0e-3,
                    "linf_error": 1.0e-2,
                    "reference_peak_abs": 1.0,
                },
            }
            for name in ("velocity", "potential", "current", "lorentz")
        },
        "layer_resolution": {
            "hartmann_layer_cells": 4.0,
            "side_layer_cells": 3.0,
            "hartmann_layer_cell_ratio": 0.5,
            "side_layer_cell_ratio": 0.5,
            "minimum_mesh_refinement_factor": 2.0,
        },
    }

    summary = summarize_observable_ladder_levels(
        [
            {"label": "coarse", "records": [bad_record]},
            {"label": "refined", "records": [good_record]},
        ]
    )
    table = write_observable_ladder_table(summary, tmp_path / "ladder.csv")

    assert summary["best_level_label"] == "refined"
    assert summary["rows"][0]["observable_offender_count"] == 4
    assert summary["rows"][0]["max_minimum_mesh_refinement_factor"] == pytest.approx(
        2.0
    )
    assert summary["rows"][1]["research_grade_validation_pass"] is True
    assert table.exists()
    assert "top_offender_observable" in table.read_text()


def test_build_case_from_freemhd_reference_covers_hartmann_velocity_mode_and_errors(
    tmp_path: Path,
):
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


def test_run_freemhd_demo_invokes_script_and_returns_output_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
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

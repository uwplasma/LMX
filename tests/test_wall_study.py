from pathlib import Path

import pytest

from lmx import (
    DEFAULT_LI_ALN_CASE,
    DEFAULT_SUBSTRATE_CONDUCTIVITIES,
    li_aln_multilayer_mesh_summary,
    li_aln_phase0_2_summary,
    li_aln_phase3_6_summary,
    li_aln_unit_audit,
    li_aln_wall_layers,
    li_aln_wall_stacks_by_side,
    write_li_aln_multilayer_mesh_artifacts,
    write_li_aln_phase0_2_artifacts,
    write_li_aln_phase3_6_artifacts,
)


pytestmark = pytest.mark.unit


def test_li_aln_unit_audit_uses_kinematic_viscosity_and_inductionless_check():
    audit = li_aln_unit_audit(DEFAULT_LI_ALN_CASE)

    assert audit["viscosity_convention"] == "kinematic_nu_m2_per_s"
    assert audit["kinematic_viscosity_m2_s"] == pytest.approx(
        DEFAULT_LI_ALN_CASE.lithium.dynamic_viscosity / DEFAULT_LI_ALN_CASE.lithium.density
    )
    assert audit["hartmann_number"] > 0.0
    assert audit["reynolds_number"] > 0.0
    assert audit["interaction_parameter"] > 0.0
    assert audit["magnetic_reynolds_number"] < 1.0e-2
    assert audit["inductionless_assumption_pass"] is True


def test_li_aln_phase0_2_summary_tracks_pinhole_limits_and_scope():
    summary = li_aln_phase0_2_summary(
        DEFAULT_LI_ALN_CASE,
        conductance_ratios=(0.0, 1.0e-4, 1.0),
        pinhole_fractions=(0.0, 1.0),
    )
    rows = summary["response_rows"]
    ideal = next(row for row in rows if row["wall_model"] == "ideal_insulator")
    bare = next(row for row in rows if row["wall_model"] == "bare_metal")

    assert summary["material_compatibility_claim"] is False
    assert summary["wall_stack"]["mesh_resolution"]["resolution_pass"] is True
    assert ideal["current_closure_proxy"] == pytest.approx(0.0)
    assert bare["current_closure_proxy"] > 0.0
    assert summary["thresholds"]["max_effective_conductance_ratio_for_10pct_deviation"] is not None


def test_li_aln_wall_layers_support_degraded_aln_conductivity():
    intact, metal = li_aln_wall_layers(DEFAULT_LI_ALN_CASE)
    degraded, _ = li_aln_wall_layers(
        DEFAULT_LI_ALN_CASE,
        aln_conductivity=DEFAULT_LI_ALN_CASE.degraded_aln_conductivity,
    )

    assert intact.name == "aln"
    assert metal.name == DEFAULT_LI_ALN_CASE.metal_name
    assert degraded.conductivity > intact.conductivity


def test_write_li_aln_phase0_2_artifacts(tmp_path: Path):
    outputs = write_li_aln_phase0_2_artifacts(
        tmp_path,
        conductance_ratios=(0.0, 1.0e-4, 1.0),
        pinhole_fractions=(0.0, 1.0e-2, 1.0),
    )

    assert [path.suffix for path in outputs] == [".json", ".csv", ".csv", ".png"]
    assert all(path.exists() and path.stat().st_size > 0 for path in outputs)


def test_li_aln_phase3_6_summary_reports_operating_and_degradation_thresholds():
    summary = li_aln_phase3_6_summary(
        DEFAULT_LI_ALN_CASE,
        magnetic_fields=(1.0, 2.0),
        velocities=(0.02, 0.04),
        aln_conductivities=(1.0e-10, 1.0e-8),
        pinhole_fractions=(0.0, 1.0e-4, 1.0e-2),
        tolerances=(0.10,),
    )
    thresholds = summary["threshold_rows"]
    ten_316l = next(row for row in thresholds if row["substrate"] == "316L")
    ten_moly = next(row for row in thresholds if row["substrate"] == "molybdenum")

    assert summary["material_compatibility_claim"] is False
    assert summary["phase_status"]["phase_5_degradation_thresholds"] == "complete_for_reduced_tangential_and_normal_thresholds"
    assert len(summary["operating_rows"]) == 4
    assert ten_316l["critical_effective_conductance_ratio"] == pytest.approx(1.0 / 9.0)
    assert ten_316l["minimum_aln_thickness_for_normal_leakage_m"] > 0.0
    assert ten_316l["maximum_aln_thickness_for_tangential_conductance_m"] > 0.0
    assert ten_moly["maximum_pinhole_fraction"] < ten_316l["maximum_pinhole_fraction"]
    assert DEFAULT_SUBSTRATE_CONDUCTIVITIES["molybdenum"] > DEFAULT_SUBSTRATE_CONDUCTIVITIES["316L"]


def test_write_li_aln_phase3_6_artifacts(tmp_path: Path):
    outputs = write_li_aln_phase3_6_artifacts(
        tmp_path,
        magnetic_fields=(1.0,),
        velocities=(0.02,),
        aln_conductivities=(1.0e-10, 1.0e-8),
        pinhole_fractions=(0.0, 1.0e-4),
    )

    assert [path.suffix for path in outputs] == [".json", ".csv", ".csv", ".csv", ".png"]
    assert all(path.exists() and path.stat().st_size > 0 for path in outputs)


def test_li_aln_multilayer_mesh_summary_tracks_interfaces_and_regions():
    summary = li_aln_multilayer_mesh_summary(DEFAULT_LI_ALN_CASE, ny=12, nz=10)
    rows = summary["wall_stack"]["interfaces"]
    regions = summary["wall_stack"]["regions"]

    assert summary["material_compatibility_claim"] is False
    assert summary["qa"]["cell_count_pass"] is True
    assert summary["qa"]["interface_faces_aligned"] is True
    assert summary["qa"]["ready_for_conservative_current_diagnostics"] is True
    assert len(rows) == 8
    assert any(row["inner_region"] == "fluid" and row["outer_region"] == "aln" for row in rows)
    assert any(row["inner_region"] == "aln" and row["outer_region"] == "316L" for row in rows)
    assert any(row["name"] == "left:aln" for row in regions)
    assert any(row["name"] == "right:316L" for row in regions)


def test_li_aln_wall_stacks_by_side_supports_degraded_aln():
    stacks = li_aln_wall_stacks_by_side(DEFAULT_LI_ALN_CASE, aln_conductivity=DEFAULT_LI_ALN_CASE.degraded_aln_conductivity)

    assert sorted(stacks) == ["bottom", "left", "right", "top"]
    assert stacks["left"][0].name == "aln"
    assert stacks["left"][0].conductivity == pytest.approx(DEFAULT_LI_ALN_CASE.degraded_aln_conductivity)
    assert stacks["left"][1].name == DEFAULT_LI_ALN_CASE.metal_name


def test_write_li_aln_multilayer_mesh_artifacts(tmp_path: Path):
    outputs = write_li_aln_multilayer_mesh_artifacts(tmp_path, ny=10, nz=10)

    assert [path.suffix for path in outputs] == [".json", ".csv", ".csv", ".csv", ".png"]
    assert all(path.exists() and path.stat().st_size > 0 for path in outputs)

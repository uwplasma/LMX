from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import lmx.wall_study as wall_study
from lmx.wall_study import (
    DEFAULT_LI_ALN_CASE,
    DEFAULT_SUBSTRATE_CONDUCTIVITIES,
    build_li_aln_multilayer_solve_case,
    li_aln_multilayer_convergence_summary,
    li_aln_multilayer_mesh_summary,
    li_aln_multilayer_solve_summary,
    li_aln_multilayer_wall_model_stacks,
    li_aln_phase0_2_summary,
    li_aln_phase3_6_summary,
    li_aln_unit_audit,
    li_aln_wall_layers,
    li_aln_wall_stacks_by_side,
    write_li_aln_multilayer_mesh_artifacts,
    write_li_aln_multilayer_convergence_artifacts,
    write_li_aln_multilayer_solve_artifacts,
    write_li_aln_phase0_2_artifacts,
    write_li_aln_phase3_6_artifacts,
)


pytestmark = pytest.mark.unit


def test_li_aln_unit_audit_uses_kinematic_viscosity_and_inductionless_check():
    audit = li_aln_unit_audit(DEFAULT_LI_ALN_CASE)

    assert audit["viscosity_convention"] == "kinematic_nu_m2_per_s"
    assert audit["kinematic_viscosity_m2_s"] == pytest.approx(
        DEFAULT_LI_ALN_CASE.lithium.dynamic_viscosity
        / DEFAULT_LI_ALN_CASE.lithium.density
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
    assert (
        summary["thresholds"]["max_effective_conductance_ratio_for_10pct_deviation"]
        is not None
    )


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
    assert (
        summary["phase_status"]["phase_5_degradation_thresholds"]
        == "complete_for_reduced_tangential_and_normal_thresholds"
    )
    assert len(summary["operating_rows"]) == 4
    assert ten_316l["critical_effective_conductance_ratio"] == pytest.approx(1.0 / 9.0)
    assert ten_316l["minimum_aln_thickness_for_normal_leakage_m"] > 0.0
    assert ten_316l["maximum_aln_thickness_for_tangential_conductance_m"] > 0.0
    assert ten_moly["maximum_pinhole_fraction"] < ten_316l["maximum_pinhole_fraction"]
    assert (
        DEFAULT_SUBSTRATE_CONDUCTIVITIES["molybdenum"]
        > DEFAULT_SUBSTRATE_CONDUCTIVITIES["316L"]
    )


def test_write_li_aln_phase3_6_artifacts(tmp_path: Path):
    outputs = write_li_aln_phase3_6_artifacts(
        tmp_path,
        magnetic_fields=(1.0,),
        velocities=(0.02,),
        aln_conductivities=(1.0e-10, 1.0e-8),
        pinhole_fractions=(0.0, 1.0e-4),
    )

    assert [path.suffix for path in outputs] == [
        ".json",
        ".csv",
        ".csv",
        ".csv",
        ".png",
    ]
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
    assert any(
        row["inner_region"] == "fluid" and row["outer_region"] == "aln" for row in rows
    )
    assert any(
        row["inner_region"] == "aln" and row["outer_region"] == "316L" for row in rows
    )
    assert any(row["name"] == "left:aln" for row in regions)
    assert any(row["name"] == "right:316L" for row in regions)


def test_li_aln_wall_stacks_by_side_supports_degraded_aln():
    stacks = li_aln_wall_stacks_by_side(
        DEFAULT_LI_ALN_CASE,
        aln_conductivity=DEFAULT_LI_ALN_CASE.degraded_aln_conductivity,
    )

    assert sorted(stacks) == ["bottom", "left", "right", "top"]
    assert stacks["left"][0].name == "aln"
    assert stacks["left"][0].conductivity == pytest.approx(
        DEFAULT_LI_ALN_CASE.degraded_aln_conductivity
    )
    assert stacks["left"][1].name == DEFAULT_LI_ALN_CASE.metal_name


def test_li_aln_multilayer_solve_case_uses_explicit_sigma_and_flow_rate():
    solver_case, mesh, stacks = build_li_aln_multilayer_solve_case(
        DEFAULT_LI_ALN_CASE,
        wall_model="intact_aln",
        ny=8,
        nz=6,
        magnetic_field=0.02,
        velocity=0.01,
        max_steps=2,
    )

    assert solver_case.forcing == 0.0
    assert solver_case.boundary_conditions[-1].kind == "inlet_flow_rate"
    assert mesh.sigma is not None
    assert mesh.fluid_mask is not None
    assert sorted(stacks) == ["bottom", "left", "right", "top"]
    assert mesh.yz_shape[0] > 8
    assert mesh.yz_shape[1] > 6


def test_li_aln_multilayer_wall_model_stacks_rank_conductive_limit():
    intact = li_aln_multilayer_wall_model_stacks(
        DEFAULT_LI_ALN_CASE, wall_model="intact_aln"
    )
    bare = li_aln_multilayer_wall_model_stacks(
        DEFAULT_LI_ALN_CASE, wall_model="bare_metal"
    )

    assert intact["left"][0].name == "aln"
    assert len(bare["left"]) == 1
    assert bare["left"][0].conductivity == pytest.approx(
        DEFAULT_LI_ALN_CASE.metal_conductivity
    )


def test_li_aln_multilayer_solve_summary_tracks_conservation(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(wall_study, "solve_steady", _fake_multilayer_solution)

    summary = li_aln_multilayer_solve_summary(
        DEFAULT_LI_ALN_CASE,
        wall_models=("intact_aln", "bare_metal"),
        ny=4,
        nz=4,
        magnetic_field=0.02,
        velocity=0.01,
        max_steps=2,
    )

    assert summary["scope"] == "solved_multilayer_internal_limiting_case"
    assert summary["external_code_parity_claim"] is False
    assert summary["qa"]["charge_balance_pass"] is True
    assert len(summary["observable_rows"]) == 2
    bare = next(
        row for row in summary["observable_rows"] if row["wall_model"] == "bare_metal"
    )
    intact = next(
        row for row in summary["observable_rows"] if row["wall_model"] == "intact_aln"
    )
    assert bare["tangential_conductance_ratio"] > intact["tangential_conductance_ratio"]


def test_write_li_aln_multilayer_solve_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(wall_study, "solve_steady", _fake_multilayer_solution)

    outputs = write_li_aln_multilayer_solve_artifacts(
        tmp_path,
        wall_models=("intact_aln", "bare_metal"),
        ny=4,
        nz=4,
        magnetic_field=0.02,
        velocity=0.01,
        max_steps=2,
    )

    assert [path.suffix for path in outputs] == [".json", ".csv", ".png"]
    assert all(path.exists() and path.stat().st_size > 0 for path in outputs)


def test_li_aln_multilayer_convergence_summary_tracks_last_step_changes(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        wall_study, "li_aln_multilayer_solve_summary", _fake_convergence_solve_summary
    )

    summary = li_aln_multilayer_convergence_summary(
        DEFAULT_LI_ALN_CASE,
        wall_models=("intact_aln", "bare_metal"),
        resolutions=(10, 12, 14),
    )

    assert summary["scope"] == "solved_multilayer_mesh_ladder_internal_gate"
    assert summary["qa"]["pressure_last_step_relative_change_pass"] is True
    assert summary["qa"]["current_last_step_relative_change_pass"] is True
    assert len(summary["convergence_rows"]) == 6
    assert len(summary["model_rows"]) == 2


def test_write_li_aln_multilayer_convergence_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        wall_study, "li_aln_multilayer_solve_summary", _fake_convergence_solve_summary
    )

    outputs = write_li_aln_multilayer_convergence_artifacts(
        tmp_path,
        wall_models=("intact_aln", "bare_metal"),
        resolutions=(10, 12, 14),
    )

    assert [path.suffix for path in outputs] == [".json", ".csv", ".png"]
    assert all(path.exists() and path.stat().st_size > 0 for path in outputs)


def _fake_multilayer_solution(case, *, mesh):
    fluid = np.asarray(mesh.fluid_mask, dtype=bool)
    u = np.where(fluid, float(case.initial_velocity), 0.0)
    zeros = np.zeros(mesh.yz_shape, dtype=float)
    diagnostics = SimpleNamespace(
        pressure_proxy_history=np.asarray([1.0]),
        current_scaled_pressure_proxy_history=np.asarray([1.0]),
        volumetric_flow_rate_history=np.asarray(
            [float(case.boundary_conditions[-1].value)]
        ),
        mean_current_magnitude_history=np.asarray([1.0e-3]),
        current_max_history=np.asarray([2.0e-3]),
        face_current_max_history=np.asarray([2.5e-3]),
        lorentz_power_history=np.asarray([1.0e-6]),
        div_current_max_history=np.asarray([1.0e-4]),
        charge_balance_residual_history=np.asarray([1.0e-8]),
        interface_current_residual_history=np.asarray([1.0e-5]),
        potential_residual_history=np.asarray([1.0e-8]),
        linear_residual_history=np.asarray([1.0e-8]),
    )
    state = SimpleNamespace(
        u=u,
        phi=zeros,
        jy=zeros,
        jz=zeros,
        lorentz_x=zeros,
        time=float(case.time_stepper.t_final),
        residual=0.0,
    )
    return SimpleNamespace(
        state=state, diagnostics=diagnostics, mesh=mesh, case_name=case.name
    )


def _fake_convergence_solve_summary(case, *, wall_models, ny, nz, **kwargs):
    rows = []
    for model in wall_models:
        base = 10.0 if model == "bare_metal" else 1.0
        rows.append(
            {
                "wall_model": model,
                "mesh_ny": int(ny) + 24,
                "mesh_nz": int(nz) + 24,
                "pressure_proxy": base * (1.0 + 1.0 / int(ny)),
                "mean_current_magnitude": base * 2.0 * (1.0 + 0.5 / int(ny)),
                "charge_balance_relative": 1.0e-5,
                "div_current_relative": 1.0e-3,
                "interface_current_relative": 5.0e-2,
            }
        )
    return {"observable_rows": rows}


def test_write_li_aln_multilayer_mesh_artifacts(tmp_path: Path):
    outputs = write_li_aln_multilayer_mesh_artifacts(tmp_path, ny=10, nz=10)

    assert [path.suffix for path in outputs] == [
        ".json",
        ".csv",
        ".csv",
        ".csv",
        ".png",
    ]
    assert all(path.exists() and path.stat().st_size > 0 for path in outputs)

from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
import json
import shutil

import jax.numpy as jnp
import pytest

import scripts.run_samper_table_i as samper_runner
from lmx.freemhd import load_samper_table_i
from scripts.build_benchmark_a_acceptance import build_acceptance, write_acceptance
from scripts.run_samper_table_i import (
    build_samper_case,
    dimensionless_flow_rate,
    freeze_campaign,
    merge_campaigns,
    parse_mesh_levels,
    reassess_campaign,
    select_rows,
    summarize_refinement,
)


RESULTS = Path("benchmarks/results")


def _compact_campaign(*, passed: bool = True) -> dict[str, object]:
    return {
        "schema_version": 1,
        "records": [
            {
                "case_kind": "shercliff",
                "hartmann_number": 5000,
                "finest_level_pass": passed,
            }
        ],
        "research_grade_validation_pass": passed,
    }


def _copy_acceptance_evidence(destination: Path) -> None:
    destination.mkdir()
    for pattern in ("benchmark-a-ha20-*.json", "samper-table-i-accepted.json"):
        for path in RESULTS.glob(pattern):
            shutil.copy2(path, destination / path.name)


def _row(case_kind: str, ha: int = 500):
    table = load_samper_table_i()
    return next(
        row
        for row in table["cases"]
        if row["case_kind"] == case_kind and row["hartmann_number"] == ha
    )


def test_build_samper_shercliff_case_makes_q_equal_q_tilde() -> None:
    case = build_samper_case(_row("shercliff"), ny=63, nz=63)
    fluid = case.regions[0]
    assert case.geometry.width == case.geometry.height == 2.0
    assert fluid.density == fluid.viscosity == fluid.conductivity == 1.0
    assert case.magnetic_field.value == pytest.approx((0.0, 500.0, 0.0))
    assert case.forcing == 1.0
    assert case.time_stepper.current_reconstruction == "face_averaged"
    assert case.solver.coupling_acceleration == "anderson"
    assert case.solver.coupling_history_depth == 6
    assert case.solver.coupling_iterations == 512
    assert case.solver.linear_solver == "solvax_pcg"
    assert case.time_stepper.max_steps == 12
    assert case.time_stepper.steady_tolerance == pytest.approx(
        samper_runner.STEADY_RELATIVE_UPDATE_TARGET
        * float(_row("shercliff")["analytical_flow_rate"])
    )
    assert case.time_stepper.potential_tolerance == pytest.approx(9.6e-5)
    high_ha = build_samper_case(_row("shercliff", 5000), ny=63, nz=63)
    assert high_ha.time_stepper.potential_tolerance == pytest.approx(9.8775e-5)
    assert (
        build_samper_case(
            _row("shercliff"), ny=63, nz=63, linear_solver="solvax_pcg"
        ).solver.linear_solver
        == "solvax_pcg"
    )


def test_build_samper_hunt_case_preserves_wall_conductance() -> None:
    case = build_samper_case(_row("hunt"), ny=63, nz=63, wall_thickness=0.01)
    fluid, wall, insulator = case.regions
    conductance = wall.conductivity * wall.wall_thickness / (fluid.conductivity * 1.0)
    assert conductance == pytest.approx(0.01)
    assert insulator.conductivity == 0.0
    assert case.geometry.wall_cells == (4, 4, 4, 4)
    assert case.time_stepper.steady_tolerance == pytest.approx(
        samper_runner.STEADY_RELATIVE_UPDATE_TARGET
        * float(_row("hunt")["analytical_flow_rate"])
    )


def test_dimensionless_flow_conversion_uses_actual_forcing() -> None:
    case = build_samper_case(_row("shercliff"), ny=63, nz=63)
    solution = SimpleNamespace(
        diagnostics=SimpleNamespace(
            volumetric_flow_rate_history=jnp.asarray([0.4]),
            applied_forcing_history=jnp.asarray([2.0]),
        )
    )
    assert dimensionless_flow_rate(case, solution) == pytest.approx(0.2)
    solution.diagnostics.applied_forcing_history = jnp.asarray([0.0])
    with pytest.raises(ValueError, match="nonzero"):
        dimensionless_flow_rate(case, solution)


def test_mesh_parser_and_table_selection_are_strict() -> None:
    assert parse_mesh_levels("63x63,79x79,99x99") == ((63, 63), (79, 79), (99, 99))
    for value, message in (
        ("63x63,79x79", "at least three"),
        ("63x63,60x60,99x99", "monotonically"),
        ("63,79x79,99x99", "expected NYxNZ"),
    ):
        with pytest.raises(ValueError, match=message):
            parse_mesh_levels(value)
    rows = select_rows(
        load_samper_table_i(), case_kinds=("hunt",), hartmann_numbers=(500, 5000)
    )
    assert [(row["case_kind"], row["hartmann_number"]) for row in rows] == [
        ("hunt", 500),
        ("hunt", 5000),
    ]


def test_implementation_fingerprint_covers_runner_and_solver_core() -> None:
    fingerprint = samper_runner._implementation_fingerprint()
    assert set(fingerprint) == {
        "runner_sha256",
        "solver_core_sha256",
        "lmx_version",
        "solvax_version",
    }
    assert len(fingerprint["runner_sha256"]) == 64
    assert len(fingerprint["solver_core_sha256"]) == 64


def test_refinement_gate_requires_order_monotonicity_and_small_finest_change() -> None:
    levels = [
        {"mesh": [63, 63], "q_tilde": 0.9, "analytical_relative_error": 0.1},
        {"mesh": [79, 79], "q_tilde": 0.998, "analytical_relative_error": 0.01},
        {"mesh": [99, 99], "q_tilde": 1.0, "analytical_relative_error": 0.001},
    ]
    summary = summarize_refinement(levels)
    assert summary["pass"] is True
    levels[-1]["q_tilde"] = 1.1
    assert summarize_refinement(levels)["pass"] is False
    with pytest.raises(ValueError, match="at least three"):
        summarize_refinement(levels[:2])


def test_campaign_writes_a_compact_aggregate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[dict[str, object]]] = []

    def fake_run_row(row, mesh_levels, checkpoint, initial_levels=(), **controls):
        calls.append(list(initial_levels))
        checkpoint(
            [
                *initial_levels,
                {"mesh": list(mesh_levels[len(initial_levels)]), "q_tilde": 0.1},
            ]
        )
        return {
            "case_kind": row["case_kind"],
            "hartmann_number": row["hartmann_number"],
            "mesh_levels": mesh_levels,
            "controls": controls,
            "finest_level_pass": True,
        }

    monkeypatch.setattr(
        samper_runner,
        "run_row",
        fake_run_row,
    )
    output = tmp_path / "table-i.json"
    summary = samper_runner.run_campaign(
        output=output,
        case_kinds=("shercliff",),
        hartmann_numbers=(500,),
        mesh_levels=((8, 8), (10, 10), (12, 12)),
        max_steps=4,
    )
    assert summary["research_grade_validation_pass"] is True
    assert summary["reference"]["sha256"] == load_samper_table_i()["sha256"]
    assert summary["records"][0]["controls"] == {"max_steps": 4}
    assert "active_record" not in summary
    assert output.read_text(encoding="utf-8").endswith("\n")
    assert not output.with_suffix(".json.tmp").exists()

    calls.clear()
    resumed = samper_runner.run_campaign(
        output=output,
        case_kinds=("shercliff",),
        hartmann_numbers=(500,),
        mesh_levels=((8, 8), (10, 10), (12, 12)),
        max_steps=4,
        resume=True,
    )
    assert resumed == samper_runner.json.loads(samper_runner.json.dumps(summary))
    assert calls == []

    partial = dict(summary)
    partial["records"] = []
    partial["research_grade_validation_pass"] = False
    partial["active_record"] = {
        "case_kind": "shercliff",
        "hartmann_number": 500,
        "levels": [{"mesh": [8, 8], "q_tilde": 0.1}],
        "complete": False,
    }
    output.write_text(samper_runner.json.dumps(partial), encoding="utf-8")
    calls.clear()
    resumed = samper_runner.run_campaign(
        output=output,
        case_kinds=("shercliff",),
        hartmann_numbers=(500,),
        mesh_levels=((8, 8), (10, 10), (12, 12)),
        max_steps=4,
        resume=True,
    )
    assert calls == [[{"mesh": [8, 8], "q_tilde": 0.1}]]
    assert resumed["research_grade_validation_pass"] is True

    with pytest.raises(ValueError, match="contract mismatch: controls"):
        samper_runner.run_campaign(
            output=output,
            case_kinds=("shercliff",),
            hartmann_numbers=(500,),
            mesh_levels=((8, 8), (10, 10), (12, 12)),
            max_steps=5,
            resume=True,
        )


def test_run_row_rejects_nonprefix_checkpoint_mesh() -> None:
    with pytest.raises(ValueError, match="not a prefix"):
        samper_runner.run_row(
            _row("shercliff"),
            ((8, 8), (10, 10), (12, 12)),
            initial_levels=({"mesh": [9, 9]},),
        )


def test_reassess_campaign_uses_physical_current_gate_for_potential() -> None:
    level = {
        "analytical_relative_error": 0.005,
        "solver": {"residual": 5.0e-10, "potential_residual": 1.0},
        "solver_convergence": {},
        "layer_resolution": {"layer_resolution_pass": True},
        "balances": {"current": {"pass": True}, "power": {"pass": True}},
    }
    summary = {
        "potential_residual_target": 2.0e-6,
        "records": [
            {
                "analytical_flow_rate": 1.0,
                "levels": [dict(level), dict(level), dict(level)],
                "refinement": {"pass": True},
                "finest_level_pass": False,
            }
        ],
    }
    reassessed = reassess_campaign(summary)
    assert reassessed["research_grade_validation_pass"] is True
    assert reassessed["records"][0]["levels"][-1]["solver_convergence"]["pass"] is True
    assert "potential_residual_target" not in reassessed
    reassessed["records"][0]["levels"][-1]["solver"]["residual"] = 2.0e-9
    assert reassess_campaign(reassessed)["research_grade_validation_pass"] is False


def test_freeze_campaign_writes_deterministic_compact_evidence(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    destination = tmp_path / "frozen.json"
    source.write_text(json.dumps(_compact_campaign()))
    payload = freeze_campaign(source, destination)
    assert payload["freeze"]["format"] == "compact-json"
    assert len(payload["freeze"]["source_sha256"]) == 64
    assert destination.read_text().endswith("\n")


def test_freeze_campaign_rejects_incomplete_or_failing_evidence(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.json"
    incomplete = _compact_campaign()
    incomplete["active_record"] = {}
    source.write_text(json.dumps(incomplete))
    with pytest.raises(ValueError, match="incomplete"):
        freeze_campaign(source, tmp_path / "frozen.json")
    source.write_text(json.dumps(_compact_campaign(passed=False)))
    with pytest.raises(ValueError, match="failing"):
        freeze_campaign(source, tmp_path / "frozen.json")


def test_merge_campaigns_deduplicates_contract_and_preserves_sources(tmp_path: Path) -> None:
    sources = []
    for index, mesh in enumerate(([[8, 8]], [[8, 8], [10, 10]])):
        payload = _compact_campaign()
        payload["mesh_levels"] = mesh
        payload["records"][0]["hartmann_number"] = 5000 + index
        source = tmp_path / f"source-{index}.json"
        source.write_text(json.dumps(payload))
        sources.append(source)
    merged = merge_campaigns(sources, tmp_path / "merged.json")
    assert merged["mesh_levels"] == [[8, 8], [10, 10]]
    assert len(merged["records"]) == 2
    assert set(merged["freeze"]["source_sha256_by_file"]) == {
        "source-0.json",
        "source-1.json",
    }

    incompatible = json.loads(sources[1].read_text())
    incompatible["schema_version"] = 2
    sources[1].write_text(json.dumps(incompatible))
    with pytest.raises(ValueError, match="one contract"):
        merge_campaigns(sources, tmp_path / "rejected.json")


def test_build_acceptance_separates_claims_and_accepts_all_rows() -> None:
    payload = build_acceptance(RESULTS)
    assert payload["research_grade_validation_pass"] is True
    assert payload["finite_grid_freemhd"]["pass"] is True
    assert payload["analytical_continuum_audit"]["pass"] is True
    assert payload["conservation_and_power"]["pass"] is True
    assert payload["richardson_diagnostic"]["acceptance_gate"] is False
    assert payload["richardson_diagnostic"]["all_extrapolated_primary_pass"] is False
    assert len(payload["literature_table_i"]["rows"]) == 8


def test_write_acceptance_is_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    write_acceptance(RESULTS, first)
    write_acceptance(RESULTS, second)
    assert first.read_bytes() == second.read_bytes()


@pytest.mark.parametrize("failure", ("fingerprint", "row"))
def test_build_acceptance_rejects_inconsistent_or_failing_evidence(
    tmp_path: Path, failure: str
) -> None:
    copied = tmp_path / "results"
    _copy_acceptance_evidence(copied)
    path = copied / "samper-table-i-accepted.json"
    payload = json.loads(path.read_text())
    if failure == "fingerprint":
        payload["implementation"]["solver_core_sha256"] = "0" * 64
    else:
        next(
            row
            for row in payload["records"]
            if row["case_kind"] == "hunt" and row["hartmann_number"] == 15000
        )["finest_level_pass"] = False
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="implementation|finest-level"):
        build_acceptance(copied)

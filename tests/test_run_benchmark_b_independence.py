from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_benchmark_b_independence as campaign


pytestmark = pytest.mark.unit


def test_variant_problem_applies_only_frozen_solver_control_changes():
    baseline = campaign._variant_problem("B1-fringing-pipe", "coarse", "baseline")
    tight = campaign._variant_problem("B1-fringing-pipe", "coarse", "tight_tolerance")
    extended = campaign._variant_problem(
        "B1-fringing-pipe", "coarse", "extended_iterations"
    )
    thin = campaign._variant_problem("B1-fringing-pipe", "coarse", "thin_wall")

    assert tight.case.solver.coupling_tolerance == pytest.approx(
        0.5 * baseline.case.solver.coupling_tolerance
    )
    assert (
        tight.case.time_stepper.potential_iterations
        == 2 * baseline.case.time_stepper.potential_iterations
    )
    assert (
        extended.case.solver.coupling_iterations
        == 2 * baseline.case.solver.coupling_iterations
    )
    assert (
        extended.case.time_stepper.max_steps == 2 * baseline.case.time_stepper.max_steps
    )
    assert thin.case.geometry.wall_thickness[0] == pytest.approx(
        0.5 * baseline.case.geometry.wall_thickness[0]
    )
    baseline_conductivity = baseline.case.regions[1].conductivity
    thin_conductivity = thin.case.regions[1].conductivity
    assert thin_conductivity == pytest.approx(2.0 * baseline_conductivity)

    with pytest.raises(ValueError, match="Unsupported independence variant"):
        campaign._variant_problem("B1-fringing-pipe", "coarse", "unknown")


def _record(observable, *, residual=1.0e-9):
    return {
        "primary_observable": observable,
        "diagnostics": {
            "max_residual": residual,
            "max_divergence_residual": 1.0e-5,
            "max_charge_balance_residual": 1.0e-5,
            "volumetric_flow_rate_span": 1.0e-5,
            "max_wall_current_leakage": 0.0,
            "net_boundary_current_residual": 0.0,
        },
    }


def test_comparison_applies_uncertainty_and_thin_wall_gates():
    records = {
        "baseline": _record([0.1, 0.2, 0.1]),
        "tight_tolerance": _record([0.1001, 0.2001, 0.1001]),
        "extended_iterations": _record([0.1001, 0.1999, 0.1001]),
        "thin_wall": _record([0.1001, 0.2001, 0.1001]),
    }
    comparison = campaign._comparison("B2-fringing-square", records)
    assert comparison["complete"] is True
    assert comparison["pass"] is True
    assert all(comparison["gates"].values())

    records["baseline"] = _record([0.1, 0.2, 0.1], residual=1.0)
    failed = campaign._comparison("B2-fringing-square", records)
    assert failed["pass"] is False
    assert failed["gates"]["steady_residual"] is False

    records["baseline"] = _record([0.1, 0.2, 0.1])
    records["baseline"]["diagnostics"]["max_divergence_residual"] = 1.0
    failed = campaign._comparison("B2-fringing-square", records)
    assert failed["pass"] is False
    assert failed["gates"]["mass_balance"] is False

    records["baseline"] = _record([0.1, 0.2, 0.1])
    records["thin_wall"] = _record([0.1001, 0.2001, 0.1001], residual=1.0)
    failed = campaign._comparison("B2-fringing-square", records)
    assert failed["pass"] is False
    assert failed["gates"]["thin-wall_steady_residual"] is False

    incomplete = campaign._comparison(
        "B2-fringing-square", {"baseline": records["baseline"]}
    )
    assert incomplete["complete"] is False
    assert "thin_wall" in incomplete["missing_variants"]


def test_dry_run_writes_deterministic_campaign_plan(tmp_path: Path):
    output = tmp_path / "campaign"
    exit_code = campaign.main(
        [
            "--output",
            str(output),
            "--cases",
            "B1-fringing-pipe",
            "--variants",
            "baseline",
            "--dry-run",
        ]
    )
    payload = json.loads((output / "benchmark-b-independence.json").read_text())
    assert exit_code == 0
    assert payload["pass"] is False
    assert payload["cases"] == [
        {"case_id": "B1-fringing-pipe", "complete": False, "dry_run": True}
    ]


def test_resume_rejects_checkpoint_from_another_fingerprint(tmp_path: Path):
    output = tmp_path / "campaign"
    checkpoint = output / "runs" / "B1-fringing-pipe-coarse-baseline.json"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text(json.dumps({"source_fingerprint": "stale"}))
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        campaign.main(
            [
                "--output",
                str(output),
                "--cases",
                "B1-fringing-pipe",
                "--variants",
                "baseline",
                "--resume",
            ]
        )


def test_variant_restart_parser_is_explicit_and_rejects_invalid_values():
    assert campaign._parse_variant_restarts(
        ["thin_wall=/tmp/thin.npz", "baseline=/tmp/base.npz"]
    ) == {
        "thin_wall": Path("/tmp/thin.npz"),
        "baseline": Path("/tmp/base.npz"),
    }
    with pytest.raises(ValueError, match="VARIANT"):
        campaign._parse_variant_restarts(["thin_wall"])
    with pytest.raises(ValueError, match="VARIANT"):
        campaign._parse_variant_restarts(["unknown=/tmp/x.npz"])


def test_gpu_device_parser_requires_unique_ids():
    assert campaign._parse_gpu_devices("0, 1") == ("0", "1")
    with pytest.raises(ValueError, match="unique"):
        campaign._parse_gpu_devices("")
    with pytest.raises(ValueError, match="unique"):
        campaign._parse_gpu_devices("0,0")


def test_gpu_wave_assigns_one_variant_per_device(monkeypatch: pytest.MonkeyPatch):
    launches = []

    class Process:
        returncode = 2

        def communicate(self):
            return "incomplete comparison", ""

    def fake_popen(command, **kwargs):
        launches.append((command, kwargs))
        return Process()

    monkeypatch.setattr(campaign.subprocess, "Popen", fake_popen)
    args = SimpleNamespace(
        output=Path("artifacts/campaign"),
        mesh_level="coarse",
        resume=True,
        initial_restart=None,
        variant_restart=[],
    )
    campaign._run_gpu_wave(
        args,
        [("B1-fringing-pipe", "baseline"), ("B2-fringing-square", "thin_wall")],
        ("0", "1"),
    )

    assert [item[1]["env"]["CUDA_VISIBLE_DEVICES"] for item in launches] == ["0", "1"]
    assert all(
        item[1]["env"]["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false" for item in launches
    )
    assert "baseline" in launches[0][0]
    assert "thin_wall" in launches[1][0]


def test_gpu_campaign_runs_restart_dependent_variants_in_second_wave(
    monkeypatch: pytest.MonkeyPatch,
):
    waves = []
    monkeypatch.setattr(
        campaign, "_run_gpu_wave", lambda args, tasks, devices: waves.append(tasks)
    )
    monkeypatch.setattr(campaign, "main", lambda argv=None: 7)
    args = SimpleNamespace(
        gpu_devices="0,1",
        output=Path("artifacts/campaign"),
        cases=["B2-fringing-square"],
        mesh_level="coarse",
        variants=list(campaign.VARIANTS),
    )

    assert campaign._run_gpu_campaign(args) == 7
    assert waves == [
        [("B2-fringing-square", "baseline"), ("B2-fringing-square", "thin_wall")],
        [
            ("B2-fringing-square", "tight_tolerance"),
            ("B2-fringing-square", "extended_iterations"),
        ],
    ]

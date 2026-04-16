from pathlib import Path

import pytest

from scripts import run_full_validation_exercise as suite


pytestmark = pytest.mark.unit


def test_run_full_validation_exercise_writes_combined_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    output = tmp_path / "full_validation"

    def fake_validation_main(argv: list[str]) -> int:
        out_index = argv.index("--output") + 1
        out_dir = Path(argv[out_index])
        ha = float(argv[argv.index("--ha") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            f"hartmann_ha{int(ha)}": {
                "case": f"hartmann_ha{int(ha)}",
                "accepted": 1.0,
                "charge_balance_residual": 1.0e-6,
                "interface_current_residual": 1.0e-3,
                "u_max": 1.0,
            },
            f"shercliff_ha{int(ha)}": {
                "case": f"shercliff_ha{int(ha)}",
                "charge_balance_residual": 1.0e-5,
                "interface_current_residual": 2.0e-3,
                "u_max": 0.8,
            },
        }
        (out_dir / "summary.json").write_text(__import__("json").dumps(payload))
        return 0

    def fake_manual_main(argv: list[str]) -> int:
        out_path = Path(argv[argv.index("--output") + 1])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "fringing_rect_duct_ha10_n8": {
                "case": "fringing_rect_duct_ha10",
                "geometry_kind": "rect_duct",
                "solver_kind": "extruded_inductionless",
                "validation_pass": 1.0,
                "max_charge_balance_residual": 1.0e-2,
                "volumetric_flow_rate_span": 1.0e-3,
                "field_mean_velocity_correlation": -0.8,
            },
            "hunt_ha10_n8": {
                "case": "hunt_ha10",
                "geometry_kind": "layered_duct",
                "solver_kind": "fully_developed_inductionless",
                "validation_pass": 1.0,
            },
        }
        out_path.write_text(__import__("json").dumps(payload))
        csv_path = out_path.with_suffix(".csv")
        csv_path.write_text("key,case\nfringing_rect_duct_ha10_n8,fringing_rect_duct_ha10\n")
        return 0

    monkeypatch.setattr(suite.validation_suite, "main", fake_validation_main)
    monkeypatch.setattr(suite.manual_validation, "main", fake_manual_main)

    exit_code = suite.main(
        [
            "--output",
            str(output),
            "--ha-values",
            "10,20",
            "--fringing-resolutions",
            "8",
        ]
    )

    summary = __import__("json").loads((output / "full_validation_summary.json").read_text())
    markdown = (output / "full_validation_summary.md").read_text()
    csv_text = (output / "full_validation_summary.csv").read_text()

    assert exit_code == 0
    assert summary["gates"]["benchmark_a_pass"] == 1
    assert summary["gates"]["benchmark_b_pass"] == 1
    assert summary["gates"]["overall_pass"] == 1
    assert "Benchmark A" in markdown
    assert "Benchmark B" in markdown
    assert "fringing_rect_duct_ha10_n8" in csv_text


def test_run_full_validation_exercise_respects_fail_on_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    output = tmp_path / "full_validation"

    def fake_validation_main(argv: list[str]) -> int:
        out_dir = Path(argv[argv.index("--output") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "summary.json").write_text(
            '{"hartmann_ha10":{"case":"hartmann_ha10","accepted":1.0,"charge_balance_residual":1e-6,"interface_current_residual":1e-3,"u_max":1.0}}'
        )
        return 0

    def fake_manual_main(argv: list[str]) -> int:
        out_path = Path(argv[argv.index("--output") + 1])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            '{"fringing_rect_duct_ha10_n8":{"case":"fringing_rect_duct_ha10","geometry_kind":"rect_duct","solver_kind":"extruded_inductionless","validation_pass":0.0,"max_charge_balance_residual":1.0,"volumetric_flow_rate_span":1.0,"field_mean_velocity_correlation":0.0}}'
        )
        return 1

    monkeypatch.setattr(suite.validation_suite, "main", fake_validation_main)
    monkeypatch.setattr(suite.manual_validation, "main", fake_manual_main)

    exit_code = suite.main(["--output", str(output), "--ha-values", "10", "--fail-on-threshold"])

    assert exit_code == 1

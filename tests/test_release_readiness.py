from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from lmx.publication import PUBLICATION_FIGURE_SPECS
from scripts.run_release_readiness import evaluate_release_readiness, main, write_release_readiness_report


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _write_release_fixture(root: Path) -> None:
    (root / "pyproject.toml").write_text(
        """
[project]
name = "lmx"
version = "1.0.0"
dependencies = ["jax", "jaxlib", "matplotlib", "numpy", "scipy"]

[project.optional-dependencies]
dev = ["pytest", "lineax", "interpax"]
docs = ["sphinx", "myst-parser", "furo", "sphinx-copybutton"]
""".strip()
        + "\n"
    )
    workflow_dir = root / ".github" / "workflows"
    workflow_dir.mkdir(parents=True)
    (workflow_dir / "ci.yml").write_text("name: ci\n")
    (workflow_dir / "docs.yml").write_text("name: docs\n")
    (workflow_dir / "release.yml").write_text(
        "permissions:\n  id-token: write\nsteps:\n  - uses: pypa/gh-action-pypi-publish@release/v1\n"
    )
    static = root / "docs" / "_static" / "generated"
    static.mkdir(parents=True)
    for name in (
        "analytic_velocity_profiles.png",
        "closed_channel_validation_ladder.png",
        "readme_hunt_startup_2d_poster.png",
        "readme_hunt_startup_3d_poster.png",
        "q2d_turbulence_decay_poster.png",
        "q2d_turbulence_reference_observables_template.csv",
        "magnetic_obstacle_benchmark.png",
        "magnetic_obstacle_schematic.png",
        "magnetic_obstacle_reference_observables_template.csv",
        "bent_pipe_overview.png",
        "dean_vortex_reference_observables_template.csv",
        "variable_field_tabulated_reconstruction.png",
        "strong_scaling.png",
    ):
        (static / name).write_bytes(b"artifact")
    for spec in PUBLICATION_FIGURE_SPECS:
        if not (static / spec.artifact).exists():
            (static / spec.artifact).write_bytes(b"artifact")
        if not (static / spec.summary).exists():
            _write_json(static / spec.summary, {"validation": {"validation_pass": True, "value": 1.0}})
    _write_json(
        static / "readme_media_manifest.json",
        {
            "media": [
                {
                    "name": "readme_hunt_startup_2d.gif",
                    "url": "https://github.com/uwplasma/LMX/releases/download/v1.0.2/readme_hunt_startup_2d.gif",
                    "poster": "readme_hunt_startup_2d_poster.png",
                },
                {
                    "name": "readme_hunt_startup_3d.gif",
                    "url": "https://github.com/uwplasma/LMX/releases/download/v1.0.2/readme_hunt_startup_3d.gif",
                    "poster": "readme_hunt_startup_3d_poster.png",
                },
                {
                    "name": "q2d_turbulence_decay.gif",
                    "url": "https://github.com/uwplasma/LMX/releases/download/v1.0.2/q2d_turbulence_decay.gif",
                    "poster": "q2d_turbulence_decay_poster.png",
                },
            ],
            "local_media": [
                {
                    "name": "wham_blanket_flow.gif",
                    "path": "docs/_static/generated/wham_blanket_flow.gif",
                    "poster": "wham_blanket_flow_poster.png",
                }
            ],
            "figures": [
                {"name": "analytic_velocity_profiles.png"},
                {"name": "strong_scaling.png"},
            ],
        },
    )
    (static / "wham_blanket_flow.gif").write_bytes(b"artifact")
    (static / "wham_blanket_flow_poster.png").write_bytes(b"artifact")
    _write_json(
        static / "straight_duct_profile_comparison_summary.json",
        {
            "hartmann": {"l2_error": 0.01},
            "shercliff": {"y_l2_error": 0.006, "z_l2_error": 0.007},
            "hunt": {"y_l2_error": 0.008, "z_l2_error": 0.006},
        },
    )
    _write_json(
        static / "straight_duct_validation_ladder_summary.json",
        {
            "hunt": [
                {"ha": 20, "z_l2_error": 0.006},
                {"ha": 100, "z_l2_error": 0.032},
            ]
        },
    )
    _write_json(
        static / "q2d_turbulence_decay_summary.json",
        {
            "validation": {
                "validation_pass": True,
                "frame_count": 72,
                "turnover_count": 0.33,
                "max_courant": 0.05,
                "max_divergence_linf": 1.0e-14,
                "research_grade_turbulence_validation_pass": False,
            }
        },
    )
    _write_json(
        static / "magnetic_obstacle_benchmark_summary.json",
        {
            "validation": {
                "benchmark_pass": True,
                "conservation_pass": True,
                "peak_centerline_deficit_ratio": 0.2,
                "peak_crosscut_distortion": 0.1,
                "max_charge_balance_residual": 1.0e-12,
                "research_grade_validation_pass": False,
            }
        },
    )
    _write_json(
        static / "bent_pipe_inductionless_summary.json",
        {
            "validation": {
                "validation_pass": True,
                "cross_section_l2_error": 0.0,
                "centerline_l2_error": 0.0,
                "max_charge_balance_residual": 0.02,
                "max_wall_current_leakage": 0.0,
                "net_boundary_current_residual": 0.0,
                "research_grade_charge_balance_pass": False,
                "research_grade_dean_validation_pass": False,
            }
        },
    )
    _write_json(
        static / "variable_field_tabulated_summary.json",
        {
            "field_quality": {"validation_pass": True, "divergence_to_field_ratio": 1.0e-3},
            "reconstruction_quality": {
                "validation_pass": True,
                "relative_l2_error": 1.0e-5,
                "relative_linf_error": 1.0e-4,
                "relative_l2_tolerance": 2.0e-3,
                "relative_linf_tolerance": 1.0e-2,
            },
            "validation": {"validation_pass": True, "max_charge_balance_residual": 1.0e-12},
        },
    )


def test_release_readiness_passes_bounded_release_and_tracks_deferred_lanes(tmp_path: Path):
    _write_release_fixture(tmp_path)

    report = evaluate_release_readiness(tmp_path)

    assert report["release_ready"] is True
    assert report["release_class"] == "bounded"
    assert report["research_grade_ready"] is False
    assert report["blockers"] == []
    assert report["research_blockers"] == report["deferred_research_lanes"]
    assert any("High-Ha Hunt side-layer" in item for item in report["deferred_research_lanes"])
    assert any("Magnetic-obstacle" in item for item in report["deferred_research_lanes"])
    assert any("Bent-pipe" in item for item in report["deferred_research_lanes"])

    output = write_release_readiness_report(report, tmp_path / "artifacts/release/release_readiness.json")
    assert output.exists()


def test_release_readiness_fails_missing_public_artifact(tmp_path: Path):
    _write_release_fixture(tmp_path)
    (tmp_path / "docs/_static/generated/strong_scaling.png").unlink()

    report = evaluate_release_readiness(tmp_path)

    assert report["release_ready"] is False
    assert "required_public_artifacts" in report["blockers"]


def test_release_readiness_fails_missing_manifest_media(tmp_path: Path):
    _write_release_fixture(tmp_path)
    (tmp_path / "docs/_static/generated/wham_blanket_flow.gif").unlink()

    report = evaluate_release_readiness(tmp_path)
    gate = next(item for item in report["gates"] if item["name"] == "readme_external_media_manifest")

    assert report["release_ready"] is False
    assert "readme_external_media_manifest" in report["blockers"]
    assert gate["details"]["missing_local_media"] == ["wham_blanket_flow.gif"]


def test_release_readiness_fails_missing_publication_manifest_artifact(tmp_path: Path):
    _write_release_fixture(tmp_path)
    (tmp_path / "docs/_static/generated/q2d_turbulence_observables.png").unlink()

    report = evaluate_release_readiness(tmp_path)
    gate = next(item for item in report["gates"] if item["name"] == "publication_figure_manifest")

    assert report["release_ready"] is False
    assert "publication_figure_manifest" in report["blockers"]
    assert gate["details"]["missing_artifacts"] == ["q2d_turbulence_observables.png"]


def test_release_readiness_omits_deferred_lanes_when_research_gates_pass(tmp_path: Path):
    _write_release_fixture(tmp_path)
    static = tmp_path / "docs/_static/generated"
    _write_json(static / "straight_duct_validation_ladder_summary.json", {"hunt": [{"ha": 100, "z_l2_error": 0.006}]})
    _write_json(
        static / "q2d_turbulence_decay_summary.json",
        {
            "validation": {
                "validation_pass": True,
                "frame_count": 72,
                "turnover_count": 0.33,
                "max_courant": 0.05,
                "max_divergence_linf": 1.0e-14,
                "research_grade_turbulence_validation_pass": True,
            },
            "external_reference_comparison": {"status": "external_reference_compared", "validation_pass": True},
        },
    )
    _write_json(
        static / "magnetic_obstacle_benchmark_summary.json",
        {
            "validation": {
                "benchmark_pass": True,
                "conservation_pass": True,
                "peak_centerline_deficit_ratio": 0.2,
                "peak_crosscut_distortion": 0.1,
                "max_charge_balance_residual": 1.0e-12,
                "research_grade_validation_pass": True,
            },
            "external_reference_comparison": {"status": "external_reference_compared", "validation_pass": True},
        },
    )
    _write_json(
        static / "bent_pipe_inductionless_summary.json",
        {
            "validation": {
                "validation_pass": True,
                "cross_section_l2_error": 0.0,
                "centerline_l2_error": 0.0,
                "max_charge_balance_residual": 1.0e-12,
                "max_wall_current_leakage": 0.0,
                "net_boundary_current_residual": 0.0,
                "research_grade_charge_balance_pass": True,
                "research_grade_dean_validation_pass": True,
            },
            "external_reference_comparison": {"status": "external_reference_compared", "validation_pass": True},
        },
    )
    for name in (
        "q2d_turbulence_reference_observables.csv",
        "magnetic_obstacle_reference_observables.csv",
        "dean_vortex_reference_observables.csv",
    ):
        (static / name).write_text("observable,value,tolerance\nx,1,1\n")

    report = evaluate_release_readiness(tmp_path)

    assert report["release_ready"] is True
    assert report["release_class"] == "research_grade"
    assert report["research_grade_ready"] is True
    assert report["research_blockers"] == []
    assert report["deferred_research_lanes"] == []


def test_release_readiness_main_writes_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _write_release_fixture(tmp_path)
    output = tmp_path / "artifacts/release/readiness.json"
    monkeypatch.setattr(sys, "argv", ["run_release_readiness.py", "--root", str(tmp_path), "--output", str(output)])

    main()

    payload = json.loads(output.read_text())
    assert payload["release_ready"] is True
    assert payload["release_class"] == "bounded"


def test_release_readiness_main_exits_on_blocker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _write_release_fixture(tmp_path)
    (tmp_path / "docs/_static/generated/strong_scaling.png").unlink()
    output = tmp_path / "artifacts/release/readiness.json"
    monkeypatch.setattr(sys, "argv", ["run_release_readiness.py", "--root", str(tmp_path), "--output", str(output)])

    with pytest.raises(SystemExit):
        main()

    payload = json.loads(output.read_text())
    assert payload["release_ready"] is False


def test_release_readiness_main_strict_research_grade_exits_on_deferred_lane(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _write_release_fixture(tmp_path)
    output = tmp_path / "artifacts/release/readiness.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_release_readiness.py",
            "--root",
            str(tmp_path),
            "--output",
            str(output),
            "--strict-research-grade",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 2
    payload = json.loads(output.read_text())
    assert payload["release_ready"] is True
    assert payload["research_grade_ready"] is False

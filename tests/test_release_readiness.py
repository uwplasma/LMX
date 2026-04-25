from __future__ import annotations

import json
from pathlib import Path

from scripts.run_release_readiness import evaluate_release_readiness, write_release_readiness_report


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
        "readme_hunt_startup_2d.gif",
        "readme_hunt_startup_3d.gif",
        "q2d_turbulence_decay.gif",
        "magnetic_obstacle_benchmark.png",
        "magnetic_obstacle_schematic.png",
        "bent_pipe_overview.png",
        "variable_field_tabulated_reconstruction.png",
        "strong_scaling.png",
    ):
        (static / name).write_bytes(b"artifact")
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
    assert report["blockers"] == []
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

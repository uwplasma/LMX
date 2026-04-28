from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:  # pragma: no cover - Python 3.10 fallback
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]


@dataclass(frozen=True)
class ReleaseGate:
    name: str
    passed: bool
    details: dict[str, Any]


REQUIRED_ARTIFACTS = (
    "analytic_velocity_profiles.png",
    "closed_channel_validation_ladder.png",
    "readme_media_manifest.json",
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
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _load_pyproject(root: Path) -> dict[str, Any]:
    with (root / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def _static_dir(root: Path) -> Path:
    return root / "docs" / "_static" / "generated"


def _gate(name: str, passed: bool, **details: Any) -> ReleaseGate:
    return ReleaseGate(name=name, passed=bool(passed), details=details)


def _required_artifact_gate(root: Path) -> ReleaseGate:
    static_dir = _static_dir(root)
    missing = [name for name in REQUIRED_ARTIFACTS if not (static_dir / name).exists()]
    return _gate(
        "required_public_artifacts",
        not missing,
        required=list(REQUIRED_ARTIFACTS),
        missing=missing,
    )


def _readme_media_gate(root: Path) -> ReleaseGate:
    static_dir = _static_dir(root)
    manifest_path = static_dir / "readme_media_manifest.json"
    try:
        manifest = _load_json(manifest_path)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        return _gate("readme_external_media_manifest", False, error=str(exc))
    media = manifest.get("media", [])
    required_names = {
        "readme_hunt_startup_2d.gif",
        "readme_hunt_startup_3d.gif",
        "q2d_turbulence_decay.gif",
    }
    seen_names = {str(item.get("name", "")) for item in media if isinstance(item, dict)}
    missing_names = sorted(required_names - seen_names)
    missing_posters: list[str] = []
    invalid_urls: list[str] = []
    for item in media:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url", ""))
        poster = str(item.get("poster", ""))
        if not url.startswith("https://github.com/uwplasma/LMX/releases/download/"):
            invalid_urls.append(url)
        if poster and not (static_dir / poster).exists():
            missing_posters.append(poster)
    passed = not missing_names and not missing_posters and not invalid_urls
    return _gate(
        "readme_external_media_manifest",
        passed,
        missing_names=missing_names,
        missing_posters=missing_posters,
        invalid_urls=invalid_urls,
    )


def _packaging_gate(root: Path) -> ReleaseGate:
    pyproject = _load_pyproject(root)
    project = pyproject.get("project", {})
    dependencies = set(project.get("dependencies", []))
    optional = project.get("optional-dependencies", {})
    required_deps = {"jax", "jaxlib", "matplotlib", "numpy", "scipy"}
    dev_deps = set(optional.get("dev", []))
    docs_deps = set(optional.get("docs", []))
    passed = (
        project.get("name") == "lmx"
        and bool(project.get("version"))
        and required_deps.issubset(dependencies)
        and {"pytest", "lineax", "interpax"}.issubset(dev_deps)
        and {"sphinx", "myst-parser", "furo", "sphinx-copybutton"}.issubset(docs_deps)
    )
    return _gate(
        "package_metadata",
        passed,
        package_name=project.get("name"),
        version=project.get("version"),
        missing_runtime_dependencies=sorted(required_deps - dependencies),
        missing_dev_dependencies=sorted({"pytest", "lineax", "interpax"} - dev_deps),
        missing_docs_dependencies=sorted({"sphinx", "myst-parser", "furo", "sphinx-copybutton"} - docs_deps),
    )


def _straight_duct_gate(root: Path, *, target_l2: float) -> tuple[ReleaseGate, list[str]]:
    summary = _load_json(_static_dir(root) / "straight_duct_profile_comparison_summary.json")
    checked = {
        "hartmann_l2": float(summary["hartmann"]["l2_error"]),
        "shercliff_y_l2": float(summary["shercliff"]["y_l2_error"]),
        "shercliff_z_l2": float(summary["shercliff"]["z_l2_error"]),
        "hunt_y_l2": float(summary["hunt"]["y_l2_error"]),
        "hunt_z_l2": float(summary["hunt"]["z_l2_error"]),
    }
    ladder = _load_json(_static_dir(root) / "straight_duct_validation_ladder_summary.json")
    deferred: list[str] = []
    for record in ladder.get("hunt", []):
        if int(float(record.get("ha", 0))) == 100 and float(record.get("z_l2_error", 0.0)) > target_l2:
            deferred.append(
                "High-Ha Hunt side-layer cut remains above the manuscript target "
                f"(Ha=100 z_l2={float(record['z_l2_error']):.3e})."
            )
    return (
        _gate(
            "straight_duct_reader_facing_profiles",
            all(value <= target_l2 for value in checked.values()),
            target_l2=target_l2,
            checked_errors=checked,
        ),
        deferred,
    )


def _q2d_gate(root: Path) -> tuple[ReleaseGate, list[str]]:
    summary = _load_json(_static_dir(root) / "q2d_turbulence_decay_summary.json")
    validation = summary["validation"]
    passed = (
        bool(validation.get("validation_pass"))
        and int(validation.get("frame_count", 0)) >= 24
        and float(validation.get("turnover_count", 0.0)) >= 0.1
        and float(validation.get("max_courant", 1.0)) < 0.45
    )
    deferred = []
    if not bool(validation.get("research_grade_turbulence_validation_pass")):
        deferred.append("Nonlinear Q2D movie is an internal SM82-style physics gate; external turbulent parity remains open.")
    return (
        _gate(
            "q2d_nonlinear_movie_gate",
            passed,
            frame_count=int(validation.get("frame_count", 0)),
            turnover_count=float(validation.get("turnover_count", 0.0)),
            max_courant=float(validation.get("max_courant", 0.0)),
            max_divergence_linf=float(validation.get("max_divergence_linf", 0.0)),
        ),
        deferred,
    )


def _magnetic_obstacle_gate(root: Path) -> tuple[ReleaseGate, list[str]]:
    summary = _load_json(_static_dir(root) / "magnetic_obstacle_benchmark_summary.json")
    validation = summary["validation"]
    passed = (
        bool(validation.get("benchmark_pass"))
        and bool(validation.get("conservation_pass"))
        and float(validation.get("max_charge_balance_residual", 1.0)) <= 1.0e-8
    )
    deferred = []
    if not bool(validation.get("research_grade_validation_pass")):
        deferred.append("Magnetic-obstacle lane is still an internal matched-no-field response gate until external observables are filled.")
    return (
        _gate(
            "magnetic_obstacle_internal_response",
            passed,
            peak_centerline_deficit_ratio=float(validation.get("peak_centerline_deficit_ratio", 0.0)),
            peak_crosscut_distortion=float(validation.get("peak_crosscut_distortion", 0.0)),
            max_charge_balance_residual=float(validation.get("max_charge_balance_residual", 0.0)),
            research_grade_validation_pass=bool(validation.get("research_grade_validation_pass")),
        ),
        deferred,
    )


def _bent_pipe_gate(root: Path) -> tuple[ReleaseGate, list[str]]:
    summary = _load_json(_static_dir(root) / "bent_pipe_inductionless_summary.json")
    validation = summary["validation"]
    passed = (
        bool(validation.get("validation_pass"))
        and float(validation.get("cross_section_l2_error", 1.0)) <= 1.0e-10
        and float(validation.get("centerline_l2_error", 1.0)) <= 1.0e-10
        and float(validation.get("max_wall_current_leakage", 1.0)) <= 1.0e-12
        and float(validation.get("net_boundary_current_residual", 1.0)) <= 1.0e-12
    )
    deferred = []
    if not bool(validation.get("research_grade_charge_balance_pass")):
        deferred.append(
            "Bent-pipe global current closure passes, but local mapped-grid |div J| remains above the research-grade target."
        )
    if not bool(validation.get("research_grade_dean_validation_pass")):
        deferred.append("Higher-inertia Dean-vortex bent-pipe validation remains open.")
    return (
        _gate(
            "bent_pipe_low_de_gate",
            passed,
            cross_section_l2_error=float(validation.get("cross_section_l2_error", 0.0)),
            centerline_l2_error=float(validation.get("centerline_l2_error", 0.0)),
            max_charge_balance_residual=float(validation.get("max_charge_balance_residual", 0.0)),
            max_wall_current_leakage=float(validation.get("max_wall_current_leakage", 0.0)),
            net_boundary_current_residual=float(validation.get("net_boundary_current_residual", 0.0)),
        ),
        deferred,
    )


def _variable_field_gate(root: Path) -> ReleaseGate:
    summary = _load_json(_static_dir(root) / "variable_field_tabulated_summary.json")
    field_quality = summary["field_quality"]
    reconstruction = summary["reconstruction_quality"]
    validation = summary["validation"]
    passed = (
        bool(field_quality.get("validation_pass"))
        and bool(reconstruction.get("validation_pass"))
        and bool(validation.get("validation_pass"))
        and float(reconstruction.get("relative_l2_error", 1.0)) <= float(reconstruction.get("relative_l2_tolerance", 0.0))
        and float(reconstruction.get("relative_linf_error", 1.0)) <= float(reconstruction.get("relative_linf_tolerance", 0.0))
    )
    return _gate(
        "tabulated_variable_field_gate",
        passed,
        divergence_to_field_ratio=float(field_quality.get("divergence_to_field_ratio", 0.0)),
        reconstruction_l2=float(reconstruction.get("relative_l2_error", 0.0)),
        reconstruction_linf=float(reconstruction.get("relative_linf_error", 0.0)),
        max_charge_balance_residual=float(validation.get("max_charge_balance_residual", 0.0)),
    )


def _workflow_gate(root: Path) -> ReleaseGate:
    workflows = [
        root / ".github" / "workflows" / "ci.yml",
        root / ".github" / "workflows" / "docs.yml",
        root / ".github" / "workflows" / "release.yml",
    ]
    missing = [str(path.relative_to(root)) for path in workflows if not path.exists()]
    release_text = (root / ".github" / "workflows" / "release.yml").read_text() if not missing else ""
    passed = not missing and "pypa/gh-action-pypi-publish@release/v1" in release_text and "id-token: write" in release_text
    return _gate("release_workflows", passed, missing=missing)


def evaluate_release_readiness(root: str | Path = ".", *, target_l2: float = 1.2e-2) -> dict[str, Any]:
    root_path = Path(root).resolve()
    gates: list[ReleaseGate] = [
        _packaging_gate(root_path),
        _workflow_gate(root_path),
        _required_artifact_gate(root_path),
        _readme_media_gate(root_path),
    ]
    deferred: list[str] = []
    straight_gate, straight_deferred = _straight_duct_gate(root_path, target_l2=target_l2)
    q2d_gate, q2d_deferred = _q2d_gate(root_path)
    obstacle_gate, obstacle_deferred = _magnetic_obstacle_gate(root_path)
    bent_gate, bent_deferred = _bent_pipe_gate(root_path)
    gates.extend(
        [
            straight_gate,
            q2d_gate,
            obstacle_gate,
            bent_gate,
            _variable_field_gate(root_path),
        ]
    )
    deferred.extend(straight_deferred)
    deferred.extend(q2d_deferred)
    deferred.extend(obstacle_deferred)
    deferred.extend(bent_deferred)
    blockers = [gate.name for gate in gates if not gate.passed]
    pyproject = _load_pyproject(root_path)
    research_grade_ready = not blockers and not deferred
    return {
        "case": "release_readiness",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "version": pyproject.get("project", {}).get("version"),
        "release_ready": not blockers,
        "release_class": "research_grade" if research_grade_ready else "bounded",
        "research_grade_ready": research_grade_ready,
        "blockers": blockers,
        "research_blockers": deferred,
        "deferred_research_lanes": deferred,
        "gates": [asdict(gate) for gate in gates],
    }


def write_release_readiness_report(report: dict[str, Any], output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the bounded LMX release-readiness gates.")
    parser.add_argument("--root", type=Path, default=Path("."), help="Repository root.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/release/release_readiness.json"))
    parser.add_argument("--target-l2", type=float, default=1.2e-2)
    parser.add_argument(
        "--strict-research-grade",
        action="store_true",
        help="Exit nonzero when any deferred research lane remains open.",
    )
    args = parser.parse_args()
    report = evaluate_release_readiness(args.root, target_l2=args.target_l2)
    write_release_readiness_report(report, args.output)
    print(json.dumps(report, indent=2))
    if not report["release_ready"]:
        raise SystemExit(1)
    if args.strict_research_grade and not report["research_grade_ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

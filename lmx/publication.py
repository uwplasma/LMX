"""Publication-figure manifest and readiness helpers."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class PublicationFigureSpec:
    """Specification for one manuscript-facing LMX figure or table."""

    family: str
    artifact: str
    summary: str
    generator: str
    reference: str
    manuscript_role: str
    readiness_status: str
    required_next_step: str


PUBLICATION_FIGURE_SPECS: tuple[PublicationFigureSpec, ...] = (
    PublicationFigureSpec(
        family="closed_duct_profiles",
        artifact="analytic_velocity_profiles.png",
        summary="straight_duct_profile_comparison_summary.json",
        generator="examples/straight_duct_profile_comparison.py",
        reference="Hartmann/Shercliff/Hunt analytical profiles and FreeMHD/Ni wall model",
        manuscript_role="Analytical velocity-profile verification with release L2 gates",
        readiness_status="ready",
        required_next_step="Add the final paper table with mesh, Ha, wall conductance, and all profile errors.",
    ),
    PublicationFigureSpec(
        family="closed_duct_ladder",
        artifact="closed_channel_validation_ladder.png",
        summary="straight_duct_validation_ladder_summary.json",
        generator="examples/straight_duct_validation_ladder.py",
        reference="FreeMHD paper closed-channel analytical solution files",
        manuscript_role="Mesh and Hartmann-number validation ladder",
        readiness_status="ready",
        required_next_step="Keep as a release gate and cite the analytical source in the figure caption.",
    ),
    PublicationFigureSpec(
        family="freemhd_observable_parity",
        artifact="freemhd_closed_channel_observable_parity.png",
        summary="freemhd_closed_channel_observable_parity_summary.json",
        generator="examples/freemhd_closed_channel_observable_parity.py",
        reference="Bundled FreeMHD paper slices and local FreeMHD/OpenFOAM rerun path",
        manuscript_role="Observable-level parity for u, phi, J, JxB and pressure-gradient fields",
        readiness_status="ready_with_documented_offenders",
        required_next_step="Add current-vector/streamline panels and keep offender rows in the paper supplement.",
    ),
    PublicationFigureSpec(
        family="wham_blanket_pressure",
        artifact="wham_blanket_pressure_sweep.png",
        summary="wham_blanket_pressure_sweep_summary.json",
        generator="examples/wham_blanket_flow_demo.py",
        reference="Liquid-metal blanket pressure-drop design literature",
        manuscript_role="Cumulative pressure drop along station and field-strength scaling",
        readiness_status="reduced_design_gate",
        required_next_step="Add velocity, radius, Ha, N, Dean-number nondimensional sweeps.",
    ),
    PublicationFigureSpec(
        family="wham_blanket_transient",
        artifact="wham_blanket_transient_flow.png",
        summary="wham_blanket_transient_flow_summary.json",
        generator="examples/wham_blanket_flow_demo.py",
        reference="Curved-pipe MHD and Dean-flow literature",
        manuscript_role="Startup, pressure approach to steady state, and bend inboard/outboard skew",
        readiness_status="reduced_design_gate",
        required_next_step="Replace the bounded Dean-skew proxy with a resolved secondary-flow solve.",
    ),
    PublicationFigureSpec(
        family="wham_blanket_autodiff",
        artifact="wham_blanket_autodiff_research.png",
        summary="wham_blanket_autodiff_research_summary.json",
        generator="examples/wham_blanket_autodiff_research_demo.py",
        reference="Differentiable reduced pressure-budget model",
        manuscript_role="Coil-separation sensitivity and field-scale inverse design",
        readiness_status="ready_reduced_autodiff",
        required_next_step="Repeat with the resolved curved-pipe operator once available.",
    ),
    PublicationFigureSpec(
        family="li_aln_wall_stack",
        artifact="li_aln_wall_stack_phase0_2.png",
        summary="li_aln_wall_stack_phase0_2_summary.json",
        generator="examples/li_aln_wall_stack_phase0_2.py",
        reference="Thin-wall conductance, coated-wall leakage, and Li/AlN wall-stack study plan",
        manuscript_role="Li/AlN unit audit, nested-wall QA, reduced conductance sweep, and pinhole sensitivity",
        readiness_status="ready_reduced_wall_stack",
        required_next_step="Add true multilayer geometry and FreeMHD/code-to-code limiting-case comparisons.",
    ),
    PublicationFigureSpec(
        family="magnetic_obstacle",
        artifact="magnetic_obstacle_benchmark.png",
        summary="magnetic_obstacle_benchmark_summary.json",
        generator="examples/magnetic_obstacle_benchmark.py",
        reference="Cuevas-Smolentsev-Abdou and Votyakov magnetic-obstacle observables",
        manuscript_role="Localized-field response, wake recovery, pressure proxy, and current closure",
        readiness_status="external_reference_open",
        required_next_step="Fill external reference observables and replace the matched no-field baseline.",
    ),
    PublicationFigureSpec(
        family="q2d_turbulence",
        artifact="q2d_turbulence_observables.png",
        summary="q2d_wall_bounded_validation_summary.json",
        generator="examples/q2d_wall_bounded_validation.py",
        reference="Sommeria-Moreau and Potherat quasi-2D MHD turbulence literature",
        manuscript_role="Energy, spectrum, Hartmann friction, and nonlinear activity observables",
        readiness_status="external_reference_open",
        required_next_step="Add external turbulent parity for energy decay and spectra.",
    ),
    PublicationFigureSpec(
        family="bent_pipe_dean",
        artifact="bent_pipe_overview.png",
        summary="bent_pipe_inductionless_summary.json",
        generator="examples/bent_pipe_inductionless_demo.py",
        reference="Curved-pipe Dean-vortex and MHD bend literature",
        manuscript_role="Curved geometry, local current closure, pressure response, and Dean-skew baseline",
        readiness_status="resolved_dean_validation_open",
        required_next_step="Add a resolved higher-inertia Dean-vortex mesh ladder.",
    ),
    PublicationFigureSpec(
        family="variable_tabulated_field",
        artifact="variable_field_tabulated_reconstruction.png",
        summary="variable_field_tabulated_summary.json",
        generator="examples/variable_field_tabulated_demo.py",
        reference="Tabulated 3D field interpolation and divergence-control checks",
        manuscript_role="Interpolation accuracy, divergence control, and variable-field response",
        readiness_status="ready_internal_gate",
        required_next_step="Add external or manufactured 3D field-line validation.",
    ),
    PublicationFigureSpec(
        family="strong_scaling",
        artifact="strong_scaling.png",
        summary="strong_scaling_summary.json",
        generator="scripts/run_strong_scaling_worker.py",
        reference="Solver-facing CPU/GPU fixed-size scaling campaign",
        manuscript_role="Performance, throughput, memory, and device-scaling evidence",
        readiness_status="ready_bounded_release",
        required_next_step="Repeat after each major operator refactor and include compile/runtime split.",
    ),
    PublicationFigureSpec(
        family="strict_validation_closure_dashboard",
        artifact="research_grade_closure_dashboard.png",
        summary="research_grade_closure_dashboard_summary.json",
        generator="examples/research_grade_closure_dashboard.py",
        reference="Q2DmhdFoam, Votyakov magnetic-obstacle, and Bayat-Rezai Dean-vortex closure artifacts",
        manuscript_role="Single-panel closure ledger for closed support gates and open strict research blockers",
        readiness_status="strict_research_blockers_open",
        required_next_step="Replace the failed magnetic-obstacle and Dean-vortex panels with passed matched solved-physics comparisons.",
    ),
)


def publication_figure_rows(static_dir: str | Path | None = None) -> list[dict[str, Any]]:
    """Return publication figure readiness rows with artifact and metric checks."""

    root = _static_dir(static_dir)
    rows: list[dict[str, Any]] = []
    for spec in PUBLICATION_FIGURE_SPECS:
        summary_path = root / spec.summary
        artifact_path = root / spec.artifact
        summary = _load_summary(summary_path)
        row = {
            **asdict(spec),
            "artifact_path": str(artifact_path),
            "summary_path": str(summary_path),
            "artifact_exists": artifact_path.exists(),
            "summary_exists": summary_path.exists(),
            "selected_metrics": _selected_metrics(spec.family, summary),
        }
        rows.append(row)
    return rows


def publication_figure_campaign_summary(static_dir: str | Path | None = None) -> dict[str, Any]:
    """Summarize manuscript figure readiness for docs and release review."""

    rows = publication_figure_rows(static_dir)
    missing_artifacts = [row["artifact"] for row in rows if not row["artifact_exists"]]
    missing_summaries = [row["summary"] for row in rows if not row["summary_exists"]]
    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row["readiness_status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    external_open = [
        row["family"]
        for row in rows
        if "open" in str(row["readiness_status"]) or "external_reference_open" == row["readiness_status"]
    ]
    return {
        "case": "publication_figure_campaign",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "figure_count": len(rows),
        "artifact_count": sum(1 for row in rows if row["artifact_exists"]),
        "summary_count": sum(1 for row in rows if row["summary_exists"]),
        "missing_artifacts": missing_artifacts,
        "missing_summaries": missing_summaries,
        "status_counts": status_counts,
        "external_or_resolved_validation_open": external_open,
        "release_blocking": bool(missing_artifacts or missing_summaries),
        "paper_ready": bool(not missing_artifacts and not missing_summaries and not external_open),
        "rows": rows,
    }


def write_publication_figure_manifest(
    out_dir: str | Path,
    *,
    static_dir: str | Path | None = None,
    filename_stem: str = "publication_figure_campaign",
) -> list[Path]:
    """Write JSON and CSV manifests for the current publication figure set."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = publication_figure_campaign_summary(static_dir)
    json_path = out / f"{filename_stem}_summary.json"
    csv_path = out / f"{filename_stem}_table.csv"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_rows_csv(csv_path, summary["rows"])
    return [json_path, csv_path]


def _static_dir(static_dir: str | Path | None) -> Path:
    if static_dir is not None:
        return Path(static_dir)
    return Path(__file__).resolve().parents[1] / "docs" / "_static" / "generated"


def _load_summary(path: Path) -> Mapping[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _selected_metrics(family: str, summary: Mapping[str, Any]) -> dict[str, Any]:
    if not summary:
        return {}
    if family == "closed_duct_profiles":
        values = [
            float(summary.get("hartmann", {}).get("l2_error", 0.0)),
            float(summary.get("shercliff", {}).get("y_l2_error", 0.0)),
            float(summary.get("shercliff", {}).get("z_l2_error", 0.0)),
            float(summary.get("hunt", {}).get("y_l2_error", 0.0)),
            float(summary.get("hunt", {}).get("z_l2_error", 0.0)),
        ]
        return {"max_l2_error": max(values)}
    if family == "wham_blanket_pressure":
        return {
            "field_scales": summary.get("field_scales", []),
            "pressure_drop_kpa": summary.get("pressure_drop_kpa", []),
        }
    if family == "wham_blanket_transient":
        metrics = summary.get("metrics", {})
        return {
            "final_mean_velocity_m_per_s": metrics.get("final_mean_velocity_m_per_s"),
            "final_pressure_drop_kpa": metrics.get("final_pressure_drop_kpa"),
            "final_bend_outboard_to_inboard_ratio": metrics.get("final_bend_outboard_to_inboard_ratio"),
            "steady_state_reached": metrics.get("steady_state_reached"),
        }
    if family == "wham_blanket_autodiff":
        return _numeric_subset(summary.get("reference", {}), limit=6)
    if family == "li_aln_wall_stack":
        audit = summary.get("unit_audit", {})
        mesh = summary.get("wall_stack", {}).get("mesh_resolution", {})
        thresholds = summary.get("thresholds", {})
        return {
            "hartmann_number": audit.get("hartmann_number"),
            "reynolds_number": audit.get("reynolds_number"),
            "magnetic_reynolds_number": audit.get("magnetic_reynolds_number"),
            "inductionless_assumption_pass": audit.get("inductionless_assumption_pass"),
            "wall_layer_count": mesh.get("layer_count"),
            "wall_mesh_resolution_pass": mesh.get("resolution_pass"),
            "max_c_eff_10pct": thresholds.get("max_effective_conductance_ratio_for_10pct_deviation"),
            "max_pinhole_10pct": thresholds.get("max_pinhole_fraction_for_10pct_deviation"),
        }
    if family == "strong_scaling":
        return {
            "record_count": len(summary.get("records", [])),
            "plot_count": len(summary.get("plots", [])),
        }
    validation = summary.get("validation") or summary.get("field_quality") or summary.get("metrics") or summary
    return _numeric_subset(validation, limit=8)


def _numeric_subset(payload: Any, *, limit: int) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    selected: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, bool) or isinstance(value, (int, float)):
            selected[str(key)] = value
        if len(selected) >= limit:
            break
    return selected


def _write_rows_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    columns = [
        "family",
        "artifact",
        "summary",
        "generator",
        "reference",
        "manuscript_role",
        "readiness_status",
        "artifact_exists",
        "summary_exists",
        "required_next_step",
        "selected_metrics_json",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **{column: row.get(column, "") for column in columns if column != "selected_metrics_json"},
                    "selected_metrics_json": json.dumps(row.get("selected_metrics", {}), sort_keys=True),
                }
            )

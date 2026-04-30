"""Research-grade closure status helpers for the strict validation lanes."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import textwrap
from typing import Any, Mapping


@dataclass(frozen=True)
class ResearchClosureLaneSpec:
    """Static metadata for one strict research-grade validation blocker."""

    lane: str
    summary: str
    external_summary: str
    reference_csv: str
    primary_artifact: str
    required_physics_gate: str
    required_external_gate: str
    closure_target: str
    next_step: str


RESEARCH_CLOSURE_LANES: tuple[ResearchClosureLaneSpec, ...] = (
    ResearchClosureLaneSpec(
        lane="q2d_turbulence_external_parity",
        summary="q2d_turbulence_decay_summary.json",
        external_summary="q2dmhdfoam_external_reference_summary.json",
        reference_csv="q2d_turbulence_reference_observables.csv",
        primary_artifact="q2d_turbulence_decay_poster.png",
        required_physics_gate="research_grade_turbulence_validation_pass",
        required_external_gate="external_reference_compared",
        closure_target="Matched LMX-vs-Q2DmhdFoam turbulent energy, enstrophy, spectrum, and turnover observables.",
        next_step="Run a matched Q2DmhdFoam turbulent case and fill q2d_turbulence_reference_observables.csv.",
    ),
    ResearchClosureLaneSpec(
        lane="magnetic_obstacle_external_validation",
        summary="magnetic_obstacle_benchmark_summary.json",
        external_summary="magnetic_obstacle_external_reference_template_summary.json",
        reference_csv="magnetic_obstacle_reference_observables.csv",
        primary_artifact="magnetic_obstacle_benchmark.png",
        required_physics_gate="research_grade_validation_pass",
        required_external_gate="external_reference_compared",
        closure_target="External centerline deficit, wake recovery, pressure/drag, and current/Lorentz-force parity.",
        next_step="Modernize an MHD_Solvers_OpenFOAM obstacle case or digitize Votyakov/Cuevas observables.",
    ),
    ResearchClosureLaneSpec(
        lane="dean_vortex_higher_inertia_validation",
        summary="bent_pipe_inductionless_summary.json",
        external_summary="dean_vortex_external_reference_template_summary.json",
        reference_csv="dean_vortex_reference_observables.csv",
        primary_artifact="bent_pipe_overview.png",
        required_physics_gate="research_grade_dean_validation_pass",
        required_external_gate="external_reference_compared",
        closure_target="Resolved secondary-flow, inboard/outboard skew, and pressure-loss parity for higher-De bend flow.",
        next_step="Acquire a Dean-flow reference and add a resolved or documented reduced secondary-flow state in LMX.",
    ),
)


def research_grade_closure_rows(static_dir: str | Path | None = None) -> list[dict[str, Any]]:
    """Return computed closure rows for the strict research-grade blockers."""

    root = _static_dir(static_dir)
    rows: list[dict[str, Any]] = []
    for spec in RESEARCH_CLOSURE_LANES:
        summary = _load_json(root / spec.summary)
        external_summary = _load_json(root / spec.external_summary)
        row = _lane_row(spec, summary, external_summary, root)
        rows.append(row)
    return rows


def research_grade_closure_status(static_dir: str | Path | None = None) -> dict[str, Any]:
    """Return a machine-readable status summary for strict closure work."""

    rows = research_grade_closure_rows(static_dir)
    closed_lanes = [row["lane"] for row in rows if row["closed"]]
    open_lanes = [row["lane"] for row in rows if not row["closed"]]
    return {
        "case": "research_grade_closure_status",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "lane_count": len(rows),
        "closed_lane_count": len(closed_lanes),
        "open_lane_count": len(open_lanes),
        "closed_lanes": closed_lanes,
        "open_lanes": open_lanes,
        "research_grade_ready": bool(rows and not open_lanes),
        "release_blocking": False,
        "strict_research_blocking": bool(open_lanes),
        "rows": rows,
    }


def write_research_grade_closure_status(
    out_dir: str | Path,
    *,
    static_dir: str | Path | None = None,
    filename_stem: str = "research_grade_closure_status",
) -> list[Path]:
    """Write JSON and CSV closure-status artifacts."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = research_grade_closure_status(static_dir)
    json_path = out / f"{filename_stem}.json"
    csv_path = out / f"{filename_stem}.csv"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_rows_csv(csv_path, summary["rows"])
    return [json_path, csv_path]


def research_grade_final_disposition(static_dir: str | Path | None = None) -> dict[str, Any]:
    """Return the final evidence-based disposition for strict open lanes.

    This is intentionally stricter than release readiness. A lane can only be
    marked ``closed`` by :func:`research_grade_closure_status`; this helper
    records whether the final local push produced closure or a documented
    future-work decision with the measured offender.
    """

    root = _static_dir(static_dir)
    closure = research_grade_closure_status(root)
    rows = [
        _q2d_final_disposition(root, closure),
        _magnetic_final_disposition(root, closure),
        _dean_final_disposition(root, closure),
    ]
    closed_rows = [row for row in rows if row["final_decision"] == "closed"]
    deferred_rows = [row for row in rows if row["final_decision"] != "closed"]
    return {
        "case": "research_grade_final_disposition",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "lane_count": len(rows),
        "closed_lane_count": len(closed_rows),
        "deferred_lane_count": len(deferred_rows),
        "research_grade_ready": bool(closure["research_grade_ready"]),
        "final_push_complete": True,
        "final_push_decision": (
            "strict_research_grade_ready"
            if closure["research_grade_ready"]
            else "bounded_release_only_strict_lanes_deferred"
        ),
        "rows": rows,
        "notes": (
            "The final push is evidence-based. Lanes that failed external or "
            "physics gates remain deferred rather than being reclassified as "
            "research-grade closure."
        ),
    }


def write_research_grade_final_disposition(
    out_dir: str | Path,
    *,
    static_dir: str | Path | None = None,
    filename_stem: str = "research_grade_final_disposition",
) -> list[Path]:
    """Write JSON, CSV, and PNG artifacts for the final strict-lane disposition."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = research_grade_final_disposition(static_dir)
    json_path = out / f"{filename_stem}.json"
    csv_path = out / f"{filename_stem}.csv"
    png_path = out / f"{filename_stem}.png"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_final_disposition_csv(csv_path, summary["rows"])
    _write_final_disposition_plot(png_path, summary)
    return [json_path, csv_path, png_path]


def research_grade_external_data_audit(
    *,
    static_dir: str | Path | None = None,
    external_codes_root: str | Path | None = None,
    freemhd_cases_root: str | Path | None = None,
) -> dict[str, Any]:
    """Return a local audit of external data needed by strict blockers.

    The audit is intentionally observational: it records which external code
    outputs and reference CSVs are present locally without promoting any lane to
    research-grade closure. Closure remains controlled by
    :func:`research_grade_closure_status`.
    """

    static_root = _static_dir(static_dir)
    external_root = _external_codes_root(external_codes_root)
    freemhd_root = _freemhd_cases_root(freemhd_cases_root)
    closure = research_grade_closure_status(static_root)
    rows = [
        _q2d_external_audit(static_root, external_root),
        _magnetic_obstacle_external_audit(static_root, external_root),
        _dean_external_audit(static_root, external_root),
        _freemhd_context_audit(freemhd_root),
    ]
    available_rows = [row for row in rows if row.get("source_or_processed_artifact_exists", row["path_exists"])]
    matched_rows = [row for row in rows if row.get("matched_reference_csv_exists")]
    return {
        "case": "research_grade_external_data_audit",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "rows": rows,
        "source_count": len(rows),
        "available_source_count": len(available_rows),
        "matched_reference_csv_count": len(matched_rows),
        "strict_closure": {
            "closed_lane_count": closure["closed_lane_count"],
            "lane_count": closure["lane_count"],
            "open_lanes": closure["open_lanes"],
            "research_grade_ready": closure["research_grade_ready"],
        },
        "notes": (
            "This audit records available external-code and digitized-data inputs. "
            "It does not turn templates or unmatched outputs into validation claims."
        ),
    }


def write_research_grade_external_data_audit(
    out_dir: str | Path,
    *,
    static_dir: str | Path | None = None,
    external_codes_root: str | Path | None = None,
    freemhd_cases_root: str | Path | None = None,
    filename_stem: str = "research_grade_external_data_audit",
) -> Path:
    """Write the strict-blocker external-data audit as JSON."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = research_grade_external_data_audit(
        static_dir=static_dir,
        external_codes_root=external_codes_root,
        freemhd_cases_root=freemhd_cases_root,
    )
    json_path = out / f"{filename_stem}.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return json_path


def _lane_row(
    spec: ResearchClosureLaneSpec,
    summary: Mapping[str, Any],
    external_summary: Mapping[str, Any],
    root: Path,
) -> dict[str, Any]:
    validation = _validation_payload(summary)
    external_comparison = summary.get("external_reference_comparison", {})
    external_status = str(external_comparison.get("status") or external_summary.get("status") or "missing_external_summary")
    physics_gate_pass = bool(validation.get(spec.required_physics_gate, False))
    comparison_pass = bool(external_comparison.get("validation_pass", False))
    external_gate_pass = external_status == spec.required_external_gate and comparison_pass
    reference_csv_exists = (root / spec.reference_csv).exists()
    primary_artifact_exists = (root / spec.primary_artifact).exists()
    closed = bool(summary and physics_gate_pass and external_gate_pass and reference_csv_exists and primary_artifact_exists)
    status = (
        "closed"
        if closed
        else _lane_status(spec, summary, external_summary, external_status, physics_gate_pass, external_gate_pass)
    )
    return {
        **asdict(spec),
        "status": status,
        "closed": closed,
        "summary_exists": bool(summary),
        "external_summary_exists": bool(external_summary),
        "reference_csv_exists": reference_csv_exists,
        "primary_artifact_exists": primary_artifact_exists,
        "physics_gate_pass": physics_gate_pass,
        "external_gate_pass": external_gate_pass,
        "external_reference_status": external_status,
        "selected_metrics": _selected_metrics(spec.lane, validation, external_comparison),
    }


def _lane_status(
    spec: ResearchClosureLaneSpec,
    summary: Mapping[str, Any],
    external_summary: Mapping[str, Any],
    external_status: str,
    physics_gate_pass: bool,
    external_gate_pass: bool,
) -> str:
    if physics_gate_pass and external_gate_pass:
        return "closed"
    if not summary:
        return "missing_lmx_summary"
    if spec.lane == "dean_vortex_higher_inertia_validation" and not physics_gate_pass:
        return "resolved_secondary_flow_open"
    if external_status == "external_reference_csv_missing":
        return "external_reference_observables_open"
    if external_summary and not external_gate_pass:
        return "external_adapter_ready_matched_parity_open"
    return "external_data_acquisition_open"


def _q2d_final_disposition(root: Path, closure: Mapping[str, Any]) -> dict[str, Any]:
    q2d_summary = _load_json(root / "q2d_turbulence_decay_summary.json")
    sidewall_summary = _load_json(root / "q2d_lmx_q2dmhdfoam_lid_driven_parity_summary.json")
    match_audit = _load_json(root / "q2dmhdfoam_lmx_turbulence_match_audit_summary.json")
    row = _closure_row_by_lane(closure, "q2d_turbulence_external_parity")
    validation = _validation_payload(q2d_summary)
    sidewall_closed = bool(sidewall_summary.get("strict_blocker_closed", False))
    reference_csv_exists = bool(row.get("reference_csv_exists", False))
    closed = bool(row.get("closed", False))
    audited_cases = int(match_audit.get("case_count", 0) or 0)
    strict_admissible_cases = list(match_audit.get("strict_admissible_cases", []))
    return {
        "lane": "q2d_turbulence_external_parity",
        "final_decision": "closed" if closed else "defer_strict_turbulent_parity",
        "support_gate": "matched_sidewall_q2dmhdfoam_closed" if sidewall_closed else "support_gate_open",
        "blocking_observable": "matched nonlinear turbulent energy/enstrophy/spectrum/turnover CSV",
        "measured_offender": (
            "q2d_turbulence_reference_observables.csv present"
            if reference_csv_exists
            else f"reference CSV absent; {audited_cases} Q2DmhdFoam cases audited, {len(strict_admissible_cases)} strict-admissible"
        ),
        "evidence_artifact": "q2dmhdfoam_lmx_turbulence_match_audit.png",
        "required_next_physics": (
            "Run Q2DmhdFoam and LMX on the same nonlinear Q2D turbulent case, "
            "with identical forcing, domain, Hartmann friction, and time window."
        ),
        "why_not_closed": (
            "The side-wall field-observable gate is closed, but the match audit "
            "rejects the available external cases as different physical cases."
        ),
        "selected_metrics": {
            "frame_count": validation.get("frame_count"),
            "turnover_count": validation.get("turnover_count"),
            "max_courant": validation.get("max_courant"),
            "support_gate_closed": sidewall_closed,
            "audited_q2dmhdfoam_case_count": audited_cases,
            "strict_admissible_case_count": len(strict_admissible_cases),
        },
    }


def _magnetic_final_disposition(root: Path, closure: Mapping[str, Any]) -> dict[str, Any]:
    summary = _load_json(root / "magnetic_obstacle_votyakov_strict_attempt_summary.json")
    curve_summary = _load_json(root / "magnetic_obstacle_votyakov_curve_validation_summary.json")
    benchmark = _load_json(root / "magnetic_obstacle_benchmark_summary.json")
    row = _closure_row_by_lane(closure, "magnetic_obstacle_external_validation")
    comparison_row = _first_comparison_row(summary)
    lmx_value = comparison_row.get("lmx_value")
    reference_value = comparison_row.get("reference_value")
    relative_error = comparison_row.get("relative_error")
    plateau_gap = curve_summary.get("absolute_gap_to_plateau")
    closed = bool(row.get("closed", False))
    return {
        "lane": "magnetic_obstacle_external_validation",
        "final_decision": "closed" if closed else "defer_inertia_capable_localized_field_solver",
        "support_gate": "internal_localized_field_response_closed",
        "blocking_observable": "minimum_centerline_velocity_ratio",
        "measured_offender": (
            f"LMX={_format_metric(lmx_value)}, Votyakov={_format_metric(reference_value)}, "
            f"relative_error={_format_metric(relative_error)}, plateau_gap={_format_metric(plateau_gap)}"
        ),
        "evidence_artifact": "magnetic_obstacle_votyakov_curve_comparison.png",
        "required_next_physics": (
            "Add or couple an inertia-capable localized-field magnetic-obstacle "
            "solve, or run a geometry-matched external code case and compare "
            "centerline velocity, wake recovery, pressure/drag, current, and Lorentz observables."
        ),
        "why_not_closed": (
            "The current reduced LMX case remains positive on the centerline, "
            "while the digitized Votyakov target is recirculating."
        ),
        "selected_metrics": {
            "peak_centerline_deficit_ratio": _validation_payload(benchmark).get("peak_centerline_deficit_ratio"),
            "max_charge_balance_residual": _validation_payload(benchmark).get("max_charge_balance_residual"),
            "external_validation_pass": _external_comparison_validation(summary),
            "target_plateau_minimum_centerline_velocity_ratio": curve_summary.get(
                "target_plateau_minimum_centerline_velocity_ratio"
            ),
            "absolute_gap_to_plateau": plateau_gap,
        },
    }


def _dean_final_disposition(root: Path, closure: Mapping[str, Any]) -> dict[str, Any]:
    summary = _load_json(root / "dean_vortex_bayat_rezai_strict_attempt_summary.json")
    bent = _load_json(root / "bent_pipe_inductionless_summary.json")
    row = _closure_row_by_lane(closure, "dean_vortex_higher_inertia_validation")
    comparison_rows = _comparison_rows(summary)
    rms_row = next((item for item in comparison_rows if item.get("observable") == "secondary_flow_rms_ratio"), {})
    peak_row = next((item for item in comparison_rows if item.get("observable") == "secondary_flow_peak_ratio"), {})
    validation = _validation_payload(bent)
    closed = bool(row.get("closed", False))
    return {
        "lane": "dean_vortex_higher_inertia_validation",
        "final_decision": "closed" if closed else "defer_resolved_secondary_flow_solver",
        "support_gate": "low_de_charge_closure_closed",
        "blocking_observable": "secondary_flow_rms_ratio and secondary_flow_peak_ratio",
        "measured_offender": (
            f"rms LMX={_format_metric(rms_row.get('lmx_value'))}, target={_format_metric(rms_row.get('reference_value'))}; "
            f"peak LMX={_format_metric(peak_row.get('lmx_value'))}, target={_format_metric(peak_row.get('reference_value'))}"
        ),
        "evidence_artifact": "dean_vortex_reference_comparison.png",
        "required_next_physics": (
            "Add resolved secondary velocity/pressure coupling for higher-De curved-pipe flow "
            "or compare against a geometry-matched external curved-pipe solve."
        ),
        "why_not_closed": (
            "The current bent-pipe solver is a low-De current-closure baseline "
            "and does not generate Dean secondary vortices."
        ),
        "selected_metrics": {
            "current_lmx_dean_number": summary.get("current_lmx_dean_number"),
            "reference_dean_number": summary.get("reference_dean_number"),
            "research_grade_charge_balance_pass": validation.get("research_grade_charge_balance_pass"),
            "external_validation_pass": _external_comparison_validation(summary),
        },
    }


def _q2d_external_audit(static_root: Path, external_root: Path) -> dict[str, Any]:
    root = external_root / "Q2DmhdFoam"
    evidence = [
        root / "run/lidDriven/IDM_output_U.txt",
        root / "run/lidDriven/lidDrivenFFT_U.png",
        root / "run/lidDriven/lidDrivenFieldProfile_U.png",
        root / "run/muck_q2d/forcesCo/250000/forceCoeffs.dat",
        root / "run/muck_q2d_FFT/postProcessing/probes/0/U",
        static_root / "q2dmhdfoam_external_reference_summary.json",
        static_root / "q2dmhdfoam_docker_reference_validation_summary.json",
        static_root / "q2dmhdfoam_lid_driven_turbulence_observables.csv",
        static_root / "q2dmhdfoam_timeseries_observables.csv",
    ]
    return _audit_row(
        lane="q2d_turbulence_external_parity",
        source="Q2DmhdFoam local checkout, Docker rerun, and adapter artifacts",
        path=root,
        evidence=evidence,
        matched_reference_csv=static_root / "q2d_turbulence_reference_observables.csv",
        closure_summary=static_root / "q2d_turbulence_decay_summary.json",
        next_step="Run a matched LMX-vs-Q2DmhdFoam turbulent case and export energy/enstrophy/spectrum/force/probe observables.",
    )


def _magnetic_obstacle_external_audit(static_root: Path, external_root: Path) -> dict[str, Any]:
    root = external_root / "MHD_Solvers_OpenFOAM"
    evidence = [
        root / "solvers/mhdEpotFoam/mhdEpotFoam.C",
        root / "Examples/mhdEpotFoam/readme.txt",
        static_root / "magnetic_obstacle_external_reference_template_summary.json",
        static_root / "magnetic_obstacle_benchmark_summary.json",
    ]
    return _audit_row(
        lane="magnetic_obstacle_external_validation",
        source="MHD_Solvers_OpenFOAM plus Cuevas/Votyakov/Andreev literature targets",
        path=root,
        evidence=evidence,
        matched_reference_csv=static_root / "magnetic_obstacle_reference_observables.csv",
        closure_summary=static_root / "magnetic_obstacle_benchmark_summary.json",
        next_step="Fill or generate centerline-deficit, wake-recovery, pressure, and current/Lorentz observables.",
    )


def _dean_external_audit(static_root: Path, external_root: Path) -> dict[str, Any]:
    root = external_root / "OpenFOAM-curved-pipe"
    fallback_root = external_root / "MHD_Solvers_OpenFOAM"
    evidence = [
        static_root / "dean_vortex_external_reference_template_summary.json",
        static_root / "bent_pipe_inductionless_summary.json",
        static_root / "dean_literature_validation_summary.json",
        static_root / "dean_literature_reference_observables.csv",
        static_root / "dean_literature_validation.png",
        fallback_root / "Examples/mhdEpotFoam/readme.txt",
    ]
    return _audit_row(
        lane="dean_vortex_higher_inertia_validation",
        source="Bayat-Rezai Dean literature gate plus a future OpenFOAM curved-pipe reference case",
        path=root,
        evidence=evidence,
        matched_reference_csv=static_root / "dean_vortex_reference_observables.csv",
        closure_summary=static_root / "bent_pipe_inductionless_summary.json",
        next_step="Use the Bayat-Rezai gate to construct/acquire solved secondary-flow, velocity-skew, centroid-shift, and pressure-loss observables.",
    )


def _freemhd_context_audit(freemhd_root: Path) -> dict[str, Any]:
    evidence = [
        freemhd_root / "FreeMHDPaperAllFigures/ClosedChannel/hunt_exactBL_Ha100_XSlice1m_4.12s.csv",
        freemhd_root / "FreeMHDPaperAllFigures/ClosedChannel/shercliff_Ha100_ConstantQ_OutletZeroGradientInletCodedUxBpotE_XSlice1m_3.94s.csv",
        freemhd_root / "freemhd_paper.pdf",
    ]
    return _audit_row(
        lane="straight_duct_and_fringing_context",
        source="FreeMHD processed paper slices and local Docker rerun context",
        path=freemhd_root,
        evidence=evidence,
        matched_reference_csv=None,
        closure_summary=None,
        next_step="Keep using these files as context for closed straight-duct and deferred fringing parity campaigns.",
    )


def _closure_row_by_lane(closure: Mapping[str, Any], lane: str) -> Mapping[str, Any]:
    for row in closure.get("rows", []):
        if isinstance(row, Mapping) and row.get("lane") == lane:
            return row
    return {}


def _first_comparison_row(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    rows = _comparison_rows(summary)
    return rows[0] if rows else {}


def _comparison_rows(summary: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    external = summary.get("external_reference_comparison")
    if isinstance(external, Mapping):
        comparison = external.get("comparison")
        if isinstance(comparison, Mapping):
            rows = comparison.get("rows", [])
            if isinstance(rows, list):
                return [row for row in rows if isinstance(row, Mapping)]
    return []


def _external_comparison_validation(summary: Mapping[str, Any]) -> bool:
    external = summary.get("external_reference_comparison")
    if isinstance(external, Mapping):
        return bool(external.get("validation_pass", False))
    return False


def _format_metric(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.4g}"
    return "missing"


def _write_final_disposition_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    columns = [
        "lane",
        "final_decision",
        "support_gate",
        "blocking_observable",
        "measured_offender",
        "evidence_artifact",
        "required_next_physics",
        "why_not_closed",
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


def _write_final_disposition_plot(path: Path, summary: Mapping[str, Any]) -> None:
    import matplotlib.pyplot as plt

    rows = [row for row in summary.get("rows", []) if isinstance(row, Mapping)]
    if not rows:
        rows = [
            {
                "lane": "strict_research_lanes",
                "final_decision": "no_rows",
                "blocking_observable": "No strict-lane rows were available.",
                "measured_offender": "missing closure summary",
                "required_next_physics": "Regenerate the closure status artifacts.",
                "why_not_closed": "No lane evidence was found.",
            }
        ]
    fig_height = 2.0 + 1.25 * len(rows)
    fig, ax = plt.subplots(figsize=(13.6, fig_height))
    ax.axis("off")
    colors = {
        "closed": "#dcfce7",
        "defer_strict_turbulent_parity": "#ffedd5",
        "defer_inertia_capable_localized_field_solver": "#fee2e2",
        "defer_resolved_secondary_flow_solver": "#fee2e2",
    }
    table_rows = []
    for row in rows:
        decision = str(row.get("final_decision", "open"))
        table_rows.append(
            [
                _wrap_final_disposition_cell(_lane_title(str(row.get("lane", ""))), width=24),
                _wrap_final_disposition_cell(decision.replace("_", " "), width=22),
                _wrap_final_disposition_cell(_plot_safe_disposition_text(row.get("blocking_observable", "")), width=25),
                _wrap_final_disposition_cell(_plot_safe_disposition_text(row.get("measured_offender", "")), width=28),
                _wrap_final_disposition_cell(_plot_safe_disposition_text(row.get("required_next_physics", "")), width=40),
            ]
        )
    table = ax.table(
        cellText=table_rows,
        colLabels=["Lane", "Decision", "Blocking observable", "Measured offender", "Required next physics"],
        cellLoc="left",
        colLoc="left",
        colWidths=[0.19, 0.16, 0.18, 0.22, 0.25],
        bbox=[0.0, 0.02, 1.0, 0.80],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.1)
    for (row_index, column_index), cell in table.get_celld().items():
        cell.set_edgecolor("#cbd5e1")
        cell.set_linewidth(0.7)
        cell.set_height(0.16 if row_index == 0 else 0.24)
        if row_index == 0:
            cell.set_facecolor("#0f172a")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            decision = str(rows[row_index - 1].get("final_decision", "open"))
            cell.set_facecolor(colors.get(decision, "#f8fafc") if column_index == 1 else "#ffffff")
            if column_index == 0:
                cell.get_text().set_weight("bold")
    title = "Final strict research-lane disposition"
    subtitle = "Evidence-based last-push result: strict blockers remain deferred unless their physics/external gates pass."
    ax.text(0.0, 0.98, title, fontsize=15, fontweight="bold", transform=ax.transAxes)
    ax.text(0.0, 0.91, subtitle, fontsize=9.5, color="#475569", transform=ax.transAxes)
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def _wrap_final_disposition_cell(value: str, *, width: int) -> str:
    return "\n".join(textwrap.wrap(value, width=width, break_long_words=False, break_on_hyphens=True))


def _plot_safe_disposition_text(value: Any) -> str:
    text = str(value)
    replacements = {
        "q2d_turbulence_reference_observables.csv": "Q2D turbulent reference CSV",
        "minimum_centerline_velocity_ratio": "minimum centerline velocity ratio",
        "secondary_flow_rms_ratio": "secondary-flow RMS ratio",
        "secondary_flow_peak_ratio": "secondary-flow peak ratio",
        "energy/enstrophy/spectrum/turnover": "energy, enstrophy, spectrum, turnover",
        "pressure/drag": "pressure or drag",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.replace("_", " ")


def _lane_title(lane: str) -> str:
    return {
        "q2d_turbulence_external_parity": "Q2D turbulence external parity",
        "magnetic_obstacle_external_validation": "Magnetic-obstacle external validation",
        "dean_vortex_higher_inertia_validation": "Higher-inertia Dean-vortex validation",
    }.get(lane, lane.replace("_", " "))


def _audit_row(
    *,
    lane: str,
    source: str,
    path: Path,
    evidence: list[Path],
    matched_reference_csv: Path | None,
    closure_summary: Path | None,
    next_step: str,
) -> dict[str, Any]:
    evidence_rows = [{"path": str(item), "exists": item.exists()} for item in evidence]
    matched_exists = bool(matched_reference_csv and matched_reference_csv.exists())
    closure_summary_exists = bool(closure_summary and closure_summary.exists())
    return {
        "lane": lane,
        "source": source,
        "path": str(path),
        "path_exists": path.exists(),
        "evidence_files": evidence_rows,
        "evidence_file_count": len(evidence_rows),
        "available_evidence_file_count": sum(1 for item in evidence_rows if item["exists"]),
        "matched_reference_csv": str(matched_reference_csv) if matched_reference_csv else "",
        "matched_reference_csv_exists": matched_exists,
        "closure_summary": str(closure_summary) if closure_summary else "",
        "closure_summary_exists": closure_summary_exists,
        "source_or_processed_artifact_exists": bool(path.exists() or any(item["exists"] for item in evidence_rows)),
        "ready_for_strict_closure_claim": bool(matched_exists and closure_summary_exists),
        "next_step": next_step,
    }


def _validation_payload(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = summary.get("validation")
    if isinstance(payload, Mapping):
        return payload
    payload = summary.get("turbulence_observables")
    if isinstance(payload, Mapping):
        return payload
    return {}


def _selected_metrics(
    lane: str,
    validation: Mapping[str, Any],
    external_comparison: Mapping[str, Any],
) -> dict[str, Any]:
    keys_by_lane = {
        "q2d_turbulence_external_parity": (
            "frame_count",
            "turnover_count",
            "max_courant",
            "max_divergence_linf",
        ),
        "magnetic_obstacle_external_validation": (
            "peak_centerline_deficit_ratio",
            "peak_crosscut_distortion",
            "max_charge_balance_residual",
            "research_grade_validation_pass",
        ),
        "dean_vortex_higher_inertia_validation": (
            "research_grade_charge_balance_pass",
            "research_grade_dean_validation_pass",
            "secondary_flow_rms_ratio",
            "secondary_flow_peak_ratio",
        ),
    }
    selected: dict[str, Any] = {}
    for key in keys_by_lane.get(lane, ()):
        if key in validation:
            selected[key] = validation[key]
    for key in ("compared_observable_count", "passed_observable_count", "validation_pass"):
        if key in external_comparison:
            selected[key] = external_comparison[key]
    return selected


def _static_dir(static_dir: str | Path | None) -> Path:
    if static_dir is not None:
        return Path(static_dir)
    return Path(__file__).resolve().parents[1] / "docs" / "_static" / "generated"


def _external_codes_root(external_codes_root: str | Path | None) -> Path:
    if external_codes_root is not None:
        return Path(external_codes_root)
    local_root = Path("/Users/rogerio/local/tests/lmx_external_codes")
    return local_root if local_root.exists() else Path("lmx_external_codes")


def _freemhd_cases_root(freemhd_cases_root: str | Path | None) -> Path:
    if freemhd_cases_root is not None:
        return Path(freemhd_cases_root)
    local_root = Path("/Users/rogerio/local/tests/freemhd_test_cases")
    return local_root if local_root.exists() else Path("freemhd_test_cases")


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, Mapping) else {}


def _write_rows_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    columns = [
        "lane",
        "status",
        "closed",
        "physics_gate_pass",
        "external_gate_pass",
        "summary_exists",
        "external_summary_exists",
        "reference_csv_exists",
        "primary_artifact_exists",
        "external_reference_status",
        "next_step",
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

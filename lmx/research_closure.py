"""Research-grade closure status helpers for the strict validation lanes."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
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
    available_rows = [row for row in rows if row["path_exists"]]
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


def _q2d_external_audit(static_root: Path, external_root: Path) -> dict[str, Any]:
    root = external_root / "Q2DmhdFoam"
    evidence = [
        root / "run/lidDriven/IDM_output_U.txt",
        root / "run/lidDriven/lidDrivenFFT_U.png",
        root / "run/lidDriven/lidDrivenFieldProfile_U.png",
        static_root / "q2dmhdfoam_external_reference_summary.json",
        static_root / "q2dmhdfoam_lid_driven_turbulence_observables.csv",
    ]
    return _audit_row(
        lane="q2d_turbulence_external_parity",
        source="Q2DmhdFoam local checkout and adapter artifacts",
        path=root,
        evidence=evidence,
        matched_reference_csv=static_root / "q2d_turbulence_reference_observables.csv",
        closure_summary=static_root / "q2d_turbulence_decay_summary.json",
        next_step="Run a matched LMX-vs-Q2DmhdFoam turbulent case and export energy/enstrophy/spectrum observables.",
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
        fallback_root / "Examples/mhdEpotFoam/readme.txt",
    ]
    return _audit_row(
        lane="dean_vortex_higher_inertia_validation",
        source="Dean-vortex literature plus a future OpenFOAM curved-pipe reference case",
        path=root,
        evidence=evidence,
        matched_reference_csv=static_root / "dean_vortex_reference_observables.csv",
        closure_summary=static_root / "bent_pipe_inductionless_summary.json",
        next_step="Acquire or generate secondary-flow, velocity-skew, centroid-shift, and pressure-loss observables.",
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

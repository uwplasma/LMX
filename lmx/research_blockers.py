"""Strict research-blocker probe artifacts for release closeout.

These helpers summarize closure attempts that are too expensive or too
incomplete to run in routine CI. They deliberately do not promote candidate
targets or unmatched external-code outputs into strict validation claims.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

from .research_closure import research_grade_external_data_audit, research_grade_closure_status


@dataclass(frozen=True)
class StrictBlockerAttempt:
    """One strict blocker closure attempt and its release decision."""

    lane: str
    status: str
    attempted_action: str
    key_result: str
    release_decision: str
    required_next_step: str


MAGNETIC_OBSTACLE_ESCALATION = {
    "target_minimum_centerline_velocity_ratio": -0.13,
    "target_source": "Votyakov et al. Fig. 7(a) candidate plateau target",
    "low_resolution_candidate": {
        "base_bz": 105.0,
        "forcing": 1.0,
        "ny": 18,
        "nz": 18,
        "nx_stations": 13,
        "minimum_centerline_velocity_ratio": -0.14,
        "max_charge_balance_residual": 3.7e-11,
        "accepted": False,
        "reason": "Candidate reverse-flow signal was not retained after the current-resolution rerun.",
    },
    "current_resolution_rerun": {
        "base_bz": 105.0,
        "forcing": 1.0,
        "ny": 40,
        "nz": 40,
        "nx_stations": 25,
        "max_steps": 32,
        "potential_iterations": 80,
        "coupling_iterations": 12,
        "minimum_centerline_velocity_ratio": 0.996559043587607,
        "centerline_velocity_deficit_ratio": 0.003440956412392947,
        "pressure_drop_proxy": 2.6590709760413764,
        "current_proxy_peak": 21.302496732725963,
        "max_charge_balance_residual": 1.6372180033138312e-12,
        "accepted": False,
        "reason": (
            "The apparent low-resolution reverse-flow match disappears on the "
            "current benchmark grid; the lane needs a matched inertial/localized "
            "magnetic-obstacle case rather than a candidate CSV."
        ),
    },
}


def strict_blocker_closure_attempt_summary(
    *,
    static_dir: str | Path | None = None,
    external_codes_root: str | Path | None = None,
) -> dict[str, Any]:
    """Return the current strict-blocker closure attempt summary.

    The returned payload is suitable for docs, release notes, and tests. It is
    not a solver execution path; expensive reruns belong in manual validation
    campaigns and should feed strict reference CSVs only after matched parity is
    obtained.
    """

    static_root = _static_dir(static_dir)
    closure = research_grade_closure_status(static_root)
    audit = research_grade_external_data_audit(
        static_dir=static_root,
        external_codes_root=external_codes_root,
    )
    lanes = [
        _magnetic_obstacle_attempt(static_root),
        _q2d_turbulence_attempt(audit),
        _dean_vortex_attempt(static_root),
    ]
    return {
        "case": "research_grade_strict_blocker_attempt",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "release_decision": "do_not_tag_research_grade_release",
        "research_grade_ready": bool(closure.get("research_grade_ready")),
        "strict_open_lanes": list(closure.get("open_lanes", [])),
        "strict_closed_lane_count": int(closure.get("closed_lane_count", 0)),
        "strict_lane_count": int(closure.get("lane_count", len(lanes))),
        "lanes": [asdict(lane) for lane in lanes],
        "magnetic_obstacle_escalation": MAGNETIC_OBSTACLE_ESCALATION,
        "external_data_audit": {
            "available_source_count": int(audit.get("available_source_count", 0)),
            "matched_reference_csv_count": int(audit.get("matched_reference_csv_count", 0)),
            "rows": audit.get("rows", []),
        },
        "notes": (
            "Bounded release readiness can remain green while this strict report "
            "is open. A research-grade tag requires every lane to have matched "
            "external references, physics gates, convergence evidence, and "
            "strict reference CSVs consumed by release readiness."
        ),
    }


def write_strict_blocker_closure_attempt(
    out_dir: str | Path,
    *,
    static_dir: str | Path | None = None,
    external_codes_root: str | Path | None = None,
    filename_stem: str = "research_grade_strict_blocker_attempt",
) -> list[Path]:
    """Write JSON, CSV, and plot artifacts for the strict blocker attempt."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = strict_blocker_closure_attempt_summary(
        static_dir=static_dir,
        external_codes_root=external_codes_root,
    )
    json_path = out / f"{filename_stem}.json"
    csv_path = out / f"{filename_stem}.csv"
    png_path = out / f"{filename_stem}.png"
    pdf_path = out / f"{filename_stem}.pdf"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_attempt_csv(csv_path, summary["lanes"])
    _write_attempt_plot(png_path, summary)
    _write_attempt_plot(pdf_path, summary)
    return [json_path, csv_path, png_path, pdf_path]


def _magnetic_obstacle_attempt(static_root: Path) -> StrictBlockerAttempt:
    current_summary = _load_json(static_root / "magnetic_obstacle_benchmark_summary.json")
    validation = current_summary.get("validation", {})
    high_res = MAGNETIC_OBSTACLE_ESCALATION["current_resolution_rerun"]
    key_result = (
        "current 40x40x25 obstacle rerun gives minimum_centerline_velocity_ratio="
        f"{high_res['minimum_centerline_velocity_ratio']:.3f}, while the candidate "
        "Votyakov reverse-flow target is about -0.13"
    )
    if validation.get("research_grade_validation_pass"):
        status = "unexpectedly_closed_check_strict_csv"
        decision = "verify_before_tag"
    else:
        status = "open_current_solver_not_matching_external_obstacle_response"
        decision = "block_research_grade_tag"
    return StrictBlockerAttempt(
        lane="magnetic_obstacle_external_validation",
        status=status,
        attempted_action=(
            "Escalated localized-field strength around base_bz=105 and reran the "
            "current 40x40x25 benchmark grid after a low-resolution reverse-flow candidate appeared."
        ),
        key_result=key_result,
        release_decision=decision,
        required_next_step=(
            "Build a matched inertial magnetic-obstacle case from literature or an "
            "external solver, then compare centerline deficit, recovery, pressure, "
            "current, and Lorentz-force observables under mesh convergence."
        ),
    )


def _q2d_turbulence_attempt(audit: Mapping[str, Any]) -> StrictBlockerAttempt:
    row = _audit_row(audit, "q2d_turbulence_external_parity")
    available = int(row.get("available_evidence_file_count", 0) or 0)
    matched = bool(row.get("matched_reference_csv_exists"))
    status = "open_matched_q2dmhdfoam_case_missing"
    decision = "block_research_grade_tag"
    if matched:
        status = "reference_csv_present_verify_observable_gate"
        decision = "verify_before_tag"
    return StrictBlockerAttempt(
        lane="q2d_turbulence_external_parity",
        status=status,
        attempted_action=(
            "Audited local Q2DmhdFoam lid-driven, Vetcha, and cylinder/duct outputs "
            "against the current LMX periodic SM82-style turbulence example."
        ),
        key_result=(
            f"{available} Q2DmhdFoam evidence files are available, but the strict "
            "q2d_turbulence_reference_observables.csv matched to the LMX case is "
            f"{'present' if matched else 'absent'}."
        ),
        release_decision=decision,
        required_next_step=(
            "Either run a matched Q2DmhdFoam turbulence case with the same geometry, "
            "forcing, friction, and observables as LMX, or add the corresponding "
            "LMX lid-driven/cylinder case before claiming parity."
        ),
    )


def _dean_vortex_attempt(static_root: Path) -> StrictBlockerAttempt:
    summary = _load_json(static_root / "bent_pipe_inductionless_summary.json")
    literature_summary = _load_json(static_root / "dean_literature_validation_summary.json")
    validation = summary.get("validation", {})
    dean_number = _safe_float(summary.get("dean_number"), default=_safe_float(validation.get("dean_number")))
    peak = _safe_float(validation.get("secondary_flow_peak_ratio"))
    status = "open_secondary_flow_physics_missing"
    decision = "block_research_grade_tag"
    if validation.get("research_grade_dean_validation_pass"):
        status = "reference_gate_present_verify_dean_csv"
        decision = "verify_before_tag"
    return StrictBlockerAttempt(
        lane="dean_vortex_higher_inertia_validation",
        status=status,
        attempted_action=(
            "Audited the current bent-pipe example against higher-inertia Dean-flow "
            "requirements after local charge closure was fixed."
        ),
        key_result=(
            "current retained example remains a low-De straight-pipe-equivalence "
            f"gate (De={dean_number:.3e}, secondary_flow_peak_ratio={peak:.3e}); "
            "it does not resolve the Dean-vortex topology. "
            f"Bayat-Rezai literature gate present={bool(literature_summary.get('validation_pass'))}."
        ),
        release_decision=decision,
        required_next_step=(
            "Add a no-field Dean-flow case with resolved secondary velocity and an "
            "external curved-pipe reference before adding MHD damping and pressure-loss parity."
        ),
    )


def _write_attempt_csv(path: Path, lanes: list[Mapping[str, Any]]) -> None:
    fields = ("lane", "status", "release_decision", "attempted_action", "key_result", "required_next_step")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in lanes:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_attempt_plot(path: Path, summary: Mapping[str, Any]) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.5), constrained_layout=True)
    _plot_magnetic_attempt(axes[0], summary)
    _plot_q2d_attempt(axes[1], summary)
    _plot_dean_attempt(axes[2], summary)
    fig.suptitle("Strict research blockers: final closure attempt", fontsize=15, fontweight="bold")
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def _plot_magnetic_attempt(ax: Any, summary: Mapping[str, Any]) -> None:
    escalation = summary["magnetic_obstacle_escalation"]
    target = escalation["target_minimum_centerline_velocity_ratio"]
    low = escalation["low_resolution_candidate"]["minimum_centerline_velocity_ratio"]
    high = escalation["current_resolution_rerun"]["minimum_centerline_velocity_ratio"]
    labels = ["Votyakov\ncandidate", "low-res\nLMX scan", "40x40x25\nLMX rerun"]
    values = [target, low, high]
    colors = ["#555555", "#d99b2b", "#b03a2e"]
    ax.axhline(0.0, color="0.2", linewidth=0.8)
    ax.bar(labels, values, color=colors, alpha=0.88)
    ax.set_title("Magnetic obstacle")
    ax.set_ylabel("min centerline u / u_ref")
    ax.set_ylim(-0.22, 1.08)
    ax.text(2, high - 0.15, "not reversed", ha="center", va="top", fontsize=9, color="white")
    ax.text(1, low - 0.02, "not retained", ha="center", va="top", fontsize=9)
    ax.grid(axis="y", alpha=0.25)


def _plot_q2d_attempt(ax: Any, summary: Mapping[str, Any]) -> None:
    audit_rows = summary.get("external_data_audit", {}).get("rows", [])
    q2d = next((row for row in audit_rows if row.get("lane") == "q2d_turbulence_external_parity"), {})
    available = 1.0 if q2d.get("path_exists") else 0.0
    evidence_total = max(float(q2d.get("evidence_file_count", 0) or 0), 1.0)
    evidence = min(float(q2d.get("available_evidence_file_count", 0) or 0) / evidence_total, 1.0)
    matched = 1.0 if q2d.get("matched_reference_csv_exists") else 0.0
    ax.bar(
        ["checkout", "evidence", "matched\nCSV"],
        [available, evidence, matched],
        color=["#2f6f9f", "#69a7c9", "#b03a2e"],
        alpha=0.88,
    )
    ax.set_title("Q2D turbulence")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("readiness fraction")
    ax.text(2, 0.06, "missing", ha="center", va="bottom", color="0.15", fontsize=9)
    ax.grid(axis="y", alpha=0.25)


def _plot_dean_attempt(ax: Any, summary: Mapping[str, Any]) -> None:
    lane = next((row for row in summary.get("lanes", []) if row.get("lane") == "dean_vortex_higher_inertia_validation"), {})
    key = str(lane.get("key_result", ""))
    dean = _extract_float_after(key, "De=")
    secondary = _extract_float_after(key, "secondary_flow_peak_ratio=")
    plotted = [max(dean, 1.0e-12), max(secondary, 1.0e-12)]
    ax.bar(["Dean\nnumber", "secondary\nflow"], plotted, color=["#3b7f5f", "#b03a2e"], alpha=0.88)
    ax.set_yscale("log")
    ax.set_ylim(5.0e-13, 2.0e-6)
    ax.set_title("Dean-vortex bend")
    ax.set_ylabel("current retained metric")
    ax.text(0, plotted[0] * 1.5, "low-De", ha="center", fontsize=9)
    ax.text(1, plotted[1] * 1.5, "no vortex", ha="center", fontsize=9)
    ax.grid(axis="y", alpha=0.25)


def _audit_row(audit: Mapping[str, Any], lane: str) -> Mapping[str, Any]:
    for row in audit.get("rows", []):
        if isinstance(row, Mapping) and row.get("lane") == lane:
            return row
    return {}


def _static_dir(static_dir: str | Path | None) -> Path:
    if static_dir is not None:
        return Path(static_dir)
    return Path(__file__).resolve().parents[1] / "docs" / "_static" / "generated"


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, Mapping) else {}


def _safe_float(value: object, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _extract_float_after(text: str, prefix: str) -> float:
    if prefix not in text:
        return 0.0
    tail = text.split(prefix, 1)[1].split(";", 1)[0].split(",", 1)[0].split(")", 1)[0]
    return _safe_float(tail.strip())

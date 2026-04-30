"""Publication-facing figures for open research-grade validation lanes."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from .q2d import (
    build_q2d_turbulence_decay_case,
    solve_q2d_turbulence_decay,
)


VOTYAKOV_FIG7A_DIGITIZED: tuple[dict[str, float | str], ...] = (
    # Approximate digitization from Votyakov, Zienicke & Kolesnikov,
    # "Constrained flow around a magnetic obstacle", Fig. 7(a).
    {"series": "experiment_Ha140", "N": 3.5, "ux_min": 0.115},
    {"series": "experiment_Ha140", "N": 4.5, "ux_min": 0.010},
    {"series": "experiment_Ha140", "N": 5.0, "ux_min": -0.020},
    {"series": "experiment_Ha140", "N": 5.5, "ux_min": -0.060},
    {"series": "experiment_Ha140", "N": 6.2, "ux_min": -0.080},
    {"series": "experiment_Ha140", "N": 7.0, "ux_min": -0.095},
    {"series": "experiment_Ha140", "N": 8.5, "ux_min": -0.105},
    {"series": "experiment_Ha140", "N": 11.0, "ux_min": -0.112},
    {"series": "experiment_Ha140", "N": 15.0, "ux_min": -0.121},
    {"series": "experiment_Ha140", "N": 18.0, "ux_min": -0.128},
    {"series": "experiment_Ha140", "N": 21.0, "ux_min": -0.134},
    {"series": "experiment_Ha140", "N": 25.0, "ux_min": -0.140},
    {"series": "experiment_Ha140", "N": 30.0, "ux_min": -0.148},
    {"series": "experiment_Ha140", "N": 34.0, "ux_min": -0.151},
    {"series": "simulation_Re196", "N": 3.7, "ux_min": 0.087},
    {"series": "simulation_Re196", "N": 5.6, "ux_min": -0.024},
    {"series": "simulation_Re196", "N": 9.2, "ux_min": -0.071},
    {"series": "simulation_Re196", "N": 16.0, "ux_min": -0.124},
    {"series": "simulation_Re100", "N": 3.8, "ux_min": 0.100},
    {"series": "simulation_Re100", "N": 4.6, "ux_min": 0.060},
    {"series": "simulation_Re100", "N": 5.2, "ux_min": 0.000},
    {"series": "simulation_Re100", "N": 6.6, "ux_min": -0.035},
    {"series": "simulation_Re100", "N": 8.4, "ux_min": -0.060},
    {"series": "simulation_Re100", "N": 16.0, "ux_min": -0.126},
    {"series": "simulation_Re100", "N": 25.0, "ux_min": -0.137},
    {"series": "simulation_Re100", "N": 36.0, "ux_min": -0.148},
)

_CANDIDATE_FIELDNAMES = (
    "observable",
    "value",
    "tolerance",
    "relative_tolerance",
    "units",
    "source",
    "note",
)


def write_research_grade_external_target_panel(
    output_dir: str | Path,
    *,
    magnetic_summary_path: str | Path,
    q2dmhdfoam_summary_path: str | Path,
    dean_summary_path: str | Path,
    output_stem: str = "research_grade_external_targets",
) -> list[Path]:
    """Write a publication-facing panel for the remaining strict blockers.

    The panel intentionally separates external target evidence from validation
    closure. It is suitable for docs/manuscript planning, but it does not fill
    the strict reference CSVs or mark a lane as closed.
    """

    import matplotlib.pyplot as plt

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    magnetic_summary = _load_json(magnetic_summary_path)
    q2dmhdfoam_summary = _load_json(q2dmhdfoam_summary_path)
    dean_summary = _load_json(dean_summary_path)

    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.8), constrained_layout=True)
    _plot_votyakov_target(axes[0], magnetic_summary)
    _plot_q2d_spectrum_target(axes[1], q2dmhdfoam_summary)
    _plot_dean_gap(axes[2], dean_summary)
    fig.suptitle(
        "External validation targets for remaining strict LMX lanes",
        fontsize=15,
        fontweight="bold",
    )

    png_path = out_dir / f"{output_stem}.png"
    pdf_path = out_dir / f"{output_stem}.pdf"
    for path in (png_path, pdf_path):
        fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return [png_path, pdf_path]


def write_research_grade_external_target_tables(
    output_dir: str | Path,
    *,
    q2dmhdfoam_summary_path: str | Path | None = None,
    dean_summary_path: str | Path | None = None,
    output_stem: str = "research_grade_external_targets",
) -> list[Path]:
    """Write candidate CSV tables used by the external target panel.

    These tables deliberately use ``*_candidate.csv`` names when the source is
    not a fully matched LMX/external comparison. Release readiness only consumes
    the strict reference CSVs, so these files document targets without creating
    a false validation pass.
    """

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    magnetic_path = out_dir / "magnetic_obstacle_votyakov_fig7a_digitized.csv"
    with magnetic_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("series", "N", "ux_min"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(VOTYAKOV_FIG7A_DIGITIZED)

    candidate_path = out_dir / "magnetic_obstacle_reference_observables_candidate.csv"
    with candidate_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CANDIDATE_FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerow(
            {
                "observable": "minimum_centerline_velocity_ratio",
                "value": "-0.13",
                "tolerance": "0.03",
                "relative_tolerance": "0.20",
                "units": "dimensionless",
                "source": (
                    "Approximate digitization of Votyakov et al. Fig. 7(a), "
                    "N > 15 plateau"
                ),
                "note": (
                    "Candidate target only; requires matched magnetic field, Re, "
                    "N, kappa, and inertia-capable LMX run before strict comparison."
                ),
            }
        )
        writer.writerow(
            {
                "observable": "normalized_recovery_distance",
                "value": "",
                "tolerance": "",
                "relative_tolerance": "0.15",
                "units": "dimensionless",
                "source": "Votyakov/Cuevas centerline or matched external solver",
                "note": (
                    "Not filled from Fig. 7(a); requires centerline profile "
                    "digitization or external run output."
                ),
            }
        )
    q2d_path = out_dir / "q2d_turbulence_reference_observables_candidate.csv"
    q2d_summary = _load_json(q2dmhdfoam_summary_path) if q2dmhdfoam_summary_path else {}
    _write_candidate_rows(q2d_path, _q2d_candidate_rows(q2d_summary))

    dean_path = out_dir / "dean_vortex_reference_observables_candidate.csv"
    dean_summary = _load_json(dean_summary_path) if dean_summary_path else {}
    _write_candidate_rows(dean_path, _dean_candidate_rows(dean_summary))

    summary_path = out_dir / f"{output_stem}_summary.json"
    summary = {
        "case": output_stem,
        "status": "external_targets_documented_not_strict_closure",
        "tables": [magnetic_path.name, candidate_path.name, q2d_path.name, dean_path.name],
        "notes": (
            "Candidate reference rows document Votyakov magnetic-obstacle, "
            "Q2DmhdFoam spectral, and Dean-vortex observable targets. They are "
            "not used by release readiness until matched LMX/external cases fill "
            "the strict reference CSVs."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return [magnetic_path, candidate_path, q2d_path, dean_path, summary_path]


def write_research_grade_closure_dashboard(
    output_dir: str | Path,
    *,
    q2d_sidewall_summary_path: str | Path,
    magnetic_strict_summary_path: str | Path,
    dean_strict_summary_path: str | Path,
    closure_status_path: str | Path,
    output_stem: str = "research_grade_closure_dashboard",
) -> list[Path]:
    """Write a publication-facing dashboard for strict closure status.

    The figure is deliberately conservative: it can show a closed support gate
    and failed strict comparisons in the same panel without promoting the open
    lanes to research-grade closure.
    """

    import matplotlib.pyplot as plt

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    q2d_summary = _load_json(q2d_sidewall_summary_path)
    magnetic_summary = _load_json(magnetic_strict_summary_path)
    dean_summary = _load_json(dean_strict_summary_path)
    closure_summary = _load_json(closure_status_path)

    fig, axes = plt.subplots(2, 2, figsize=(15.8, 10.2), constrained_layout=True)
    _plot_q2d_sidewall_closure(axes[0, 0], q2d_summary)
    _plot_magnetic_strict_gap(axes[0, 1], magnetic_summary)
    _plot_dean_strict_gap(axes[1, 0], dean_summary)
    _plot_closure_ledger(axes[1, 1], closure_summary)
    fig.suptitle(
        "Strict research-grade validation closure dashboard",
        fontsize=15.5,
        fontweight="bold",
    )

    png_path = out_dir / f"{output_stem}.png"
    pdf_path = out_dir / f"{output_stem}.pdf"
    for path in (png_path, pdf_path):
        fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    summary_path = out_dir / f"{output_stem}_summary.json"
    summary_path.write_text(
        json.dumps(
            _closure_dashboard_summary(
                q2d_summary,
                magnetic_summary,
                dean_summary,
                closure_summary,
                plots=[png_path.name, pdf_path.name],
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return [png_path, pdf_path, summary_path]


def _closure_dashboard_summary(
    q2d_summary: Mapping[str, object],
    magnetic_summary: Mapping[str, object],
    dean_summary: Mapping[str, object],
    closure_summary: Mapping[str, object],
    *,
    plots: list[str],
) -> dict[str, object]:
    magnetic_comparison = _external_comparison_payload(magnetic_summary)
    dean_comparison = _external_comparison_payload(dean_summary)
    return {
        "case": "research_grade_closure_dashboard",
        "status": "strict_research_lanes_documented",
        "plots": plots,
        "q2d_sidewall_gate_closed": bool(q2d_summary.get("strict_blocker_closed", False)),
        "magnetic_obstacle_external_validation_pass": bool(magnetic_comparison.get("validation_pass", False)),
        "dean_vortex_external_validation_pass": bool(dean_comparison.get("validation_pass", False)),
        "closed_lane_count": int(closure_summary.get("closed_lane_count", 0) or 0),
        "lane_count": int(closure_summary.get("lane_count", 0) or 0),
        "research_grade_ready": bool(closure_summary.get("research_grade_ready", False)),
        "open_lanes": list(closure_summary.get("open_lanes", [])),
        "notes": (
            "This dashboard is a documentation artifact. It shows closed support "
            "evidence and strict reference mismatches without marking open "
            "research-grade lanes as closed."
        ),
    }


def _write_candidate_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CANDIDATE_FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _plot_q2d_sidewall_closure(ax, summary: Mapping[str, object]) -> None:
    rows = _direct_comparison_rows(summary)
    if not rows:
        _missing_panel(ax, "Q2D side-wall gate", "comparison rows missing")
        return
    labels = [_compact_observable_label(str(row.get("observable", ""))) for row in rows]
    ratios = [
        _safe_float(row.get("relative_error")) / max(_safe_float(row.get("relative_tolerance")), 1.0e-30)
        for row in rows
    ]
    passes = [bool(row.get("validation_pass", False)) for row in rows]
    colors = ["#0f766e" if flag else "#b45309" for flag in passes]
    ax.bar(labels, ratios, color=colors)
    ax.axhline(1.0, color="#111827", linestyle="--", linewidth=1.0, label="tolerance")
    ax.set_ylim(0.0, max(1.2, 1.12 * max(ratios)))
    ax.set_ylabel("relative error / tolerance")
    ax.set_title("Closed matched side-wall Q2D gate")
    ax.grid(True, axis="y", alpha=0.24)
    ax.tick_params(axis="x", rotation=20)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    status = "closed" if bool(summary.get("strict_blocker_closed", False)) else "open"
    ax.text(
        0.03,
        0.16,
        (
            f"support gate: {status}\n"
            "cell-centered Q2DmhdFoam fields\n"
            "strict nonlinear turbulence parity remains separate"
        ),
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.92},
    )


def _plot_magnetic_strict_gap(ax, summary: Mapping[str, object]) -> None:
    rows = _external_comparison_rows(summary)
    if not rows:
        _missing_panel(ax, "Magnetic-obstacle strict gate", "external comparison rows missing")
        return
    row = rows[0]
    lmx = _safe_float(row.get("lmx_value"))
    reference = _safe_float(row.get("reference_value"))
    tolerance = _safe_float(row.get("effective_tolerance"))
    ax.bar(["LMX", "Votyakov target"], [lmx, reference], color=["#2563eb", "#dc2626"], width=0.58)
    ax.axhline(0.0, color="#111827", linewidth=1.0)
    ax.fill_between(
        [-0.4, 1.4],
        [reference - tolerance] * 2,
        [reference + tolerance] * 2,
        color="#fecaca",
        alpha=0.55,
        label="target tolerance",
    )
    ax.set_xlim(-0.45, 1.45)
    ax.set_ylim(min(-0.25, reference - 3.0 * tolerance), max(1.1, lmx * 1.08))
    ax.set_ylabel(r"minimum centerline $u_x/U_0$")
    ax.set_title("Magnetic obstacle: reverse-flow mismatch")
    ax.grid(True, axis="y", alpha=0.24)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    rel_error = _safe_float(row.get("relative_error"))
    ax.text(
        0.03,
        0.91,
        (
            f"relative error = {rel_error:.2g}\n"
            f"LMX min = {lmx:.3f}; target = {reference:.2f}\n"
            "requires inertia-capable localized-field parity"
        ),
        transform=ax.transAxes,
        fontsize=8,
        va="top",
        bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.92},
    )


def _plot_dean_strict_gap(ax, summary: Mapping[str, object]) -> None:
    rows = _external_comparison_rows(summary)
    if not rows:
        _missing_panel(ax, "Dean-vortex strict gate", "external comparison rows missing")
        return
    labels = [_compact_observable_label(str(row.get("observable", ""))) for row in rows]
    x = np.arange(len(rows), dtype=float)
    lmx = np.asarray([_safe_float(row.get("lmx_value")) for row in rows], dtype=float)
    reference = np.asarray([_safe_float(row.get("reference_value")) for row in rows], dtype=float)
    width = 0.34
    ax.bar(x - width / 2.0, lmx, width=width, color="#2563eb", label="LMX current")
    ax.bar(x + width / 2.0, reference, width=width, color="#dc2626", label="Bayat-Rezai target")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("secondary-flow ratio")
    ax.set_title("Dean vortex: higher-inertia mismatch")
    ax.grid(True, axis="y", alpha=0.24)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_ylim(0.0, max(0.1, 1.18 * float(np.max(reference))))
    ax.text(
        0.03,
        0.92,
        (
            f"current De = {_safe_float(summary.get('current_lmx_dean_number')):.2e}\n"
            f"target De = {_safe_float(summary.get('reference_dean_number')):.1f}\n"
            "remaining work is resolved secondary-flow physics"
        ),
        transform=ax.transAxes,
        fontsize=8,
        va="top",
        bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.92},
    )


def _plot_closure_ledger(ax, closure_summary: Mapping[str, object]) -> None:
    rows = list(closure_summary.get("rows", []))
    if not rows:
        _missing_panel(ax, "Strict closure ledger", "closure status missing")
        return
    from matplotlib.patches import FancyBboxPatch

    ax.axis("off")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Strict closure ledger")
    headers = [("lane", 0.03), ("physics", 0.34), ("external", 0.50), ("strict status", 0.66)]
    for label, x in headers:
        ax.text(x, 0.86, label, transform=ax.transAxes, fontsize=9, weight="bold", color="#0f172a")
    y_positions = np.linspace(0.70, 0.34, num=max(len(rows), 1))
    for row, y in zip(rows, y_positions, strict=False):
        if not isinstance(row, Mapping):
            continue
        ax.text(
            0.03,
            y,
            _short_lane_name(str(row.get("lane", ""))),
            transform=ax.transAxes,
            fontsize=8.5,
            va="center",
        )
        _draw_badge(ax, 0.34, y, "pass" if bool(row.get("physics_gate_pass", False)) else "open")
        _draw_badge(ax, 0.50, y, "pass" if bool(row.get("external_gate_pass", False)) else "open")
        status = "closed" if bool(row.get("closed", False)) else _friendly_status(row)
        face = "#dcfce7" if status == "closed" else "#ffedd5"
        patch = FancyBboxPatch(
            (0.66, y - 0.045),
            0.31,
            0.09,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            linewidth=1.0,
            edgecolor="#cbd5e1",
            facecolor=face,
            transform=ax.transAxes,
        )
        ax.add_patch(patch)
        ax.text(0.675, y, status, transform=ax.transAxes, fontsize=8, va="center")
    closed = int(closure_summary.get("closed_lane_count", 0) or 0)
    total = int(closure_summary.get("lane_count", len(rows)) or len(rows))
    ax.text(
        0.0,
        0.05,
        (
            f"strict research-grade lanes closed: {closed}/{total}\n"
            "bounded release can remain green while these manual research gates stay open"
        ),
        transform=ax.transAxes,
        fontsize=8.5,
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.92},
    )


def _draw_badge(ax, x: float, y: float, label: str) -> None:
    from matplotlib.patches import FancyBboxPatch

    face = "#dcfce7" if label == "pass" else "#fee2e2"
    patch = FancyBboxPatch(
        (x, y - 0.035),
        0.10,
        0.07,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.0,
        edgecolor="#cbd5e1",
        facecolor=face,
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    ax.text(x + 0.022, y, label, transform=ax.transAxes, fontsize=8, va="center")


def _friendly_status(row: Mapping[str, object]) -> str:
    lane = str(row.get("lane", ""))
    status = str(row.get("status", "open"))
    if lane == "q2d_turbulence_external_parity":
        return "turbulent parity open"
    if lane == "magnetic_obstacle_external_validation":
        return "reverse-flow mismatch"
    if lane == "dean_vortex_higher_inertia_validation":
        return "secondary flow open"
    return status.replace("_", " ")


def _missing_panel(ax, title: str, message: str) -> None:
    ax.axis("off")
    ax.set_title(title)
    ax.text(
        0.5,
        0.5,
        message,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.92},
    )


def _direct_comparison_rows(summary: Mapping[str, object]) -> list[Mapping[str, object]]:
    comparison = summary.get("comparison")
    if isinstance(comparison, Mapping):
        rows = comparison.get("rows", [])
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, Mapping)]
    return []


def _external_comparison_rows(summary: Mapping[str, object]) -> list[Mapping[str, object]]:
    rows = _external_comparison_payload(summary).get("rows", [])
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, Mapping)]
    return []


def _external_comparison_payload(summary: Mapping[str, object]) -> Mapping[str, object]:
    external = summary.get("external_reference_comparison")
    if isinstance(external, Mapping):
        comparison = external.get("comparison")
        if isinstance(comparison, Mapping):
            return comparison
    return {}


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return default
    return output if np.isfinite(output) else default


def _compact_observable_label(label: str) -> str:
    replacements = {
        "speed_mean": "mean speed",
        "speed_rms": "RMS speed",
        "vorticity_peak": "peak vorticity",
        "secondary_flow_rms_ratio": "secondary RMS",
        "secondary_flow_peak_ratio": "secondary peak",
        "minimum_centerline_velocity_ratio": "min centerline speed",
    }
    return replacements.get(label, label.replace("_", " "))


def _short_lane_name(lane: str) -> str:
    replacements = {
        "q2d_turbulence_external_parity": "Q2D turbulence",
        "magnetic_obstacle_external_validation": "magnetic obstacle",
        "dean_vortex_higher_inertia_validation": "Dean vortex",
    }
    return replacements.get(lane, lane.replace("_", " "))


def _q2d_candidate_rows(summary: Mapping[str, object]) -> list[dict[str, str]]:
    turbulence = dict(summary.get("turbulence_observables", {}))
    source = str(turbulence.get("source_path", "Q2DmhdFoam lid-driven spectral summary"))
    weak_weighted = turbulence.get("weak_weighted_wavenumber", "")
    source_path = turbulence.get("source_path")
    high_k_fraction: str | float = ""
    if source_path:
        weak_k, weak_peak, strong_k, strong_peak = _read_q2dmhdfoam_idm_spectrum(source_path)
        k = np.asarray(weak_k + strong_k, dtype=float)
        amplitude = np.asarray(weak_peak + strong_peak, dtype=float)
        if k.size and float(np.sum(amplitude)) > 0.0:
            high_k_fraction = float(np.sum(amplitude[k >= 4.0]) / np.sum(amplitude))
    return [
        {
            "observable": "energy_decay_ratio",
            "value": "",
            "tolerance": "",
            "relative_tolerance": "0.10",
            "units": "dimensionless",
            "source": source,
            "note": (
                "Candidate row left open; Q2DmhdFoam spectral summary does not "
                "include matched LMX interval energy history."
            ),
        },
        {
            "observable": "enstrophy_decay_ratio",
            "value": "",
            "tolerance": "",
            "relative_tolerance": "0.15",
            "units": "dimensionless",
            "source": source,
            "note": "Candidate row left open; requires matched vorticity/enstrophy history.",
        },
        {
            "observable": "final_spectral_centroid",
            "value": _candidate_value(weak_weighted),
            "tolerance": "",
            "relative_tolerance": "0.15",
            "units": "mode index",
            "source": source,
            "note": "Candidate Q2DmhdFoam spectral target only; not yet a matched LMX turbulent parity run.",
        },
        {
            "observable": "final_high_k_energy_fraction",
            "value": _candidate_value(high_k_fraction),
            "tolerance": "",
            "relative_tolerance": "0.20",
            "units": "dimensionless",
            "source": source,
            "note": (
                "Computed from normalized spectral amplitudes at mode index >= 4; "
                "replace with matched shell-energy definition before strict closure."
            ),
        },
        {
            "observable": "turnover_count",
            "value": "",
            "tolerance": "",
            "relative_tolerance": "0.10",
            "units": "dimensionless",
            "source": source,
            "note": "Candidate row left open; requires matched time integration interval.",
        },
    ]


def _dean_candidate_rows(summary: Mapping[str, object]) -> list[dict[str, str]]:
    validation = dict(summary.get("validation", {}))
    source = "Dean curved-pipe/duct literature or matched OpenFOAM curved-pipe run"
    current_de = float(validation.get("dean_number", 0.0))
    return [
        {
            "observable": "secondary_flow_rms_ratio",
            "value": "",
            "tolerance": "",
            "relative_tolerance": "0.15",
            "units": "dimensionless",
            "source": source,
            "note": (
                f"Strict value open. Current LMX low-De baseline is De={current_de:.3e}, "
                "not a higher-inertia Dean-vortex reference."
            ),
        },
        {
            "observable": "secondary_flow_peak_ratio",
            "value": "",
            "tolerance": "",
            "relative_tolerance": "0.15",
            "units": "dimensionless",
            "source": source,
            "note": (
                "Strict value open until the resolved secondary-flow or external "
                "curved-pipe reference is available."
            ),
        },
        {
            "observable": "normalized_velocity_centroid_shift",
            "value": "",
            "tolerance": "",
            "relative_tolerance": "0.15",
            "units": "dimensionless",
            "source": source,
            "note": "Strict value open; this should measure axial-speed skew toward the outer bend.",
        },
        {
            "observable": "inner_outer_velocity_ratio",
            "value": "",
            "tolerance": "",
            "relative_tolerance": "0.10",
            "units": "dimensionless",
            "source": source,
            "note": "Strict value open; compare outer/inner axial-speed cut in the bend.",
        },
        {
            "observable": "pressure_loss_proxy",
            "value": "",
            "tolerance": "",
            "relative_tolerance": "0.15",
            "units": "case-specific",
            "source": source,
            "note": "Strict value open; use the same nondimensional pressure-loss convention as the reference.",
        },
    ]


def _candidate_value(value: object) -> str:
    if isinstance(value, (int, float, np.floating)) and np.isfinite(float(value)):
        return f"{float(value):.12g}"
    return ""


def _plot_votyakov_target(ax, magnetic_summary: Mapping[str, object]) -> None:
    data = list(VOTYAKOV_FIG7A_DIGITIZED)
    style = {
        "experiment_Ha140": ("o", "none", "#2563eb", "Votyakov exp., Ha=140"),
        "simulation_Re196": ("s", "#1d4ed8", "#1d4ed8", "Votyakov sim., Re=196"),
        "simulation_Re100": ("^", "#1d4ed8", "#1d4ed8", "Votyakov sim., Re=100"),
    }
    for series, (marker, face, edge, label) in style.items():
        rows = [row for row in data if row["series"] == series]
        n = [float(row["N"]) for row in rows]
        ux = [float(row["ux_min"]) for row in rows]
        ax.plot(n, ux, color=edge, linewidth=1.2, alpha=0.85)
        ax.scatter(n, ux, marker=marker, facecolors=face, edgecolors=edge, s=38, label=label, zorder=3)
    ax.axhline(0.0, color="#111827", linewidth=1.0)
    ax.axvline(5.2, color="#64748b", linestyle="--", linewidth=1.0)
    ax.text(5.35, 0.17, r"$N_{c,m}$", fontsize=9, color="#475569")
    ax.set_xlim(0.0, 40.0)
    ax.set_ylim(-0.22, 0.22)
    ax.set_xlabel(r"interaction parameter $N$")
    ax.set_ylabel(r"minimum centerline $u_x/U_0$")
    ax.set_title("Magnetic obstacle target")
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")

    observables = _magnetic_observables(magnetic_summary)
    current_min = observables.get("minimum_centerline_velocity_ratio")
    current_deficit = observables.get("centerline_velocity_deficit_ratio")
    note = "LMX current gate: internal response only"
    if current_min is not None and current_deficit is not None:
        note += f"\nmin u/U0={float(current_min):.3f}; deficit={float(current_deficit):.3g}"
    ax.text(
        0.03,
        0.05,
        note,
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.92},
    )


def _plot_q2d_spectrum_target(ax, q2dmhdfoam_summary: Mapping[str, object]) -> None:
    case = build_q2d_turbulence_decay_case(nx=48, ny=48, t_final=1.0, frame_count=24)
    solution = solve_q2d_turbulence_decay(case)
    spectrum = solution.final_spectrum
    k = np.asarray(spectrum["wavenumber"], dtype=float) / (2.0 * np.pi / case.lx)
    energy = np.asarray(spectrum["energy"], dtype=float)
    positive = (k > 0.0) & (energy > 0.0)
    if np.any(positive):
        ax.plot(
            k[positive],
            energy[positive] / max(float(np.max(energy[positive])), 1.0e-30),
            "o-",
            color="#0f766e",
            label="LMX final spectrum",
        )

    turbulence = dict(q2dmhdfoam_summary.get("turbulence_observables", {}))
    weak_k = []
    weak_peak = []
    source_path = turbulence.get("source_path")
    if source_path:
        weak_k, weak_peak, strong_k, strong_peak = _read_q2dmhdfoam_idm_spectrum(source_path)
        if weak_k:
            ax.plot(weak_k, weak_peak, "s--", color="#1d4ed8", label="Q2DmhdFoam weak modes")
        if strong_k:
            ax.plot(strong_k, strong_peak, "D", color="#dc2626", label="Q2DmhdFoam strong mode")
    ax.set_xlabel("mode index")
    ax.set_ylabel("normalized spectral amplitude")
    ax.set_title("Q2D turbulence target")
    ax.set_xlim(0.0, 8.5)
    ax.set_ylim(0.0, 1.08)
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False, fontsize=7.5)
    ax.text(
        0.03,
        0.05,
        "Shape diagnostic only\nmatched forcing/domain still open",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.92},
    )


def _plot_dean_gap(ax, dean_summary: Mapping[str, object]) -> None:
    validation = dict(dean_summary.get("validation", {}))
    labels = [
        "observable\ncontract",
        "low-De\ncharge gate",
        "external\nDean data",
        "resolved\nsecondary flow",
    ]
    charge_pass = bool(validation.get("research_grade_charge_balance_pass", False))
    values = [1.0, 1.0 if charge_pass else 0.0, 0.0, 0.0]
    colors = ["#0f766e", "#0f766e" if charge_pass else "#f59e0b", "#b45309", "#b45309"]
    ax.bar(labels, values, color=colors)
    ax.set_ylim(0.0, 1.15)
    ax.set_ylabel("readiness fraction")
    ax.set_title("Dean-vortex validation target")
    ax.grid(True, axis="y", alpha=0.22)
    dean_number = float(validation.get("dean_number", 0.0))
    ax.text(
        0.03,
        0.92,
        (
            f"current De={dean_number:.2e}\n"
            "low-De current closure is documented\n"
            "higher-inertia reference and resolved secondary flow remain open"
        ),
        transform=ax.transAxes,
        fontsize=8,
        va="top",
        bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.92},
    )


def _read_q2dmhdfoam_idm_spectrum(
    path: str | Path,
) -> tuple[list[float], list[float], list[float], list[float]]:
    text = Path(path).read_text(encoding="utf-8")
    weak = _extract_payload(text, "Weak turbulence")
    strong = _extract_payload(text, "Strong turbulence")
    weak_k = [float(row[0]) for row in weak if len(row) >= 2]
    weak_peak = [float(row[1]) for row in weak if len(row) >= 2]
    strong_k = [float(row[0]) for row in strong if len(row) >= 2]
    strong_peak = [float(row[1]) for row in strong if len(row) >= 2]
    normalizer = max(weak_peak + strong_peak + [1.0e-30])
    weak_peak = [value / normalizer for value in weak_peak]
    strong_peak = [value / normalizer for value in strong_peak]
    return weak_k, weak_peak, strong_k, strong_peak


def _extract_payload(text: str, label: str) -> list[list[float]]:
    import ast
    import re

    match = re.search(rf"{re.escape(label)}\s*:\s*(\[[^\n\r]*\])", text)
    if not match:
        return []
    payload = ast.literal_eval(match.group(1))
    return payload if isinstance(payload, list) else []


def _magnetic_observables(summary: Mapping[str, object]) -> dict[str, float]:
    external = summary.get("external_readiness", {})
    if isinstance(external, Mapping):
        observables = external.get("observables", {})
        if isinstance(observables, Mapping):
            return {
                str(key): float(value)
                for key, value in observables.items()
                if isinstance(value, (int, float))
            }
    return {}


def _load_json(path: str | Path) -> Mapping[str, object]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, Mapping) else {}

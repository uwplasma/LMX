"""External literature-reference validation helpers.

These utilities intentionally stay independent from the solvers. They define
the data contract used to turn digitized literature/experimental observables
into repeatable validation gates and publication-ready comparison tables.
"""

from __future__ import annotations

import ast
import csv
import json
import re
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


SCALAR_REFERENCE_COLUMNS = ("observable", "value", "tolerance")
MAGNETIC_OBSTACLE_REFERENCE_COLUMNS = SCALAR_REFERENCE_COLUMNS
EXTERNAL_VALIDATION_READY_SCORE = 3.0
FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def load_scalar_reference_observables(
    path: str | Path,
    *,
    context: str = "Scalar reference CSV",
) -> dict[str, dict[str, float | str]]:
    """Load scalar literature/experimental observables from CSV.

    Required columns are ``observable``, ``value``, and ``tolerance``. Optional
    metadata columns such as ``relative_tolerance``, ``units``, ``source``, and
    ``note`` are preserved. This generic loader is used by the magnetic-
    obstacle, Q2D turbulence, and Dean-vortex external-reference contracts.
    """

    source = Path(path)
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        missing = [column for column in SCALAR_REFERENCE_COLUMNS if column not in fieldnames]
        if missing:
            missing_text = ", ".join(missing)
            raise ValueError(f"{context} is missing required columns: {missing_text}")
        records: dict[str, dict[str, float | str]] = {}
        for row_number, row in enumerate(reader, start=2):
            observable = (row.get("observable") or "").strip()
            if not observable:
                raise ValueError(f"{context} row {row_number} has an empty observable")
            value = _parse_float(row.get("value"), row_number=row_number, column="value", context=context)
            tolerance = _parse_float(row.get("tolerance"), row_number=row_number, column="tolerance", context=context)
            if tolerance < 0.0:
                raise ValueError(f"{context} row {row_number} has a negative tolerance")
            payload: dict[str, float | str] = {"value": value, "tolerance": tolerance}
            relative_text = (row.get("relative_tolerance") or "").strip()
            if relative_text:
                relative_tolerance = _parse_float(
                    relative_text,
                    row_number=row_number,
                    column="relative_tolerance",
                    context=context,
                )
                if relative_tolerance < 0.0:
                    raise ValueError(f"{context} row {row_number} has a negative relative_tolerance")
                payload["relative_tolerance"] = relative_tolerance
            for key, item in row.items():
                if key not in {"observable", "value", "tolerance", "relative_tolerance"} and item:
                    payload[key] = item.strip()
            records[observable] = payload
    return records


def compare_scalar_reference_observables(
    lmx_observables: Mapping[str, float],
    reference_observables: Mapping[str, Mapping[str, float | str]],
    *,
    missing_status: str = "missing_lmx_observable",
) -> dict[str, object]:
    """Compare scalar LMX observables with loaded reference rows."""

    rows: list[dict[str, float | str | bool]] = []
    compared = 0
    passed = 0
    for observable, reference in reference_observables.items():
        if observable not in lmx_observables:
            rows.append(
                {
                    "observable": observable,
                    "status": missing_status,
                    "validation_pass": False,
                }
            )
            continue
        reference_value = float(reference["value"])
        lmx_value = float(lmx_observables[observable])
        absolute_error = abs(lmx_value - reference_value)
        absolute_tolerance = float(reference.get("tolerance", 0.0))
        relative_tolerance = float(reference.get("relative_tolerance", 0.0))
        effective_tolerance = max(absolute_tolerance, relative_tolerance * max(abs(reference_value), 1.0e-20))
        validation_pass = bool(absolute_error <= effective_tolerance)
        compared += 1
        passed += int(validation_pass)
        rows.append(
            {
                "observable": observable,
                "lmx_value": lmx_value,
                "reference_value": reference_value,
                "absolute_error": absolute_error,
                "effective_tolerance": effective_tolerance,
                "relative_error": absolute_error / max(abs(reference_value), 1.0e-20),
                "validation_pass": validation_pass,
                "status": "compared",
                "units": str(reference.get("units", "")),
                "source": str(reference.get("source", "")),
            }
        )
    extra_observables = sorted(set(lmx_observables) - set(reference_observables))
    missing_count = sum(1 for row in rows if row["status"] == missing_status)
    return {
        "compared_observable_count": compared,
        "passed_observable_count": passed,
        "missing_lmx_observable_count": missing_count,
        "extra_lmx_observables": extra_observables,
        "rows": rows,
        "validation_pass": bool(compared > 0 and passed == compared and all(row["validation_pass"] for row in rows)),
    }


def write_scalar_reference_comparison_table(
    comparison: Mapping[str, object],
    path: str | Path,
) -> Path:
    """Write a publication-ready scalar-observable comparison CSV."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = list(comparison.get("rows", []))
    columns = [
        "observable",
        "lmx_value",
        "reference_value",
        "absolute_error",
        "relative_error",
        "effective_tolerance",
        "validation_pass",
        "status",
        "units",
        "source",
    ]
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
    return out


def write_scalar_reference_comparison_plots(
    comparison: Mapping[str, object],
    output_dir: str | Path,
    *,
    output_stem: str,
    title: str,
    no_data_label: str,
) -> list[Path]:
    """Write PNG/PDF scalar-observable comparison plots."""

    import matplotlib.pyplot as plt

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [dict(row) for row in comparison.get("rows", []) if dict(row).get("status") == "compared"]
    missing_count = int(comparison.get("missing_lmx_observable_count", 0))

    fig, axes = plt.subplots(2, 1, figsize=(8.4, 6.2), constrained_layout=True)
    if not rows:
        for ax in axes:
            ax.axis("off")
        axes[0].text(
            0.5,
            0.5,
            no_data_label,
            ha="center",
            va="center",
            fontsize=13,
            transform=axes[0].transAxes,
        )
        axes[1].text(
            0.5,
            0.5,
            f"Missing LMX observables: {missing_count}",
            ha="center",
            va="center",
            fontsize=11,
            transform=axes[1].transAxes,
        )
    else:
        labels = [_compact_observable_label(str(row["observable"])) for row in rows]
        x = np.arange(len(rows), dtype=float)
        lmx_values = np.asarray([float(row["lmx_value"]) for row in rows], dtype=float)
        reference_values = np.asarray([float(row["reference_value"]) for row in rows], dtype=float)
        absolute_errors = np.asarray([float(row["absolute_error"]) for row in rows], dtype=float)
        tolerances = np.asarray([float(row["effective_tolerance"]) for row in rows], dtype=float)
        pass_flags = [bool(row["validation_pass"]) for row in rows]

        width = 0.36
        axes[0].bar(x - width / 2.0, reference_values, width=width, label="reference", color="#2f5f8f")
        axes[0].bar(x + width / 2.0, lmx_values, width=width, label="LMX", color="#d46f2c")
        axes[0].set_ylabel("observable value")
        axes[0].set_title(title)
        axes[0].set_xticks(x, labels)
        axes[0].legend(frameon=False, ncols=2)
        axes[0].grid(True, axis="y", alpha=0.25)

        ratios = np.divide(
            absolute_errors,
            tolerances,
            out=np.full_like(absolute_errors, np.inf),
            where=tolerances > 0.0,
        )
        finite = ratios[np.isfinite(ratios)]
        fallback_height = float(max(2.0, np.max(finite) * 1.15)) if finite.size else 2.0
        plot_ratios = np.where(np.isfinite(ratios), ratios, fallback_height)
        colors = ["#2a9d8f" if flag else "#c2410c" for flag in pass_flags]
        axes[1].bar(x, plot_ratios, color=colors)
        axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1.0, label="tolerance")
        axes[1].set_ylabel("|LMX - ref| / tolerance")
        axes[1].set_xticks(x, labels)
        axes[1].set_ylim(0.0, max(1.25, float(np.max(plot_ratios)) * 1.18))
        axes[1].grid(True, axis="y", alpha=0.25)
        axes[1].legend(frameon=False)
        if missing_count:
            axes[1].text(
                0.99,
                0.96,
                f"missing LMX observables: {missing_count}",
                ha="right",
                va="top",
                fontsize=9,
                transform=axes[1].transAxes,
            )
        for xi, ratio in zip(x, ratios, strict=True):
            if not np.isfinite(ratio):
                axes[1].text(xi, fallback_height, "inf", ha="center", va="bottom", fontsize=8)

    png_path = out_dir / f"{output_stem}.png"
    pdf_path = out_dir / f"{output_stem}.pdf"
    for path in (png_path, pdf_path):
        fig.savefig(path, dpi=180)
    plt.close(fig)
    return [png_path, pdf_path]


def write_scalar_reference_template(path: str | Path, rows: list[dict[str, str]]) -> Path:
    """Write a scalar-observable external-reference CSV template."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    columns = ("observable", "value", "tolerance", "relative_tolerance", "units", "source", "note")
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return out


def external_validation_readiness_rows() -> list[dict[str, object]]:
    """Return the current external-code readiness matrix for open lanes.

    The score is intentionally coarse:

    - ``0``: no external path identified;
    - ``1``: literature or code path identified, but not executable here;
    - ``2``: executable code or processed data are available, but the LMX
      observable parity gate still needs filled reference rows or case
      construction;
    - ``3``: executable or processed external data are already wired into an
      LMX validation artifact.
    """

    return [
        {
            "lane": "Straight-duct Hunt/Shercliff parity",
            "external_code": "FreeMHD Docker + paper slices",
            "score": 3.0,
            "status": "wired artifact",
            "observables": "u, phi, J, JxB, Q, pressure-gradient proxy",
            "next_step": "keep heavier constant-Q ladder as manual artifact",
        },
        {
            "lane": "Q2D turbulence parity",
            "external_code": "Q2DmhdFoam",
            "score": 2.7,
            "status": "external adapter wired",
            "observables": "profiles, force coefficients, probes, spectra, turnover count",
            "next_step": "run matched LMX-vs-Q2DmhdFoam turbulent case",
        },
        {
            "lane": "Magnetic-obstacle validation",
            "external_code": "MHD_Solvers_OpenFOAM + literature",
            "score": 2.0,
            "status": "compiled and smoke-run",
            "observables": "deficit, wake recovery, pressure/drag, J/Lorentz",
            "next_step": "modernize obstacle case or digitize Votyakov/Cuevas",
        },
        {
            "lane": "Fringing mapped-pipe parity",
            "external_code": "FreeMHD paper pipe slices",
            "score": 2.0,
            "status": "processed data available",
            "observables": "tap pressure drop, profile distortion, potential",
            "next_step": "deferred mesh/operator parity campaign",
        },
        {
            "lane": "Dean-vortex bent-pipe parity",
            "external_code": "Bayat-Rezai Dean literature + OpenFOAM curved-pipe path",
            "score": 2.0,
            "status": "literature gate wired",
            "observables": "secondary-flow intensity, centroid shift, pressure loss",
            "next_step": "construct solved hydrodynamic curved-pipe reference",
        },
        {
            "lane": "Variable/tabulated 3D fields",
            "external_code": "WHAM coil-model script + manufactured fields",
            "score": 2.0,
            "status": "external field script wired",
            "observables": "interpolation error, div B, pressure response, autodiff",
            "next_step": "add matched field-response validation data",
        },
    ]


def summarize_external_validation_readiness(
    rows: list[Mapping[str, object]] | None = None,
    *,
    ready_score: float = EXTERNAL_VALIDATION_READY_SCORE,
) -> dict[str, object]:
    """Summarize which external-validation lanes are ready or still open."""

    records = list(external_validation_readiness_rows() if rows is None else rows)
    ready_lanes = [str(row["lane"]) for row in records if float(row["score"]) >= ready_score]
    runnable_lanes = [str(row["lane"]) for row in records if float(row["score"]) >= 2.0]
    open_lanes = [str(row["lane"]) for row in records if float(row["score"]) < ready_score]
    return {
        "lane_count": len(records),
        "ready_lane_count": len(ready_lanes),
        "runnable_or_data_lane_count": len(runnable_lanes),
        "open_lane_count": len(open_lanes),
        "ready_lanes": ready_lanes,
        "runnable_or_data_lanes": runnable_lanes,
        "open_lanes": open_lanes,
        "research_grade_validation_pass": bool(records and len(open_lanes) == 0),
    }


def write_external_validation_readiness_panel(
    rows: list[Mapping[str, object]] | None,
    output_dir: str | Path,
    *,
    output_stem: str = "external_validation_readiness",
    write_pdf: bool = False,
) -> list[Path]:
    """Write a publication-style readiness panel for external validation lanes."""

    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    records = list(external_validation_readiness_rows() if rows is None else rows)
    if not records:
        raise ValueError("At least one external validation readiness row is required")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = [str(row["lane"]) for row in records]
    scores = np.asarray([float(row["score"]) for row in records], dtype=float)
    y = np.arange(len(records), dtype=float)
    colors = [_readiness_color(score) for score in scores]

    fig = plt.figure(figsize=(13.8, 9.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.35])
    ax = fig.add_subplot(grid[0, 0])
    ax_table = fig.add_subplot(grid[1, 0])

    ax.barh(y, scores, color=colors, edgecolor="#1f2937", linewidth=0.6)
    ax.axvline(EXTERNAL_VALIDATION_READY_SCORE, color="#111827", linestyle="--", linewidth=1.0)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0.0, EXTERNAL_VALIDATION_READY_SCORE + 0.25)
    ax.set_xlabel("external validation readiness")
    ax.set_title("External validation readiness by lane")
    ax.set_xticks([0, 1, 2, 3], ["none", "identified", "runnable/data", "wired"])
    ax.grid(True, axis="x", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    legend_handles = [
        Patch(facecolor="#c2410c", edgecolor="#1f2937", label="open reference"),
        Patch(facecolor="#d97706", edgecolor="#1f2937", label="runnable/data path"),
        Patch(facecolor="#2a9d8f", edgecolor="#1f2937", label="wired artifact"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=False, fontsize=9, ncols=3)

    ax_table.axis("off")
    row_text = []
    for row in records:
        row_text.append(
            [
                _wrap_text(str(row["lane"]), 24),
                _wrap_text(str(row["external_code"]), 25),
                _wrap_text(str(row["observables"]), 34),
                _wrap_text(str(row["next_step"]), 34),
            ]
        )
    table = ax_table.table(
        cellText=row_text,
        colLabels=["lane", "external source", "observable gate", "next validation artifact"],
        loc="center",
        cellLoc="left",
        colLoc="left",
        colWidths=[0.22, 0.23, 0.27, 0.28],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.0)
    table.scale(1.0, 2.4)
    for (row_idx, _col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#d1d5db")
        if row_idx == 0:
            cell.set_facecolor("#111827")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#f8fafc" if row_idx % 2 else "white")

    summary = summarize_external_validation_readiness(records)
    fig.suptitle("LMX executable external-code validation map", fontsize=15, fontweight="bold")
    fig.text(
        0.01,
        0.01,
        (
            f"Ready lanes: {summary['ready_lane_count']}/{summary['lane_count']} | "
            f"runnable/data lanes: {summary['runnable_or_data_lane_count']}/{summary['lane_count']} | "
            "routine CI does not require external executables"
        ),
        fontsize=8.5,
        color="#374151",
    )

    png_path = out_dir / f"{output_stem}.png"
    pdf_path = out_dir / f"{output_stem}.pdf"
    paths = [png_path]
    if write_pdf:
        paths.append(pdf_path)
    for path in paths:
        fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return paths


def load_q2dmhdfoam_line_profile(
    path: str | Path,
    *,
    coordinate_column: int = 0,
    velocity_column: int = -1,
    coordinate_half_width: float | None = None,
) -> dict[str, object]:
    """Load a Q2DmhdFoam line-sampled velocity profile.

    Q2DmhdFoam validation files commonly store ``coordinate, theta, Ux`` or
    ``coordinate, Ux`` columns. The loader normalizes the coordinate to
    ``[-1, 1]`` while preserving the raw coordinate and velocity arrays.
    Filename tokens such as ``lineSampled_theta_Ux_250_500_1e6`` are parsed as
    Hartmann, Reynolds, and Grashof metadata when present.
    """

    source = Path(path)
    data = np.loadtxt(source)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 2:
        raise ValueError(f"Q2DmhdFoam line profile {source} needs at least two columns")

    coordinate = np.asarray(data[:, coordinate_column], dtype=float)
    velocity = np.asarray(data[:, velocity_column], dtype=float)
    finite = np.isfinite(coordinate) & np.isfinite(velocity)
    if finite.sum() < 3:
        raise ValueError(f"Q2DmhdFoam line profile {source} has fewer than three finite samples")

    coordinate = coordinate[finite]
    velocity = velocity[finite]
    order = np.argsort(coordinate)
    coordinate = coordinate[order]
    velocity = velocity[order]

    span = float(np.max(coordinate) - np.min(coordinate))
    if span <= 0.0:
        raise ValueError(f"Q2DmhdFoam line profile {source} has zero coordinate span")
    half_width = float(coordinate_half_width) if coordinate_half_width is not None else 0.5 * span
    if half_width <= 0.0:
        raise ValueError("coordinate_half_width must be positive")
    center = 0.5 * float(np.max(coordinate) + np.min(coordinate))
    position = (coordinate - center) / half_width
    metadata = _q2dmhdfoam_conditions_from_name(source.name)

    return {
        "source_path": str(source),
        "label": _q2dmhdfoam_profile_label(source.name, metadata),
        "position": position,
        "raw_coordinate": coordinate,
        "velocity": velocity,
        "sample_count": int(position.size),
        **metadata,
    }


def q2dmhdfoam_profile_observables(profile: Mapping[str, object]) -> dict[str, float | str | int]:
    """Compute scalar observables from one Q2DmhdFoam profile.

    The returned metrics are intentionally code-agnostic: they summarize
    normalized shape, symmetry, wall gradients, and edge damping so they can be
    compared with future LMX Q2D runs or digitized reference curves.
    """

    x = np.asarray(profile["position"], dtype=float)
    u = np.asarray(profile["velocity"], dtype=float)
    if x.ndim != 1 or u.ndim != 1 or x.size != u.size:
        raise ValueError("Q2DmhdFoam profile observables require matching 1D position and velocity arrays")
    if x.size < 3:
        raise ValueError("Q2DmhdFoam profile observables require at least three samples")

    order = np.argsort(x)
    x = x[order]
    u = u[order]
    span = float(np.max(x) - np.min(x))
    if span <= 0.0:
        raise ValueError("Q2DmhdFoam profile position span must be positive")
    mean_velocity = float(np.trapezoid(u, x) / span)
    normalization = mean_velocity if abs(mean_velocity) > 1.0e-30 else float(np.max(np.abs(u)))
    if abs(normalization) <= 1.0e-30:
        normalized = np.zeros_like(u)
    else:
        normalized = u / normalization
    peak = float(np.max(u))
    trough = float(np.min(u))
    peak_to_mean = float(peak / mean_velocity) if abs(mean_velocity) > 1.0e-30 else float("nan")
    mirrored = np.interp(-x, x, normalized)
    symmetry_l2 = float(np.sqrt(np.mean((normalized - mirrored) ** 2)))
    edge_count = max(1, min(8, x.size // 12))
    edge_velocity = float(0.5 * (np.mean(normalized[:edge_count]) + np.mean(normalized[-edge_count:])))
    wall_gradient_proxy = float(np.max(np.abs(np.gradient(normalized, x)))) if x.size >= 4 else 0.0
    center_velocity = float(np.interp(0.0, x, normalized))

    result: dict[str, float | str | int] = {
        "label": str(profile.get("label", "Q2DmhdFoam profile")),
        "sample_count": int(x.size),
        "mean_velocity": mean_velocity,
        "peak_velocity": peak,
        "trough_velocity": trough,
        "peak_to_mean_velocity": peak_to_mean,
        "center_normalized_velocity": center_velocity,
        "edge_normalized_velocity": edge_velocity,
        "symmetry_l2": symmetry_l2,
        "wall_gradient_proxy": wall_gradient_proxy,
    }
    for key in ("hartmann", "reynolds", "grashof", "source_path"):
        if key in profile:
            value = profile[key]
            result[key] = float(value) if isinstance(value, (int, float, np.floating)) else str(value)
    return result


def write_q2dmhdfoam_profile_observable_table(
    records: Sequence[Mapping[str, float | str | int]],
    path: str | Path,
) -> Path:
    """Write scalar Q2DmhdFoam profile observables to CSV."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "label",
        "hartmann",
        "reynolds",
        "grashof",
        "sample_count",
        "mean_velocity",
        "peak_velocity",
        "trough_velocity",
        "peak_to_mean_velocity",
        "center_normalized_velocity",
        "edge_normalized_velocity",
        "symmetry_l2",
        "wall_gradient_proxy",
        "source_path",
    ]
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow({column: record.get(column, "") for column in columns})
    return out


def load_q2dmhdfoam_lid_driven_observables(path: str | Path) -> dict[str, float | int | str]:
    """Load turbulence-summary observables emitted by Q2DmhdFoam lid-driven runs."""

    source = Path(path)
    text = source.read_text(encoding="utf-8")
    weak = _parse_q2dmhdfoam_list_payload(text, "Weak turbulence")
    strong = _parse_q2dmhdfoam_list_payload(text, "Strong turbulence")

    result: dict[str, float | int | str] = {"source_path": str(source)}
    if weak:
        weak_arr = np.asarray(weak, dtype=float)
        if weak_arr.ndim != 2 or weak_arr.shape[1] < 2:
            raise ValueError(f"{source} has malformed weak-turbulence rows")
        peak = weak_arr[:, 1]
        result.update(
            {
                "weak_mode_count": int(weak_arr.shape[0]),
                "weak_peak_over_max_max": float(np.max(peak)),
                "weak_peak_over_max_mean": float(np.mean(peak)),
                "weak_dominant_wavenumber": float(weak_arr[int(np.argmax(peak)), 0]),
                "weak_weighted_wavenumber": float(np.sum(weak_arr[:, 0] * peak) / max(np.sum(peak), 1.0e-30)),
            }
        )
    if strong:
        strong_arr = np.asarray(strong, dtype=float)
        if strong_arr.ndim != 2 or strong_arr.shape[1] < 3:
            raise ValueError(f"{source} has malformed strong-turbulence rows")
        strong_peak = strong_arr[:, 1]
        result.update(
            {
                "strong_mode_count": int(strong_arr.shape[0]),
                "strong_peak_over_max_max": float(np.max(strong_peak)),
                "strong_avg_over_max_max": float(np.max(strong_arr[:, 2])),
                "strong_dominant_wavenumber": float(strong_arr[int(np.argmax(strong_peak)), 0]),
            }
        )
    return result


def load_q2dmhdfoam_force_coefficients(path: str | Path, *, tail_fraction: float = 0.25) -> dict[str, float | str]:
    """Load Q2DmhdFoam force-coefficient history and return tail statistics."""

    source = Path(path)
    data = np.loadtxt(source, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 4:
        raise ValueError(f"Q2DmhdFoam force coefficient file {source} needs Time, Cd, Cl, Cm columns")
    finite = np.all(np.isfinite(data[:, :4]), axis=1)
    data = data[finite]
    if data.shape[0] < 3:
        raise ValueError(f"Q2DmhdFoam force coefficient file {source} has fewer than three finite rows")
    start = max(0, int(np.floor((1.0 - float(tail_fraction)) * data.shape[0])))
    tail = data[start:, :]
    time = data[:, 0]
    duration = float(time[-1] - time[0])
    return {
        "source_path": str(source),
        "sample_count": int(data.shape[0]),
        "tail_sample_count": int(tail.shape[0]),
        "time_start": float(time[0]),
        "time_end": float(time[-1]),
        "duration": duration,
        "cd_tail_mean": float(np.mean(tail[:, 1])),
        "cd_tail_rms": float(np.sqrt(np.mean((tail[:, 1] - np.mean(tail[:, 1])) ** 2))),
        "cl_tail_mean": float(np.mean(tail[:, 2])),
        "cl_tail_rms": float(np.sqrt(np.mean((tail[:, 2] - np.mean(tail[:, 2])) ** 2))),
        "cm_tail_mean": float(np.mean(tail[:, 3])),
        "cm_tail_rms": float(np.sqrt(np.mean((tail[:, 3] - np.mean(tail[:, 3])) ** 2))),
        "cd_drift": float(tail[-1, 1] - tail[0, 1]) if tail.shape[0] > 1 else 0.0,
        "cl_drift": float(tail[-1, 2] - tail[0, 2]) if tail.shape[0] > 1 else 0.0,
    }


def load_q2dmhdfoam_probe_velocity_history(path: str | Path) -> dict[str, float | int | str]:
    """Load an OpenFOAM probes U file and return velocity-history observables."""

    source = Path(path)
    rows: list[list[float]] = []
    for line in source.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        values = [float(item) for item in re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", stripped)]
        if len(values) >= 4:
            rows.append(values)
    if len(rows) < 3:
        raise ValueError(f"Q2DmhdFoam probe history {source} has fewer than three numeric rows")
    data = np.asarray(rows, dtype=float)
    time = data[:, 0]
    vectors = data[:, 1:]
    probe_count = vectors.shape[1] // 3
    if probe_count < 1:
        raise ValueError(f"Q2DmhdFoam probe history {source} has no vector probes")
    vectors = vectors[:, : 3 * probe_count].reshape(data.shape[0], probe_count, 3)
    speed = np.linalg.norm(vectors, axis=2)
    tail_start = max(0, int(0.75 * speed.shape[0]))
    tail_speed = speed[tail_start:, :]
    return {
        "source_path": str(source),
        "sample_count": int(speed.shape[0]),
        "probe_count": int(probe_count),
        "time_start": float(time[0]),
        "time_end": float(time[-1]),
        "duration": float(time[-1] - time[0]),
        "speed_tail_mean": float(np.mean(tail_speed)),
        "speed_tail_rms": float(np.sqrt(np.mean((tail_speed - np.mean(tail_speed)) ** 2))),
        "speed_peak": float(np.max(speed)),
        "streamwise_tail_mean": float(np.mean(vectors[tail_start:, :, 0])),
        "transverse_tail_rms": float(np.sqrt(np.mean(vectors[tail_start:, :, 1:] ** 2))),
    }


def write_q2dmhdfoam_timeseries_observable_table(
    records: Sequence[Mapping[str, float | str | int]],
    path: str | Path,
) -> Path:
    """Write Q2DmhdFoam force/probe time-history observables to CSV."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for record in records for key in record})
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow({column: record.get(column, "") for column in columns})
    return out


def load_q2dmhdfoam_docker_reference_profile(
    profile_csv: str | Path,
    summary_json: str | Path | None = None,
) -> dict[str, object]:
    """Load the Docker-generated Q2DmhdFoam fully developed reference profile.

    The Docker runner writes a compact CSV extracted from a foam-extend 4.1
    Q2DmhdFoam tutorial rerun. The loader keeps the raw dimensional coordinate
    and velocity, and exposes ``position`` as ``y / b`` so it can reuse the
    generic Q2DmhdFoam profile-observable utilities.
    """

    source = Path(profile_csv)
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if len(rows) < 3:
        raise ValueError(f"Q2DmhdFoam Docker profile {source} has fewer than three rows")

    y = np.asarray([float(row["y"]) for row in rows], dtype=float)
    position = np.asarray([float(row["y_over_b"]) for row in rows], dtype=float)
    velocity = np.asarray([float(row["ux"]) for row in rows], dtype=float)
    finite = np.isfinite(y) & np.isfinite(position) & np.isfinite(velocity)
    if finite.sum() < 3:
        raise ValueError(f"Q2DmhdFoam Docker profile {source} has fewer than three finite samples")
    order = np.argsort(position[finite])

    summary: dict[str, object] = {}
    if summary_json is not None:
        summary_path = Path(summary_json)
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
    hartmann = float(summary["hartmann"]) if "hartmann" in summary else float("nan")
    label = f"Docker Q2DmhdFoam Ha={hartmann:.3g}" if np.isfinite(hartmann) else "Docker Q2DmhdFoam"

    return {
        "source_path": str(source),
        "summary_path": str(summary_json) if summary_json is not None else "",
        "label": label,
        "position": position[finite][order],
        "raw_coordinate": y[finite][order],
        "velocity": velocity[finite][order],
        "sample_count": int(finite.sum()),
        "hartmann": hartmann,
        "summary": summary,
    }


def q2dmhdfoam_docker_reference_observables(profile: Mapping[str, object]) -> dict[str, float | str | int | bool]:
    """Return executable-run observables for the Docker Q2DmhdFoam gate."""

    observables = dict(q2dmhdfoam_profile_observables(profile))
    summary = dict(profile.get("summary", {}) if isinstance(profile.get("summary", {}), Mapping) else {})
    for key in (
        "final_time",
        "rank_count",
        "cell_count",
        "flow_rate_relative_error",
        "target_mean_velocity",
        "magnetic_field",
    ):
        if key in summary:
            value = summary[key]
            observables[key] = float(value) if isinstance(value, (int, float, np.floating)) else str(value)
    observables["steady_state_reached"] = summary.get("status") == "external_reference_case_complete"
    observables["reference_gate"] = "q2dmhdfoam_docker_fully_developed"
    return observables


def write_q2dmhdfoam_docker_reference_panel(
    profile: Mapping[str, object],
    observables: Mapping[str, float | str | int | bool],
    output_dir: str | Path,
    *,
    output_stem: str = "q2dmhdfoam_docker_reference",
) -> list[Path]:
    """Write a publication-facing panel for the Docker Q2DmhdFoam rerun."""

    import matplotlib.pyplot as plt

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    x = np.asarray(profile["position"], dtype=float)
    u = np.asarray(profile["velocity"], dtype=float)
    mean_velocity = float(observables["mean_velocity"])
    u_norm = u / mean_velocity if abs(mean_velocity) > 1.0e-30 else u
    summary = dict(profile.get("summary", {}) if isinstance(profile.get("summary", {}), Mapping) else {})

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.5), constrained_layout=True)
    axes[0].plot(x, u_norm, color="#0f766e", linewidth=2.2)
    axes[0].axvline(0.0, color="#64748b", linewidth=1.0, linestyle="--", alpha=0.65)
    axes[0].set_title("Q2DmhdFoam fully developed profile")
    axes[0].set_xlabel(r"$y / b$")
    axes[0].set_ylabel(r"$U_x / \overline{U}_x$")
    axes[0].grid(True, alpha=0.25)

    axes[1].axis("off")
    rows = [
        ("container", "foam-extend 4.1 + Q2DmhdFoam"),
        ("case", str(summary.get("case", "Q2DfullyDeveloped"))),
        ("status", str(summary.get("status", "unknown"))),
        ("Ha", f"{float(observables.get('hartmann', float('nan'))):.3g}"),
        ("MPI ranks", f"{int(float(observables.get('rank_count', 1))):d}"),
        ("cells", f"{int(float(observables.get('cell_count', 0))):d}"),
        ("flow-rate error", f"{float(observables.get('flow_rate_relative_error', float('nan'))):.2e}"),
        ("symmetry L2", f"{float(observables.get('symmetry_l2', float('nan'))):.3g}"),
    ]
    table = axes[1].table(
        cellText=rows,
        colLabels=("observable", "value"),
        cellLoc="left",
        colLoc="left",
        loc="center",
        bbox=(0.0, 0.08, 1.0, 0.84),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#cbd5e1")
        if row == 0:
            cell.set_facecolor("#0f172a")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f8fafc")
    axes[1].set_title("Executable external-code gate")

    fig.suptitle("Docker-rerun Q2DmhdFoam reference artifact", fontsize=14.5, fontweight="bold")
    paths = [out_dir / f"{output_stem}.png", out_dir / f"{output_stem}.pdf"]
    for path in paths:
        fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return paths


def load_q2dmhdfoam_vtk_vector_field(
    vtk_path: str | Path,
    *,
    field_name: str = "U",
    data_kind: str = "point",
) -> dict[str, object]:
    """Load an ASCII foamToVTK vector field from a Q2DmhdFoam rerun.

    The external Docker runner writes legacy ASCII VTK files. This parser only
    targets the FIELD-array layout emitted by `foamToVTK -ascii`; it is enough
    for validation dashboards without adding a hard dependency on VTK/PyVista.
    """

    source = Path(vtk_path)
    tokens = source.read_text(encoding="utf-8").split()
    points_index = _require_token(tokens, "POINTS", source)
    point_count = int(tokens[points_index + 1])
    point_start = points_index + 3
    point_stop = point_start + 3 * point_count
    if point_stop > len(tokens):
        raise ValueError(f"VTK file {source} ended before all POINTS were read")
    points = np.asarray(tokens[point_start:point_stop], dtype=float).reshape(point_count, 3)

    marker = "POINT_DATA" if data_kind == "point" else "CELL_DATA"
    data_index = _require_token(tokens, marker, source)
    tuple_count = int(tokens[data_index + 1])
    field_index = data_index + 2
    if tokens[field_index] != "FIELD":
        field_index = _require_token(tokens[data_index + 2 :], "FIELD", source) + data_index + 2
    array_count = int(tokens[field_index + 2])
    arrays: dict[str, np.ndarray] = {}
    cursor = field_index + 3
    for _ in range(array_count):
        if cursor + 4 > len(tokens):
            raise ValueError(f"VTK file {source} ended inside FIELD metadata")
        name = tokens[cursor]
        component_count = int(tokens[cursor + 1])
        local_tuple_count = int(tokens[cursor + 2])
        dtype_name = tokens[cursor + 3]
        value_count = component_count * local_tuple_count
        cursor += 4
        values = tokens[cursor : cursor + value_count]
        if len(values) != value_count:
            raise ValueError(f"VTK file {source} ended inside FIELD array {name}")
        if dtype_name.lower().startswith("int"):
            array = np.asarray(values, dtype=int)
        else:
            array = np.asarray(values, dtype=float)
        arrays[name] = array.reshape(local_tuple_count, component_count)
        cursor += value_count
    if field_name not in arrays:
        raise ValueError(f"VTK file {source} does not contain FIELD array {field_name!r}")
    field = np.asarray(arrays[field_name], dtype=float)
    if data_kind == "point" and field.shape[0] != point_count:
        raise ValueError(f"Point FIELD array {field_name!r} has {field.shape[0]} rows but POINTS has {point_count}")
    if data_kind != "point" and field.shape[0] != tuple_count:
        raise ValueError(f"Cell FIELD array {field_name!r} has {field.shape[0]} rows but {marker} has {tuple_count}")
    return {
        "source_path": str(source),
        "data_kind": data_kind,
        "field_name": field_name,
        "points": points,
        "vectors": field,
        "arrays": arrays,
        "tuple_count": int(tuple_count),
        "point_count": int(point_count),
    }


def q2dmhdfoam_vtk_velocity_observables(field: Mapping[str, object]) -> dict[str, float | int | str]:
    """Compute compact velocity observables from a Q2DmhdFoam VTK vector field."""

    vectors = np.asarray(field["vectors"], dtype=float)
    if vectors.ndim != 2 or vectors.shape[1] < 2:
        raise ValueError("Q2DmhdFoam VTK velocity observables require a vector array with at least two components")
    speed = np.linalg.norm(vectors[:, : min(3, vectors.shape[1])], axis=1)
    points = np.asarray(field["points"], dtype=float)
    arrays = field.get("arrays", {})
    vorticity_peak = float("nan")
    if isinstance(arrays, Mapping) and "vorticity" in arrays:
        vort = np.asarray(arrays["vorticity"], dtype=float)
        if vort.shape[0] == vectors.shape[0]:
            vorticity_peak = float(np.max(np.linalg.norm(vort[:, : min(3, vort.shape[1])], axis=1)))
    return {
        "source_path": str(field.get("source_path", "")),
        "sample_count": int(vectors.shape[0]),
        "x_min": float(np.min(points[:, 0])),
        "x_max": float(np.max(points[:, 0])),
        "y_min": float(np.min(points[:, 1])),
        "y_max": float(np.max(points[:, 1])),
        "z_min": float(np.min(points[:, 2])),
        "z_max": float(np.max(points[:, 2])),
        "speed_mean": float(np.mean(speed)),
        "speed_max": float(np.max(speed)),
        "speed_rms": float(np.sqrt(np.mean(speed**2))),
        "ux_mean": float(np.mean(vectors[:, 0])),
        "uy_mean": float(np.mean(vectors[:, 1])),
        "vorticity_peak": vorticity_peak,
        "reference_gate": "q2dmhdfoam_vtk_field_ingestion",
    }


def load_q2dmhdfoam_lid_driven_cell_field(case_dir: str | Path) -> dict[str, object]:
    """Load cell-centered velocity/vorticity from a Q2DmhdFoam lid-driven case.

    The VTK files emitted by ``foamToVTK`` are useful for visualization, but
    their point arrays are not area-weighted on the graded OpenFOAM mesh. This
    loader reads the reconstructed OpenFOAM time directory directly and returns
    the cell centers, cell widths, velocity, and optional vorticity used for
    scalar parity observables.
    """

    root = Path(case_dir)
    latest = _latest_openfoam_time_dir(root)
    variables = _openfoam_scalar_assignments(root / "constant" / "polyMesh" / "blockMeshDict")
    nx = int(round(float(variables["Nx"])))
    x_length = float(variables["x"])
    x = (np.arange(nx, dtype=float) + 0.5) * x_length / max(nx, 1)
    y, y_widths = _q2dmhdfoam_lid_driven_y_cells(variables)
    velocity = _read_openfoam_vector_internal_field(latest / "U").reshape(len(y), nx, 3)
    arrays: dict[str, np.ndarray] = {}
    vorticity_path = latest / "vorticity"
    if vorticity_path.exists():
        arrays["vorticity"] = _read_openfoam_vector_internal_field(vorticity_path).reshape(len(y), nx, 3)
    return {
        "source_path": str(latest),
        "x": x,
        "y": y,
        "x_width": float(x_length / max(nx, 1)),
        "y_widths": y_widths,
        "vectors": velocity,
        "arrays": arrays,
        "sample_count": int(velocity.shape[0] * velocity.shape[1]),
    }


def q2dmhdfoam_cell_velocity_observables(field: Mapping[str, object]) -> dict[str, float | int | str]:
    """Compute area-weighted velocity observables from Q2DmhdFoam cell data."""

    vectors = np.asarray(field["vectors"], dtype=float)
    if vectors.ndim != 3 or vectors.shape[2] < 2:
        raise ValueError("Q2DmhdFoam cell observables require a (ny, nx, component) vector array")
    y_widths = np.asarray(field["y_widths"], dtype=float)
    if y_widths.ndim != 1 or y_widths.size != vectors.shape[0]:
        raise ValueError("Q2DmhdFoam cell observables require one y-width per cell row")
    weights = y_widths[:, None]
    denominator = max(float(np.sum(weights) * vectors.shape[1]), 1.0e-300)
    speed = np.linalg.norm(vectors[:, :, : min(3, vectors.shape[2])], axis=2)
    arrays = field.get("arrays", {})
    vorticity_peak = float("nan")
    if isinstance(arrays, Mapping) and "vorticity" in arrays:
        vort = np.asarray(arrays["vorticity"], dtype=float)
        if vort.shape[:2] == vectors.shape[:2]:
            vorticity_peak = float(np.max(np.linalg.norm(vort[:, :, : min(3, vort.shape[2])], axis=2)))
    return {
        "source_path": str(field.get("source_path", "")),
        "sample_count": int(field.get("sample_count", vectors.shape[0] * vectors.shape[1])),
        "x_min": float(np.min(np.asarray(field["x"], dtype=float))),
        "x_max": float(np.max(np.asarray(field["x"], dtype=float))),
        "y_min": float(np.min(np.asarray(field["y"], dtype=float))),
        "y_max": float(np.max(np.asarray(field["y"], dtype=float))),
        "speed_mean": float(np.sum(speed * weights) / denominator),
        "speed_max": float(np.max(speed)),
        "speed_rms": float(np.sqrt(np.sum(speed**2 * weights) / denominator)),
        "ux_mean": float(np.sum(vectors[:, :, 0] * weights) / denominator),
        "uy_mean": float(np.sum(vectors[:, :, 1] * weights) / denominator),
        "vorticity_peak": vorticity_peak,
        "reference_gate": "q2dmhdfoam_cell_field_observables",
        "weighting": "cell_area",
    }


def write_q2dmhdfoam_vtk_velocity_panel(
    field: Mapping[str, object],
    observables: Mapping[str, float | int | str],
    output_dir: str | Path,
    *,
    output_stem: str = "q2dmhdfoam_lid_driven_vtk",
) -> list[Path]:
    """Write a publication-facing panel for a Q2DmhdFoam VTK velocity field."""

    import matplotlib.pyplot as plt

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    points = np.asarray(field["points"], dtype=float)
    vectors = np.asarray(field["vectors"], dtype=float)
    x_grid, y_grid, ux_grid, uy_grid, speed_grid = _point_vector_grid(points, vectors)
    x_mid = 0.5 * (float(x_grid[0]) + float(x_grid[-1]))
    y_mid = 0.5 * (float(y_grid[0]) + float(y_grid[-1]))
    x_index = int(np.argmin(np.abs(x_grid - x_mid)))
    y_index = int(np.argmin(np.abs(y_grid - y_mid)))
    peak = max(float(np.nanmax(speed_grid)), 1.0e-30)

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.0), constrained_layout=True)
    image = axes[0, 0].pcolormesh(x_grid, y_grid, speed_grid, shading="auto", cmap="magma")
    axes[0, 0].set_title("Q2DmhdFoam velocity magnitude")
    axes[0, 0].set_xlabel("x [m]")
    axes[0, 0].set_ylabel("y [m]")
    axes[0, 0].set_aspect("equal")
    fig.colorbar(image, ax=axes[0, 0], fraction=0.046, pad=0.04, label="|U| [m/s]")

    stride_x = max(1, len(x_grid) // 24)
    stride_y = max(1, len(y_grid) // 18)
    axes[0, 1].pcolormesh(x_grid, y_grid, speed_grid / peak, shading="auto", cmap="Blues", vmin=0.0, vmax=1.0)
    axes[0, 1].quiver(
        x_grid[::stride_x],
        y_grid[::stride_y],
        ux_grid[::stride_y, ::stride_x],
        uy_grid[::stride_y, ::stride_x],
        color="#111827",
        width=0.0028,
        scale=3.2 * peak,
    )
    axes[0, 1].set_title("Velocity direction over normalized speed")
    axes[0, 1].set_xlabel("x [m]")
    axes[0, 1].set_ylabel("y [m]")
    axes[0, 1].set_aspect("equal")

    axes[1, 0].plot(x_grid, uy_grid[y_index, :] / peak, color="#0f766e", linewidth=2.0, label=r"$U_y/U_{max}$ at mid-y")
    axes[1, 0].plot(y_grid, ux_grid[:, x_index] / peak, color="#b45309", linewidth=2.0, label=r"$U_x/U_{max}$ at mid-x")
    axes[1, 0].set_title("Centerline velocity components")
    axes[1, 0].set_xlabel("coordinate [m]")
    axes[1, 0].set_ylabel("normalized component")
    axes[1, 0].grid(True, alpha=0.25)
    axes[1, 0].legend(frameon=False)

    axes[1, 1].axis("off")
    rows = [
        ("case", "Q2DmhdFoam generic VTK"),
        ("samples", f"{int(observables['sample_count']):d}"),
        ("domain x", f"{float(observables['x_min']):.3g} .. {float(observables['x_max']):.3g} m"),
        ("domain y", f"{float(observables['y_min']):.3g} .. {float(observables['y_max']):.3g} m"),
        ("mean |U|", f"{float(observables['speed_mean']):.4g} m/s"),
        ("max |U|", f"{float(observables['speed_max']):.4g} m/s"),
        ("rms |U|", f"{float(observables['speed_rms']):.4g} m/s"),
        ("peak |omega|", f"{float(observables['vorticity_peak']):.4g} 1/s"),
    ]
    table = axes[1, 1].table(cellText=rows, colLabels=("observable", "value"), cellLoc="left", colLoc="left", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row, _column), cell in table.get_celld().items():
        cell.set_edgecolor("#cbd5e1")
        if row == 0:
            cell.set_facecolor("#0f172a")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f8fafc")
    axes[1, 1].set_title("Executable external-code field observables")

    fig.suptitle("Q2DmhdFoam lid-driven rerun: VTK field ingestion", fontsize=15.0, fontweight="bold")
    paths = [out_dir / f"{output_stem}.png", out_dir / f"{output_stem}.pdf"]
    for path in paths:
        fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return paths


def write_q2dmhdfoam_external_reference_panel(
    profiles: Sequence[Mapping[str, object]],
    profile_observables: Sequence[Mapping[str, float | str | int]],
    output_dir: str | Path,
    *,
    turbulence_observables: Mapping[str, float | int | str] | None = None,
    force_observables: Sequence[Mapping[str, float | int | str]] | None = None,
    probe_observables: Sequence[Mapping[str, float | int | str]] | None = None,
    output_stem: str = "q2dmhdfoam_external_reference",
) -> list[Path]:
    """Write a publication-facing Q2DmhdFoam external-reference panel."""

    import matplotlib.pyplot as plt

    if not profiles:
        raise ValueError("At least one Q2DmhdFoam profile is required")
    if len(profiles) != len(profile_observables):
        raise ValueError("profiles and profile_observables must have the same length")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.6), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(profiles)))

    for color, profile, observables in zip(colors, profiles, profile_observables, strict=True):
        x = np.asarray(profile["position"], dtype=float)
        u = np.asarray(profile["velocity"], dtype=float)
        mean_velocity = float(observables["mean_velocity"])
        u_norm = u / mean_velocity if abs(mean_velocity) > 1.0e-30 else u
        axes[0, 0].plot(x, u_norm, color=color, linewidth=1.8, label=str(observables["label"]))
    axes[0, 0].set_title("External line-profile reference curves")
    axes[0, 0].set_xlabel("normalized line coordinate")
    axes[0, 0].set_ylabel(r"$U / \overline{U}$")
    axes[0, 0].grid(True, alpha=0.25)
    axes[0, 0].legend(frameon=False, fontsize=7.5, ncols=1, loc="upper right")

    labels = [f"{idx + 1}" for idx in range(len(profile_observables))]
    x_idx = np.arange(len(profile_observables), dtype=float)
    axes[0, 1].bar(
        x_idx,
        [float(row["peak_to_mean_velocity"]) for row in profile_observables],
        color=colors,
        edgecolor="#1f2937",
        linewidth=0.5,
    )
    axes[0, 1].set_title("Shape observable")
    axes[0, 1].set_ylabel(r"$U_{max} / \overline{U}$")
    axes[0, 1].set_xticks(x_idx, labels)
    axes[0, 1].set_xlabel("profile id (legend order)")
    axes[0, 1].grid(True, axis="y", alpha=0.25)

    axes[1, 0].bar(
        x_idx,
        [float(row["symmetry_l2"]) for row in profile_observables],
        color=colors,
        edgecolor="#1f2937",
        linewidth=0.5,
    )
    axes[1, 0].set_title("Profile asymmetry metric")
    axes[1, 0].set_ylabel(r"$L_2(U(x)-U(-x))$")
    axes[1, 0].set_xticks(x_idx, labels)
    axes[1, 0].set_xlabel("profile id (legend order)")
    axes[1, 0].grid(True, axis="y", alpha=0.25)

    axes[1, 1].axis("off")
    turbulence_observables = dict(turbulence_observables or {})
    note = (
        "Adapter status: external data are wired into LMX artifacts; "
        "matched LMX parity remains a separate validation gate."
    )
    force_observables = list(force_observables or [])
    probe_observables = list(probe_observables or [])
    if turbulence_observables or force_observables or probe_observables:
        lines = ["Q2DmhdFoam lid-driven turbulence summary"]
        for key in (
            "weak_mode_count",
            "weak_peak_over_max_max",
            "weak_dominant_wavenumber",
            "strong_mode_count",
            "strong_peak_over_max_max",
            "strong_avg_over_max_max",
        ):
            if key in turbulence_observables:
                value = turbulence_observables[key]
                if isinstance(value, float):
                    lines.append(f"{key}: {value:.4g}")
                else:
                    lines.append(f"{key}: {value}")
        if force_observables:
            force = dict(force_observables[0])
            lines.extend(
                [
                    "",
                    "Cylinder/duct force tail statistics",
                    f"Cd mean: {float(force.get('cd_tail_mean', 0.0)):.4g}",
                    f"Cl rms: {float(force.get('cl_tail_rms', 0.0)):.4g}",
                ]
            )
        if probe_observables:
            probe = dict(probe_observables[0])
            lines.extend(
                [
                    "",
                    "Probe velocity history",
                    f"speed tail mean: {float(probe.get('speed_tail_mean', 0.0)):.4g}",
                    f"transverse tail rms: {float(probe.get('transverse_tail_rms', 0.0)):.4g}",
                ]
            )
        lines.extend(["", note])
        axes[1, 1].text(0.03, 0.95, "\n".join(lines), va="top", fontsize=10, transform=axes[1, 1].transAxes)
    else:
        axes[1, 1].text(
            0.5,
            0.5,
            "No Q2DmhdFoam turbulence summary found",
            ha="center",
            va="center",
            fontsize=11,
            transform=axes[1, 1].transAxes,
        )

    fig.suptitle("Executable Q2DmhdFoam reference-data adapter", fontsize=15, fontweight="bold")
    png_path = out_dir / f"{output_stem}.png"
    pdf_path = out_dir / f"{output_stem}.pdf"
    for path in (png_path, pdf_path):
        fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return [png_path, pdf_path]


def _latest_openfoam_time_dir(case_dir: Path) -> Path:
    candidates: list[tuple[float, Path]] = []
    for path in case_dir.iterdir():
        if not path.is_dir():
            continue
        try:
            value = float(path.name)
        except ValueError:
            continue
        candidates.append((value, path))
    if not candidates:
        raise ValueError(f"No numeric OpenFOAM time directories found under {case_dir}")
    return max(candidates, key=lambda item: item[0])[1]


def _openfoam_scalar_assignments(path: Path) -> dict[str, float]:
    result: dict[str, float] = {}
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_]*)\s+(" + FLOAT_PATTERN + r")\s*;", re.MULTILINE)
    for match in pattern.finditer(text):
        result[match.group(1)] = float(match.group(2))
    return result


def _q2dmhdfoam_lid_driven_y_cells(variables: Mapping[str, float]) -> tuple[np.ndarray, np.ndarray]:
    segments = (
        (variables["yNeg"], variables["yNegBL"], int(round(variables["NyBL"])), variables["GyBL"]),
        (variables["yNegBL"], 0.0, int(round(variables["Ny"])), variables["Gy"]),
        (0.0, variables["yBL"], int(round(variables["Ny"])), variables["GyInv"]),
        (variables["yBL"], variables["y"], int(round(variables["NyBL"])), variables["GyBLinv"]),
    )
    centers: list[float] = []
    widths: list[float] = []
    for start, end, count, ratio in segments:
        segment_centers, segment_widths = _graded_cell_centers(float(start), float(end), int(count), float(ratio))
        centers.extend(segment_centers)
        widths.extend(segment_widths)
    return np.asarray(centers, dtype=float), np.asarray(widths, dtype=float)


def _graded_cell_centers(start: float, end: float, count: int, ratio: float) -> tuple[list[float], list[float]]:
    if count <= 0:
        return [], []
    length = end - start
    if abs(ratio - 1.0) <= 1.0e-12 or count == 1:
        widths = [length / count] * count
    else:
        per_cell_ratio = ratio ** (1.0 / (count - 1))
        first = length * (1.0 - per_cell_ratio) / (1.0 - per_cell_ratio**count)
        widths = [first * per_cell_ratio**index for index in range(count)]
    centers: list[float] = []
    position = start
    for width in widths:
        centers.append(position + 0.5 * width)
        position += width
    return centers, widths


def _read_openfoam_vector_internal_field(path: Path) -> np.ndarray:
    text = path.read_text(encoding="utf-8")
    match = re.search(
        r"internalField\s+nonuniform\s+List<vector>\s+\d+\s*\((.*?)\)\s*;",
        text,
        re.S,
    )
    if not match:
        raise ValueError(f"{path} does not contain a nonuniform List<vector> internalField")
    rows = [
        (float(x), float(y), float(z))
        for x, y, z in re.findall(
            r"\((" + FLOAT_PATTERN + r")\s+(" + FLOAT_PATTERN + r")\s+(" + FLOAT_PATTERN + r")\)",
            match.group(1),
        )
    ]
    if not rows:
        raise ValueError(f"{path} contained no vector rows")
    return np.asarray(rows, dtype=float)


def q2dmhdfoam_case_manifest(case_dir: str | Path) -> dict[str, object]:
    """Return a compact physical/numerical manifest for a Q2DmhdFoam case.

    The manifest is intentionally based on case dictionaries rather than solver
    output. It is used to decide whether an external Q2DmhdFoam run can be
    promoted into a strict LMX-vs-Q2DmhdFoam Q2D turbulence reference.
    """

    root = Path(case_dir)
    control = _read_text_if_exists(root / "system" / "controlDict")
    transport = _read_text_if_exists(root / "constant" / "transportProperties")
    b_field = _read_text_if_exists(root / "0" / "B")
    mesh = _read_text_if_exists(root / "constant" / "polyMesh" / "blockMeshDict")
    if not mesh:
        mesh = _read_text_if_exists(root / "constant" / "polyMesh.org" / "blockMeshDict")
    patch_types = _q2dmhdfoam_patch_types(mesh)
    vertices = _q2dmhdfoam_vertices(mesh)
    cells = _q2dmhdfoam_block_cell_counts(mesh)
    b_internal = _openfoam_internal_uniform_scalar(b_field)
    ubar = _openfoam_dimensioned_vector(transport, "Ubar")
    ubar_magnitude = float(np.linalg.norm(ubar))
    q0 = _openfoam_dimensioned_scalar(transport, "q0")
    nu = _openfoam_dimensioned_scalar(transport, "nu")
    rho0 = _openfoam_dimensioned_scalar(transport, "rho0")
    sigma = _openfoam_dimensioned_scalar(transport, "sigma")
    a = _openfoam_dimensioned_scalar(transport, "a")
    b = _openfoam_dimensioned_scalar(transport, "b")
    if vertices.size:
        domain = {
            "x_min": float(np.min(vertices[:, 0])),
            "x_max": float(np.max(vertices[:, 0])),
            "y_min": float(np.min(vertices[:, 1])),
            "y_max": float(np.max(vertices[:, 1])),
            "z_min": float(np.min(vertices[:, 2])),
            "z_max": float(np.max(vertices[:, 2])),
        }
    else:
        domain = {}
    return {
        "case_dir": str(root),
        "case_name": root.name,
        "application": _openfoam_word(control, "application"),
        "end_time": _openfoam_scalar_assignment(control, "endTime"),
        "delta_t": _openfoam_scalar_assignment(control, "deltaT"),
        "write_interval": _openfoam_scalar_assignment(control, "writeInterval"),
        "probe_count": _q2dmhdfoam_probe_count(control),
        "nu": nu,
        "rho0": rho0,
        "sigma": sigma,
        "a": a,
        "b": b,
        "q0": q0,
        "ubar": [float(value) for value in ubar],
        "ubar_magnitude": ubar_magnitude,
        "magnetic_field_internal": b_internal,
        "hartmann_friction_nonzero": bool(abs(b_internal) > 0.0),
        "thermal_forcing_nonzero": bool(abs(q0) > 0.0),
        "forced_mean_flow": bool(ubar_magnitude > 0.0),
        "patch_types": patch_types,
        "patch_names": sorted(patch_types),
        "has_cylinder_obstacle": "cylinder" in patch_types,
        "has_inlet_outlet": "xinlet" in patch_types or "xoutlet" in patch_types,
        "has_side_walls": "sideWalls" in patch_types,
        "has_empty_hartmann_walls": patch_types.get("hartmannWalls") == "empty",
        "has_cyclic_patch": any(kind == "cyclic" for kind in patch_types.values()),
        "domain": domain,
        "block_count": len(cells),
        "total_cell_count": int(sum(np.prod(count) for count in cells)) if cells else 0,
    }


def audit_q2dmhdfoam_lmx_turbulence_match(
    case_dir: str | Path,
    *,
    lmx_case: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Audit whether a Q2DmhdFoam case can close strict LMX Q2D parity.

    The current strict target is the LMX periodic Sommeria-Moreau-style
    nonlinear vorticity solve used by the README movie. A Q2DmhdFoam case must
    match topology, forcing, Hartmann friction, time window, and observables
    before it is allowed to populate ``q2d_turbulence_reference_observables``.
    """

    manifest = q2dmhdfoam_case_manifest(case_dir)
    target = dict(lmx_case or _default_lmx_q2d_turbulence_target())
    gates = [
        _audit_gate(
            "application",
            "Q2DmhdFoam",
            manifest.get("application") == "Q2DmhdFoam",
            str(manifest.get("application", "")),
        ),
        _audit_gate(
            "topology",
            "periodic box, no inlet/outlet, no obstacle",
            not bool(manifest["has_inlet_outlet"]) and not bool(manifest["has_cylinder_obstacle"]) and bool(manifest["has_cyclic_patch"]),
            _topology_label(manifest),
        ),
        _audit_gate(
            "hartmann_friction",
            "nonzero Hartmann friction matching LMX alpha",
            bool(manifest["hartmann_friction_nonzero"]),
            f"B={float(manifest['magnetic_field_internal']):.6g}",
        ),
        _audit_gate(
            "forcing",
            str(target["forcing_kind"]),
            False,
            _forcing_label(manifest),
        ),
        _audit_gate(
            "time_window",
            f"dt={target['dt']}, t_final={target['t_final']}",
            _close_or_missing(manifest.get("delta_t"), float(target["dt"])) and _close_or_missing(manifest.get("end_time"), float(target["t_final"])),
            f"dt={manifest.get('delta_t')}, endTime={manifest.get('end_time')}",
        ),
        _audit_gate(
            "observables",
            "energy, enstrophy, spectrum, turnover, force/probe histories",
            False,
            f"probe_count={manifest.get('probe_count')}; no energy/enstrophy contract in case dictionaries",
        ),
    ]
    blockers = [gate["criterion"] for gate in gates if not gate["passed"]]
    strict_admissible = not blockers
    return {
        "case": "q2dmhdfoam_lmx_turbulence_match_audit",
        "case_name": manifest["case_name"],
        "manifest": manifest,
        "lmx_target": target,
        "gate_results": gates,
        "blockers": blockers,
        "strict_admissible": strict_admissible,
        "matched_parity": False,
        "strict_blocker_closed": False,
        "decision": "admissible_for_strict_csv" if strict_admissible else "not_admissible_for_strict_csv",
        "required_next_step": (
            "Either create a Q2DmhdFoam case with the same periodic topology, "
            "Hartmann friction, deterministic forcing, timestep window, and "
            "energy/enstrophy/spectrum observables as the LMX SM82 case, or add "
            "the corresponding LMX inlet/outlet obstacle/cavity physics before "
            "using the existing Q2DmhdFoam outputs as strict turbulence parity."
        ),
    }


def write_q2dmhdfoam_lmx_turbulence_match_audit(
    audits: Sequence[Mapping[str, object]],
    output_dir: str | Path,
    *,
    output_stem: str = "q2dmhdfoam_lmx_turbulence_match_audit",
) -> list[Path]:
    """Write JSON/CSV/PNG/PDF artifacts for Q2DmhdFoam-vs-LMX match audits."""

    import matplotlib.pyplot as plt

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_rows = [dict(audit) for audit in audits]
    json_path = out_dir / f"{output_stem}.json"
    csv_path = out_dir / f"{output_stem}.csv"
    png_path = out_dir / f"{output_stem}.png"
    pdf_path = out_dir / f"{output_stem}.pdf"
    json_path.write_text(json.dumps({"case": output_stem, "audits": audit_rows}, indent=2) + "\n", encoding="utf-8")
    _write_q2dmhdfoam_match_audit_csv(audit_rows, csv_path)

    criteria = sorted({str(gate["criterion"]) for audit in audit_rows for gate in audit.get("gate_results", [])})
    if not criteria:
        criteria = ["no gates"]
    matrix = np.zeros((len(audit_rows), len(criteria)), dtype=float)
    for row_index, audit in enumerate(audit_rows):
        by_name = {str(gate["criterion"]): bool(gate["passed"]) for gate in audit.get("gate_results", [])}
        for column_index, criterion in enumerate(criteria):
            matrix[row_index, column_index] = 1.0 if by_name.get(criterion, False) else 0.0

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4), constrained_layout=True)
    image = axes[0].imshow(matrix, vmin=0.0, vmax=1.0, cmap="RdYlGn", aspect="auto")
    axes[0].set_yticks(np.arange(len(audit_rows)), [str(audit.get("case_name", f"case {idx+1}")) for idx, audit in enumerate(audit_rows)])
    axes[0].set_xticks(np.arange(len(criteria)), criteria, rotation=35, ha="right")
    axes[0].set_title("Strict Q2D parity admissibility gates")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axes[0].text(column, row, "pass" if matrix[row, column] > 0.5 else "open", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=axes[0], ticks=[0, 1], fraction=0.046, pad=0.04)

    axes[1].axis("off")
    lines = ["Decision summary"]
    for audit in audit_rows:
        blockers = list(audit.get("blockers", []))
        lines.append("")
        lines.append(f"{audit.get('case_name')}: {audit.get('decision')}")
        lines.append(f"open gates: {', '.join(map(str, blockers)) if blockers else 'none'}")
    lines.extend(
        [
            "",
            "No row is promoted into q2d_turbulence_reference_observables.csv unless all strict gates pass.",
        ]
    )
    axes[1].text(0.02, 0.98, "\n".join(lines), va="top", fontsize=10.0, transform=axes[1].transAxes)
    fig.suptitle("Q2DmhdFoam-to-LMX nonlinear Q2D match audit", fontsize=15, fontweight="bold")
    for path in (png_path, pdf_path):
        fig.savefig(path, dpi=185, bbox_inches="tight")
    plt.close(fig)
    return [json_path, csv_path, png_path, pdf_path]


def _write_q2dmhdfoam_match_audit_csv(audits: Sequence[Mapping[str, object]], path: Path) -> None:
    columns = [
        "case_name",
        "criterion",
        "passed",
        "expected",
        "observed",
        "decision",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for audit in audits:
            for gate in audit.get("gate_results", []):
                writer.writerow(
                    {
                        "case_name": audit.get("case_name", ""),
                        "criterion": gate.get("criterion", ""),
                        "passed": gate.get("passed", ""),
                        "expected": gate.get("expected", ""),
                        "observed": gate.get("observed", ""),
                        "decision": audit.get("decision", ""),
                    }
                )


def _default_lmx_q2d_turbulence_target() -> dict[str, object]:
    return {
        "model": "periodic_sommeria_moreau_vorticity",
        "geometry_kind": "periodic_box",
        "lx": 2.0,
        "ly": 2.0,
        "viscosity": 8.0e-4,
        "hartmann_friction": 0.08,
        "forcing_kind": "deterministic_periodic_vorticity_forcing",
        "dt": 2.0e-3,
        "t_final": 3.0,
        "required_observables": [
            "energy_decay_ratio",
            "enstrophy_decay_ratio",
            "final_spectral_centroid",
            "final_high_k_energy_fraction",
            "turnover_count",
        ],
    }


def _audit_gate(criterion: str, expected: str, passed: bool, observed: str) -> dict[str, object]:
    return {
        "criterion": criterion,
        "expected": expected,
        "observed": observed,
        "passed": bool(passed),
    }


def _topology_label(manifest: Mapping[str, object]) -> str:
    flags = []
    if manifest.get("has_cylinder_obstacle"):
        flags.append("cylinder")
    if manifest.get("has_inlet_outlet"):
        flags.append("inlet/outlet")
    if manifest.get("has_side_walls"):
        flags.append("sideWalls")
    if manifest.get("has_cyclic_patch"):
        flags.append("cyclic")
    if manifest.get("has_empty_hartmann_walls"):
        flags.append("empty_hartmannWalls")
    return ", ".join(flags) if flags else "no recognized topology flags"


def _forcing_label(manifest: Mapping[str, object]) -> str:
    pieces = []
    if manifest.get("thermal_forcing_nonzero"):
        pieces.append(f"thermal q0={float(manifest.get('q0', 0.0)):.6g}")
    if manifest.get("forced_mean_flow"):
        pieces.append(f"mean-flow Ubar={float(manifest.get('ubar_magnitude', 0.0)):.6g}")
    if not pieces:
        pieces.append("no q0/Ubar forcing in dictionaries")
    return ", ".join(pieces)


def _close_or_missing(value: object, target: float, *, rel_tol: float = 1.0e-8, abs_tol: float = 1.0e-12) -> bool:
    if value is None:
        return False
    try:
        return bool(abs(float(value) - target) <= max(abs_tol, rel_tol * max(abs(target), 1.0)))
    except (TypeError, ValueError):
        return False


def _read_text_if_exists(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def _strip_cpp_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//.*", "", text)


def _openfoam_word(text: str, key: str) -> str:
    clean = _strip_cpp_comments(text)
    match = re.search(r"\b" + re.escape(key) + r"\s+([A-Za-z_][A-Za-z0-9_]*)\s*;", clean)
    return match.group(1) if match else ""


def _openfoam_scalar_assignment(text: str, key: str) -> float | None:
    clean = _strip_cpp_comments(text)
    match = re.search(r"\b" + re.escape(key) + r"\s+(" + FLOAT_PATTERN + r")\s*;", clean)
    return float(match.group(1)) if match else None


def _openfoam_dimensioned_scalar(text: str, key: str) -> float:
    clean = _strip_cpp_comments(text)
    pattern = (
        r"\b"
        + re.escape(key)
        + r"\s+(?:[A-Za-z_][A-Za-z0-9_]*\s+)?(?:\[[^\]]+\]\s+)?("
        + FLOAT_PATTERN
        + r")\s*;"
    )
    match = re.search(pattern, clean)
    return float(match.group(1)) if match else 0.0


def _openfoam_dimensioned_vector(text: str, key: str) -> np.ndarray:
    clean = _strip_cpp_comments(text)
    pattern = (
        r"\b"
        + re.escape(key)
        + r"\s+(?:[A-Za-z_][A-Za-z0-9_]*\s+)?(?:\[[^\]]+\]\s+)?\(("
        + FLOAT_PATTERN
        + r")\s+("
        + FLOAT_PATTERN
        + r")\s+("
        + FLOAT_PATTERN
        + r")\)\s*;"
    )
    match = re.search(pattern, clean)
    if not match:
        return np.zeros(3, dtype=float)
    return np.asarray([float(match.group(1)), float(match.group(2)), float(match.group(3))], dtype=float)


def _openfoam_internal_uniform_scalar(text: str) -> float:
    clean = _strip_cpp_comments(text)
    match = re.search(r"\binternalField\s+uniform\s+(" + FLOAT_PATTERN + r")\s*;", clean)
    return float(match.group(1)) if match else 0.0


def _q2dmhdfoam_patch_types(mesh_text: str) -> dict[str, str]:
    clean = _strip_cpp_comments(mesh_text)
    patch_types: dict[str, str] = {}
    for kind, name in re.findall(r"\b(patch|wall|empty|cyclic|symmetryPlane)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", clean):
        patch_types[name] = kind
    return patch_types


def _q2dmhdfoam_vertices(mesh_text: str) -> np.ndarray:
    clean = _strip_cpp_comments(mesh_text)
    match = re.search(r"\bvertices\s*\((.*?)\)\s*;", clean, flags=re.S)
    if not match:
        return np.empty((0, 3), dtype=float)
    rows = [
        (float(x), float(y), float(z))
        for x, y, z in re.findall(
            r"\((" + FLOAT_PATTERN + r")\s+(" + FLOAT_PATTERN + r")\s+(" + FLOAT_PATTERN + r")\)",
            match.group(1),
        )
    ]
    return np.asarray(rows, dtype=float) if rows else np.empty((0, 3), dtype=float)


def _q2dmhdfoam_block_cell_counts(mesh_text: str) -> list[tuple[int, int, int]]:
    clean = _strip_cpp_comments(mesh_text)
    return [
        (int(nx), int(ny), int(nz))
        for nx, ny, nz in re.findall(
            r"\bhex\s*\([^)]+\)\s*\(\s*(\d+)\s+(\d+)\s+(\d+)\s*\)",
            clean,
        )
    ]


def _q2dmhdfoam_probe_count(control_text: str) -> int:
    clean = _strip_cpp_comments(control_text)
    match = re.search(r"\bprobeLocations\s*\((.*?)\)\s*;", clean, flags=re.S)
    if not match:
        return 0
    return len(
        re.findall(
            r"\((" + FLOAT_PATTERN + r")\s+(" + FLOAT_PATTERN + r")\s+(" + FLOAT_PATTERN + r")\)",
            match.group(1),
        )
    )


def load_magnetic_obstacle_reference_observables(path: str | Path) -> dict[str, dict[str, float | str]]:
    """Load scalar magnetic-obstacle reference observables from CSV.

    Required columns are ``observable``, ``value``, and ``tolerance``. Optional
    columns such as ``units``, ``source``, and ``note`` are preserved in each
    record. Tolerances are interpreted as absolute tolerances unless a row also
    supplies ``relative_tolerance``.
    """

    return load_scalar_reference_observables(path, context="Magnetic-obstacle reference CSV")


def compare_magnetic_obstacle_reference_observables(
    lmx_observables: Mapping[str, float],
    reference_observables: Mapping[str, Mapping[str, float | str]],
) -> dict[str, object]:
    """Compare LMX magnetic-obstacle observables with loaded reference rows."""

    return compare_scalar_reference_observables(lmx_observables, reference_observables)


def write_magnetic_obstacle_reference_comparison_table(
    comparison: Mapping[str, object],
    path: str | Path,
) -> Path:
    """Write a CSV table from ``compare_magnetic_obstacle_reference_observables``."""

    return write_scalar_reference_comparison_table(comparison, path)


def write_magnetic_obstacle_reference_comparison_plots(
    comparison: Mapping[str, object],
    output_dir: str | Path,
    *,
    output_stem: str = "magnetic_obstacle_reference_comparison",
) -> list[Path]:
    """Write PNG/PDF plots for a magnetic-obstacle reference comparison.

    The figure is intentionally observable-level rather than field-level: the
    external magnetic-obstacle references are expected to come from digitized
    literature or experimental scalar observables first. Field overlays belong
    in case-specific examples once a fully matched reference dataset exists.
    """

    return write_scalar_reference_comparison_plots(
        comparison,
        output_dir,
        output_stem=output_stem,
        title="Magnetic-obstacle external-reference observables",
        no_data_label="No compared magnetic-obstacle observables",
    )


def magnetic_obstacle_reference_template_rows() -> list[dict[str, str]]:
    """Return the scalar-observable CSV template expected for external parity."""

    return [
        {
            "observable": "centerline_velocity_deficit_ratio",
            "value": "",
            "tolerance": "",
            "relative_tolerance": "0.10",
            "units": "dimensionless",
            "source": "digitized literature or experiment",
            "note": "Peak or station-matched centerline wake deficit.",
        },
        {
            "observable": "wake_recovery_ratio",
            "value": "",
            "tolerance": "",
            "relative_tolerance": "0.10",
            "units": "dimensionless",
            "source": "digitized literature or experiment",
            "note": "Downstream recovery metric at the documented outlet or recovery station.",
        },
        {
            "observable": "minimum_centerline_velocity_ratio",
            "value": "",
            "tolerance": "",
            "relative_tolerance": "0.10",
            "units": "dimensionless",
            "source": "Votyakov et al. JFM 610 figure 7 or matched external solver",
            "note": "Minimum streamwise centerline velocity normalized by the inlet/upstream speed; negative values indicate reverse flow.",
        },
        {
            "observable": "normalized_recovery_distance",
            "value": "",
            "tolerance": "",
            "relative_tolerance": "0.15",
            "units": "dimensionless",
            "source": "digitized literature or matched external solver",
            "note": "Distance from peak field to recovery station normalized by downstream length.",
        },
        {
            "observable": "pressure_drop_proxy",
            "value": "",
            "tolerance": "",
            "relative_tolerance": "0.15",
            "units": "case-specific",
            "source": "digitized literature or experiment",
            "note": "Use the same nondimensionalization as the reference paper.",
        },
        {
            "observable": "current_proxy_peak",
            "value": "",
            "tolerance": "",
            "relative_tolerance": "0.15",
            "units": "case-specific",
            "source": "digitized literature or experiment",
            "note": "Peak induced-current or Lorentz-force proxy if directly available.",
        },
    ]


def write_magnetic_obstacle_reference_template(path: str | Path) -> Path:
    """Write the external magnetic-obstacle observable CSV template."""

    return write_scalar_reference_template(path, magnetic_obstacle_reference_template_rows())


def q2d_turbulence_reference_template_rows() -> list[dict[str, str]]:
    """Return scalar-observable rows for external Q2D turbulent parity."""

    return [
        {
            "observable": "energy_decay_ratio",
            "value": "",
            "tolerance": "",
            "relative_tolerance": "0.10",
            "units": "dimensionless",
            "source": "digitized Sommeria-Moreau-style turbulent reference",
            "note": "Final-to-initial kinetic-energy ratio at matched Hartmann friction and time.",
        },
        {
            "observable": "enstrophy_decay_ratio",
            "value": "",
            "tolerance": "",
            "relative_tolerance": "0.15",
            "units": "dimensionless",
            "source": "digitized Sommeria-Moreau-style turbulent reference",
            "note": "Final-to-initial enstrophy proxy ratio at matched parameters.",
        },
        {
            "observable": "final_spectral_centroid",
            "value": "",
            "tolerance": "",
            "relative_tolerance": "0.10",
            "units": "1/length",
            "source": "digitized spectrum or reference output",
            "note": "Shell-energy spectral centroid at the final compared time.",
        },
        {
            "observable": "final_high_k_energy_fraction",
            "value": "",
            "tolerance": "",
            "relative_tolerance": "0.15",
            "units": "dimensionless",
            "source": "digitized spectrum or reference output",
            "note": "Fraction of shell energy above the documented high-wavenumber cutoff.",
        },
        {
            "observable": "turnover_count",
            "value": "",
            "tolerance": "",
            "relative_tolerance": "0.10",
            "units": "dimensionless",
            "source": "reference runtime diagnostics",
            "note": "Integrated eddy-turnover proxy over the compared interval.",
        },
    ]


def dean_vortex_reference_template_rows() -> list[dict[str, str]]:
    """Return scalar-observable rows for higher-inertia Dean-vortex parity."""

    return [
        {
            "observable": "secondary_flow_rms_ratio",
            "value": "",
            "tolerance": "",
            "relative_tolerance": "0.15",
            "units": "dimensionless",
            "source": "curved-duct or curved-pipe literature/reference solver",
            "note": "RMS secondary-flow speed normalized by axial speed.",
        },
        {
            "observable": "secondary_flow_peak_ratio",
            "value": "",
            "tolerance": "",
            "relative_tolerance": "0.15",
            "units": "dimensionless",
            "source": "curved-duct or curved-pipe literature/reference solver",
            "note": "Peak secondary-flow speed normalized by peak axial speed.",
        },
        {
            "observable": "normalized_velocity_centroid_shift",
            "value": "",
            "tolerance": "",
            "relative_tolerance": "0.15",
            "units": "dimensionless",
            "source": "curved-duct or curved-pipe literature/reference solver",
            "note": "Axial-velocity centroid displacement normalized by pipe radius.",
        },
        {
            "observable": "inner_outer_velocity_ratio",
            "value": "",
            "tolerance": "",
            "relative_tolerance": "0.10",
            "units": "dimensionless",
            "source": "curved-duct or curved-pipe literature/reference solver",
            "note": "Outer-wall to inner-wall axial-speed ratio on the diameter cut.",
        },
        {
            "observable": "pressure_loss_proxy",
            "value": "",
            "tolerance": "",
            "relative_tolerance": "0.15",
            "units": "case-specific",
            "source": "curved-duct or curved-pipe literature/reference solver",
            "note": "Use the same nondimensional pressure-loss convention as the reference.",
        },
    ]


def write_q2d_turbulence_reference_template(path: str | Path) -> Path:
    """Write the external Q2D turbulence observable CSV template."""

    return write_scalar_reference_template(path, q2d_turbulence_reference_template_rows())


def write_dean_vortex_reference_template(path: str | Path) -> Path:
    """Write the external Dean-vortex observable CSV template."""

    return write_scalar_reference_template(path, dean_vortex_reference_template_rows())


def _require_token(tokens: Sequence[str], token: str, source: Path) -> int:
    items = tokens if isinstance(tokens, list) else list(tokens)
    try:
        return items.index(token)
    except ValueError as exc:
        raise ValueError(f"VTK file {source} does not contain token {token!r}") from exc


def _point_vector_grid(points: np.ndarray, vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    coords = np.asarray(points, dtype=float)
    values = np.asarray(vectors, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("Q2DmhdFoam VTK grid conversion expects point coordinates with shape (n, 3)")
    if values.ndim != 2 or values.shape[0] != coords.shape[0] or values.shape[1] < 2:
        raise ValueError("Q2DmhdFoam VTK grid conversion expects matching point-vector rows")

    z_values = np.unique(coords[:, 2])
    z_target = z_values[int(np.argmin(np.abs(z_values - float(np.median(z_values)))))]
    z_tolerance = max(1.0e-12, 10.0 * np.finfo(float).eps * max(1.0, abs(float(z_target))))
    z_mask = np.abs(coords[:, 2] - z_target) <= z_tolerance
    section = coords[z_mask]
    section_values = values[z_mask]
    x_unique = np.unique(section[:, 0])
    y_unique = np.unique(section[:, 1])
    if x_unique.size < 2 or y_unique.size < 2:
        raise ValueError("Q2DmhdFoam VTK point field does not define a 2D section")

    ux = np.full((y_unique.size, x_unique.size), np.nan, dtype=float)
    uy = np.full_like(ux, np.nan)
    speed = np.full_like(ux, np.nan)
    ix = np.searchsorted(x_unique, section[:, 0])
    iy = np.searchsorted(y_unique, section[:, 1])
    magnitudes = np.linalg.norm(section_values[:, : min(3, section_values.shape[1])], axis=1)
    ux[iy, ix] = section_values[:, 0]
    uy[iy, ix] = section_values[:, 1]
    speed[iy, ix] = magnitudes
    return x_unique, y_unique, ux, uy, speed


def _q2dmhdfoam_conditions_from_name(name: str) -> dict[str, float]:
    match = re.search(r"lineSampled_theta_Ux_([0-9.eE+-]+)_([0-9.eE+-]+)_([0-9.eE+-]+)", name)
    if not match:
        return {}
    return {
        "hartmann": float(match.group(1)),
        "reynolds": float(match.group(2)),
        "grashof": float(match.group(3)),
    }


def _q2dmhdfoam_profile_label(name: str, metadata: Mapping[str, float]) -> str:
    if {"hartmann", "reynolds", "grashof"} <= set(metadata):
        return f"Ha={metadata['hartmann']:g}, Re={metadata['reynolds']:g}, Gr={metadata['grashof']:g}"
    return Path(name).stem.replace("_", " ")


def _parse_q2dmhdfoam_list_payload(text: str, label: str) -> list[list[float]]:
    match = re.search(rf"{re.escape(label)}\s*:\s*(\[[^\n\r]*\])", text)
    if not match:
        return []
    payload = ast.literal_eval(match.group(1))
    if not isinstance(payload, list):
        raise ValueError(f"{label} payload must be a list")
    return payload


def _parse_float(value: str | None, *, row_number: int, column: str, context: str = "Scalar reference CSV") -> float:
    text = (value or "").strip()
    if not text:
        raise ValueError(f"{context} row {row_number} has an empty {column}")
    parsed = float(text)
    if not np.isfinite(parsed):
        raise ValueError(f"{context} row {row_number} has a non-finite {column}")
    return parsed


def _compact_observable_label(observable: str) -> str:
    tokens = observable.replace("_", " ").split()
    lines: list[str] = []
    current: list[str] = []
    current_length = 0
    for token in tokens:
        proposed_length = current_length + len(token) + (1 if current else 0)
        if current and proposed_length > 18:
            lines.append(" ".join(current))
            current = [token]
            current_length = len(token)
        else:
            current.append(token)
            current_length = proposed_length
    if current:
        lines.append(" ".join(current))
    return "\n".join(lines)


def _readiness_color(score: float) -> str:
    if score >= EXTERNAL_VALIDATION_READY_SCORE:
        return "#2a9d8f"
    if score >= 2.0:
        return "#d97706"
    return "#c2410c"


def _wrap_text(text: str, width: int) -> str:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    length = 0
    for word in words:
        proposed = length + len(word) + (1 if current else 0)
        if current and proposed > width:
            lines.append(" ".join(current))
            current = [word]
            length = len(word)
        else:
            current.append(word)
            length = proposed
    if current:
        lines.append(" ".join(current))
    return "\n".join(lines)

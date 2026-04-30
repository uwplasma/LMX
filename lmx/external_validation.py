"""External literature-reference validation helpers.

These utilities intentionally stay independent from the solvers. They define
the data contract used to turn digitized literature/experimental observables
into repeatable validation gates and publication-ready comparison tables.
"""

from __future__ import annotations

import ast
import csv
import re
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


SCALAR_REFERENCE_COLUMNS = ("observable", "value", "tolerance")
MAGNETIC_OBSTACLE_REFERENCE_COLUMNS = SCALAR_REFERENCE_COLUMNS
EXTERNAL_VALIDATION_READY_SCORE = 3.0


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
            "score": 2.5,
            "status": "external adapter wired",
            "observables": "profiles, energy, enstrophy, spectra, turnover count",
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
            "external_code": "OpenFOAM curved-pipe + Dean literature",
            "score": 1.0,
            "status": "reference path identified",
            "observables": "secondary-flow intensity, centroid shift, pressure loss",
            "next_step": "construct hydrodynamic curved-pipe reference",
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


def write_q2dmhdfoam_external_reference_panel(
    profiles: Sequence[Mapping[str, object]],
    profile_observables: Sequence[Mapping[str, float | str | int]],
    output_dir: str | Path,
    *,
    turbulence_observables: Mapping[str, float | int | str] | None = None,
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
    if turbulence_observables:
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

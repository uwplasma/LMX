"""External literature-reference validation helpers.

These utilities intentionally stay independent from the solvers. They define
the data contract used to turn digitized literature/experimental observables
into repeatable validation gates and publication-ready comparison tables.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping

import numpy as np


SCALAR_REFERENCE_COLUMNS = ("observable", "value", "tolerance")
MAGNETIC_OBSTACLE_REFERENCE_COLUMNS = SCALAR_REFERENCE_COLUMNS


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
        writer = csv.DictWriter(handle, fieldnames=columns)
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
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return out


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

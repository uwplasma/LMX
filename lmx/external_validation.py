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


MAGNETIC_OBSTACLE_REFERENCE_COLUMNS = ("observable", "value", "tolerance")


def load_magnetic_obstacle_reference_observables(path: str | Path) -> dict[str, dict[str, float | str]]:
    """Load scalar magnetic-obstacle reference observables from CSV.

    Required columns are ``observable``, ``value``, and ``tolerance``. Optional
    columns such as ``units``, ``source``, and ``note`` are preserved in each
    record. Tolerances are interpreted as absolute tolerances unless a row also
    supplies ``relative_tolerance``.
    """

    source = Path(path)
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        missing = [column for column in MAGNETIC_OBSTACLE_REFERENCE_COLUMNS if column not in fieldnames]
        if missing:
            missing_text = ", ".join(missing)
            raise ValueError(f"Magnetic-obstacle reference CSV is missing required columns: {missing_text}")
        records: dict[str, dict[str, float | str]] = {}
        for row_number, row in enumerate(reader, start=2):
            observable = (row.get("observable") or "").strip()
            if not observable:
                raise ValueError(f"Magnetic-obstacle reference CSV row {row_number} has an empty observable")
            value = _parse_float(row.get("value"), row_number=row_number, column="value")
            tolerance = _parse_float(row.get("tolerance"), row_number=row_number, column="tolerance")
            if tolerance < 0.0:
                raise ValueError(f"Magnetic-obstacle reference CSV row {row_number} has a negative tolerance")
            payload: dict[str, float | str] = {"value": value, "tolerance": tolerance}
            relative_text = (row.get("relative_tolerance") or "").strip()
            if relative_text:
                relative_tolerance = _parse_float(relative_text, row_number=row_number, column="relative_tolerance")
                if relative_tolerance < 0.0:
                    raise ValueError(
                        f"Magnetic-obstacle reference CSV row {row_number} has a negative relative_tolerance"
                    )
                payload["relative_tolerance"] = relative_tolerance
            for key, item in row.items():
                if key not in {"observable", "value", "tolerance", "relative_tolerance"} and item:
                    payload[key] = item.strip()
            records[observable] = payload
    return records


def compare_magnetic_obstacle_reference_observables(
    lmx_observables: Mapping[str, float],
    reference_observables: Mapping[str, Mapping[str, float | str]],
) -> dict[str, object]:
    """Compare LMX magnetic-obstacle observables with loaded reference rows."""

    rows: list[dict[str, float | str | bool]] = []
    compared = 0
    passed = 0
    for observable, reference in reference_observables.items():
        if observable not in lmx_observables:
            rows.append(
                {
                    "observable": observable,
                    "status": "missing_lmx_observable",
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
    missing_reference_observables = sorted(set(lmx_observables) - set(reference_observables))
    return {
        "compared_observable_count": compared,
        "passed_observable_count": passed,
        "missing_lmx_observable_count": sum(1 for row in rows if row["status"] == "missing_lmx_observable"),
        "extra_lmx_observables": missing_reference_observables,
        "rows": rows,
        "validation_pass": bool(compared > 0 and passed == compared and all(row["validation_pass"] for row in rows)),
    }


def write_magnetic_obstacle_reference_comparison_table(
    comparison: Mapping[str, object],
    path: str | Path,
) -> Path:
    """Write a CSV table from ``compare_magnetic_obstacle_reference_observables``."""

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

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = magnetic_obstacle_reference_template_rows()
    columns = ("observable", "value", "tolerance", "relative_tolerance", "units", "source", "note")
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return out


def _parse_float(value: str | None, *, row_number: int, column: str) -> float:
    text = (value or "").strip()
    if not text:
        raise ValueError(f"Magnetic-obstacle reference CSV row {row_number} has an empty {column}")
    parsed = float(text)
    if not np.isfinite(parsed):
        raise ValueError(f"Magnetic-obstacle reference CSV row {row_number} has a non-finite {column}")
    return parsed

"""Reduced Li/AlN wall-stack study helpers.

These helpers implement the Phase 0--2 part of the Li/AlN wall-stack plan:
unit/nondimensional audits, reduced tangential/normal wall conductance models,
smooth pinhole sweeps, and plot/table artifacts.  They are intentionally
electrical-performance reductions, not material-compatibility claims.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable, Sequence

from .units import (
    dynamic_to_kinematic_viscosity,
    hartmann_number,
    interaction_parameter,
    magnetic_reynolds_number,
    normal_leakage_ratio,
    reynolds_number,
    wall_conductance_ratio,
)
from .wall_models import (
    WallLayer,
    effective_pinhole_conductance_ratio,
    nested_wall_layer_resolution_summary,
    normal_stack_leakage_ratio,
    tangential_stack_conductance_ratio,
)


@dataclass(frozen=True)
class LithiumMaterial:
    """SI lithium properties used by reduced wall-stack studies."""

    temperature_c: float
    density: float
    dynamic_viscosity: float
    electrical_conductivity: float

    @property
    def kinematic_viscosity(self) -> float:
        return dynamic_to_kinematic_viscosity(self.dynamic_viscosity, self.density)


@dataclass(frozen=True)
class WallStackStudyCase:
    """Minimal reproducible inputs for a reduced Li/AlN wall-stack study."""

    name: str
    length_scale: float
    velocity: float
    magnetic_field: float
    lithium: LithiumMaterial
    aln_thickness: float
    aln_cells: int
    metal_name: str
    metal_conductivity: float
    metal_thickness: float
    metal_cells: int
    intact_aln_conductivity: float
    degraded_aln_conductivity: float


DEFAULT_LI_ALN_CASE = WallStackStudyCase(
    name="li_aln_rectangular_wall_stack_phase0_2",
    length_scale=0.05,
    velocity=0.04,
    magnetic_field=2.0,
    lithium=LithiumMaterial(
        temperature_c=250.0,
        density=500.0,
        dynamic_viscosity=4.0e-4,
        electrical_conductivity=3.2e6,
    ),
    aln_thickness=2.0e-4,
    aln_cells=4,
    metal_name="316L",
    metal_conductivity=1.35e6,
    metal_thickness=1.0e-3,
    metal_cells=8,
    intact_aln_conductivity=1.0e-8,
    degraded_aln_conductivity=1.0e-3,
)


def li_aln_phase0_2_summary(
    case: WallStackStudyCase = DEFAULT_LI_ALN_CASE,
    *,
    conductance_ratios: Sequence[float] | None = None,
    pinhole_fractions: Sequence[float] | None = None,
    minimum_cells_per_layer: int = 3,
) -> dict[str, object]:
    """Return a Phase 0--2 reduced Li/AlN wall-stack summary.

    The scalar response is a nondimensional current-closure/drag proxy that
    increases monotonically with effective wall conductance and interaction
    parameter.  It is useful for ranking reduced electrical wall models before
    committing to full multilayer solves.
    """

    c_values = tuple(conductance_ratios or (0.0, 1.0e-10, 1.0e-8, 1.0e-6, 1.0e-4, 1.0e-2, 1.0e-1, 1.0, 10.0))
    f_values = tuple(pinhole_fractions or (0.0, 1.0e-6, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0))
    unit_audit = li_aln_unit_audit(case)
    stack = li_aln_wall_layers(case, aln_conductivity=case.intact_aln_conductivity)
    degraded_stack = li_aln_wall_layers(case, aln_conductivity=case.degraded_aln_conductivity)
    metal_c = wall_conductance_ratio(
        wall_conductivity=case.metal_conductivity,
        wall_thickness=case.metal_thickness,
        fluid_conductivity=case.lithium.electrical_conductivity,
        length_scale=case.length_scale,
    )
    intact_c = wall_conductance_ratio(
        wall_conductivity=case.intact_aln_conductivity,
        wall_thickness=case.aln_thickness,
        fluid_conductivity=case.lithium.electrical_conductivity,
        length_scale=case.length_scale,
    )
    degraded_c = wall_conductance_ratio(
        wall_conductivity=case.degraded_aln_conductivity,
        wall_thickness=case.aln_thickness,
        fluid_conductivity=case.lithium.electrical_conductivity,
        length_scale=case.length_scale,
    )
    response_rows = [
        _wall_response_row("ideal_insulator", 0.0, 0.0, unit_audit["interaction_parameter"], 0.0),
        _wall_response_row("bare_metal", metal_c, metal_c, unit_audit["interaction_parameter"], 1.0),
        _wall_response_row("intact_aln", intact_c, intact_c, unit_audit["interaction_parameter"], 0.0),
        _wall_response_row("degraded_aln", degraded_c, degraded_c, unit_audit["interaction_parameter"], 0.0),
    ]
    for c_aln in c_values:
        for f_p in f_values:
            c_eff = effective_pinhole_conductance_ratio(
                intact_conductance_ratio=float(c_aln),
                metal_conductance_ratio=metal_c,
                pinhole_fraction=float(f_p),
            )
            response_rows.append(
                _wall_response_row(
                    "pinhole_sweep",
                    float(c_aln),
                    c_eff,
                    unit_audit["interaction_parameter"],
                    float(f_p),
                )
            )
    return {
        "case": case.name,
        "scope": "reduced_mhd_electrical_performance_only",
        "material_compatibility_claim": False,
        "inputs": _case_payload(case),
        "unit_audit": unit_audit,
        "wall_stack": {
            "intact_layers": [asdict(layer) for layer in stack],
            "degraded_layers": [asdict(layer) for layer in degraded_stack],
            "intact_tangential_conductance_ratio": tangential_stack_conductance_ratio(
                stack,
                fluid_conductivity=case.lithium.electrical_conductivity,
                length_scale=case.length_scale,
            ),
            "intact_normal_leakage_ratio": normal_stack_leakage_ratio(
                stack,
                fluid_conductivity=case.lithium.electrical_conductivity,
                length_scale=case.length_scale,
            ),
            "degraded_tangential_conductance_ratio": tangential_stack_conductance_ratio(
                degraded_stack,
                fluid_conductivity=case.lithium.electrical_conductivity,
                length_scale=case.length_scale,
            ),
            "degraded_normal_leakage_ratio": normal_stack_leakage_ratio(
                degraded_stack,
                fluid_conductivity=case.lithium.electrical_conductivity,
                length_scale=case.length_scale,
            ),
            "mesh_resolution": nested_wall_layer_resolution_summary(
                stack,
                minimum_cells_per_layer=minimum_cells_per_layer,
            ),
        },
        "response_rows": response_rows,
        "thresholds": _wall_thresholds(response_rows),
        "phase_status": {
            "phase_0_repository_preparation": "recorded_by_artifact_metadata",
            "phase_1_units_and_properties": "complete_for_reduced_case",
            "phase_2_reduced_wall_models": "complete_for_conductance_and_pinhole_sweeps",
            "true_multilayer_geometry": "planned_solver_extension",
        },
    }


def li_aln_unit_audit(case: WallStackStudyCase = DEFAULT_LI_ALN_CASE) -> dict[str, float | str | bool]:
    """Return unit and nondimensional-number checks for a Li/AlN case."""

    lithium = case.lithium
    nu = lithium.kinematic_viscosity
    ha = hartmann_number(
        magnetic_field=case.magnetic_field,
        length_scale=case.length_scale,
        conductivity=lithium.electrical_conductivity,
        density=lithium.density,
        kinematic_viscosity=nu,
    )
    re = reynolds_number(
        velocity=case.velocity,
        length_scale=case.length_scale,
        kinematic_viscosity=nu,
    )
    interaction = interaction_parameter(
        magnetic_field=case.magnetic_field,
        length_scale=case.length_scale,
        conductivity=lithium.electrical_conductivity,
        density=lithium.density,
        velocity=case.velocity,
    )
    rm = magnetic_reynolds_number(
        velocity=case.velocity,
        length_scale=case.length_scale,
        conductivity=lithium.electrical_conductivity,
    )
    return {
        "viscosity_convention": "kinematic_nu_m2_per_s",
        "dynamic_viscosity_pa_s": float(lithium.dynamic_viscosity),
        "density_kg_m3": float(lithium.density),
        "kinematic_viscosity_m2_s": float(nu),
        "electrical_conductivity_s_m": float(lithium.electrical_conductivity),
        "hartmann_number": float(ha),
        "reynolds_number": float(re),
        "interaction_parameter": float(interaction),
        "magnetic_reynolds_number": float(rm),
        "inductionless_assumption_pass": bool(rm < 1.0e-2),
    }


def li_aln_wall_layers(case: WallStackStudyCase = DEFAULT_LI_ALN_CASE, *, aln_conductivity: float | None = None) -> tuple[WallLayer, WallLayer]:
    """Return the reduced fluid-facing AlN plus metal wall stack."""

    return (
        WallLayer(
            "aln",
            conductivity=float(case.intact_aln_conductivity if aln_conductivity is None else aln_conductivity),
            thickness=float(case.aln_thickness),
            cells=int(case.aln_cells),
        ),
        WallLayer(
            case.metal_name,
            conductivity=float(case.metal_conductivity),
            thickness=float(case.metal_thickness),
            cells=int(case.metal_cells),
        ),
    )


def write_li_aln_phase0_2_artifacts(
    out_dir: str | Path,
    *,
    case: WallStackStudyCase = DEFAULT_LI_ALN_CASE,
    conductance_ratios: Sequence[float] | None = None,
    pinhole_fractions: Sequence[float] | None = None,
    filename_stem: str = "li_aln_wall_stack_phase0_2",
) -> list[Path]:
    """Write JSON, CSV, and PNG artifacts for the reduced Li/AlN study."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = li_aln_phase0_2_summary(
        case,
        conductance_ratios=conductance_ratios,
        pinhole_fractions=pinhole_fractions,
    )
    json_path = out / f"{filename_stem}_summary.json"
    response_csv = out / f"{filename_stem}_response.csv"
    units_csv = out / f"{filename_stem}_unit_audit.csv"
    png_path = out / f"{filename_stem}.png"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_mapping_csv(units_csv, summary["unit_audit"])
    _write_response_csv(response_csv, summary["response_rows"])
    _write_li_aln_phase0_2_plot(png_path, summary)
    return [json_path, response_csv, units_csv, png_path]


def _wall_response_row(
    model: str,
    conductance_ratio: float,
    effective_conductance_ratio: float,
    interaction: float,
    pinhole_fraction: float,
) -> dict[str, float | str | bool]:
    closure_factor = _closure_factor(effective_conductance_ratio)
    drag_proxy = float(interaction) * closure_factor
    return {
        "wall_model": model,
        "conductance_ratio": float(conductance_ratio),
        "effective_conductance_ratio": float(effective_conductance_ratio),
        "pinhole_fraction": float(pinhole_fraction),
        "current_closure_proxy": closure_factor,
        "lorentz_drag_proxy": drag_proxy,
        "ideal_insulator_deviation_fraction": closure_factor,
        "mhd_performance_only": True,
    }


def _closure_factor(conductance_ratio: float) -> float:
    conductance = max(float(conductance_ratio), 0.0)
    return conductance / (1.0 + conductance)


def _wall_thresholds(rows: Iterable[dict[str, float | str | bool]]) -> dict[str, float | None]:
    sweep = [row for row in rows if row["wall_model"] == "pinhole_sweep"]
    thresholds: dict[str, float | None] = {}
    for tolerance in (0.05, 0.10, 0.25):
        accepted = [
            float(row["effective_conductance_ratio"])
            for row in sweep
            if float(row["ideal_insulator_deviation_fraction"]) <= tolerance
        ]
        thresholds[f"max_effective_conductance_ratio_for_{int(tolerance * 100)}pct_deviation"] = max(accepted) if accepted else None
        accepted_pinhole = [
            float(row["pinhole_fraction"])
            for row in sweep
            if float(row["ideal_insulator_deviation_fraction"]) <= tolerance
        ]
        thresholds[f"max_pinhole_fraction_for_{int(tolerance * 100)}pct_deviation"] = max(accepted_pinhole) if accepted_pinhole else None
    return thresholds


def _case_payload(case: WallStackStudyCase) -> dict[str, object]:
    payload = asdict(case)
    payload["lithium"]["kinematic_viscosity"] = case.lithium.kinematic_viscosity
    return payload


def _write_mapping_csv(path: Path, values: dict[str, object]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["quantity", "value"])
        for key, value in values.items():
            writer.writerow([key, value])


def _write_response_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    columns = [
        "wall_model",
        "conductance_ratio",
        "effective_conductance_ratio",
        "pinhole_fraction",
        "current_closure_proxy",
        "lorentz_drag_proxy",
        "ideal_insulator_deviation_fraction",
        "mhd_performance_only",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _write_li_aln_phase0_2_plot(path: Path, summary: dict[str, object]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    rows = [row for row in summary["response_rows"] if row["wall_model"] == "pinhole_sweep"]
    c_eff = np.asarray([float(row["effective_conductance_ratio"]) for row in rows], dtype=float)
    c_aln = np.asarray([float(row["conductance_ratio"]) for row in rows], dtype=float)
    pinhole = np.asarray([float(row["pinhole_fraction"]) for row in rows], dtype=float)
    closure = np.asarray([float(row["current_closure_proxy"]) for row in rows], dtype=float)
    drag = np.asarray([float(row["lorentz_drag_proxy"]) for row in rows], dtype=float)
    positive = c_eff > 0.0

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.6), constrained_layout=True)
    scatter = axes[0, 0].scatter(
        np.maximum(c_aln, 1.0e-12),
        closure,
        c=np.maximum(pinhole, 1.0e-12),
        cmap="viridis",
        s=58,
        edgecolor="#0f172a",
        linewidth=0.25,
    )
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_xlabel("AlN conductance ratio c_AlN")
    axes[0, 0].set_ylabel("current-closure proxy")
    axes[0, 0].set_title("Reduced pinhole/conductance sweep")
    axes[0, 0].grid(True, which="both", alpha=0.25)
    colorbar = fig.colorbar(scatter, ax=axes[0, 0])
    colorbar.set_label("pinhole fraction")

    axes[0, 1].loglog(c_eff[positive], drag[positive], marker="o", linestyle="", color="#b45309", alpha=0.75)
    axes[0, 1].set_xlabel("effective conductance ratio c_eff")
    axes[0, 1].set_ylabel("Lorentz-drag proxy")
    axes[0, 1].set_title("MHD penalty increases with c_eff")
    axes[0, 1].grid(True, which="both", alpha=0.25)

    labels = []
    values = []
    for model in ("ideal_insulator", "intact_aln", "degraded_aln", "bare_metal"):
        row = next(item for item in summary["response_rows"] if item["wall_model"] == model)
        labels.append(model.replace("_", "\n"))
        values.append(float(row["current_closure_proxy"]))
    axes[1, 0].bar(labels, values, color=["#2563eb", "#0891b2", "#f59e0b", "#991b1b"])
    axes[1, 0].set_ylabel("current-closure proxy")
    axes[1, 0].set_title("Baseline wall-model ranking")
    axes[1, 0].grid(True, axis="y", alpha=0.25)

    axes[1, 1].axis("off")
    unit_audit = summary["unit_audit"]
    stack = summary["wall_stack"]
    lines = [
        "Phase 0-2 status",
        f"Ha = {float(unit_audit['hartmann_number']):.3g}",
        f"Re = {float(unit_audit['reynolds_number']):.3g}",
        f"N = {float(unit_audit['interaction_parameter']):.3g}",
        f"Rm = {float(unit_audit['magnetic_reynolds_number']):.3g}",
        f"inductionless: {bool(unit_audit['inductionless_assumption_pass'])}",
        "",
        "Nested wall-layer QA",
        f"layers = {int(stack['mesh_resolution']['layer_count'])}",
        f"cells = {int(stack['mesh_resolution']['total_cells'])}",
        f"resolution pass = {bool(stack['mesh_resolution']['resolution_pass'])}",
        "",
        "Scope: MHD electrical performance only.",
    ]
    axes[1, 1].text(0.02, 0.98, "\n".join(lines), va="top", fontsize=10.5, transform=axes[1, 1].transAxes)
    fig.suptitle("Li/AlN wall-stack Phase 0-2 reduced study", fontsize=15.5, fontweight="bold")
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)

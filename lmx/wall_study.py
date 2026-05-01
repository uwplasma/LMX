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

import numpy as np

from .mesh import StructuredMesh, generate_multilayer_duct_mesh
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


DEFAULT_SUBSTRATE_CONDUCTIVITIES: dict[str, float] = {
    "316L": 1.35e6,
    "IN625": 0.80e6,
    "molybdenum": 1.87e7,
}


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


def li_aln_phase3_6_summary(
    case: WallStackStudyCase = DEFAULT_LI_ALN_CASE,
    *,
    magnetic_fields: Sequence[float] | None = None,
    velocities: Sequence[float] | None = None,
    substrate_conductivities: dict[str, float] | None = None,
    aln_conductivities: Sequence[float] | None = None,
    pinhole_fractions: Sequence[float] | None = None,
    tolerances: Sequence[float] = (0.05, 0.10, 0.25),
) -> dict[str, object]:
    """Return Phase 3--6 reduced Li/AlN threshold and substrate sweeps.

    This remains a reduced MHD electrical-performance model. It distinguishes
    tangential wall conductance from normal leakage so thickness guidance is not
    overinterpreted: increasing AlN thickness raises tangential conductance but
    reduces through-layer leakage.
    """

    b_values = tuple(magnetic_fields or (1.0, 2.0, 4.0))
    u_values = tuple(velocities or (0.01, 0.02, 0.04))
    sigma_values = tuple(aln_conductivities or (1.0e-10, 1.0e-9, 1.0e-8, 1.0e-7, 1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3))
    f_values = tuple(pinhole_fractions or (0.0, 1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1))
    substrates = dict(substrate_conductivities or DEFAULT_SUBSTRATE_CONDUCTIVITIES)
    unit_rows = _li_aln_operating_rows(case, magnetic_fields=b_values, velocities=u_values)
    threshold_rows = _li_aln_threshold_rows(case, substrates, tolerances=tolerances)
    substrate_rows = _li_aln_substrate_rows(
        case,
        substrates,
        aln_conductivities=sigma_values,
        pinhole_fractions=f_values,
        operating_rows=unit_rows,
    )
    return {
        "case": case.name,
        "scope": "reduced_mhd_electrical_performance_only",
        "material_compatibility_claim": False,
        "inputs": _case_payload(case),
        "phase_status": {
            "phase_3_case_matrix": "complete_for_reduced_B_U_sweep",
            "phase_4_parametric_sweeps": "complete_for_reduced_conductance_pinhole_substrate_model",
            "phase_5_degradation_thresholds": "complete_for_reduced_tangential_and_normal_thresholds",
            "phase_6_aln_metal_stack_comparison": "complete_for_effective_stack_model_true_multilayer_open",
            "true_multilayer_geometry": "planned_solver_extension",
        },
        "operating_rows": unit_rows,
        "threshold_rows": threshold_rows,
        "substrate_rows": substrate_rows,
        "substrate_conductivities": substrates,
        "notes": (
            "Thresholds use c/(1+c) as a reduced current-closure deviation. "
            "Tangential conductance gives a maximum allowable AlN thickness for a "
            "given conductivity, while normal leakage gives a minimum thickness. "
            "Both are reported because they are different electrical paths."
        ),
    }


def write_li_aln_phase3_6_artifacts(
    out_dir: str | Path,
    *,
    case: WallStackStudyCase = DEFAULT_LI_ALN_CASE,
    magnetic_fields: Sequence[float] | None = None,
    velocities: Sequence[float] | None = None,
    substrate_conductivities: dict[str, float] | None = None,
    aln_conductivities: Sequence[float] | None = None,
    pinhole_fractions: Sequence[float] | None = None,
    filename_stem: str = "li_aln_wall_stack_phase3_6",
) -> list[Path]:
    """Write Phase 3--6 Li/AlN wall-study JSON, CSV, and PNG artifacts."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = li_aln_phase3_6_summary(
        case,
        magnetic_fields=magnetic_fields,
        velocities=velocities,
        substrate_conductivities=substrate_conductivities,
        aln_conductivities=aln_conductivities,
        pinhole_fractions=pinhole_fractions,
    )
    json_path = out / f"{filename_stem}_summary.json"
    operating_csv = out / f"{filename_stem}_operating_matrix.csv"
    thresholds_csv = out / f"{filename_stem}_thresholds.csv"
    substrate_csv = out / f"{filename_stem}_substrates.csv"
    png_path = out / f"{filename_stem}.png"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_generic_rows_csv(operating_csv, summary["operating_rows"])
    _write_generic_rows_csv(thresholds_csv, summary["threshold_rows"])
    _write_generic_rows_csv(substrate_csv, summary["substrate_rows"])
    _write_li_aln_phase3_6_plot(png_path, summary)
    return [json_path, operating_csv, thresholds_csv, substrate_csv, png_path]


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


def li_aln_wall_stacks_by_side(
    case: WallStackStudyCase = DEFAULT_LI_ALN_CASE,
    *,
    aln_conductivity: float | None = None,
    metal_conductivity: float | None = None,
) -> dict[str, tuple[WallLayer, WallLayer]]:
    """Return fluid-outward AlN/metal layers on all four duct sides."""

    stack = (
        WallLayer(
            "aln",
            conductivity=float(case.intact_aln_conductivity if aln_conductivity is None else aln_conductivity),
            thickness=float(case.aln_thickness),
            cells=int(case.aln_cells),
        ),
        WallLayer(
            case.metal_name,
            conductivity=float(case.metal_conductivity if metal_conductivity is None else metal_conductivity),
            thickness=float(case.metal_thickness),
            cells=int(case.metal_cells),
        ),
    )
    return {side: stack for side in ("left", "right", "bottom", "top")}


def li_aln_multilayer_mesh_summary(
    case: WallStackStudyCase = DEFAULT_LI_ALN_CASE,
    *,
    ny: int = 48,
    nz: int = 48,
    wall_layers: dict[str, Sequence[WallLayer]] | None = None,
    minimum_cells_per_layer: int = 3,
) -> dict[str, object]:
    """Return a true ``fluid | AlN | metal`` rectangular mesh QA summary."""

    stacks = {side: tuple(layers) for side, layers in (wall_layers or li_aln_wall_stacks_by_side(case)).items()}
    mesh = generate_multilayer_duct_mesh(
        width=2.0 * case.length_scale,
        height=2.0 * case.length_scale,
        length=case.length_scale,
        nx=1,
        ny=ny,
        nz=nz,
        wall_layers=stacks,
        fluid_conductivity=case.lithium.electrical_conductivity,
    )
    layer_rows = _multilayer_layer_rows(stacks)
    interface_rows = _multilayer_interface_rows(case, stacks, mesh)
    side_rows = _multilayer_side_rows(case, stacks)
    region_rows = _multilayer_region_rows(mesh)
    interface_aligned = all(bool(row["face_aligned"]) for row in interface_rows)
    cell_count_pass = all(int(row["cells"]) >= minimum_cells_per_layer for row in layer_rows)
    return {
        "case": f"{case.name}_multilayer_mesh",
        "scope": "explicit_multilayer_geometry_qa_only",
        "material_compatibility_claim": False,
        "inputs": _case_payload(case),
        "mesh": {
            "geometry": mesh.geometry,
            "ny": mesh.ny,
            "nz": mesh.nz,
            "fluid_ny": int(ny),
            "fluid_nz": int(nz),
            "region_count": len(mesh.region_names),
            "fluid_cell_count": int(region_rows[0]["cell_count"]),
            "solid_cell_count": int(mesh.ny * mesh.nz - int(region_rows[0]["cell_count"])),
            "minimum_dy": float(np.min(np.asarray(mesh.dy))),
            "minimum_dz": float(np.min(np.asarray(mesh.dz))),
            "maximum_dy": float(np.max(np.asarray(mesh.dy))),
            "maximum_dz": float(np.max(np.asarray(mesh.dz))),
        },
        "wall_stack": {
            "sides": side_rows,
            "layers": layer_rows,
            "interfaces": interface_rows,
            "regions": region_rows,
        },
        "qa": {
            "minimum_required_cells_per_layer": int(minimum_cells_per_layer),
            "cell_count_pass": bool(cell_count_pass),
            "interface_faces_aligned": bool(interface_aligned),
            "explicit_conductivity_field": bool(mesh.sigma is not None),
            "explicit_region_ids": bool(mesh.region_ids is not None),
            "ready_for_conservative_current_diagnostics": bool(cell_count_pass and interface_aligned and mesh.sigma is not None),
        },
        "phase_status": {
            "true_fluid_aln_metal_geometry": "complete_for_rectangular_mesh_qa",
            "interface_current_diagnostics": "available_when_mesh_is_used_by_solver",
            "freemhd_limiting_case_comparison": "next",
        },
    }


def write_li_aln_multilayer_mesh_artifacts(
    out_dir: str | Path,
    *,
    case: WallStackStudyCase = DEFAULT_LI_ALN_CASE,
    ny: int = 48,
    nz: int = 48,
    wall_layers: dict[str, Sequence[WallLayer]] | None = None,
    filename_stem: str = "li_aln_multilayer_mesh_qa",
) -> list[Path]:
    """Write JSON, CSV, and PNG artifacts for the Li/AlN multilayer mesh QA."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stacks = {side: tuple(layers) for side, layers in (wall_layers or li_aln_wall_stacks_by_side(case)).items()}
    summary = li_aln_multilayer_mesh_summary(case, ny=ny, nz=nz, wall_layers=stacks)
    mesh = generate_multilayer_duct_mesh(
        width=2.0 * case.length_scale,
        height=2.0 * case.length_scale,
        length=case.length_scale,
        nx=1,
        ny=ny,
        nz=nz,
        wall_layers=stacks,
        fluid_conductivity=case.lithium.electrical_conductivity,
    )
    json_path = out / f"{filename_stem}_summary.json"
    layer_csv = out / f"{filename_stem}_layers.csv"
    interface_csv = out / f"{filename_stem}_interfaces.csv"
    region_csv = out / f"{filename_stem}_regions.csv"
    png_path = out / f"{filename_stem}.png"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_generic_rows_csv(layer_csv, summary["wall_stack"]["layers"])
    _write_generic_rows_csv(interface_csv, summary["wall_stack"]["interfaces"])
    _write_generic_rows_csv(region_csv, summary["wall_stack"]["regions"])
    _write_li_aln_multilayer_mesh_plot(png_path, mesh, summary)
    return [json_path, layer_csv, interface_csv, region_csv, png_path]


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


def _li_aln_operating_rows(
    case: WallStackStudyCase,
    *,
    magnetic_fields: Sequence[float],
    velocities: Sequence[float],
) -> list[dict[str, float | bool]]:
    rows: list[dict[str, float | bool]] = []
    for magnetic_field in magnetic_fields:
        for velocity in velocities:
            audit = li_aln_unit_audit(
                WallStackStudyCase(
                    **{
                        **asdict(case),
                        "lithium": case.lithium,
                        "magnetic_field": float(magnetic_field),
                        "velocity": float(velocity),
                    }
                )
            )
            rows.append(
                {
                    "magnetic_field_t": float(magnetic_field),
                    "velocity_m_s": float(velocity),
                    "hartmann_number": float(audit["hartmann_number"]),
                    "reynolds_number": float(audit["reynolds_number"]),
                    "interaction_parameter": float(audit["interaction_parameter"]),
                    "magnetic_reynolds_number": float(audit["magnetic_reynolds_number"]),
                    "inductionless_assumption_pass": bool(audit["inductionless_assumption_pass"]),
                }
            )
    return rows


def _li_aln_threshold_rows(
    case: WallStackStudyCase,
    substrate_conductivities: dict[str, float],
    *,
    tolerances: Sequence[float],
) -> list[dict[str, float | str | bool | None]]:
    rows: list[dict[str, float | str | bool | None]] = []
    sigma_li = case.lithium.electrical_conductivity
    length = case.length_scale
    c_aln_intact = wall_conductance_ratio(
        wall_conductivity=case.intact_aln_conductivity,
        wall_thickness=case.aln_thickness,
        fluid_conductivity=sigma_li,
        length_scale=length,
    )
    for tolerance in tolerances:
        c_crit = _conductance_for_deviation(float(tolerance))
        sigma_tangential_crit = c_crit * sigma_li * length / case.aln_thickness
        g_perp_crit = c_crit
        t_min_normal = case.intact_aln_conductivity * length / (sigma_li * g_perp_crit)
        t_max_tangential = c_crit * sigma_li * length / case.intact_aln_conductivity
        for metal_name, metal_sigma in substrate_conductivities.items():
            metal_c = wall_conductance_ratio(
                wall_conductivity=metal_sigma,
                wall_thickness=case.metal_thickness,
                fluid_conductivity=sigma_li,
                length_scale=length,
            )
            f_p_max = _pinhole_fraction_for_threshold(
                intact_conductance_ratio=c_aln_intact,
                metal_conductance_ratio=metal_c,
                critical_conductance_ratio=c_crit,
            )
            rows.append(
                {
                    "tolerance_fraction": float(tolerance),
                    "substrate": metal_name,
                    "substrate_conductivity_s_m": float(metal_sigma),
                    "critical_effective_conductance_ratio": c_crit,
                    "critical_aln_conductivity_for_current_thickness_s_m": sigma_tangential_crit,
                    "normal_leakage_threshold": g_perp_crit,
                    "minimum_aln_thickness_for_normal_leakage_m": t_min_normal,
                    "maximum_aln_thickness_for_tangential_conductance_m": t_max_tangential,
                    "maximum_pinhole_fraction": f_p_max,
                    "intact_aln_passes_tangential_threshold": bool(c_aln_intact <= c_crit),
                    "mhd_performance_only": True,
                }
            )
    return rows


def _li_aln_substrate_rows(
    case: WallStackStudyCase,
    substrate_conductivities: dict[str, float],
    *,
    aln_conductivities: Sequence[float],
    pinhole_fractions: Sequence[float],
    operating_rows: Sequence[dict[str, float | bool]],
) -> list[dict[str, float | str | bool]]:
    rows: list[dict[str, float | str | bool]] = []
    sigma_li = case.lithium.electrical_conductivity
    interaction_values = [float(row["interaction_parameter"]) for row in operating_rows]
    interaction_ref = max(interaction_values) if interaction_values else li_aln_unit_audit(case)["interaction_parameter"]
    for metal_name, metal_sigma in substrate_conductivities.items():
        metal_c = wall_conductance_ratio(
            wall_conductivity=metal_sigma,
            wall_thickness=case.metal_thickness,
            fluid_conductivity=sigma_li,
            length_scale=case.length_scale,
        )
        for aln_sigma in aln_conductivities:
            c_aln = wall_conductance_ratio(
                wall_conductivity=float(aln_sigma),
                wall_thickness=case.aln_thickness,
                fluid_conductivity=sigma_li,
                length_scale=case.length_scale,
            )
            for f_p in pinhole_fractions:
                c_eff = effective_pinhole_conductance_ratio(
                    intact_conductance_ratio=c_aln,
                    metal_conductance_ratio=metal_c,
                    pinhole_fraction=float(f_p),
                )
                closure = _closure_factor(c_eff)
                rows.append(
                    {
                        "substrate": metal_name,
                        "substrate_conductivity_s_m": float(metal_sigma),
                        "aln_conductivity_s_m": float(aln_sigma),
                        "aln_conductance_ratio": c_aln,
                        "pinhole_fraction": float(f_p),
                        "metal_conductance_ratio": metal_c,
                        "effective_conductance_ratio": c_eff,
                        "current_closure_proxy": closure,
                        "worst_case_lorentz_drag_proxy": float(interaction_ref) * closure,
                        "mhd_performance_only": True,
                    }
                )
    return rows


def _conductance_for_deviation(tolerance: float) -> float:
    if not 0.0 < tolerance < 1.0:
        raise ValueError("deviation tolerance must lie between 0 and 1")
    return tolerance / (1.0 - tolerance)


def _pinhole_fraction_for_threshold(
    *,
    intact_conductance_ratio: float,
    metal_conductance_ratio: float,
    critical_conductance_ratio: float,
) -> float | None:
    if intact_conductance_ratio > critical_conductance_ratio:
        return None
    if metal_conductance_ratio <= intact_conductance_ratio:
        return 1.0
    return max(
        0.0,
        min(
            1.0,
            (critical_conductance_ratio - intact_conductance_ratio)
            / (metal_conductance_ratio - intact_conductance_ratio),
        ),
    )


def _write_generic_rows_csv(path: Path, rows: Sequence[object]) -> None:
    payload = [dict(row) for row in rows]
    columns: list[str] = []
    for row in payload:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in payload:
            writer.writerow({column: row.get(column, "") for column in columns})


def _multilayer_layer_rows(wall_layers: dict[str, Sequence[WallLayer]]) -> list[dict[str, float | int | str | bool]]:
    rows: list[dict[str, float | int | str | bool]] = []
    for side, layers in wall_layers.items():
        distance = 0.0
        for index, layer in enumerate(layers):
            rows.append(
                {
                    "side": side,
                    "layer_index_fluid_outward": int(index),
                    "name": layer.name,
                    "conductivity_s_m": float(layer.conductivity),
                    "thickness_m": float(layer.thickness),
                    "cells": int(layer.cells),
                    "normal_cell_width_m": float(layer.thickness) / int(layer.cells),
                    "inner_distance_from_fluid_m": distance,
                    "outer_distance_from_fluid_m": distance + float(layer.thickness),
                    "mhd_performance_only": True,
                }
            )
            distance += float(layer.thickness)
    return rows


def _multilayer_side_rows(
    case: WallStackStudyCase,
    wall_layers: dict[str, Sequence[WallLayer]],
) -> list[dict[str, float | str | int | bool]]:
    rows: list[dict[str, float | str | int | bool]] = []
    for side, layers in wall_layers.items():
        rows.append(
            {
                "side": side,
                "layer_count": len(layers),
                "total_thickness_m": sum(float(layer.thickness) for layer in layers),
                "total_cells": sum(int(layer.cells) for layer in layers),
                "tangential_conductance_ratio": tangential_stack_conductance_ratio(
                    layers,
                    fluid_conductivity=case.lithium.electrical_conductivity,
                    length_scale=case.length_scale,
                ),
                "normal_leakage_ratio": normal_stack_leakage_ratio(
                    layers,
                    fluid_conductivity=case.lithium.electrical_conductivity,
                    length_scale=case.length_scale,
                ),
                "mhd_performance_only": True,
            }
        )
    return rows


def _multilayer_interface_rows(
    case: WallStackStudyCase,
    wall_layers: dict[str, Sequence[WallLayer]],
    mesh: StructuredMesh,
) -> list[dict[str, float | str | bool]]:
    rows: list[dict[str, float | str | bool]] = []
    y_faces = np.asarray(mesh.y_faces, dtype=float)
    z_faces = np.asarray(mesh.z_faces, dtype=float)
    half_width = case.length_scale
    half_height = case.length_scale
    for side, layers in wall_layers.items():
        distance = 0.0
        axis = "y" if side in {"left", "right"} else "z"
        faces = y_faces if axis == "y" else z_faces
        for index, layer in enumerate(layers):
            if side == "left":
                coordinate = -half_width - distance
            elif side == "right":
                coordinate = half_width + distance
            elif side == "bottom":
                coordinate = -half_height - distance
            else:
                coordinate = half_height + distance
            rows.append(
                {
                    "side": side,
                    "axis": axis,
                    "coordinate_m": coordinate,
                    "distance_from_fluid_m": distance,
                    "inner_region": "fluid" if index == 0 else layers[index - 1].name,
                    "outer_region": layer.name,
                    "face_aligned": _face_is_aligned(coordinate, faces),
                    "mhd_performance_only": True,
                }
            )
            distance += float(layer.thickness)
    return rows


def _face_is_aligned(value: float, faces: np.ndarray, *, tolerance: float = 1.0e-8) -> bool:
    return bool(np.min(np.abs(faces - float(value))) <= tolerance * max(1.0, abs(float(value))))


def _multilayer_region_rows(mesh: StructuredMesh) -> list[dict[str, float | int | str | bool]]:
    if mesh.region_ids is None or mesh.sigma is None:
        return []
    region_ids = np.asarray(mesh.region_ids, dtype=int)
    sigma = np.asarray(mesh.sigma, dtype=float)
    rows: list[dict[str, float | int | str | bool]] = []
    for region_id, name in enumerate(mesh.region_names):
        mask = region_ids == region_id
        values = sigma[mask]
        rows.append(
            {
                "region_id": int(region_id),
                "name": name,
                "cell_count": int(np.sum(mask)),
                "conductivity_s_m": float(values[0]) if values.size else 0.0,
                "is_fluid": bool(region_id == 0),
            }
        )
    return rows


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


def _write_li_aln_multilayer_mesh_plot(path: Path, mesh: StructuredMesh, summary: dict[str, object]) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Rectangle
    import numpy as local_np

    y_faces = local_np.asarray(mesh.y_faces, dtype=float)
    z_faces = local_np.asarray(mesh.z_faces, dtype=float)
    region_ids = local_np.asarray(mesh.region_ids, dtype=int)
    sigma = local_np.asarray(mesh.sigma, dtype=float)
    inputs = dict(summary["inputs"])
    half_width = float(inputs["length_scale"])
    half_height = float(inputs["length_scale"])

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.0), constrained_layout=True)
    region_plot = axes[0, 0].pcolormesh(y_faces, z_faces, region_ids.T, shading="flat", cmap="tab20")
    axes[0, 0].add_patch(
        Rectangle(
            (-half_width, -half_height),
            2.0 * half_width,
            2.0 * half_height,
            fill=False,
            edgecolor="black",
            linewidth=1.2,
        )
    )
    axes[0, 0].set_aspect("equal")
    axes[0, 0].set_xlabel("y [m]")
    axes[0, 0].set_ylabel("z [m]")
    axes[0, 0].set_title("Explicit fluid | AlN | metal regions")
    cbar = fig.colorbar(region_plot, ax=axes[0, 0])
    cbar.set_label("region id")

    sigma_plot = axes[0, 1].pcolormesh(
        y_faces,
        z_faces,
        local_np.maximum(sigma.T, 1.0e-30),
        shading="flat",
        cmap="viridis",
        norm=LogNorm(),
    )
    axes[0, 1].add_patch(
        Rectangle(
            (-half_width, -half_height),
            2.0 * half_width,
            2.0 * half_height,
            fill=False,
            edgecolor="white",
            linewidth=1.0,
        )
    )
    top_wall = max(float(row["total_thickness_m"]) for row in summary["wall_stack"]["sides"])
    axes[0, 1].set_xlim(-half_width, half_width)
    axes[0, 1].set_ylim(half_height - 2.0 * float(inputs["aln_thickness"]), half_height + 1.05 * top_wall)
    axes[0, 1].set_xlabel("y [m]")
    axes[0, 1].set_ylabel("z [m]")
    axes[0, 1].set_title("Electrical conductivity field, top-wall zoom")
    cbar = fig.colorbar(sigma_plot, ax=axes[0, 1])
    cbar.set_label("sigma [S/m]")

    side_rows = [dict(row) for row in summary["wall_stack"]["sides"]]
    layer_rows = [dict(row) for row in summary["wall_stack"]["layers"]]
    sides = [str(row["side"]) for row in side_rows]
    left_offsets = {side: 0.0 for side in sides}
    colors = {"aln": "#2563eb", "316L": "#9ca3af", "IN625": "#6b7280", "molybdenum": "#52525b"}
    for row in layer_rows:
        side = str(row["side"])
        thickness = float(row["thickness_m"])
        name = str(row["name"])
        axes[1, 0].barh(
            side,
            thickness,
            left=left_offsets[side],
            color=colors.get(name, "#64748b"),
            edgecolor="#0f172a",
            label=name if name not in axes[1, 0].get_legend_handles_labels()[1] else None,
        )
        left_offsets[side] += thickness
    axes[1, 0].set_xlabel("distance from fluid boundary [m]")
    axes[1, 0].set_title("Wall stack by side")
    axes[1, 0].grid(True, axis="x", alpha=0.25)
    axes[1, 0].legend(frameon=False, loc="lower right")

    axes[1, 1].axis("off")
    qa = summary["qa"]
    mesh_summary = summary["mesh"]
    text = [
        "Mesh QA",
        f"ny x nz = {int(mesh_summary['ny'])} x {int(mesh_summary['nz'])}",
        f"fluid cells = {int(mesh_summary['fluid_cell_count'])}",
        f"solid cells = {int(mesh_summary['solid_cell_count'])}",
        f"min dy = {float(mesh_summary['minimum_dy']):.3e} m",
        f"min dz = {float(mesh_summary['minimum_dz']):.3e} m",
        "",
        f"cell-count pass = {bool(qa['cell_count_pass'])}",
        f"interfaces aligned = {bool(qa['interface_faces_aligned'])}",
        f"explicit sigma = {bool(qa['explicit_conductivity_field'])}",
        f"ready for current diagnostics = {bool(qa['ready_for_conservative_current_diagnostics'])}",
        "",
        "Scope: geometry and MHD electrical performance only.",
    ]
    axes[1, 1].text(0.02, 0.98, "\n".join(text), va="top", fontsize=11.0, transform=axes[1, 1].transAxes)
    fig.suptitle("Li/AlN explicit multilayer wall-stack mesh QA", fontsize=15.5, fontweight="bold")
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def _write_li_aln_phase3_6_plot(path: Path, summary: dict[str, object]) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import numpy as np

    operating = [dict(row) for row in summary["operating_rows"]]
    thresholds = [dict(row) for row in summary["threshold_rows"]]
    substrate_rows = [dict(row) for row in summary["substrate_rows"]]
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.2), constrained_layout=True)

    b_values = sorted({float(row["magnetic_field_t"]) for row in operating})
    u_values = sorted({float(row["velocity_m_s"]) for row in operating})
    n_grid = np.full((len(b_values), len(u_values)), np.nan, dtype=float)
    for row in operating:
        i = b_values.index(float(row["magnetic_field_t"]))
        j = u_values.index(float(row["velocity_m_s"]))
        n_grid[i, j] = float(row["interaction_parameter"])
    im = axes[0, 0].imshow(n_grid, origin="lower", aspect="auto", cmap="magma", norm=LogNorm())
    axes[0, 0].set_xticks(np.arange(len(u_values)), [f"{value:.2g}" for value in u_values])
    axes[0, 0].set_yticks(np.arange(len(b_values)), [f"{value:.2g}" for value in b_values])
    axes[0, 0].set_xlabel("velocity [m/s]")
    axes[0, 0].set_ylabel("B [T]")
    axes[0, 0].set_title("Interaction-parameter matrix")
    for i in range(n_grid.shape[0]):
        for j in range(n_grid.shape[1]):
            color = "white" if n_grid[i, j] < np.nanmax(n_grid) / 4.0 else "#111827"
            axes[0, 0].text(j, i, f"{n_grid[i, j]:.2g}", ha="center", va="center", color=color, fontsize=8)
    cbar = fig.colorbar(im, ax=axes[0, 0])
    cbar.set_label("N")

    ten_pct = [row for row in thresholds if abs(float(row["tolerance_fraction"]) - 0.10) < 1.0e-12]
    substrates = [str(row["substrate"]) for row in ten_pct]
    fp = [float(row["maximum_pinhole_fraction"]) if row["maximum_pinhole_fraction"] is not None else 0.0 for row in ten_pct]
    axes[0, 1].bar(substrates, fp, color="#0f766e")
    axes[0, 1].set_ylim(0.0, 1.05)
    axes[0, 1].set_ylabel("max pinhole fraction")
    axes[0, 1].set_title("10% deviation pinhole limit by substrate")
    for index, value in enumerate(fp):
        axes[0, 1].text(index, min(value + 0.025, 1.02), f"{value:.2g}", ha="center", va="bottom", fontsize=9)
    axes[0, 1].grid(True, axis="y", alpha=0.25)

    grouped: dict[str, list[dict[str, object]]] = {}
    for row in substrate_rows:
        if float(row["pinhole_fraction"]) in {0.0, 1.0e-3, 1.0e-2} and str(row["substrate"]) == "316L":
            grouped.setdefault(f"f={float(row['pinhole_fraction']):.0e}", []).append(row)
    for label in ("f=0e+00", "f=1e-03", "f=1e-02"):
        rows = grouped.get(label, [])
        if not rows:
            continue
        ordered = sorted(rows, key=lambda item: float(item["aln_conductivity_s_m"]))
        axes[1, 0].loglog(
            [float(row["aln_conductivity_s_m"]) for row in ordered],
            [float(row["current_closure_proxy"]) for row in ordered],
            marker="o",
            linewidth=1.6,
            label=label,
        )
    axes[1, 0].set_xlabel("AlN conductivity [S/m]")
    axes[1, 0].set_ylabel("current-closure proxy")
    axes[1, 0].set_title("Degradation sweep for 316L substrate")
    axes[1, 0].grid(True, which="both", alpha=0.25)
    axes[1, 0].legend(frameon=False)

    case_payload = dict(summary["inputs"])
    sigma_li = float(case_payload["lithium"]["electrical_conductivity"])
    length = float(case_payload["length_scale"])
    t_current = float(case_payload["aln_thickness"])
    ccrit = float(ten_pct[0]["critical_effective_conductance_ratio"]) if ten_pct else _conductance_for_deviation(0.10)
    sigma = np.logspace(-10, -3, 120)
    tangential_c = sigma * t_current / (sigma_li * length)
    normal_g = sigma * length / (sigma_li * t_current)
    axes[1, 1].loglog(
        sigma,
        tangential_c / ccrit,
        color="#b45309",
        linewidth=2.0,
        label="tangential c / c_crit",
    )
    axes[1, 1].loglog(
        sigma,
        normal_g / ccrit,
        color="#2563eb",
        linewidth=2.0,
        label="normal leakage / g_crit",
    )
    axes[1, 1].axhline(1.0, color="black", linestyle="--", linewidth=1.0, label="10% threshold")
    axes[1, 1].set_xlabel("AlN conductivity [S/m]")
    axes[1, 1].set_ylabel("margin at current thickness")
    axes[1, 1].set_title("10% electrical margins at t_AlN = 200 microns")
    axes[1, 1].grid(True, which="both", alpha=0.25)
    axes[1, 1].legend(frameon=False, fontsize=8.5)

    fig.suptitle("Li/AlN wall-stack Phase 3-6 reduced parametric assessment", fontsize=15.5, fontweight="bold")
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)

"""Differentiable WHAM blanket pressure-drop design study.

This example keeps the WHAM blanket route and reduced fixed-flow pressure
budget explicit. It answers two research-design questions with autodiff:

1. How does the pressure drop respond to mirror-coil separation?
2. What magnetic-field multiplier reaches a target pressure-drop constraint?
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import jax
import jax.numpy as jnp
import numpy as np

from lmx.blanket_flow import (
    BlanketFlowSettings,
    LiquidMetalProperties,
    blanket_pressure_budget_from_transverse_field,
    wham_blanket_pressure_drop_history,
    wham_blanket_pressure_drop_sensitivity,
    write_wham_blanket_autodiff_research_plots,
)
from lmx.blanket_geometry import WhamBlanketLoop, build_wham_blanket_centerline
from lmx.field_models import load_wham_coil_model_script


OUTPUT_DIR = Path("artifacts/examples/wham_blanket_autodiff_research")
DOCS_OUTPUT_DIR = Path("docs/_static/generated")
WHAM_COIL_MODEL_SCRIPT = Path("/Users/rogerio/Downloads/coil_model_WHAM-1.txt")

# Blanket route and mesh-free centerline sampling.
PIPE_RADIUS = 0.12
BEND_RADIUS = 0.90
ENTRY_LENGTH = 1.35
CENTRAL_CELL_RADIUS = 0.42
STRAIGHT_POINTS = 72
BEND_POINTS = 144

# PbLi-like fixed-flow operating point.
MEAN_VELOCITY = 0.20
DENSITY = 9300.0
DYNAMIC_VISCOSITY = 1.8e-3
ELECTRICAL_CONDUCTIVITY = 7.9e5

# Same design-field multiplier and loss factors as the flow-movie example.
FIELD_SCALE = 8.0
MHD_DRAG_FACTOR = 0.35
BEND_LOSS_COEFFICIENT = 0.35

# Autodiff study controls.
REFERENCE_COIL_SEPARATION = 1.96
SEPARATION_SWEEP = np.linspace(1.50, 2.30, 9)
TARGET_PRESSURE_DROP_KPA = 20.0
INVERSE_DESIGN_STEPS = 7
NEWTON_RELAXATION = 0.7


def _coil_parameters() -> dict[str, float | int]:
    if not WHAM_COIL_MODEL_SCRIPT.exists():
        return {
            "coil_separation": REFERENCE_COIL_SEPARATION,
            "inner_radius": 0.043,
            "outer_radius": 0.365,
            "coil_axial_thickness": 0.1144,
            "radial_loops": 12,
            "axial_loops": 4,
            "current_scale": 100323.62459546926,
        }
    parsed = load_wham_coil_model_script(WHAM_COIL_MODEL_SCRIPT, radial_loops=12, axial_loops=4)
    return {
        "coil_separation": float(parsed["coil_separation"]),
        "inner_radius": float(parsed["inner_radius"]),
        "outer_radius": float(parsed["outer_radius"]),
        "coil_axial_thickness": float(parsed["coil_axial_thickness"]),
        "radial_loops": int(parsed["radial_loops"]),
        "axial_loops": int(parsed["axial_loops"]),
        "current_scale": float(parsed["current_scale"]),
    }


def _field_scale_inverse_design(
    reference: dict[str, jnp.ndarray],
    *,
    geometry: WhamBlanketLoop,
    properties: LiquidMetalProperties,
    settings: BlanketFlowSettings,
) -> list[dict[str, float]]:
    station = reference["station"]
    base_b_perp = reference["b_perp"] / jnp.maximum(reference["field_scale"], 1.0e-12)
    curvature = reference["curvature"]
    target = jnp.asarray(TARGET_PRESSURE_DROP_KPA * 1000.0, dtype=jnp.float32)

    def pressure_for_scale(scale):
        budget = blanket_pressure_budget_from_transverse_field(
            station,
            scale * base_b_perp,
            curvature,
            pipe_radius=geometry.pipe_radius,
            mean_velocity=settings.mean_velocity,
            density=properties.density,
            dynamic_viscosity=properties.dynamic_viscosity,
            electrical_conductivity=properties.electrical_conductivity,
            mhd_drag_factor=settings.mhd_drag_factor,
            bend_loss_coefficient=settings.bend_loss_coefficient,
        )
        return budget["pressure_drop"]

    history = []
    scale = jnp.asarray(FIELD_SCALE, dtype=jnp.float32)
    for step in range(INVERSE_DESIGN_STEPS):
        pressure, gradient = jax.value_and_grad(pressure_for_scale)(scale)
        history.append(
            {
                "step": step,
                "field_scale": float(scale),
                "pressure_drop_kpa": float(pressure / 1000.0),
                "d_pressure_drop_d_field_scale_kpa": float(gradient / 1000.0),
            }
        )
        scale = jnp.clip(
            scale - NEWTON_RELAXATION * (pressure - target) / jnp.maximum(jnp.abs(gradient), 1.0e-12),
            0.25,
            16.0,
        )
    pressure, gradient = jax.value_and_grad(pressure_for_scale)(scale)
    history.append(
        {
            "step": INVERSE_DESIGN_STEPS,
            "field_scale": float(scale),
            "pressure_drop_kpa": float(pressure / 1000.0),
            "d_pressure_drop_d_field_scale_kpa": float(gradient / 1000.0),
        }
    )
    return history


def run_wham_blanket_autodiff_research_demo() -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    coil_parameters = _coil_parameters()
    geometry = WhamBlanketLoop(
        pipe_radius=PIPE_RADIUS,
        bend_radius=BEND_RADIUS,
        entry_length=ENTRY_LENGTH,
        central_cell_radius=CENTRAL_CELL_RADIUS,
        coil_separation=float(coil_parameters["coil_separation"]),
        coil_inner_radius=float(coil_parameters["inner_radius"]),
        coil_outer_radius=float(coil_parameters["outer_radius"]),
        coil_axial_thickness=float(coil_parameters["coil_axial_thickness"]),
    )
    centerline = build_wham_blanket_centerline(
        geometry,
        straight_points=STRAIGHT_POINTS,
        bend_points=BEND_POINTS,
    )
    properties = LiquidMetalProperties(
        density=DENSITY,
        dynamic_viscosity=DYNAMIC_VISCOSITY,
        electrical_conductivity=ELECTRICAL_CONDUCTIVITY,
    )
    settings = BlanketFlowSettings(
        mean_velocity=MEAN_VELOCITY,
        field_scale=FIELD_SCALE,
        mhd_drag_factor=MHD_DRAG_FACTOR,
        bend_loss_coefficient=BEND_LOSS_COEFFICIENT,
        radial_loops=int(coil_parameters["radial_loops"]),
        axial_loops=int(coil_parameters["axial_loops"]),
    )

    reference = wham_blanket_pressure_drop_sensitivity(
        centerline,
        geometry=geometry,
        properties=properties,
        settings=settings,
        coil_parameters=coil_parameters,
        coil_separation=REFERENCE_COIL_SEPARATION,
        field_scale=FIELD_SCALE,
        mean_velocity=MEAN_VELOCITY,
    )
    separation_pressure = [
        float(
            wham_blanket_pressure_drop_history(
                centerline,
                geometry=geometry,
                properties=properties,
                settings=settings,
                coil_parameters=coil_parameters,
                coil_separation=float(separation),
                field_scale=FIELD_SCALE,
                mean_velocity=MEAN_VELOCITY,
            )["pressure_drop"]
            / 1000.0
        )
        for separation in SEPARATION_SWEEP
    ]
    field_history = _field_scale_inverse_design(
        reference,
        geometry=geometry,
        properties=properties,
        settings=settings,
    )
    study = {
        "case": "wham_blanket_autodiff_research",
        "research_questions": [
            "local pressure-drop sensitivity with respect to mirror-coil separation",
            "field multiplier required to hit a pressure-drop constraint at fixed flow rate",
        ],
        "reference": reference,
        "separation_sweep": SEPARATION_SWEEP.tolist(),
        "separation_pressure_drop_kpa": separation_pressure,
        "target_pressure_drop_kpa": TARGET_PRESSURE_DROP_KPA,
        "field_scale_design_history": field_history,
    }
    outputs = write_wham_blanket_autodiff_research_plots(study, OUTPUT_DIR)

    copied = []
    for output in outputs:
        if output.suffix.lower() in {".png", ".json", ".csv"}:
            target = DOCS_OUTPUT_DIR / output.name
            shutil.copy2(output, target)
            copied.append(target.name)

    summary = {
        "case": "wham_blanket_autodiff_research",
        "docs_artifacts": copied,
        "pressure_drop_kpa": float(reference["pressure_drop"] / 1000.0),
        "d_pressure_drop_d_coil_separation_kpa_per_m": float(reference["d_pressure_drop_d_coil_separation"] / 1000.0),
        "d_pressure_drop_d_field_scale_kpa": float(reference["d_pressure_drop_d_field_scale"] / 1000.0),
        "d_pressure_drop_d_mean_velocity_kpa_per_m_per_s": float(reference["d_pressure_drop_d_mean_velocity"] / 1000.0),
        "elasticity_coil_separation": float(reference["elasticity_coil_separation"]),
        "elasticity_field_scale": float(reference["elasticity_field_scale"]),
        "elasticity_mean_velocity": float(reference["elasticity_mean_velocity"]),
        "target_pressure_drop_kpa": TARGET_PRESSURE_DROP_KPA,
        "target_field_scale": float(field_history[-1]["field_scale"]),
        "target_pressure_drop_error_kpa": float(field_history[-1]["pressure_drop_kpa"] - TARGET_PRESSURE_DROP_KPA),
        "model_limitations": (
            "Differentiable reduced fixed-flow-rate pressure-budget model; it is a "
            "research-design gate for field/geometry sensitivity before the full "
            "curved-pipe pressure-velocity solver is promoted."
        ),
    }
    summary_path = OUTPUT_DIR / "wham_blanket_autodiff_research_demo_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(summary_path, DOCS_OUTPUT_DIR / summary_path.name)
    print(f"WHAM blanket autodiff artifacts written to {OUTPUT_DIR}")
    print(f"pressure_drop = {summary['pressure_drop_kpa']:.3f} kPa")
    print(f"d(pressure_drop)/d(coil_separation) = {summary['d_pressure_drop_d_coil_separation_kpa_per_m']:.3f} kPa/m")
    print(f"target field scale = {summary['target_field_scale']:.3f}")
    return summary


if __name__ == "__main__":
    run_wham_blanket_autodiff_research_demo()

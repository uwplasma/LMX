"""Run the first reduced liquid-metal flow preview on the WHAM blanket route."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
import shutil

from lmx.blanket_flow import (
    BlanketFlowSettings,
    BlanketTransientFlowSettings,
    LiquidMetalProperties,
    solve_wham_blanket_reduced_flow,
    solve_wham_blanket_transient_flow,
    write_wham_blanket_flow_plots,
    write_wham_blanket_pressure_sweep_plots,
    write_wham_blanket_transient_flow_movie,
    write_wham_blanket_transient_flow_plots,
)
from lmx.blanket_geometry import WhamBlanketLoop, build_wham_blanket_centerline
from lmx.field_models import load_wham_coil_model_script


OUTPUT_DIR = Path("artifacts/examples/wham_blanket_flow")
DOCS_OUTPUT_DIR = Path("docs/_static/generated")
WHAM_COIL_MODEL_SCRIPT = Path("/Users/rogerio/Downloads/coil_model_WHAM-1.txt")

# Approved geometry preview parameters.
PIPE_RADIUS = 0.12
BEND_RADIUS = 0.90
ENTRY_LENGTH = 1.35
CENTRAL_CELL_RADIUS = 0.42

# PbLi-like operating point for a first blanket pump-loop estimate.
MEAN_VELOCITY = 0.20
DENSITY = 9300.0
DYNAMIC_VISCOSITY = 1.8e-3
ELECTRICAL_CONDUCTIVITY = 7.9e5

# The parsed WHAM coil script preserves source ampere-turns on a reduced loop
# representation. FIELD_SCALE is an explicit design-field multiplier so the
# pipe, which is outside the central-cell envelope, sees a visible MHD response.
FIELD_SCALE = 8.0
FIELD_SCALE_SWEEP = (4.0, 6.0, 8.0, 10.0)
MHD_DRAG_FACTOR = 0.35
BEND_LOSS_COEFFICIENT = 0.35

STRAIGHT_POINTS = 72
BEND_POINTS = 144
CROSS_SECTION_POINTS = 55
TRANSIENT_PRESSURE_DRIVE_FACTOR = 1.10
TRANSIENT_TIME_STEP = 0.05
TRANSIENT_FINAL_TIME = 15.0
TRANSIENT_FRAMES = 72
MOVIE_FPS = 12


def _coil_parameters() -> dict[str, float | int]:
    if not WHAM_COIL_MODEL_SCRIPT.exists():
        return {
            "coil_separation": 1.96,
            "inner_radius": 0.043,
            "outer_radius": 0.365,
            "coil_axial_thickness": 0.1144,
            "radial_loops": 12,
            "axial_loops": 4,
            "current_scale": 100323.62459546926,
        }
    parsed = load_wham_coil_model_script(
        WHAM_COIL_MODEL_SCRIPT, radial_loops=12, axial_loops=4
    )
    return {
        "coil_separation": float(parsed["coil_separation"]),
        "inner_radius": float(parsed["inner_radius"]),
        "outer_radius": float(parsed["outer_radius"]),
        "coil_axial_thickness": float(parsed["coil_axial_thickness"]),
        "radial_loops": int(parsed["radial_loops"]),
        "axial_loops": int(parsed["axial_loops"]),
        "current_scale": float(parsed["current_scale"]),
    }


def run_wham_blanket_flow_demo() -> dict[str, object]:
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
        cross_section_points=CROSS_SECTION_POINTS,
    )
    flow = solve_wham_blanket_reduced_flow(
        centerline,
        geometry=geometry,
        properties=properties,
        settings=settings,
        coil_parameters=coil_parameters,
    )
    sweep_flows = []
    for field_scale in FIELD_SCALE_SWEEP:
        if abs(float(field_scale) - float(settings.field_scale)) < 1.0e-12:
            sweep_flows.append(flow)
            continue
        sweep_flows.append(
            solve_wham_blanket_reduced_flow(
                centerline,
                geometry=geometry,
                properties=properties,
                settings=replace(settings, field_scale=float(field_scale)),
                coil_parameters=coil_parameters,
            )
        )
    transient = solve_wham_blanket_transient_flow(
        flow,
        settings=BlanketTransientFlowSettings(
            pressure_drive_factor=TRANSIENT_PRESSURE_DRIVE_FACTOR,
            time_step=TRANSIENT_TIME_STEP,
            final_time=TRANSIENT_FINAL_TIME,
            frame_count=TRANSIENT_FRAMES,
        ),
    )
    plot_outputs = write_wham_blanket_flow_plots(flow, OUTPUT_DIR)
    sweep_outputs = write_wham_blanket_pressure_sweep_plots(sweep_flows, OUTPUT_DIR)
    transient_plot_outputs = write_wham_blanket_transient_flow_plots(
        transient, OUTPUT_DIR
    )
    movie_outputs = write_wham_blanket_transient_flow_movie(
        transient,
        OUTPUT_DIR,
        fps=MOVIE_FPS,
    )

    copied = []
    for output in [
        *plot_outputs,
        *sweep_outputs,
        *transient_plot_outputs,
        *movie_outputs,
    ]:
        if output.suffix.lower() in {".png", ".gif", ".json", ".csv"}:
            target = DOCS_OUTPUT_DIR / output.name
            shutil.copy2(output, target)
            copied.append(target.name)

    summary = {
        "case": "wham_blanket_flow",
        "docs_artifacts": copied,
        "metrics": flow["metrics"],
        "transient_metrics": transient["metrics"],
        "pressure_sweep_kpa": {
            str(float(item["settings"].field_scale)): float(
                item["metrics"]["pressure_drop_kpa"]
            )
            for item in sweep_flows
        },
        "model_limitations": (
            "The steady panel is a fixed-flow pressure budget. The movie and "
            "transient panel use a centerline pressure/velocity solve with "
            "turbulent pipe-friction closure, local MHD drag, and bend losses. "
            "Resolved 3D secondary flow and turbulence remain future solver "
            "validation work."
        ),
    }
    (OUTPUT_DIR / "wham_blanket_flow_demo_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"WHAM blanket reduced-flow artifacts written to {OUTPUT_DIR}")
    print(f"pressure_drop = {flow['metrics']['pressure_drop_kpa']:.3f} kPa")
    print(
        f"transient_final_pressure_drop = {transient['metrics']['final_pressure_drop_kpa']:.3f} kPa"
    )
    print(
        f"transient_final_mean_velocity = {transient['metrics']['final_mean_velocity_m_per_s']:.3f} m/s"
    )
    print(f"peak_Ha = {flow['metrics']['peak_hartmann_number']:.1f}")
    return summary


if __name__ == "__main__":
    run_wham_blanket_flow_demo()

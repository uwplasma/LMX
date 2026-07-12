from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lmx.field_models import (
    load_wham_coil_model_script,
    sample_wham_mirror_field,
    tabulated_field_quality_metrics,
    write_wham_mirror_field_npz,
)


OUTPUT_DIR = Path("artifacts/examples/wham_coil_model_field_adapter")
COIL_MODEL_SCRIPT = Path("/Users/rogerio/Downloads/coil_model_WHAM-1.txt")
RADIAL_LOOPS = 20
AXIAL_LOOPS = 5
PIPE_RADIUS = 0.22
PIPE_LENGTH = 1.40
FIELD_NX = 25
FIELD_NY = 15
FIELD_NZ = 31
CONTOUR_X = np.linspace(0.0, 2.5, 72)
CONTOUR_Z = np.linspace(-2.0, 2.0, 96)


def run_wham_coil_model_field_adapter(
    *,
    out_dir: Path = OUTPUT_DIR,
    coil_model_script: Path = COIL_MODEL_SCRIPT,
) -> dict[str, object]:
    """Convert the WHAM coil-model script into a tabulated LMX field artifact."""

    out_dir.mkdir(parents=True, exist_ok=True)
    script = Path(coil_model_script)
    if not script.exists():
        summary = {
            "case": "wham_coil_model_field_adapter",
            "status": "coil_model_script_missing",
            "coil_model_script": str(script),
        }
        (out_dir / "wham_coil_model_field_adapter_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        return summary

    params = load_wham_coil_model_script(
        script,
        radial_loops=RADIAL_LOOPS,
        axial_loops=AXIAL_LOOPS,
        preserve_ampere_turns=True,
    )
    x_table = np.linspace(0.0, PIPE_LENGTH, FIELD_NX)
    y_table = np.linspace(-PIPE_RADIUS, PIPE_RADIUS, FIELD_NY)
    z_table = np.linspace(-PIPE_RADIUS, PIPE_RADIUS, FIELD_NZ)
    table_path = write_wham_mirror_field_npz(
        out_dir / "wham_coil_model_field.npz",
        x=x_table,
        y=y_table,
        z=z_table,
        coil_separation=float(params["coil_separation"]),
        current_scale=float(params["current_scale"]),
        inner_radius=float(params["inner_radius"]),
        outer_radius=float(params["outer_radius"]),
        coil_axial_thickness=float(params["coil_axial_thickness"]),
        radial_loops=int(params["radial_loops"]),
        axial_loops=int(params["axial_loops"]),
        x_offset=-0.5 * PIPE_LENGTH,
    )
    quality = tabulated_field_quality_metrics(table_path)
    plot_paths = _write_wham_coil_model_plot(params, out_dir)
    summary = {
        "case": "wham_coil_model_field_adapter",
        "status": "wham_script_ingested",
        "coil_model_script": str(script),
        "field_table": table_path.name,
        "plots": [path.name for path in plot_paths],
        "parameters": params,
        "field_quality": quality,
        "notes": (
            "The reduced loop count preserves total ampere-turns from the WHAM "
            "script while keeping docs/example runtime bounded."
        ),
    }
    (out_dir / "wham_coil_model_field_adapter_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary


def _write_wham_coil_model_plot(params: dict[str, object], out_dir: Path) -> list[Path]:
    xx, zz = np.meshgrid(CONTOUR_X, CONTOUR_Z, indexing="ij")
    yy = np.zeros_like(xx)
    field = np.asarray(
        sample_wham_mirror_field(
            xx,
            yy,
            zz,
            coil_separation=float(params["coil_separation"]),
            current_scale=float(params["current_scale"]),
            inner_radius=float(params["inner_radius"]),
            outer_radius=float(params["outer_radius"]),
            coil_axial_thickness=float(params["coil_axial_thickness"]),
            radial_loops=int(params["radial_loops"]),
            axial_loops=int(params["axial_loops"]),
        ),
        dtype=float,
    )
    bmag = np.linalg.norm(field, axis=-1)
    axis_z = CONTOUR_Z
    axis_field = np.asarray(
        sample_wham_mirror_field(
            np.zeros_like(axis_z),
            np.zeros_like(axis_z),
            axis_z,
            coil_separation=float(params["coil_separation"]),
            current_scale=float(params["current_scale"]),
            inner_radius=float(params["inner_radius"]),
            outer_radius=float(params["outer_radius"]),
            coil_axial_thickness=float(params["coil_axial_thickness"]),
            radial_loops=int(params["radial_loops"]),
            axial_loops=int(params["axial_loops"]),
        ),
        dtype=float,
    )
    axis_b = np.linalg.norm(axis_field, axis=-1)

    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.8), constrained_layout=True)
    positive = bmag[bmag > 0.0]
    level_min = max(float(np.min(positive)), 1.0e-4) if positive.size else 1.0e-4
    level_max = max(float(np.max(bmag)), 1.0e-3)
    if level_max <= level_min * (1.0 + 1.0e-9):
        levels = np.linspace(level_min, level_min * 1.01, 18)
    else:
        levels = np.geomspace(level_min, level_max, 18)
    contour = axes[0].contourf(
        CONTOUR_X, CONTOUR_Z, bmag.T, levels=levels, cmap="magma"
    )
    axes[0].axhline(
        float(params["coil_center_negative"]),
        color="white",
        linestyle="--",
        linewidth=1.0,
    )
    axes[0].axhline(
        float(params["coil_center_positive"]),
        color="white",
        linestyle="--",
        linewidth=1.0,
    )
    axes[0].plot([0.0, PIPE_LENGTH], [0.0, 0.0], color="#38bdf8", linewidth=3.0)
    axes[0].text(0.04, 0.08, "pipe path", color="#38bdf8", fontsize=9, weight="bold")
    axes[0].set_title("WHAM script field magnitude")
    axes[0].set_xlabel("transverse distance [m]")
    axes[0].set_ylabel("mirror-axis coordinate [m]")
    fig.colorbar(contour, ax=axes[0], label="|B| [model units]")

    axes[1].plot(
        axis_z,
        axis_b / max(float(np.max(axis_b)), 1.0e-12),
        color="#0f766e",
        linewidth=2.0,
    )
    axes[1].axvline(
        float(params["coil_center_negative"]),
        color="#111827",
        linestyle="--",
        linewidth=1.0,
    )
    axes[1].axvline(
        float(params["coil_center_positive"]),
        color="#111827",
        linestyle="--",
        linewidth=1.0,
    )
    axes[1].set_title("Normalized mirror-axis field")
    axes[1].set_xlabel("mirror-axis coordinate [m]")
    axes[1].set_ylabel(r"$|B|/|B|_{max}$")
    axes[1].grid(True, alpha=0.25)

    axes[2].axis("off")
    lines = [
        "Parsed WHAM coil model",
        f"coil separation: {float(params['coil_separation']):.3f} m",
        f"inner/outer radius: {float(params['inner_radius']):.3f} / {float(params['outer_radius']):.3f} m",
        f"source loops: {int(params['source_radial_loops'])} x {int(params['source_axial_loops'])}",
        f"reduced loops: {int(params['radial_loops'])} x {int(params['axial_loops'])}",
        f"ampere-turns: {float(params['source_ampere_turns']):.3e}",
        f"reduced loop current: {float(params['current_scale']):.3e}",
    ]
    axes[2].text(
        0.03, 0.95, "\n".join(lines), va="top", fontsize=11, transform=axes[2].transAxes
    )

    png_path = out_dir / "wham_coil_model_field_adapter.png"
    pdf_path = out_dir / "wham_coil_model_field_adapter.pdf"
    for path in (png_path, pdf_path):
        fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return [png_path, pdf_path]


if __name__ == "__main__":
    run_wham_coil_model_field_adapter()

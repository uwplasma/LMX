from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lmx.cases import make_hunt_case, make_shercliff_case
from lmx.mesh import generate_layered_duct_mesh, generate_layered_duct_mesh_from_fluid_faces, generate_rect_duct_mesh
from lmx.reference_data import default_closed_channel_reference_root, load_processed_slice, processed_slice_point_mesh
from lmx.validation import duct_layer_resolution_gate


OUTPUT_DIR = Path("artifacts/examples/reference_slice_mesh_diagnostic")
REFERENCE_ROOT = default_closed_channel_reference_root()
CASE_KIND = "hunt"
HA = 20
X_SLICE = "1m"

WIDTH = 0.2
HEIGHT = 0.2
NY = 49
NZ = 37
WALL_THICKNESS = 0.001
WALL_CELLS = 2


def _case_and_generated_mesh(case_kind: str):
    if case_kind == "hunt":
        case = make_hunt_case(
            ha=HA,
            width=WIDTH,
            height=HEIGHT,
            ny=NY,
            nz=NZ,
            wall_cells=WALL_CELLS,
            wall_thickness=WALL_THICKNESS,
            fluid_conductivity=1.0e6,
            wall_conductivity=5.0e6,
            insulator_conductivity=1.0e-6,
            density=1.0e3,
            viscosity=1.0e-3,
        )
        mesh = generate_layered_duct_mesh(
            width=WIDTH,
            height=HEIGHT,
            ny=NY,
            nz=NZ,
            wall_thickness=case.geometry.wall_thickness,
            wall_cells=case.geometry.wall_cells,
            target_ha=HA,
            magnetic_axis="y",
        )
        return case, mesh
    if case_kind == "shercliff":
        case = make_shercliff_case(ha=HA, width=WIDTH, height=HEIGHT, ny=NY, nz=NZ)
        mesh = generate_rect_duct_mesh(width=WIDTH, height=HEIGHT, ny=NY, nz=NZ, target_ha=HA, magnetic_axis="y")
        return case, mesh
    raise ValueError(f"Unsupported CASE_KIND={case_kind!r}")


def _reference_like_mesh(reference, case_kind: str):
    point_mesh = processed_slice_point_mesh(reference)
    if case_kind == "hunt":
        return generate_layered_duct_mesh_from_fluid_faces(
            fluid_y_faces=point_mesh.y_faces,
            fluid_z_faces=point_mesh.z_faces,
            width=WIDTH,
            height=HEIGHT,
            wall_thickness=(WALL_THICKNESS, WALL_THICKNESS, WALL_THICKNESS, WALL_THICKNESS),
            wall_cells=(WALL_CELLS, WALL_CELLS, WALL_CELLS, WALL_CELLS),
        )
    return point_mesh


def _spacing_summary(mesh) -> dict[str, float | int]:
    return {
        "ny": int(mesh.ny),
        "nz": int(mesh.nz),
        "min_dy": float(np.min(np.asarray(mesh.dy))),
        "max_dy": float(np.max(np.asarray(mesh.dy))),
        "min_dz": float(np.min(np.asarray(mesh.dz))),
        "max_dz": float(np.max(np.asarray(mesh.dz))),
    }


def _draw_grid(ax, mesh, title: str) -> None:
    y_faces = np.asarray(mesh.y_faces, dtype=float)
    z_faces = np.asarray(mesh.z_faces, dtype=float)
    y_stride = max(1, y_faces.size // 80)
    z_stride = max(1, z_faces.size // 80)
    for y in y_faces[::y_stride]:
        ax.plot([y, y], [z_faces[0], z_faces[-1]], color="#1f2937", linewidth=0.35, alpha=0.35)
    for z in z_faces[::z_stride]:
        ax.plot([y_faces[0], y_faces[-1]], [z, z], color="#1f2937", linewidth=0.35, alpha=0.35)
    ax.plot(
        [-0.5 * WIDTH, 0.5 * WIDTH, 0.5 * WIDTH, -0.5 * WIDTH, -0.5 * WIDTH],
        [-0.5 * HEIGHT, -0.5 * HEIGHT, 0.5 * HEIGHT, 0.5 * HEIGHT, -0.5 * HEIGHT],
        color="#dc2626",
        linewidth=1.5,
    )
    ax.set_title(title)
    ax.set_xlabel("y [m]")
    ax.set_ylabel("z [m]")
    ax.set_aspect("equal", adjustable="box")


def run_reference_slice_mesh_diagnostic() -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    reference = load_processed_slice(CASE_KIND, HA, x_slice=X_SLICE, reference_root=REFERENCE_ROOT)
    case, generated_mesh = _case_and_generated_mesh(CASE_KIND)
    reference_mesh = _reference_like_mesh(reference, CASE_KIND)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), constrained_layout=True)
    _draw_grid(axes[0], generated_mesh, "LMX generated benchmark mesh")
    _draw_grid(axes[1], reference_mesh, "Processed-slice point mesh")
    fig.suptitle(f"{CASE_KIND.capitalize()} Ha={HA} mesh diagnostic", fontsize=14)
    png_path = OUTPUT_DIR / "reference_slice_mesh_diagnostic.png"
    pdf_path = OUTPUT_DIR / "reference_slice_mesh_diagnostic.pdf"
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)

    summary = {
        "case_kind": CASE_KIND,
        "ha": HA,
        "x_slice": X_SLICE,
        "reference_path": reference.path,
        "generated_mesh": _spacing_summary(generated_mesh),
        "reference_point_mesh": _spacing_summary(reference_mesh),
        "generated_layer_gate": duct_layer_resolution_gate(case, generated_mesh),
        "reference_point_layer_gate": duct_layer_resolution_gate(case, reference_mesh),
        "plots": [str(png_path), str(pdf_path)],
    }
    (OUTPUT_DIR / "reference_slice_mesh_diagnostic_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    run_reference_slice_mesh_diagnostic()

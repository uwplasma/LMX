from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp

from lmx import generate_rect_duct_mesh, write_operator_verification_plots
from lmx.operators import gradient_scalar, laplacian_scalar


OUTPUT_DIR = Path("artifacts/examples/operator_verification")
RESOLUTIONS = (16, 32, 64)


def _gradient_reference(y: jnp.ndarray, z: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    field = jnp.sin(jnp.pi * y) * jnp.cos(0.5 * jnp.pi * z)
    exact_y = jnp.pi * jnp.cos(jnp.pi * y) * jnp.cos(0.5 * jnp.pi * z)
    exact_z = -0.5 * jnp.pi * jnp.sin(jnp.pi * y) * jnp.sin(0.5 * jnp.pi * z)
    return field, exact_y, exact_z


def _laplacian_reference(y: jnp.ndarray, z: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    field = jnp.sin(jnp.pi * y) * jnp.sin(jnp.pi * z)
    exact = -2.0 * (jnp.pi**2) * field
    return field, exact


def run_operator_verification_demo() -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, float]] = []

    for resolution in RESOLUTIONS:
        mesh = generate_rect_duct_mesh(width=2.0, height=2.0, ny=resolution, nz=resolution)
        y, z = jnp.meshgrid(mesh.y_centers, mesh.z_centers, indexing="ij")

        gradient_field, exact_grad_y, exact_grad_z = _gradient_reference(y, z)
        grad_y, grad_z = gradient_scalar(gradient_field, mesh)
        sl = (slice(2, -2), slice(2, -2))
        gradient_y_l2_error = float(jnp.sqrt(jnp.mean((grad_y[sl] - exact_grad_y[sl]) ** 2)))
        gradient_z_l2_error = float(jnp.sqrt(jnp.mean((grad_z[sl] - exact_grad_z[sl]) ** 2)))

        lap_field, exact_lap = _laplacian_reference(y, z)
        lap = laplacian_scalar(lap_field, mesh)
        sl_lap = (slice(4, -4), slice(4, -4))
        laplacian_l2_error = float(jnp.sqrt(jnp.mean((lap[sl_lap] - exact_lap[sl_lap]) ** 2)))

        records.append(
            {
                "resolution": float(resolution),
                "max_spacing": float(jnp.max(mesh.dy)),
                "gradient_y_l2_error": gradient_y_l2_error,
                "gradient_z_l2_error": gradient_z_l2_error,
                "laplacian_l2_error": laplacian_l2_error,
            }
        )

    plots = write_operator_verification_plots(records, OUTPUT_DIR, case_title="LMX operator verification")

    spacing = jnp.asarray([row["max_spacing"] for row in records], dtype=float)
    grad_y_error = jnp.asarray([row["gradient_y_l2_error"] for row in records], dtype=float)
    grad_z_error = jnp.asarray([row["gradient_z_l2_error"] for row in records], dtype=float)
    lap_error = jnp.asarray([row["laplacian_l2_error"] for row in records], dtype=float)
    grad_y_order = float(jnp.mean(jnp.log(grad_y_error[:-1] / grad_y_error[1:]) / jnp.log(spacing[:-1] / spacing[1:])))
    grad_z_order = float(jnp.mean(jnp.log(grad_z_error[:-1] / grad_z_error[1:]) / jnp.log(spacing[:-1] / spacing[1:])))
    lap_order = float(jnp.mean(jnp.log(lap_error[:-1] / lap_error[1:]) / jnp.log(spacing[:-1] / spacing[1:])))

    summary = {
        "case": "operator_verification",
        "records": records,
        "observed_order": {
            "gradient_y": grad_y_order,
            "gradient_z": grad_z_order,
            "laplacian": lap_order,
        },
        "literature_references": [
            "Manufactured-solution and observed-order verification practice for elliptic operators",
            "Samper et al., verification and validation ladder for fusion MHD codes",
        ],
        "plots": [path.name for path in plots],
    }
    (OUTPUT_DIR / "operator_verification_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    run_operator_verification_demo()

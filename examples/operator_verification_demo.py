"""Measure the observed accuracy of LMX gradient and Laplacian operators.

Edit ``RESOLUTIONS`` below, then run this file from top to bottom. The example
writes a compact convergence record and plot beneath ``artifacts/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp

from lmx.mesh import generate_rect_duct_mesh
from lmx.operators import gradient_scalar, laplacian_scalar
from lmx.plotting import write_operator_verification_plots


# Inputs: domain, refinement ladder, and output directory.
OUTPUT_DIR = Path("artifacts/examples/operator_verification")
WIDTH = 2.0
HEIGHT = 2.0
RESOLUTIONS = (16, 32, 64)


def _gradient_reference(
    y: jnp.ndarray, z: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return a smooth scalar field and its exact first derivatives."""

    field = jnp.sin(jnp.pi * y) * jnp.cos(0.5 * jnp.pi * z)
    exact_y = jnp.pi * jnp.cos(jnp.pi * y) * jnp.cos(0.5 * jnp.pi * z)
    exact_z = -0.5 * jnp.pi * jnp.sin(jnp.pi * y) * jnp.sin(0.5 * jnp.pi * z)
    return field, exact_y, exact_z


def _laplacian_reference(
    y: jnp.ndarray, z: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return a smooth scalar field and its exact Laplacian."""

    field = jnp.sin(jnp.pi * y) * jnp.sin(jnp.pi * z)
    return field, -2.0 * (jnp.pi**2) * field


# Run the same manufactured fields on every mesh in the refinement ladder.
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
records: list[dict[str, float]] = []
for resolution in RESOLUTIONS:
    mesh = generate_rect_duct_mesh(
        width=WIDTH, height=HEIGHT, ny=resolution, nz=resolution
    )
    y, z = jnp.meshgrid(mesh.y_centers, mesh.z_centers, indexing="ij")

    gradient_field, exact_grad_y, exact_grad_z = _gradient_reference(y, z)
    grad_y, grad_z = gradient_scalar(gradient_field, mesh)
    gradient_slice = (slice(2, -2), slice(2, -2))

    laplacian_field, exact_laplacian = _laplacian_reference(y, z)
    laplacian = laplacian_scalar(laplacian_field, mesh)
    laplacian_slice = (slice(4, -4), slice(4, -4))
    records.append(
        {
            "resolution": float(resolution),
            "max_spacing": float(jnp.max(mesh.dy)),
            "gradient_y_l2_error": float(
                jnp.sqrt(
                    jnp.mean(
                        (grad_y[gradient_slice] - exact_grad_y[gradient_slice]) ** 2
                    )
                )
            ),
            "gradient_z_l2_error": float(
                jnp.sqrt(
                    jnp.mean(
                        (grad_z[gradient_slice] - exact_grad_z[gradient_slice]) ** 2
                    )
                )
            ),
            "laplacian_l2_error": float(
                jnp.sqrt(
                    jnp.mean(
                        (
                            laplacian[laplacian_slice]
                            - exact_laplacian[laplacian_slice]
                        )
                        ** 2
                    )
                )
            ),
        }
    )

# Calculate observed orders, then save the evidence and its plot.
spacing = jnp.asarray([row["max_spacing"] for row in records])
observed_order = {}
for name in ("gradient_y", "gradient_z", "laplacian"):
    errors = jnp.asarray([row[f"{name}_l2_error"] for row in records])
    observed_order[name] = float(
        jnp.mean(
            jnp.log(errors[:-1] / errors[1:])
            / jnp.log(spacing[:-1] / spacing[1:])
        )
    )

plots = write_operator_verification_plots(
    records, OUTPUT_DIR, case_title="LMX operator verification"
)
summary = {
    "case": "operator_verification",
    "records": records,
    "observed_order": observed_order,
    "literature_references": [
        "Manufactured-solution and observed-order verification for elliptic operators",
        "Samper et al. verification and validation ladder for fusion MHD codes",
    ],
    "plots": [path.name for path in plots],
}
summary_path = OUTPUT_DIR / "operator_verification_summary.json"
summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))

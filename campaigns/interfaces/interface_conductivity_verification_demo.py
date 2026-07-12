from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from lmx import generate_rect_duct_mesh
from lmx.plotting import write_interface_verification_plots
import lmx.solvers as solvers


OUTPUT_DIR = Path("artifacts/examples/interface_conductivity_verification")
RESOLUTIONS = (24, 48, 96)
SIGMA_LEFT = 1.0
SIGMA_RIGHT = 5.0
BOUNDARY_LEFT = 0.0
BOUNDARY_RIGHT = 1.0


def _exact_piecewise_linear_solution(y: jnp.ndarray) -> tuple[jnp.ndarray, float]:
    left_length = 1.0
    right_length = 1.0
    flux = (BOUNDARY_RIGHT - BOUNDARY_LEFT) / (
        left_length / SIGMA_LEFT + right_length / SIGMA_RIGHT
    )
    interface_value = BOUNDARY_LEFT + flux * left_length / SIGMA_LEFT
    exact = jnp.where(
        y < 0.0,
        BOUNDARY_LEFT + flux * (y + 1.0) / SIGMA_LEFT,
        interface_value + flux * y / SIGMA_RIGHT,
    )
    return exact, float(flux)


def _solve_interface_problem(mesh) -> tuple[np.ndarray, float]:
    ny = mesh.ny
    y = mesh.y_centers
    dy = mesh.dy
    sigma_1d = jnp.where(y < 0.0, SIGMA_LEFT, SIGMA_RIGHT)
    sigma_2d = jnp.repeat(sigma_1d[:, None], 2, axis=1)
    interface = np.asarray(solvers._interface_conductance_y(mesh, sigma_2d))[:, 0]

    matrix = np.zeros((ny, ny), dtype=float)
    rhs = np.zeros(ny, dtype=float)

    left_boundary_conductance = float(sigma_1d[0] / max(0.5 * dy[0], 1e-12))
    right_boundary_conductance = float(sigma_1d[-1] / max(0.5 * dy[-1], 1e-12))

    for i in range(ny):
        west = left_boundary_conductance if i == 0 else float(interface[i - 1])
        east = right_boundary_conductance if i == ny - 1 else float(interface[i])
        matrix[i, i] = west + east
        if i > 0:
            matrix[i, i - 1] = -west
        else:
            rhs[i] += west * BOUNDARY_LEFT
        if i < ny - 1:
            matrix[i, i + 1] = -east
        else:
            rhs[i] += east * BOUNDARY_RIGHT

    numeric = np.linalg.solve(matrix, rhs)
    interface_index = int(np.searchsorted(np.asarray(y), 0.0) - 1)
    interface_index = max(0, min(interface_index, ny - 2))
    left_flux = interface[interface_index] * (
        numeric[interface_index + 1] - numeric[interface_index]
    )
    return numeric, float(left_flux)


def run_interface_conductivity_verification_demo() -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, float]] = []
    profile_payload: dict[str, np.ndarray] | None = None

    for resolution in RESOLUTIONS:
        mesh = generate_rect_duct_mesh(width=2.0, height=1.0, ny=resolution, nz=2)
        y = mesh.y_centers
        exact, exact_flux = _exact_piecewise_linear_solution(y)
        numeric, numeric_flux = _solve_interface_problem(mesh)
        profile_l2_error = float(
            jnp.sqrt(jnp.mean((jnp.asarray(numeric) - exact) ** 2))
        )
        flux_error = abs(float(numeric_flux) - exact_flux)

        records.append(
            {
                "resolution": float(resolution),
                "max_spacing": float(jnp.max(mesh.dy)),
                "profile_l2_error": profile_l2_error,
                "flux_error": flux_error,
            }
        )

        if resolution == RESOLUTIONS[-1]:
            profile_payload = {
                "y": np.asarray(y),
                "u_exact": np.asarray(exact),
                "u_numeric": np.asarray(numeric),
            }

    assert profile_payload is not None
    plots = write_interface_verification_plots(
        records,
        profile_payload,
        OUTPUT_DIR,
        case_title="LMX interface-conductivity verification",
        interface_location=0.0,
    )

    spacing = jnp.asarray([row["max_spacing"] for row in records], dtype=float)
    profile_error = jnp.asarray(
        [row["profile_l2_error"] for row in records], dtype=float
    )
    flux_error = jnp.asarray([row["flux_error"] for row in records], dtype=float)
    if bool(jnp.all(profile_error > 1e-12)):
        profile_order = float(
            jnp.mean(
                jnp.log(profile_error[:-1] / profile_error[1:])
                / jnp.log(spacing[:-1] / spacing[1:])
            )
        )
    else:
        profile_order = None
    if bool(jnp.all(flux_error > 1e-12)):
        flux_order = float(
            jnp.mean(
                jnp.log(flux_error[:-1] / flux_error[1:])
                / jnp.log(spacing[:-1] / spacing[1:])
            )
        )
    else:
        flux_order = None

    summary = {
        "case": "interface_conductivity_verification",
        "conductivity_jump": {
            "left": SIGMA_LEFT,
            "right": SIGMA_RIGHT,
        },
        "records": records,
        "observed_order": {
            "profile": profile_order,
            "flux": flux_order,
        },
        "literature_references": [
            "Finite-volume verification with discontinuous coefficients and harmonic averaging",
            "Samper et al., verification and validation ladder for fusion MHD codes",
        ],
        "plots": [path.name for path in plots],
    }
    (OUTPUT_DIR / "interface_conductivity_verification_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary


if __name__ == "__main__":
    run_interface_conductivity_verification_demo()

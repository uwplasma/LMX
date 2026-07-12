from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lmx.autodiff import (
    build_fringing_autodiff_problem,
    wham_mirror_pressure_drop_sensitivity,
)


OUTPUT_DIR = Path("artifacts/examples/autodiff_wham_pressure_sensitivity")
NX_STATIONS = 25
LENGTH = 1.40
NY = 12
NZ = 12
FORCING = 1.0
PEAK_HARTMANN_NUMBER = 20.0
REFERENCE_SEPARATION = 1.96
SEPARATION_SWEEP = np.linspace(1.50, 2.30, 9)
RADIAL_LOOPS = 16
AXIAL_LOOPS = 4


def run_autodiff_wham_pressure_sensitivity() -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    problem = build_fringing_autodiff_problem(
        nx_stations=NX_STATIONS,
        length=LENGTH,
        ny=NY,
        nz=NZ,
        macro_iterations=3,
        potential_iterations=12,
        velocity_iterations=16,
    )
    reference = wham_mirror_pressure_drop_sensitivity(
        problem,
        forcing=FORCING,
        peak_hartmann_number=PEAK_HARTMANN_NUMBER,
        coil_separation=REFERENCE_SEPARATION,
        radial_loops=RADIAL_LOOPS,
        axial_loops=AXIAL_LOOPS,
    )
    sweep = [
        wham_mirror_pressure_drop_sensitivity(
            problem,
            forcing=FORCING,
            peak_hartmann_number=PEAK_HARTMANN_NUMBER,
            coil_separation=float(separation),
            radial_loops=RADIAL_LOOPS,
            axial_loops=AXIAL_LOOPS,
        )
        for separation in SEPARATION_SWEEP
    ]
    pressure_proxy = np.asarray(
        [float(item["pressure_drop_proxy"]) for item in sweep], dtype=float
    )
    pressure_gradient = np.asarray(
        [float(item["d_pressure_drop_d_separation"]) for item in sweep], dtype=float
    )
    x = np.asarray(reference["x"], dtype=float)

    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "legend.fontsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.7), constrained_layout=True)
    axes[0].plot(
        x,
        np.asarray(reference["field_scale"], dtype=float),
        color="#1d4ed8",
        label=r"$B/B_{max}$",
    )
    axes[0].plot(
        x,
        np.asarray(reference["pressure_span"], dtype=float)
        / max(
            float(np.max(np.asarray(reference["pressure_span"], dtype=float))), 1.0e-12
        ),
        color="#b91c1c",
        label=r"$\Delta p/\Delta p_{max}$",
    )
    axes[0].set_title("Reference response")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("Normalized response")
    axes[0].legend(loc="best")

    axes[1].plot(SEPARATION_SWEEP, pressure_proxy, marker="o", color="#0f766e")
    axes[1].axvline(
        REFERENCE_SEPARATION, color="#111827", linestyle="--", linewidth=1.0
    )
    axes[1].set_title("Pressure-drop proxy")
    axes[1].set_xlabel("coil separation")
    axes[1].set_ylabel("proxy")

    axes[2].plot(SEPARATION_SWEEP, pressure_gradient, marker="o", color="#7c3aed")
    axes[2].axvline(
        REFERENCE_SEPARATION, color="#111827", linestyle="--", linewidth=1.0
    )
    axes[2].set_title("Autodiff sensitivity")
    axes[2].set_xlabel("coil separation")
    axes[2].set_ylabel(r"$d(\Delta p)/ds$")

    png_path = OUTPUT_DIR / "autodiff_wham_pressure_sensitivity.png"
    pdf_path = OUTPUT_DIR / "autodiff_wham_pressure_sensitivity.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "case": "autodiff_wham_pressure_sensitivity",
        "plots": [png_path.name, pdf_path.name],
        "reference_separation": REFERENCE_SEPARATION,
        "pressure_drop_proxy": float(reference["pressure_drop_proxy"]),
        "d_pressure_drop_d_separation": float(
            reference["d_pressure_drop_d_separation"]
        ),
        "separation_sweep": SEPARATION_SWEEP.tolist(),
        "pressure_drop_curve": pressure_proxy.tolist(),
        "sensitivity_curve": pressure_gradient.tolist(),
    }
    (OUTPUT_DIR / "autodiff_wham_pressure_sensitivity_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary


if __name__ == "__main__":
    run_autodiff_wham_pressure_sensitivity()

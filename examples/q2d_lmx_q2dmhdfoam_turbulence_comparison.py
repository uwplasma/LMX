from __future__ import annotations

import json
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np

from lmx import (
    build_q2d_turbulence_decay_case,
    load_q2dmhdfoam_lid_driven_observables,
    solve_q2d_turbulence_decay,
    validate_q2d_turbulence_decay_observables,
    write_q2d_turbulence_decay_movie,
)


OUTPUT_DIR = Path("artifacts/examples/q2d_lmx_q2dmhdfoam_turbulence_comparison")
DOCS_OUTPUT_DIR = Path("docs/_static/generated")
Q2DMHDFOAM_ROOT = Path("/Users/rogerio/local/tests/lmx_external_codes/Q2DmhdFoam")
Q2DMHDFOAM_LID_DRIVEN_SUMMARY = Q2DMHDFOAM_ROOT / "run/lidDriven/IDM_output_U.txt"

NX = 96
NY = 96
VISCOSITY = 8.0e-4
HARTMANN_FRICTION = 0.08
AMPLITUDE = 6.0
FORCING_AMPLITUDE = 0.08
FORCING_WAVENUMBER = 4
DT = 2.0e-3
T_FINAL = 3.0
FRAME_COUNT = 72
FPS = 14
COPY_TO_DOCS = True


def run_q2d_lmx_q2dmhdfoam_turbulence_comparison(
    *,
    out_dir: Path = OUTPUT_DIR,
    docs_output_dir: Path = DOCS_OUTPUT_DIR,
    q2dmhdfoam_lid_driven_summary: Path = Q2DMHDFOAM_LID_DRIVEN_SUMMARY,
    copy_to_docs: bool = COPY_TO_DOCS,
) -> dict[str, object]:
    """Write the current LMX/Q2DmhdFoam Q2D turbulence comparison artifact.

    This example compares LMX nonlinear Q2D observables with the available
    Q2DmhdFoam lid-driven spectral summary. It intentionally records
    ``matched_parity = False`` because the archived Q2DmhdFoam lid-driven case
    is not the same physical case as the periodic LMX SM82-style movie.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    docs_output_dir.mkdir(parents=True, exist_ok=True)
    case = build_q2d_turbulence_decay_case(
        nx=NX,
        ny=NY,
        viscosity=VISCOSITY,
        hartmann_friction=HARTMANN_FRICTION,
        amplitude=AMPLITUDE,
        forcing_amplitude=FORCING_AMPLITUDE,
        forcing_wavenumber=FORCING_WAVENUMBER,
        dt=DT,
        t_final=T_FINAL,
        frame_count=FRAME_COUNT,
    )
    solution = solve_q2d_turbulence_decay(case)
    validation = validate_q2d_turbulence_decay_observables(case, solution)
    q2dmhdfoam_observables = _load_external_summary(q2dmhdfoam_lid_driven_summary)
    plots = _write_comparison_panel(solution, validation, q2dmhdfoam_observables, out_dir)
    movie_paths = write_q2d_turbulence_decay_movie(
        solution,
        out_dir,
        title="LMX Q2D turbulence benchmark movie",
        fps=FPS,
    )
    comparison_gif = out_dir / "q2d_lmx_q2dmhdfoam_turbulence_comparison.gif"
    shutil.copy2(movie_paths[0], comparison_gif)

    copied: list[str] = []
    if copy_to_docs:
        for path in [*plots, comparison_gif]:
            target = docs_output_dir / path.name
            shutil.copy2(path, target)
            copied.append(target.name)

    summary = {
        "case": "q2d_lmx_q2dmhdfoam_turbulence_comparison",
        "lmx_case": {
            "nx": NX,
            "ny": NY,
            "viscosity": VISCOSITY,
            "hartmann_friction": HARTMANN_FRICTION,
            "dt": DT,
            "t_final": T_FINAL,
            "frame_count": FRAME_COUNT,
        },
        "lmx_validation": validation,
        "q2dmhdfoam_observables": q2dmhdfoam_observables,
        "matched_parity": False,
        "strict_blocker_closed": False,
        "plots": [path.name for path in plots],
        "movie": comparison_gif.name,
        "docs_artifacts": copied,
        "next_step": (
            "Run Q2DmhdFoam and LMX on the same Q2D geometry, forcing, "
            "Hartmann friction, integration time, and observable definitions; "
            "then fill q2d_turbulence_reference_observables.csv."
        ),
    }
    summary_path = out_dir / "q2d_lmx_q2dmhdfoam_turbulence_comparison_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if copy_to_docs:
        shutil.copy2(summary_path, docs_output_dir / summary_path.name)
    return summary


def _load_external_summary(path: Path) -> dict[str, object]:
    if not path.exists():
        return {
            "status": "q2dmhdfoam_lid_driven_summary_missing",
            "source_path": str(path),
        }
    payload = dict(load_q2dmhdfoam_lid_driven_observables(path))
    payload["status"] = "q2dmhdfoam_lid_driven_summary_loaded"
    return payload


def _write_comparison_panel(
    solution,
    validation: dict[str, object],
    q2dmhdfoam_observables: dict[str, object],
    out_dir: Path,
) -> list[Path]:
    final_field = np.asarray(solution.frames[-1], dtype=float)
    spectrum = solution.final_spectrum
    wavenumber = np.asarray(spectrum["wavenumber"], dtype=float)
    energy = np.asarray(spectrum["energy"], dtype=float)
    positive = (wavenumber > 0.0) & (energy > 0.0)
    energy_history = np.asarray(solution.kinetic_energy, dtype=float)
    enstrophy_history = np.asarray(solution.enstrophy_proxy, dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 9.2), constrained_layout=True)
    vmax = max(float(np.max(np.abs(final_field))), 1.0e-12)
    image = axes[0, 0].imshow(
        final_field.T,
        origin="lower",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        extent=(float(solution.x[0]), float(solution.x[-1]), float(solution.y[0]), float(solution.y[-1])),
        aspect="equal",
    )
    axes[0, 0].set_title("LMX final Q2D vorticity")
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel("y")
    fig.colorbar(image, ax=axes[0, 0], fraction=0.046, pad=0.04)

    time = np.asarray(solution.time, dtype=float)
    axes[0, 1].plot(time, energy_history / max(energy_history[0], 1.0e-20), color="#0f766e", linewidth=2.0, label="LMX kinetic energy")
    axes[0, 1].plot(time, enstrophy_history / max(enstrophy_history[0], 1.0e-20), color="#b45309", linewidth=2.0, label="LMX enstrophy proxy")
    axes[0, 1].set_title("LMX time-history gates")
    axes[0, 1].set_xlabel("time")
    axes[0, 1].set_ylabel("normalized value")
    axes[0, 1].grid(True, alpha=0.25)
    axes[0, 1].legend(frameon=False)

    axes[1, 0].loglog(wavenumber[positive], energy[positive], marker="o", linewidth=1.8, color="#1d4ed8", label="LMX final spectrum")
    for key, color, label in (
        ("weak_dominant_wavenumber", "#dc2626", "Q2DmhdFoam weak dominant"),
        ("strong_dominant_wavenumber", "#7c3aed", "Q2DmhdFoam strong dominant"),
    ):
        if key in q2dmhdfoam_observables:
            axes[1, 0].axvline(float(q2dmhdfoam_observables[key]), color=color, linestyle="--", linewidth=1.2, label=label)
    axes[1, 0].set_title("Spectrum comparison vocabulary")
    axes[1, 0].set_xlabel("|k|")
    axes[1, 0].set_ylabel("shell energy")
    axes[1, 0].grid(True, which="both", alpha=0.25)
    axes[1, 0].legend(frameon=False, fontsize=8)

    axes[1, 1].axis("off")
    lines = [
        "Validation status",
        f"LMX nonlinear movie gate: {bool(validation.get('validation_pass'))}",
        f"LMX frames: {int(validation.get('frame_count', 0))}",
        f"LMX turnover count: {float(validation.get('turnover_count', 0.0)):.3g}",
        f"LMX max CFL: {float(validation.get('max_courant', 0.0)):.3g}",
        "",
        "Q2DmhdFoam evidence",
        f"status: {q2dmhdfoam_observables.get('status', 'missing')}",
    ]
    for key in (
        "weak_mode_count",
        "weak_weighted_wavenumber",
        "weak_peak_over_max_max",
        "strong_mode_count",
        "strong_peak_over_max_max",
    ):
        if key in q2dmhdfoam_observables:
            value = q2dmhdfoam_observables[key]
            lines.append(f"{key}: {float(value):.4g}" if isinstance(value, (float, int)) else f"{key}: {value}")
    lines.extend(
        [
            "",
            "Strict parity: open",
            "Reason: Q2DmhdFoam lid-driven case is not matched to the periodic LMX SM82-style run.",
        ]
    )
    axes[1, 1].text(0.02, 0.98, "\n".join(lines), va="top", fontsize=10.0, transform=axes[1, 1].transAxes)

    fig.suptitle("LMX and Q2DmhdFoam Q2D turbulence-observable comparison", fontsize=15.5, fontweight="bold")
    paths = [
        out_dir / "q2d_lmx_q2dmhdfoam_turbulence_comparison.png",
        out_dir / "q2d_lmx_q2dmhdfoam_turbulence_comparison.pdf",
    ]
    for path in paths:
        fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return paths


if __name__ == "__main__":
    run_q2d_lmx_q2dmhdfoam_turbulence_comparison()

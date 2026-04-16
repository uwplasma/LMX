from __future__ import annotations

import argparse
import json
from dataclasses import is_dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lmx.fringing import build_pipe_ogrid_extruded_problem, solve_extruded_inductionless


def _set_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.figsize": (13.5, 8.0),
            "axes.grid": True,
            "grid.alpha": 0.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
        }
    )


def _replace_fields(obj, **changes):
    if is_dataclass(obj):
        return replace(obj, **changes)
    for name, value in changes.items():
        setattr(obj, name, value)
    return obj


def _load_reference_profile(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    data = np.genfromtxt(path, delimiter=",", names=True)
    coord = np.asarray(data["Points2"], dtype=float)
    velocity = np.asarray(data["U2"], dtype=float)
    x_offset = float(np.mean(np.asarray(data["Points0"], dtype=float)))
    coord_scale = max(np.max(np.abs(coord)), 1.0e-12)
    return coord / coord_scale, velocity, x_offset / coord_scale


def _extract_pipe_profile(
    bundle,
    *,
    x_offset_fraction: float,
    samples: int = 121,
) -> tuple[np.ndarray, np.ndarray]:
    peak_index = int(np.argmax(np.abs(np.asarray(bundle.field_scale, dtype=float))))
    radius = float(np.max(np.asarray(bundle.y, dtype=float)))
    u_station = np.asarray(bundle.u[peak_index], dtype=float)
    radial = np.asarray(bundle.y, dtype=float)
    theta = np.asarray(bundle.z, dtype=float)
    target_x = x_offset_fraction * radius
    z_limit = max(np.sqrt(max(radius**2 - target_x**2, 0.0)), 1.0e-12)
    z_targets = np.linspace(-z_limit, z_limit, samples)
    profile = np.zeros_like(z_targets)

    def sample_value(r_target: float, theta_target: float) -> float:
        r_target = float(np.clip(r_target, radial[0], radial[-1]))
        theta_target = float(np.mod(theta_target, 2.0 * np.pi))
        r_hi = int(np.searchsorted(radial, r_target, side="right"))
        r_hi = min(max(r_hi, 1), radial.size - 1)
        r_lo = r_hi - 1
        r_span = max(radial[r_hi] - radial[r_lo], 1.0e-12)
        r_weight = (r_target - radial[r_lo]) / r_span

        theta_extended = np.concatenate([theta, [theta[0] + 2.0 * np.pi]])
        u_extended = np.concatenate([u_station, u_station[:, :1]], axis=1)
        t_hi = int(np.searchsorted(theta_extended, theta_target, side="right"))
        t_hi = min(max(t_hi, 1), theta_extended.size - 1)
        t_lo = t_hi - 1
        t_span = max(theta_extended[t_hi] - theta_extended[t_lo], 1.0e-12)
        t_weight = (theta_target - theta_extended[t_lo]) / t_span

        v00 = u_extended[r_lo, t_lo]
        v01 = u_extended[r_lo, t_hi]
        v10 = u_extended[r_hi, t_lo]
        v11 = u_extended[r_hi, t_hi]
        return float(
            (1.0 - r_weight) * ((1.0 - t_weight) * v00 + t_weight * v01)
            + r_weight * ((1.0 - t_weight) * v10 + t_weight * v11)
        )

    for idx, z_target in enumerate(z_targets):
        r_target = float(np.sqrt(target_x**2 + z_target**2))
        theta_target = float(np.arctan2(z_target, target_x if abs(target_x) > 1.0e-12 else 1.0e-12))
        if abs(target_x) <= 1.0e-12:
            theta_target = np.pi / 2.0 if z_target >= 0.0 else 3.0 * np.pi / 2.0
        profile[idx] = sample_value(r_target, theta_target)
    return z_targets / z_limit, profile


def run_pipe_reference_comparison_demo(
    *,
    out_dir: Path,
    ha_peak: float = 20.0,
    nr: int = 18,
    ntheta: int = 48,
    nx_stations: int = 7,
    max_steps: int = 18,
    coupling_iterations: int = 10,
    potential_iterations: int = 60,
    reference_dir: Path | None = None,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    base_dir = (
        reference_dir
        if reference_dir is not None
        else Path(__file__).resolve().parents[1]
        / "external"
        / "FreeMHDPaperAllFigures"
        / "FreeMHDPaperAllFigures"
        / "FringingBPipe"
    )
    reference_paths = {
        "center": base_dir / "Buhler2020PaperProperties_Ha2k_Re20k_coarserZMesh5x_CenterLine_5.89s.csv",
        "negative": base_dir / "Buhler2020PaperProperties_Ha2k_Re20k_coarserZMesh5x_NegXLine_5.89s.csv",
        "positive": base_dir / "Buhler2020PaperProperties_Ha2k_Re20k_coarserZMesh5x_PosXLine_5.89s.csv",
    }

    problem = build_pipe_ogrid_extruded_problem(
        ha_peak=ha_peak,
        nr=nr,
        ntheta=ntheta,
        nx_stations=nx_stations,
    )
    case_updates = {
        "solver": _replace_fields(problem.case.solver, coupling_iterations=coupling_iterations),
    }
    if hasattr(problem.case, "time_stepper"):
        case_updates["time_stepper"] = _replace_fields(
            problem.case.time_stepper,
            max_steps=max_steps,
            potential_iterations=potential_iterations,
        )
    problem = _replace_fields(problem, case=_replace_fields(problem.case, **case_updates))
    solution = solve_extruded_inductionless(problem)

    reference_profiles = {name: _load_reference_profile(path) for name, path in reference_paths.items()}
    reference_velocity_scale = max(
        max(np.max(np.abs(velocity)), 1.0e-12) for _, velocity, _ in reference_profiles.values()
    )
    lmx_profiles = {
        name: _extract_pipe_profile(solution.bundle, x_offset_fraction=offset)
        for name, (_, _, offset) in reference_profiles.items()
    }
    lmx_velocity_scale = max(max(np.max(np.abs(profile)), 1.0e-12) for _, profile in lmx_profiles.values())

    _set_style()
    fig, axes = plt.subplots(2, 2, constrained_layout=True)
    fig.suptitle("LMX mapped-pipe reference comparison", fontsize=16)
    colors = {"center": "#0f766e", "negative": "#b45309", "positive": "#1d4ed8"}
    labels = {"center": "Center line", "negative": "Negative x offset", "positive": "Positive x offset"}
    summary_profiles: dict[str, dict[str, float]] = {}

    for ax, name in zip(axes.ravel()[:3], ("center", "negative", "positive"), strict=False):
        ref_coord, ref_velocity_raw, offset = reference_profiles[name]
        lmx_coord, lmx_velocity_raw = lmx_profiles[name]
        ref_velocity = ref_velocity_raw / reference_velocity_scale
        lmx_velocity = lmx_velocity_raw / lmx_velocity_scale
        interp_velocity = np.interp(ref_coord, lmx_coord, lmx_velocity)
        l2_error = float(np.sqrt(np.mean((interp_velocity - ref_velocity) ** 2)))
        linf_error = float(np.max(np.abs(interp_velocity - ref_velocity)))
        summary_profiles[name] = {
            "x_offset_fraction": float(offset),
            "normalized_l2_error": l2_error,
            "normalized_linf_error": linf_error,
        }
        ax.plot(ref_coord, ref_velocity, color=colors[name], linewidth=2.2, label="External reference")
        ax.plot(lmx_coord, lmx_velocity, color="#111827", linestyle="--", linewidth=2.0, label="LMX")
        ax.set_title(f"{labels[name]}\n$L_2$={l2_error:.3f} | $L_\\infty$={linf_error:.3f}")
        ax.set_xlabel("Normalized transverse coordinate")
        ax.set_ylabel("Normalized axial velocity")
        ax.set_xlim(-1.02, 1.02)
        ax.legend(frameon=False)

    x = np.asarray(solution.bundle.x, dtype=float)
    pressure_span = np.max(np.asarray(solution.bundle.p), axis=(1, 2)) - np.min(np.asarray(solution.bundle.p), axis=(1, 2))
    axes[1, 1].plot(x, np.asarray(solution.bundle.field_scale), color="#7c3aed", label="Field scale")
    axes[1, 1].plot(x, np.asarray(solution.bundle.mean_velocity), color="#0f766e", label="Mean velocity")
    axes[1, 1].plot(x, pressure_span, color="#b45309", label="Pressure span")
    axes[1, 1].semilogy(
        x,
        np.maximum(np.asarray(solution.bundle.charge_balance_residual), 1.0e-16),
        color="#dc2626",
        linestyle="--",
        label="Charge balance",
    )
    axes[1, 1].set_title("LMX pipe response and conservation")
    axes[1, 1].set_xlabel("x")
    axes[1, 1].legend(frameon=False)

    png_path = out_dir / "pipe_reference_comparison.png"
    pdf_path = out_dir / "pipe_reference_comparison.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "case": "pipe_reference_comparison_demo",
        "geometry_kind": "pipe_ogrid",
        "velocity_normalization": "shared_peak_by_dataset",
        "notes": (
            "The external reference profile comes from a fringing-pipe benchmark dataset. "
            "All three transverse lines are normalized by one shared peak velocity per dataset "
            "so center and off-center errors can be compared on the same scale."
        ),
        "profiles": summary_profiles,
        "validation": {
            "max_charge_balance_residual": solution.validation.max_charge_balance_residual,
            "max_wall_current_leakage": solution.validation.max_wall_current_leakage,
            "net_boundary_current_residual": solution.validation.net_boundary_current_residual,
            "peak_velocity_span": solution.validation.peak_velocity_span,
            "pressure_span_range": solution.validation.pressure_span_range,
        },
        "plots": [png_path.name, pdf_path.name],
    }
    (out_dir / "pipe_reference_comparison_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare the LMX mapped-pipe slice against external fringing-pipe reference profiles.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/examples/pipe_reference_comparison"))
    parser.add_argument("--ha-peak", type=float, default=20.0)
    parser.add_argument("--nr", type=int, default=18)
    parser.add_argument("--ntheta", type=int, default=48)
    parser.add_argument("--nx-stations", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=18)
    parser.add_argument("--coupling-iterations", type=int, default=10)
    parser.add_argument("--potential-iterations", type=int, default=60)
    parser.add_argument("--reference-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    run_pipe_reference_comparison_demo(
        out_dir=args.output,
        ha_peak=args.ha_peak,
        nr=args.nr,
        ntheta=args.ntheta,
        nx_stations=args.nx_stations,
        max_steps=args.max_steps,
        coupling_iterations=args.coupling_iterations,
        potential_iterations=args.potential_iterations,
        reference_dir=args.reference_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

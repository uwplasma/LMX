from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_manual_solver_family_validation import main as run_manual_validation_main


def _set_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.figsize": (13.5, 7.5),
            "axes.grid": True,
            "grid.alpha": 0.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
        }
    )


def run_extruded_validation_campaign(
    *,
    out_dir: Path,
    ha_values: str = "10,20",
    resolutions: str = "10,14,18",
    fringing_nx: int = 7,
    fringing_geometries: str = "rect_duct,layered_duct,pipe_ogrid",
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "solver_family_summary.json"
    exit_code = run_manual_validation_main(
        [
            "--output",
            str(summary_path),
            "--ha-values",
            ha_values,
            "--resolutions",
            resolutions,
            "--include-fringing",
            "--fringing-geometries",
            fringing_geometries,
            "--fringing-nx",
            str(fringing_nx),
            "--max-steps",
            "10",
            "--potential-iterations",
            "40",
            "--coupling-iterations",
            "5",
            "--write-csv",
            "--write-plot",
            "--fail-on-threshold",
        ]
    )
    if exit_code != 0:
        raise RuntimeError(f"manual validation campaign failed with exit code {exit_code}")
    payload = json.loads(summary_path.read_text())
    fringing_rows = [
        value
        for value in payload.values()
        if value.get("solver_kind") == "extruded_inductionless"
    ]
    grouped: dict[tuple[str, float], list[dict[str, float | str]]] = {}
    for row in fringing_rows:
        grouped.setdefault((str(row["geometry_kind"]), float(row["ha"])), []).append(row)

    _set_style()
    fig, axes = plt.subplots(2, 2, constrained_layout=True)
    fig.suptitle("LMX larger 3D validation campaign", fontsize=16)
    palette = {
        "rect_duct": "#0f766e",
        "layered_duct": "#b45309",
        "pipe_ogrid": "#1d4ed8",
    }
    line_styles = {10.0: "-", 20.0: "--", 30.0: ":", 40.0: "-."}
    for (geometry_kind, ha_value), rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda row: (float(row["ha"]), float(row["resolution"])))
        resolution = [float(row["resolution"]) for row in rows]
        charge = [max(float(row["max_charge_balance_residual"]), 1.0e-16) for row in rows]
        axial_current_span = [max(float(row.get("axial_current_span", 0.0)), 1.0e-16) for row in rows]
        peak_velocity_span = [max(float(row.get("peak_velocity_span", 0.0)), 1.0e-16) for row in rows]
        pressure_span_range = [max(float(row.get("pressure_span_range", 0.0)), 1.0e-16) for row in rows]
        color = palette.get(geometry_kind, "#7c3aed")
        label = f"{geometry_kind}, Ha={int(ha_value)}"
        style = line_styles.get(ha_value, "-")
        axes[0, 0].semilogy(resolution, charge, marker="o", linewidth=2.0, linestyle=style, color=color, label=label)
        axes[0, 1].semilogy(resolution, axial_current_span, marker="o", linewidth=2.0, linestyle=style, color=color, label=label)
        axes[1, 0].semilogy(resolution, peak_velocity_span, marker="o", linewidth=2.0, linestyle=style, color=color, label=label)
        axes[1, 1].semilogy(resolution, pressure_span_range, marker="o", linewidth=2.0, linestyle=style, color=color, label=label)
    axes[0, 0].set_title("Charge-balance residual")
    axes[0, 0].set_xlabel("Cross-section resolution")
    axes[0, 0].set_ylabel(r"$\max |\nabla \cdot J|$")
    axes[0, 1].set_title("Axial-current span")
    axes[0, 1].set_xlabel("Cross-section resolution")
    axes[0, 1].set_ylabel(r"$\Delta \int J_x\,dA$")
    axes[1, 0].set_title("Peak-velocity span")
    axes[1, 0].set_xlabel("Cross-section resolution")
    axes[1, 0].set_ylabel(r"$\Delta u_{max}$")
    axes[1, 1].set_title("Pressure-span range")
    axes[1, 1].set_xlabel("Cross-section resolution")
    axes[1, 1].set_ylabel(r"$\Delta (\max p - \min p)$")
    axes[0, 0].legend(frameon=False, ncols=2)

    png_path = out_dir / "extruded_validation_campaign.png"
    pdf_path = out_dir / "extruded_validation_campaign.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "summary_json": str(summary_path),
        "summary_csv": str(summary_path.with_suffix(".csv")),
        "default_fringing_plot": str(summary_path.with_name(f"{summary_path.stem}_fringing.png")),
        "summary_plot": str(png_path),
        "ha_values": ha_values,
        "resolutions": resolutions,
        "fringing_nx": fringing_nx,
        "fringing_geometries": fringing_geometries,
    }
    (out_dir / "extruded_validation_campaign_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the larger LMX extruded 3D validation campaign.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/examples/extruded_validation_campaign"))
    parser.add_argument("--ha-values", type=str, default="10,20")
    parser.add_argument("--resolutions", type=str, default="10,14,18")
    parser.add_argument("--fringing-nx", type=int, default=7)
    parser.add_argument("--fringing-geometries", type=str, default="rect_duct,layered_duct,pipe_ogrid")
    args = parser.parse_args(argv)
    run_extruded_validation_campaign(
        out_dir=args.output,
        ha_values=args.ha_values,
        resolutions=args.resolutions,
        fringing_nx=args.fringing_nx,
        fringing_geometries=args.fringing_geometries,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

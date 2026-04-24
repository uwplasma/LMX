from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from examples import freemhd_closed_channel_observable_parity as observable
from lmx.freemhd import summarize_observable_ladder_levels, write_observable_ladder_table


OUTPUT_DIR = Path("artifacts/examples/freemhd_observable_mesh_ladder")
L2_TARGET = 1.0e-2
DRIVE_MODE = "pressure_gradient"
INITIAL_PROFILE = "analytic"
LADDER = (
    {
        "label": "retained_49x37",
        "case_settings": {
            "shercliff": {
                "ny": 49,
                "nz": 37,
                "max_steps": 64,
                "current_reconstruction": "face_averaged",
                "velocity_update_limit": 0.1,
            },
            "hunt": {
                "ny": 49,
                "nz": 37,
                "max_steps": 64,
                "current_reconstruction": "face_averaged",
                "velocity_update_limit": 0.1,
            },
        },
    },
    {
        "label": "side_refined_57x43",
        "case_settings": {
            "shercliff": {
                "ny": 57,
                "nz": 43,
                "max_steps": 64,
                "current_reconstruction": "face_averaged",
                "velocity_update_limit": 0.1,
            },
            "hunt": {
                "ny": 57,
                "nz": 43,
                "max_steps": 64,
                "current_reconstruction": "face_averaged",
                "velocity_update_limit": 0.1,
            },
        },
    },
)


def _write_ladder_plot(summary: dict[str, object], out_dir: Path) -> list[Path]:
    rows = list(summary.get("rows", []))
    labels = [str(row["label"]) for row in rows]
    x = range(len(rows))
    max_errors = [float(row["max_offender_l2_error"]) for row in rows]
    offender_counts = [int(row["observable_offender_count"]) for row in rows]
    hartmann_ratios = [float(row["min_hartmann_layer_cell_ratio"]) for row in rows]
    side_ratios = [float(row["min_side_layer_cell_ratio"]) for row in rows]

    plt.style.use("default")
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0), constrained_layout=True)
    axes[0].plot(list(x), max_errors, marker="o", color="#1f4e79")
    axes[0].axhline(float(summary["l2_target"]), color="#b91c1c", linestyle="--", linewidth=1.2, label="target")
    axes[0].set_title("Worst over-target cut")
    axes[0].set_ylabel("normalized L2")
    axes[0].legend(loc="best", frameon=False)

    axes[1].bar(list(x), offender_counts, color="#7c2d12")
    axes[1].set_title("Observable offenders")
    axes[1].set_ylabel("count")

    axes[2].plot(list(x), hartmann_ratios, marker="o", label="Hartmann", color="#0f766e")
    axes[2].plot(list(x), side_ratios, marker="s", label="side", color="#7c3aed")
    axes[2].axhline(1.0, color="#404040", linestyle="--", linewidth=1.0)
    axes[2].set_title("Layer-cell readiness")
    axes[2].set_ylabel("cells / required cells")
    axes[2].legend(loc="best", frameon=False)

    for ax in axes:
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.grid(True, color="#e5e7eb", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("LMX FreeMHD observable mesh ladder", fontsize=14, fontweight="bold")
    png_path = out_dir / "freemhd_observable_mesh_ladder.png"
    pdf_path = out_dir / "freemhd_observable_mesh_ladder.pdf"
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)
    return [png_path, pdf_path]


def run_freemhd_observable_mesh_ladder(
    *,
    out_dir: Path | None = None,
    ladder: tuple[dict[str, object], ...] = LADDER,
) -> dict[str, object]:
    if out_dir is None:
        out_dir = OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    levels: list[dict[str, object]] = []
    for level in ladder:
        label = str(level["label"])
        case_settings = level["case_settings"]
        records = [
            observable._observable_record(
                case_kind,
                drive_mode=DRIVE_MODE,
                initial_profile=INITIAL_PROFILE,
                case_settings=case_settings,  # type: ignore[arg-type]
            )
            for case_kind in ("shercliff", "hunt")
        ]
        levels.append({"label": label, "case_settings": case_settings, "records": records})

    ladder_summary = summarize_observable_ladder_levels(levels, l2_target=L2_TARGET)
    table_path = write_observable_ladder_table(ladder_summary, out_dir / "freemhd_observable_mesh_ladder.csv")
    plots = _write_ladder_plot(ladder_summary, out_dir)
    summary = {
        "case": "freemhd_observable_mesh_ladder",
        "ha": observable.HA,
        "x_slice": observable.X_SLICE,
        "drive_mode": DRIVE_MODE,
        "initial_profile": INITIAL_PROFILE,
        "l2_target": L2_TARGET,
        "ladder": levels,
        "ladder_summary": ladder_summary,
        "best_level_label": ladder_summary["best_level_label"],
        "table": table_path.name,
        "plots": [path.name for path in plots],
    }
    (out_dir / "freemhd_observable_mesh_ladder_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    run_freemhd_observable_mesh_ladder()

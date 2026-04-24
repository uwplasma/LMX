from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples import freemhd_closed_channel_observable_parity as observable
from lmx import write_freemhd_observable_parity_plots
from lmx.freemhd import summarize_observable_offenders


OUTPUT_DIR = Path("artifacts/examples/freemhd_closed_channel_flow_rate_parity")
DRIVE_MODE = "flow_rate"
INITIAL_PROFILE = "analytic"
FLOW_RATE_TARGET_MEAN_VELOCITY: float | None = None
CASE_SETTINGS = observable.CASE_SETTINGS


def run_freemhd_closed_channel_flow_rate_parity(
    *,
    out_dir: Path | None = None,
) -> dict[str, object]:
    if out_dir is None:
        out_dir = OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    records = [
        observable._observable_record(
            case_kind,
            drive_mode=DRIVE_MODE,
            initial_profile=INITIAL_PROFILE,
            flow_rate_target_mean_velocity=FLOW_RATE_TARGET_MEAN_VELOCITY,
            case_settings=CASE_SETTINGS,
        )
        for case_kind in ("shercliff", "hunt")
    ]
    plots = write_freemhd_observable_parity_plots(
        records,
        out_dir,
        case_title=f"LMX vs FreeMHD constant-flow observables (Ha={observable.HA})",
        output_stem="freemhd_closed_channel_flow_rate_parity",
    )
    summary = {
        "case": "freemhd_closed_channel_flow_rate_parity",
        "ha": observable.HA,
        "x_slice": observable.X_SLICE,
        "initial_profile": INITIAL_PROFILE,
        "drive_mode": DRIVE_MODE,
        "target_mean_velocity": FLOW_RATE_TARGET_MEAN_VELOCITY,
        "target_mean_velocity_by_case": {str(record["case_kind"]): record["target_mean_velocity"] for record in records},
        "target_mean_velocity_source": (
            "processed_slice_area_mean" if FLOW_RATE_TARGET_MEAN_VELOCITY is None else "configured"
        ),
        "settings": CASE_SETTINGS,
        "records": records,
        "top_observable_offenders": summarize_observable_offenders(records, l2_target=1.0e-2, top_n=8),
        "plots": [path.name for path in plots],
    }
    (out_dir / "freemhd_closed_channel_flow_rate_parity_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    run_freemhd_closed_channel_flow_rate_parity()

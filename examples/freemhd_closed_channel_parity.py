from __future__ import annotations

import json
import time
from pathlib import Path

from lmx import write_freemhd_parity_plots
from lmx.freemhd import (
    build_case_from_freemhd_reference,
    parse_freemhd_execution_seconds,
    run_freemhd_demo,
)
from lmx.solvers import solve_transient
from lmx.validation import latest_field_minmax_record, reference_profile_validation
from lmx.validation import read_field_minmax


OUTPUT_DIR = Path("artifacts/examples/freemhd_closed_channel_parity")
FREEMHD_INSTALL_DIR = Path("/Users/rogerio/local/tests/freemhd_install")
RERUN_FREEMHD = False
NPROC = 2
NY = 25
NZ = 25
DT = 1.25e-6
T_FINAL = 1.0e-5
MAX_STEPS = 8
CASES = (
    ("shercliff", 20.0),
    ("hunt", 20.0),
)


def _run_case(case_kind: str, ha: float, reference_run_dir: Path) -> dict[str, object]:
    case = build_case_from_freemhd_reference(
        case_kind=case_kind,
        ha=ha,
        ny=NY,
        nz=NZ,
        dt=DT,
        t_final=T_FINAL,
        max_steps=MAX_STEPS,
        reference_run_dir=reference_run_dir,
        forcing=None,
    )
    solve_start = time.perf_counter()
    solution = solve_transient(case)
    lmx_execution_seconds = time.perf_counter() - solve_start
    sample_validation = reference_profile_validation(solution, reference_run_dir)
    latest_u = latest_field_minmax_record(reference_run_dir, field="mag(U)")
    lmx_u_max = float(solution.state.u.max())
    u_max_abs_diff = abs(lmx_u_max - float(latest_u.max_value if latest_u is not None else lmx_u_max))
    freemhd_history_records = []
    for path in reference_run_dir.glob("postProcessing/**/fieldMinMax.dat"):
        freemhd_history_records.extend(record for record in read_field_minmax(path) if record.field == "mag(U)")
    freemhd_history_records.sort(key=lambda record: record.time)
    return {
        "case_kind": case_kind,
        "ha": ha,
        "reference_run_dir": str(reference_run_dir),
        "freemhd_execution_seconds": float(parse_freemhd_execution_seconds(reference_run_dir / "run.log") or 0.0),
        "lmx_execution_seconds": float(lmx_execution_seconds),
        "u_max_abs_diff": float(u_max_abs_diff),
        "y_l2_error": float(sample_validation.y_profile.l2_error),
        "z_l2_error": float(sample_validation.z_profile.l2_error),
        "freemhd_u_max_history": {
            "time": [float(record.time) for record in freemhd_history_records],
            "value": [float(record.max_value) for record in freemhd_history_records],
        },
        "lmx_u_max_history": {
            "time": solution.diagnostics.time_history.tolist(),
            "value": solution.diagnostics.u_max_history.tolist(),
        },
        "y_profile": {
            "coordinate": sample_validation.y_profile.coordinate.tolist(),
            "simulated": sample_validation.y_profile.simulated.tolist(),
            "reference": sample_validation.y_profile.reference.tolist(),
        },
        "z_profile": {
            "coordinate": sample_validation.z_profile.coordinate.tolist(),
            "simulated": sample_validation.z_profile.simulated.tolist(),
            "reference": sample_validation.z_profile.reference.tolist(),
        },
    }


def run_freemhd_closed_channel_parity() -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if RERUN_FREEMHD:
        for case_kind, _ in CASES:
            run_freemhd_demo(FREEMHD_INSTALL_DIR, demo_kind=case_kind, nproc=NPROC)
    records = []
    for case_kind, ha in CASES:
        reference_run_dir = FREEMHD_INSTALL_DIR / "freemhd_output" / case_kind
        records.append(_run_case(case_kind, ha, reference_run_dir))
    plots = write_freemhd_parity_plots(records, OUTPUT_DIR, case_title="LMX vs FreeMHD closed-channel parity")
    summary = {
        "case": "freemhd_closed_channel_parity",
        "records": records,
        "plots": [path.name for path in plots],
        "reran_freemhd": RERUN_FREEMHD,
        "nproc": NPROC,
        "resolution": {"ny": NY, "nz": NZ},
        "time_controls": {"dt": DT, "t_final": T_FINAL, "max_steps": MAX_STEPS},
    }
    (OUTPUT_DIR / "freemhd_closed_channel_parity_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    run_freemhd_closed_channel_parity()

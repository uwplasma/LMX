"""Run a small fixed-work CPU scaling calibration.

This example isolates each device count in a fresh process so JAX sees the
requested CPU topology before it initializes.  It is deliberately small and
interactive: the output demonstrates placement, timing, and report generation,
but it is not publishable strong-scaling evidence.  The monitored, multi-minute
B2 campaign lives in ``scripts/run_strong_scaling_worker.py --campaign``.

Runtime: about one minute on a four-device development machine.
Outputs: per-rung JSON, a CSV table, a compact summary, and optional plots.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from lmx.plotting import write_strong_scaling_plots
from lmx.scaling import (
    summarize_strong_scaling_records,
    write_strong_scaling_summary_table,
)

# Inputs: edit this compact block to change the fixed global workload.
DEVICE_COUNTS = (1, 2, 4)  # Compare the same problem on these CPU shard counts.
GRID = (96, 24, 24)  # Axial size must divide every requested device count.
ITERATIONS = 16  # Increase only for a more stable, longer calibration.
REPEATS = 2  # One compilation run followed by one warm measurement.
OUTPUT_DIR = Path("artifacts/examples/strong_scaling")

# Run each topology in a clean process; compilation is excluded from warm timing.
repo_root = Path(__file__).resolve().parents[1]
worker = repo_root / "scripts" / "run_strong_scaling_worker.py"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
records: list[dict[str, object]] = []

for count in DEVICE_COUNTS:
    if GRID[0] % count:
        raise ValueError(f"Axial grid size {GRID[0]} is not divisible by {count}")

    # Keep unrelated XLA flags while selecting the requested local CPU topology.
    environment = os.environ.copy()
    flags = [
        flag
        for flag in shlex.split(environment.get("XLA_FLAGS", ""))
        if not flag.startswith("--xla_force_host_platform_device_count=")
    ]
    flags.append(f"--xla_force_host_platform_device_count={count}")
    environment.update(
        JAX_PLATFORMS="cpu",
        JAX_ENABLE_X64="true",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        XLA_FLAGS=" ".join(flags),
    )

    record_path = OUTPUT_DIR / f"cpu_{count}.json"
    command = [
        sys.executable,
        str(worker),
        "--benchmark-kind",
        "extruded3d",
        "--platform",
        "CPU",
        "--num-devices",
        str(count),
        "--nx",
        str(GRID[0]),
        "--ny",
        str(GRID[1]),
        "--nz",
        str(GRID[2]),
        "--iterations",
        str(ITERATIONS),
        "--repeats",
        str(REPEATS),
        "--output",
        str(record_path),
    ]
    subprocess.run(command, cwd=repo_root, env=environment, check=True)
    records.append(json.loads(record_path.read_text()))

# Report speedup and placement without promoting the short run as evidence.
diagnostics = summarize_strong_scaling_records(records)
table = write_strong_scaling_summary_table(
    records, OUTPUT_DIR / "strong_scaling_table.csv"
)
try:
    plots = write_strong_scaling_plots(
        records, OUTPUT_DIR, case_title="LMX local fixed-work calibration"
    )
except ModuleNotFoundError as error:
    if error.name != "matplotlib":
        raise
    plots = []

summary = {
    "classification": "debug-or-calibration",
    "records": [f"cpu_{count}.json" for count in DEVICE_COUNTS],
    "table": table.name,
    "plots": [path.name for path in plots],
    "diagnostics": diagnostics,
}
(OUTPUT_DIR / "strong_scaling_summary.json").write_text(
    json.dumps(summary, indent=2) + "\n"
)
print(
    f"Wrote {len(records)} calibration rungs to {OUTPUT_DIR}; "
    f"best speedup={diagnostics['best_speedup']:.3f}x"
)

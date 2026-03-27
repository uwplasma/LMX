from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import jax

from .cases import make_hartmann_case
from .solvers import solve_steady


def benchmark_solver(repeats: int = 3, ha: float = 20.0, ny: int = 48, nz: int = 48) -> dict[str, float | str]:
    case = make_hartmann_case(ha=ha, ny=ny, nz=nz)
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        solve_steady(case)
        timings.append(time.perf_counter() - start)
    cold = timings[0]
    warm = min(timings[1:] or timings)
    return {
        "case": case.name,
        "ha": ha,
        "ny": float(ny),
        "nz": float(nz),
        "repeats": float(repeats),
        "cold_seconds": cold,
        "warm_seconds": warm,
        "mean_seconds": sum(timings) / len(timings),
        "backend": jax.default_backend(),
        "device_kind": jax.devices()[0].device_kind,
        "jax_version": jax.__version__,
        "python_version": platform.python_version(),
    }


def write_benchmark_report(report: dict[str, float | str], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    return path

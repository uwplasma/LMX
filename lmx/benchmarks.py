from __future__ import annotations

import time

from .cases import make_hartmann_case
from .solvers import solve_steady


def benchmark_solver(repeats: int = 3) -> dict[str, float]:
    case = make_hartmann_case(ha=20.0, ny=48, nz=48)
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        solve_steady(case)
        timings.append(time.perf_counter() - start)
    cold = timings[0]
    warm = min(timings[1:] or timings)
    return {"cold_seconds": cold, "warm_seconds": warm, "repeats": float(repeats)}

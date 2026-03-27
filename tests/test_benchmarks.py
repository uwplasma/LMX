from pathlib import Path

import pytest

from lmx.benchmarks import benchmark_solver, write_benchmark_report


pytestmark = pytest.mark.unit


def test_benchmark_solver_returns_positive_timings():
    report = benchmark_solver(repeats=2, ha=5.0, ny=16, nz=16)
    assert float(report["cold_seconds"]) > 0.0
    assert float(report["warm_seconds"]) > 0.0
    assert report["backend"]


def test_benchmark_writer(tmp_path: Path):
    path = write_benchmark_report({"cold_seconds": 1.0, "warm_seconds": 0.5}, tmp_path / "benchmark.json")
    assert path.exists()

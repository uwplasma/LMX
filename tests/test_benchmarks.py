from pathlib import Path
from types import SimpleNamespace

import pytest

import lmx.benchmarks as benchmarks
from lmx.benchmarks import benchmark_solver, write_benchmark_report


pytestmark = pytest.mark.unit


def test_benchmark_solver_returns_positive_timings(monkeypatch: pytest.MonkeyPatch):
    times = iter([10.0, 10.4, 10.4, 10.7])

    monkeypatch.setattr(benchmarks, "make_hartmann_case", lambda ha, ny, nz: SimpleNamespace(name="hartmann_ha5"))
    monkeypatch.setattr(benchmarks, "solve_steady", lambda case: SimpleNamespace())
    monkeypatch.setattr(benchmarks.time, "perf_counter", lambda: next(times))
    monkeypatch.setattr(benchmarks.jax, "default_backend", lambda: "cpu")
    monkeypatch.setattr(benchmarks.jax, "devices", lambda: [SimpleNamespace(device_kind="cpu")])
    monkeypatch.setattr(benchmarks.platform, "python_version", lambda: "3.13.7")

    report = benchmark_solver(repeats=2, ha=5.0, ny=16, nz=16)
    assert float(report["cold_seconds"]) > 0.0
    assert float(report["warm_seconds"]) > 0.0
    assert report["backend"]


def test_benchmark_writer(tmp_path: Path):
    path = write_benchmark_report({"cold_seconds": 1.0, "warm_seconds": 0.5}, tmp_path / "benchmark.json")
    assert path.exists()

from pathlib import Path
import subprocess

import jax
import pytest

from lmx.scaling import benchmark_sharded_extruded_operator, benchmark_sharded_stencil, write_scaling_report
from examples.strong_scaling_demo import _default_visible_devices


pytestmark = pytest.mark.unit


def test_benchmark_sharded_stencil_runs_on_single_device():
    record = benchmark_sharded_stencil(ny=32, nz=32, iterations=4, repeats=1, num_devices=1)

    assert record.num_devices == 1
    assert record.mean_seconds >= 0.0
    assert record.ny == 32
    assert record.nz == 32


def test_write_scaling_report_writes_json(tmp_path: Path):
    record = benchmark_sharded_stencil(ny=16, nz=16, iterations=2, repeats=1, num_devices=1)
    path = write_scaling_report([record], tmp_path / "scaling.json")

    assert path.exists()
    assert '"num_devices": 1' in path.read_text()


def test_benchmark_sharded_stencil_rejects_invalid_device_count():
    with pytest.raises(ValueError):
        benchmark_sharded_stencil(ny=18, nz=16, iterations=1, repeats=1, num_devices=max(2, len(jax.devices()) + 1))


def test_benchmark_sharded_extruded_operator_runs_on_single_device():
    record = benchmark_sharded_extruded_operator(nx=16, ny=12, nz=10, iterations=2, repeats=1, num_devices=1)

    assert record.num_devices == 1
    assert record.nx == 16
    assert record.ny == 12
    assert record.nz == 10
    assert record.benchmark_kind == "extruded3d"


def test_benchmark_sharded_extruded_operator_rejects_invalid_device_count():
    with pytest.raises(ValueError):
        benchmark_sharded_extruded_operator(nx=16, ny=12, nz=10, iterations=1, repeats=1, num_devices=max(2, len(jax.devices()) + 1))


def test_default_visible_devices_uses_highest_indices(monkeypatch: pytest.MonkeyPatch):
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="0\n1\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert _default_visible_devices("office", 1) == "1"
    assert _default_visible_devices("office", 2) == "0,1"

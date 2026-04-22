from pathlib import Path
import subprocess

import jax
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding

import lmx.scaling as scaling
from lmx.scaling import (
    _build_extruded_operator_problem,
    _build_operator_problem,
    _factor_device_mesh,
    _row_or_replicated_sharding,
    _two_axis_mesh_and_sharding,
    benchmark_sharded_extruded_operator,
    benchmark_sharded_stencil,
    write_scaling_report,
)
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


def test_scaling_problem_builders_return_expected_shapes():
    field, potential, forcing = _build_operator_problem(12, 10)
    assert field.shape == (12, 10)
    assert potential.shape == (12, 10)
    assert forcing.shape == (12, 10)
    assert field.dtype == np.float32

    u, v, w, phi, src, sigma = _build_extruded_operator_problem(8, 6, 4)
    assert u.shape == (8, 6, 4)
    assert v.shape == (8, 6, 4)
    assert w.shape == (8, 6, 4)
    assert phi.shape == (8, 6, 4)
    assert src.shape == (8, 6, 4)
    assert sigma.shape == (8, 6, 4)


def test_factor_device_mesh_prefers_near_square_factoring():
    assert _factor_device_mesh(1) == (1, 1)
    assert _factor_device_mesh(4) == (2, 2)
    assert _factor_device_mesh(6) == (2, 3)


def test_row_or_replicated_sharding_covers_row_and_replicated_paths():
    devices = np.asarray(jax.devices()[:1], dtype=object)
    mesh = Mesh(devices, ("d",))

    row = _row_or_replicated_sharding(mesh, (4, 3), 1)
    repl = _row_or_replicated_sharding(mesh, (), 1)

    assert isinstance(row, NamedSharding)
    assert isinstance(repl, NamedSharding)


def test_two_axis_mesh_and_sharding_covers_elongated_and_replicated_paths():
    devices = list(jax.devices()[:1])

    mesh1, sharding1 = _two_axis_mesh_and_sharding(devices, num_devices=1, shape=(16, 4, 4))
    mesh2, sharding2 = _two_axis_mesh_and_sharding(devices, num_devices=1, shape=(5, 3))

    assert isinstance(mesh1, Mesh)
    assert isinstance(sharding1, NamedSharding)
    assert isinstance(mesh2, Mesh)
    assert isinstance(sharding2, NamedSharding)


def test_two_axis_mesh_and_sharding_rejects_incompatible_multi_axis_shape(monkeypatch: pytest.MonkeyPatch):
    class FakeMesh:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class FakeSharding:
        def __init__(self, mesh, spec):
            self.mesh = mesh
            self.spec = spec

    monkeypatch.setattr(scaling, "Mesh", FakeMesh)
    monkeypatch.setattr(scaling, "NamedSharding", FakeSharding)

    with pytest.raises(ValueError, match="not compatible"):
        _two_axis_mesh_and_sharding([object(), object(), object(), object()], num_devices=4, shape=(3, 5, 2))


def test_benchmark_sharded_stencil_rejects_missing_devices(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(jax, "devices", lambda: [])
    with pytest.raises(RuntimeError, match="No JAX devices"):
        benchmark_sharded_stencil(ny=8, nz=8, iterations=1, repeats=1, num_devices=1)


def test_benchmark_sharded_extruded_operator_rejects_missing_devices(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(jax, "devices", lambda: [])
    with pytest.raises(RuntimeError, match="No JAX devices"):
        benchmark_sharded_extruded_operator(nx=8, ny=8, nz=8, iterations=1, repeats=1, num_devices=1)

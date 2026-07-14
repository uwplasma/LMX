from pathlib import Path
import json
import subprocess
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding

import lmx.scaling as scaling
from examples import strong_scaling_demo
from lmx.scaling import (
    StrongScalingRecord,
    _array_nbytes,
    _build_extruded_operator_problem,
    _factor_device_mesh,
    _float_or_none,
    _int_or_none,
    _row_or_replicated_sharding,
    _shard_placement,
    _two_axis_mesh_and_sharding,
    benchmark_extruded_inductionless_solve,
    benchmark_sharded_extruded_operator,
    summarize_strong_scaling_records,
    write_scaling_report,
    write_strong_scaling_summary_table,
)
from examples.strong_scaling_demo import _default_visible_devices
from scripts import run_strong_scaling_worker


pytestmark = pytest.mark.unit


def test_scaling_demo_requires_restart_for_production() -> None:
    with pytest.raises(SystemExit):
        strong_scaling_demo.main(["--benchmark-kind", "extruded_solve"])


def test_scaling_worker_command_forwards_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "record.json"
    output.write_text("{}")
    commands: list[list[str]] = []
    monkeypatch.setattr(
        strong_scaling_demo.subprocess,
        "run",
        lambda command, **kwargs: commands.append(command),
    )

    strong_scaling_demo._run_worker(
        python_executable="python",
        repo_root=tmp_path,
        output_path=output,
        platform="CPU",
        benchmark_kind="extruded_solve",
        nx=8,
        num_devices=2,
        ny=6,
        nz=4,
        iterations=3,
        repeats=1,
        restart_path=tmp_path / "restart.npz",
    )

    assert commands[0][-2:] == ["--restart", str(tmp_path / "restart.npz")]


def test_write_scaling_report_writes_json(tmp_path: Path):
    record = benchmark_sharded_extruded_operator(
        nx=16, ny=8, nz=8, iterations=2, repeats=1, num_devices=1
    )
    path = write_scaling_report([record], tmp_path / "scaling.json")

    assert path.exists()
    assert '"num_devices": 1' in path.read_text()


def test_strong_scaling_summary_table_computes_solver_diagnostics(tmp_path: Path):
    baseline = StrongScalingRecord(
        backend="cpu",
        device_kind="cpu",
        num_devices=1,
        nx=8,
        ny=6,
        nz=4,
        iterations=5,
        repeats=2,
        cold_seconds=11.0,
        warm_seconds=10.0,
        mean_seconds=10.5,
        python_version="3.12",
        jax_version="0.test",
        benchmark_kind="extruded_solve",
        operator_path="solve_extruded_inductionless",
        total_cells=192,
        cell_updates=960,
        warm_cell_updates_per_second=96.0,
        memory_bytes_estimate=2 * 1024 * 1024,
        profile_path="profiles/cpu_1",
        velocity_l2=3.0,
        potential_l2=2.0,
        current_l2=1.0,
        validation_passed=True,
    )
    two_device = {
        **baseline.__dict__,
        "num_devices": 2,
        "cold_seconds": 6.0,
        "warm_seconds": 5.0,
        "mean_seconds": 5.5,
        "profile_path": None,
    }

    summary = summarize_strong_scaling_records([baseline, two_device])
    table = write_strong_scaling_summary_table(
        [baseline, two_device], tmp_path / "strong_scaling_table.csv"
    )

    assert summary["validation_status"] == "solver_faithful_records_present"
    assert summary["solver_faithful_record_count"] == 2
    assert summary["profiled_record_count"] == 1
    assert summary["physics_equivalent_record_count"] == 2
    assert summary["best_speedup"] == pytest.approx(2.0)
    rows = summary["rows"]
    assert rows[0]["speedup"] == pytest.approx(1.0)
    assert rows[1]["speedup"] == pytest.approx(2.0)
    assert rows[1]["parallel_efficiency"] == pytest.approx(1.0)
    assert rows[0]["warm_mcell_updates_per_second"] == pytest.approx(9.6e-5)
    assert rows[0]["memory_mib"] == pytest.approx(2.0)
    assert rows[1]["physics_equivalent"]
    assert table.exists()
    assert "parallel_efficiency" in table.read_text()


def test_benchmark_sharded_extruded_operator_runs_on_single_device():
    record = benchmark_sharded_extruded_operator(
        nx=16, ny=12, nz=10, iterations=2, repeats=1, num_devices=1
    )

    assert record.num_devices == 1
    assert record.nx == 16
    assert record.ny == 12
    assert record.nz == 10
    assert record.benchmark_kind == "extruded3d"
    assert record.operator_path == "sharded_extruded_operator_surrogate"
    assert record.total_cells == 16 * 12 * 10
    assert record.spatially_sharded is False
    assert record.global_shard_count == 1


def test_shard_placement_reports_partitioning_and_rejects_replication():
    single = SimpleNamespace(
        global_shards=[object()],
        sharding=SimpleNamespace(is_fully_replicated=True),
    )
    distributed = SimpleNamespace(
        global_shards=[object(), object()],
        sharding=SimpleNamespace(is_fully_replicated=False),
    )
    replicated = SimpleNamespace(
        global_shards=[object(), object()],
        sharding=SimpleNamespace(is_fully_replicated=True),
    )

    assert _shard_placement(single, 1) == (False, 1)
    assert _shard_placement(distributed, 2) == (True, 2)
    with pytest.raises(RuntimeError, match="not spatially partitioned"):
        _shard_placement(replicated, 2)
    with pytest.raises(RuntimeError, match="global_shards=2"):
        _shard_placement(distributed, 4)


def test_tracked_mac_sharding_record_reports_only_actual_partitions():
    payload = json.loads(
        Path("benchmarks/results/mac-cpu-sharding-20260714.json").read_text()
    )
    points = payload["points"]

    assert all(payload["gates"].values())
    assert all(point["global_shards"] == point["devices"] for point in points)
    assert all(point["spatially_sharded"] for point in points[1:])
    assert payload["interpretation"]["fastest_device_count"] == 4
    assert payload["interpretation"]["production_solver_claim"] is False


def test_benchmark_sharded_extruded_operator_rejects_invalid_device_count():
    with pytest.raises(ValueError):
        benchmark_sharded_extruded_operator(
            nx=16,
            ny=12,
            nz=10,
            iterations=1,
            repeats=1,
            num_devices=max(2, len(jax.devices()) + 1),
        )


def test_benchmark_extruded_inductionless_solve_records_solver_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    calls = []
    restart_path = tmp_path / "steady.npz"
    restart_path.write_bytes(b"restart")
    initial_bundle = object()

    def fake_solve(problem, *, num_devices=None, initial_bundle=None):
        calls.append(problem)
        assert num_devices == 1
        assert initial_bundle is not None
        shape = (
            problem.case.geometry.nx,
            problem.case.geometry.ny,
            problem.case.geometry.nz,
        )
        bundle = SimpleNamespace(
            u=jnp.ones(shape),
            v=jnp.zeros(shape),
            w=jnp.zeros(shape),
            p=jnp.zeros(shape),
            phi=jnp.ones(shape),
            jx=jnp.ones(shape),
            jy=jnp.zeros(shape),
            jz=jnp.zeros(shape),
            lorentz_x=jnp.zeros(shape),
            lorentz_y=jnp.zeros(shape),
            lorentz_z=jnp.zeros(shape),
            charge_balance_residual=jnp.asarray([1.0e-6]),
            boundary_current_residual=jnp.asarray([2.0e-6]),
            iteration_electric_linear_history=jnp.asarray(
                [[1.0e-8, 1.0e-9, 3.0e-6, 4.0, 1.0, 1.0]]
            ),
            iteration_residual_history=jnp.asarray([5.0e-5, 4.0e-5, 3.0e-5]),
            iteration_component_residual_history=jnp.asarray(
                [[3.0e-5, 0.0, 0.0, 1.0e-6, 0.0, 3.0e-6]] * 3
            ),
        )
        return SimpleNamespace(bundle=bundle)

    monkeypatch.setattr(scaling, "solve_extruded_inductionless", fake_solve)
    monkeypatch.setattr(
        scaling,
        "load_extruded_restart_bundle",
        lambda path: SimpleNamespace(bundle=initial_bundle),
    )
    validated = []
    monkeypatch.setattr(
        scaling,
        "validate_extruded_restart_bundle",
        lambda restart, case: validated.append((restart, case)),
    )

    record = benchmark_extruded_inductionless_solve(
        nx=6,
        ny=5,
        nz=4,
        max_steps=3,
        potential_iterations=2,
        coupling_iterations=1,
        repeats=1,
        num_devices=1,
        restart_path=restart_path,
    )

    assert calls
    assert record.benchmark_kind == "extruded_solve"
    assert record.operator_path == "solve_extruded_inductionless"
    assert record.total_cells == 6 * 5 * 4
    assert record.cell_updates == 6 * 5 * 4 * record.iterations
    assert record.memory_bytes_estimate is not None and record.memory_bytes_estimate > 0
    assert not record.spatially_sharded
    assert record.global_shard_count == 1
    assert record.velocity_l2 is not None
    assert record.validation_passed
    assert record.electric_solves_converged
    assert validated
    assert record.initialization == "restart"
    assert record.restart_sha256 == scaling.hashlib.sha256(b"restart").hexdigest()


def test_solver_scaling_rejects_invalid_devices_and_physics(
    monkeypatch: pytest.MonkeyPatch,
):
    with pytest.raises(ValueError, match="only"):
        benchmark_extruded_inductionless_solve(
            repeats=1, num_devices=len(jax.devices()) + 1
        )

    shape = (4, 4, 4)
    zero = jnp.zeros(shape)
    bundle = SimpleNamespace(
        **{
            name: zero
            for name in (
                "u",
                "v",
                "w",
                "p",
                "phi",
                "jx",
                "jy",
                "jz",
                "lorentz_x",
                "lorentz_y",
                "lorentz_z",
            )
        }
    )
    monkeypatch.setattr(
        scaling,
        "solve_extruded_inductionless",
        lambda problem, **kwargs: SimpleNamespace(bundle=bundle),
    )
    with pytest.raises(RuntimeError, match="physics signature"):
        benchmark_extruded_inductionless_solve(
            nx=4, ny=4, nz=4, repeats=1, num_devices=1
        )


def test_solver_scaling_rejects_failed_conservation(monkeypatch: pytest.MonkeyPatch):
    shape = (4, 4, 4)
    one = jnp.ones(shape)
    zero = jnp.zeros(shape)
    bundle = SimpleNamespace(
        **{
            name: one if name in {"u", "phi", "jx"} else zero
            for name in (
                "u",
                "v",
                "w",
                "p",
                "phi",
                "jx",
                "jy",
                "jz",
                "lorentz_x",
                "lorentz_y",
                "lorentz_z",
            )
        },
        charge_balance_residual=jnp.asarray([2.0e-3]),
        boundary_current_residual=jnp.asarray([0.0]),
        iteration_electric_linear_history=jnp.asarray(
            [[1.0e-8, 1.0e-9, 2.0e-3, 4.0, 0.0, 1.0]]
        ),
    )
    monkeypatch.setattr(
        scaling,
        "solve_extruded_inductionless",
        lambda problem, **kwargs: SimpleNamespace(bundle=bundle),
    )

    with pytest.raises(RuntimeError, match="failed conservation"):
        benchmark_extruded_inductionless_solve(
            nx=4, ny=4, nz=4, repeats=1, num_devices=1
        )


def test_default_visible_devices_uses_highest_indices(monkeypatch: pytest.MonkeyPatch):
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args, returncode=0, stdout="0\n1\n", stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert _default_visible_devices("office", 1) == "1"
    assert _default_visible_devices("office", 2) == "0,1"


def test_scaling_problem_builders_return_expected_shapes():
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
    assert _factor_device_mesh(0) == (1, 0)


def test_scaling_helpers_handle_missing_and_invalid_values():
    class InvalidArray:
        shape = (2,)
        dtype = object()

    assert _float_or_none(None) is None
    assert _float_or_none("not-a-float") is None
    assert _int_or_none(None) is None
    assert _int_or_none("not-an-int") is None
    assert _array_nbytes(InvalidArray()) == 0


def test_row_or_replicated_sharding_covers_row_and_replicated_paths():
    devices = np.asarray(jax.devices()[:1], dtype=object)
    mesh = Mesh(devices, ("d",))

    row = _row_or_replicated_sharding(mesh, (4, 3), 1)
    repl = _row_or_replicated_sharding(mesh, (), 1)

    assert isinstance(row, NamedSharding)
    assert isinstance(repl, NamedSharding)


def test_two_axis_mesh_and_sharding_covers_elongated_and_replicated_paths():
    devices = list(jax.devices()[:1])

    mesh1, sharding1 = _two_axis_mesh_and_sharding(
        devices, num_devices=1, shape=(16, 4, 4)
    )
    mesh2, sharding2 = _two_axis_mesh_and_sharding(devices, num_devices=1, shape=(5, 3))

    assert isinstance(mesh1, Mesh)
    assert isinstance(sharding1, NamedSharding)
    assert isinstance(mesh2, Mesh)
    assert isinstance(sharding2, NamedSharding)


def test_two_axis_mesh_and_sharding_covers_multi_axis_and_flattened_partitions(
    monkeypatch: pytest.MonkeyPatch,
):
    class FakeMesh:
        def __init__(self, devices, axis_names):
            self.devices = devices
            self.axis_names = axis_names

    class FakeSharding:
        def __init__(self, mesh, spec):
            self.mesh = mesh
            self.spec = spec

    monkeypatch.setattr(scaling, "Mesh", FakeMesh)
    monkeypatch.setattr(scaling, "NamedSharding", FakeSharding)

    mesh1, sharding1 = _two_axis_mesh_and_sharding(
        [object(), object(), object(), object()], num_devices=4, shape=(4, 4, 2)
    )
    mesh2, sharding2 = _two_axis_mesh_and_sharding(
        [object(), object(), object(), object()], num_devices=4, shape=(4, 3, 2)
    )

    assert mesh1.axis_names == ("x", "y")
    assert sharding1.spec == scaling.P("x", "y", None)
    assert mesh2.axis_names == ("x", "y")
    assert sharding2.spec == scaling.P(("x", "y"), None, None)


def test_two_axis_mesh_and_sharding_rejects_incompatible_multi_axis_shape(
    monkeypatch: pytest.MonkeyPatch,
):
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
        _two_axis_mesh_and_sharding(
            [object(), object(), object(), object()], num_devices=4, shape=(3, 5, 2)
        )


def test_benchmark_sharded_extruded_operator_rejects_missing_devices(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(jax, "devices", lambda: [])
    with pytest.raises(RuntimeError, match="No JAX devices"):
        benchmark_sharded_extruded_operator(
            nx=8, ny=8, nz=8, iterations=1, repeats=1, num_devices=1
        )


def test_scaling_worker_writes_expected_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    def fake_benchmark_sharded_extruded_operator(**kwargs):
        assert kwargs == {
            "nx": 48,
            "ny": 64,
            "nz": 32,
            "iterations": 5,
            "repeats": 2,
            "num_devices": 1,
        }
        return StrongScalingRecord(
            backend="cpu",
            device_kind="cpu",
            num_devices=1,
            ny=64,
            nz=32,
            iterations=5,
            repeats=2,
            cold_seconds=0.2,
            warm_seconds=0.1,
            mean_seconds=0.15,
            python_version="3.x",
            jax_version="0.x",
            nx=48,
            benchmark_kind="extruded3d",
        )

    monkeypatch.setattr(
        run_strong_scaling_worker,
        "benchmark_sharded_extruded_operator",
        fake_benchmark_sharded_extruded_operator,
    )
    output_path = tmp_path / "worker.json"
    rc = run_strong_scaling_worker.main(
        [
            "--benchmark-kind",
            "extruded3d",
            "--nx",
            "48",
            "--ny",
            "64",
            "--nz",
            "32",
            "--iterations",
            "5",
            "--repeats",
            "2",
            "--num-devices",
            "1",
            "--platform",
            "CPU",
            "--output",
            str(output_path),
        ]
    )

    assert rc == 0
    payload = json.loads(output_path.read_text())
    assert payload["platform"] == "CPU"
    assert payload["warm_seconds"] == 0.1
    assert payload["benchmark_kind"] == "extruded3d"
    assert len(payload["source_fingerprint"]) == 64
    assert json.loads(capsys.readouterr().out)["num_devices"] == 1


def test_scaling_worker_covers_solver_faithful_branch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    def fake_benchmark_extruded_inductionless_solve(**kwargs):
        assert kwargs == {
            "nx": 12,
            "ny": 10,
            "nz": 8,
            "max_steps": 6,
            "repeats": 1,
            "num_devices": 1,
            "profile_dir": tmp_path / "profile",
            "restart_path": tmp_path / "steady.npz",
        }
        return StrongScalingRecord(
            backend="cpu",
            device_kind="cpu",
            num_devices=1,
            ny=10,
            nz=8,
            iterations=6,
            repeats=1,
            cold_seconds=0.5,
            warm_seconds=0.5,
            mean_seconds=0.5,
            python_version="3.x",
            jax_version="0.x",
            nx=12,
            benchmark_kind="extruded_solve",
            operator_path="solve_extruded_inductionless",
        )

    monkeypatch.setattr(
        run_strong_scaling_worker,
        "benchmark_extruded_inductionless_solve",
        fake_benchmark_extruded_inductionless_solve,
    )
    output_path = tmp_path / "worker_solve.json"
    rc = run_strong_scaling_worker.main(
        [
            "--benchmark-kind",
            "extruded_solve",
            "--nx",
            "12",
            "--ny",
            "10",
            "--nz",
            "8",
            "--iterations",
            "6",
            "--repeats",
            "1",
            "--num-devices",
            "1",
            "--profile-dir",
            str(tmp_path / "profile"),
            "--restart",
            str(tmp_path / "steady.npz"),
            "--output",
            str(output_path),
        ]
    )

    assert rc == 0
    payload = json.loads(output_path.read_text())
    assert payload["benchmark_kind"] == "extruded_solve"
    assert payload["operator_path"] == "solve_extruded_inductionless"
    assert len(payload["source_fingerprint"]) == 64

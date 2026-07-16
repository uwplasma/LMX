from pathlib import Path
import json
import subprocess
import sys
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
    _shard_placement,
    _two_axis_mesh_and_sharding,
    benchmark_extruded_inductionless_solve,
    benchmark_sharded_extruded_operator,
    summarize_pressure_linear_history,
    summarize_strong_scaling_records,
    write_scaling_report,
    write_strong_scaling_summary_table,
)
from examples.strong_scaling_demo import _default_visible_devices
from scripts import run_freemhd_parity_suite, run_strong_scaling_worker


def test_b2_repeat_signature_ignores_gauge_and_detects_shard_changes():
    field = np.arange(24.0).reshape(4, 3, 2)
    attributes = {name: field.copy() for name in ("u", "v", "w", "p", "jx", "jy", "jz")}
    attributes.update(phi=field.copy(), rho_phi_plus=np.stack((field,) * 3),
        rho_phi_inlet=np.zeros((3, 2)))
    reference = run_strong_scaling_worker._b2_repeat_signature(SimpleNamespace(**attributes))
    attributes["phi"] += 7.0
    np.testing.assert_array_equal(
        run_strong_scaling_worker._b2_repeat_signature(SimpleNamespace(**attributes)),
        reference,
    )
    attributes["u"][2] += 1.0
    assert not np.allclose(
        run_strong_scaling_worker._b2_repeat_signature(SimpleNamespace(**attributes)),
        reference,
    )


def test_schema6_anderson_diagnostics_cover_restart_weights_and_placement():
    mapped0 = jnp.arange(32.0).reshape(4, 2, 2, 2)
    residual0 = mapped0 * 1.0e-3 + 1.0e-4
    mapped1, residual1 = mapped0 + 0.25, residual0 * 0.8
    flux0 = jnp.arange(24.0).reshape(3, 2, 2, 2)
    inlet0 = jnp.arange(4.0).reshape(2, 2)
    checkpoint = SimpleNamespace(
        anderson_state=(mapped0, residual0, flux0, inlet0),
        stopping_state=(3, 0, "in_progress"),
    )
    direct = SimpleNamespace(
        anderson_state=(mapped1, residual1, flux0 + 0.5, inlet0 + 0.5),
        stopping_state=(6, 0, "step_limit"),
    )
    resumed = SimpleNamespace(**direct.__dict__)
    serialized = SimpleNamespace(
        bundle=SimpleNamespace(**checkpoint.__dict__),
        metadata={"restart_schema": "b2_diagnostics_v6"},
    )
    problem = SimpleNamespace(case=SimpleNamespace(
        solver=SimpleNamespace(
            coupling_acceleration="anderson",
            coupling_history_depth=2,
            coupling_regularization=1.0e-8,
        ),
        time_stepper=SimpleNamespace(max_steps=6),
    ))

    diagnostics = run_strong_scaling_worker._anderson_diagnostics(
        problem, checkpoint, direct, resumed, serialized, num_devices=1
    )

    assert diagnostics["schema6_active"]
    assert diagnostics["anderson_depth_two_update_executed"]
    assert diagnostics["anderson_validation_passed"]
    assert diagnostics["anderson_serialized_max_abs"] == 0.0
    assert diagnostics["anderson_replay_max_abs"] == 0.0
    assert diagnostics["anderson_replay_field_relative_l2"] == 0.0
    assert diagnostics["anderson_replay_field_tolerance_ratio"] == 0.0
    assert diagnostics["anderson_weights_sum_error"] <= 1.0e-12
    for delta, expected in ((5.0e-10, True), (1.0e-2, False)):
        resumed.anderson_state = (mapped1 + delta, residual1, flux0 + 0.5, inlet0 + 0.5)
        noisy = run_strong_scaling_worker._anderson_diagnostics(
            problem, checkpoint, direct, resumed, serialized, num_devices=1
        )
        assert noisy["anderson_replay_field_relative_l2"] > 0.0
        assert (noisy["anderson_replay_field_tolerance_ratio"] <= 1.0) is expected
        assert noisy["anderson_validation_passed"] is expected


def test_matched_b2_topology_gate_compares_schema6_gram_and_contract():
    base = {
        "validation_passed": True,
        "source_fingerprint": "source",
        "input_sha256": "input",
        "evaluator_sha256": "evaluator",
        "restart_schema": "b2_diagnostics_v6",
        "coupling_acceleration": "anderson",
        "coupling_history_depth": 2,
        "schema6_active": True,
        "anderson_validation_passed": True,
        "anderson_gram": [[2.0, 1.0], [1.0, 2.0]],
        "anderson_weights": [0.5, 0.5],
        "velocity_l2": 3.0,
        "potential_l2": 2.0,
        "current_l2": 1.0,
        "observables": {
            "pressure_observable": [1.0, 2.0],
            "courant_mean": [1.0e-5, 1.1e-5],
            "courant_max": [2.0e-5, 2.1e-5],
        },
    }
    strong_scaling_demo._validate_matched_b2_topologies([
        base, base | {"velocity_l2": 3.0 + 1.0e-10}
    ])
    with pytest.raises(RuntimeError, match="anderson_weights"):
        strong_scaling_demo._validate_matched_b2_topologies([
            base, base | {"anderson_weights": [0.6, 0.4]}
        ])


def test_scaling_fingerprint_owns_packaged_benchmark_resources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    paths = run_strong_scaling_worker._source_fingerprint_paths()
    relative = {path.relative_to(run_strong_scaling_worker.ROOT).as_posix()
        for path in paths}
    assert any(name.startswith("lmx/data/benchmarks/specs/") for name in relative)
    assert any(name.startswith("lmx/data/benchmarks/references/") for name in relative)

    probe = tmp_path / "resource.toml"
    probe.write_text("value = 1")
    monkeypatch.setattr(run_strong_scaling_worker, "ROOT", tmp_path)
    monkeypatch.setattr(
        run_strong_scaling_worker, "_source_fingerprint_paths", lambda: (probe,)
    )
    first = run_strong_scaling_worker._source_fingerprint()
    probe.write_text("value = 2")
    assert run_strong_scaling_worker._source_fingerprint() != first


def test_remote_scaling_archive_has_one_package_resource_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    added = []

    class Archive:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def add(self, path, *, arcname):
            added.append((Path(path), arcname))

    monkeypatch.setattr(strong_scaling_demo.tarfile, "open", lambda *args: Archive())
    monkeypatch.setattr(strong_scaling_demo.subprocess, "run", lambda *args, **kwargs: None)

    strong_scaling_demo._sync_repo_to_remote(
        repo_root=tmp_path, remote_host="office", remote_dir="/tmp/lmx"
    )

    assert [arcname for _, arcname in added] == [
        "lmx", "scripts/run_strong_scaling_worker.py",
        "scripts/run_freemhd_parity_suite.py",
    ]
    assert not any(arcname.startswith("benchmarks/") for _, arcname in added)


def test_matched_scaling_allows_a_gpu_only_run(tmp_path: Path):
    assert strong_scaling_demo.run_local_cpu_scaling(
        repo_root=tmp_path, out_dir=tmp_path, device_counts=(),
        benchmark_kind="matched_b2_smoke", nx=None, ny=7, nz=7,
        iterations=2, repeats=4, python_executable=sys.executable) == []


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        ({"validation_passed": False}, "during validation: remote worker exited with status 1"),
        ({"validation_passed": False, "failure": {
            "phase": "restart", "message": "schema mismatch",
        }}, "during restart: schema mismatch"),
    ],
)
def test_remote_scaling_retrieves_failed_worker_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    record: dict[str, object], expected: str,
):
    commands = []
    monkeypatch.setattr(strong_scaling_demo, "_sync_repo_to_remote", lambda **kwargs: None)
    monkeypatch.setattr(strong_scaling_demo, "_default_visible_devices", lambda *args: "0")

    def fake_run(command, **kwargs):
        commands.append(command)
        if command[0] == "ssh":
            raise subprocess.CalledProcessError(1, command)
        Path(command[-1]).write_text(json.dumps(record))

    monkeypatch.setattr(strong_scaling_demo.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match=expected) as caught:
        strong_scaling_demo.run_remote_gpu_scaling(
            repo_root=tmp_path, out_dir=tmp_path, remote_host="office",
            remote_dir="/tmp/lmx", device_counts=(1,),
            benchmark_kind="matched_b2_smoke", nx=None, ny=7, nz=7,
            iterations=2, repeats=1,
        )

    evidence = tmp_path / "gpu_1.json"
    assert json.loads(evidence.read_text()) == record
    assert commands[-1] == ["scp", "office:/tmp/lmx/artifacts/strong_scaling/gpu_1.json", str(evidence)]
    assert isinstance(caught.value.__cause__, subprocess.CalledProcessError)


def test_strong_scaling_demo_supports_direct_help(tmp_path: Path):
    script = Path(strong_scaling_demo.__file__).resolve()
    environment = strong_scaling_demo.os.environ.copy()
    environment["PYTHONPATH"] = ""
    completed = subprocess.run(
        [sys.executable, str(script), "--help"], cwd=tmp_path, env=environment,
        capture_output=True, text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "matched_b2_smoke" in completed.stdout


pytestmark = pytest.mark.unit


def _worker_record(**updates) -> StrongScalingRecord:
    values = dict(
        backend="cpu", device_kind="cpu", num_devices=1, nx=48, ny=64, nz=32,
        iterations=5, repeats=2, cold_seconds=0.2, warm_seconds=0.1,
        mean_seconds=0.15, python_version="3.x", jax_version="0.x",
        benchmark_kind="extruded3d",
    )
    return StrongScalingRecord(**(values | updates))


def test_scaling_demo_requires_restart_for_production(monkeypatch) -> None:
    with pytest.raises(SystemExit):
        strong_scaling_demo.main(["--benchmark-kind", "extruded_solve"])
    with pytest.raises(SystemExit):
        strong_scaling_demo.main(["--benchmark-kind", "matched_b2_smoke"])
    options = {}
    monkeypatch.setattr(strong_scaling_demo, "run_strong_scaling_demo", options.update)
    arguments = ["--benchmark-kind", "matched_b2_smoke", "--repeats", "4",
        "--cpu-nx", "16", "--worker-timeout", "45"]
    assert strong_scaling_demo.main(arguments) == 0
    assert options["cpu_problem"] == (16, 7, 7) and options["gpu_problem"] == (8, 7, 7)
    assert options["timeout_seconds"] == 45.0
    assert options["cpu_iterations"] == options["gpu_iterations"] == 2
    assert strong_scaling_demo.main([
        *arguments, "--iterations", "6", "--minimum-warm-seconds", "120",
    ]) == 0
    assert options["cpu_iterations"] == options["gpu_iterations"] == 6
    assert options["minimum_warm_seconds"] == 120.0


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
    strong_scaling_demo._run_worker(
        python_executable="python", repo_root=tmp_path, output_path=output,
        platform="CPU", benchmark_kind="matched_b2_smoke", nx=None, num_devices=1,
        ny=7, nz=7, iterations=6, repeats=4,
        matched_input=tmp_path / "input.json", evaluator=tmp_path / "evaluator.json",
        minimum_warm_seconds=120.0)
    assert {commands[1][commands[1].index(flag) + 1] for flag in (
        "--matched-input", "--evaluator")} == {
            str(tmp_path / "input.json"), str(tmp_path / "evaluator.json")}
    assert commands[1][commands[1].index("--iterations") + 1] == "6"
    assert commands[1][commands[1].index("--minimum-warm-seconds") + 1] == "120.0"
    monkeypatch.setenv("XLA_FLAGS", "--xla_dump_to=/tmp/lmx-safe --xla_cpu_multi_thread_eigen=true")
    env = strong_scaling_demo._forced_cpu_environment(2)
    assert "--xla_dump_to=/tmp/lmx-safe" in env["XLA_FLAGS"]
    assert env["XLA_FLAGS"].count("--xla_force_host_platform_device_count=2") == 1
    assert "--xla_cpu_multi_thread_eigen=true" not in env["XLA_FLAGS"]


def test_matched_scaling_worker_rejects_step_contract_mismatch(tmp_path: Path):
    matched_input = tmp_path / "matched.json"
    run_freemhd_parity_suite.materialize_matched_b2_lmx_input(
        matched_input, executed_steps=6
    )
    evaluator = tmp_path / "evaluator.json"
    evaluator.write_text("{}")

    with pytest.raises(ValueError, match="executes 6 steps"):
        run_strong_scaling_worker._matched_b2_smoke_benchmark(
            matched_input, evaluator, repeats=4, num_devices=1, iterations=2
        )
    with pytest.raises(ValueError, match="minimum_warm_seconds"):
        run_strong_scaling_worker._matched_b2_smoke_benchmark(
            matched_input, evaluator, repeats=4, num_devices=1, iterations=6,
            minimum_warm_seconds=-1.0,
        )


def test_sustained_scaling_requires_predeclared_multiminute_samples():
    classify = run_strong_scaling_worker._sustained_timing_passed
    assert not classify(0.0, [121.0, 122.0, 123.0])
    assert not classify(120.0, [120.0, 121.0])
    assert not classify(120.0, [120.0, 119.999, 121.0])
    assert classify(120.0, [120.0, 121.0, 122.0])

    cpu = json.loads(Path(
        "benchmarks/results/b2-schema6-cpu-scaling-20260716.json").read_text())
    cpu_sustained = cpu["physical_core_sustained_confirmation"]
    gpu = json.loads(Path(
        "benchmarks/results/b2-gpu-scaling-calibration-20260715.json").read_text())
    gpu_sustained = gpu["sustained_shared_host_calibration"]
    for record in (cpu_sustained, gpu_sustained):
        threshold = record.get("promotion_thresholds", record.get("problem"))
        assert threshold["minimum_warm_seconds"] >= 120.0
        assert all(
            sample >= 120.0 for run in record["runs"].values()
            for sample in run["warm_samples_seconds"])
    assert not gpu_sustained["gates"]["authoritative_idle_host"]
    assert not gpu_sustained["gates"]["timing_claim"]


def test_write_scaling_report_writes_json(tmp_path: Path):
    record = benchmark_sharded_extruded_operator(
        nx=16, ny=8, nz=8, iterations=2, repeats=1, num_devices=1
    )
    path = write_scaling_report([record], tmp_path / "scaling.json")

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

    baseline_record = {**baseline.__dict__, "sustained_timing_eligible": True}
    two_device["sustained_timing_eligible"] = True
    summary = summarize_strong_scaling_records([baseline_record, two_device])
    table = write_strong_scaling_summary_table(
        [baseline_record, two_device], tmp_path / "strong_scaling_table.csv"
    )

    assert summary["validation_status"] == "solver_faithful_records_present"
    assert summary["solver_faithful_record_count"] == 2
    assert summary["profiled_record_count"] == 1
    assert summary["physics_equivalent_record_count"] == 2
    assert summary["sustained_timing_record_count"] == 2
    assert summary["best_speedup"] == pytest.approx(2.0)
    assert summary["best_sustained_speedup"] == pytest.approx(2.0)
    rows = summary["rows"]
    assert rows[0]["speedup"] == pytest.approx(1.0)
    assert rows[1]["speedup"] == pytest.approx(2.0)
    assert rows[1]["parallel_efficiency"] == pytest.approx(1.0)
    assert rows[0]["warm_mcell_updates_per_second"] == pytest.approx(9.6e-5)
    assert rows[0]["memory_mib"] == pytest.approx(2.0)
    assert rows[1]["physics_equivalent"]
    assert "pressure_linear_iterations_mean" in table.read_text()


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


@pytest.mark.timeout(110)
def test_forced_cpu_duct_step_matches_one_and_two_devices(tmp_path: Path):
    one, two = strong_scaling_demo.run_local_cpu_scaling(
        repo_root=Path(__file__).resolve().parents[1], out_dir=tmp_path,
        device_counts=(1, 2), benchmark_kind="duct_step_gate", nx=8, ny=4, nz=3,
        iterations=192, repeats=1, python_executable=sys.executable,
        timeout_seconds=50)
    np.testing.assert_allclose(one["signature"], two["signature"], rtol=2e-8, atol=2e-9)
    for record in (one, two):
        assert record["momentum_converged"] and record["mixed_pressure_converged"]
        assert record["pressure_solves_converged"] and record["pressure_linear_diagnostics_complete"]
        assert max(record[key] for key in (
            "divergence", "flow_error", "momentum_residual", "lower_wall_flux",
            "mixed_pressure_local_residual")) < 1e-8
        assert min(record["convection_flux_l2"], record["mixed_pressure_l2"]) > 1e-3
        assert record["cut_boundary_separation"] > 1e-7
    for name in ("initial_flux", "velocity", "pressure", "corrected_flux", "mixed_pressure", "momentum"):
        placement = two["placement"][name]
        assert (placement["global_shards"], placement["addressable_shards"]) == (2, 2)
        assert not placement["replicated"]
    assert two["num_devices"] == 2 and two["placement"]["inlet_flux"]["replicated"]


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
            iteration_pressure_linear_history=jnp.asarray([[1e-8, 1e-9, 7, 1, 1]] * 3),
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
        iteration_pressure_linear_history=jnp.asarray([[1e-8, 1e-9, 7, 1, 1]]),
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
    assert {value.shape for value in (u, v, w, phi, src, sigma)} == {(8, 6, 4)}


def test_factor_device_mesh_prefers_near_square_factoring():
    assert tuple(map(_factor_device_mesh, (1, 4, 6, 0))) == (
        (1, 1), (2, 2), (2, 3), (1, 0))


def test_scaling_helpers_handle_missing_and_invalid_values():
    class InvalidArray:
        shape = (2,)
        dtype = object()
    assert tuple(map(_float_or_none, (None, "not-a-float"))) == (None, None)
    assert tuple(map(_int_or_none, (None, "not-an-int"))) == (None, None)
    assert _array_nbytes(InvalidArray()) == 0
    summary = summarize_pressure_linear_history(
        [[1e-8, 2e-9, 7, 1, 1], [np.nan, np.nan, 0, 0, -1]], expected_steps=2)
    assert tuple(summary[key] for key in (
        "pressure_linear_iterations_max", "pressure_linear_diagnostics_complete")) == (7, False)


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
        return _worker_record()

    monkeypatch.setattr(
        run_strong_scaling_worker,
        "benchmark_sharded_extruded_operator",
        fake_benchmark_sharded_extruded_operator,
    )
    output_path = tmp_path / "worker.json"
    rc = run_strong_scaling_worker.main(
        ["--benchmark-kind", "extruded3d", "--nx", "48", "--ny", "64",
         "--nz", "32", "--iterations", "5", "--repeats", "2",
         "--num-devices", "1", "--platform", "CPU", "--output", str(output_path)]
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
        return _worker_record(
            nx=12, ny=10, nz=8, iterations=6, repeats=1,
            cold_seconds=0.5, warm_seconds=0.5, mean_seconds=0.5,
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
        ["--benchmark-kind", "extruded_solve", "--nx", "12", "--ny", "10",
         "--nz", "8", "--iterations", "6", "--repeats", "1",
         "--num-devices", "1", "--profile-dir", str(tmp_path / "profile"),
         "--restart", str(tmp_path / "steady.npz"), "--output", str(output_path)]
    )

    assert rc == 0
    payload = json.loads(output_path.read_text())
    assert payload["benchmark_kind"] == "extruded_solve"
    assert payload["operator_path"] == "solve_extruded_inductionless"
    matched_calls = []
    monkeypatch.setattr(run_strong_scaling_worker, "_matched_b2_smoke_benchmark",
        lambda input_path, evaluator, **options: matched_calls.append(
            (input_path, evaluator, options)) or {
                "benchmark_kind": "matched_b2_smoke", "warm_seconds": 0.2,
                "validation_passed": True})
    (tmp_path / "input.json").write_text("input")
    (tmp_path / "evaluator.json").write_text("evaluator")
    output_path = tmp_path / "matched.json"
    assert run_strong_scaling_worker.main([
        "--benchmark-kind", "matched_b2_smoke", "--matched-input", str(tmp_path / "input.json"),
        "--evaluator", str(tmp_path / "evaluator.json"), "--repeats", "4",
        "--iterations", "6", "--minimum-warm-seconds", "120",
        "--num-devices", "1", "--profile-dir", str(tmp_path / "profile"),
        "--output", str(output_path)]) == 0
    matched = json.loads(output_path.read_text())
    assert matched["validation_passed"]
    assert len(matched["source_fingerprint"]) == 64
    input_path, evaluator, options = matched_calls.pop()
    assert (input_path.name, evaluator.name) == ("input.json", "evaluator.json")
    assert options == {
        "repeats": 4, "num_devices": 1, "profile_dir": tmp_path / "profile",
        "iterations": 6, "minimum_warm_seconds": 120.0,
    }
    assert run_strong_scaling_worker.main([
        "--benchmark-kind", "matched_b2_smoke", "--repeats", "4",
        "--matched-input", str(tmp_path / "input.json"),
        "--evaluator", str(tmp_path / "evaluator.json"),
        "--num-devices", "1", "--output", str(output_path)]) == 0
    default_options = matched_calls.pop()[2]
    assert default_options["iterations"] is None
    assert default_options["minimum_warm_seconds"] == 0.0
    assert matched_calls == []
    monkeypatch.setattr(run_strong_scaling_worker, "_matched_b2_smoke_benchmark",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("stopped")))
    assert run_strong_scaling_worker.main([
        "--benchmark-kind", "matched_b2_smoke", "--matched-input", str(tmp_path / "input.json"),
        "--evaluator", str(tmp_path / "evaluator.json"), "--repeats", "4",
        "--num-devices", "1", "--output", str(output_path)]) == 1
    assert json.loads(output_path.read_text())["failure"]["message"] == "stopped"

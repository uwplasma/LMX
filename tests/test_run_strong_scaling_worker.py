from pathlib import Path
import json

import pytest

from lmx.scaling import StrongScalingRecord
from scripts import run_strong_scaling_worker


pytestmark = pytest.mark.unit


def test_run_strong_scaling_worker_writes_expected_json(
    monkeypatch, tmp_path: Path, capsys
):
    def fake_benchmark_sharded_extruded_operator(**kwargs):
        assert kwargs["nx"] == 48
        assert kwargs["ny"] == 64
        assert kwargs["nz"] == 32
        assert kwargs["iterations"] == 5
        assert kwargs["repeats"] == 2
        assert kwargs["num_devices"] == 1
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
    assert json.loads(capsys.readouterr().out)["num_devices"] == 1


def test_run_strong_scaling_worker_covers_stencil_branch(monkeypatch, tmp_path: Path):
    def fake_benchmark_sharded_stencil(**kwargs):
        assert kwargs["ny"] == 32
        assert kwargs["nz"] == 24
        assert kwargs["iterations"] == 7
        assert kwargs["repeats"] == 1
        assert kwargs["num_devices"] == 2
        return StrongScalingRecord(
            backend="cpu",
            device_kind="cpu",
            num_devices=2,
            ny=32,
            nz=24,
            iterations=7,
            repeats=1,
            cold_seconds=0.4,
            warm_seconds=0.4,
            mean_seconds=0.4,
            python_version="3.x",
            jax_version="0.x",
            benchmark_kind="stencil2d",
        )

    monkeypatch.setattr(
        run_strong_scaling_worker,
        "benchmark_sharded_stencil",
        fake_benchmark_sharded_stencil,
    )
    output_path = tmp_path / "worker_stencil.json"

    rc = run_strong_scaling_worker.main(
        [
            "--benchmark-kind",
            "stencil2d",
            "--ny",
            "32",
            "--nz",
            "24",
            "--iterations",
            "7",
            "--repeats",
            "1",
            "--num-devices",
            "2",
            "--output",
            str(output_path),
        ]
    )

    assert rc == 0
    payload = json.loads(output_path.read_text())
    assert payload["benchmark_kind"] == "stencil2d"
    assert payload["num_devices"] == 2


def test_run_strong_scaling_worker_covers_solver_faithful_branch(
    monkeypatch, tmp_path: Path
):
    def fake_benchmark_extruded_inductionless_solve(**kwargs):
        assert kwargs["nx"] == 12
        assert kwargs["ny"] == 10
        assert kwargs["nz"] == 8
        assert kwargs["max_steps"] == 6
        assert kwargs["repeats"] == 1
        assert kwargs["num_devices"] == 1
        assert kwargs["profile_dir"] == tmp_path / "profile"
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
            "--output",
            str(output_path),
        ]
    )

    assert rc == 0
    payload = json.loads(output_path.read_text())
    assert payload["benchmark_kind"] == "extruded_solve"
    assert payload["operator_path"] == "solve_extruded_inductionless"

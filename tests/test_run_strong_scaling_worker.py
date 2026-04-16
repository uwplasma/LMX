from pathlib import Path
import json

from lmx.scaling import StrongScalingRecord
from scripts import run_strong_scaling_worker


def test_run_strong_scaling_worker_writes_expected_json(monkeypatch, tmp_path: Path, capsys):
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

    monkeypatch.setattr(run_strong_scaling_worker, "benchmark_sharded_extruded_operator", fake_benchmark_sharded_extruded_operator)
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

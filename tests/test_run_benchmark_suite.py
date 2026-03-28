from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_benchmark_suite as suite


pytestmark = pytest.mark.unit


def test_run_benchmark_suite_writes_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    output = tmp_path / "artifacts" / "benchmark.json"
    recorded: dict[str, object] = {}

    monkeypatch.setattr(
        suite.argparse.ArgumentParser,
        "parse_args",
        lambda self: SimpleNamespace(output=output, repeats=2, ha=5.0, ny=8, nz=8),
    )
    monkeypatch.setattr(
        suite,
        "benchmark_solver",
        lambda repeats, ha, ny, nz: {
            "case": "hartmann_ha5",
            "cold_seconds": 1.0,
            "warm_seconds": 0.5,
            "mean_seconds": 0.75,
            "repeats": float(repeats),
        },
    )
    monkeypatch.setattr(suite, "write_benchmark_report", lambda payload, path: recorded.update(payload=payload, path=path) or Path(path))

    exit_code = suite.main()

    assert exit_code == 0
    assert recorded["path"] == output
    assert recorded["payload"]["case"] == "hartmann_ha5"
    assert '"case": "hartmann_ha5"' in capsys.readouterr().out

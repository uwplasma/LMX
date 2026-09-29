"""The FreeMHD B2 comparator without Docker: inputs, provenance, the output observer and the core-flow side."""

import subprocess

import numpy as np
import pytest

from validation import freemhd

pytestmark = pytest.mark.unit


def test_materialized_input_is_deterministic_and_carries_the_csv_field(tmp_path):
    first = freemhd.materialize_freemhd_input(tmp_path / "a")
    assert (
        first == freemhd.materialize_freemhd_input(tmp_path / "b") == freemhd.artifact_sha256(tmp_path / "a")
    )
    with pytest.raises(FileExistsError):
        freemhd.materialize_freemhd_input(tmp_path / "a")
    field = (tmp_path / "a/system/liquid/setExprFieldsDict").read_text()
    assert freemhd.artifact_sha256(freemhd.REFERENCE) in field and "lmxExtrapolation forbidden" in field
    assert "probeLocations" in (tmp_path / "a/system/controlDict").read_text()
    (tmp_path / "a/0/U").write_text("changed")
    assert freemhd.artifact_sha256(tmp_path / "a") != first


def test_the_field_expression_is_the_piecewise_linear_interpolant():
    reference = freemhd.load_reference()
    x_anchor, b_anchor = reference["x_over_L"], reference["b_over_B0"]
    expression = freemhd._field_expression(x_anchor.size).replace("pos(", "(0<")
    names = {f"x{chr(97 + i)}": v for i, v in enumerate(x_anchor)} | {
        f"b{chr(97 + i)}": v for i, v in enumerate(b_anchor)
    }
    for x in np.linspace(-15.0, 10.0, 37):
        assert eval(expression, {}, names | {"x": x}) == pytest.approx(
            np.interp(x, x_anchor, b_anchor), abs=1e-12
        )


def test_tree_hash_rejects_links(tmp_path):
    (tmp_path / "file").write_text("x")
    (tmp_path / "link").symlink_to(tmp_path / "file")
    with pytest.raises(ValueError):
        freemhd.artifact_sha256(tmp_path)


def _synthetic_output(root, *, courant=0.2, leak=0.0):
    times = [freemhd.DT, 2 * freemhd.DT]
    lines = ["Starting"]
    for t in times:
        lines += [f"Time = {t!r}", f"Region: liquid Courant Number mean: 0.1 max: {courant}"]
    (root / "postProcessing/b2PressureTaps/0").mkdir(parents=True)
    (root / "run.log").write_text("\n".join([*lines, "End", ""]))
    x = np.linspace(-15.0 + 25.0 / 16.0, 10.0 - 25.0 / 16.0, 8)
    header = [f"# Probe {i} ({xi:.17g} 0.8 0)" for i, xi in enumerate(x)]
    header += [f"# Probe {i + 8} ({xi:.17g} 0 0.8)" for i, xi in enumerate(x)]
    side = 540.0 * np.exp(-(x**2))
    rows = [" ".join(f"{v:.17g}" for v in [t, *np.zeros(8), *side]) for t in times]
    (root / "postProcessing/b2PressureTaps/0/p").write_text("\n".join([*header, "# Time", *rows, ""]))
    values = {"massIn": -4.0, "massOut": 4.0 + leak, "currentIn": 0.0, "currentOut": 0.0}
    values |= {"currentIntoSolid": 0.0, "currentIntoSolidMagnitude": 1.0}
    for name, value in values.items():
        (root / "postProcessing" / name / "0").mkdir(parents=True)
        text = "\n".join(["# Time sum", *(f"{t!r} {value!r}" for t in times), ""])
        (root / "postProcessing" / name / "0/surfaceFieldValue.dat").write_text(text)
    return x, np.exp(-(x**2))


def test_the_observer_reads_native_output_and_applies_the_smoke_gates(tmp_path):
    limits = freemhd.load_spec()["harness_smoke_execution"]
    x, excess = _synthetic_output(tmp_path / "pass")
    observed = freemhd.observe_freemhd(tmp_path / "pass")
    plateau = (x <= -7.5) | (x >= 5.0)
    assert observed["x_over_L"] == pytest.approx(x)
    assert observed["pressure_observable"] == pytest.approx(excess - excess[plateau].mean())
    assert observed["steps"] == 2 and observed["mass_balance"] == 0.0
    assert freemhd.freemhd_failures(observed, limits) == []
    _synthetic_output(tmp_path / "fail", courant=0.5, leak=0.1)
    failed = freemhd.freemhd_failures(freemhd.observe_freemhd(tmp_path / "fail"), limits)
    assert failed == ["courant", "mass_balance"]
    (tmp_path / "pass/run.log").write_text("--> FOAM FATAL ERROR\nEnd\n")
    with pytest.raises(ValueError, match="FOAM FATAL"):
        freemhd.observe_freemhd(tmp_path / "pass")


def test_the_core_flow_side_passes_its_gates_on_a_coarse_mesh():
    coarse = freemhd.solve_core_flow(nx=100, nz=8, ny=8)
    assert freemhd.core_flow_failures(coarse) == []
    assert coarse["floored_nodes"] == coarse["weak_field_nodes"] > 0
    x, observable = np.asarray(coarse["x_over_L"]), np.asarray(coarse["pressure_observable"])
    assert 0.02 < observable.max() < 0.035 and -2.0 < x[observable.argmax()] < 0.0
    assert freemhd.core_flow_failures(coarse | {"axial_flux_spread": 1e-3}) == ["axial_flux"]


def test_the_source_snapshot_requires_the_pinned_commit(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    with pytest.raises(ValueError, match="pinned commit"):
        freemhd.snapshot_freemhd_source(tmp_path, tmp_path / "out")
    with pytest.raises(ValueError, match="worktree root"):
        freemhd.snapshot_freemhd_source(tmp_path / ".git", tmp_path / "out")


def test_preflight_hashes_inputs_without_docker(tmp_path, capsys):
    argv = ["--preflight", "--output", str(tmp_path / "run"), "--freemhd-source-repo", str(tmp_path / "none")]
    assert freemhd.main(argv) == 0
    assert set(freemhd.json.loads((tmp_path / "run/preflight.json").read_text())["artifacts"]) == {
        "spec",
        "reference",
        "lmhdx_coreflow",
        "freemhd_input",
    }

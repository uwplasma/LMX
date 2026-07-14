from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from lmx._fringing_types import ExtrudedFieldBundle
from scripts import run_benchmark_b_independence as campaign


pytestmark = pytest.mark.unit


def test_campaign_imports_lmx_from_its_own_source_tree():
    assert campaign.ROOT in Path(campaign.lmx.__file__).resolve().parents


def test_variant_problem_applies_only_frozen_solver_control_changes():
    baseline = campaign._variant_problem("B1-fringing-pipe", "coarse", "baseline")
    tight = campaign._variant_problem("B1-fringing-pipe", "coarse", "tight_tolerance")
    extended = campaign._variant_problem(
        "B1-fringing-pipe", "coarse", "extended_iterations"
    )
    thin = campaign._variant_problem("B1-fringing-pipe", "coarse", "thin_wall")

    assert tight.case.solver.coupling_tolerance == pytest.approx(
        0.5 * baseline.case.solver.coupling_tolerance
    )
    assert (
        tight.case.time_stepper.steady_tolerance
        == baseline.case.time_stepper.steady_tolerance
    )
    assert (
        tight.case.time_stepper.potential_iterations
        == 2 * baseline.case.time_stepper.potential_iterations
    )
    assert (
        tight.case.solver.coupling_iterations
        == 4 * baseline.case.solver.coupling_iterations
    )
    assert (
        extended.case.solver.coupling_iterations
        == 2 * baseline.case.solver.coupling_iterations
    )
    assert (
        extended.case.time_stepper.max_steps == 2 * baseline.case.time_stepper.max_steps
    )
    assert thin.case.geometry.wall_thickness[0] == pytest.approx(
        0.5 * baseline.case.geometry.wall_thickness[0]
    )
    baseline_conductivity = baseline.case.regions[1].conductivity
    thin_conductivity = thin.case.regions[1].conductivity
    assert thin_conductivity == pytest.approx(2.0 * baseline_conductivity)
    assert campaign._effective_iteration_limits(baseline) == {
        "electric_iterations": 4000,
        "projection_iterations": 4000,
        "momentum_iterations": 400,
    }
    b2 = campaign._variant_problem("B2-fringing-square", "coarse", "baseline")
    assert campaign._effective_iteration_limits(b2)["electric_iterations"] == 600
    assert b2.case.solver.coupling_min_relaxation == pytest.approx(2.0)
    assert (
        campaign._variant_problem(
            "B2-fringing-square", "coarse", "baseline", num_devices=2
        ).case.geometry.nx
        == 102
    )

    with pytest.raises(ValueError, match="Unsupported independence variant"):
        campaign._variant_problem("B1-fringing-pipe", "coarse", "unknown")


def _record(observable, *, residual=1.0e-9, coupling_tolerance=None):
    return {
        "primary_observable": observable,
        "controls": (
            {}
            if coupling_tolerance is None
            else {"coupling_tolerance": coupling_tolerance}
        ),
        "diagnostics": {
            "max_residual": residual,
            "max_divergence_residual": 1.0e-5,
            "max_charge_balance_residual": 1.0e-5,
            "volumetric_flow_rate_span": 1.0e-5,
            "max_wall_current_leakage": 0.0,
            "net_boundary_current_residual": 0.0,
            "final_steady_streak": 3,
            "stop_reason": "converged",
        },
    }


def test_comparison_enforces_tighter_variant_steady_tolerance():
    records = {
        "baseline": _record([0.1, 0.2, 0.1], residual=4.0e-5),
        "tight_tolerance": _record(
            [0.1, 0.2, 0.1], residual=3.0e-5, coupling_tolerance=2.5e-5
        ),
        "extended_iterations": _record([0.1, 0.2, 0.1]),
        "thin_wall": _record([0.1, 0.2, 0.1]),
    }

    comparison = campaign._comparison("B2-fringing-square", records)

    assert comparison["gates"]["steady_residual"]
    assert not comparison["gates"]["tight-tolerance_steady_residual"]
    assert not comparison["pass"]


def test_comparison_applies_uncertainty_and_thin_wall_gates():
    records = {
        "baseline": _record([0.1, 0.2, 0.1]),
        "tight_tolerance": _record([0.1001, 0.2001, 0.1001]),
        "extended_iterations": _record([0.1001, 0.1999, 0.1001]),
        "thin_wall": _record([0.1001, 0.2001, 0.1001]),
    }
    comparison = campaign._comparison("B2-fringing-square", records)
    assert comparison["complete"] is True
    assert comparison["pass"] is True
    assert all(comparison["gates"].values())

    records["baseline"] = _record([0.1, 0.2, 0.1], residual=1.0)
    failed = campaign._comparison("B2-fringing-square", records)
    assert failed["pass"] is False
    assert failed["gates"]["steady_residual"] is False

    records["baseline"] = _record([0.1, 0.2, 0.1])
    records["baseline"]["diagnostics"]["max_divergence_residual"] = 1.0
    failed = campaign._comparison("B2-fringing-square", records)
    assert failed["pass"] is False
    assert failed["gates"]["mass_balance"] is False

    records["baseline"] = _record([0.1, 0.2, 0.1])
    records["thin_wall"] = _record([0.1001, 0.2001, 0.1001])
    records["thin_wall"]["diagnostics"]["final_steady_streak"] = 2
    failed = campaign._comparison("B2-fringing-square", records)
    assert failed["pass"] is False
    assert failed["gates"]["thin-wall_sustained_stopping"] is False

    incomplete = campaign._comparison(
        "B2-fringing-square", {"baseline": records["baseline"]}
    )
    assert incomplete["complete"] is False
    assert "thin_wall" in incomplete["missing_variants"]


def test_dry_run_writes_deterministic_campaign_plan(tmp_path: Path):
    output = tmp_path / "campaign"
    exit_code = campaign.main(
        [
            "--output",
            str(output),
            "--cases",
            "B1-fringing-pipe",
            "--variants",
            "baseline",
            "--dry-run",
        ]
    )
    payload = json.loads((output / "benchmark-b-independence.json").read_text())
    assert exit_code == 0
    assert payload["pass"] is False
    assert payload["cases"] == [
        {"case_id": "B1-fringing-pipe", "complete": False, "dry_run": True}
    ]


def test_resume_rejects_checkpoint_from_another_fingerprint(tmp_path: Path):
    output = tmp_path / "campaign"
    checkpoint = output / "runs" / "B1-fringing-pipe-coarse-baseline.json"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text(json.dumps({"source_fingerprint": "stale"}))
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        campaign.main(
            [
                "--output",
                str(output),
                "--cases",
                "B1-fringing-pipe",
                "--variants",
                "baseline",
                "--resume",
            ]
        )


def test_progress_writer_keeps_latest_atomic_partial_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    progress_path = tmp_path / "run.progress.json"
    restart_path = tmp_path / "run.partial.npz"
    monkeypatch.setattr(campaign, "_source_fingerprint", lambda: "source")

    def fake_restart_writer(bundle, case, path):
        Path(path).write_bytes(bundle)

    monkeypatch.setattr(
        campaign, "write_extruded_bundle_restart_npz", fake_restart_writer
    )
    writer = campaign._progress_writer(
        problem=SimpleNamespace(case=object()),
        case_id="B1-fringing-pipe",
        variant="baseline",
        progress_path=progress_path,
        partial_restart_path=restart_path,
        started=campaign.time.perf_counter(),
    )
    common = {
        "total_steps": 8,
        "residual": 1.0e-4,
        "component_residuals": (1.0e-4,) * 6,
        "pressure_residual": 1.0e-5,
        "potential_residual": 1.0e-6,
    }
    writer(SimpleNamespace(step=2, checkpoint=b"restart", **common))
    writer(SimpleNamespace(step=3, checkpoint=None, **common))

    payload = json.loads(progress_path.read_text())
    assert restart_path.read_bytes() == b"restart"
    assert payload["step"] == 3
    assert payload["checkpoint"]["step"] == 2
    assert payload["checkpoint"]["sha256"] == campaign._file_sha256(restart_path)


def test_partial_restart_requires_matching_progress_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    restart_path = tmp_path / "run.partial.npz"
    progress_path = tmp_path / "run.progress.json"
    restart_path.touch()

    with pytest.raises(ValueError, match="no progress metadata"):
        campaign._load_partial_restart(restart_path, progress_path, "source")

    progress_path.write_text(json.dumps({"source_fingerprint": "stale"}))
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        campaign._load_partial_restart(restart_path, progress_path, "source")

    bundle = object()
    progress_path.write_text(
        json.dumps(
            {
                "source_fingerprint": "source",
                "checkpoint": {"sha256": campaign._file_sha256(restart_path)},
            }
        )
    )
    monkeypatch.setattr(
        campaign,
        "load_extruded_restart_bundle",
        lambda path: SimpleNamespace(bundle=bundle),
    )
    assert (
        campaign._load_partial_restart(restart_path, progress_path, "source") is bundle
    )

    restart_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        campaign._load_partial_restart(restart_path, progress_path, "source")


def test_acceptance_observable_is_reloaded_from_persisted_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    bundle = SimpleNamespace(
        x=campaign.np.asarray([-8.0, 0.0, 6.0]),
        transverse_pressure_difference=campaign.np.asarray([1.0, 4.0, 3.0]),
    )
    monkeypatch.setattr(
        campaign,
        "load_extruded_restart_bundle",
        lambda path: SimpleNamespace(bundle=bundle),
    )

    x, observable = campaign._load_restart_observable(
        tmp_path / "accepted.npz", "B2-fringing-square"
    )

    assert x == pytest.approx([-8.0, 0.0, 6.0])
    assert observable == pytest.approx([-1.0 / 540.0, 2.0 / 540.0, 1.0 / 540.0])


def test_variant_restart_parser_is_explicit_and_rejects_invalid_values():
    assert campaign._parse_variant_restarts(
        ["thin_wall=/tmp/thin.npz", "baseline=/tmp/base.npz"]
    ) == {
        "thin_wall": Path("/tmp/thin.npz"),
        "baseline": Path("/tmp/base.npz"),
    }
    with pytest.raises(ValueError, match="VARIANT"):
        campaign._parse_variant_restarts(["thin_wall"])
    with pytest.raises(ValueError, match="VARIANT"):
        campaign._parse_variant_restarts(["unknown=/tmp/x.npz"])


def test_acceptance_assembly_reuses_three_campaigns_without_solver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    case_id = "B1-fringing-pipe"
    arguments = ["--output", str(tmp_path / "accepted"), "--cases", case_id]
    for level in campaign.MESH_LEVELS:
        directory = tmp_path / level
        runs = directory / "runs"
        runs.mkdir(parents=True)
        (directory / "benchmark-b-independence.json").write_text(
            json.dumps(
                {
                    "mesh_level": level,
                    "source_fingerprint": "source",
                    "cases": [{"case_id": case_id, "complete": True, "pass": True}],
                }
            )
        )
        (runs / f"{case_id}-{level}-baseline.json").write_text(
            json.dumps(
                {
                    "case_id": case_id,
                    "mesh_level": level,
                    "source_fingerprint": "source",
                }
            )
        )
        arguments.extend(("--acceptance-mesh", f"{level}={directory}"))
    freemhd = tmp_path / "freemhd.json"
    freemhd.write_text(json.dumps({"case_id": case_id, "pass": True}))
    arguments.extend(("--freemhd-record", f"{case_id}={freemhd}"))
    monkeypatch.setattr(campaign, "_source_fingerprint", lambda: "source")

    def evaluate(selected, meshes, reference, *, matched_freemhd_artifact_root):
        assert selected == case_id
        assert set(meshes) == set(campaign.MESH_LEVELS)
        assert reference["case_id"] == case_id
        assert matched_freemhd_artifact_root == tmp_path.resolve()
        return {"case_id": case_id, "pass": True}

    monkeypatch.setattr(campaign, "_evaluate_acceptance", evaluate)
    assert campaign.main(arguments) == 0
    payload = json.loads(
        (tmp_path / "accepted" / "benchmark-b-acceptance.json").read_text()
    )
    assert payload["pass"] is True


def test_acceptance_path_parser_requires_every_unique_name():
    with pytest.raises(ValueError, match="each of coarse, medium, fine once"):
        campaign._parse_acceptance_paths(
            ["coarse=/a", "coarse=/b"], campaign.MESH_LEVELS, "--acceptance-mesh"
        )


@pytest.mark.parametrize("case_id", campaign.CASE_IDS)
def test_acceptance_combines_literature_mesh_and_freemhd(case_id, monkeypatch):
    reference = campaign.load_benchmark_b_reference(case_id)
    x = list(reference["x_over_L"])
    expected = list(reference["pressure_observable"])
    meshes = {
        level: {
            "source_fingerprint": "source",
            "baseline": {
                "case_id": case_id,
                "mesh_level": level,
                "source_fingerprint": "source",
                "x_over_L": x,
                "primary_observable": [value + offset for value in expected],
            },
            "independence": {"case_id": case_id, "complete": True, "pass": True},
        }
        for level, offset in zip(campaign.MESH_LEVELS, (1.0e-4, 5.0e-5, 0.0))
    }
    roots = []

    def validate(record, **kwargs):
        roots.append(kwargs["artifact_root"])
        return {
            "acceptance_pass": record.get("validated") is True,
            "schema_complete": record.get("validated") is True,
        }

    monkeypatch.setattr(campaign, "validate_matched_b_record", validate)
    freemhd = {"case_id": case_id, "validated": True}
    artifact_root = Path("/evidence")

    result = campaign._evaluate_acceptance(
        case_id, meshes, freemhd, matched_freemhd_artifact_root=artifact_root
    )

    assert result["pass"]
    assert all(result["gates"].values())
    assert result["literature"]["fine"]["weighted_rms"] == pytest.approx(0.0)
    assert roots == [artifact_root]


def test_acceptance_reports_missing_and_rejects_bad_curves():
    incomplete = campaign._evaluate_acceptance("B1-fringing-pipe", {})
    assert incomplete["missing_mesh_levels"] == list(campaign.MESH_LEVELS)
    bad = {
        level: {
            "source_fingerprint": "source",
            "baseline": {
                "case_id": "B1-fringing-pipe",
                "mesh_level": level,
                "source_fingerprint": "source",
                "x_over_L": [0.0, -1.0],
                "primary_observable": [0.0, 0.0],
            },
            "independence": {
                "case_id": "B1-fringing-pipe",
                "complete": True,
                "pass": True,
            },
        }
        for level in campaign.MESH_LEVELS
    }
    with pytest.raises(ValueError, match="baseline curve is invalid"):
        campaign._evaluate_acceptance("B1-fringing-pipe", bad)


def test_gpu_device_parser_requires_unique_ids():
    assert campaign._parse_gpu_devices("0, 1") == ("0", "1")
    with pytest.raises(ValueError, match="unique"):
        campaign._parse_gpu_devices("")
    with pytest.raises(ValueError, match="unique"):
        campaign._parse_gpu_devices("0,0")


def test_spatial_placement_records_and_enforces_actual_shards():
    field = SimpleNamespace(
        addressable_shards=(
            SimpleNamespace(device="cuda:0"),
            SimpleNamespace(device="cuda:1"),
        )
    )
    assert campaign._spatial_placement(field, 2) == {
        "spatial_devices": 2,
        "actual_spatial_shards": 2,
        "spatial_device_ids": ["cuda:0", "cuda:1"],
    }
    with pytest.raises(RuntimeError, match="solution has 2 shards"):
        campaign._spatial_placement(field, 1)


def test_b2_restart_prolongation_is_trilinear_in_physical_coordinates():
    problem = campaign._variant_problem("B2-fringing-square", "coarse", "baseline")
    geometry = campaign.replace(problem.case.geometry, nx=4)
    case = campaign.replace(problem.case, geometry=geometry)
    mesh = campaign._cross_section_mesh(case)
    profile = campaign.replace(
        problem.profile,
        x=mesh.x_centers,
        field_scale=campaign.np.ones(mesh.nx),
    )
    problem = campaign.replace(problem, case=case, profile=profile)
    axes = tuple(
        campaign.np.asarray([values[0], values[-1]])
        for values in (mesh.x_centers, mesh.y_centers, mesh.z_centers)
    )
    x, y, z = campaign.np.meshgrid(*axes, indexing="ij")
    linear = x + 2.0 * y + 3.0 * z
    bundle = ExtrudedFieldBundle(
        x=axes[0],
        y=axes[1],
        z=axes[2],
        field_scale=campaign.np.ones(2),
        geometry_kind="layered_duct",
        solver_kind="extruded_inductionless",
        rho_phi_plus=campaign.np.ones((3, 2, 2, 2)),
        rho_phi_inlet=campaign.np.ones((2, 2)),
        **dict.fromkeys(("u", "v", "w", "p", "phi"), linear),
    )

    prolonged, record = campaign._prolong_b2_restart(bundle, problem)

    target_x, target_y, target_z = campaign.np.meshgrid(
        mesh.x_centers, mesh.y_centers, mesh.z_centers, indexing="ij"
    )
    assert prolonged.u.shape == (mesh.nx, mesh.ny, mesh.nz)
    assert prolonged.u == pytest.approx(target_x + 2.0 * target_y + 3.0 * target_z)
    assert prolonged.rho_phi_plus is prolonged.rho_phi_inlet is None
    assert record == {
        "method": "trilinear_physical_coordinates",
        "compact_flux": "reinitialized_on_target_mesh",
        "source_shape": [2, 2, 2],
        "target_shape": [mesh.nx, mesh.ny, mesh.nz],
    }


def test_gpu_wave_assigns_one_variant_per_device(monkeypatch: pytest.MonkeyPatch):
    launches = []

    class Process:
        returncode = 2

        def communicate(self):
            return "incomplete comparison", ""

    def fake_popen(command, **kwargs):
        launches.append((command, kwargs))
        return Process()

    monkeypatch.setattr(campaign.subprocess, "Popen", fake_popen)
    args = SimpleNamespace(
        output=Path("artifacts/campaign"),
        mesh_level="coarse",
        resume=True,
        initial_restart=None,
        variant_restart=[],
        checkpoint_interval=8,
    )
    campaign._run_gpu_wave(
        args,
        [("B1-fringing-pipe", "baseline"), ("B2-fringing-square", "thin_wall")],
        ("0", "1"),
    )

    assert [item[1]["env"]["CUDA_VISIBLE_DEVICES"] for item in launches] == ["0", "1"]
    assert all(
        item[1]["env"]["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false" for item in launches
    )
    assert all(
        "lmx-jax-cache" in item[1]["env"]["JAX_COMPILATION_CACHE_DIR"]
        for item in launches
    )
    assert "baseline" in launches[0][0]
    assert "thin_wall" in launches[1][0]
    assert launches[0][0][-3:-1] == ["--checkpoint-interval", "8"]


@pytest.mark.parametrize("physics_passes", [False, True])
def test_gpu_campaign_gates_restart_dependent_second_wave(
    monkeypatch: pytest.MonkeyPatch, physics_passes: bool
):
    waves = []
    monkeypatch.setattr(
        campaign, "_run_gpu_wave", lambda args, tasks, devices: waves.append(tasks)
    )
    monkeypatch.setattr(
        campaign, "_wave_physics_passes", lambda args, tasks: physics_passes
    )
    monkeypatch.setattr(campaign, "main", lambda argv=None: 7)
    args = SimpleNamespace(
        gpu_devices="0,1",
        output=Path("artifacts/campaign"),
        cases=["B2-fringing-square"],
        mesh_level="coarse",
        variants=list(campaign.VARIANTS),
        checkpoint_interval=8,
    )

    assert campaign._run_gpu_campaign(args) == (7 if physics_passes else 2)
    assert len(waves) == (2 if physics_passes else 1)


def test_b2_evidence_plot_mode_uses_only_existing_records(tmp_path, monkeypatch):
    record = {"x_over_L": [-1.0, 0.0, 1.0], "primary_observable": [0.0, 0.1, 0.0]}
    transverse = tmp_path / "transverse.json"
    consistent = tmp_path / "consistent.json"
    transverse.write_text(json.dumps(record))
    consistent.write_text(json.dumps(record))
    field = tmp_path / "field.npz"
    campaign.np.savez(
        field,
        x=campaign.np.linspace(14.0, 16.0, 5),
        y=campaign.np.linspace(-1.0, 1.0, 4),
        bx=campaign.np.ones((5, 4, 1)) * 0.1,
        by=campaign.np.ones((5, 4, 1)),
    )
    output = tmp_path / "evidence.webp"
    monkeypatch.setattr(
        campaign, "_run_record", lambda *args, **kwargs: pytest.fail("solver ran")
    )
    assert (
        campaign.main(
            [
                "--plot-evidence",
                str(output),
                "--plot-transverse-record",
                str(transverse),
                "--plot-consistent-record",
                str(consistent),
                "--plot-field-record",
                str(field),
            ]
        )
        == 0
    )
    assert output.is_file() and output.stat().st_size < 100_000


def test_gpu_wave_physics_gate_reads_worker_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    runs = tmp_path / "runs"
    runs.mkdir()
    path = runs / "B2-fringing-square-coarse-baseline.json"
    path.write_text('{"variant": "baseline"}')
    args = SimpleNamespace(output=tmp_path, mesh_level="coarse")
    task = [("B2-fringing-square", "baseline")]
    monkeypatch.setattr(
        campaign,
        "_comparison",
        lambda case_id, records: {"gates": {"steady_residual": True}},
    )
    assert campaign._wave_physics_passes(args, task)
    monkeypatch.setattr(
        campaign,
        "_comparison",
        lambda case_id, records: {"gates": {"steady_residual": False}},
    )
    assert not campaign._wave_physics_passes(args, task)

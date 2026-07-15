from pathlib import Path
from types import SimpleNamespace
from copy import deepcopy
from dataclasses import replace
import hashlib
import json

import pytest

import lmx.benchmarks as benchmarks
from lmx.benchmarks import (
    benchmark_b_pressure_observable,
    build_benchmark_b_problem,
    build_benchmark_b_field_profile,
    benchmark_solver,
    load_benchmark_b_reference,
    load_benchmark_b_spec,
    write_benchmark_report,
)
from lmx.fringing import (
    _cross_section_mesh,
    _unpack_duct_mass_flux,
    solve_extruded_inductionless,
)
from lmx.io import load_extruded_restart_bundle, write_extruded_bundle_restart_npz
from scripts.analyze_freemhd_benchmark_a_ladder import (
    analyze_ladder,
    compact_evidence,
    freeze_summary,
)


pytestmark = pytest.mark.unit

_MATCHED = ("matched_contract",)
_SHARED = _MATCHED + ("shared",)
_EQUATIONS = _SHARED + ("equations",)
_MESH_LEVELS = ("mesh", "levels")
_SEMANTICS = "matched formulation semantics differ"
_FREEMHD_COMMIT = ("free_mhd_discretization_reference", "repository_commit")
_STEADY_STEPS = _MATCHED + ("roles", "b2-production", "stopping_rules", "steady_steps_min")
_STOPPING = "matched stopping contract differs"
_REFERENCE_HEADER = (
    "x_over_L,b_over_B0,b_uncertainty,pressure_observable,pressure_uncertainty"
)


def _freemhd_record(case_kind: str, cells: int) -> dict:
    h = 1.0 / cells
    reference = [0.0, 1.0, 0.0]
    shape = [0.5, -1.0, 0.5]
    simulated = [value + h**2 * delta for value, delta in zip(reference, shape)]
    cut = {
        "coordinate": [-1.0, 0.0, 1.0],
        "reference": reference,
        "simulated": simulated,
        "l2_error": (0.5 * h**4) ** 0.5,
    }
    return {
        "case_kind": case_kind,
        "settings": {"ny": cells, "nz": cells},
        "benchmark_spec": {
            "id": f"A2-{case_kind}-ha20",
            "path": f"{case_kind}.toml",
            "sha256": "a" * 64,
        },
        "solver_diagnostics": {
            "potential_iterations_used": 20,
            "potential_residual": 1.0e-10,
        },
        "integral_observables": {
            "applied_pressure_gradient": 10.0 + 100.0 * h**2,
            "reference_pressure_gradient": 10.0,
            "pressure_gradient_relative_error": 0.006,
        },
        "observables": {
            name: {axis: deepcopy(cut) for axis in ("y", "z")}
            for name in ("velocity", "potential", "current", "lorentz")
        },
        "current_balance": {"acceptance_target": 0.001},
        "power_balance": {"mechanical_power_relative_error": 1.0e-5},
        "continuum_velocity_audit": {
            "reference_path": f"/nonportable/reference/{case_kind}.txt",
            "axes": {"y": {"lmx_raw_analytical": {"l2_error": 0.007}}},
        },
    }


def _freemhd_levels(
    levels=(("coarse", 10), ("medium", 20), ("fine", 40)),
) -> list[dict]:
    return [
        {
            "label": label,
            "records": [
                _freemhd_record("shercliff", cells),
                _freemhd_record("hunt", cells),
            ],
        }
        for label, cells in levels
    ]


def _freemhd_summary() -> dict[str, object]:
    return {
        "best_level_label": "confirmation",
        "implementation": {
            "runner_sha256": "1" * 64,
            "solver_core_sha256": "2" * 64,
            "lmx_version": "1.2.3",
            "solvax_version": "0.4.0",
        },
        "ladder": _freemhd_levels(
            (label, 85) for label in ("coarse", "medium", "fine", "confirmation")
        ),
        "richardson": {
            "schema_version": 1,
            "levels": ["medium", "fine", "confirmation"],
            "cases": {},
            "research_grade_validation_pass": False,
        },
    }


def _assert_iteration_histories(bundle):
    history = bundle.iteration_residual_history
    pressure = bundle.iteration_pressure_residual_history
    potential = bundle.iteration_potential_residual_history
    assert pressure.shape == potential.shape == history.shape
    assert benchmarks.jnp.all(
        history[:, None] >= benchmarks.jnp.stack((pressure, potential), 1)
    )
    electric = bundle.iteration_electric_linear_history
    assert electric.shape == (history.size, 6)
    assert benchmarks.jnp.all(electric[:, 3] > 0)
    assert benchmarks.jnp.all(electric[:, 2] <= 1.0e-3)
    return history


def _set_nested(mapping, path, value):
    root = mapping
    for key in path[:-1]:
        mapping = mapping[next(iter(mapping))] if key is None else mapping[key]
    if value is None:
        value = (
            root["wall"]["nominal_thickness_over_L"]
            if path[0] == "wall"
            else root["mesh"]["levels"][0][path[-1]]
        )
    mapping[path[-1]] = value


def _install_reference(tmp_path, monkeypatch, rows):
    (tmp_path / "reference.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    spec = deepcopy(load_benchmark_b_spec("B1-fringing-pipe"))
    spec["reference"]["data_path"] = "reference.csv"
    monkeypatch.setattr(benchmarks, "load_benchmark_b_spec", lambda *_: spec)


def test_analyze_freemhd_ladder_recovers_second_order() -> None:
    result = analyze_ladder(_freemhd_levels())
    for case in result["cases"].values():
        assert case["profiles"]["velocity_y"]["observed_order"] == pytest.approx(2.0)
        assert case["profiles"]["velocity_y"]["extrapolated_l2"] == pytest.approx(
            0.0, abs=1.0e-14
        )
        assert case["pressure_gradient"]["observed_order"] == pytest.approx(2.0)
        assert case["pressure_gradient"][
            "extrapolated_relative_error"
        ] == pytest.approx(0.0, abs=1.0e-14)
        assert case["extrapolated_primary_pass"] is True
    assert result["research_grade_validation_pass"] is True


def test_analyze_freemhd_ladder_rejects_bad_inputs_and_unavailable_order() -> None:
    with pytest.raises(ValueError, match="exactly three levels"):
        analyze_ladder(_freemhd_levels()[:2])
    levels = _freemhd_levels()
    levels[1]["records"][0]["benchmark_spec"]["sha256"] = "b" * 64
    with pytest.raises(ValueError, match="specification changed"):
        analyze_ladder(levels)

    levels = _freemhd_levels()
    for level in levels:
        cut = level["records"][0]["observables"]["velocity"]["y"]
        cut["simulated"] = cut["reference"]
        cut["l2_error"] = 0.0
    profile = analyze_ladder(levels)["cases"]["shercliff"]["profiles"]["velocity_y"]
    assert profile["observed_order"] is None
    assert profile["extrapolation_status"] == "order_unavailable"
    assert profile["extrapolated_pass"] is False


def test_compact_freemhd_evidence_is_portable_and_deterministic(tmp_path: Path) -> None:
    summary = _freemhd_summary()
    evidence = compact_evidence(summary, source_sha256="b" * 64)
    assert set(evidence) == {"richardson", "continuum", "power"}
    assert evidence["power"]["confirmation_level"] == "confirmation"
    assert evidence["power"]["implementation"]["solver_core_sha256"] == "2" * 64
    continuum = evidence["continuum"]["cases"]["shercliff"]
    assert continuum["reference_file"] == "shercliff.txt"
    assert "reference_path" not in continuum

    source = tmp_path / "summary.json"
    source.write_text(json.dumps(summary), encoding="utf-8")
    first = freeze_summary(source, tmp_path / "first")
    second = freeze_summary(source, tmp_path / "second")
    assert first.keys() == second.keys()
    assert all(first[kind].read_bytes() == second[kind].read_bytes() for kind in first)


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda summary: summary.update(ladder=summary["ladder"][:3]), "at least four"),
        (lambda summary: summary.update(best_level_label="fine"), "Best level"),
        (lambda summary: summary.pop("implementation"), "implementation fingerprint"),
        (
            lambda summary: summary["richardson"].update(
                levels=["coarse", "medium", "fine"]
            ),
            "final three",
        ),
        (
            lambda summary: summary["ladder"][-1].update(
                records=[_freemhd_record("hunt", 85)]
            ),
            "one Shercliff",
        ),
    ],
)
def test_compact_freemhd_evidence_rejects_inconsistent_campaign(
    mutation, message: str
) -> None:
    summary = _freemhd_summary()
    mutation(summary)
    with pytest.raises(ValueError, match=message):
        compact_evidence(summary, source_sha256="c" * 64)


def test_benchmark_b_specification_index_is_complete_and_deterministic():
    expected = json.loads(
        Path("benchmarks/results/benchmark-b-specification.json").read_text()
    )
    assert expected["specification_freeze_pass"] is True
    assert expected["production_results_included"] is False
    assert {case["id"] for case in expected["cases"]} == {
        "B1-fringing-pipe",
        "B2-fringing-square",
    }
    assert all(
        case["mesh_levels"] == ["coarse", "medium", "fine"]
        and case["numerical_independence"]["tolerance_uncertainty_fraction_max"] == 0.25
        for case in expected["cases"]
    )
    for case in expected["cases"]:
        for prefix in ("spec", "data"):
            digest = hashlib.sha256(Path(case[f"{prefix}_path"]).read_bytes()).hexdigest()
            assert digest == case[f"{prefix}_sha256"]
    assert len(expected["production_blockers"]) == 2


def test_benchmark_solver_returns_positive_timings(monkeypatch: pytest.MonkeyPatch):
    times = iter([10.0, 10.4, 10.4, 10.7])

    monkeypatch.setattr(
        benchmarks,
        "make_hartmann_case",
        lambda ha, ny, nz: SimpleNamespace(name="hartmann_ha5"),
    )
    monkeypatch.setattr("lmx.solvers.solve_steady", lambda case: SimpleNamespace())
    monkeypatch.setattr(benchmarks.time, "perf_counter", lambda: next(times))
    monkeypatch.setattr(benchmarks.jax, "default_backend", lambda: "cpu")
    monkeypatch.setattr(
        benchmarks.jax, "devices", lambda: [SimpleNamespace(device_kind="cpu")]
    )
    monkeypatch.setattr(benchmarks.platform, "python_version", lambda: "3.13.7")

    report = benchmark_solver(repeats=2, ha=5.0, ny=16, nz=16)
    assert float(report["cold_seconds"]) > 0.0
    assert float(report["warm_seconds"]) > 0.0
    assert report["backend"]


def test_benchmark_writer(tmp_path: Path):
    path = write_benchmark_report(
        {"cold_seconds": 1.0, "warm_seconds": 0.5}, tmp_path / "benchmark.json"
    )
    assert path.exists()


@pytest.mark.parametrize(
    ("case_id", "ha", "interaction", "conductance", "row_count"),
    [
        ("B1-fringing-pipe", 6600.0, 10700.0, 0.027, 16),
        ("B2-fringing-square", 2900.0, 540.0, 0.07, 18),
    ],
)
def test_frozen_benchmark_b_specs_and_reference_data(
    case_id: str,
    ha: float,
    interaction: float,
    conductance: float,
    row_count: int,
):
    spec = load_benchmark_b_spec(case_id)
    reference = load_benchmark_b_reference(case_id)
    assert spec["status"] == "frozen"
    assert spec["physics"]["hartmann_number"] == ha
    assert spec["physics"]["interaction_parameter"] == interaction
    assert spec["wall"]["wall_conductance_ratio"] == conductance
    assert len(reference["x_over_L"]) == row_count
    assert max(reference["pressure_uncertainty"]) == pytest.approx(
        spec["reference"]["combined_uncertainty_absolute"]
    )


def test_benchmark_b_loader_rejects_unknown_case():
    with pytest.raises(ValueError, match="Unsupported Benchmark B"):
        load_benchmark_b_spec("B3")


@pytest.mark.parametrize("case_id", ["B1-fringing-pipe", "B2-fringing-square"])
def test_benchmark_b_field_profile_is_cell_centered_monotone_and_bounded(case_id):
    profile = build_benchmark_b_field_profile(case_id, axial_stations=101)

    assert profile.axis == "y"
    assert profile.x.shape == (101,)
    assert profile.field_scale.shape == (101,)
    assert float(profile.x[0]) > -15.0
    assert float(profile.x[-1]) < 10.0
    assert min(profile.field_scale) >= 0.0
    assert max(profile.field_scale) <= 1.0
    assert all(
        right <= left
        for left, right in zip(profile.field_scale, profile.field_scale[1:])
    )


def test_benchmark_b_field_profile_rejects_degenerate_station_count():
    with pytest.raises(ValueError, match="at least 2"):
        build_benchmark_b_field_profile("B1-fringing-pipe", axial_stations=1)


@pytest.mark.parametrize(
    ("case_id", "expected_shape", "expected_conductance", "expected_acceleration"),
    [
        ("B1-fringing-pipe", (101, 64, 128), 0.027, "anderson"),
        ("B2-fringing-square", (101, 65, 65), 0.07, "aitken"),
    ],
)
def test_benchmark_b_problem_binds_frozen_nondimensional_contract(
    case_id, expected_shape, expected_conductance, expected_acceleration
):
    problem = build_benchmark_b_problem(case_id, mesh_level="coarse")
    case = problem.case
    fluid, wall = case.regions
    recovered_conductance = wall.conductivity * wall.wall_thickness / fluid.conductivity

    assert case.geometry.nx == expected_shape[0]
    if case_id == "B1-fringing-pipe":
        assert case.geometry.nr == expected_shape[1]
        assert case.geometry.ntheta == expected_shape[2]
    else:
        assert case.geometry.ny == expected_shape[1]
        assert case.geometry.nz == expected_shape[2]
    assert recovered_conductance == pytest.approx(expected_conductance)
    assert case.solver.coupling_acceleration == expected_acceleration
    if case_id == "B2-fringing-square":
        assert case.solver.coupling_max_relaxation == pytest.approx(2.0)
    assert case.initial_velocity == 1.0
    assert case.forcing == 0.0
    assert case.geometry.axial_origin == -15.0
    assert problem.profile.axis == "y"
    inlet, outlet = case.boundary_conditions[1:3]
    expected_flow = 3.141592653589793 if case_id == "B1-fringing-pipe" else 4.0
    assert (inlet.name, inlet.kind, inlet.axis, inlet.value) == (
        "inlet",
        "inlet_flow_rate",
        "x",
        pytest.approx(expected_flow),
    )
    assert (outlet.name, outlet.kind, outlet.axis, outlet.value) == (
        "outlet",
        "outlet_pressure",
        "x",
        0.0,
    )

    mesh = _cross_section_mesh(case)
    assert mesh.x_centers.tolist() == pytest.approx(problem.profile.x.tolist())


def test_benchmark_b_pipe_mesh_resolves_frozen_hartmann_layer():
    problem = build_benchmark_b_problem("B1-fringing-pipe", mesh_level="coarse")
    mesh = _cross_section_mesh(problem.case)
    radial_cells = int(problem.case.geometry.nr)
    layer_cells = int(problem.case.geometry.hartmann_layer_cells)
    resolved_thickness = float(
        mesh.y_faces[radial_cells] - mesh.y_faces[radial_cells - layer_cells]
    )
    assert resolved_thickness == pytest.approx(1.0 / 6600.0, rel=2.0e-4)
    assert float(mesh.y_faces[radial_cells]) == pytest.approx(1.0)


def test_benchmark_b_problem_rejects_unfrozen_choices():
    with pytest.raises(ValueError, match="mesh_level"):
        build_benchmark_b_problem("B1-fringing-pipe", mesh_level="tiny")
    with pytest.raises(ValueError, match="wall_realization"):
        build_benchmark_b_problem(
            "B1-fringing-pipe", mesh_level="coarse", wall_realization="thick"
        )
    with pytest.raises(ValueError, match="num_devices"):
        build_benchmark_b_problem(
            "B2-fringing-square", mesh_level="coarse", num_devices=0
        )


def test_benchmark_b_sharded_mesh_rounds_frozen_axial_minimum_upward():
    problem = build_benchmark_b_problem(
        "B2-fringing-square", mesh_level="coarse", num_devices=2
    )
    assert problem.case.geometry.nx == 102
    assert problem.profile.x.size == 102


def test_benchmark_b1_reduced_production_path_closes_fixed_flow_and_is_finite():
    problem = build_benchmark_b_problem("B1-fringing-pipe", mesh_level="coarse")
    # Preserve radial, azimuthal, wall, restart, and retained-modal paths on the
    # smallest mesh that still closes every production-path conservation gate.
    case = replace(
        problem.case,
        geometry=replace(
            problem.case.geometry,
            nx=7,
            nr=7,
            ntheta=12,
            wall_cells=(1, 1, 1, 1),
            target_ha=20.0,
            hartmann_layer_cells=2,
        ),
        time_stepper=replace(
            problem.case.time_stepper,
            max_steps=2,
            potential_iterations=200,
        ),
        solver=replace(
            problem.case.solver,
            coupling_iterations=2,
            coupling_tolerance=1.0e-9,
        ),
    )
    profile = build_benchmark_b_field_profile(
        "B1-fringing-pipe", axial_stations=case.geometry.nx
    )
    reduced_problem = replace(problem, case=case, profile=profile)
    progress = []
    solution = solve_extruded_inductionless(
        reduced_problem,
        progress_callback=progress.append,
        checkpoint_interval=1,
    )

    assert [item.step for item in progress] == list(range(1, len(progress) + 1))
    assert all(isinstance(item.potential_residual, float) for item in progress)
    assert all(isinstance(value, float) for value in progress[-1].component_residuals)
    assert all(item.checkpoint is not None for item in progress)
    assert progress[-1].checkpoint.u.shape == solution.bundle.u.shape
    assert progress[-1].checkpoint.iteration_residual_history.size == len(progress)
    assert float(benchmarks.jnp.mean(solution.bundle.mean_velocity)) == pytest.approx(
        1.0, abs=1.0e-8
    )
    assert solution.validation.mean_velocity_span < 1.0e-3
    assert solution.validation.volumetric_flow_rate_span < 1.0e-3
    assert solution.validation.max_charge_balance_residual < 1.0e-3
    assert solution.validation.net_boundary_current_residual < 1.0e-3
    assert benchmarks.jnp.allclose(solution.bundle.u[:, -1, :], 0.0)
    assert benchmarks.jnp.isfinite(solution.bundle.p).all()
    assert benchmarks.jnp.isfinite(solution.bundle.phi).all()
    assert benchmarks.jnp.isfinite(solution.bundle.axial_pressure_loss_gradient).all()
    _assert_iteration_histories(solution.bundle)

    restarted = solve_extruded_inductionless(
        reduced_problem, initial_bundle=solution.bundle
    )
    assert float(benchmarks.jnp.mean(restarted.bundle.mean_velocity)) == pytest.approx(
        1.0, abs=1.0e-8
    )
    assert restarted.bundle.iteration_pressure_residual_history[0] < 1.0e-3
    assert restarted.validation.max_charge_balance_residual < 1.0e-3

    with pytest.raises(ValueError, match="checkpoint_interval"):
        solve_extruded_inductionless(reduced_problem, checkpoint_interval=0)


def test_benchmark_b2_reduced_path_closes_boundaries_and_restarts_exactly(tmp_path):
    problem = build_benchmark_b_problem("B2-fringing-square", mesh_level="coarse")
    case = replace(
        problem.case,
        geometry=replace(
            problem.case.geometry,
            nx=7,
            ny=7,
            nz=7,
            wall_cells=(1, 1, 1, 1),
            target_ha=20.0,
            hartmann_layer_cells=2,
        ),
        time_stepper=replace(
            problem.case.time_stepper,
            max_steps=3,
            potential_iterations=160,
        ),
        solver=replace(
            problem.case.solver,
            coupling_iterations=2,
            coupling_tolerance=1.0e-9,
        ),
    )
    profile = build_benchmark_b_field_profile(
        "B2-fringing-square", axial_stations=case.geometry.nx
    )
    progress = []
    solution = solve_extruded_inductionless(
        replace(problem, case=case, profile=profile),
        progress_callback=progress.append,
        checkpoint_interval=1,
    )

    assert float(benchmarks.jnp.mean(solution.bundle.mean_velocity)) == pytest.approx(
        1.0, abs=1.0e-10
    )
    assert solution.validation.max_charge_balance_residual < 1.0e-3
    assert solution.validation.net_boundary_current_residual < 1.0e-3
    history = _assert_iteration_histories(solution.bundle)
    assert solution.bundle.iteration_courant_history.shape == (history.size, 3)
    assert benchmarks.jnp.all(solution.bundle.iteration_courant_history >= 0.0)
    assert solution.bundle.stopping_state[0] == history.size

    fx, fy, fz = _unpack_duct_mass_flux(
        solution.bundle.rho_phi_plus, solution.bundle.rho_phi_inlet
    )
    flux_divergence = (
        fx[1:] - fx[:-1] + fy[:, 1:] - fy[:, :-1] + fz[:, :, 1:] - fz[:, :, :-1]
    )
    inlet_flux, outlet_flux = map(
        float, (benchmarks.jnp.sum(fx[0]), benchmarks.jnp.sum(fx[-1]))
    )
    assert inlet_flux > 0.0
    assert outlet_flux == pytest.approx(inlet_flux, abs=1.0e-10)
    assert float(benchmarks.jnp.max(benchmarks.jnp.abs(flux_divergence))) < 1.0e-8

    path = write_extruded_bundle_restart_npz(
        progress[0].checkpoint, case, tmp_path / "b2.npz"
    )
    restart = load_extruded_restart_bundle(path)
    continuation_case = replace(
        case, time_stepper=replace(case.time_stepper, max_steps=2)
    )
    with pytest.raises(ValueError, match="both compact flux arrays"):
        solve_extruded_inductionless(
            replace(problem, case=continuation_case, profile=profile),
            initial_bundle=replace(restart.bundle, rho_phi_inlet=None))
    resumed = solve_extruded_inductionless(
        replace(problem, case=continuation_case, profile=profile),
        initial_bundle=restart.bundle,
    )
    assert restart.metadata["restart_schema"] == "b2_diagnostics_v2"
    for name in (
        "u",
        "v",
        "w",
        "p",
        "phi",
        "rho_phi_plus",
        "rho_phi_inlet",
        "iteration_residual_history",
        "iteration_component_residual_history",
        "iteration_pressure_residual_history",
        "iteration_electric_linear_history",
        "iteration_potential_residual_history",
        "iteration_courant_history",
    ):
        assert benchmarks.jnp.allclose(
            getattr(resumed.bundle, name),
            getattr(solution.bundle, name),
            rtol=0.0,
            atol=1.0e-12,
        )
    assert resumed.bundle.stopping_state == solution.bundle.stopping_state


def test_benchmark_b_primary_pressure_observables_use_direct_fields():
    b1 = SimpleNamespace(
        bundle=SimpleNamespace(
            x=benchmarks.jnp.asarray([-15.0, 0.0, 7.5, 10.0]),
            axial_pressure_loss_gradient=benchmarks.jnp.asarray([4.0, 3.0, 2.0, 2.0]),
            transverse_pressure_difference=benchmarks.jnp.zeros(4),
        )
    )
    b2 = SimpleNamespace(
        bundle=SimpleNamespace(
            x=benchmarks.jnp.asarray([-15.0, 0.0, 10.0]),
            axial_pressure_loss_gradient=benchmarks.jnp.zeros(3),
            transverse_pressure_difference=benchmarks.jnp.asarray([0.0, 54.0, 0.0]),
        )
    )
    assert benchmark_b_pressure_observable(
        b1, "B1-fringing-pipe"
    ).tolist() == pytest.approx([2.0 / 10700.0, 1.0 / 10700.0, 0.0, 0.0])
    assert benchmark_b_pressure_observable(
        b2, "B2-fringing-square"
    ).tolist() == pytest.approx([0.0, 0.1, 0.0])


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("id",), "B3", "Unsupported Benchmark B"),
        (("schema_version",), 0, "schema 1"),
        (("tolerances_frozen_before_production",), False, "tolerances"),
        (("physics", "hartmann_number"), 1.0, "parameters differ"),
        (_EQUATIONS + ("inertia",), "omitted", _SEMANTICS),
        (_EQUATIONS + ("gradient_discretization",), "Gauss linear", _SEMANTICS),
        (
            _SHARED + ("boundary_drive", "flow_constraint_scope"),
            "stationwise",
            _SEMANTICS,
        ),
        (_FREEMHD_COMMIT, "0" * 40, "FreeMHD discretization reference differs"),
        (("matched_contract", "roles"), {}, "matched production role differs"),
        (_STEADY_STEPS, 2, _STOPPING),
        (("harness_smoke_execution", "restart_absolute_tolerance"), 1.0e-6, "smoke execution contract"),
        (("sources",), [], "both review"),
        (("sources", 0, "pages"), "", "pages"),
        (("field", "representation"), "spline", "field reconstruction"),
        (_MESH_LEVELS + (0, "name"), "tiny", "coarse, medium, and fine"),
        (_MESH_LEVELS + (1, "axial_stations_min"), None, "must increase"),
        (_MESH_LEVELS + (1, "radial_cells_min"), None, "radial_cells_min"),
        (("wall", "confirmation_thickness_over_L"), None, "thin-wall"),
        (("acceptance", "weighted_rms_max"), 2.0, "acceptance contract"),
        (("data_rights", "redistribution"), "none", "redistribution policy"),
        (("reference", "data_sha256"), "0" * 64, "SHA-256"),
    ],
)
def test_benchmark_b_spec_validation_rejects_contract_drift(path, value, message):
    case_id = "B2-fringing-square" if message in {_STOPPING, "smoke execution contract"} else "B1-fringing-pipe"
    spec = deepcopy(load_benchmark_b_spec(case_id))
    _set_nested(spec, path, value)
    with pytest.raises(ValueError, match=message):
        benchmarks._validate_benchmark_b_spec(spec, Path.cwd())


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (["bad,header", "0,1"], "columns"),
        ([_REFERENCE_HEADER, "-15,nan,0.1,0,0.1"], "finite"),
        (
            [_REFERENCE_HEADER, *[f"{x},1,0.1,0,0.1" for x in range(9)]],
            "strictly increasing",
        ),
    ],
)
def test_benchmark_b_reference_rejects_malformed_data(
    tmp_path, monkeypatch, rows, message
):
    _install_reference(tmp_path, monkeypatch, rows)
    with pytest.raises(ValueError, match=message):
        load_benchmark_b_reference("B1-fringing-pipe", tmp_path)


@pytest.mark.parametrize(
    ("x_start", "bad_column", "bad_value", "message"),
    [
        (-14.0, None, None, "span"),
        (-15.0, "b_uncertainty", 0.0, "uncertainties"),
        (-15.0, "b_over_B0", 2.0, "physical range"),
    ],
)
def test_benchmark_b_reference_rejects_physical_contract_violations(
    tmp_path, monkeypatch, x_start, bad_column, bad_value, message
):
    columns = _REFERENCE_HEADER.split(",")
    rows = [",".join(columns)]
    for index in range(10):
        values = {
            "x_over_L": x_start + (10.0 - x_start) * index / 9,
            "b_over_B0": 1.0 - index / 10,
            "b_uncertainty": 0.1,
            "pressure_observable": 0.0,
            "pressure_uncertainty": 0.1,
        }
        if index == 5 and bad_column is not None:
            values[bad_column] = bad_value
        rows.append(",".join(str(values[column]) for column in columns))
    _install_reference(tmp_path, monkeypatch, rows)
    with pytest.raises(ValueError, match=message):
        load_benchmark_b_reference("B1-fringing-pipe", tmp_path)


def test_benchmark_b_field_profile_rejects_extrapolation(monkeypatch):
    spec = deepcopy(load_benchmark_b_spec("B1-fringing-pipe"))
    reference = load_benchmark_b_reference("B1-fringing-pipe")
    spec["geometry"]["x_over_L_min"] = -20.0
    monkeypatch.setattr(benchmarks, "load_benchmark_b_spec", lambda *_: spec)
    monkeypatch.setattr(benchmarks, "load_benchmark_b_reference", lambda *_: reference)
    with pytest.raises(ValueError, match="cannot extrapolate"):
        build_benchmark_b_field_profile("B1-fringing-pipe", axial_stations=20)


def test_benchmark_b_field_profile_rejects_nonmonotone_field(monkeypatch):
    spec = load_benchmark_b_spec("B1-fringing-pipe")
    reference = dict(load_benchmark_b_reference("B1-fringing-pipe"))
    field = list(reference["b_over_B0"])
    field[8] = field[7] + 0.1
    reference["b_over_B0"] = tuple(field)
    monkeypatch.setattr(benchmarks, "load_benchmark_b_spec", lambda *_: spec)
    monkeypatch.setattr(benchmarks, "load_benchmark_b_reference", lambda *_: reference)
    with pytest.raises(ValueError, match="must remain monotone"):
        build_benchmark_b_field_profile("B1-fringing-pipe", axial_stations=101)


@pytest.mark.parametrize(
    ("case_id", "field", "message"),
    [
        ("B1-fringing-pipe", "axial_pressure_loss_gradient", "B1 requires"),
        ("B2-fringing-square", "transverse_pressure_difference", "B2 requires"),
    ],
)
def test_benchmark_b_pressure_observable_rejects_missing_direct_fields(
    case_id, field, message
):
    solution = SimpleNamespace(
        bundle=SimpleNamespace(**{field: benchmarks.jnp.zeros(0)})
    )
    with pytest.raises(ValueError, match=message):
        benchmark_b_pressure_observable(solution, case_id)


def test_benchmark_b_pressure_observable_has_coordinate_free_fallback():
    solution = SimpleNamespace(
        bundle=SimpleNamespace(
            axial_pressure_loss_gradient=benchmarks.jnp.asarray([3.0, 2.0, 1.0])
        )
    )
    assert benchmark_b_pressure_observable(
        solution, "B1-fringing-pipe"
    ).tolist() == pytest.approx([1.0 / 10700.0, 0.0, -1.0 / 10700.0])

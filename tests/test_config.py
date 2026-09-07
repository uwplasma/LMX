import ast
import inspect
import json
import os
import re
import subprocess
import sys
import tarfile
import textwrap
import zipfile
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib

import lmx
from lmx.cases import _wall_conductivity_from_conductance_ratio
from lmx.specs import _parse_boundary_value, load_run_config
from scripts.audit_architecture import (
    _checkout_size,
    architecture_budget_errors,
    build_inventory,
    inspect_sdist,
    inspect_wheel,
    measure_import,
)
from scripts.run_full_test_suite import _ALL_TESTS, _test_environment, _tests_for_changes

pytestmark = pytest.mark.unit


def test_ci_tiers_cover_collection_without_overlapping_pr_work():
    from scripts.run_full_test_suite import _TEST_TIERS

    def collect(expression):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests",
                "--collect-only",
                "-o",
                "addopts=",
                "-q",
                "-m",
                expression,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode in (0, 5), result.stdout + result.stderr
        return {line for line in result.stdout.splitlines() if line.startswith("tests/") and "::" in line}

    all_tests = collect("")
    tiers = {name: collect(expression) for name, expression in _TEST_TIERS.items()}
    assert all_tests and all_tests == set.union(*tiers.values())
    assert tiers["unit"] and tiers["regression"]
    assert not tiers["unit"] & tiers["regression"]
    deferred = tiers["slow"] | tiers["gpu"] | tiers["external"]
    assert not (tiers["unit"] | tiers["regression"]) & deferred

    workflow = Path(".github/workflows/ci.yml").read_text()
    for job in ("compatibility", "coverage"):
        # Read the job condition without relying on its step implementation.
        condition = workflow.split(f"\n  {job}:\n", 1)[1].split("    if: ", 1)[1].splitlines()[0]
        assert "github.event_name != 'pull_request'" in condition
    pr_job = re.split(r"\n  \w[\w-]*:\n", workflow.split("\n  pr-tests:\n", 1)[1])[0]
    assert "--no-coverage" in pr_job
    # Heavy evidence must keep a per-test timeout above the slowest recorded case.
    assert "--test-timeout-seconds 300" in pr_job

    entries = [
        (
            block.split("tier: ", 1)[1].splitlines()[0].strip(),
            block.split("shard: ", 1)[1].splitlines()[0].strip(' "'),
        )
        for block in pr_job.split("- tier: ")[1:]
        for block in ["tier: " + block]
    ]
    assert entries, pr_job
    covered: set[str] = set()
    for tier, shard in entries:
        selection = tiers[tier] if not shard else tiers[tier] & _shard_members(all_tests, shard)
        assert selection, f"PR job {tier}/{shard} selects nothing"
        assert not covered & selection, f"PR job {tier}/{shard} repeats work"
        covered |= selection
    assert covered == tiers["unit"] | tiers["regression"]


def _shard_members(all_tests: set[str], shard: str) -> set[str]:
    """Return the tests a file shard runs, mirroring the runner's own selection."""
    from scripts.run_full_test_suite import _HEAVY_FRINGING_TEST, _TEST_SHARDS

    entries = _TEST_SHARDS[shard]
    files = {entry for entry in entries if "::" not in entry}
    nodes = tuple(entry for entry in entries if "::" in entry)
    members = {test for test in all_tests if test.split("::", 1)[0] in files}
    members |= {test for test in all_tests if test.startswith(nodes)} if nodes else set()
    if shard == "fringing":
        members -= {test for test in all_tests if _HEAVY_FRINGING_TEST in test}
    return members


@pytest.mark.parametrize("x64", ["false", "true"])
def test_explicit_precision_in_fresh_process(x64):
    code = """
import warnings
import jax
import jax.numpy as jnp
import lmx
from dataclasses import replace
from lmx import cases, mesh, physics, fringing, q2d
initial = jax.config.x64_enabled
assert initial == EXPECTED
case32 = lmx.make_hartmann_case(ha=2, ny=8, nz=8, dtype="float32")
assert jax.config.x64_enabled == initial
values = []
for dtype in ("float32", "float64"):
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        case = lmx.make_hartmann_case(ha=2, ny=8, nz=8, dtype=dtype)
    assert len(recorded) == int(dtype == "float64" and not initial)
    if recorded:
        assert recorded[0].category is DeprecationWarning
    grid, materials, _, _ = cases._prepare_fully_developed_case(case)
    assert grid.y_faces.dtype == materials.conductivity.dtype == case.dtype
    objective = lambda x: jnp.mean(lmx.solve_fully_developed_fields(case, forcing=x)[0])
    x = jnp.asarray(1., dtype=case.dtype)
    value, grad = jax.jit(jax.value_and_grad(objective))(x)
    tangent = jax.jit(lambda x: jax.jvp(objective, (x,), (jnp.ones_like(x),))[1])(x)
    assert value.dtype == grad.dtype == tangent.dtype == case.dtype
    assert bool(jnp.isfinite(grad))
    # Linear forcing response is an exact derivative oracle at fixed field.
    assert bool(jnp.allclose(grad, value, rtol=2e-5, atol=1e-7))
    assert bool(jnp.allclose(grad, tangent, rtol=2e-5, atol=1e-7))
    values.append(float(value))
    short = replace(case, time_stepper=replace(case.time_stepper, max_steps=2))
    result = lmx.solve(short)
    assert result.state.u.dtype == result.state.phi.dtype == case.dtype
assert abs(values[0] - values[1]) < 2e-6
lmx.enable_x64()
assert jnp.asarray(1.).dtype == jnp.float64
""".replace("EXPECTED", str(x64 == "true"))
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        timeout=90,
        env={**os.environ, "JAX_ENABLE_X64": x64},
    )


def _write_minimal_config(
    tmp_path: Path,
    name: str,
    *,
    case_extra: str = "",
    geometry_kind: str | None = "rect_duct",
    geometry_extra: str = "",
    magnetic_kind: str = "constant",
    solver: str = "",
) -> Path:
    geometry_type = "" if geometry_kind is None else f'kind = "{geometry_kind}"'
    magnetic_value = "value = [0.0, 0.0, 1.0]" if magnetic_kind == "constant" else ""
    solver_table = f"[solver]\n{solver}" if solver else ""
    path = tmp_path / f"{name}.toml"
    path.write_text(
        f"""
[case]
name = "{name}"
{case_extra}

[geometry]
{geometry_type}
width = 1.0
height = 1.0
ny = 4
nz = 4
{geometry_extra}

[magnetic_field]
kind = "{magnetic_kind}"
{magnetic_value}

{solver_table}

[time_stepper]
dt = 0.1
t_final = 0.1
max_steps = 1

[[regions]]
name = "fluid"
kind = "fluid"
conductivity = 1.0

[[boundary_conditions]]
name = "wall"
kind = "no_slip"
""".strip()
    )
    return path


def test_every_test_file_belongs_to_a_covered_shard():
    """A file outside every shard never runs in the coverage lane and scores zero."""
    from scripts.run_full_test_suite import _TEST_SHARDS

    sharded = {entry.split("::", 1)[0] for shard in _TEST_SHARDS.values() for entry in shard}
    present = {f"tests/{path.name}" for path in Path("tests").glob("test_*.py")}
    assert present - sharded == set(), "add these files to a shard in _TEST_SHARDS"
    assert sharded - present == set(), "these shard entries no longer exist"


@pytest.mark.parametrize(
    "changed,full,targeted,docs,external",
    [
        (".github/workflows/ci.yml", "true", "false", "false", "false"),
        (".github/workflows/docs.yml", "true", "false", "true", "false"),
        (".github/workflows/external-validation.yml", "true", "false", "false", "true"),
        (".github/actions/setup-lmx/action.yml", "true", "false", "true", "true"),
        ("src/lmx/mesh.py", "true", "false", "true", "true"),
        ("src/lmx/q2d.py", "true", "false", "true", "false"),
        ("src/lmx/validation.py", "true", "false", "true", "true"),
        ("tests/test_mesh.py", "true", "false", "false", "false"),
        ("scripts/run_full_test_suite.py", "false", "true", "false", "false"),
        ("docs/index.md", "false", "false", "true", "false"),
        ("README.md", "false", "false", "true", "false"),
    ],
)
def test_ci_scope_and_superseded_work_policy(tmp_path, changed, full, targeted, docs, external):
    root = Path(__file__).resolve().parents[1] / ".github/workflows"

    def git(*args):
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=LMX test",
                "-c",
                "user.email=test@example.invalid",
                "-c",
                "core.hooksPath=/dev/null",
                "-c",
                "commit.gpgsign=false",
                *args,
            ],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

    git("init", "-q")
    git("commit", "--allow-empty", "-qm", "baseline")
    git("update-ref", "refs/remotes/origin/main", "HEAD")
    path = tmp_path / changed
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("BENCHMARK_B\n")
    git("add", changed)
    git("commit", "-qm", "candidate")
    groups = set()
    for name, expected in (
        ("ci", {"full": full, "targeted": targeted}),
        ("docs", {"docs": docs}),
        ("external-validation", {"b2": external}),
    ):
        workflow = (root / f"{name}.yml").read_text()
        script = textwrap.dedent(workflow.split("run: |\n", 1)[1].split("\n\n  ", 1)[0])
        script = script.replace("${{ github.event_name }}", "pull_request").replace(
            "${{ github.base_ref }}", "main"
        )
        output = tmp_path / f"{name}.outputs"
        subprocess.run(
            ["bash", "-e", "-o", "pipefail", "-c", script],
            cwd=tmp_path,
            check=True,
            env={**os.environ, "GITHUB_OUTPUT": str(output)},
        )
        assert dict(line.split("=", 1) for line in output.read_text().splitlines()) == expected
        policy = workflow.split("concurrency:\n", 1)[1].split("\n\n", 1)[0]
        assert "cancel-in-progress: true" in policy and "github.event.pull_request.number" in policy
        group = policy.split("group: ", 1)[1].splitlines()[0]
        assert group not in groups
        groups.add(group)


def test_load_run_config_reads_complete_toml(tmp_path: Path):
    input_file = tmp_path / "hartmann.toml"
    input_file.write_text(
        """
[case]
name = "hartmann_toml_demo"
forcing = 1.0
initial_velocity = 0.0
reference_pressure_gradient = -1.0
reference_phi_cell = [0, 0]
notes = "unit test"

[geometry]
kind = "rect_duct"
width = 2.0
height = 2.0
length = 1.0
nx = 1
ny = 8
nz = 8
wall_thickness = [0.0, 0.0, 0.0, 0.0]
wall_cells = [0, 0, 0, 0]
target_ha = 20.0

[magnetic_field]
kind = "constant"
value = [0.0, 0.0, 20.0]
ramp_start = 0.0
ramp_duration = 0.0

[solver]
kind = "fully_developed_inductionless"
mode = "steady"
preconditioner = "jacobi"
time_scheme = "implicit_euler"
coupling_iterations = 9
coupling_tolerance = 1.0e-7
coupling_acceleration = "aitken"
coupling_min_relaxation = 0.1
coupling_max_relaxation = 12.0
coupling_history_depth = 5
coupling_regularization = 1.0e-9
coupling_damping = 0.8

[time_stepper]
dt = 0.001
t_final = 0.01
max_steps = 10
potential_iterations = 100
potential_relaxation = 1.0
potential_solver = "cg"
steady_tolerance = 1e-8

[output]
directory = "./out"
write_paraview = true
write_csv_profiles = true
write_npz = true
write_json_summary = true
write_plots = false
copy_input_file = true
write_stride = 1
history_stride = 3

[logging]
enabled = true
banner = true
print_footer = true
flush = true
step_stride = 2

[restart]
enabled = true
path = "./previous_results.npz"
reset_histories = false
write_restart = true
restart_filename = "hartmann_restart.npz"

[[regions]]
name = "fluid"
kind = "fluid"
conductivity = 1.0
density = 1.0
viscosity = 0.01

[[boundary_conditions]]
name = "y_min_wall"
kind = "no_slip"
axis = "y"
side = "min"

[[boundary_conditions]]
name = "y_max_wall"
kind = "no_slip"
axis = "y"
side = "max"

[[boundary_conditions]]
name = "z_min_wall"
kind = "insulating"
axis = "z"
side = "min"

[[boundary_conditions]]
name = "z_max_wall"
kind = "insulating"
axis = "z"
side = "max"
""".strip()
    )

    config = load_run_config(input_file)

    assert config.case.name == "hartmann_toml_demo"
    assert config.case.geometry.kind == "rect_duct"
    assert config.case.output.directory == str((tmp_path / "out").resolve())
    assert config.case.output.history_stride == 3
    assert config.case.solver.kind == "fully_developed_inductionless"
    assert config.case.solver.mode == "steady"
    assert config.case.solver.preconditioner == "jacobi"
    assert config.case.solver.coupling_iterations == 9
    assert config.case.solver.coupling_acceleration == "aitken"
    assert config.case.solver.coupling_min_relaxation == pytest.approx(0.1)
    assert config.case.solver.coupling_max_relaxation == pytest.approx(12.0)
    assert config.case.solver.coupling_history_depth == 5
    assert config.case.solver.coupling_regularization == pytest.approx(1.0e-9)
    assert config.case.solver.coupling_damping == pytest.approx(0.8)
    assert config.case.time_stepper.potential_solver == "cg"
    assert config.logging.step_stride == 2
    assert config.restart.enabled is True
    assert config.restart.path == (tmp_path / "previous_results.npz").resolve()
    assert config.restart.reset_histories is False
    assert config.restart.write_restart is True
    assert config.restart.restart_filename == "hartmann_restart.npz"
    assert config.fringing.enabled is False
    assert len(config.case.regions) == 1
    assert len(config.case.boundary_conditions) == 4


@pytest.mark.parametrize("formulation", ["stokes_projection", "b1_finite_volume", "b2_finite_volume"])
def test_load_run_config_reads_extruded_fringing_controls(tmp_path: Path, formulation):
    input_file = tmp_path / "fringing.toml"
    input_file.write_text(
        """
[case]
name = "fringing_rect_demo"

[geometry]
kind = "rect_duct"
width = 2.0
height = 2.0
length = 6.0
nx = 7
ny = 6
nz = 6

[magnetic_field]
kind = "constant"
value = [0.0, 0.0, 8.0]

[solver]
kind = "extruded_inductionless"
mode = "steady"

[fringing]
enabled = true
entry_center = 1.0
exit_center = 4.0
transition_width = 0.5
axis = "z"

[time_stepper]
dt = 0.01
t_final = 0.1
max_steps = 8

[[regions]]
name = "fluid"
kind = "fluid"
conductivity = 1.0
density = 1.0
viscosity = 0.05

[[boundary_conditions]]
name = "wall"
kind = "no_slip"
""".strip().replace('mode = "steady"', f'mode = "steady"\nextruded_formulation = "{formulation}"')
    )

    config = load_run_config(input_file)

    assert config.case.solver.kind == "extruded_inductionless"
    assert config.case.solver.extruded_formulation == formulation
    assert config.fringing.enabled is True
    assert config.fringing.entry_center == pytest.approx(1.0)
    assert config.fringing.exit_center == pytest.approx(4.0)
    assert config.fringing.transition_width == pytest.approx(0.5)
    assert config.fringing.axis == "z"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"magnetic_kind": "analytic"}, "analytic magnetic-field"),
        ({"solver": 'kind = "invalid"'}, "Unsupported solver kind"),
        ({"solver": 'extruded_formulation = "invalid"'}, "Unsupported extruded formulation"),
        ({"geometry_kind": None}, "Missing required TOML key 'kind'"),
        ({"geometry_extra": "wall_thickness = [0.1, 0.2]"}, "must have length 4"),
        ({"solver": 'mode = "invalid"'}, "Unsupported solve mode"),
    ],
)
def test_load_run_config_rejects_invalid_inputs(tmp_path: Path, kwargs: dict[str, str | None], message: str):
    input_file = _write_minimal_config(tmp_path, "rejected", **kwargs)

    with pytest.raises(ValueError, match=message):
        load_run_config(input_file)


def test_shipped_example_toml_files_parse():
    root = Path(__file__).resolve().parents[1]

    for relative in ("examples/hartmann_case.toml",):
        config = load_run_config(root / relative)
        assert config.case.name
        assert config.case.regions


def test_tutorials_map_to_executable_examples_or_numerical_tests():
    root = Path(__file__).resolve().parents[1]
    tutorial_paths = sorted((root / "docs/tutorials").glob("*.md"))
    expected = {
        "differentiation.md",
        "fringing.md",
        "fully_developed.md",
        "q2d.md",
        "walls_and_fields.md",
    }
    assert {path.name for path in tutorial_paths} == expected

    catalog = tomllib.loads((root / "examples/catalog.toml").read_text())
    documented = {Path(item["docs"]).name for item in catalog["example"]}
    index = (root / "docs/index.md").read_text()
    for path in tutorial_paths:
        if path.name != "differentiation.md":
            assert path.name in documented
        assert f"tutorials/{path.stem}" in index
        assert "```python" in path.read_text()


def test_parse_boundary_value_accepts_scalar_and_vector_and_rejects_bad_inputs():
    assert _parse_boundary_value(None) is None
    assert _parse_boundary_value(1.25) == pytest.approx(1.25)
    assert _parse_boundary_value([1, 2, 3]) == pytest.approx((1.0, 2.0, 3.0))

    with pytest.raises(ValueError, match="length 3"):
        _parse_boundary_value([1, 2])

    with pytest.raises(ValueError, match="Unsupported boundary-condition value"):
        _parse_boundary_value({"bad": True})


@pytest.mark.parametrize(
    ("wall_thickness", "hartmann_half_spacing", "message"),
    ((0.0, 1.0, "wall_thickness"), (1.0, 0.0, "hartmann_half_spacing")),
)
def test_wall_conductivity_rejects_nonpositive_geometry(
    wall_thickness: float, hartmann_half_spacing: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _wall_conductivity_from_conductance_ratio(
            wall_conductance_ratio=1.0,
            fluid_conductivity=1.0,
            wall_thickness=wall_thickness,
            hartmann_half_spacing=hartmann_half_spacing,
        )


EXPECTED_ROOT_API = {
    "enable_x64",
    "enable_compilation_cache",
    "make_hartmann_case",
    "make_shercliff_case",
    "make_hunt_case",
    "make_q2d_case",
    "evolve_q2d",
    "solve_fully_developed_fields",
    "Q2DProblem",
    "solve",
    "generate_rect_duct_mesh",
    "generate_rect_duct_mesh_from_faces",
    "generate_layered_duct_mesh",
    "generate_layered_duct_mesh_from_fluid_faces",
    "generate_multilayer_duct_mesh",
    "WallLayer",
    "dynamic_to_kinematic_viscosity",
    "kinematic_to_dynamic_viscosity",
    "hartmann_number",
    "reynolds_number",
    "interaction_parameter",
    "magnetic_reynolds_number",
    "magnetic_field_from_hartmann",
    "wall_conductance_ratio",
    "effective_pinhole_conductance_ratio",
    "tangential_stack_conductance_ratio",
    "normal_stack_leakage_ratio",
    "equivalent_single_layer",
    "nested_wall_layer_resolution_summary",
}


def test_architecture_inventory_is_deterministic_without_timing() -> None:
    assert build_inventory() == build_inventory()
    assert Path(_test_environment()["PYTHONPATH"].split(os.pathsep)[0]) == Path("src").resolve()


def test_change_gate_selects_affected_tests_and_fails_closed() -> None:
    assert _tests_for_changes(("docs/index.md", "plan.md")) == ()
    assert _tests_for_changes(("src/lmx/_fringing_pipe.py",)) == (
        "tests/test_fringing.py",
        "tests/test_benchmarks.py",
        "tests/test_freemhd.py",
        "tests/test_example_runner.py",
    )
    assert _tests_for_changes(("src/lmx/q2d.py", "examples/q2d_vortex.py")) == (
        "tests/test_physics.py",
        "tests/test_example_runner.py",
    )
    assert _tests_for_changes(("tests/test_io.py",)) == ("tests/test_io.py",)
    assert _tests_for_changes(("unknown executable",)) == _ALL_TESTS


def test_stable_root_api_is_small_lazy_and_resolvable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert set(lmx.__all__) == EXPECTED_ROOT_API
    assert EXPECTED_ROOT_API <= set(dir(lmx))
    assert all(callable(getattr(lmx, name)) for name in lmx.__all__)
    assert not [name for name in lmx.__all__ if not inspect.getdoc(getattr(lmx, name))]
    api_reference = Path("docs/reference/api.md").read_text()
    assert all(f"`{name}`" in api_reference for name in lmx.__all__)

    updates = []
    monkeypatch.setattr("lmx.io.jax.config.update", lambda *args: updates.append(args))
    cache = lmx.enable_compilation_cache(
        tmp_path / "jax-cache", min_compile_time_secs=2.0, min_entry_size_bytes=4096
    )
    assert cache.is_dir()
    assert updates == [
        ("jax_compilation_cache_dir", str(cache)),
        ("jax_persistent_cache_min_entry_size_bytes", 4096),
        ("jax_persistent_cache_min_compile_time_secs", 2.0),
    ]


def test_advanced_api_uses_owning_module() -> None:
    assert not hasattr(lmx, "solve_extruded_inductionless")
    from lmx.fringing import solve_extruded_inductionless

    assert callable(solve_extruded_inductionless)


def test_unknown_root_attribute_has_standard_error() -> None:
    with pytest.raises(AttributeError, match="not_an_api"):
        lmx.not_an_api


def test_architecture_inventory_ignores_generated_egg_info(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "source.py").write_bytes(b"x")
    metadata = tmp_path / "package.egg-info"
    metadata.mkdir()
    (metadata / "PKG-INFO").write_bytes(b"generated")

    def missing_git(*_args, **_kwargs):
        raise FileNotFoundError

    monkeypatch.setattr("scripts.audit_architecture.subprocess.run", missing_git)
    assert _checkout_size(tmp_path) == 1


def test_root_import_is_lazy_and_within_budget() -> None:
    payload = build_inventory()
    payload["import_measurement"] = measure_import(repeats=3)
    assert architecture_budget_errors(payload) == []


def test_numerical_modules_do_not_import_optional_visualization() -> None:
    code = """
import sys
import lmx.io
assert not any(name == 'matplotlib' or name.startswith('matplotlib.') for name in sys.modules)
assert not any(name == 'PIL' or name.startswith('PIL.') for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_wheel_audit_rejects_nonpackage_payload(tmp_path: Path) -> None:
    wheel = tmp_path / "lmx-test.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("lmx/__init__.py", "")
        archive.writestr("lmx/py.typed", "")
        archive.writestr("lmx-1.dist-info/METADATA", "")
        archive.writestr("benchmarks/raw.bin", b"large output")
    assert inspect_wheel(wheel)["forbidden_members"] == ["benchmarks/raw.bin"]
    assert "outside lmx/" in architecture_budget_errors(build_inventory(), wheel=wheel)[0]


def test_root_api_is_pep561_marked_and_fully_annotated() -> None:
    assert (Path("src/lmx") / "py.typed").is_file()
    for name in lmx.__all__:
        value = getattr(lmx, name)
        if not (inspect.isfunction(value) or inspect.isclass(value)):
            continue
        signature = inspect.signature(value)
        assert signature.return_annotation is not inspect.Signature.empty, name
        assert all(
            parameter.annotation is not inspect.Parameter.empty
            for parameter in signature.parameters.values()
            if parameter.name not in {"self", "cls"}
        ), name


def test_sdist_audit_rejects_repository_tests(tmp_path: Path) -> None:
    source = tmp_path / "lmx-1" / "tests" / "test_solver.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"large output")
    sdist = tmp_path / "lmx-test.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        archive.add(source, arcname="lmx-1/tests/test_solver.py")
    assert inspect_sdist(sdist)["forbidden_members"] == ["tests/test_solver.py"]
    assert "outside its source payload" in architecture_budget_errors(build_inventory(), sdist=sdist)[0]


def test_curated_examples_use_submodules_and_linear_scripts_are_editable() -> None:
    inventory = build_inventory()["inventory"]
    stable = set(lmx.__all__)
    for item in inventory["curated_examples"]:
        path = Path(item["path"])
        if path.suffix != ".py":
            continue
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        imports = (
            node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module == "lmx"
        )
        root_imports = {alias.name for node in imports for alias in node.names}
        assert root_imports <= stable, f"{path} imports unsupported root APIs: {root_imports - stable}"
        linear_limits = {
            "fringing_benchmark_demo.py": 160,
            "hartmann_example.py": 160,
            "hunt_example.py": 160,
            "li_aln_wall_stack_example.py": 260,
            "q2d_turbulence_demo.py": 140,
            "variable_field_extruded_demo.py": 190,
        }
        if path.name in linear_limits:
            assert ast.get_docstring(tree)
            assert "# Inputs:" in source and "# Run" in source
            assert len(source.splitlines()) <= linear_limits[path.name]
            functions = (node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
            assert all(node.name != "main" and ast.get_docstring(node) for node in functions)
            assert "argparse" not in source and "__name__" not in source


def test_curated_examples_declare_user_facing_contracts(tmp_path: Path) -> None:
    inventory = build_inventory()["inventory"]
    curated = inventory["curated_examples"]
    assert {item["path"] for item in curated} == set(inventory["examples"])
    assert len(curated) == 7
    for item in curated:
        assert item["command"]
        assert item["outputs"]
        assert item["runtime"] in {"portable", "accelerator-optional"}
        assert Path(item["docs"]).is_file()
    q2d = Path(__file__).resolve().parents[1] / "examples/q2d_turbulence_demo.py"
    root = q2d.parents[1]
    environment = {
        **os.environ,
        "PATH": str(tmp_path),
        "PYTHONPATH": os.pathsep.join((str(root / "src"), os.environ.get("PYTHONPATH", ""))),
    }
    subprocess.run([sys.executable, q2d], cwd=tmp_path, timeout=30, check=True, env=environment)
    summary_path = next((tmp_path / "artifacts").rglob("q2d_vortex_decay.json"))
    summary = json.loads(summary_path.read_text())
    assert summary["status"] == "completed"
    assert summary["frames"] == 41
    assert summary["diagnostics"]["kinetic_energy_final"] < summary["diagnostics"]["kinetic_energy_initial"]
    assert (summary_path.parent / summary["poster"]).is_file()
    if summary["movie"] is not None:
        assert (summary_path.parent / summary["movie"]).is_file()

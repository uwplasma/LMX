import ast
import inspect
import json
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

import lmx
from lmx.cases import _wall_conductivity_from_conductance_ratio
from lmx.config import LoggingSpec, _parse_boundary_value, load_run_config
from scripts.audit_architecture import (
    _checkout_size,
    architecture_budget_errors,
    build_inventory,
    inspect_sdist,
    inspect_wheel,
    measure_import,
    write_inventory,
)

pytestmark = pytest.mark.unit


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
linear_solver = "solvax_pcg"
preconditioner = "block_jacobi"
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
checkpoint_stride = 1

[output]
directory = "./out"
write_paraview = true
write_csv_profiles = true
write_npz = true
write_json_summary = true
write_plots = false
copy_input_file = true
write_stride = 1

[logging]
enabled = true
verbose = true
verbosity = "debug"
banner = true
print_regions = true
print_boundaries = true
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
    assert config.case.solver.kind == "fully_developed_inductionless"
    assert config.case.solver.mode == "steady"
    assert config.case.solver.linear_solver == "solvax_pcg"
    assert config.case.solver.preconditioner == "block_jacobi"
    assert config.case.solver.coupling_iterations == 9
    assert config.case.solver.coupling_acceleration == "aitken"
    assert config.case.solver.coupling_min_relaxation == pytest.approx(0.1)
    assert config.case.solver.coupling_max_relaxation == pytest.approx(12.0)
    assert config.case.solver.coupling_history_depth == 5
    assert config.case.solver.coupling_regularization == pytest.approx(1.0e-9)
    assert config.case.solver.coupling_damping == pytest.approx(0.8)
    assert config.case.time_stepper.potential_solver == "cg"
    assert config.logging.verbosity == "debug"
    assert config.logging.step_stride == 2
    assert config.restart.enabled is True
    assert config.restart.path == (tmp_path / "previous_results.npz").resolve()
    assert config.restart.reset_histories is False
    assert config.restart.write_restart is True
    assert config.restart.restart_filename == "hartmann_restart.npz"
    assert config.fringing.enabled is False
    assert len(config.case.regions) == 1
    assert len(config.case.boundary_conditions) == 4


def test_load_run_config_reads_extruded_fringing_controls(tmp_path: Path):
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
""".strip()
    )

    config = load_run_config(input_file)

    assert config.case.solver.kind == "extruded_inductionless"
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


def test_parse_boundary_value_accepts_scalar_and_vector_and_rejects_bad_inputs():
    assert _parse_boundary_value(None) is None
    assert _parse_boundary_value(1.25) == pytest.approx(1.25)
    assert _parse_boundary_value([1, 2, 3]) == pytest.approx((1.0, 2.0, 3.0))

    with pytest.raises(ValueError, match="length 3"):
        _parse_boundary_value([1, 2])

    with pytest.raises(ValueError, match="Unsupported boundary-condition value"):
        _parse_boundary_value({"bad": True})


def test_logging_spec_from_user_controls_supports_verbose_alias_and_quiet():
    detailed = LoggingSpec.from_user_controls(verbose=True)
    assert detailed.enabled is True
    assert detailed.verbosity == "detailed"

    quiet = LoggingSpec.from_user_controls(verbose=False)
    assert quiet.enabled is False
    assert quiet.verbosity == "quiet"

    debug = LoggingSpec.from_user_controls(enabled=True, verbosity="debug")
    assert debug.enabled is True
    assert debug.verbosity == "debug"
    assert debug.verbosity_rank() == 3
    assert debug.is_enabled() is True


def test_logging_spec_rejects_invalid_verbosity():
    with pytest.raises(ValueError, match="Unsupported logging verbosity"):
        LoggingSpec.from_user_controls(verbosity="loud")


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
    "enable_compilation_cache",
    "make_hartmann_case",
    "make_shercliff_case",
    "make_hunt_case",
    "solve_steady",
    "solve_transient",
    "fully_developed_power_balance",
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
    "load_shercliff_analytical",
    "load_hunt_analytical",
    "load_closed_channel_analytical",
    "load_processed_slice",
}


def test_architecture_inventory_is_deterministic_without_timing(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    write_inventory(first)
    write_inventory(second)
    assert first.read_bytes() == second.read_bytes()


def test_stable_root_api_is_small_lazy_and_resolvable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert set(lmx.__all__) == EXPECTED_ROOT_API
    assert EXPECTED_ROOT_API <= set(dir(lmx))
    assert all(callable(getattr(lmx, name)) for name in lmx.__all__)
    assert all(inspect.getdoc(getattr(lmx, name)) for name in lmx.__all__)

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


def test_architecture_inventory_ignores_generated_egg_info(tmp_path: Path) -> None:
    (tmp_path / "source.py").write_bytes(b"x")
    metadata = tmp_path / "package.egg-info"
    metadata.mkdir()
    (metadata / "PKG-INFO").write_bytes(b"generated")
    assert _checkout_size(tmp_path) == 1


def test_root_import_is_lazy_and_within_budget() -> None:
    payload = build_inventory()
    payload["import_measurement"] = measure_import(repeats=3)
    assert architecture_budget_errors(payload) == []


def test_numerical_modules_do_not_import_optional_visualization() -> None:
    code = """
import sys
import lmx.plotting
assert not any(name == 'matplotlib' or name.startswith('matplotlib.') for name in sys.modules)
assert not any(name == 'PIL' or name.startswith('PIL.') for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_wheel_audit_rejects_nonpackage_payload(tmp_path: Path) -> None:
    wheel = tmp_path / "lmx-test.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("lmx/__init__.py", "")
        archive.writestr("lmx-1.dist-info/METADATA", "")
        archive.writestr("benchmarks/raw.bin", b"large output")
    assert inspect_wheel(wheel)["forbidden_members"] == ["benchmarks/raw.bin"]
    assert "outside lmx/" in architecture_budget_errors(build_inventory(), wheel=wheel)[0]


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
            "autodiff_design_demo.py": 160,
            "freemhd_closed_channel_observable_parity.py": 700,
            "fringing_benchmark_demo.py": 160,
            "hartmann_example.py": 160,
            "hunt_example.py": 160,
            "li_aln_wall_stack_example.py": 260,
            "variable_field_extruded_demo.py": 160,
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
    assert len(curated) == 8
    for item in curated:
        assert item["command"]
        assert item["outputs"]
        assert item["runtime"] in {"portable", "external", "accelerator-optional"}
        assert Path(item["docs"]).is_file()
    autodiff = Path(__file__).resolve().parents[1] / "examples/autodiff_design_demo.py"
    subprocess.run([sys.executable, autodiff], cwd=tmp_path, timeout=30, check=True)
    design_path = next((tmp_path / "artifacts").rglob("autodiff_summary.json"))
    design = json.loads(design_path.read_text())
    assert design["recovered"]["forcing"] == pytest.approx(1.0, abs=0.02)
    assert design["recovered"]["loss"] < design["optimization_history"][0]["loss"] * 1.0e-3
    assert all((design_path.parent / name).is_file() for name in design["plots"])

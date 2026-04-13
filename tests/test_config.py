from pathlib import Path

import pytest

from lmx.config import LoggingSpec, _parse_boundary_value, load_run_config


pytestmark = pytest.mark.unit


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
linear_solver = "gmres"
preconditioner = "block_jacobi"
time_scheme = "implicit_euler"
coupling_iterations = 9
coupling_tolerance = 1.0e-7

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
    assert config.solve_mode == "steady"
    assert config.case.geometry.kind == "rect_duct"
    assert config.case.output.directory == str((tmp_path / "out").resolve())
    assert config.case.solver.kind == "fully_developed_inductionless"
    assert config.case.solver.mode == "steady"
    assert config.case.solver.linear_solver == "gmres"
    assert config.case.solver.preconditioner == "block_jacobi"
    assert config.case.solver.coupling_iterations == 9
    assert config.case.time_stepper.potential_solver == "cg"
    assert config.logging.verbosity == "debug"
    assert config.logging.step_stride == 2
    assert config.restart.enabled is True
    assert config.restart.path == (tmp_path / "previous_results.npz").resolve()
    assert config.restart.reset_histories is False
    assert config.restart.write_restart is True
    assert config.restart.restart_filename == "hartmann_restart.npz"
    assert len(config.case.regions) == 1
    assert len(config.case.boundary_conditions) == 4


def test_load_run_config_rejects_analytic_callable_fields(tmp_path: Path):
    input_file = tmp_path / "bad.toml"
    input_file.write_text(
        """
[case]
name = "bad"

[geometry]
kind = "rect_duct"
width = 1.0
height = 1.0
ny = 4
nz = 4

[magnetic_field]
kind = "analytic"

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

    with pytest.raises(ValueError, match="analytic magnetic-field"):
        load_run_config(input_file)


def test_shipped_example_toml_files_parse():
    root = Path(__file__).resolve().parents[1] / "examples"

    for name in ("hartmann_case.toml", "hartmann_restart_case.toml", "shercliff_case.toml", "hunt_case.toml"):
        config = load_run_config(root / name)
        assert config.case.name
        assert config.case.regions


def test_shipped_hunt_example_uses_insulating_side_walls_and_conducting_hartmann_walls():
    root = Path(__file__).resolve().parents[1] / "examples"
    config = load_run_config(root / "hunt_case.toml")

    boundaries = {boundary.name: boundary for boundary in config.case.boundary_conditions}
    assert boundaries["left_wall"].kind == "insulating"
    assert boundaries["right_wall"].kind == "insulating"
    assert boundaries["bottom_wall"].kind == "conducting_wall"
    assert boundaries["top_wall"].kind == "conducting_wall"


def test_case_solve_mode_is_accepted_for_backward_compatibility(tmp_path: Path):
    input_file = tmp_path / "compatibility.toml"
    input_file.write_text(
        """
[case]
name = "compatibility"
solve_mode = "transient"

[geometry]
kind = "rect_duct"
width = 1.0
height = 1.0
ny = 4
nz = 4

[magnetic_field]
kind = "constant"
value = [0.0, 0.0, 1.0]

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

    config = load_run_config(input_file)
    assert config.solve_mode == "transient"
    assert config.case.solver.mode == "transient"


def test_conflicting_case_and_solver_modes_are_rejected(tmp_path: Path):
    input_file = tmp_path / "conflict.toml"
    input_file.write_text(
        """
[case]
name = "conflict"
solve_mode = "steady"

[geometry]
kind = "rect_duct"
width = 1.0
height = 1.0
ny = 4
nz = 4

[magnetic_field]
kind = "constant"
value = [0.0, 0.0, 1.0]

[solver]
mode = "transient"

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

    with pytest.raises(ValueError, match="disagree"):
        load_run_config(input_file)


def test_load_run_config_rejects_missing_required_key(tmp_path: Path):
    input_file = tmp_path / "missing.toml"
    input_file.write_text(
        """
[case]
name = "missing"

[geometry]
width = 1.0
height = 1.0
ny = 4
nz = 4

[magnetic_field]
kind = "constant"
value = [0.0, 0.0, 1.0]

[time_stepper]
dt = 0.1
t_final = 0.1
max_steps = 1
""".strip()
    )

    with pytest.raises(ValueError, match="Missing required TOML key 'kind'"):
        load_run_config(input_file)


def test_load_run_config_rejects_invalid_tuple_length(tmp_path: Path):
    input_file = tmp_path / "bad_length.toml"
    input_file.write_text(
        """
[case]
name = "bad_length"

[geometry]
kind = "rect_duct"
width = 1.0
height = 1.0
ny = 4
nz = 4
wall_thickness = [0.1, 0.2]

[magnetic_field]
kind = "constant"
value = [0.0, 0.0, 1.0]

[time_stepper]
dt = 0.1
t_final = 0.1
max_steps = 1
""".strip()
    )

    with pytest.raises(ValueError, match="must have length 4"):
        load_run_config(input_file)


def test_parse_boundary_value_accepts_scalar_and_vector_and_rejects_bad_inputs():
    assert _parse_boundary_value(None) is None
    assert _parse_boundary_value(1.25) == pytest.approx(1.25)
    assert _parse_boundary_value([1, 2, 3]) == pytest.approx((1.0, 2.0, 3.0))

    with pytest.raises(ValueError, match="length 3"):
        _parse_boundary_value([1, 2])

    with pytest.raises(ValueError, match="Unsupported boundary-condition value"):
        _parse_boundary_value({"bad": True})


def test_load_run_config_rejects_unsupported_solver_mode(tmp_path: Path):
    input_file = tmp_path / "bad_mode.toml"
    input_file.write_text(
        """
[case]
name = "bad_mode"

[geometry]
kind = "rect_duct"
width = 1.0
height = 1.0
ny = 4
nz = 4

[magnetic_field]
kind = "constant"
value = [0.0, 0.0, 1.0]

[solver]
mode = "invalid"

[time_stepper]
dt = 0.1
t_final = 0.1
max_steps = 1
""".strip()
    )

    with pytest.raises(ValueError, match="Unsupported solve mode"):
        load_run_config(input_file)


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

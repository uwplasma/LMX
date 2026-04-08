from pathlib import Path

import pytest

from lmx.config import load_run_config


pytestmark = pytest.mark.unit


def test_load_run_config_reads_complete_toml(tmp_path: Path):
    input_file = tmp_path / "hartmann.toml"
    input_file.write_text(
        """
[case]
name = "hartmann_toml_demo"
solve_mode = "steady"
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

[time_stepper]
dt = 0.001
t_final = 0.01
max_steps = 10
outer_iterations = 2
potential_iterations = 100
potential_relaxation = 1.0
potential_solver = "cg"
current_reconstruction = "cell_centered"
steady_tolerance = 1e-8
relaxation = 0.35
velocity_update_limit = 1e-3
velocity_update_limiter = "global_scale"
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
banner = true
print_regions = true
print_boundaries = true
print_footer = true
flush = true
step_stride = 2

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
    assert config.case.time_stepper.potential_solver == "cg"
    assert config.logging.step_stride == 2
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

    for name in ("hartmann_case.toml", "shercliff_case.toml", "hunt_case.toml"):
        config = load_run_config(root / name)
        assert config.case.name
        assert config.case.regions

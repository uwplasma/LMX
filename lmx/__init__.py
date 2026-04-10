"""LMX public API."""

from .benchmarks import benchmark_solver
from .cases import make_hartmann_case, make_hunt_case, make_shercliff_case
from .example_runner import run_case_example, run_theory_meeting_demo
from .io import write_paraview
from .mesh import generate_layered_duct_mesh, generate_pipe_ogrid_mesh, generate_rect_duct_mesh
from .reference_data import load_closed_channel_analytical, load_processed_slice
from .solvers import solve_steady, solve_transient
from .validation import compare_with_reference_outputs

__all__ = [
    "benchmark_solver",
    "compare_with_reference_outputs",
    "generate_layered_duct_mesh",
    "generate_pipe_ogrid_mesh",
    "generate_rect_duct_mesh",
    "load_closed_channel_analytical",
    "load_processed_slice",
    "make_hartmann_case",
    "make_hunt_case",
    "make_shercliff_case",
    "run_case_example",
    "run_theory_meeting_demo",
    "solve_steady",
    "solve_transient",
]

"""LMX public API with lazy imports for lightweight startup."""

from __future__ import annotations

from importlib import import_module


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
    "build_hartmann_autodiff_problem",
    "solve_differentiable_hartmann",
    "benchmark_sharded_stencil",
    "write_benchmark_report",
    "write_scaling_report",
]


_EXPORTS = {
    "benchmark_solver": ("lmx.benchmarks", "benchmark_solver"),
    "write_benchmark_report": ("lmx.benchmarks", "write_benchmark_report"),
    "make_hartmann_case": ("lmx.cases", "make_hartmann_case"),
    "make_hunt_case": ("lmx.cases", "make_hunt_case"),
    "make_shercliff_case": ("lmx.cases", "make_shercliff_case"),
    "run_case_example": ("lmx.example_runner", "run_case_example"),
    "run_theory_meeting_demo": ("lmx.example_runner", "run_theory_meeting_demo"),
    "generate_layered_duct_mesh": ("lmx.mesh", "generate_layered_duct_mesh"),
    "generate_pipe_ogrid_mesh": ("lmx.mesh", "generate_pipe_ogrid_mesh"),
    "generate_rect_duct_mesh": ("lmx.mesh", "generate_rect_duct_mesh"),
    "load_closed_channel_analytical": ("lmx.reference_data", "load_closed_channel_analytical"),
    "load_processed_slice": ("lmx.reference_data", "load_processed_slice"),
    "solve_steady": ("lmx.solvers", "solve_steady"),
    "solve_transient": ("lmx.solvers", "solve_transient"),
    "compare_with_reference_outputs": ("lmx.validation", "compare_with_reference_outputs"),
    "build_hartmann_autodiff_problem": ("lmx.autodiff", "build_hartmann_autodiff_problem"),
    "solve_differentiable_hartmann": ("lmx.autodiff", "solve_differentiable_hartmann"),
    "benchmark_sharded_stencil": ("lmx.scaling", "benchmark_sharded_stencil"),
    "write_scaling_report": ("lmx.scaling", "write_scaling_report"),
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value

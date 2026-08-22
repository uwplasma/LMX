"""Small, lazy convenience API for LMX.

Research and advanced APIs live in their named modules, for example
``lmx.fringing`` and ``lmx.autodiff``. Keeping the package root deliberately
small makes supported concepts discoverable and avoids importing JAX-heavy
solver modules until a symbol is used.
"""

from __future__ import annotations

from importlib import import_module

__all__ = [
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
]


_EXPORTS = {
    "enable_compilation_cache": ("lmx.io", "enable_compilation_cache"),
    "make_hartmann_case": ("lmx.cases", "make_hartmann_case"),
    "make_shercliff_case": ("lmx.cases", "make_shercliff_case"),
    "make_hunt_case": ("lmx.cases", "make_hunt_case"),
    "solve_steady": ("lmx.solvers", "solve_steady"),
    "solve_transient": ("lmx.solvers", "solve_transient"),
    "fully_developed_power_balance": ("lmx.solvers", "fully_developed_power_balance"),
    "generate_rect_duct_mesh": ("lmx.mesh", "generate_rect_duct_mesh"),
    "generate_rect_duct_mesh_from_faces": ("lmx.mesh", "generate_rect_duct_mesh_from_faces"),
    "generate_layered_duct_mesh": ("lmx.mesh", "generate_layered_duct_mesh"),
    "generate_layered_duct_mesh_from_fluid_faces": (
        "lmx.mesh",
        "generate_layered_duct_mesh_from_fluid_faces",
    ),
    "generate_multilayer_duct_mesh": ("lmx.mesh", "generate_multilayer_duct_mesh"),
    "WallLayer": ("lmx.wall_models", "WallLayer"),
    "dynamic_to_kinematic_viscosity": ("lmx.units", "dynamic_to_kinematic_viscosity"),
    "kinematic_to_dynamic_viscosity": ("lmx.units", "kinematic_to_dynamic_viscosity"),
    "hartmann_number": ("lmx.units", "hartmann_number"),
    "reynolds_number": ("lmx.units", "reynolds_number"),
    "interaction_parameter": ("lmx.units", "interaction_parameter"),
    "magnetic_reynolds_number": ("lmx.units", "magnetic_reynolds_number"),
    "magnetic_field_from_hartmann": ("lmx.units", "magnetic_field_from_hartmann"),
    "wall_conductance_ratio": ("lmx.units", "wall_conductance_ratio"),
    "effective_pinhole_conductance_ratio": ("lmx.wall_models", "effective_pinhole_conductance_ratio"),
    "tangential_stack_conductance_ratio": ("lmx.wall_models", "tangential_stack_conductance_ratio"),
    "normal_stack_leakage_ratio": ("lmx.wall_models", "normal_stack_leakage_ratio"),
    "equivalent_single_layer": ("lmx.wall_models", "equivalent_single_layer"),
    "nested_wall_layer_resolution_summary": ("lmx.wall_models", "nested_wall_layer_resolution_summary"),
    "load_shercliff_analytical": ("lmx.reference_data", "load_shercliff_analytical"),
    "load_hunt_analytical": ("lmx.reference_data", "load_hunt_analytical"),
    "load_closed_channel_analytical": ("lmx.reference_data", "load_closed_channel_analytical"),
    "load_processed_slice": ("lmx.reference_data", "load_processed_slice"),
}


def __getattr__(name: str):
    """Load a documented root export on first access."""

    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module 'lmx' has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return module globals plus the lazy public surface for discovery."""

    return sorted(set(globals()) | set(__all__))

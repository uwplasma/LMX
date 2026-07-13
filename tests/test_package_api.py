import lmx


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


def test_stable_root_api_is_small_lazy_and_resolvable():
    assert set(lmx.__all__) == EXPECTED_ROOT_API
    assert EXPECTED_ROOT_API <= set(dir(lmx))
    assert all(callable(getattr(lmx, name)) for name in lmx.__all__)


def test_advanced_api_uses_owning_module():
    assert not hasattr(lmx, "solve_extruded_inductionless")
    from lmx.fringing import solve_extruded_inductionless

    assert callable(solve_extruded_inductionless)


def test_unknown_root_attribute_has_standard_error():
    try:
        lmx.not_an_api
    except AttributeError as exc:
        assert "not_an_api" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("unknown root attribute unexpectedly resolved")

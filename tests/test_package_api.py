import lmx


def test_lmx_lazy_exports_resolve_expected_symbols():
    assert callable(lmx.generate_rect_duct_mesh)
    assert callable(lmx.enable_compilation_cache)
    assert callable(lmx.load_fringing_pipe_profile)
    assert callable(lmx.load_shercliff_analytical)
    assert callable(lmx.load_hunt_analytical)
    assert callable(lmx.build_hartmann_autodiff_problem)
    assert callable(lmx.build_extruded_response_targets)
    assert callable(lmx.hartmann_mean_velocity_gradients)
    assert callable(lmx.hartmann_profile_loss_gradients)
    assert callable(lmx.run_hartmann_profile_inverse_design)
    assert callable(lmx.build_square_duct_fringing_benchmark)
    assert callable(lmx.build_layered_duct_extruded_problem)
    assert callable(lmx.build_pipe_ogrid_extruded_problem)
    assert callable(lmx.run_extruded_inductionless_slice)
    assert callable(lmx.solve_extruded_inductionless)
    assert callable(lmx.build_square_duct_extruded_problem)
    assert callable(lmx.build_fringing_autodiff_problem)
    assert callable(lmx.extruded_rect_response_history)
    assert callable(lmx.extruded_rect_projection_iteration_history)
    assert callable(lmx.extruded_rect_projection_field_loss_gradients)
    assert callable(lmx.extruded_rect_projection_trajectory_loss_gradients)
    assert callable(lmx.fringing_mean_velocity_history)
    assert callable(lmx.fringing_response_history)
    assert callable(lmx.extruded_rect_response_loss_gradients)
    assert callable(lmx.run_fringing_history_inverse_design)
    assert callable(lmx.run_fringing_response_inverse_design)
    assert callable(lmx.run_extruded_rect_inverse_design)
    assert callable(lmx.run_extruded_rect_projection_field_inverse_design)
    assert callable(lmx.run_extruded_rect_projection_trajectory_inverse_design)
    assert callable(lmx.run_extruded_target_inverse_design)
    assert callable(lmx.benchmark_sharded_stencil)
    assert callable(lmx.write_scaling_report)
    assert callable(lmx.solve_case_snapshots)
    assert callable(lmx.write_case_overview_plots)
    assert callable(lmx.write_extruded_overview_plots)
    assert callable(lmx.write_geometry_gallery_plots)
    assert callable(lmx.write_geometry_preview_plots)
    assert callable(lmx.write_transient_movies)
    assert callable(lmx.write_strong_scaling_plots)
    assert callable(lmx.write_autodiff_plots)
    assert callable(lmx.solve_closed_channel_benchmark)
    assert callable(lmx.write_lm_duct_geometry_setup_figure)
    assert callable(lmx.write_structured_mesh_figure)
    assert callable(lmx.write_boundary_layer_figure)
    assert callable(lmx.write_annotated_layer_figure)
    assert callable(lmx.write_velocity_profile_volume_figure)
    assert callable(lmx.write_closed_channel_profile_comparison_figure)
    assert callable(lmx.write_closed_channel_startup_movies)
    assert "solve_steady" in lmx.__all__
    assert "build_hartmann_autodiff_problem" in lmx.__all__
    assert "write_case_overview_plots" in lmx.__all__


def test_lmx_lazy_exports_reject_unknown_name():
    try:
        getattr(lmx, "definitely_missing_symbol")
    except AttributeError as exc:
        assert str(exc) == "definitely_missing_symbol"
    else:
        raise AssertionError("Expected AttributeError for missing lazy export.")

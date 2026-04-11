import lmx


def test_lmx_lazy_exports_resolve_expected_symbols():
    assert callable(lmx.generate_rect_duct_mesh)
    assert callable(lmx.build_hartmann_autodiff_problem)
    assert callable(lmx.hartmann_mean_velocity_gradients)
    assert callable(lmx.build_square_duct_fringing_benchmark)
    assert callable(lmx.benchmark_sharded_stencil)
    assert callable(lmx.write_scaling_report)
    assert "solve_steady" in lmx.__all__
    assert "build_hartmann_autodiff_problem" in lmx.__all__


def test_lmx_lazy_exports_reject_unknown_name():
    try:
        getattr(lmx, "definitely_missing_symbol")
    except AttributeError as exc:
        assert str(exc) == "definitely_missing_symbol"
    else:
        raise AssertionError("Expected AttributeError for missing lazy export.")

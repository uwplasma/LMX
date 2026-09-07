"""Reference tests explicitly request double precision before collection."""

import pytest

import lmx

lmx.enable_x64()


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items):
    """Map semantic evidence markers to explicit execution tiers before selection."""
    for item in items:
        marks = {mark.name for mark in item.iter_markers()}
        if "curated" in marks:
            item.add_marker(pytest.mark.slow)
        if marks & {"numerical", "physics", "regression", "validation"} or item.path.name in {
            "test_solver.py",
            "test_fringing.py",
            "test_benchmarks.py",
            "test_physics.py",
        }:
            item.add_marker(pytest.mark.regression)
        elif not marks & {"unit", "slow", "gpu", "external"}:
            item.add_marker(pytest.mark.unit)

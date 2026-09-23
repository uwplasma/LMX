"""Reference tests explicitly request double precision before collection."""

import os

import jax
import pytest

# Tests measure compiles and must not share programs through the user's disk cache.
os.environ.setdefault("LMX_COMPILATION_CACHE", "0")

import lmx  # noqa: E402

lmx.enable_x64()


@pytest.fixture
def true_float32_matmuls():
    """Pin float32 matmuls, as mixed precision requires, and restore the setting.

    JAX's default on Ampere GPUs is TensorFloat-32, under which the float64
    correction of a float32 solve stalls near 1e-4. On CPU the setting is inert,
    so pinning it keeps these tests meaningful on either device.
    """
    previous = jax.config.jax_default_matmul_precision
    jax.config.update("jax_default_matmul_precision", "float32")
    try:
        yield
    finally:
        jax.config.update("jax_default_matmul_precision", previous)


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

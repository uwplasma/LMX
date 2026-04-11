from types import SimpleNamespace

import jax.numpy as jnp
import pytest

import lmx.linear as linear


pytestmark = pytest.mark.unit


def _poisson_coefficients():
    diagonal = jnp.ones((2, 2)) * 4.0
    west = jnp.ones((2, 2))
    east = jnp.ones((2, 2))
    south = jnp.ones((2, 2))
    north = jnp.ones((2, 2))
    rhs = jnp.zeros((2, 2))
    return diagonal, west, east, south, north, rhs


def test_solve_poisson_lineax_falls_back_without_lineax(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(linear, "lx", None)
    diagonal, west, east, south, north, rhs = _poisson_coefficients()
    solution, info = linear.solve_poisson_lineax(diagonal, west, east, south, north, rhs, anchor=(0, 0))

    assert solution.shape == (2, 2)
    assert info.backend == "jax-jacobi"
    assert info.iterations == 400
    assert info.residual == pytest.approx(0.0)


def test_solve_poisson_lineax_uses_lineax_backend(monkeypatch: pytest.MonkeyPatch):
    diagonal, west, east, south, north, rhs = _poisson_coefficients()

    class FakeLinearOperator:
        def __init__(self, mv, shape, tags=None):
            self.mv = mv
            self.shape = shape
            self.tags = tags

    class FakeSolver:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def fake_linear_solve(op, rhs_vec, solver=None):
        op.mv(jnp.ones_like(rhs_vec))
        return SimpleNamespace(value=jnp.zeros_like(rhs_vec), stats={"num_steps": 7})

    fake_lx = SimpleNamespace(
        FunctionLinearOperator=FakeLinearOperator,
        positive_semidefinite_tag=object(),
        CG=FakeSolver,
        linear_solve=fake_linear_solve,
    )
    monkeypatch.setattr(linear, "lx", fake_lx)

    solution, info = linear.solve_poisson_lineax(diagonal, west, east, south, north, rhs, anchor=(0, 0))

    assert solution.shape == (2, 2)
    assert info.backend == "lineax-cg"
    assert info.iterations == 7


def test_solve_poisson_jacobi_can_stop_early_on_residual_tolerance():
    diagonal, west, east, south, north, rhs = _poisson_coefficients()
    solution, info = linear.solve_poisson_jacobi(
        diagonal,
        west,
        east,
        south,
        north,
        rhs,
        anchor=(0, 0),
        iterations=50,
        tolerance=1e-6,
    )

    assert solution.shape == (2, 2)
    assert info.backend == "jax-jacobi"
    assert info.iterations < 50
    assert info.residual <= 2e-6


def test_solve_poisson_cg_converges_on_zero_rhs():
    diagonal, west, east, south, north, rhs = _poisson_coefficients()

    solution, info = linear.solve_poisson_cg(
        diagonal,
        west,
        east,
        south,
        north,
        rhs,
        anchor=(0, 0),
        iterations=20,
        tolerance=1e-8,
    )

    assert solution.shape == (2, 2)
    assert info.backend == "jax-cg"
    assert info.iterations == 0
    assert info.residual == pytest.approx(0.0)


def test_poisson_residual_norm_is_zero_for_exact_zero_solution():
    diagonal, west, east, south, north, rhs = _poisson_coefficients()
    residual = linear.poisson_residual_norm(
        diagonal,
        west,
        east,
        south,
        north,
        rhs,
        jnp.zeros_like(rhs),
        anchor=(0, 0),
    )

    assert float(residual) == pytest.approx(0.0)


def _five_point_coefficients():
    diagonal = jnp.ones((2, 2)) * 4.0
    west = jnp.ones((2, 2))
    east = jnp.ones((2, 2))
    south = jnp.ones((2, 2))
    north = jnp.ones((2, 2))
    rhs = jnp.ones((2, 2))
    return diagonal, west, east, south, north, rhs


def test_solve_five_point_cg_state_supports_none_preconditioner_and_rejects_unknown():
    diagonal, west, east, south, north, rhs = _five_point_coefficients()

    field, residual, iterations = linear.solve_five_point_cg_state(
        diagonal,
        west,
        east,
        south,
        north,
        rhs,
        iterations=5,
        tolerance=None,
        preconditioner="none",
    )

    assert field.shape == rhs.shape
    assert int(iterations) <= 5
    assert float(residual) >= 0.0

    with pytest.raises(ValueError, match="Unsupported preconditioner"):
        linear.solve_five_point_cg_state(
            diagonal,
            west,
            east,
            south,
            north,
            rhs,
            iterations=2,
            preconditioner="bad",
        )


def test_solve_five_point_lineax_falls_back_without_lineax(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(linear, "lx", None)
    diagonal, west, east, south, north, rhs = _five_point_coefficients()

    field, info = linear.solve_five_point_lineax(
        diagonal,
        west,
        east,
        south,
        north,
        rhs,
        linear_solver="cg",
        tolerance=1e-8,
        max_steps=6,
    )

    assert field.shape == rhs.shape
    assert info.backend == "jax-cg"
    assert info.iterations <= 6


def test_solve_five_point_lineax_supports_gmres_and_bicgstab_and_rejects_unknown(monkeypatch: pytest.MonkeyPatch):
    diagonal, west, east, south, north, rhs = _five_point_coefficients()
    created = []

    class FakeLinearOperator:
        def __init__(self, mv, shape):
            self.mv = mv
            self.shape = shape

    class FakeSolver:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def fake_linear_solve(op, rhs_vec, solver=None):
        created.append(type(solver).__name__)
        op.mv(jnp.ones_like(rhs_vec))
        return SimpleNamespace(value=jnp.zeros_like(rhs_vec), stats={"num_steps": "fallback"})

    fake_lx = SimpleNamespace(
        FunctionLinearOperator=FakeLinearOperator,
        GMRES=FakeSolver,
        BiCGStab=FakeSolver,
        CG=FakeSolver,
        linear_solve=fake_linear_solve,
    )
    monkeypatch.setattr(linear, "lx", fake_lx)

    _, gmres_info = linear.solve_five_point_lineax(
        diagonal, west, east, south, north, rhs, linear_solver="gmres", max_steps=9
    )
    _, bicg_info = linear.solve_five_point_lineax(
        diagonal, west, east, south, north, rhs, linear_solver="bicgstab", max_steps=11
    )

    assert gmres_info.backend == "lineax-gmres"
    assert gmres_info.iterations == 9
    assert bicg_info.backend == "lineax-bicgstab"
    assert bicg_info.iterations == 11
    assert created == ["FakeSolver", "FakeSolver"]

    with pytest.raises(ValueError, match="Unsupported lineax solver"):
        linear.solve_five_point_lineax(diagonal, west, east, south, north, rhs, linear_solver="bad")


def test_solve_five_point_system_supports_auto_gmres_and_rejects_unknown(monkeypatch: pytest.MonkeyPatch):
    diagonal, west, east, south, north, rhs = _five_point_coefficients()

    field, info = linear.solve_five_point_system(
        diagonal,
        west,
        east,
        south,
        north,
        rhs,
        linear_solver="auto",
        max_steps=4,
    )
    assert field.shape == rhs.shape
    assert info.backend == "jax-cg"

    monkeypatch.setattr(
        linear,
        "solve_five_point_lineax",
        lambda *args, **kwargs: (jnp.zeros_like(rhs), linear.LinearSolveInfo("lineax-gmres", 3, 1e-6)),
    )
    _, gmres_info = linear.solve_five_point_system(
        diagonal,
        west,
        east,
        south,
        north,
        rhs,
        linear_solver="gmres",
    )
    assert gmres_info.backend == "lineax-gmres"

    with pytest.raises(ValueError, match="Unsupported linear solver"):
        linear.solve_five_point_system(diagonal, west, east, south, north, rhs, linear_solver="bad")

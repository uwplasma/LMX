"""The TM-228 inertialess core-flow model: Walker's limits, the ANL fringe, symmetry and the adjoint."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lmx.core3d import fringe_field
from lmx.coreflow import CoreFlow, fully_developed_gradient, midplane_field
from lmx.grid import Grid, uniform_faces

pytestmark = pytest.mark.unit


def _anl(x, half_length=3.0):
    """TM-228 eq. (16): ``B_y = (1 - sin(pi x / 2 x0)) / 2`` inside the fringe, 1 upstream, 0 downstream."""
    inside = np.clip(x, -half_length, half_length)
    return 0.5 * (1.0 - np.sin(np.pi * inside / (2.0 * half_length)))


def _anl_square_integral(lower, upper=2.0):
    """Closed form of the integral of ``B_y^2`` from ``lower <= -3`` to ``upper`` in ``[-3, 3]`` (x0 = 3)."""

    def antiderivative(u):
        return 1.5 * u + 2.0 * np.cos(u) - np.sin(2.0 * u) / 4.0

    fringe = (antiderivative(np.pi * upper / 6.0) - antiderivative(-np.pi / 2.0)) * 6.0 / (4.0 * np.pi)
    return (-3.0 - lower) + fringe


@pytest.mark.physics
@pytest.mark.parametrize(
    ("aspect", "c_t", "c_s", "scale"),
    [(1.0, 0.02, 0.02, 1.0), (0.5, 0.05, 0.1, 2.0), (2.0, 0.1, 0.01, 0.7)],
)
def test_a_uniform_field_gives_walkers_fully_developed_gradient_at_second_order(aspect, c_t, c_s, scale):
    exact = scale**2 * fully_developed_gradient(c_t, c_s, aspect)
    errors = []
    for cells in (8, 16):
        model = CoreFlow(np.linspace(0.0, 2.0, 9), aspect=aspect, nz=cells, ny=cells)
        result = model.solve(np.ones(9), c_t=c_t, c_s=c_s, field_scale=scale)
        errors.append(float(result.pressure_drop) / 2.0 / exact - 1.0)
        assert float(jnp.ptp(result.pressure[4])) < 1e-12
        assert float(jnp.ptp(result.axial_flux)) < 1e-12
        assert float(jnp.mean(result.axial_flux)) == pytest.approx(aspect, rel=1e-12)
    # The only error is the trapezoidal side-wall integral of a quadratic potential.
    assert abs(errors[1]) < 1e-3
    assert errors[0] / errors[1] == pytest.approx(4.0, rel=0.02)


@pytest.mark.physics
def test_perfectly_conducting_side_walls_give_the_hartmann_wall_limit():
    model = CoreFlow(np.linspace(0.0, 2.0, 9), nz=8, ny=8)
    result = model.solve(np.ones(9), c_t=0.02, c_s=1e4)
    assert float(result.pressure_drop) / 2.0 == pytest.approx(0.02 / 1.02, rel=2e-6)


def test_the_coupled_operator_is_symmetric_and_indefinite():
    x = np.linspace(-6.0, 2.0, 17)
    model = CoreFlow(x, aspect=0.8, nz=4, ny=5)
    matrix = model.operator(_anl(x), c_t=0.03, c_s=0.05, field_scale=1.3)
    asymmetry = abs(matrix - matrix.T).max() / abs(matrix).max()
    assert asymmetry < 1e-12
    eigenvalues = np.linalg.eigvalsh(matrix.toarray())
    assert eigenvalues.min() < 0.0 < eigenvalues.max()
    assert np.min(np.abs(eigenvalues)) > 1e-10 * np.max(np.abs(eigenvalues))


@pytest.mark.validation
def test_the_anl_fringe_reproduces_tm228():
    """ANL/FPP/TM-228 section 4.1: x0 = 3, c_t = c_s = 0.02, a = 1, 0.0932 against 0.0754 from x = -6 to 2.

    The domain runs past the fringe into the zero field, as TM-228's cap on
    ``1/B`` implies. The quadrature of TM-228's own fully developed gradient
    over ``[-6, 2]`` is 0.07757, 2.9 % above the quoted 0.0754; the quoted pair
    is consistent with an interval starting at ``x_s = -5.853``, where that
    quadrature is exactly 0.0754, so both the three-dimensional excess and the
    drop over ``[x_s, 2]`` are gated.
    """
    gradient = fully_developed_gradient(0.02, 0.02)
    local = gradient * _anl_square_integral(-6.0)
    assert local == pytest.approx(0.0775728, rel=1e-6)
    x = np.linspace(-10.0, 6.0, 161)
    model = CoreFlow(x, nz=20, ny=20)
    result = model.solve(_anl(x), c_t=0.02, c_s=0.02)
    # 1/B is capped at 1000 from x = 2.9 (B = 6.9e-4) to the end of the domain.
    assert int(result.floored_nodes) == 32
    assert float(jnp.ptp(result.axial_flux)) < 1e-10
    pressure = np.asarray(result.pressure).mean(axis=1)
    drop = np.interp(-6.0, x, pressure) - np.interp(2.0, x, pressure)
    assert drop - local == pytest.approx(0.0932 - 0.0754, rel=0.01)
    assert drop / local == pytest.approx(0.0932 / 0.0754, rel=0.01)
    start = -6.0 + (local - 0.0754) / gradient
    assert gradient * _anl_square_integral(start) == pytest.approx(0.0754, rel=1e-12)
    assert np.interp(start, x, pressure) - np.interp(2.0, x, pressure) == pytest.approx(0.0932, rel=0.01)
    # Figure 10: the three-dimensional excess is k c^(1/2) with k = 0.126 for x0 = 3.
    assert drop - local == pytest.approx(0.126 * 0.02**0.5, rel=0.01)


def test_the_fringe_field_provider_feeds_the_model():
    grid = Grid(uniform_faces(40, -6.0, 2.0), uniform_faces(4, -1.0, 1.0), uniform_faces(2, -1.0, 1.0))
    x, field = midplane_field(fringe_field(grid, solenoidal=False))
    # Each cell holds the mean of its two faces, a difference of the vector potential.
    np.testing.assert_allclose(field, _anl(x), atol=1e-3)
    result = CoreFlow(x, nz=4, ny=4).solve(field, c_t=0.02, c_s=0.02)
    assert 0.08 < float(result.pressure_drop) < 0.1


@pytest.mark.numerical
def test_the_adjoint_matches_central_differences():
    x = np.linspace(-4.0, 2.0, 25)
    model = CoreFlow(x, aspect=0.9, nz=6, ny=5)
    field = _anl(x)

    def drop(parameters):
        c_t, c_s, scale = parameters
        return model.solve(field, c_t=c_t, c_s=c_s, field_scale=scale).pressure_drop

    def flux(parameters):
        c_t, c_s, drive = parameters
        return jnp.mean(model.solve(field, c_t=c_t, c_s=c_s, drive=drive, mean_velocity=None).axial_flux)

    derivatives = {}
    for objective, point in ((drop, (0.03, 0.05, 1.2)), (flux, (0.03, 0.05, 0.7))):
        point = jnp.asarray(point)
        derivatives[objective] = jax.jit(jax.grad(objective))
        gradient = np.asarray(derivatives[objective](point))
        tangent = np.asarray(jax.jit(jax.jacfwd(objective))(point))
        objective = jax.jit(objective)
        steps = 1e-5 * np.asarray(point)
        central = np.array(
            [
                (objective(point.at[k].add(steps[k])) - objective(point.at[k].add(-steps[k])))
                / (2 * steps[k])
                for k in range(3)
            ]
        )
        np.testing.assert_allclose(gradient, central, rtol=1e-6)
        np.testing.assert_allclose(tangent, gradient, rtol=1e-10)
    # The raw flux is linear in the drive, and the drop scales as the field squared.
    point = jnp.asarray((0.03, 0.05, 1.2))
    assert float(derivatives[drop](point)[2]) == pytest.approx(2.0 * float(drop(point)) / 1.2, rel=1e-8)


def test_invalid_meshes_and_fields_are_refused():
    with pytest.raises(ValueError, match="uniformly spaced"):
        CoreFlow(np.array([0.0, 1.0, 3.0, 4.0]))
    with pytest.raises(ValueError, match="aspect"):
        CoreFlow(np.linspace(0.0, 1.0, 5), nz=1)
    with pytest.raises(ValueError, match="stations"):
        CoreFlow(np.linspace(0.0, 1.0, 5)).solve(np.ones(4), c_t=0.1, c_s=0.1)

import jax.numpy as jnp
import pytest

from lmx.cases import make_hartmann_case, make_shercliff_case
from lmx.solvers import solve_steady


pytestmark = pytest.mark.regression


_EXPECTED_HARTMANN_CENTERLINE = jnp.asarray(
    [
        0.0,
        0.038497839,
        0.057696119,
        0.066464774,
        0.070100561,
        0.071358547,
        0.071358182,
        0.070099525,
        0.066463225,
        0.057694353,
        0.038496420,
        0.0,
    ]
)

_EXPECTED_SHERCLIFF_CENTERLINE = jnp.asarray(
    [
        0.0,
        0.037473299,
        0.057078741,
        0.066200368,
        0.069920868,
        0.071173675,
        0.071425311,
        0.070404597,
        0.066600353,
        0.057347629,
        0.037609905,
        0.0,
    ]
)


def _centerline(solution):
    return solution.state.u[:, solution.state.u.shape[1] // 2]


def test_hartmann_centerline_regression():
    case = make_hartmann_case(ha=2.0, ny=12, nz=12)
    solution = solve_steady(case)
    assert jnp.allclose(_centerline(solution), _EXPECTED_HARTMANN_CENTERLINE, atol=1e-6)


def test_shercliff_centerline_regression():
    case = make_shercliff_case(ha=2.0, ny=12, nz=12)
    solution = solve_steady(case)
    assert jnp.allclose(_centerline(solution), _EXPECTED_SHERCLIFF_CENTERLINE, atol=1e-6)

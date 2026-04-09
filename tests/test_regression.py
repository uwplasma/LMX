import jax.numpy as jnp
import pytest

from lmx.cases import make_hartmann_case, make_shercliff_case
from lmx.solvers import solve_steady


pytestmark = pytest.mark.regression


_EXPECTED_HARTMANN_CENTERLINE = jnp.asarray(
    [
        0.0,
        0.06030579,
        0.08260150,
        0.09641168,
        0.10430133,
        0.10786588,
        0.10786589,
        0.10430134,
        0.09641170,
        0.08260150,
        0.06030578,
        0.0,
    ]
)

_EXPECTED_SHERCLIFF_CENTERLINE = jnp.asarray(
    [
        0.0,
        0.12307610,
        0.17726284,
        0.21458733,
        0.23789105,
        0.24907115,
        0.24907114,
        0.23789103,
        0.21458730,
        0.17726278,
        0.12307601,
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

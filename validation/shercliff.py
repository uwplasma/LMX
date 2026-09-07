"""A spectral reference for insulating-duct MHD flow, independent of LMX.

Fully developed flow along a square duct with a transverse field has no closed
form that is comfortable to transcribe correctly, and a series copied out of a
paper is a reference only as far as the transcription is trusted. This module
solves the governing system directly instead, with Chebyshev collocation and a
dense linear solve, so the reference stands on the equations rather than on a
published expansion. It shares no operator, mesh or solver with the package.

With :math:`\\mathbf u = u(y,z)\\hat x`, :math:`\\mathbf B = B\\hat y` and
constant properties the inductionless system reduces to

.. math::
   \\mu\\nabla^2 u - \\sigma B^2 u + \\sigma B\\,\\partial_z\\varphi + f = 0,
   \\qquad \\nabla^2\\varphi = B\\,\\partial_z u,

with no slip on all four walls and, because an insulating wall carries no
current, :math:`\\partial_n\\varphi = 0` on all four. The discretization is
spectral, so the answer is converged to eight digits by about forty points per
direction and can be treated as exact when a finite-volume result is compared
against it. At :math:`B=0` it reproduces the analytic Poiseuille duct maximum
0.29468541 for unit forcing on ``[-1, 1]^2``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["chebyshev_weights", "duct_flow", "flow_rate"]


def _differentiation_matrix(points: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the Chebyshev differentiation matrix and its nodes on ``[-1, 1]``."""
    nodes = np.cos(np.pi * np.arange(points + 1) / points)
    scale = np.hstack([2.0, np.ones(points - 1), 2.0]) * (-1.0) ** np.arange(points + 1)
    spread = np.tile(nodes, (points + 1, 1)).T
    difference = spread - spread.T
    matrix = np.outer(scale, 1.0 / scale) / (difference + np.eye(points + 1))
    return matrix - np.diag(matrix.sum(axis=1)), nodes


def duct_flow(
    hartmann: float,
    points: int = 40,
    *,
    forcing: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the nodes, velocity and potential of an insulating square duct.

    Non-dimensionalised so that the half width, density, kinematic viscosity and
    conductivity are one; the field is then ``B = hartmann`` along ``y`` and the
    duct occupies ``[-1, 1]`` in both transverse directions.
    """
    if points < 4:
        raise ValueError("the spectral reference needs at least four points per direction")
    derivative, nodes = _differentiation_matrix(points)
    count = points + 1
    identity = np.eye(count)
    laplacian = np.kron(derivative @ derivative, identity) + np.kron(identity, derivative @ derivative)
    along_y = np.kron(derivative, identity)
    along_z = np.kron(identity, derivative)
    y = np.repeat(nodes, count)
    z = np.tile(nodes, count)
    edge = (np.abs(y) > 1.0 - 1e-12) | (np.abs(z) > 1.0 - 1e-12)

    size = count * count
    field = float(hartmann)
    operator = np.zeros((2 * size, 2 * size))
    source = np.zeros(2 * size)
    operator[:size, :size] = laplacian - field**2 * np.eye(size)
    operator[:size, size:] = field * along_z
    source[:size] = -float(forcing)
    operator[size:, :size] = -field * along_z
    operator[size:, size:] = laplacian
    for row in np.flatnonzero(edge):
        operator[row, :] = 0.0
        operator[row, row] = 1.0
        source[row] = 0.0
        operator[size + row, :] = 0.0
        operator[size + row, size:] = along_y[row] if abs(y[row]) > 1.0 - 1e-12 else along_z[row]
        source[size + row] = 0.0
    anchor = size // 2
    operator[size + anchor, :] = 0.0
    operator[size + anchor, size + anchor] = 1.0
    source[size + anchor] = 0.0

    solution = np.linalg.solve(operator, source)
    return nodes, solution[:size].reshape(count, count), solution[size:].reshape(count, count)


def chebyshev_weights(points: int) -> np.ndarray:
    """Return Clenshaw-Curtis quadrature weights for the Chebyshev nodes."""
    angles = np.pi * np.arange(points + 1) / points
    weights = np.zeros(points + 1)
    interior = np.ones(points - 1)
    for order in range(2, points, 2):
        interior -= 2.0 * np.cos(order * angles[1:points]) / (order**2 - 1)
    if points % 2 == 0:
        weights[0] = weights[points] = 1.0 / (points**2 - 1)
        interior -= np.cos(points * angles[1:points]) / (points**2 - 1)
    else:
        weights[0] = weights[points] = 1.0 / points**2
    weights[1:points] = 2.0 * interior / points
    return weights


def flow_rate(hartmann: float, points: int = 40, *, forcing: float = 1.0) -> float:
    """Return the mean velocity of the cross-section, ``Q / A``."""
    _, velocity, _ = duct_flow(hartmann, points, forcing=forcing)
    weights = chebyshev_weights(points)
    return float(weights @ velocity @ weights) / 4.0

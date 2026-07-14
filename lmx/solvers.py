"""Fully developed steady and transient inductionless solvers."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import jax
import numpy as np

import jax.numpy as jnp
from jax.scipy.linalg import cho_factor, cho_solve

from .core import Diagnostics, MHDState, Solution
from .linear import (
    apply_five_point_operator,
    apply_poisson_operator,
    five_point_residual_norm,
    poisson_residual_norm,
    solve_five_point_system,
    solve_poisson_cg_state,
    solve_poisson_jacobi_state,
    solve_poisson_lineax,
)
from .mesh import (
    StructuredMesh,
    generate_layered_duct_mesh,
    generate_rect_duct_mesh,
    generate_rect_duct_mesh_from_faces,
)
from .operators import gradient_scalar
from .physics import build_material_fields, magnetic_field_components
from .runtime_logging import RestartLogInfo, SolverStepRecord
from .specs import BoundaryCondition, CaseSpec

try:
    from solvax import (
        aitken_relaxation as _solvax_aitken_relaxation,
        anderson_mixing as _solvax_anderson_mixing,
        gmres as _solvax_gmres,
        p_multigrid as _solvax_p_multigrid,
        tridiagonal_solve as _solvax_tridiagonal_solve,
    )
except ImportError:  # pragma: no cover - exercised in minimum installs
    _solvax_aitken_relaxation = None
    _solvax_anderson_mixing = None
    _solvax_gmres = None
    _solvax_p_multigrid = None
    _solvax_tridiagonal_solve = None


_POTENTIAL_ADDITIVE_LINE_MIN_CELLS = 110
_POTENTIAL_ADDITIVE_LINE_DIAGONAL_RATIO = 3.0e4
_POTENTIAL_COARSE_STRIDE = 8
_POTENTIAL_INEXACT_COUPLING_TOLERANCE = 1.0e-4
_POTENTIAL_COUPLING_NORMALIZED_GATE = 1.0e-5
_LINEAR_RESIDUAL_FLOOR = 1.0e-9
_MIN_STRICT_POTENTIAL_COUPLING_SOLVES = 3
_POTENTIAL_FGMRES_RELATIVE_TOLERANCE = 1.0e-12


def _coupling_potential_tolerance(
    requested: float | None,
    *,
    velocity_residual: float,
    coupling_tolerance: float,
    flexible: bool,
) -> float | None:
    """Use an inexact potential solve only while the fixed point is far away."""
    if (
        requested is None
        or flexible
        or velocity_residual <= 10.0 * coupling_tolerance
    ):
        return requested
    return max(float(requested), _POTENTIAL_INEXACT_COUPLING_TOLERANCE)


def _nested_velocity_tolerance(coupling_tolerance: float, dtype) -> float:
    """Keep the momentum solve below the fixed-point error it supports."""
    roundoff_floor = 10.0 * float(jnp.finfo(dtype).eps)
    requested = min(1.0e-10, 0.01 * max(float(coupling_tolerance), 0.0))
    return max(roundoff_floor, requested)


def _potential_y_line_preconditioner(
    diagonal: jnp.ndarray,
    west: jnp.ndarray,
    east: jnp.ndarray,
    anchor: tuple[int, int],
):
    if _solvax_tridiagonal_solve is None:
        return None
    line_diagonal = diagonal.at[anchor].set(1.0)
    lower = (-west).at[anchor].set(0.0)
    upper = (-east).at[anchor].set(0.0)

    def apply(residual: jnp.ndarray) -> jnp.ndarray:
        solved = _solvax_tridiagonal_solve(
            lower,
            line_diagonal,
            upper,
            residual.at[anchor].set(0.0),
        )
        return solved.at[anchor].set(0.0)

    return apply


def _potential_z_line_preconditioner(
    diagonal: jnp.ndarray,
    south: jnp.ndarray,
    north: jnp.ndarray,
    anchor: tuple[int, int],
):
    transposed = _potential_y_line_preconditioner(
        diagonal.T, south.T, north.T, (anchor[1], anchor[0])
    )
    if transposed is None:
        return None

    def apply(residual: jnp.ndarray) -> jnp.ndarray:
        return transposed(residual.T).T

    return apply


def _potential_additive_line_preconditioner(
    diagonal: jnp.ndarray,
    west: jnp.ndarray,
    east: jnp.ndarray,
    south: jnp.ndarray,
    north: jnp.ndarray,
    anchor: tuple[int, int],
):
    if _solvax_tridiagonal_solve is None:
        return None
    y_line = _potential_y_line_preconditioner(diagonal, west, east, anchor)
    anchor_t = (anchor[1], anchor[0])
    z_diagonal = diagonal.T.at[anchor_t].set(1.0)
    z_lower = (-south.T).at[anchor_t].set(0.0)
    z_upper = (-north.T).at[anchor_t].set(0.0)

    def apply(residual: jnp.ndarray) -> jnp.ndarray:
        rhs = residual.at[anchor].set(0.0)
        solved_y = y_line(rhs)
        solved_z = _solvax_tridiagonal_solve(
            z_lower,
            z_diagonal,
            z_upper,
            rhs.T,
        ).T
        return (0.5 * (solved_y + solved_z)).at[anchor].set(0.0)

    return apply


def _potential_deflated_line_preconditioner(
    diagonal: jnp.ndarray,
    west: jnp.ndarray,
    east: jnp.ndarray,
    south: jnp.ndarray,
    north: jnp.ndarray,
    anchor: tuple[int, int],
    *,
    coarse_stride: int = _POTENTIAL_COARSE_STRIDE,
):
    """Combine SPD line solves with an exact Galerkin coarse correction."""
    if _solvax_p_multigrid is None or coarse_stride < 2:
        return None
    mean_y = float(np.asarray(jnp.mean(west + east)))
    mean_z = float(np.asarray(jnp.mean(south + north)))
    if mean_y >= 4.0 * max(mean_z, np.finfo(float).tiny):
        line = _potential_y_line_preconditioner(diagonal, west, east, anchor)
    elif mean_z >= 4.0 * max(mean_y, np.finfo(float).tiny):
        line = _potential_z_line_preconditioner(diagonal, south, north, anchor)
    else:
        line = _potential_additive_line_preconditioner(
            diagonal, west, east, south, north, anchor
        )
    if line is None:
        return None

    fine_shape = diagonal.shape
    coarse_shape = tuple(
        (size - 1 + coarse_stride - 1) // coarse_stride + 1 for size in fine_shape
    )

    def prolong(coarse: jnp.ndarray) -> jnp.ndarray:
        fine = jax.image.resize(coarse, fine_shape, method="linear")
        return fine.at[anchor].set(0.0)

    coarse_zero = jnp.zeros(coarse_shape, dtype=diagonal.dtype)

    def restrict(fine: jnp.ndarray) -> jnp.ndarray:
        return jax.linear_transpose(prolong, coarse_zero)(fine)[0]

    def fine_matvec(field: jnp.ndarray) -> jnp.ndarray:
        return apply_poisson_operator(
            diagonal, west, east, south, north, field, anchor
        )

    def coarse_matvec(field: jnp.ndarray) -> jnp.ndarray:
        return restrict(fine_matvec(prolong(field)))

    coarse_size = coarse_shape[0] * coarse_shape[1]
    basis = jnp.eye(coarse_size, dtype=diagonal.dtype)
    coarse_matrix = jax.vmap(
        lambda column: coarse_matvec(column.reshape(coarse_shape)).reshape(-1)
    )(basis).T
    coarse_matrix = 0.5 * (coarse_matrix + coarse_matrix.T)
    coarse_factors = cho_factor(coarse_matrix, lower=True)

    def coarse_solve(rhs: jnp.ndarray) -> jnp.ndarray:
        return cho_solve(coarse_factors, rhs.reshape(-1)).reshape(coarse_shape)

    def no_smoothing(_matvec, iterate: jnp.ndarray, _rhs: jnp.ndarray) -> jnp.ndarray:
        return iterate

    coarse_correction = _solvax_p_multigrid(
        (fine_matvec,),
        (restrict,),
        (prolong,),
        coarse_solve,
        smoothers=(no_smoothing,),
    )

    def apply(residual: jnp.ndarray) -> jnp.ndarray:
        line_part = line(residual)
        coarse_part = coarse_correction(residual - fine_matvec(line_part))
        corrected = line_part + coarse_part - line(fine_matvec(coarse_part))
        return corrected.at[anchor].set(0.0)

    return apply


@partial(jax.jit, static_argnames=("anchor", "preconditioner"))
def _solve_potential_fgmres_state(
    diagonal: jnp.ndarray,
    west: jnp.ndarray,
    east: jnp.ndarray,
    south: jnp.ndarray,
    north: jnp.ndarray,
    rhs: jnp.ndarray,
    anchor: tuple[int, int],
    *,
    initial: jnp.ndarray,
    residual_scale: jnp.ndarray,
    preconditioner: Callable[[jnp.ndarray], jnp.ndarray],
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Solve the anchored potential system with SOLVAX flexible GMRES."""
    if _solvax_gmres is None:
        raise ImportError("SOLVAX flexible GMRES is unavailable")
    shape = rhs.shape

    def matvec_flat(vector: jnp.ndarray) -> jnp.ndarray:
        field = vector.reshape(shape)
        return apply_poisson_operator(
            diagonal, west, east, south, north, field, anchor
        ).reshape(-1)

    def precondition_flat(vector: jnp.ndarray) -> jnp.ndarray:
        return preconditioner(vector.reshape(shape)).reshape(-1)

    solution = _solvax_gmres(
        matvec_flat,
        rhs.at[anchor].set(0.0).reshape(-1),
        x0=initial.at[anchor].set(0.0).reshape(-1),
        precond=precondition_flat,
        restart=20,
        rtol=_POTENTIAL_FGMRES_RELATIVE_TOLERANCE,
        max_restarts=10,
    )
    phi = solution.x.reshape(shape).at[anchor].set(0.0)
    physical_residual = rhs - apply_five_point_operator(
        diagonal, west, east, south, north, phi
    )
    residual = jnp.max(
        jnp.abs(physical_residual) / jnp.maximum(residual_scale, 1.0e-30)
    )
    return phi, residual, solution.iterations


def _potential_fast_diagonalization_preconditioner(
    mesh: StructuredMesh,
    west: jnp.ndarray,
    east: jnp.ndarray,
    south: jnp.ndarray,
    north: jnp.ndarray,
    anchor: tuple[int, int],
):
    """Approximate the tensor-product Poisson inverse by fast diagonalization.

    The volume-scaled uniform-conductivity operator is the generalized
    Kronecker sum ``Ty (x) Mz + My (x) Tz``.  The mildly stretched z problem
    is diagonalized, while every resulting y problem is solved by SOLVAX's
    batched tridiagonal solver.  Avoiding an eigendecomposition in the strongly
    stretched Hartmann direction is substantially more accurate; outer PCG
    removes the remaining z-eigensolver roundoff.
    """
    if _solvax_tridiagonal_solve is None:
        return None
    dy = mesh.dy.astype(west.dtype)
    dz = mesh.dz.astype(west.dtype)
    west_y = jnp.mean(west / dz[None, :], axis=1)
    east_y = jnp.mean(east / dz[None, :], axis=1)
    south_z = jnp.mean(south / dy[:, None], axis=0)
    north_z = jnp.mean(north / dy[:, None], axis=0)
    operator_z = (
        jnp.diag(south_z + north_z)
        + jnp.diag(-north_z[:-1], 1)
        + jnp.diag(-south_z[1:], -1)
    )
    inv_sqrt_dz = jax.lax.rsqrt(dz)
    eigenvectors_z, eigenvalues_z, _ = jnp.linalg.svd(
        inv_sqrt_dz[:, None] * operator_z * inv_sqrt_dz[None, :]
    )
    eigenvalues_z = eigenvalues_z[::-1]
    eigenvectors_z = eigenvectors_z[:, ::-1]
    modes_z = inv_sqrt_dz[:, None] * eigenvectors_z
    line_diagonal = (
        west_y[:, None]
        + east_y[:, None]
        + dy[:, None] * eigenvalues_z[None, :]
    )
    line_lower = jnp.broadcast_to(-west_y[:, None], line_diagonal.shape)
    line_upper = jnp.broadcast_to(-east_y[:, None], line_diagonal.shape)
    anchor_y = anchor[0]
    line_diagonal = line_diagonal.at[anchor_y, 0].set(1.0)
    line_lower = line_lower.at[anchor_y, 0].set(0.0)
    line_upper = line_upper.at[anchor_y, 0].set(0.0)
    if anchor_y > 0:
        line_upper = line_upper.at[anchor_y - 1, 0].set(0.0)
    if anchor_y + 1 < line_diagonal.shape[0]:
        line_lower = line_lower.at[anchor_y + 1, 0].set(0.0)

    def apply(residual: jnp.ndarray) -> jnp.ndarray:
        # The anchored system replaces one redundant Poisson equation by the
        # gauge. Restore that omitted row so the unanchored tensor operator
        # receives a compatible (zero-sum) right-hand side.
        compatible = residual.at[anchor].set(residual[anchor] - jnp.sum(residual))
        transformed = compatible @ modes_z
        transformed = transformed.at[anchor_y, 0].set(0.0)
        solved_modes = _solvax_tridiagonal_solve(
            line_lower, line_diagonal, line_upper, transformed
        )
        solved = solved_modes @ modes_z.T
        solved = solved - solved[anchor]
        return solved.at[anchor].set(0.0)

    return apply


def _potential_conducting_rectangle_preconditioner(
    mesh: StructuredMesh,
    sigma: jnp.ndarray,
    west: jnp.ndarray,
    east: jnp.ndarray,
    south: jnp.ndarray,
    north: jnp.ndarray,
    anchor: tuple[int, int],
):
    """Build the tensor inverse on a contiguous uniform conducting rectangle."""
    sigma_host = np.asarray(sigma)
    conductive = sigma_host > 0.0
    active_y = np.flatnonzero(np.any(conductive, axis=1))
    active_z = np.flatnonzero(np.any(conductive, axis=0))
    if not active_y.size or not active_z.size:
        return None
    if np.any(np.diff(active_y) != 1) or np.any(np.diff(active_z) != 1):
        return None
    y_slice = slice(int(active_y[0]), int(active_y[-1]) + 1)
    z_slice = slice(int(active_z[0]), int(active_z[-1]) + 1)
    rectangle = np.zeros_like(conductive)
    rectangle[y_slice, z_slice] = True
    positive = sigma_host[conductive]
    if not np.array_equal(conductive, rectangle) or not np.allclose(
        positive, positive[0], rtol=1.0e-12, atol=0.0
    ):
        return None
    if not (conductive[anchor] and y_slice.start <= anchor[0] < y_slice.stop):
        return None

    submesh = generate_rect_duct_mesh_from_faces(
        y_faces=mesh.y_faces[y_slice.start : y_slice.stop + 1],
        z_faces=mesh.z_faces[z_slice.start : z_slice.stop + 1],
        length=float(mesh.x_faces[-1] - mesh.x_faces[0]),
        nx=mesh.nx,
    )
    subanchor = (anchor[0] - y_slice.start, anchor[1] - z_slice.start)
    subsolve = _potential_fast_diagonalization_preconditioner(
        submesh,
        west[y_slice, z_slice],
        east[y_slice, z_slice],
        south[y_slice, z_slice],
        north[y_slice, z_slice],
        subanchor,
    )
    if subsolve is None:
        return None
    conductive_array = jnp.asarray(conductive)

    def apply(residual: jnp.ndarray) -> jnp.ndarray:
        # Disconnected cells carry unit equations after volume scaling.
        solved = jnp.where(conductive_array, 0.0, residual)
        return solved.at[y_slice, z_slice].set(subsolve(residual[y_slice, z_slice]))

    return apply


def _select_potential_preconditioner(
    diagonal: jnp.ndarray,
    west: jnp.ndarray,
    east: jnp.ndarray,
    south: jnp.ndarray,
    north: jnp.ndarray,
    anchor: tuple[int, int],
):
    diagonal_host = np.asarray(diagonal)
    positive = diagonal_host[diagonal_host > 0.0]
    diagonal_ratio = float(positive.max() / positive.min()) if positive.size else 1.0
    if diagonal_ratio >= _POTENTIAL_ADDITIVE_LINE_DIAGONAL_RATIO:
        deflated = _potential_deflated_line_preconditioner(
            diagonal, west, east, south, north, anchor
        )
        if deflated is not None:
            return deflated
    if (
        min(diagonal.shape) < _POTENTIAL_ADDITIVE_LINE_MIN_CELLS
        and diagonal_ratio < _POTENTIAL_ADDITIVE_LINE_DIAGONAL_RATIO
    ):
        return None
    return _potential_additive_line_preconditioner(
        diagonal, west, east, south, north, anchor
    )


def _potential_preconditioner_for_materials(
    mesh: StructuredMesh,
    sigma: jnp.ndarray,
    diagonal: jnp.ndarray,
    west: jnp.ndarray,
    east: jnp.ndarray,
    south: jnp.ndarray,
    north: jnp.ndarray,
    anchor: tuple[int, int],
) -> tuple[Callable[[jnp.ndarray], jnp.ndarray] | None, bool]:
    """Select the strongest valid potential preconditioner for the material map.

    A uniform conducting rectangle may occupy only part of the storage array:
    exact insulating cells carry disconnected unit equations after volume
    scaling.  This is the Hunt topology, so requiring every stored cell to
    conduct would incorrectly bypass the tensor inverse.
    """
    preconditioner = _potential_conducting_rectangle_preconditioner(
        mesh, sigma, west, east, south, north, anchor
    )
    if preconditioner is not None:
        return preconditioner, _solvax_gmres is not None

    preconditioner = _select_potential_preconditioner(
        diagonal, west, east, south, north, anchor
    )
    if preconditioner is None:
        preconditioner = _potential_additive_line_preconditioner(
            diagonal, west, east, south, north, anchor
        )
    return preconditioner, False


def _build_mesh(case: CaseSpec) -> StructuredMesh:
    g = case.geometry
    magnetic_axis = None
    if case.magnetic_field.kind == "constant" and case.magnetic_field.value is not None:
        bx, by, bz = case.magnetic_field.value
        magnitudes = {"x": abs(bx), "y": abs(by), "z": abs(bz)}
        dominant = max(magnitudes, key=magnitudes.get)
        magnetic_axis = dominant if magnitudes[dominant] > 0.0 else None
    if g.kind == "rect_duct":
        return generate_rect_duct_mesh(
            width=g.width,
            height=g.height,
            length=g.length,
            nx=g.nx,
            ny=g.ny,
            nz=g.nz,
            target_ha=g.target_ha,
            magnetic_axis=magnetic_axis,
        )
    if g.kind == "layered_duct":
        return generate_layered_duct_mesh(
            width=g.width,
            height=g.height,
            length=g.length,
            nx=g.nx,
            ny=g.ny,
            nz=g.nz,
            wall_thickness=g.wall_thickness,
            wall_cells=g.wall_cells,
            target_ha=g.target_ha,
            magnetic_axis=magnetic_axis,
        )
    raise NotImplementedError(f"Geometry {g.kind} is not supported by the laminar solver yet.")


def _bounded_time_step_count(*, start_time: float, dt: float, t_final: float, max_steps: int) -> int:
    if dt <= 0.0:
        raise ValueError("Time-step size dt must be positive")
    if max_steps <= 0:
        return 0
    remaining_time = max(0.0, float(t_final) - float(start_time))
    if remaining_time <= 0.0:
        return 0
    ratio = remaining_time / float(dt)
    tolerance = 16.0 * np.finfo(float).eps * max(1.0, abs(ratio))
    allowed_steps = int(np.floor(ratio + tolerance))
    return min(int(max_steps), max(0, allowed_steps))


def _interface_conductance_y(mesh: StructuredMesh, sigma: jnp.ndarray) -> jnp.ndarray:
    left_distance = 0.5 * mesh.dy[:-1, None]
    right_distance = 0.5 * mesh.dy[1:, None]
    sigma_left = sigma[:-1, :]
    sigma_right = sigma[1:, :]
    connected = (sigma_left > 0.0) & (sigma_right > 0.0)
    safe_left = jnp.where(connected, sigma_left, 1.0)
    safe_right = jnp.where(connected, sigma_right, 1.0)
    resistance = left_distance / safe_left + right_distance / safe_right
    return jnp.where(connected, 1.0 / jnp.maximum(resistance, 1e-30), 0.0)


def _face_conductance_y(mesh: StructuredMesh, sigma: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    conductance = _interface_conductance_y(mesh, sigma)
    west = jnp.pad(conductance, ((1, 0), (0, 0))) / mesh.dy[:, None]
    east = jnp.pad(conductance, ((0, 1), (0, 0))) / mesh.dy[:, None]
    return west, east


def _interface_conductance_z(mesh: StructuredMesh, sigma: jnp.ndarray) -> jnp.ndarray:
    left_distance = 0.5 * mesh.dz[None, :-1]
    right_distance = 0.5 * mesh.dz[None, 1:]
    sigma_left = sigma[:, :-1]
    sigma_right = sigma[:, 1:]
    connected = (sigma_left > 0.0) & (sigma_right > 0.0)
    safe_left = jnp.where(connected, sigma_left, 1.0)
    safe_right = jnp.where(connected, sigma_right, 1.0)
    resistance = left_distance / safe_left + right_distance / safe_right
    return jnp.where(connected, 1.0 / jnp.maximum(resistance, 1e-30), 0.0)


def _face_conductance_z(mesh: StructuredMesh, sigma: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    conductance = _interface_conductance_z(mesh, sigma)
    south = jnp.pad(conductance, ((0, 0), (1, 0))) / mesh.dz[None, :]
    north = jnp.pad(conductance, ((0, 0), (0, 1))) / mesh.dz[None, :]
    return south, north


def _face_emf_y(mesh: StructuredMesh, sigma: jnp.ndarray, source: jnp.ndarray) -> jnp.ndarray:
    left_distance = 0.5 * mesh.dy[:-1, None]
    right_distance = 0.5 * mesh.dy[1:, None]
    conductance = _interface_conductance_y(mesh, sigma)
    return conductance * (left_distance * source[:-1, :] + right_distance * source[1:, :])


def _face_emf_z(mesh: StructuredMesh, sigma: jnp.ndarray, source: jnp.ndarray) -> jnp.ndarray:
    left_distance = 0.5 * mesh.dz[None, :-1]
    right_distance = 0.5 * mesh.dz[None, 1:]
    conductance = _interface_conductance_z(mesh, sigma)
    return conductance * (left_distance * source[:, :-1] + right_distance * source[:, 1:])


def _potential_coefficients(mesh: StructuredMesh, sigma: jnp.ndarray) -> tuple[jnp.ndarray, ...]:
    west, east = _face_conductance_y(mesh, sigma)
    south, north = _face_conductance_z(mesh, sigma)
    diagonal = west + east + south + north
    diagonal = jnp.where(diagonal > 0.0, diagonal, 1.0)
    return diagonal, west, east, south, north


def _cell_metric(mesh: StructuredMesh) -> jnp.ndarray:
    return mesh.dy[:, None] * mesh.dz[None, :]


def _active_velocity_mask_for_solver(fluid_mask: jnp.ndarray, solver_kind: str) -> jnp.ndarray:
    if solver_kind == "fully_developed_inductionless":
        return _active_velocity_mask(fluid_mask)
    return fluid_mask


def _connected_interface_diffusivity_y(mesh: StructuredMesh, diffusivity: jnp.ndarray, active_mask: jnp.ndarray) -> jnp.ndarray:
    connected = active_mask[:-1, :] & active_mask[1:, :]
    left_distance = 0.5 * mesh.dy[:-1, None]
    right_distance = 0.5 * mesh.dy[1:, None]
    diffusivity_left = jnp.maximum(diffusivity[:-1, :], 1e-12)
    diffusivity_right = jnp.maximum(diffusivity[1:, :], 1e-12)
    conductance = 1.0 / jnp.maximum(left_distance / diffusivity_left + right_distance / diffusivity_right, 1e-12)
    return jnp.where(connected, conductance, 0.0)


def _connected_interface_diffusivity_z(mesh: StructuredMesh, diffusivity: jnp.ndarray, active_mask: jnp.ndarray) -> jnp.ndarray:
    connected = active_mask[:, :-1] & active_mask[:, 1:]
    left_distance = 0.5 * mesh.dz[None, :-1]
    right_distance = 0.5 * mesh.dz[None, 1:]
    diffusivity_left = jnp.maximum(diffusivity[:, :-1], 1e-12)
    diffusivity_right = jnp.maximum(diffusivity[:, 1:], 1e-12)
    conductance = 1.0 / jnp.maximum(left_distance / diffusivity_left + right_distance / diffusivity_right, 1e-12)
    return jnp.where(connected, conductance, 0.0)


def _velocity_system_coefficients(
    mesh: StructuredMesh,
    diffusivity: jnp.ndarray,
    reaction: jnp.ndarray,
    active_mask: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    dy = mesh.dy[:, None]
    dz = mesh.dz[None, :]
    active = active_mask.astype(diffusivity.dtype)

    west_connected = active_mask & jnp.pad(active_mask[:-1, :], ((1, 0), (0, 0)))
    east_connected = active_mask & jnp.pad(active_mask[1:, :], ((0, 1), (0, 0)))
    south_connected = active_mask & jnp.pad(active_mask[:, :-1], ((0, 0), (1, 0)))
    north_connected = active_mask & jnp.pad(active_mask[:, 1:], ((0, 0), (0, 1)))

    interface_y = _connected_interface_diffusivity_y(mesh, diffusivity, active_mask)
    interface_z = _connected_interface_diffusivity_z(mesh, diffusivity, active_mask)
    west = jnp.where(
        west_connected,
        jnp.pad(interface_y, ((1, 0), (0, 0))) / jnp.maximum(dy, 1e-12),
        jnp.where(active_mask, 2.0 * diffusivity / jnp.maximum(dy**2, 1e-12), 0.0),
    )
    east = jnp.where(
        east_connected,
        jnp.pad(interface_y, ((0, 1), (0, 0))) / jnp.maximum(dy, 1e-12),
        jnp.where(active_mask, 2.0 * diffusivity / jnp.maximum(dy**2, 1e-12), 0.0),
    )
    south = jnp.where(
        south_connected,
        jnp.pad(interface_z, ((0, 0), (1, 0))) / jnp.maximum(dz, 1e-12),
        jnp.where(active_mask, 2.0 * diffusivity / jnp.maximum(dz**2, 1e-12), 0.0),
    )
    north = jnp.where(
        north_connected,
        jnp.pad(interface_z, ((0, 0), (0, 1))) / jnp.maximum(dz, 1e-12),
        jnp.where(active_mask, 2.0 * diffusivity / jnp.maximum(dz**2, 1e-12), 0.0),
    )
    diagonal = west + east + south + north + jnp.where(active_mask, reaction, 1.0 - active)
    diagonal = jnp.where(active_mask, diagonal, 1.0)
    west = jnp.where(active_mask, west, 0.0)
    east = jnp.where(active_mask, east, 0.0)
    south = jnp.where(active_mask, south, 0.0)
    north = jnp.where(active_mask, north, 0.0)
    return diagonal, west, east, south, north


def _solve_velocity_system(
    *,
    mesh: StructuredMesh,
    diffusivity: jnp.ndarray,
    reaction: jnp.ndarray,
    rhs: jnp.ndarray,
    active_mask: jnp.ndarray,
    linear_solver: str,
    preconditioner: str,
    max_steps: int,
    tolerance: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    diagonal, west, east, south, north = _velocity_system_coefficients(mesh, diffusivity, reaction, active_mask)
    rhs_masked = jnp.where(active_mask, rhs, 0.0)
    cell_metric = _cell_metric(mesh).astype(rhs_masked.dtype)
    initial_residual = five_point_residual_norm(
        diagonal * cell_metric,
        west * cell_metric,
        east * cell_metric,
        south * cell_metric,
        north * cell_metric,
        rhs_masked * cell_metric,
        jnp.zeros_like(rhs_masked),
    )
    field, info = solve_five_point_system(
        diagonal * cell_metric,
        west * cell_metric,
        east * cell_metric,
        south * cell_metric,
        north * cell_metric,
        rhs_masked * cell_metric,
        linear_solver=linear_solver,
        preconditioner=preconditioner,
        tolerance=tolerance,
        max_steps=max_steps,
    )
    field = jnp.where(active_mask, field, 0.0)
    return (
        field,
        jnp.asarray(info.residual, dtype=rhs.dtype),
        jnp.asarray(info.iterations, dtype=jnp.int32),
        jnp.asarray(initial_residual, dtype=rhs.dtype),
    )


def _fully_developed_rhs(
    *,
    mesh: StructuredMesh,
    sigma: jnp.ndarray,
    rho: jnp.ndarray,
    fluid_mask: jnp.ndarray,
    u: jnp.ndarray,
    phi: jnp.ndarray,
    by: jnp.ndarray,
    bz: jnp.ndarray,
    forcing: jnp.ndarray,
    current_reconstruction: str = "cell_centered",
) -> tuple[jnp.ndarray, jnp.ndarray]:
    _, _, lorentz_source = _compute_current_and_lorentz(
        mesh,
        sigma,
        fluid_mask,
        u,
        phi,
        by,
        bz,
        reconstruction=current_reconstruction,
    )
    rhs = jnp.where(fluid_mask, (forcing + lorentz_source) / rho, 0.0)
    return rhs, lorentz_source


def _volume_scaled_potential_system(
    mesh: StructuredMesh,
    diagonal: jnp.ndarray,
    west: jnp.ndarray,
    east: jnp.ndarray,
    south: jnp.ndarray,
    north: jnp.ndarray,
    rhs: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    cell_metric = _cell_metric(mesh)
    connected = (west + east + south + north) > 0.0
    return (
        jnp.where(connected, diagonal * cell_metric, 1.0),
        west * cell_metric,
        east * cell_metric,
        south * cell_metric,
        north * cell_metric,
        jnp.where(connected, rhs * cell_metric, 0.0),
    )


def _solve_potential(
    mesh: StructuredMesh,
    sigma: jnp.ndarray,
    fluid_mask: jnp.ndarray,
    u: jnp.ndarray,
    by: jnp.ndarray,
    bz: jnp.ndarray,
    anchor: tuple[int, int],
    iterations: int,
    tolerance: float | None = None,
    relaxation: float = 1.0,
    solver: str = "jacobi",
    initial_phi: jnp.ndarray | None = None,
    potential_preconditioner: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    potential_flexible: bool = False,
    return_solver_residual: bool = False,
) -> tuple[jnp.ndarray, ...]:
    uxb_y = jnp.where(fluid_mask, -u * bz, 0.0)
    uxb_z = jnp.where(fluid_mask, u * by, 0.0)
    conv_y = _face_emf_y(mesh, sigma, uxb_y)
    conv_z = _face_emf_z(mesh, sigma, uxb_z)
    face_conv_y = jnp.pad(conv_y, ((1, 1), (0, 0)))
    face_conv_z = jnp.pad(conv_z, ((0, 0), (1, 1)))
    rhs = -(
        (face_conv_y[1:, :] - face_conv_y[:-1, :]) / mesh.dy[:, None]
        + (face_conv_z[:, 1:] - face_conv_z[:, :-1]) / mesh.dz[None, :]
    )
    cell_metric = _cell_metric(mesh).astype(rhs.dtype)
    conductive_weight = jnp.where(sigma > 0.0, cell_metric, 0.0)
    conductive_total_weight = jnp.maximum(jnp.sum(conductive_weight), 1.0e-20)
    rhs_mean = jnp.sum(conductive_weight * rhs) / conductive_total_weight
    rhs = jnp.where(sigma > 0.0, rhs - rhs_mean, 0.0)

    diagonal, west, east, south, north = _potential_coefficients(mesh, sigma)
    warm_start = jnp.zeros_like(rhs) if initial_phi is None else jnp.asarray(initial_phi)
    if warm_start.shape != rhs.shape:
        raise ValueError("Potential initial guess must match the potential field shape")
    use_warm_start = tolerance is not None and solver in {"cg", "cg_volume"}
    solve_start = warm_start if use_warm_start else jnp.zeros_like(rhs)
    initial_residual = poisson_residual_norm(
        diagonal,
        west,
        east,
        south,
        north,
        rhs,
        solve_start,
        anchor,
    )
    if solver == "jacobi":
        phi, residual, iteration_count = solve_poisson_jacobi_state(
            diagonal,
            west,
            east,
            south,
            north,
            rhs,
            anchor,
            iterations,
            tolerance=tolerance,
            relaxation=relaxation,
        )
        solver_residual = residual
    elif solver == "cg":
        phi, residual, iteration_count = solve_poisson_cg_state(
            diagonal,
            west,
            east,
            south,
            north,
            rhs,
            anchor,
            iterations,
            tolerance=tolerance,
            initial=solve_start,
        )
        solver_residual = residual
    elif solver == "cg_volume":
        diagonal_scaled, west_scaled, east_scaled, south_scaled, north_scaled, rhs_scaled = _volume_scaled_potential_system(
            mesh,
            diagonal,
            west,
            east,
            south,
            north,
            rhs,
        )
        selected_preconditioner = (
            potential_preconditioner
            if potential_preconditioner is not None
            else _select_potential_preconditioner(
                diagonal_scaled,
                west_scaled,
                east_scaled,
                south_scaled,
                north_scaled,
                anchor,
            )
        )
        if potential_flexible and selected_preconditioner is not None:
            phi, solver_residual, iteration_count = _solve_potential_fgmres_state(
                diagonal_scaled,
                west_scaled,
                east_scaled,
                south_scaled,
                north_scaled,
                rhs_scaled,
                anchor,
                # The tensor inverse makes a zero-start solve inexpensive and
                # keeps the coupled fixed-point map independent of the prior
                # potential iterate.  An inexact warm start otherwise changes
                # the map at the residual floor and prevents strict outer
                # convergence on high-Ha Hunt cases.
                initial=jnp.zeros_like(solve_start),
                residual_scale=_cell_metric(mesh),
                preconditioner=selected_preconditioner,
            )
        else:
            phi, solver_residual, iteration_count = solve_poisson_cg_state(
                diagonal_scaled,
                west_scaled,
                east_scaled,
                south_scaled,
                north_scaled,
                rhs_scaled,
                anchor,
                iterations,
                tolerance=tolerance,
                initial=solve_start,
                residual_scale=_cell_metric(mesh),
                preconditioner=selected_preconditioner,
            )
        residual = poisson_residual_norm(diagonal, west, east, south, north, rhs, phi, anchor)
    elif solver == "lineax_cg":
        phi, info = solve_poisson_lineax(
            diagonal,
            west,
            east,
            south,
            north,
            rhs,
            anchor,
            fallback_iterations=iterations,
            max_steps=iterations,
        )
        residual = jnp.asarray(info.residual, dtype=rhs.dtype)
        solver_residual = residual
        iteration_count = jnp.asarray(info.iterations, dtype=jnp.int32)
    else:
        raise ValueError(f"Unsupported potential solver backend {solver!r}")
    result = (phi, residual, iteration_count, initial_residual)
    if return_solver_residual:
        return (*result, solver_residual)
    return result


def _resolve_potential_solver(solver: str, fluid_mask: jnp.ndarray | None) -> str:
    if solver != "auto":
        return solver
    if fluid_mask is None:
        return "cg"
    return "cg" if bool(np.asarray(fluid_mask).all()) else "cg_volume"


def _has_uniform_spacing(mesh: StructuredMesh, *, tolerance: float = 1.0e-12) -> bool:
    dy = np.asarray(mesh.dy, dtype=float)
    dz = np.asarray(mesh.dz, dtype=float)
    dy_uniform = np.allclose(dy, dy[0], rtol=0.0, atol=tolerance) if dy.size else True
    dz_uniform = np.allclose(dz, dz[0], rtol=0.0, atol=tolerance) if dz.size else True
    return bool(dy_uniform and dz_uniform)


def _compute_current_and_lorentz(
    mesh: StructuredMesh,
    sigma: jnp.ndarray,
    fluid_mask: jnp.ndarray,
    u: jnp.ndarray,
    phi: jnp.ndarray,
    by: jnp.ndarray,
    bz: jnp.ndarray,
    reconstruction: str = "cell_centered",
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    use_face_currents = reconstruction in {"face_averaged", "hybrid_face_lorentz"}
    if use_face_currents:
        uxb_y = jnp.where(fluid_mask, -u * bz, 0.0)
        uxb_z = jnp.where(fluid_mask, u * by, 0.0)
        face_jy = _interface_conductance_y(mesh, sigma) * (phi[:-1, :] - phi[1:, :]) + _face_emf_y(mesh, sigma, uxb_y)
        face_jz = _interface_conductance_z(mesh, sigma) * (phi[:, :-1] - phi[:, 1:]) + _face_emf_z(mesh, sigma, uxb_z)
        face_jy_centered = 0.5 * (jnp.pad(face_jy, ((1, 0), (0, 0))) + jnp.pad(face_jy, ((0, 1), (0, 0))))
        face_jz_centered = 0.5 * (jnp.pad(face_jz, ((0, 0), (1, 0))) + jnp.pad(face_jz, ((0, 0), (0, 1))))
    if reconstruction == "face_averaged":
        jy = face_jy_centered
        jz = face_jz_centered
        lorentz_x = jy * bz - jz * by
    else:
        jy, jz = _conductive_current_components(mesh, sigma, fluid_mask, u, phi, by, bz)
        lorentz_x = jy * bz - jz * by
        if reconstruction == "hybrid_face_lorentz":
            lorentz_x = face_jy_centered * bz - face_jz_centered * by
    jy = jnp.where(fluid_mask, jy, 0.0)
    jz = jnp.where(fluid_mask, jz, 0.0)
    lorentz_x = jnp.where(fluid_mask, lorentz_x, 0.0)
    return jy, jz, lorentz_x


def _face_current_components(
    mesh: StructuredMesh,
    sigma: jnp.ndarray,
    fluid_mask: jnp.ndarray,
    u: jnp.ndarray,
    phi: jnp.ndarray,
    by: jnp.ndarray,
    bz: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    uxb_y = jnp.where(fluid_mask, -u * bz, 0.0)
    uxb_z = jnp.where(fluid_mask, u * by, 0.0)
    emf_y = _face_emf_y(mesh, sigma, uxb_y)
    emf_z = _face_emf_z(mesh, sigma, uxb_z)
    face_jy = _interface_conductance_y(mesh, sigma) * (phi[:-1, :] - phi[1:, :]) + emf_y
    face_jz = _interface_conductance_z(mesh, sigma) * (phi[:, :-1] - phi[:, 1:]) + emf_z
    return face_jy, face_jz, emf_y, emf_z


def _conductive_current_components(
    mesh: StructuredMesh,
    sigma: jnp.ndarray,
    fluid_mask: jnp.ndarray,
    u: jnp.ndarray,
    phi: jnp.ndarray,
    by: jnp.ndarray,
    bz: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    dphi_dy, dphi_dz = gradient_scalar(phi, mesh)
    fluid_velocity = jnp.where(fluid_mask, u, 0.0)
    jy = sigma * (-dphi_dy - fluid_velocity * bz)
    jz = sigma * (-dphi_dz + fluid_velocity * by)
    return jy, jz


def _face_current_emf_and_lorentz_max(
    mesh: StructuredMesh,
    sigma: jnp.ndarray,
    fluid_mask: jnp.ndarray,
    u: jnp.ndarray,
    phi: jnp.ndarray,
    by: jnp.ndarray,
    bz: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    face_jy, face_jz, emf_y, emf_z = _face_current_components(mesh, sigma, fluid_mask, u, phi, by, bz)
    max_face_current = jnp.maximum(jnp.max(jnp.abs(face_jy)), jnp.max(jnp.abs(face_jz)))
    max_emf = jnp.maximum(jnp.max(jnp.abs(emf_y)), jnp.max(jnp.abs(emf_z)))
    face_bz = 0.5 * (bz[:-1, :] + bz[1:, :])
    face_by = 0.5 * (by[:, :-1] + by[:, 1:])
    max_face_lorentz = jnp.maximum(
        jnp.max(jnp.abs(face_jy * face_bz)),
        jnp.max(jnp.abs(face_jz * face_by)),
    )
    return max_face_current, max_emf, max_face_lorentz


def _integral_diagnostics(
    *,
    mesh: StructuredMesh,
    sigma: jnp.ndarray,
    fluid_mask: jnp.ndarray,
    u: jnp.ndarray,
    phi: jnp.ndarray,
    jy: jnp.ndarray,
    jz: jnp.ndarray,
    lorentz: jnp.ndarray,
    by: jnp.ndarray,
    bz: jnp.ndarray,
    anchor: tuple[int, int],
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    cell_metric = _cell_metric(mesh).astype(u.dtype)
    fluid_weight = jnp.where(fluid_mask, cell_metric, 0.0)
    fluid_total_weight = jnp.maximum(jnp.sum(fluid_weight), 1e-20)
    volumetric_flow_rate = jnp.sum(fluid_weight * u)
    mean_current_magnitude = jnp.sum(fluid_weight * jnp.sqrt(jy**2 + jz**2)) / fluid_total_weight
    lorentz_power = jnp.sum(fluid_weight * lorentz * u)
    face_jy, face_jz, _, _ = _face_current_components(mesh, sigma, fluid_mask, u, phi, by, bz)
    padded_face_jy = jnp.pad(face_jy, ((1, 1), (0, 0)))
    padded_face_jz = jnp.pad(face_jz, ((0, 0), (1, 1)))
    div_current = (
        (padded_face_jy[1:, :] - padded_face_jy[:-1, :]) / mesh.dy[:, None]
        + (padded_face_jz[:, 1:] - padded_face_jz[:, :-1]) / mesh.dz[None, :]
    )
    div_current_max = jnp.max(jnp.where(fluid_mask, jnp.abs(div_current), 0.0))
    charge_balance_residual = jnp.abs(jnp.sum(fluid_weight * div_current)) / fluid_total_weight
    gauge_residual = jnp.abs(phi[anchor])
    conductivity_jump_y = jnp.abs(sigma[:-1, :] - sigma[1:, :]) > 1e-12
    conductivity_jump_z = jnp.abs(sigma[:, :-1] - sigma[:, 1:]) > 1e-12
    interface_mask_y = conductivity_jump_y | (fluid_mask[:-1, :] != fluid_mask[1:, :])
    interface_mask_z = conductivity_jump_z | (fluid_mask[:, :-1] != fluid_mask[:, 1:])
    interface_cells = jnp.zeros_like(fluid_mask, dtype=bool)
    interface_cells = interface_cells.at[:-1, :].set(interface_cells[:-1, :] | interface_mask_y)
    interface_cells = interface_cells.at[1:, :].set(interface_cells[1:, :] | interface_mask_y)
    interface_cells = interface_cells.at[:, :-1].set(interface_cells[:, :-1] | interface_mask_z)
    interface_cells = interface_cells.at[:, 1:].set(interface_cells[:, 1:] | interface_mask_z)
    # A conservative face is shared by its two neighboring cells, so comparing
    # that face with a cell-centered average measures a physical gradient, not
    # a continuity defect.  Instead report the finite-volume current imbalance
    # in interface-adjacent cells, converted from A/m^3 to an A/m^2 flux scale
    # with the local characteristic cell length.
    local_length = jnp.sqrt(cell_metric)
    interface_current_residual = jnp.max(
        jnp.where(interface_cells, jnp.abs(div_current) * local_length, 0.0),
        initial=0.0,
    )
    return (
        volumetric_flow_rate,
        mean_current_magnitude,
        lorentz_power,
        div_current_max,
        charge_balance_residual,
        gauge_residual,
        interface_current_residual,
    )


def fully_developed_power_balance(case: CaseSpec, solution: Solution) -> dict[str, float]:
    """Return final per-unit-length mechanical and electrical power balances.

    Pressure, Lorentz, viscous, and Joule terms are all dimensional W/m.  The
    Joule audit reconstructs current in explicit wall cells instead of using
    the fluid-masked state fields.
    """

    mesh = solution.mesh
    materials = build_material_fields(case, mesh)
    fluid_mask = materials.fluid_mask
    metric = _cell_metric(mesh).astype(solution.state.u.dtype)
    fluid_metric = jnp.where(fluid_mask, metric, 0.0)
    _, by, bz = magnetic_field_components(case.magnetic_field, mesh, time=solution.state.time)
    face_jy, face_jz, emf_y, emf_z = _face_current_components(
        mesh,
        materials.conductivity,
        fluid_mask,
        solution.state.u,
        solution.state.phi,
        by,
        bz,
    )
    conductance_y = _interface_conductance_y(mesh, materials.conductivity)
    conductance_z = _interface_conductance_z(mesh, materials.conductivity)
    face_area_y = mesh.dz[None, :]
    face_area_z = mesh.dy[:, None]
    joule_dissipation = jnp.sum(
        face_area_y
        * jnp.where(conductance_y > 0.0, face_jy**2 / jnp.maximum(conductance_y, 1.0e-30), 0.0)
    ) + jnp.sum(
        face_area_z
        * jnp.where(conductance_z > 0.0, face_jz**2 / jnp.maximum(conductance_z, 1.0e-30), 0.0)
    )
    emf_field_y = jnp.where(
        conductance_y > 0.0, emf_y / jnp.maximum(conductance_y, 1.0e-30), 0.0
    )
    emf_field_z = jnp.where(
        conductance_z > 0.0, emf_z / jnp.maximum(conductance_z, 1.0e-30), 0.0
    )
    emf_power = jnp.sum(face_area_y * face_jy * emf_field_y) + jnp.sum(
        face_area_z * face_jz * emf_field_z
    )
    lorentz_work = jnp.sum(fluid_metric * solution.state.lorentz_x * solution.state.u)
    flow_rate = jnp.sum(fluid_metric * solution.state.u)
    applied_forcing = (
        solution.diagnostics.applied_forcing_history[-1]
        if solution.diagnostics.applied_forcing_history.size
        else jnp.asarray(case.forcing, dtype=solution.state.u.dtype)
    )
    pressure_power = applied_forcing * flow_rate

    dynamic_viscosity = materials.density * materials.viscosity
    zeros = jnp.zeros_like(dynamic_viscosity)
    diagonal, west, east, south, north = _velocity_system_coefficients(
        mesh,
        dynamic_viscosity,
        zeros,
        fluid_mask,
    )
    viscous_operator_u = apply_five_point_operator(
        diagonal * metric,
        west * metric,
        east * metric,
        south * metric,
        north * metric,
        solution.state.u,
    )
    viscous_dissipation = jnp.sum(solution.state.u * viscous_operator_u)
    electrical_residual = joule_dissipation + lorentz_work
    network_electrical_residual = joule_dissipation - emf_power
    lorentz_transfer_residual = lorentz_work + emf_power
    mechanical_residual = pressure_power + lorentz_work - viscous_dissipation
    electrical_scale = jnp.maximum(
        jnp.maximum(jnp.abs(joule_dissipation), jnp.abs(lorentz_work)), 1.0e-30
    )
    mechanical_scale = jnp.maximum(
        jnp.maximum(jnp.abs(pressure_power), jnp.abs(lorentz_work) + jnp.abs(viscous_dissipation)),
        1.0e-30,
    )
    return {
        "pressure_power": float(pressure_power),
        "lorentz_work": float(lorentz_work),
        "viscous_dissipation": float(viscous_dissipation),
        "joule_dissipation": float(joule_dissipation),
        "emf_power": float(emf_power),
        "electrical_power_residual": float(electrical_residual),
        "electrical_power_relative_error": float(jnp.abs(electrical_residual) / electrical_scale),
        "network_electrical_residual": float(network_electrical_residual),
        "network_electrical_relative_error": float(
            jnp.abs(network_electrical_residual) / electrical_scale
        ),
        "lorentz_transfer_residual": float(lorentz_transfer_residual),
        "lorentz_transfer_relative_error": float(
            jnp.abs(lorentz_transfer_residual) / electrical_scale
        ),
        "mechanical_power_residual": float(mechanical_residual),
        "mechanical_power_relative_error": float(jnp.abs(mechanical_residual) / mechanical_scale),
    }


def _enforce_velocity_bc(
    u: jnp.ndarray,
    mesh: StructuredMesh,
    fluid_mask: jnp.ndarray,
    *,
    interpolate_direct_fluid_walls: bool = False,
) -> jnp.ndarray:
    zero = jnp.asarray(0.0, dtype=u.dtype)
    u = jnp.where(fluid_mask, u, zero)
    if not interpolate_direct_fluid_walls:
        u = u.at[0, :].set(zero)
        u = u.at[-1, :].set(zero)
        u = u.at[:, 0].set(zero)
        u = u.at[:, -1].set(zero)
        return u
    direct_west = fluid_mask[0, :]
    direct_east = fluid_mask[-1, :]
    direct_south = fluid_mask[:, 0]
    direct_north = fluid_mask[:, -1]
    if u.shape[0] > 1:
        west_ratio = (mesh.y_centers[0] - mesh.y_faces[0]) / jnp.maximum(mesh.y_centers[1] - mesh.y_faces[0], 1e-12)
        east_ratio = (mesh.y_faces[-1] - mesh.y_centers[-1]) / jnp.maximum(mesh.y_faces[-1] - mesh.y_centers[-2], 1e-12)
        west_scale = jnp.where(
            direct_west,
            west_ratio * fluid_mask[1, :].astype(u.dtype),
            zero,
        ).astype(u.dtype)
        east_scale = jnp.where(
            direct_east,
            east_ratio * fluid_mask[-2, :].astype(u.dtype),
            zero,
        ).astype(u.dtype)
        u = u.at[0, :].set(jnp.where(direct_west, west_scale * u[1, :], u[0, :]))
        u = u.at[-1, :].set(jnp.where(direct_east, east_scale * u[-2, :], u[-1, :]))
    if u.shape[1] > 1:
        south_ratio = (mesh.z_centers[0] - mesh.z_faces[0]) / jnp.maximum(mesh.z_centers[1] - mesh.z_faces[0], 1e-12)
        north_ratio = (mesh.z_faces[-1] - mesh.z_centers[-1]) / jnp.maximum(mesh.z_faces[-1] - mesh.z_centers[-2], 1e-12)
        south_scale = jnp.where(
            direct_south,
            south_ratio * fluid_mask[:, 1].astype(u.dtype),
            zero,
        ).astype(u.dtype)
        north_scale = jnp.where(
            direct_north,
            north_ratio * fluid_mask[:, -2].astype(u.dtype),
            zero,
        ).astype(u.dtype)
        u = u.at[:, 0].set(jnp.where(direct_south, south_scale * u[:, 1], u[:, 0]))
        u = u.at[:, -1].set(jnp.where(direct_north, north_scale * u[:, -2], u[:, -1]))
    return u


def _limited_velocity_update(
    current: jnp.ndarray,
    trial: jnp.ndarray,
    fluid_mask: jnp.ndarray,
    max_delta: float = 1e-3,
    limiter: str = "global_scale",
) -> jnp.ndarray:
    delta = jnp.where(fluid_mask, trial - current, 0.0)
    if limiter == "local_clip":
        clipped_delta = jnp.clip(delta, -max_delta, max_delta)
        return jnp.where(fluid_mask, current + clipped_delta, 0.0)
    if limiter != "global_scale":
        raise ValueError(f"Unsupported velocity update limiter {limiter!r}")
    peak_delta = jnp.max(jnp.abs(delta))
    scale = jnp.minimum(1.0, max_delta / jnp.maximum(peak_delta, 1e-12))
    return jnp.where(fluid_mask, current + scale * delta, 0.0)


def _enforce_target_mean_velocity(
    u: jnp.ndarray,
    mesh: StructuredMesh,
    fluid_mask: jnp.ndarray,
    target_mean_velocity: float | None,
) -> jnp.ndarray:
    if target_mean_velocity is None:
        return jnp.where(fluid_mask, u, 0.0)
    cell_metric = _cell_metric(mesh).astype(u.dtype)
    fluid_weight = jnp.where(fluid_mask, cell_metric, 0.0)
    fluid_total_weight = jnp.maximum(jnp.sum(fluid_weight), 1e-20)
    current_mean = jnp.sum(fluid_weight * u) / fluid_total_weight
    target = jnp.asarray(target_mean_velocity, dtype=u.dtype)
    zero_target = jnp.abs(target) <= 1e-20
    safe_mean = jnp.where(jnp.abs(current_mean) > 1e-20, current_mean, 1.0)
    scaled = jnp.where(zero_target, jnp.zeros_like(u), u * (target / safe_mean))
    return jnp.where(fluid_mask, scaled, 0.0)


def _velocity_update_statistics(
    current: jnp.ndarray,
    trial: jnp.ndarray,
    fluid_mask: jnp.ndarray,
    *,
    max_delta: float,
    limiter: str,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    delta = jnp.where(fluid_mask, trial - current, 0.0)
    peak_delta = jnp.max(jnp.abs(delta))
    active_count = jnp.maximum(jnp.sum(fluid_mask.astype(delta.dtype)), 1.0)
    limited_fraction = jnp.sum(
        jnp.where(fluid_mask, (jnp.abs(delta) > max_delta).astype(delta.dtype), 0.0)
    ) / active_count
    if limiter not in {"global_scale", "local_clip"}:
        raise ValueError(f"Unsupported velocity update limiter {limiter!r}")
    scale = jnp.minimum(1.0, max_delta / jnp.maximum(peak_delta, 1e-12))
    return peak_delta, scale, limited_fraction


def _active_velocity_mask(fluid_mask: jnp.ndarray) -> jnp.ndarray:
    active = jnp.array(fluid_mask, copy=True)
    active = active.at[0, :].set(False)
    active = active.at[-1, :].set(False)
    active = active.at[:, 0].set(False)
    active = active.at[:, -1].set(False)
    return active


def _inlet_speed(boundary: BoundaryCondition, case: CaseSpec) -> float | None:
    if boundary.kind == "inlet_velocity":
        value = boundary.value
        if isinstance(value, tuple):
            axis = (boundary.axis or "x").lower()
            component = {"x": 0, "y": 1, "z": 2}.get(axis, 0)
            return float(value[component])
        if isinstance(value, (int, float)):
            return float(value)
    if boundary.kind == "inlet_flow_rate" and isinstance(boundary.value, (int, float)):
        area = case.geometry.width * case.geometry.height
        if area > 0.0:
            return float(boundary.value) / area
    return None


def _target_mean_velocity(case: CaseSpec) -> float | None:
    if abs(case.forcing) != 0.0:
        return None
    for boundary in case.boundary_conditions:
        if boundary.kind != "inlet_flow_rate":
            continue
        speed = _inlet_speed(boundary, case)
        if speed is not None:
            return speed
    return None


def _reference_mean_velocity(case: CaseSpec) -> float | None:
    for boundary in case.boundary_conditions:
        speed = _inlet_speed(boundary, case)
        if speed is not None:
            return speed
    if abs(case.initial_velocity) > 0.0:
        return float(case.initial_velocity)
    return None


def _explicit_forcing(explicit_forcing: float, dtype: jnp.dtype) -> jnp.ndarray:
    return jnp.asarray(explicit_forcing, dtype=dtype)


def _emit_solver_header(
    logger,
    *,
    case,
    mesh,
    materials,
    mode,
    potential_solver,
    target_mean_velocity,
    reference_mean_velocity,
    restart: RestartLogInfo | None = None,
):
    if logger is None:
        return
    logger.emit_header(
        case=case,
        mesh=mesh,
        materials=materials,
        mode=mode,
        potential_solver=potential_solver,
        target_mean_velocity=target_mean_velocity,
        reference_mean_velocity=reference_mean_velocity,
        restart=restart,
    )


def _initial_solver_state(
    *,
    case: CaseSpec,
    mesh: StructuredMesh,
    fluid_mask: jnp.ndarray,
    interpolate_direct_fluid_walls: bool,
    initial_state: MHDState | None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, float]:
    if initial_state is None:
        initial_u = jnp.where(fluid_mask, case.initial_velocity, 0.0)
        initial_u = _enforce_velocity_bc(
            initial_u,
            mesh,
            fluid_mask,
            interpolate_direct_fluid_walls=interpolate_direct_fluid_walls,
        )
        zeros = jnp.zeros_like(initial_u)
        return initial_u, zeros, zeros, zeros, zeros, 0.0
    initial_u = _enforce_velocity_bc(
        jnp.asarray(initial_state.u),
        mesh,
        fluid_mask,
        interpolate_direct_fluid_walls=interpolate_direct_fluid_walls,
    )
    return (
        initial_u,
        jnp.asarray(initial_state.phi),
        jnp.asarray(initial_state.jy),
        jnp.asarray(initial_state.jz),
        jnp.asarray(initial_state.lorentz_x),
        float(initial_state.time),
    )


def _concat_history(
    previous: jnp.ndarray | None,
    current: jnp.ndarray,
    *,
    append: bool,
) -> jnp.ndarray:
    if not append or previous is None or previous.size == 0:
        return current
    return jnp.concatenate((previous, current))


def _pressure_proxy_reference_current(diagnostics: Diagnostics | None) -> float | None:
    if diagnostics is None:
        return None
    if diagnostics.face_current_max_history.size:
        return float(diagnostics.face_current_max_history[0])
    if diagnostics.current_max_history.size:
        return float(diagnostics.current_max_history[0])
    return None


def _scaled_pressure_proxy_value(
    pressure_proxy: float,
    current_max: float,
    face_current_max: float,
    reference_current: float | None,
) -> tuple[float, float]:
    current_source = float(face_current_max) if abs(float(face_current_max)) > 0.0 else float(current_max)
    if reference_current is None or abs(reference_current) < 1e-20:
        reference_current = current_source if abs(current_source) >= 1e-20 else 1.0
    scaled = float(pressure_proxy) * current_source / reference_current
    return scaled, reference_current


def _emit_solver_step(
    logger,
    *,
    step_index: int,
    step_time: float,
    dt: float,
    u_max_value: float,
    mean_velocity: float,
    max_current: float,
    face_current_max: float,
    emf_max: float,
    max_lorentz: float,
    face_lorentz_max: float,
    residual_value: float,
    potential_residual: float,
    potential_iteration_count: float,
    linear_residual: float,
    linear_iteration_count: float,
    applied_forcing: float,
    pressure_proxy: float,
    current_scaled_pressure_proxy: float,
    raw_update_max: float,
    limiter_scale: float,
    limited_fraction: float,
    courant_like: float,
    ohmic: float,
    volumetric_flow_rate: float,
    mean_current_magnitude: float,
    lorentz_power: float,
    div_current_max: float,
    charge_balance_residual: float,
    gauge_residual: float,
    interface_current_residual: float,
    potential_initial_residual: float = 0.0,
    linear_initial_residual: float = 0.0,
):
    if logger is None:
        return
    logger.emit_step(
        SolverStepRecord(
            step_index=step_index,
            time=step_time,
            dt=dt,
            u_max=u_max_value,
            mean_velocity=mean_velocity,
            current_max=max_current,
            face_current_max=face_current_max,
            emf_max=emf_max,
            lorentz_max=max_lorentz,
            face_lorentz_max=face_lorentz_max,
            residual=residual_value,
            potential_residual=potential_residual,
            potential_iterations=potential_iteration_count,
            linear_residual=linear_residual,
            linear_iterations=linear_iteration_count,
            potential_initial_residual=potential_initial_residual,
            linear_initial_residual=linear_initial_residual,
            applied_forcing=applied_forcing,
            pressure_proxy=pressure_proxy,
            current_scaled_pressure_proxy=current_scaled_pressure_proxy,
            raw_update_max=raw_update_max,
            limiter_scale=limiter_scale,
            limited_fraction=limited_fraction,
            courant_like=courant_like,
            ohmic_power=ohmic,
            volumetric_flow_rate=volumetric_flow_rate,
            mean_current_magnitude=mean_current_magnitude,
            lorentz_power=lorentz_power,
            div_current_max=div_current_max,
            charge_balance_residual=charge_balance_residual,
            gauge_residual=gauge_residual,
            interface_current_residual=interface_current_residual,
        )
    )


def _fully_developed_case_step(
    *,
    case: CaseSpec,
    mesh: StructuredMesh,
    materials,
    u_previous: jnp.ndarray,
    step_time: float,
    potential_solver: str,
    target_mean_velocity: float | None,
    linear_solver: str,
    preconditioner: str,
    coupling_iterations: int,
    coupling_tolerance: float,
    phi_previous: jnp.ndarray | None = None,
) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
]:
    _, by, bz = magnetic_field_components(case.magnetic_field, mesh, time=step_time)
    forcing = _explicit_forcing(case.forcing, by.dtype)
    fluid_mask = materials.fluid_mask
    active_mask = fluid_mask
    u_iter = u_previous
    phi_iter = jnp.zeros_like(u_previous) if phi_previous is None else phi_previous
    dt = case.time_stepper.dt
    fluid_weight = jnp.where(fluid_mask, _cell_metric(mesh).astype(u_previous.dtype), 0.0)
    fluid_total_weight = jnp.maximum(jnp.sum(fluid_weight), 1e-20)
    velocity_residual = jnp.asarray(jnp.inf, dtype=u_previous.dtype)
    potential_residual = jnp.asarray(jnp.inf, dtype=u_previous.dtype)
    potential_iteration_count = jnp.asarray(0, dtype=jnp.int32)
    potential_initial_residual = jnp.asarray(0.0, dtype=u_previous.dtype)
    linear_iteration_count = jnp.asarray(0, dtype=jnp.int32)
    linear_residual = jnp.asarray(jnp.inf, dtype=u_previous.dtype)
    linear_initial_residual = jnp.asarray(0.0, dtype=u_previous.dtype)
    applied_forcing = forcing
    steady_mode = case.solver.mode == "steady"
    potential_preconditioner = None
    potential_flexible = False
    if potential_solver == "cg_volume":
        coefficients = _potential_coefficients(mesh, materials.conductivity)
        scaled = _volume_scaled_potential_system(
            mesh, *coefficients, jnp.zeros_like(u_previous)
        )
        potential_preconditioner, potential_flexible = (
            _potential_preconditioner_for_materials(
                mesh,
                materials.conductivity,
                *scaled[:5],
                case.reference_phi_cell,
            )
        )
    acceleration = case.solver.coupling_acceleration
    if acceleration not in {"none", "aitken", "anderson"}:
        raise ValueError(f"Unsupported coupling acceleration {acceleration!r}")
    if (
        case.solver.coupling_min_relaxation <= 0.0
        or case.solver.coupling_max_relaxation < case.solver.coupling_min_relaxation
    ):
        raise ValueError("Coupling relaxation bounds must satisfy 0 < min <= max")
    if acceleration == "aitken" and _solvax_aitken_relaxation is None:
        raise ImportError(
            "coupling_acceleration='aitken' requires the optional accelerated dependencies; "
            "install LMX with `pip install lmx[accelerated]`"
        )
    if acceleration == "anderson" and _solvax_anderson_mixing is None:
        raise ImportError(
            "coupling_acceleration='anderson' requires the optional accelerated dependencies; "
            "install LMX with `pip install lmx[accelerated]`"
        )
    if case.solver.coupling_history_depth < 1:
        raise ValueError("Anderson coupling history depth must be positive")
    if case.solver.coupling_regularization < 0.0:
        raise ValueError("Anderson coupling regularization must be non-negative")
    if not 0.0 <= case.solver.coupling_damping <= 1.0:
        raise ValueError("Anderson coupling damping must lie in [0, 1]")
    previous_fixed_point_residual: jnp.ndarray | None = None
    coupling_relaxation = jnp.asarray(1.0, dtype=u_previous.dtype)
    anderson_iterates: list[jnp.ndarray] = []
    anderson_residuals: list[jnp.ndarray] = []
    strict_potential_solves = 0
    velocity_linear_tolerance = _nested_velocity_tolerance(
        coupling_tolerance, u_previous.dtype
    )

    if case.solver.time_scheme != "implicit_euler" and not steady_mode:
        raise NotImplementedError("fully_developed_inductionless currently supports implicit_euler only")

    for _ in range(max(1, coupling_iterations)):
        potential_iteration_tolerance = _coupling_potential_tolerance(
            case.time_stepper.potential_tolerance,
            velocity_residual=float(velocity_residual),
            coupling_tolerance=float(coupling_tolerance),
            flexible=potential_flexible,
        )
        potential_result = _solve_potential(
            mesh,
            materials.conductivity,
            fluid_mask,
            u_iter,
            by,
            bz,
            case.reference_phi_cell,
            case.time_stepper.potential_iterations,
            tolerance=potential_iteration_tolerance,
            relaxation=case.time_stepper.potential_relaxation,
            solver=potential_solver,
            initial_phi=phi_iter,
            potential_preconditioner=potential_preconditioner,
            potential_flexible=potential_flexible,
            return_solver_residual=True,
        )
        requested_potential_tolerance = case.time_stepper.potential_tolerance
        if (
            requested_potential_tolerance is not None
            and potential_iteration_tolerance is not None
            and potential_iteration_tolerance <= requested_potential_tolerance
        ):
            strict_potential_solves += 1
        if len(potential_result) == 4:  # compatibility for injected legacy backends
            (
                phi,
                potential_residual,
                potential_iteration_count,
                potential_initial_residual,
            ) = potential_result
            potential_solver_residual = potential_residual
        else:
            (
                phi,
                potential_residual,
                potential_iteration_count,
                potential_initial_residual,
                potential_solver_residual,
            ) = potential_result
        phi = jnp.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0)
        phi_iter = phi
        magnetic_reaction = jnp.where(
            active_mask,
            materials.conductivity * (by**2 + bz**2) / materials.density,
            0.0,
        )
        reaction = magnetic_reaction
        if not steady_mode:
            reaction = reaction + jnp.where(active_mask, 1.0 / dt, 0.0)
        rhs_base, _ = _fully_developed_rhs(
            mesh=mesh,
            sigma=materials.conductivity,
            rho=materials.density,
            fluid_mask=fluid_mask,
            u=u_iter,
            phi=phi,
            by=by,
            bz=bz,
            forcing=jnp.asarray(0.0, dtype=u_previous.dtype),
            current_reconstruction=case.time_stepper.current_reconstruction,
        )
        rhs_base = rhs_base + magnetic_reaction * jnp.where(active_mask, u_iter, 0.0)
        if not steady_mode:
            rhs_base = rhs_base + jnp.where(active_mask, u_previous / dt, 0.0)
        if target_mean_velocity is None:
            rhs = rhs_base + jnp.where(active_mask, forcing / materials.density, 0.0)
            u_next, velocity_linear_residual, linear_iteration_count, linear_initial_residual = _solve_velocity_system(
                mesh=mesh,
                diffusivity=materials.viscosity,
                reaction=reaction,
                rhs=rhs,
                active_mask=active_mask,
                linear_solver=linear_solver,
                preconditioner=preconditioner,
                max_steps=max(case.time_stepper.max_steps, case.solver.coupling_iterations * 25),
                tolerance=velocity_linear_tolerance,
            )
            applied_forcing = forcing
        else:
            unit_rhs = jnp.where(active_mask, 1.0 / materials.density, 0.0)
            u_base, velocity_linear_residual, linear_iteration_count, linear_initial_residual = _solve_velocity_system(
                mesh=mesh,
                diffusivity=materials.viscosity,
                reaction=reaction,
                rhs=rhs_base,
                active_mask=active_mask,
                linear_solver=linear_solver,
                preconditioner=preconditioner,
                max_steps=max(case.time_stepper.max_steps, case.solver.coupling_iterations * 25),
                tolerance=velocity_linear_tolerance,
            )
            u_sensitivity, _, _, sensitivity_initial_residual = _solve_velocity_system(
                mesh=mesh,
                diffusivity=materials.viscosity,
                reaction=reaction,
                rhs=unit_rhs,
                active_mask=active_mask,
                linear_solver=linear_solver,
                preconditioner=preconditioner,
                max_steps=max(case.time_stepper.max_steps, case.solver.coupling_iterations * 25),
                tolerance=velocity_linear_tolerance,
            )
            linear_initial_residual = jnp.maximum(linear_initial_residual, sensitivity_initial_residual)
            mean_base = jnp.sum(fluid_weight * u_base) / fluid_total_weight
            mean_sensitivity = jnp.sum(fluid_weight * u_sensitivity) / fluid_total_weight
            applied_forcing = jnp.where(
                mean_sensitivity > 1e-20,
                (jnp.asarray(target_mean_velocity, dtype=u_previous.dtype) - mean_base) / mean_sensitivity,
                jnp.asarray(0.0, dtype=u_previous.dtype),
            )
            u_next = u_base + applied_forcing * u_sensitivity
        linear_residual = velocity_linear_residual
        u_next = _limited_velocity_update(
            u_iter,
            u_next,
            fluid_mask,
            max_delta=case.time_stepper.velocity_update_limit,
            limiter=case.time_stepper.velocity_update_limiter,
        )
        u_next = _enforce_velocity_bc(
            jnp.where(fluid_mask, u_next, 0.0),
            mesh,
            fluid_mask,
            interpolate_direct_fluid_walls=case.geometry.kind == "rect_duct",
        )
        u_next = _enforce_target_mean_velocity(u_next, mesh, fluid_mask, target_mean_velocity)
        fixed_point_residual = u_next - u_iter
        if acceleration == "aitken" and previous_fixed_point_residual is not None:
            coupling_relaxation = _solvax_aitken_relaxation(
                previous_fixed_point_residual,
                fixed_point_residual,
                coupling_relaxation,
                min_relaxation=case.solver.coupling_min_relaxation,
                max_relaxation=case.solver.coupling_max_relaxation,
            )
            u_next = u_iter + coupling_relaxation * fixed_point_residual
            u_next = _enforce_velocity_bc(
                jnp.where(fluid_mask, u_next, 0.0),
                mesh,
                fluid_mask,
                interpolate_direct_fluid_walls=case.geometry.kind == "rect_duct",
            )
            u_next = _enforce_target_mean_velocity(
                u_next, mesh, fluid_mask, target_mean_velocity
            )
        elif acceleration == "anderson":
            anderson_iterates.append(u_iter)
            anderson_residuals.append(fixed_point_residual)
            depth = case.solver.coupling_history_depth
            anderson_iterates = anderson_iterates[-depth:]
            anderson_residuals = anderson_residuals[-depth:]
            u_next = _solvax_anderson_mixing(
                jnp.stack(anderson_iterates),
                jnp.stack(anderson_residuals),
                regularization=case.solver.coupling_regularization,
                damping=case.solver.coupling_damping,
            )
            u_next = _enforce_velocity_bc(
                jnp.where(fluid_mask, u_next, 0.0),
                mesh,
                fluid_mask,
                interpolate_direct_fluid_walls=case.geometry.kind == "rect_duct",
            )
            u_next = _enforce_target_mean_velocity(
                u_next, mesh, fluid_mask, target_mean_velocity
            )
        # Convergence is defined by the unrelaxed fixed-point residual at the
        # current iterate.  If it passes, retain that certified iterate rather
        # than returning the subsequently extrapolated Anderson/Aitken point,
        # whose residual has not been evaluated.  Acceleration is used only to
        # choose the next iterate when another map evaluation is required.
        velocity_residual = jnp.max(jnp.abs(fixed_point_residual))
        velocity_converged = float(velocity_residual) <= float(coupling_tolerance)
        auxiliary_converged = (
            strict_potential_solves >= _MIN_STRICT_POTENTIAL_COUPLING_SOLVES
            and float(potential_residual) <= _POTENTIAL_COUPLING_NORMALIZED_GATE
            and float(linear_residual)
            <= max(float(coupling_tolerance), _LINEAR_RESIDUAL_FLOOR)
        )
        if velocity_converged and auxiliary_converged:
            break
        if velocity_converged:
            # Re-evaluate the auxiliary solves at the certified velocity until
            # their own gates pass; do not perturb it merely to accumulate the
            # required strict-solve evidence.
            previous_fixed_point_residual = None
            continue
        previous_fixed_point_residual = fixed_point_residual
        u_iter = u_next

    phi, potential_residual, potential_iteration_count, potential_initial_residual = _solve_potential(
        mesh,
        materials.conductivity,
        fluid_mask,
        u_iter,
        by,
        bz,
        case.reference_phi_cell,
        case.time_stepper.potential_iterations,
        tolerance=case.time_stepper.potential_tolerance,
        relaxation=case.time_stepper.potential_relaxation,
        solver=potential_solver,
        initial_phi=phi_iter,
        potential_preconditioner=potential_preconditioner,
        potential_flexible=potential_flexible,
    )
    jy, jz, lorentz = _compute_current_and_lorentz(
        mesh,
        materials.conductivity,
        fluid_mask,
        u_iter,
        phi,
        by,
        bz,
        reconstruction=case.time_stepper.current_reconstruction,
    )
    face_current_max, emf_max, face_lorentz_max = _face_current_emf_and_lorentz_max(
        mesh,
        materials.conductivity,
        fluid_mask,
        u_iter,
        phi,
        by,
        bz,
    )
    mean_velocity = jnp.sum(fluid_weight * u_iter) / fluid_total_weight
    return (
        u_iter,
        phi,
        jy,
        jz,
        lorentz,
        velocity_residual,
        potential_residual,
        potential_iteration_count,
        linear_residual,
        linear_iteration_count,
        face_current_max,
        emf_max,
        face_lorentz_max,
        mean_velocity,
        applied_forcing,
        potential_initial_residual,
        linear_initial_residual,
    )


def _solve_fully_developed(
    case: CaseSpec,
    logger=None,
    *,
    mesh: StructuredMesh | None = None,
    initial_state: MHDState | None = None,
    initial_diagnostics: Diagnostics | None = None,
    append_diagnostics: bool = False,
    restart_info: RestartLogInfo | None = None,
) -> Solution:
    mesh = _build_mesh(case) if mesh is None else mesh
    materials = build_material_fields(case, mesh)
    target_mean_velocity = _target_mean_velocity(case)
    reference_mean_velocity = _reference_mean_velocity(case)
    potential_solver = _resolve_potential_solver(case.time_stepper.potential_solver, materials.fluid_mask)
    if (
        case.time_stepper.potential_solver == "auto"
        and potential_solver == "cg"
        and not _has_uniform_spacing(mesh)
    ):
        potential_solver = "cg_volume"
    linear_solver = (
        "solvax_pcg"
        if case.solver.linear_solver == "auto"
        else case.solver.linear_solver
    )
    if case.geometry.kind not in {"rect_duct", "layered_duct"}:
        raise NotImplementedError(f"Solver {case.solver.kind!r} does not yet support geometry {case.geometry.kind!r}")
    interpolate_direct_fluid_walls = case.geometry.kind == "rect_duct"
    initial_u, initial_phi, initial_jy, initial_jz, initial_lorentz, start_time = _initial_solver_state(
        case=case,
        mesh=mesh,
        fluid_mask=materials.fluid_mask,
        interpolate_direct_fluid_walls=interpolate_direct_fluid_walls,
        initial_state=initial_state,
    )
    dt = case.time_stepper.dt
    steady_mode = case.solver.mode == "steady"
    steps = _bounded_time_step_count(
        start_time=start_time,
        dt=dt,
        t_final=case.time_stepper.t_final,
        max_steps=case.time_stepper.max_steps,
    )
    if steady_mode:
        step_coupling_iterations = case.solver.coupling_iterations
        step_coupling_tolerance = float(case.time_stepper.steady_tolerance)
    else:
        step_coupling_iterations = case.solver.coupling_iterations
        step_coupling_tolerance = case.solver.coupling_tolerance
    _emit_solver_header(
        logger,
        case=case,
        mesh=mesh,
        materials=materials,
        mode=case.solver.mode,
        potential_solver=f"{potential_solver} / {linear_solver}",
        target_mean_velocity=target_mean_velocity,
        reference_mean_velocity=reference_mean_velocity,
        restart=restart_info,
    )

    u = initial_u
    phi = initial_phi
    jy = initial_jy
    jz = initial_jz
    lorentz = initial_lorentz
    time_history: list[float] = []
    u_max_history: list[float] = []
    mean_velocity_history: list[float] = []
    applied_forcing_history: list[float] = []
    pressure_proxy_history: list[float] = []
    current_scaled_pressure_proxy_history: list[float] = []
    residual_history: list[float] = []
    courant_history: list[float] = []
    ohmic_history: list[float] = []
    current_max_history: list[float] = []
    face_current_max_history: list[float] = []
    emf_max_history: list[float] = []
    lorentz_max_history: list[float] = []
    face_lorentz_max_history: list[float] = []
    potential_history: list[float] = []
    potential_iteration_history: list[float] = []
    linear_residual_history: list[float] = []
    linear_iteration_history: list[float] = []
    volumetric_flow_rate_history: list[float] = []
    mean_current_magnitude_history: list[float] = []
    lorentz_power_history: list[float] = []
    div_current_max_history: list[float] = []
    charge_balance_residual_history: list[float] = []
    gauge_residual_history: list[float] = []
    interface_current_residual_history: list[float] = []
    raw_update_max_history: list[float] = []
    limiter_scale_history: list[float] = []
    limited_fraction_history: list[float] = []
    pressure_proxy_reference_current = _pressure_proxy_reference_current(initial_diagnostics if append_diagnostics else None)
    residual_value = float(initial_state.residual if initial_state is not None else 0.0)
    step_count = 0

    for step_index in range(steps):
        step_time = float(start_time + (step_index + 1) * dt)
        u_before_step = u
        (
            u,
            phi,
            jy,
            jz,
            lorentz,
            residual,
            potential_residual,
            potential_iteration_count,
            linear_residual,
            _linear_iteration_count,
            face_current_max,
            emf_max,
            face_lorentz_max,
            mean_velocity,
            applied_forcing,
            potential_initial_residual,
            linear_initial_residual,
        ) = _fully_developed_case_step(
            case=case,
            mesh=mesh,
            materials=materials,
            u_previous=u,
            step_time=step_time,
            potential_solver=potential_solver,
            target_mean_velocity=target_mean_velocity,
            linear_solver=linear_solver,
            preconditioner=case.solver.preconditioner,
            coupling_iterations=step_coupling_iterations,
            coupling_tolerance=step_coupling_tolerance,
            phi_previous=phi,
        )
        outer_update = float(jnp.max(jnp.abs(u - u_before_step)))
        residual_value = max(float(residual), outer_update)
        u_max_value = float(jnp.max(jnp.abs(u)))
        courant_like = float(u_max_value * dt / jnp.min(mesh.dy))
        ohmic = float(jnp.mean(jy**2 + jz**2))
        max_current = float(jnp.max(jnp.sqrt(jy**2 + jz**2)))
        max_lorentz = float(jnp.max(jnp.abs(lorentz)))
        _, by_step, bz_step = magnetic_field_components(case.magnetic_field, mesh, time=step_time)
        (
            volumetric_flow_rate,
            mean_current_magnitude,
            lorentz_power,
            div_current_max,
            charge_balance_residual,
            gauge_residual,
            interface_current_residual,
        ) = _integral_diagnostics(
            mesh=mesh,
            sigma=materials.conductivity,
            fluid_mask=materials.fluid_mask,
            u=u,
            phi=phi,
            jy=jy,
            jz=jz,
            lorentz=lorentz,
            by=by_step,
            bz=bz_step,
            anchor=case.reference_phi_cell,
        )
        time_history.append(step_time)
        u_max_history.append(u_max_value)
        mean_velocity_history.append(float(mean_velocity))
        applied_forcing_history.append(float(applied_forcing))
        pressure_proxy_history.append(float(applied_forcing))
        residual_history.append(residual_value)
        courant_history.append(courant_like)
        ohmic_history.append(ohmic)
        current_max_history.append(max_current)
        face_current_max_history.append(float(face_current_max))
        emf_max_history.append(float(emf_max))
        lorentz_max_history.append(max_lorentz)
        face_lorentz_max_history.append(float(face_lorentz_max))
        potential_history.append(float(potential_residual))
        potential_iteration_history.append(float(potential_iteration_count))
        linear_residual_history.append(float(linear_residual))
        linear_iteration_history.append(float(_linear_iteration_count))
        volumetric_flow_rate_history.append(float(volumetric_flow_rate))
        mean_current_magnitude_history.append(float(mean_current_magnitude))
        lorentz_power_history.append(float(lorentz_power))
        div_current_max_history.append(float(div_current_max))
        charge_balance_residual_history.append(float(charge_balance_residual))
        gauge_residual_history.append(float(gauge_residual))
        interface_current_residual_history.append(float(interface_current_residual))
        raw_update_max_history.append(residual_value)
        limiter_scale_history.append(1.0)
        limited_fraction_history.append(0.0)
        current_scaled_pressure_proxy, pressure_proxy_reference_current = _scaled_pressure_proxy_value(
            float(applied_forcing),
            max_current,
            float(face_current_max),
            pressure_proxy_reference_current,
        )
        current_scaled_pressure_proxy_history.append(float(current_scaled_pressure_proxy))
        _emit_solver_step(
            logger,
            step_index=step_index + 1,
            step_time=step_time,
            dt=dt,
            u_max_value=u_max_value,
            mean_velocity=float(mean_velocity),
            max_current=max_current,
            face_current_max=float(face_current_max),
            emf_max=float(emf_max),
            max_lorentz=max_lorentz,
            face_lorentz_max=float(face_lorentz_max),
            residual_value=residual_value,
            potential_residual=float(potential_residual),
            potential_iteration_count=float(potential_iteration_count),
            linear_residual=float(linear_residual),
            linear_iteration_count=float(_linear_iteration_count),
            applied_forcing=float(applied_forcing),
            pressure_proxy=float(applied_forcing),
            current_scaled_pressure_proxy=float(current_scaled_pressure_proxy),
            raw_update_max=residual_value,
            limiter_scale=1.0,
            limited_fraction=0.0,
            courant_like=courant_like,
            ohmic=ohmic,
            volumetric_flow_rate=float(volumetric_flow_rate),
            mean_current_magnitude=float(mean_current_magnitude),
            lorentz_power=float(lorentz_power),
            div_current_max=float(div_current_max),
            charge_balance_residual=float(charge_balance_residual),
            gauge_residual=float(gauge_residual),
            interface_current_residual=float(interface_current_residual),
            potential_initial_residual=float(potential_initial_residual),
            linear_initial_residual=float(linear_initial_residual),
        )
        step_count = step_index + 1
        potential_gate = case.time_stepper.steady_potential_tolerance
        if potential_gate is None:
            potential_gate = case.time_stepper.potential_tolerance
        if potential_gate is None:
            potential_gate = case.time_stepper.steady_tolerance
        linear_gate = max(
            float(case.time_stepper.steady_tolerance), _LINEAR_RESIDUAL_FLOOR
        )
        if (
            steady_mode
            and residual_value <= float(case.time_stepper.steady_tolerance)
            and float(linear_residual) <= linear_gate
            and float(potential_residual) <= float(potential_gate)
        ):
            break

    state = MHDState(
        u=u,
        phi=phi,
        jy=jy,
        jz=jz,
        lorentz_x=lorentz,
        time=float(start_time + step_count * dt),
        residual=residual_value,
    )
    diagnostics = Diagnostics(
        time_history=_concat_history(initial_diagnostics.time_history if initial_diagnostics is not None else None, jnp.asarray(time_history, dtype=float), append=append_diagnostics),
        u_max_history=_concat_history(initial_diagnostics.u_max_history if initial_diagnostics is not None else None, jnp.asarray(u_max_history, dtype=float), append=append_diagnostics),
        mean_velocity_history=_concat_history(initial_diagnostics.mean_velocity_history if initial_diagnostics is not None else None, jnp.asarray(mean_velocity_history, dtype=float), append=append_diagnostics),
        applied_forcing_history=_concat_history(initial_diagnostics.applied_forcing_history if initial_diagnostics is not None else None, jnp.asarray(applied_forcing_history, dtype=float), append=append_diagnostics),
        pressure_proxy_history=_concat_history(initial_diagnostics.pressure_proxy_history if initial_diagnostics is not None else None, jnp.asarray(pressure_proxy_history, dtype=float), append=append_diagnostics),
        current_scaled_pressure_proxy_history=_concat_history(initial_diagnostics.current_scaled_pressure_proxy_history if initial_diagnostics is not None else None, jnp.asarray(current_scaled_pressure_proxy_history, dtype=float), append=append_diagnostics),
        raw_update_max_history=_concat_history(initial_diagnostics.raw_update_max_history if initial_diagnostics is not None else None, jnp.asarray(raw_update_max_history, dtype=float), append=append_diagnostics),
        limiter_scale_history=_concat_history(initial_diagnostics.limiter_scale_history if initial_diagnostics is not None else None, jnp.asarray(limiter_scale_history, dtype=float), append=append_diagnostics),
        limited_fraction_history=_concat_history(initial_diagnostics.limited_fraction_history if initial_diagnostics is not None else None, jnp.asarray(limited_fraction_history, dtype=float), append=append_diagnostics),
        residual_history=_concat_history(initial_diagnostics.residual_history if initial_diagnostics is not None else None, jnp.asarray(residual_history, dtype=float), append=append_diagnostics),
        courant_like=_concat_history(initial_diagnostics.courant_like if initial_diagnostics is not None else None, jnp.asarray(courant_history, dtype=float), append=append_diagnostics),
        ohmic_power=_concat_history(initial_diagnostics.ohmic_power if initial_diagnostics is not None else None, jnp.asarray(ohmic_history, dtype=float), append=append_diagnostics),
        current_max_history=_concat_history(initial_diagnostics.current_max_history if initial_diagnostics is not None else None, jnp.asarray(current_max_history, dtype=float), append=append_diagnostics),
        face_current_max_history=_concat_history(initial_diagnostics.face_current_max_history if initial_diagnostics is not None else None, jnp.asarray(face_current_max_history, dtype=float), append=append_diagnostics),
        emf_max_history=_concat_history(initial_diagnostics.emf_max_history if initial_diagnostics is not None else None, jnp.asarray(emf_max_history, dtype=float), append=append_diagnostics),
        lorentz_max_history=_concat_history(initial_diagnostics.lorentz_max_history if initial_diagnostics is not None else None, jnp.asarray(lorentz_max_history, dtype=float), append=append_diagnostics),
        face_lorentz_max_history=_concat_history(initial_diagnostics.face_lorentz_max_history if initial_diagnostics is not None else None, jnp.asarray(face_lorentz_max_history, dtype=float), append=append_diagnostics),
        potential_residual_history=_concat_history(initial_diagnostics.potential_residual_history if initial_diagnostics is not None else None, jnp.asarray(potential_history, dtype=float), append=append_diagnostics),
        potential_iterations_history=_concat_history(initial_diagnostics.potential_iterations_history if initial_diagnostics is not None else None, jnp.asarray(potential_iteration_history, dtype=float), append=append_diagnostics),
        linear_residual_history=_concat_history(initial_diagnostics.linear_residual_history if initial_diagnostics is not None else None, jnp.asarray(linear_residual_history, dtype=float), append=append_diagnostics),
        linear_iterations_history=_concat_history(initial_diagnostics.linear_iterations_history if initial_diagnostics is not None else None, jnp.asarray(linear_iteration_history, dtype=float), append=append_diagnostics),
        volumetric_flow_rate_history=_concat_history(initial_diagnostics.volumetric_flow_rate_history if initial_diagnostics is not None else None, jnp.asarray(volumetric_flow_rate_history, dtype=float), append=append_diagnostics),
        mean_current_magnitude_history=_concat_history(initial_diagnostics.mean_current_magnitude_history if initial_diagnostics is not None else None, jnp.asarray(mean_current_magnitude_history, dtype=float), append=append_diagnostics),
        lorentz_power_history=_concat_history(initial_diagnostics.lorentz_power_history if initial_diagnostics is not None else None, jnp.asarray(lorentz_power_history, dtype=float), append=append_diagnostics),
        div_current_max_history=_concat_history(initial_diagnostics.div_current_max_history if initial_diagnostics is not None else None, jnp.asarray(div_current_max_history, dtype=float), append=append_diagnostics),
        charge_balance_residual_history=_concat_history(initial_diagnostics.charge_balance_residual_history if initial_diagnostics is not None else None, jnp.asarray(charge_balance_residual_history, dtype=float), append=append_diagnostics),
        gauge_residual_history=_concat_history(initial_diagnostics.gauge_residual_history if initial_diagnostics is not None else None, jnp.asarray(gauge_residual_history, dtype=float), append=append_diagnostics),
        interface_current_residual_history=_concat_history(initial_diagnostics.interface_current_residual_history if initial_diagnostics is not None else None, jnp.asarray(interface_current_residual_history, dtype=float), append=append_diagnostics),
    )
    solution = Solution(mesh=mesh, state=state, diagnostics=diagnostics, case_name=case.name)
    if logger is not None:
        logger.emit_footer(solution)
    return solution


def solve_transient(
    case: CaseSpec,
    logger=None,
    *,
    mesh: StructuredMesh | None = None,
    initial_state: MHDState | None = None,
    initial_diagnostics: Diagnostics | None = None,
    append_diagnostics: bool = False,
    restart_info: RestartLogInfo | None = None,
) -> Solution:
    """Advance a supported case in transient mode, optionally from a restart."""

    solver_kind = getattr(getattr(case, "solver", None), "kind", "fully_developed_inductionless")
    if solver_kind == "fully_developed_inductionless":
        transient_case = case if case.solver.mode == "transient" else case.__class__(**{**case.__dict__, "solver": case.solver.__class__(**{**case.solver.__dict__, "mode": "transient"})})
        return _solve_fully_developed(
            transient_case,
            logger=logger,
            mesh=mesh,
            initial_state=initial_state,
            initial_diagnostics=initial_diagnostics,
            append_diagnostics=append_diagnostics,
            restart_info=restart_info,
        )
    raise NotImplementedError(f"Solver kind {solver_kind!r} is not implemented for transient runs")


def solve_steady(
    case: CaseSpec,
    logger=None,
    *,
    mesh: StructuredMesh | None = None,
    initial_state: MHDState | None = None,
    initial_diagnostics: Diagnostics | None = None,
    append_diagnostics: bool = False,
    restart_info: RestartLogInfo | None = None,
) -> Solution:
    """Solve a supported case to its configured steady-state stopping gate."""

    solver_kind = getattr(getattr(case, "solver", None), "kind", "fully_developed_inductionless")
    if solver_kind == "fully_developed_inductionless":
        steady_case = case if case.solver.mode == "steady" else case.__class__(**{**case.__dict__, "solver": case.solver.__class__(**{**case.solver.__dict__, "mode": "steady"})})
        return _solve_fully_developed(
            steady_case,
            logger=logger,
            mesh=mesh,
            initial_state=initial_state,
            initial_diagnostics=initial_diagnostics,
            append_diagnostics=append_diagnostics,
            restart_info=restart_info,
        )
    raise NotImplementedError(f"Solver kind {solver_kind!r} is not implemented for steady runs")

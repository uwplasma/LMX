"""Fully developed steady and transient inductionless solvers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import cho_factor, cho_solve
from solvax import (
    additive_preconditioner as _solvax_additive_preconditioner,
)
from solvax import (
    fixed_point_iteration as _solvax_fixed_point_iteration,
)
from solvax import (
    galerkin_deflation as _solvax_galerkin_deflation,
)
from solvax import (
    gmres as _solvax_gmres,
)
from solvax import (
    jacobi as _solvax_jacobi,
)
from solvax import (
    linear_solve as _solvax_linear_solve,
)
from solvax import (
    pcg_linear_solve as _solvax_pcg_linear_solve,
)
from solvax import (
    tridiagonal_solve as _solvax_tridiagonal_solve,
)

from .mesh import (
    StructuredMesh,
    apply_five_point_operator,
    apply_poisson_operator,
    five_point_residual_norm,
    generate_layered_duct_mesh,
    generate_rect_duct_mesh,
    generate_rect_duct_mesh_from_faces,
    gradient_scalar,
    poisson_residual_norm,
)
from .physics import build_material_fields, magnetic_field_components
from .specs import (
    BoundaryCondition,
    CaseSpec,
    MHDState,
    RestartLogInfo,
    Solution,
    SolverStepRecord,
)

_POTENTIAL_ADDITIVE_LINE_MIN_CELLS = 110
_POTENTIAL_ADDITIVE_LINE_DIAGONAL_RATIO = 3.0e4
_POTENTIAL_COARSE_STRIDE = 8
_POTENTIAL_INEXACT_COUPLING_TOLERANCE = 1.0e-4
_POTENTIAL_COUPLING_NORMALIZED_GATE = 1.0e-5
_LINEAR_RESIDUAL_FLOOR = 1.0e-9
_MIN_STRICT_POTENTIAL_COUPLING_SOLVES = 3
_POTENTIAL_FGMRES_RELATIVE_TOLERANCE = 1.0e-12


@dataclass(frozen=True)
class _PotentialSystem:
    coefficients: tuple[jnp.ndarray, ...]
    volume_coefficients: tuple[jnp.ndarray, ...] | None
    face_conductance: tuple[jnp.ndarray, jnp.ndarray]
    residual_scale: jnp.ndarray
    residual_scale_min: float
    preconditioner: Callable[[jnp.ndarray], jnp.ndarray] | None
    flexible: bool


def solve_poisson_jacobi_state(
    diagonal: jnp.ndarray,
    west: jnp.ndarray,
    east: jnp.ndarray,
    south: jnp.ndarray,
    north: jnp.ndarray,
    rhs: jnp.ndarray,
    anchor: tuple[int, int],
    iterations: int,
    tolerance: float | None = None,
    relaxation: float = 1.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    phi0 = jnp.zeros_like(rhs)

    def jacobi_update(phi: jnp.ndarray) -> jnp.ndarray:
        west_phi = jnp.pad(phi[:-1, :], ((1, 0), (0, 0)))
        east_phi = jnp.pad(phi[1:, :], ((0, 1), (0, 0)))
        south_phi = jnp.pad(phi[:, :-1], ((0, 0), (1, 0)))
        north_phi = jnp.pad(phi[:, 1:], ((0, 0), (0, 1)))
        updated = (rhs + west * west_phi + east * east_phi + south * south_phi + north * north_phi) / diagonal
        return updated.at[anchor].set(0.0)

    def residual_norm(phi: jnp.ndarray) -> jnp.ndarray:
        return poisson_residual_norm(diagonal, west, east, south, north, rhs, phi, anchor)

    fixed_steps = tolerance is None or tolerance <= 0.0
    solution = _solvax_fixed_point_iteration(
        jacobi_update,
        phi0,
        residual_norm=residual_norm,
        relaxation=relaxation,
        rtol=0.0,
        atol=0.0 if fixed_steps else tolerance,
        max_steps=iterations,
        fixed_steps=fixed_steps,
    )
    return solution.x, solution.residual_norm, solution.iterations


@partial(
    jax.jit,
    static_argnames=("anchor", "iterations", "tolerance", "residual_scale_min", "preconditioner"),
)
def solve_poisson_cg_state(
    diagonal: jnp.ndarray,
    west: jnp.ndarray,
    east: jnp.ndarray,
    south: jnp.ndarray,
    north: jnp.ndarray,
    rhs: jnp.ndarray,
    anchor: tuple[int, int],
    iterations: int,
    tolerance: float | None = None,
    initial: jnp.ndarray | None = None,
    residual_scale: jnp.ndarray | None = None,
    residual_scale_min: float | None = None,
    preconditioner: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Solve a symmetric anchored Poisson system with SOLVAX implicit PCG."""

    phi0 = jnp.zeros_like(rhs) if initial is None else jnp.asarray(initial).at[anchor].set(0.0)
    if phi0.shape != rhs.shape:
        raise ValueError("Poisson CG initial guess must match the right-hand side shape")
    if residual_scale is not None:
        residual_scale = jnp.asarray(residual_scale)
        if residual_scale.shape != rhs.shape:
            raise ValueError("Poisson CG residual scale must match the right-hand side shape")
    tiny = jnp.asarray(jnp.finfo(rhs.dtype).tiny, dtype=rhs.dtype)
    inverse_diagonal = 1.0 / jnp.maximum(diagonal, tiny)

    def apply_preconditioner(residual: jnp.ndarray) -> jnp.ndarray:
        # Extend the gauge-subspace preconditioner with an identity anchor.
        projected = residual.at[anchor].set(0.0)
        solved = inverse_diagonal * projected if preconditioner is None else preconditioner(projected)
        return solved.at[anchor].set(residual[anchor])

    def matvec(field: jnp.ndarray) -> jnp.ndarray:
        return apply_poisson_operator(diagonal, west, east, south, north, field, anchor)

    requested = 0.0 if tolerance is None else tolerance
    scaled_stopping = residual_scale is not None and residual_scale_min is not None
    rtol = 0.0 if scaled_stopping else requested / (rhs.size**0.5)
    atol = requested * residual_scale_min if scaled_stopping else 0.0
    solution = _solvax_pcg_linear_solve(
        matvec,
        rhs.at[anchor].set(0.0),
        x0=phi0,
        precond=apply_preconditioner,
        rtol=rtol,
        atol=atol,
        max_steps=iterations,
    )
    phi = solution.x.at[anchor].set(0.0)
    if residual_scale is None:
        residual = poisson_residual_norm(diagonal, west, east, south, north, rhs, phi, anchor)
    else:
        physical_residual = rhs - apply_five_point_operator(diagonal, west, east, south, north, phi)
        residual = jnp.max(jnp.abs(physical_residual) / jnp.maximum(residual_scale, tiny))
    return phi, residual, solution.iterations


def _coupling_potential_tolerance(
    requested: float | None,
    *,
    velocity_residual: float,
    coupling_tolerance: float,
    flexible: bool,
) -> float | None:
    """Use an inexact potential solve only while the fixed point is far away."""
    if requested is None or flexible or velocity_residual <= 10.0 * coupling_tolerance:
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
    transposed = _potential_y_line_preconditioner(diagonal.T, south.T, north.T, (anchor[1], anchor[0]))

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
    y_line = _potential_y_line_preconditioner(diagonal, west, east, anchor)
    z_line = _potential_z_line_preconditioner(diagonal, south, north, anchor)
    return _solvax_additive_preconditioner((y_line, z_line))


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
    if coarse_stride < 2:
        return None
    mean_y = float(np.asarray(jnp.mean(west + east)))
    mean_z = float(np.asarray(jnp.mean(south + north)))
    if mean_y >= 4.0 * max(mean_z, np.finfo(float).tiny):
        line = _potential_y_line_preconditioner(diagonal, west, east, anchor)
    elif mean_z >= 4.0 * max(mean_y, np.finfo(float).tiny):
        line = _potential_z_line_preconditioner(diagonal, south, north, anchor)
    else:
        line = _potential_additive_line_preconditioner(diagonal, west, east, south, north, anchor)
    fine_shape = diagonal.shape
    coarse_shape = tuple((size - 1 + coarse_stride - 1) // coarse_stride + 1 for size in fine_shape)

    def prolong(coarse: jnp.ndarray) -> jnp.ndarray:
        fine = jax.image.resize(coarse, fine_shape, method="linear")
        return fine.at[anchor].set(0.0)

    coarse_zero = jnp.zeros(coarse_shape, dtype=diagonal.dtype)

    def restrict(fine: jnp.ndarray) -> jnp.ndarray:
        return jax.linear_transpose(prolong, coarse_zero)(fine)[0]

    def fine_matvec(field: jnp.ndarray) -> jnp.ndarray:
        return apply_poisson_operator(diagonal, west, east, south, north, field, anchor)

    def coarse_matvec(field: jnp.ndarray) -> jnp.ndarray:
        return restrict(fine_matvec(prolong(field)))

    coarse_size = coarse_shape[0] * coarse_shape[1]
    basis = jnp.eye(coarse_size, dtype=diagonal.dtype)
    coarse_matrix = jax.vmap(lambda column: coarse_matvec(column.reshape(coarse_shape)).reshape(-1))(basis).T
    coarse_matrix = 0.5 * (coarse_matrix + coarse_matrix.T)
    coarse_factors = cho_factor(coarse_matrix, lower=True)

    def coarse_solve(rhs: jnp.ndarray) -> jnp.ndarray:
        return cho_solve(coarse_factors, rhs.reshape(-1)).reshape(coarse_shape)

    return _solvax_galerkin_deflation(
        fine_matvec,
        line,
        prolong,
        coarse_solve,
        coarse_zero,
    )


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
    """Solve the anchored potential system with implicit SOLVAX FGMRES."""
    shape = rhs.shape

    def matvec_flat(vector: jnp.ndarray) -> jnp.ndarray:
        field = vector.reshape(shape)
        return apply_poisson_operator(diagonal, west, east, south, north, field, anchor).reshape(-1)

    def precondition_flat(vector: jnp.ndarray) -> jnp.ndarray:
        return preconditioner(vector.reshape(shape)).reshape(-1)

    zero = jnp.zeros(rhs.size, dtype=rhs.dtype)

    def transpose_precondition_flat(vector: jnp.ndarray) -> jnp.ndarray:
        return jax.linear_transpose(precondition_flat, zero)(vector)[0]

    def solve_with(precondition):
        def solve(operator, source):
            solution = _solvax_gmres(
                operator,
                source,
                precond=precondition,
                restart=20,
                rtol=_POTENTIAL_FGMRES_RELATIVE_TOLERANCE,
                max_restarts=10,
            )
            return solution.x, solution.iterations

        return solve

    solve = solve_with(precondition_flat)
    transpose_solve = solve_with(transpose_precondition_flat)
    source = rhs.at[anchor].set(0.0).reshape(-1)
    start = initial.at[anchor].set(0.0).reshape(-1)
    correction, iterations = _solvax_linear_solve(
        matvec_flat,
        source - matvec_flat(start),
        solve,
        transpose_solver=transpose_solve,
        has_aux=True,
    )
    phi = (start + correction).reshape(shape).at[anchor].set(0.0)
    physical_residual = rhs - apply_five_point_operator(diagonal, west, east, south, north, phi)
    residual = jnp.max(jnp.abs(physical_residual) / jnp.maximum(residual_scale, 1.0e-30))
    return phi, residual, iterations


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
    dy = mesh.dy.astype(west.dtype)
    dz = mesh.dz.astype(west.dtype)
    west_y = jnp.mean(west / dz[None, :], axis=1)
    east_y = jnp.mean(east / dz[None, :], axis=1)
    south_z = jnp.mean(south / dy[:, None], axis=0)
    north_z = jnp.mean(north / dy[:, None], axis=0)
    operator_z = jnp.diag(south_z + north_z) + jnp.diag(-north_z[:-1], 1) + jnp.diag(-south_z[1:], -1)
    inv_sqrt_dz = jax.lax.rsqrt(dz)
    eigenvectors_z, eigenvalues_z, _ = jnp.linalg.svd(
        inv_sqrt_dz[:, None] * operator_z * inv_sqrt_dz[None, :]
    )
    eigenvalues_z = eigenvalues_z[::-1]
    eigenvectors_z = eigenvectors_z[:, ::-1]
    modes_z = inv_sqrt_dz[:, None] * eigenvectors_z
    line_diagonal = west_y[:, None] + east_y[:, None] + dy[:, None] * eigenvalues_z[None, :]
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
        solved_modes = _solvax_tridiagonal_solve(line_lower, line_diagonal, line_upper, transformed)
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
        return _potential_deflated_line_preconditioner(diagonal, west, east, south, north, anchor)
    if (
        min(diagonal.shape) < _POTENTIAL_ADDITIVE_LINE_MIN_CELLS
        and diagonal_ratio < _POTENTIAL_ADDITIVE_LINE_DIAGONAL_RATIO
    ):
        return None
    return _potential_additive_line_preconditioner(diagonal, west, east, south, north, anchor)


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
        return preconditioner, True

    preconditioner = _select_potential_preconditioner(diagonal, west, east, south, north, anchor)
    if preconditioner is None:
        preconditioner = _potential_additive_line_preconditioner(diagonal, west, east, south, north, anchor)
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


def _interface_conductance(mesh: StructuredMesh, sigma: jnp.ndarray, *, axis: int) -> jnp.ndarray:
    widths = (mesh.dy, mesh.dz)[axis]
    field = jnp.moveaxis(sigma, axis, 0)
    left_distance = 0.5 * widths[:-1, None]
    right_distance = 0.5 * widths[1:, None]
    sigma_left = field[:-1]
    sigma_right = field[1:]
    connected = (sigma_left > 0.0) & (sigma_right > 0.0)
    safe_left = jnp.where(connected, sigma_left, 1.0)
    safe_right = jnp.where(connected, sigma_right, 1.0)
    resistance = left_distance / safe_left + right_distance / safe_right
    conductance = jnp.where(connected, 1.0 / jnp.maximum(resistance, 1e-30), 0.0)
    return jnp.moveaxis(conductance, 0, axis)


def _face_conductance(
    mesh: StructuredMesh, sigma: jnp.ndarray, *, axis: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    conductance = _interface_conductance(mesh, sigma, axis=axis)
    spacing = jnp.moveaxis((mesh.dy, mesh.dz)[axis][:, None], 0, axis)
    lower_pad, upper_pad = [[(0, 0), (0, 0)] for _ in range(2)]
    lower_pad[axis], upper_pad[axis] = (1, 0), (0, 1)
    return jnp.pad(conductance, lower_pad) / spacing, jnp.pad(conductance, upper_pad) / spacing


def _face_emf(
    mesh: StructuredMesh,
    sigma: jnp.ndarray,
    source: jnp.ndarray,
    *,
    axis: int,
    conductance: jnp.ndarray | None = None,
) -> jnp.ndarray:
    widths = (mesh.dy, mesh.dz)[axis]
    field = jnp.moveaxis(source, axis, 0)
    face_conductance = _interface_conductance(mesh, sigma, axis=axis) if conductance is None else conductance
    face_conductance = jnp.moveaxis(face_conductance, axis, 0)
    emf = face_conductance * (0.5 * widths[:-1, None] * field[:-1] + 0.5 * widths[1:, None] * field[1:])
    return jnp.moveaxis(emf, 0, axis)


def _potential_coefficients(mesh: StructuredMesh, sigma: jnp.ndarray) -> tuple[jnp.ndarray, ...]:
    west, east = _face_conductance(mesh, sigma, axis=0)
    south, north = _face_conductance(mesh, sigma, axis=1)
    diagonal = west + east + south + north
    diagonal = jnp.where(diagonal > 0.0, diagonal, 1.0)
    return diagonal, west, east, south, north


def _cell_metric(mesh: StructuredMesh) -> jnp.ndarray:
    return mesh.dy[:, None] * mesh.dz[None, :]


def _connected_interface_diffusivity(
    mesh: StructuredMesh, diffusivity: jnp.ndarray, active_mask: jnp.ndarray, *, axis: int
) -> jnp.ndarray:
    widths = (mesh.dy, mesh.dz)[axis]
    field = jnp.moveaxis(diffusivity, axis, 0)
    active = jnp.moveaxis(active_mask, axis, 0)
    connected = active[:-1] & active[1:]
    left_distance = 0.5 * widths[:-1, None]
    right_distance = 0.5 * widths[1:, None]
    diffusivity_left = jnp.maximum(field[:-1], 1e-12)
    diffusivity_right = jnp.maximum(field[1:], 1e-12)
    conductance = 1.0 / jnp.maximum(
        left_distance / diffusivity_left + right_distance / diffusivity_right, 1e-12
    )
    return jnp.moveaxis(jnp.where(connected, conductance, 0.0), 0, axis)


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

    interface_y = _connected_interface_diffusivity(mesh, diffusivity, active_mask, axis=0)
    interface_z = _connected_interface_diffusivity(mesh, diffusivity, active_mask, axis=1)
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


@partial(jax.jit, static_argnames=("max_steps", "tolerance", "preconditioner"))
def _solve_velocity_coefficients(
    coefficients: tuple[jnp.ndarray, ...],
    rhs: jnp.ndarray,
    *,
    max_steps: int,
    tolerance: float,
    preconditioner: str,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compose LMX coefficients and residual scaling with SOLVAX PCG."""

    def matvec(field: jnp.ndarray) -> jnp.ndarray:
        return apply_five_point_operator(*coefficients, field)

    tiny = jnp.asarray(jnp.finfo(rhs.dtype).tiny, dtype=rhs.dtype)
    if preconditioner == "jacobi":
        apply_preconditioner = _solvax_jacobi(jnp.maximum(coefficients[0], tiny))
    elif preconditioner == "none":
        apply_preconditioner = None
    else:
        raise ValueError(f"Unsupported preconditioner {preconditioner!r}")
    l2_tolerance = tolerance / (rhs.size**0.5)
    solution = _solvax_pcg_linear_solve(
        matvec,
        rhs,
        precond=apply_preconditioner,
        rtol=l2_tolerance,
        atol=0.0,
        max_steps=max_steps,
        transpose_rtol=l2_tolerance,
        transpose_atol=0.0,
        transpose_max_steps=max_steps,
    )
    residual = five_point_residual_norm(*coefficients, rhs, solution.x)
    return solution.x, residual, solution.iterations


def _solve_velocity_system(
    *,
    coefficients: tuple[jnp.ndarray, ...],
    cell_metric: jnp.ndarray,
    rhs: jnp.ndarray,
    active_mask: jnp.ndarray,
    preconditioner: str,
    max_steps: int,
    tolerance: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    rhs_masked = jnp.where(active_mask, rhs, 0.0)
    rhs_scaled = rhs_masked * cell_metric
    initial_residual = five_point_residual_norm(
        *coefficients,
        rhs_scaled,
        jnp.zeros_like(rhs_masked),
    )
    field, residual, iterations = _solve_velocity_coefficients(
        coefficients,
        rhs_scaled,
        max_steps=max_steps,
        tolerance=tolerance,
        preconditioner=preconditioner,
    )
    field = jnp.where(active_mask, field, 0.0)
    return (
        field,
        jnp.asarray(residual, dtype=rhs.dtype),
        jnp.asarray(iterations, dtype=jnp.int32),
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
) -> tuple[jnp.ndarray, jnp.ndarray]:
    _, _, lorentz_source = _compute_current_and_lorentz(
        mesh,
        sigma,
        fluid_mask,
        u,
        phi,
        by,
        bz,
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


def _prepare_potential_system(
    mesh: StructuredMesh,
    sigma: jnp.ndarray,
    anchor: tuple[int, int],
    solver: str,
) -> _PotentialSystem:
    """Assemble invariant potential coefficients and preconditioning once."""

    coefficients = _potential_coefficients(mesh, sigma)
    face_conductance = (
        _interface_conductance(mesh, sigma, axis=0),
        _interface_conductance(mesh, sigma, axis=1),
    )
    residual_scale = _cell_metric(mesh)
    residual_scale_min = float(np.min(np.asarray(residual_scale)))
    if solver != "cg_volume":
        return _PotentialSystem(
            coefficients,
            None,
            face_conductance,
            residual_scale,
            residual_scale_min,
            None,
            False,
        )
    volume_coefficients = _volume_scaled_potential_system(
        mesh,
        *coefficients,
        jnp.zeros_like(sigma),
    )[:5]
    preconditioner, flexible = _potential_preconditioner_for_materials(
        mesh,
        sigma,
        *volume_coefficients,
        anchor,
    )
    return _PotentialSystem(
        coefficients,
        volume_coefficients,
        face_conductance,
        residual_scale,
        residual_scale_min,
        preconditioner,
        flexible,
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
    system: _PotentialSystem | None = None,
    return_solver_residual: bool = False,
) -> tuple[jnp.ndarray, ...]:
    if system is None:
        system = _prepare_potential_system(mesh, sigma, anchor, solver)
    uxb_y = jnp.where(fluid_mask, -u * bz, 0.0)
    uxb_z = jnp.where(fluid_mask, u * by, 0.0)
    conv_y = _face_emf(mesh, sigma, uxb_y, axis=0, conductance=system.face_conductance[0])
    conv_z = _face_emf(mesh, sigma, uxb_z, axis=1, conductance=system.face_conductance[1])
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

    diagonal, west, east, south, north = system.coefficients
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
        if system.volume_coefficients is None:
            raise ValueError("Volume-scaled potential coefficients are required for cg_volume")
        diagonal_scaled, west_scaled, east_scaled, south_scaled, north_scaled = system.volume_coefficients
        rhs_scaled = jnp.where((west + east + south + north) > 0.0, rhs * system.residual_scale, 0.0)
        if system.flexible and system.preconditioner is not None:
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
                residual_scale=system.residual_scale,
                preconditioner=system.preconditioner,
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
                residual_scale=system.residual_scale,
                residual_scale_min=system.residual_scale_min,
                preconditioner=system.preconditioner,
            )
        residual = poisson_residual_norm(diagonal, west, east, south, north, rhs, phi, anchor)
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
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    jy, jz = _conductive_current_components(mesh, sigma, fluid_mask, u, phi, by, bz)
    lorentz_x = jy * bz - jz * by
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
    emf_y = _face_emf(mesh, sigma, uxb_y, axis=0)
    emf_z = _face_emf(mesh, sigma, uxb_z, axis=1)
    face_jy = _interface_conductance(mesh, sigma, axis=0) * (phi[:-1, :] - phi[1:, :]) + emf_y
    face_jz = _interface_conductance(mesh, sigma, axis=1) * (phi[:, :-1] - phi[:, 1:]) + emf_z
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
    div_current = (padded_face_jy[1:, :] - padded_face_jy[:-1, :]) / mesh.dy[:, None] + (
        padded_face_jz[:, 1:] - padded_face_jz[:, :-1]
    ) / mesh.dz[None, :]
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
    conductance_y = _interface_conductance(mesh, materials.conductivity, axis=0)
    conductance_z = _interface_conductance(mesh, materials.conductivity, axis=1)
    face_area_y = mesh.dz[None, :]
    face_area_z = mesh.dy[:, None]
    joule_dissipation = jnp.sum(
        face_area_y * jnp.where(conductance_y > 0.0, face_jy**2 / jnp.maximum(conductance_y, 1.0e-30), 0.0)
    ) + jnp.sum(
        face_area_z * jnp.where(conductance_z > 0.0, face_jz**2 / jnp.maximum(conductance_z, 1.0e-30), 0.0)
    )
    emf_field_y = jnp.where(conductance_y > 0.0, emf_y / jnp.maximum(conductance_y, 1.0e-30), 0.0)
    emf_field_z = jnp.where(conductance_z > 0.0, emf_z / jnp.maximum(conductance_z, 1.0e-30), 0.0)
    emf_power = jnp.sum(face_area_y * face_jy * emf_field_y) + jnp.sum(face_area_z * face_jz * emf_field_z)
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
    electrical_scale = jnp.maximum(jnp.maximum(jnp.abs(joule_dissipation), jnp.abs(lorentz_work)), 1.0e-30)
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
        "network_electrical_relative_error": float(jnp.abs(network_electrical_residual) / electrical_scale),
        "lorentz_transfer_residual": float(lorentz_transfer_residual),
        "lorentz_transfer_relative_error": float(jnp.abs(lorentz_transfer_residual) / electrical_scale),
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
        west_ratio = (mesh.y_centers[0] - mesh.y_faces[0]) / jnp.maximum(
            mesh.y_centers[1] - mesh.y_faces[0], 1e-12
        )
        east_ratio = (mesh.y_faces[-1] - mesh.y_centers[-1]) / jnp.maximum(
            mesh.y_faces[-1] - mesh.y_centers[-2], 1e-12
        )
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
        south_ratio = (mesh.z_centers[0] - mesh.z_faces[0]) / jnp.maximum(
            mesh.z_centers[1] - mesh.z_faces[0], 1e-12
        )
        north_ratio = (mesh.z_faces[-1] - mesh.z_centers[-1]) / jnp.maximum(
            mesh.z_faces[-1] - mesh.z_centers[-2], 1e-12
        )
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
) -> jnp.ndarray:
    delta = jnp.where(fluid_mask, trial - current, 0.0)
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


def _emit_solver_header(
    logger,
    *,
    case,
    mesh,
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


def _emit_solver_step(
    logger,
    *,
    step_index: int,
    step_time: float,
    u_max_value: float,
    mean_velocity: float,
    max_current: float,
    max_lorentz: float,
    residual_value: float,
    potential_residual: float,
    potential_iteration_count: float,
    linear_residual: float,
    linear_iteration_count: float,
    applied_forcing: float,
    courant_like: float,
    ohmic: float,
    volumetric_flow_rate: float,
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
            u_max=u_max_value,
            mean_velocity=mean_velocity,
            current_max=max_current,
            lorentz_max=max_lorentz,
            residual=residual_value,
            potential_residual=potential_residual,
            potential_iterations=potential_iteration_count,
            linear_residual=linear_residual,
            linear_iterations=linear_iteration_count,
            potential_initial_residual=potential_initial_residual,
            linear_initial_residual=linear_initial_residual,
            applied_forcing=applied_forcing,
            courant_like=courant_like,
            ohmic_power=ohmic,
            volumetric_flow_rate=volumetric_flow_rate,
            div_current_max=div_current_max,
            charge_balance_residual=charge_balance_residual,
            gauge_residual=gauge_residual,
            interface_current_residual=interface_current_residual,
        )
    )

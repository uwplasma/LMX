"""Fully developed duct cases solved on the staggered core.

A :class:`~lmx.specs.CaseSpec` for a rectangular duct becomes a
:class:`~lmx.core3d.ChannelProblem` with one periodic axial cell, solved by the
conjugate-gradient steady solve of :mod:`lmx.steady` and reported on the
case's cross-section. That is the route :func:`lmx.solve` and
:func:`lmx.solve_fully_developed_fields` take.

*Mesh.* ``ny`` and ``nz`` are the fluid cells along ``y`` and ``z``. The faces
follow :func:`lmx.core3d.duct_problem`: the walls normal to the field are
clustered to the Hartmann layer ``delta = sqrt(rho nu / sigma) / |B|``, the
others to the side layer ``sqrt(a delta)``, with ``a`` the half-width along the
field, six cells in each layer (``hartmann_layer_cells`` overrides) and the
gentlest stretching that spans the duct. A 2 x 2 duct with unit properties
gets exactly the faces of ``duct_problem(hartmann=Ha, cells=n)``.

*Walls.* An insulating wall is the homogeneous Neumann closure. A conducting
wall of a ``layered_duct`` is the thin wall of :mod:`lmx.poisson`: the sheet
conductance ``sigma_w t_w / sigma`` of its region and thickness (a length,
the wall conductance ratio times the half-width), with no cells of its own, so
the solution covers the fluid alone. Both walls of an axis must match.

*Drive.* ``forcing`` is the axial force density. With zero forcing and an
``inlet_flow_rate`` boundary, the flow rate is met by scaling the unit-drive
solution, which is exact because the problem is linear in the drive.

The pseudo-time controls of the case (time step, relaxation, potential and
coupling iterations, steady tolerance) do not enter; the steady state is one
preconditioned CG solve to a relative residual of 1e-9. Without
:func:`lmx.enable_x64` it runs in float32 to 1e-5, which the solve reaches at
Ha 20 on 32 cells and not at Ha 100 on 48, where it raises; a float32 case
with float64 enabled is solved in float64 and returned in float32.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

from .bc import NEUMANN, PERIODIC, BoundaryCondition
from .core3d import ChannelProblem, face_currents, zero_velocity
from .em import lorentz_force, wall_insulated
from .grid import Grid, uniform_faces, wall_resolving_faces
from .mesh import StructuredMesh, generate_rect_duct_mesh_from_faces
from .ops import divergence
from .specs import CaseSpec, Diagnostics, MHDState, Solution, require_finite
from .steady import solve_steady_state, steady_residual

__all__ = ["case_mesh", "channel_problem", "solve_fully_developed", "solve_fully_developed_fields"]

# Relative CG tolerance by the precision JAX computes in: float32 floors near 1e-6.
_TOLERANCE = {"float64": 1.0e-9, "float32": 1.0e-5}
_CELLS_IN_LAYER = 6
_SIDES = {"left": (1, 0), "right": (1, 1), "bottom": (2, 0), "top": (2, 1)}
_PAIRS = {"left_right": ("left", "right"), "top_bottom": ("bottom", "top")}
_IGNORED = {"no_slip", "inlet_velocity", "inlet_flow_rate", "outlet_pressure"}


def channel_problem(case: CaseSpec) -> ChannelProblem:
    """Return the staggered-core problem a fully developed case solves, at unit drive.

    Raise ``NotImplementedError`` for what the core does not represent: other
    geometries, a field that is not a constant in the cross-section, thick or
    mismatched conducting walls, several fluids, or an imposed current.
    """
    _check(case)
    fluid = _fluid(case)
    density, viscosity, conductivity = (
        float(fluid.density or 1.0),
        float(fluid.viscosity or 1.0),
        fluid.conductivity,
    )
    _, by, bz = (float(value) for value in case.magnetic_field.value)
    geometry = case.geometry
    halves = (0.5 * geometry.width, 0.5 * geometry.height)
    along = 0 if abs(by) >= abs(bz) else 1
    # Ha on the half-width along the field; the layers are a/Ha and a/sqrt(Ha) there.
    hartmann = halves[along] * float(np.hypot(by, bz)) * np.sqrt(conductivity / (density * viscosity))
    layers = [np.inf, np.inf]
    if hartmann > 0.0:
        layers = [halves[along] / hartmann] * 2
        layers[1 - along] = halves[along] / np.sqrt(hartmann)
    cells_in_layer = geometry.hartmann_layer_cells or _CELLS_IN_LAYER
    faces = [
        _faces(count, half, layer, cells_in_layer)
        for count, half, layer in zip((geometry.ny, geometry.nz), halves, layers, strict=True)
    ]
    insulating = BoundaryCondition(NEUMANN)
    return ChannelProblem(
        grid=Grid(uniform_faces(1, 0.0, 1.0), *faces),
        conditions=(BoundaryCondition(PERIODIC), insulating, insulating),
        density=density,
        viscosity=viscosity,
        conductivity=conductivity,
        magnetic_field=(0.0, by, bz),
        forcing=(1.0, 0.0, 0.0),
        dt=min(halves) ** 2 / viscosity,
        wall_conductance=_wall_conductance(case, conductivity),
    )


def case_mesh(case: CaseSpec) -> StructuredMesh:
    """Return the fluid cross-section the core solves ``case`` on, as a :class:`StructuredMesh`."""
    grid = channel_problem(case).grid
    return generate_rect_duct_mesh_from_faces(
        y_faces=jnp.asarray(grid.y_faces, dtype=case.dtype),
        z_faces=jnp.asarray(grid.z_faces, dtype=case.dtype),
        length=case.geometry.length,
    )


def solve_fully_developed_fields(
    case: CaseSpec,
    *,
    forcing: float | jax.Array | None = None,
    magnetic_field_scale: float | jax.Array = 1.0,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return the steady velocity, potential, currents and Lorentz force on :func:`case_mesh`.

    All five are cell-centred ``(ny, nz)`` arrays in the case's dtype: axial
    velocity, potential (zero volume mean), the ``y`` and ``z`` currents
    averaged from their faces, and the axial Lorentz force density.
    ``forcing`` and ``magnetic_field_scale`` are continuous design inputs,
    differentiable through the implicit solve of :func:`lmx.steady.solve_steady_state`;
    the case itself is static, so close over it outside :func:`jax.jit`. A
    solve that fails raises :class:`~lmx.specs.NumericalFailure` when called
    with concrete inputs and gives nonfinite fields under tracing.
    """
    problem = channel_problem(case)
    target = _target_flow_rate(case) if forcing is None else None
    drive = target if target is not None else (case.forcing if forcing is None else forcing)
    fields, _, _ = _compiled(problem, target is not None, case.dtype)(drive, magnetic_field_scale)
    if not any(isinstance(value, jax.core.Tracer) for value in (drive, magnetic_field_scale)):
        require_finite("fully developed solve", velocity=fields[0])
    return fields


def solve_fully_developed(case: CaseSpec, *, logger=None, start_time: float = 0.0) -> Solution:
    """Solve a fully developed case to its steady state on the core and report it as a :class:`Solution`.

    ``status`` is ``"converged"`` with ``residual`` the relative steady
    residual ``||R(u)|| / ||R(0)||``; a solve that fails raises. ``steps`` is
    zero: the solve is one CG call with no outer iterations. The diagnostics
    hold one record.
    """
    problem = channel_problem(case)
    mesh = case_mesh(case)
    target = _target_flow_rate(case)
    if logger is not None:
        mean = None if target is None else target / (case.geometry.width * case.geometry.height)
        logger.emit_header(
            case=case,
            mesh=mesh,
            mode="steady",
            potential_solver="staggered core / fast diagonalization / pcg",
            target_mean_velocity=mean,
            reference_mean_velocity=mean,
            restart=None,
        )
    run = _compiled(problem, target is not None, case.dtype)
    fields, drive, evidence = run(case.forcing if target is None else target, 1.0)
    u, phi, jy, jz, lorentz = fields
    require_finite("fully developed solve", velocity=u, potential=phi, residual=evidence["residual"])
    areas = jnp.asarray(mesh.dy, dtype=u.dtype)[:, None] * jnp.asarray(mesh.dz, dtype=u.dtype)[None, :]
    flow_rate = jnp.sum(areas * u)
    area = jnp.sum(areas)
    record = {
        "residual_history": evidence["residual"],
        "courant_like": 0.0,
        "ohmic_power": jnp.sum(areas * (jy**2 + jz**2)) / max(problem.conductivity, 1e-300),
        "u_max_history": jnp.max(jnp.abs(u)),
        "mean_velocity_history": flow_rate / area,
        "applied_forcing_history": drive,
        "current_max_history": jnp.max(jnp.hypot(jy, jz)),
        "lorentz_max_history": jnp.max(jnp.abs(lorentz)),
        "volumetric_flow_rate_history": flow_rate,
        "mean_current_magnitude_history": jnp.sum(areas * jnp.hypot(jy, jz)) / area,
        "lorentz_power_history": jnp.sum(areas * u * lorentz),
        "div_current_max_history": evidence["div_current_max"],
    }
    values = {name: float(value) for name, value in jax.device_get(record).items()}
    diagnostics = Diagnostics(
        time_history=jnp.asarray([start_time]),
        **{name: jnp.asarray([value]) for name, value in values.items()},
    )
    state = MHDState(u, phi, jy, jz, lorentz, float(start_time), values["residual_history"])
    solution = Solution(mesh, state, diagnostics, case.name, converged=True, status="converged", steps=0)
    if logger is not None:
        logger.emit_footer(solution)
    return solution


@functools.lru_cache(maxsize=16)
def _compiled(problem: ChannelProblem, fixed_flow: bool, dtype: str):
    """Compile one steady solve per problem, with the drive and the field scale as arguments.

    One program replaces the dispatch of every operation from the host, which
    is most of an eager solve's time, cold or warm. ``drive`` is the force
    density, or the flow rate when ``fixed_flow`` is set; that flow rate is met
    by scaling the unit-drive solution. The fields and the report share the
    program, so :func:`lmx.solve` and :func:`solve_fully_developed_fields` agree
    bit for bit and compile once between them.
    """

    def run(drive, field_scale):
        forcing = 1.0 if fixed_flow else drive
        velocity = solve_steady_state(
            problem,
            forcing=(forcing, 0.0, 0.0),
            field_scale=field_scale,
            tolerance=_TOLERANCE[jnp.result_type(float).name],
        ).velocity
        if fixed_flow:
            _, dy, dz = problem.grid.widths
            weights = jnp.asarray(dy)[:, None] * jnp.asarray(dz)[None, :]
            forcing = drive / jnp.sum(weights * velocity[0].data[0])
            velocity = tuple(component.replace_data(forcing * component.data) for component in velocity)
        fields, currents = _fields(problem, velocity, field_scale, dtype)
        drives = {"forcing": (forcing, 0.0, 0.0), "field_scale": field_scale}
        scale = _norm(steady_residual(zero_velocity(problem), problem, **drives))
        evidence = {
            "residual": _norm(steady_residual(velocity, problem, **drives)) / jnp.maximum(scale, 1e-300),
            "div_current_max": jnp.max(jnp.abs(divergence(currents).data)),
        }
        return fields, forcing, evidence

    return jax.jit(run)


def _fields(problem: ChannelProblem, velocity, field_scale, dtype):
    potential, currents, field = face_currents(velocity, problem, field_scale=field_scale)
    scalar = problem.scalar_conditions
    closed = tuple(wall_insulated(current, axis, scalar[axis]) for axis, current in enumerate(currents))
    force = lorentz_force(closed, field, scalar)
    current_y, current_z = currents[1].data[0], currents[2].data[0]
    jy = 0.5 * (current_y[:-1] + current_y[1:])
    jz = 0.5 * (current_z[:, :-1] + current_z[:, 1:])
    fields = (velocity[0].data[0], potential.data[0], jy, jz, force[0].data[0])
    return tuple(value.astype(dtype) for value in fields), currents


def _faces(count: int, half: float, layer: float, cells_in_layer: int) -> np.ndarray:
    """Wall-resolving faces as in ``duct_problem``; fewer layer cells when the mesh is too coarse."""
    if layer < half:
        for cells in range(cells_in_layer, 0, -1):
            try:
                return wall_resolving_faces(
                    count, -half, half, layer_thickness=layer, cells_in_layer=cells, max_ratio=None
                )
            except ValueError:
                if count % 2:
                    raise ValueError("a duct with a field needs an even cell count along each axis") from None
    return uniform_faces(count, -half, half)


def _fluid(case: CaseSpec):
    fluids = [region for region in case.regions if region.kind == "fluid"]
    if len(fluids) != 1:
        raise NotImplementedError("the staggered core solves one fluid region")
    return fluids[0]


def _check(case: CaseSpec) -> None:
    field = case.magnetic_field
    if case.solver.kind != "fully_developed_inductionless":
        raise ValueError("case must select the fully developed inductionless solver")
    if case.geometry.kind not in {"rect_duct", "layered_duct"}:
        raise NotImplementedError(f"the staggered core does not solve geometry {case.geometry.kind!r}")
    if field.kind != "constant" or field.value is None:
        raise NotImplementedError("the fully developed route needs a constant imposed field")
    if float(field.value[0]) != 0.0:
        raise NotImplementedError("a fully developed duct takes no axial field component")
    if field.ramp_duration > 0.0:
        raise ValueError("steady fields require an unramped magnetic field")
    for boundary in case.boundary_conditions:
        if boundary.kind not in _IGNORED | {"insulating", "conducting_wall"}:
            raise NotImplementedError(f"the staggered core does not impose {boundary.kind!r} boundaries")


def _wall_conductance(case: CaseSpec, conductivity: float) -> tuple[float, float, float]:
    """Return the thin-wall sheet conductance ``sigma_w t_w / sigma`` of each axis.

    A side is conducting when a ``conducting_wall`` boundary names it, and
    insulating when an ``insulating`` one does or when it has no wall cells. A
    wall layer that no boundary names would conduct through its cells as a thick
    wall, which the core does not model, unless its material is a perfect insulator.
    """
    regions = {region.name: region for region in case.regions}
    sides: dict[str, float] = {}
    for boundary in case.boundary_conditions:
        if boundary.kind not in {"insulating", "conducting_wall"}:
            continue
        names = _side_names(boundary)
        if boundary.kind == "insulating":
            for name in names:
                sides.setdefault(name, 0.0)
            continue
        region = regions.get(boundary.region)
        if case.geometry.kind != "layered_duct" or region is None or not names:
            raise NotImplementedError("a conducting wall needs a layered duct, a solid region and its sides")
        for name in names:
            thickness = case.geometry.wall_thickness[list(_SIDES).index(name)]
            if thickness <= 0.0:
                raise NotImplementedError(f"conducting wall {name!r} has no thickness")
            sides[name] = region.conductivity * thickness / conductivity if conductivity > 0.0 else 0.0
    solids = [region for region in case.regions if region.kind == "solid"]
    for index, name in enumerate(_SIDES):
        cells = case.geometry.wall_cells[index] if case.geometry.kind == "layered_duct" else 0
        if name not in sides and cells and solids and solids[0].conductivity > 0.0:
            raise NotImplementedError(
                f"wall {name!r} is a thick conducting layer; the core models thin walls"
            )
    conductance = [0.0, 0.0, 0.0]
    for axis in (1, 2):
        lower, upper = (sides.get(name, 0.0) for name, (index, _) in _SIDES.items() if index == axis)
        if lower != upper:
            raise NotImplementedError(
                "the thin-wall solve needs equal conductances on the two walls of an axis"
            )
        conductance[axis] = lower
    if conductance[1] > 0.0 and conductance[2] > 0.0:
        raise NotImplementedError("the thin-wall solve needs one insulating axis")
    return tuple(conductance)


def _side_names(boundary) -> tuple[str, ...]:
    """Return the walls a boundary names: ``left_right``, ``top_bottom``, a list, or an axis end."""
    side = (boundary.side or "").lower()
    if side in _PAIRS:
        return _PAIRS[side]
    if side in {"min", "max"} and (boundary.axis or "").lower() in {"y", "z"}:
        return ({"y": ("left", "right"), "z": ("bottom", "top")}[boundary.axis.lower()][side == "max"],)
    names = tuple(part.strip() for part in side.split(",") if part.strip())
    return names if all(name in _SIDES for name in names) else ()


def _target_flow_rate(case: CaseSpec) -> float | None:
    """Return the prescribed flow rate of a case driven by zero forcing."""
    if case.forcing != 0.0:
        return None
    for boundary in case.boundary_conditions:
        if boundary.kind == "inlet_flow_rate" and isinstance(boundary.value, (int, float)):
            return float(boundary.value)
    return None


def _norm(velocity) -> jax.Array:
    return jnp.sqrt(sum(jnp.sum(component.data**2) for component in velocity))

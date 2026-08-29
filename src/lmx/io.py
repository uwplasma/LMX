from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .mesh import StructuredMesh
from .specs import EXTRUDED_HISTORY_WIDTHS, Diagnostics, MHDState, Solution
from .validation import extract_midplane_profile

_DIAGNOSTIC_FIELDS = tuple(item.name for item in fields(Diagnostics))
_EXTRUDED_STATE_SCHEMA = "extruded_state_v1"
_EXTRUDED_FLUX_SCHEMA = "extruded_flux_v1"
_EXTRUDED_AITKEN_SCHEMA = "extruded_aitken_v1"
_EXTRUDED_ANDERSON_SCHEMA = "extruded_anderson_v1"
_EXTRUDED_RESTART_SCHEMAS = {
    _EXTRUDED_STATE_SCHEMA,
    _EXTRUDED_FLUX_SCHEMA,
    _EXTRUDED_AITKEN_SCHEMA,
    _EXTRUDED_ANDERSON_SCHEMA,
}
_B2_ANDERSON_FIELDS = (
    "anderson_mapped_state",
    "anderson_residual",
    "anderson_mapped_flux",
    "anderson_mapped_inlet",
)
_EXTRUDED_ARRAY_FIELDS = """x y z field_scale u v w p phi jx jy jz lorentz_x lorentz_y lorentz_z
residual volumetric_flow_rate mean_velocity axial_current wall_current_leakage current_scaled_pressure_proxy
charge_balance_residual boundary_current_residual""".split()
_EXTRUDED_FIELD_NAMES = tuple(_EXTRUDED_ARRAY_FIELDS[4:15])


def _portable_path(path: str | Path, *, relative_to: str | Path | None = None) -> str:
    candidate = Path(path)
    base = Path(relative_to) if relative_to is not None else Path.cwd()
    try:
        return str(candidate.relative_to(base))
    except ValueError:
        try:
            return str(candidate.resolve().relative_to(base.resolve()))
        except ValueError:
            return candidate.name if candidate.name else str(candidate)


def enable_compilation_cache(
    cache_dir: str | Path | None = None,
    *,
    min_compile_time_secs: float = 0.0,
    min_entry_size_bytes: int = -1,
) -> Path:
    """Enable JAX's persistent compilation cache before a heavy compile."""

    target = Path(cache_dir or (Path.home() / ".cache" / "lmx" / "jax_compilation"))
    target.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(target))
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", min_entry_size_bytes)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", min_compile_time_secs)
    return target


@dataclass(frozen=True)
class RestartBundle:
    path: Path
    state: MHDState
    diagnostics: Diagnostics
    metadata: dict[str, object]
    y_faces: np.ndarray
    z_faces: np.ndarray
    geometry_kind: str


@dataclass(frozen=True)
class ExtrudedRestartBundle:
    path: Path
    bundle: object
    station_history: tuple[dict[str, float], ...]
    metadata: dict[str, object]
    geometry_kind: str
    solver_kind: str


@dataclass(frozen=True)
class ExtrudedOutputLayout:
    root: Path
    system_dir: Path
    fields_dir: Path
    post_dir: Path
    plots_dir: Path
    restart_dir: Path
    logs_dir: Path


def _array_text(array: jnp.ndarray) -> str:
    return " ".join(f"{float(value):.12e}" for value in jnp.ravel(array))


def _rectilinear_points(mesh: StructuredMesh) -> str:
    return (
        f"<Coordinates>\n"
        f'<DataArray type="Float64" Name="X" format="ascii">{_array_text(mesh.x_faces)}</DataArray>\n'
        f'<DataArray type="Float64" Name="Y" format="ascii">{_array_text(mesh.y_faces)}</DataArray>\n'
        f'<DataArray type="Float64" Name="Z" format="ascii">{_array_text(mesh.z_faces)}</DataArray>\n'
        f"</Coordinates>"
    )


def _cell_data(solution: Solution) -> str:
    fields = {
        "u": solution.state.u,
        "phi": solution.state.phi,
        "jy": solution.state.jy,
        "jz": solution.state.jz,
        "lorentz_x": solution.state.lorentz_x,
    }
    arrays = []
    for name, field in fields.items():
        cell = field[None, :, :]
        arrays.append(
            f'<DataArray type="Float64" Name="{name}" format="ascii">{_array_text(cell)}</DataArray>'
        )
    return "<CellData>\n" + "\n".join(arrays) + "\n</CellData>"


def write_vtr(solution: Solution, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{solution.case_name}.vtr"
    mesh = solution.mesh
    content = (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="RectilinearGrid" version="0.1" byte_order="LittleEndian">\n'
        f'<RectilinearGrid WholeExtent="0 {mesh.nx} 0 {mesh.ny} 0 {mesh.nz}">\n'
        f'<Piece Extent="0 {mesh.nx} 0 {mesh.ny} 0 {mesh.nz}">\n'
        f"{_cell_data(solution)}\n"
        f"{_rectilinear_points(mesh)}\n"
        "</Piece>\n"
        "</RectilinearGrid>\n"
        "</VTKFile>\n"
    )
    target.write_text(content)
    return target


def write_vtu(mesh: StructuredMesh, out_dir: str | Path, name: str = "pipe_mesh") -> Path:
    if mesh.point_coordinates is None:
        raise ValueError("Mapped mesh requires point_coordinates")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{name}.vtu"
    points = mesh.point_coordinates.reshape((-1, 3))
    content = (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n'
        "<UnstructuredGrid>\n"
        f'<Piece NumberOfPoints="{points.shape[0]}" NumberOfCells="0">\n'
        "<Points>\n"
        f'<DataArray type="Float64" NumberOfComponents="3" format="ascii">{_array_text(points)}</DataArray>\n'
        "</Points>\n"
        "<Cells>\n"
        '<DataArray type="Int64" Name="connectivity" format="ascii"></DataArray>\n'
        '<DataArray type="Int64" Name="offsets" format="ascii"></DataArray>\n'
        '<DataArray type="UInt8" Name="types" format="ascii"></DataArray>\n'
        "</Cells>\n"
        "</Piece>\n"
        "</UnstructuredGrid>\n"
        "</VTKFile>\n"
    )
    target.write_text(content)
    return target


def write_pvd(entries: list[tuple[float, str]], out_dir: str | Path, name: str = "series") -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{name}.pvd"
    datasets = "\n".join(
        f'<DataSet timestep="{time:.8f}" group="" part="0" file="{filename}"/>' for time, filename in entries
    )
    content = (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n'
        "<Collection>\n"
        f"{datasets}\n"
        "</Collection>\n"
        "</VTKFile>\n"
    )
    target.write_text(content)
    return target


def write_paraview(solution: Solution, out_dir: str | Path) -> list[Path]:
    paths = []
    paths.append(write_vtr(solution, out_dir))
    paths.append(write_pvd([(solution.state.time, paths[0].name)], out_dir, name=solution.case_name))
    return paths


def write_solution_npz(solution: Solution, case, path: str | Path) -> Path:
    from .physics import build_material_fields

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    materials = build_material_fields(case, solution.mesh)
    metadata = {
        "case": solution.case_name,
        "time": float(solution.state.time),
        "residual": float(solution.state.residual),
        "description": "LMX solution dump",
        "geometry_kind": case.geometry.kind,
        "notes": case.notes,
        "restart_capable": True,
    }
    diag = solution.diagnostics
    np.savez_compressed(
        path,
        metadata_json=json.dumps(metadata),
        y_centers=np.asarray(solution.mesh.y_centers),
        z_centers=np.asarray(solution.mesh.z_centers),
        y_faces=np.asarray(solution.mesh.y_faces),
        z_faces=np.asarray(solution.mesh.z_faces),
        u=np.asarray(solution.state.u),
        phi=np.asarray(solution.state.phi),
        jy=np.asarray(solution.state.jy),
        jz=np.asarray(solution.state.jz),
        lorentz_x=np.asarray(solution.state.lorentz_x),
        state_time=np.asarray(float(solution.state.time)),
        state_residual=np.asarray(float(solution.state.residual)),
        conductivity=np.asarray(materials.conductivity),
        density=np.asarray(materials.density),
        viscosity=np.asarray(materials.viscosity),
        fluid_mask=np.asarray(materials.fluid_mask),
        **{name: np.asarray(getattr(diag, name)) for name in _DIAGNOSTIC_FIELDS},
    )
    return path


def write_extruded_solution_npz(solution, case, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = solution.bundle
    validation = solution.validation
    metadata = {
        "case": case.name,
        "geometry_kind": case.geometry.kind,
        "solver_kind": case.solver.kind,
        "station_count": int(bundle.x.shape[0]),
        "description": "LMX extruded inductionless solution dump",
    }
    np.savez_compressed(
        path,
        metadata_json=json.dumps(metadata),
        **{name: np.asarray(getattr(bundle, name)) for name in _EXTRUDED_ARRAY_FIELDS},
        axial_pressure_loss_gradient=np.asarray(
            getattr(bundle, "axial_pressure_loss_gradient", np.zeros_like(bundle.x))
        ),
        transverse_pressure_difference=np.asarray(
            getattr(bundle, "transverse_pressure_difference", np.zeros_like(bundle.x))
        ),
        validation_station_count=np.asarray(validation.station_count),
        validation_max_residual=np.asarray(validation.max_residual),
        validation_max_charge_balance_residual=np.asarray(validation.max_charge_balance_residual),
        validation_mean_velocity_span=np.asarray(validation.mean_velocity_span),
        validation_volumetric_flow_rate_span=np.asarray(validation.volumetric_flow_rate_span),
        validation_axial_current_span=np.asarray(validation.axial_current_span),
        validation_max_wall_current_leakage=np.asarray(validation.max_wall_current_leakage),
        validation_net_boundary_current_residual=np.asarray(validation.net_boundary_current_residual),
        validation_field_mean_velocity_correlation=np.asarray(validation.field_mean_velocity_correlation),
    )
    return path


def write_extruded_bundle_restart_npz(
    bundle,
    case,
    path: str | Path,
    *,
    station_history: tuple[dict[str, float], ...] = (),
) -> Path:
    """Write a solved or in-progress extruded bundle in the restart schema."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rho_phi_plus, rho_phi_inlet = _compact_flux_restart_arrays(bundle)
    has_compact_flux = rho_phi_plus is not None
    aitken_state = getattr(bundle, "aitken_state", None)
    anderson_state = getattr(bundle, "anderson_state", None)
    if aitken_state is not None and anderson_state is not None:
        raise ValueError("Restart cannot contain both Aitken and Anderson state")
    if anderson_state is not None:
        if len(anderson_state) != 4 or any(value is None for value in anderson_state):
            raise ValueError("B2 Anderson restart state must be all-or-none")
        anderson_arrays = tuple(np.asarray(value) for value in anderson_state)
        if (
            anderson_arrays[0].shape != (4, *np.asarray(bundle.u).shape)
            or anderson_arrays[0].shape != anderson_arrays[1].shape
            or not has_compact_flux
            or anderson_arrays[2].shape != np.asarray(rho_phi_plus).shape
            or anderson_arrays[3].shape != np.asarray(rho_phi_inlet).shape
        ):
            raise ValueError("B2 Anderson restart state has inconsistent shape")
    else:
        anderson_arrays = (np.zeros(0),) * 4
    courant_history = np.asarray(getattr(bundle, "iteration_courant_history", np.zeros((0, 3))))
    pressure_linear_history = np.asarray(
        getattr(bundle, "iteration_pressure_linear_history", np.zeros((0, 5)))
    )
    momentum_defect_history = np.asarray(getattr(bundle, "iteration_momentum_defect_history", np.zeros(0)))
    stopping_state = getattr(bundle, "stopping_state", (0, 0, "not_recorded"))
    has_diagnostics = (
        (aitken_state is not None or anderson_state is not None)
        and courant_history.shape[1:] == (3,)
        and 0 < len(courant_history) <= stopping_state[0]
    )
    has_pressure_diagnostics = pressure_linear_history.shape == (
        len(courant_history),
        5,
    )
    has_momentum_diagnostics = momentum_defect_history.shape == (len(courant_history),)
    if pressure_linear_history.size and not has_pressure_diagnostics:
        raise ValueError("Pressure linear history has inconsistent shape")
    if momentum_defect_history.size and not has_momentum_diagnostics:
        raise ValueError("Momentum defect history has inconsistent shape")
    if anderson_state is not None and not (
        has_diagnostics and has_pressure_diagnostics and has_momentum_diagnostics
    ):
        raise ValueError("B2 Anderson restart requires complete diagnostic histories")
    if (
        has_compact_flux
        and aitken_state is not None
        and not (has_diagnostics and has_pressure_diagnostics and has_momentum_diagnostics)
    ):
        raise ValueError("B2 Aitken restart requires complete diagnostic histories")
    restart_schema = (
        _EXTRUDED_ANDERSON_SCHEMA
        if anderson_state is not None
        else _EXTRUDED_AITKEN_SCHEMA
        if has_compact_flux and aitken_state is not None
        else _EXTRUDED_FLUX_SCHEMA
        if has_compact_flux
        else _EXTRUDED_STATE_SCHEMA
    )
    metadata = {
        "case": case.name,
        "geometry_kind": case.geometry.kind,
        "solver_kind": case.solver.kind,
        "station_count": int(bundle.x.shape[0]),
        "restart_schema": restart_schema,
        "stopping_state": list(stopping_state),
    }
    np.savez_compressed(
        path,
        metadata_json=json.dumps(metadata),
        station_history_json=json.dumps(station_history),
        **{name: np.asarray(getattr(bundle, name)) for name in _EXTRUDED_ARRAY_FIELDS},
        rho_phi_plus=np.asarray(rho_phi_plus) if has_compact_flux else np.zeros(0),
        rho_phi_inlet=np.asarray(rho_phi_inlet) if has_compact_flux else np.zeros(0),
        aitken_residual=np.asarray(aitken_state[0])
        if aitken_state is not None and aitken_state[0] is not None
        else np.zeros(0),
        aitken_relaxation=np.asarray(aitken_state[1] if aitken_state is not None else 1.0),
        steady_streak=np.asarray(aitken_state[2] if aitken_state is not None else stopping_state[1]),
        **dict(zip(_B2_ANDERSON_FIELDS, anderson_arrays, strict=True)),
        axial_pressure_loss_gradient=np.asarray(
            getattr(bundle, "axial_pressure_loss_gradient", np.zeros_like(bundle.x))
        ),
        transverse_pressure_difference=np.asarray(
            getattr(bundle, "transverse_pressure_difference", np.zeros_like(bundle.x))
        ),
        **{
            name: np.asarray(getattr(bundle, name, np.zeros((0, width)) if width else np.zeros(0)))
            for name, width in EXTRUDED_HISTORY_WIDTHS
        },
    )
    return path


def write_extruded_restart_npz(solution, case, path: str | Path) -> Path:
    return write_extruded_bundle_restart_npz(
        solution.bundle,
        case,
        path,
        station_history=solution.station_history,
    )


def write_restart_npz(solution: Solution, case, path: str | Path) -> Path:
    return write_solution_npz(solution, case, path)


def _load_optional_array(data: np.lib.npyio.NpzFile, key: str) -> np.ndarray:
    if key not in data:
        return np.zeros((0,), dtype=float)
    return np.asarray(data[key])


def _compact_flux_restart_arrays(bundle) -> tuple[object | None, object | None]:
    plus, inlet = (getattr(bundle, name, None) for name in ("rho_phi_plus", "rho_phi_inlet"))
    if (plus is None) != (inlet is None):
        raise ValueError("Compact restart flux requires both plus faces and inlet")
    return plus, inlet


def load_restart_bundle(path: str | Path) -> RestartBundle:
    path = Path(path).resolve()
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"])) if "metadata_json" in data else {}
        state_time = float(data["state_time"]) if "state_time" in data else float(metadata.get("time", 0.0))
        if "state_residual" in data:
            state_residual = float(data["state_residual"])
        else:
            residual_history = _load_optional_array(data, "residual_history")
            state_residual = (
                float(residual_history[-1]) if residual_history.size else float(metadata.get("residual", 0.0))
            )
        state = MHDState(
            u=jnp.asarray(data["u"]),
            phi=jnp.asarray(data["phi"]),
            jy=jnp.asarray(data["jy"]),
            jz=jnp.asarray(data["jz"]),
            lorentz_x=jnp.asarray(data["lorentz_x"]),
            time=state_time,
            residual=state_residual,
        )
        diagnostics = Diagnostics(
            **{name: jnp.asarray(_load_optional_array(data, name)) for name in _DIAGNOSTIC_FIELDS}
        )
        return RestartBundle(
            path=path,
            state=state,
            diagnostics=diagnostics,
            metadata=metadata,
            y_faces=np.asarray(data["y_faces"]),
            z_faces=np.asarray(data["z_faces"]),
            geometry_kind=str(metadata.get("geometry_kind", "unknown")),
        )


def load_extruded_restart_bundle(path: str | Path) -> ExtrudedRestartBundle:
    from .fringing import ExtrudedFieldBundle

    path = Path(path).resolve()
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"])) if "metadata_json" in data else {}
        has_plus = "rho_phi_plus" in data and data["rho_phi_plus"].size > 0
        has_inlet = "rho_phi_inlet" in data and data["rho_phi_inlet"].size > 0
        if has_plus != has_inlet:
            raise ValueError("Compact restart flux requires both plus faces and inlet")
        schema = metadata.get("restart_schema")
        if schema not in _EXTRUDED_RESTART_SCHEMAS:
            raise ValueError(f"Unsupported extruded restart schema: {schema!r}")
        has_aitken = schema == _EXTRUDED_AITKEN_SCHEMA
        if has_aitken and not {
            "aitken_residual",
            "aitken_relaxation",
            "steady_streak",
        } <= set(data.files):
            raise ValueError("B2 Aitken restart is missing accelerator state")
        anderson_present = tuple(field in data and data[field].size > 0 for field in _B2_ANDERSON_FIELDS)
        if any(anderson_present) and not all(anderson_present):
            raise ValueError("B2 Anderson restart state must be all-or-none")
        has_anderson = schema == _EXTRUDED_ANDERSON_SCHEMA
        if has_anderson and not all(anderson_present):
            raise ValueError("B2 Anderson restart is missing accelerator state")
        has_compact = schema != _EXTRUDED_STATE_SCHEMA
        if has_compact != (has_plus and has_inlet):
            raise ValueError("Extruded restart compact flux state is inconsistent")
        if all(anderson_present) and not has_anderson:
            raise ValueError("B2 Anderson restart schema is inconsistent")
        diagnostic_fields = (
            ("iteration_courant_history", "CFL"),
            ("iteration_pressure_linear_history", "pressure linear"),
            ("iteration_momentum_defect_history", "momentum defect"),
        )
        has_diagnostics = has_aitken or has_anderson
        for field, label in diagnostic_fields:
            if has_diagnostics and field not in data:
                raise ValueError(f"B2 diagnostic restart is missing {label} history")
        station_history = (
            tuple(json.loads(str(data["station_history_json"]))) if "station_history_json" in data else ()
        )
        retained_steps = _load_optional_array(data, "iteration_residual_history").size
        stopping_state = metadata.get("stopping_state")
        if stopping_state is None:
            raise ValueError("Extruded restart is missing stopping state")
        completed_steps = int(stopping_state[0]) if len(stopping_state) == 3 else -1
        if retained_steps > completed_steps or bool(retained_steps) != bool(completed_steps):
            raise ValueError("B2 restart stopping state has inconsistent step count")
        histories = {
            name: jnp.asarray(_load_optional_array(data, name)).reshape((-1, width))
            if width
            else jnp.asarray(_load_optional_array(data, name))
            for name, width in EXTRUDED_HISTORY_WIDTHS
        }
        anderson_state = (
            tuple(jnp.asarray(data[field]) for field in _B2_ANDERSON_FIELDS) if has_anderson else None
        )
        if has_anderson and (
            anderson_state[0].shape != (4, *data["u"].shape)
            or anderson_state[0].shape != anderson_state[1].shape
            or anderson_state[2].shape != data["rho_phi_plus"].shape
            or anderson_state[3].shape != data["rho_phi_inlet"].shape
        ):
            raise ValueError("B2 Anderson restart state has inconsistent shape")
        bundle = ExtrudedFieldBundle(
            **{name: jnp.asarray(data[name]) for name in _EXTRUDED_ARRAY_FIELDS},
            rho_phi_plus=jnp.asarray(data["rho_phi_plus"]) if has_plus else None,
            rho_phi_inlet=jnp.asarray(data["rho_phi_inlet"]) if has_inlet else None,
            aitken_state=(
                (
                    jnp.asarray(data["aitken_residual"]) if data["aitken_residual"].size else None,
                    float(data["aitken_relaxation"]),
                    int(data["steady_streak"]),
                )
                if has_aitken
                else None
            ),
            anderson_state=anderson_state,
            stopping_state=(
                int(stopping_state[0]),
                int(stopping_state[1]),
                str(stopping_state[2]),
            ),
            geometry_kind=str(metadata.get("geometry_kind", "unknown")),
            solver_kind=str(metadata.get("solver_kind", "extruded_inductionless")),
            axial_pressure_loss_gradient=jnp.asarray(
                _load_optional_array(data, "axial_pressure_loss_gradient")
            ),
            transverse_pressure_difference=jnp.asarray(
                _load_optional_array(data, "transverse_pressure_difference")
            ),
            **histories,
        )
        for field, _ in diagnostic_fields:
            history = getattr(bundle, field)
            if has_diagnostics and history.shape[0] != retained_steps:
                raise ValueError("B2 diagnostic restart histories have inconsistent lengths")
        if has_diagnostics and bundle.iteration_momentum_defect_history.ndim != 1:
            raise ValueError("B2 diagnostic restart histories have inconsistent lengths")
        return ExtrudedRestartBundle(
            path=path,
            bundle=bundle,
            station_history=station_history,
            metadata=metadata,
            geometry_kind=str(metadata.get("geometry_kind", "unknown")),
            solver_kind=str(metadata.get("solver_kind", "unknown")),
        )


def validate_restart_bundle(
    bundle: RestartBundle, *, mesh: StructuredMesh, geometry_kind: str, case_name: str
) -> None:
    if bundle.geometry_kind not in {"unknown", geometry_kind}:
        raise ValueError(
            f"Restart geometry_kind {bundle.geometry_kind!r} does not match current case geometry {geometry_kind!r}"
        )
    if bundle.state.u.shape != mesh.yz_shape:
        raise ValueError(
            f"Restart field shape {bundle.state.u.shape!r} does not match current mesh shape {mesh.yz_shape!r}"
        )
    if bundle.y_faces.shape != np.asarray(mesh.y_faces).shape or not np.allclose(
        bundle.y_faces, np.asarray(mesh.y_faces)
    ):
        raise ValueError("Restart y_faces do not match the current mesh")
    if bundle.z_faces.shape != np.asarray(mesh.z_faces).shape or not np.allclose(
        bundle.z_faces, np.asarray(mesh.z_faces)
    ):
        raise ValueError("Restart z_faces do not match the current mesh")
    restart_case = str(bundle.metadata.get("case", case_name))
    if restart_case != case_name:
        metadata_name = bundle.metadata.get("case")
        if metadata_name is not None:
            raise ValueError(f"Restart case {metadata_name!r} does not match current case name {case_name!r}")


def validate_extruded_restart_bundle(bundle: ExtrudedRestartBundle, *, case) -> None:
    from .mesh import _cross_section_mesh

    if bundle.geometry_kind not in {"unknown", case.geometry.kind}:
        raise ValueError(
            f"Extruded restart geometry_kind {bundle.geometry_kind!r} does not match current case geometry {case.geometry.kind!r}"
        )
    if bundle.solver_kind not in {"unknown", case.solver.kind}:
        raise ValueError(
            f"Extruded restart solver_kind {bundle.solver_kind!r} does not match current solver {case.solver.kind!r}"
        )
    if str(bundle.metadata.get("case", case.name)) != case.name:
        metadata_name = bundle.metadata.get("case")
        if metadata_name is not None:
            raise ValueError(
                f"Extruded restart case {metadata_name!r} does not match current case name {case.name!r}"
            )
    if int(bundle.bundle.x.shape[0]) != int(case.geometry.nx):
        raise ValueError("Extruded restart station count does not match current geometry.nx")
    mesh = _cross_section_mesh(case)
    expected_y, expected_z = mesh.yz_shape
    if int(bundle.bundle.y.shape[0]) != int(expected_y):
        raise ValueError("Extruded restart y resolution does not match the current extruded cross-section")
    if int(bundle.bundle.z.shape[0]) != int(expected_z):
        raise ValueError(
            "Extruded restart z/theta resolution does not match the current extruded cross-section"
        )
    plus, inlet = _compact_flux_restart_arrays(bundle.bundle)
    expected = None if inlet is None or inlet.ndim != 2 else (3, len(bundle.bundle.x), *inlet.shape)
    if plus is not None and (plus.ndim != 4 or plus.shape != expected):
        raise ValueError("Compact restart flux has inconsistent shape")


def prepare_extruded_output_layout(out_dir: str | Path) -> ExtrudedOutputLayout:
    root = Path(out_dir)
    layout = ExtrudedOutputLayout(
        root=root,
        system_dir=root / "system",
        fields_dir=root / "fields",
        post_dir=root / "postProcessing",
        plots_dir=root / "postProcessing" / "plots",
        restart_dir=root / "restart",
        logs_dir=root / "logs",
    )
    for directory in (
        layout.root,
        layout.system_dir,
        layout.fields_dir,
        layout.post_dir,
        layout.plots_dir,
        layout.restart_dir,
        layout.logs_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    return layout


def _write_extruded_station_archives(solution, case, layout: ExtrudedOutputLayout) -> list[Path]:
    stations_dir = layout.fields_dir / "stations"
    stations_dir.mkdir(parents=True, exist_ok=True)
    stride = max(int(getattr(case.output, "write_stride", 1)), 1)
    bundle = solution.bundle
    files: list[Path] = []
    station_indices = list(range(0, int(bundle.x.shape[0]), stride))
    if station_indices[-1] != int(bundle.x.shape[0]) - 1:
        station_indices.append(int(bundle.x.shape[0]) - 1)
    for index in station_indices:
        target = stations_dir / f"station_{index:04d}.npz"
        np.savez_compressed(
            target,
            station_index=int(index),
            x=float(bundle.x[index]),
            field_scale=float(bundle.field_scale[index]),
            y=np.asarray(bundle.y),
            z=np.asarray(bundle.z),
            u=np.asarray(bundle.u[index]),
            v=np.asarray(bundle.v[index]),
            w=np.asarray(bundle.w[index]),
            p=np.asarray(bundle.p[index]),
            phi=np.asarray(bundle.phi[index]),
            jx=np.asarray(bundle.jx[index]),
            jy=np.asarray(bundle.jy[index]),
            jz=np.asarray(bundle.jz[index]),
            lorentz_x=np.asarray(bundle.lorentz_x[index]),
            lorentz_y=np.asarray(bundle.lorentz_y[index]),
            lorentz_z=np.asarray(bundle.lorentz_z[index]),
            residual=float(bundle.residual[index]),
            volumetric_flow_rate=float(bundle.volumetric_flow_rate[index]),
            mean_velocity=float(bundle.mean_velocity[index]),
            axial_current=float(bundle.axial_current[index]),
            wall_current_leakage=float(bundle.wall_current_leakage[index]),
            current_scaled_pressure_proxy=float(bundle.current_scaled_pressure_proxy[index]),
            axial_pressure_loss_gradient=float(
                getattr(bundle, "axial_pressure_loss_gradient", jnp.zeros_like(bundle.x))[index]
            ),
            transverse_pressure_difference=float(
                getattr(bundle, "transverse_pressure_difference", jnp.zeros_like(bundle.x))[index]
            ),
            charge_balance_residual=float(bundle.charge_balance_residual[index]),
            boundary_current_residual=float(bundle.boundary_current_residual[index]),
        )
        files.append(target)
    manifest = {
        "case": case.name,
        "geometry_kind": case.geometry.kind,
        "solver_kind": case.solver.kind,
        "station_stride": stride,
        "station_count": int(bundle.x.shape[0]),
        "archived_station_indices": station_indices,
        "archived_files": [path.relative_to(layout.root).as_posix() for path in files],
        "fields": _EXTRUDED_FIELD_NAMES,
    }
    manifest_path = layout.system_dir / f"{case.name}_extruded_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return [manifest_path, *files]


def write_solution_outputs(
    solution: Solution,
    case,
    out_dir: str | Path,
    *,
    write_npz: bool = True,
    write_plots: bool = False,
) -> dict[str, list[Path]]:
    from .validation import (
        extract_centerline,
        extract_midplane_profile,
        write_profile_csv,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, list[Path]] = {"paraview": [], "csv": [], "npz": [], "plots": []}

    if case.output.write_paraview:
        payload["paraview"] = write_paraview(solution, out_dir)
    if case.output.write_csv_profiles:
        payload["csv"] = [
            write_profile_csv(out_dir / f"{case.name}_centerline.csv", extract_centerline(solution)),
            write_profile_csv(
                out_dir / f"{case.name}_midplane_y.csv",
                extract_midplane_profile(solution, axis="y", fluid_only=True),
            ),
            write_profile_csv(
                out_dir / f"{case.name}_midplane_z.csv",
                extract_midplane_profile(solution, axis="z", fluid_only=True),
            ),
        ]
    if write_npz and case.output.write_npz:
        payload["npz"] = [write_solution_npz(solution, case, out_dir / f"{case.name}_results.npz")]
    if write_plots and case.output.write_plots:
        payload["plots"] = write_case_overview_plots(solution, out_dir, case_title=case.name)
    return payload


def write_extruded_solution_outputs(
    solution,
    case,
    out_dir: str | Path,
    *,
    write_npz: bool = True,
    write_plots: bool = False,
) -> dict[str, list[Path]]:
    layout = prepare_extruded_output_layout(out_dir)
    payload: dict[str, list[Path]] = {"csv": [], "npz": [], "plots": [], "archive": []}
    station_csv = layout.post_dir / f"{case.name}_station_history.csv"
    station_csv.write_text(
        "x,field_scale,u_max,mean_velocity,volumetric_flow_rate,axial_current,wall_current_leakage,current_scaled_pressure_proxy,axial_pressure_loss_gradient,transverse_pressure_difference,residual,charge_balance_residual,boundary_current_residual\n"
        + "\n".join(
            ",".join(
                [
                    f"{float(record['x']):.12e}",
                    f"{float(record['field_scale']):.12e}",
                    f"{float(record['u_max']):.12e}",
                    f"{float(record['mean_velocity']):.12e}",
                    f"{float(record['volumetric_flow_rate']):.12e}",
                    f"{float(record['axial_current']):.12e}",
                    f"{float(record['wall_current_leakage']):.12e}",
                    f"{float(record['current_scaled_pressure_proxy']):.12e}",
                    f"{float(record.get('axial_pressure_loss_gradient', 0.0)):.12e}",
                    f"{float(record.get('transverse_pressure_difference', 0.0)):.12e}",
                    f"{float(record['residual']):.12e}",
                    f"{float(record['charge_balance_residual']):.12e}",
                    f"{float(record.get('boundary_current_residual', 0.0)):.12e}",
                ]
            )
            for record in solution.station_history
        )
        + "\n",
        encoding="utf-8",
    )
    payload["csv"].append(station_csv)
    if write_npz and case.output.write_npz:
        payload["npz"] = [
            write_extruded_solution_npz(
                solution, case, layout.fields_dir / f"{case.name}_extruded_results.npz"
            )
        ]
        payload["archive"] = _write_extruded_station_archives(solution, case, layout)
    if write_plots and case.output.write_plots:
        payload["plots"] = write_extruded_overview_plots(solution, layout.plots_dir, case_title=case.name)
    return payload


def _load_matplotlib() -> None:
    global plt, colors
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colors


def _set_plot_style() -> None:
    _load_matplotlib()
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "axes.linewidth": 0.9,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.5,
            "grid.color": "#4f4f4f",
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "legend.facecolor": "white",
            "legend.edgecolor": "#cbd5e1",
            "legend.fontsize": 13,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "lines.linewidth": 2.0,
        }
    )


def _prepare_plot_output(out_dir: str | Path) -> Path:
    """Load plotting dependencies, apply the house style, and create output."""

    _set_plot_style()
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    return output


def _save_figure_pair(
    fig,
    out_dir: Path,
    stem: str,
    *,
    dpi: int | None = None,
    tight: bool = True,
) -> list[Path]:
    """Save one figure as PNG and PDF, then release its Matplotlib state."""

    from matplotlib import pyplot

    save_options = {"bbox_inches": "tight"} if tight else {}
    if dpi is not None:
        save_options["dpi"] = dpi
    paths = [out_dir / f"{stem}.png", out_dir / f"{stem}.pdf"]
    for path in paths:
        fig.savefig(path, **save_options)
    pyplot.close(fig)
    return paths


def _plot_field(ax: plt.Axes, solution: Solution, field: jnp.ndarray, *, title: str, cmap: str) -> None:
    _load_matplotlib()
    mesh = solution.mesh
    field_min = float(jnp.min(field))
    field_max = float(jnp.max(field))
    if field_min >= 0.0:
        cmap = "magma"
        norm = colors.Normalize(vmin=field_min, vmax=max(field_max, field_min + 1e-12))
    elif field_max <= 0.0:
        cmap = "magma_r"
        norm = colors.Normalize(vmin=min(field_min, field_max - 1e-12), vmax=field_max)
    else:
        vmax = float(jnp.max(jnp.abs(field)))
        vmax = max(vmax, 1e-12)
        norm = colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    image = ax.pcolormesh(
        mesh.z_faces,
        mesh.y_faces,
        field,
        shading="auto",
        cmap=cmap,
        norm=norm,
    )
    ax.set_title(title)
    ax.set_xlabel("z")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def _plot_profile(
    ax: plt.Axes,
    coordinate: jnp.ndarray,
    values: jnp.ndarray,
    *,
    axis_name: str,
    title: str,
    reference_coordinate: jnp.ndarray | None = None,
    reference_values: jnp.ndarray | None = None,
    reference_label: str | None = None,
) -> None:
    coord_scale = float(jnp.max(jnp.abs(coordinate)))
    coord_scale = coord_scale if coord_scale > 0.0 else 1.0
    value_scale = float(jnp.max(jnp.abs(values)))
    value_scale = value_scale if value_scale > 0.0 else 1.0
    ax.plot(coordinate / coord_scale, values / value_scale, color="#0f766e", label="LMX")
    if reference_coordinate is not None and reference_values is not None:
        ref_coord_scale = float(jnp.max(jnp.abs(reference_coordinate)))
        ref_coord_scale = ref_coord_scale if ref_coord_scale > 0.0 else 1.0
        ref_value_scale = float(jnp.max(jnp.abs(reference_values)))
        ref_value_scale = ref_value_scale if ref_value_scale > 0.0 else 1.0
        ax.plot(
            reference_coordinate / ref_coord_scale,
            reference_values / ref_value_scale,
            color="#b45309",
            linestyle="--",
            label=reference_label or "Reference",
        )
    ax.set_title(title)
    ax.set_xlabel(f"Normalized {axis_name}")
    ax.set_ylabel("Normalized velocity")
    ax.set_xlim(-1.02, 1.02)
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98))


def write_case_overview_plots(
    solution: Solution,
    out_dir: str | Path,
    *,
    case_title: str,
    y_reference_coordinate: jnp.ndarray | None = None,
    y_reference_values: jnp.ndarray | None = None,
    z_reference_coordinate: jnp.ndarray | None = None,
    z_reference_values: jnp.ndarray | None = None,
    reference_label: str = "Reference",
) -> list[Path]:
    out_dir = _prepare_plot_output(out_dir)

    y_profile = extract_midplane_profile(solution, axis="y", fluid_only=True)
    z_profile = extract_midplane_profile(solution, axis="z", fluid_only=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    fig.suptitle(case_title, fontsize=16, y=1.02)

    _plot_field(axes[0, 0], solution, solution.state.u, title="Velocity u", cmap="RdBu_r")
    _plot_field(
        axes[0, 1],
        solution,
        solution.state.phi,
        title="Electric potential φ",
        cmap="PuOr_r",
    )
    _plot_profile(
        axes[1, 0],
        y_profile["y"],
        y_profile["u"],
        axis_name="y",
        title="Midplane y profile",
        reference_coordinate=y_reference_coordinate,
        reference_values=y_reference_values,
        reference_label=reference_label,
    )
    _plot_profile(
        axes[1, 1],
        z_profile["z"],
        z_profile["u"],
        axis_name="z",
        title="Midplane z profile",
        reference_coordinate=z_reference_coordinate,
        reference_values=z_reference_values,
        reference_label=reference_label,
    )

    overview_paths = _save_figure_pair(fig, out_dir, "overview")

    diagnostics_paths: list[Path] = []
    if solution.diagnostics.time_history.size > 0:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
        time_history = solution.diagnostics.time_history
        axes[0].plot(
            time_history,
            solution.diagnostics.u_max_history,
            color="#1d4ed8",
            label="max |u|",
        )
        if solution.diagnostics.current_max_history.size:
            axes[0].plot(
                time_history,
                solution.diagnostics.current_max_history,
                color="#b91c1c",
                label="max |J|",
            )
        if solution.diagnostics.lorentz_max_history.size:
            axes[0].plot(
                time_history,
                solution.diagnostics.lorentz_max_history,
                color="#6d28d9",
                label="max |J×B|",
            )
        axes[0].set_title("Trace magnitudes")
        axes[0].set_xlabel("time")
        axes[0].set_ylabel("magnitude")
        axes[0].legend(loc="upper left", bbox_to_anchor=(0.02, 0.98))

        axes[1].plot(
            time_history,
            solution.diagnostics.residual_history,
            color="#0f766e",
            label="velocity residual",
        )
        if solution.diagnostics.potential_residual_history.size:
            axes[1].plot(
                time_history,
                solution.diagnostics.potential_residual_history,
                color="#b45309",
                label="potential residual",
            )
        axes[1].set_title("Solver residuals")
        axes[1].set_xlabel("time")
        axes[1].set_ylabel("residual")
        axes[1].set_yscale("log")
        axes[1].legend(loc="upper left", bbox_to_anchor=(0.02, 0.98))

        diagnostics_paths = _save_figure_pair(fig, out_dir, "diagnostics")

    return [*overview_paths, *diagnostics_paths]


def _centers_to_edges(values: np.ndarray) -> np.ndarray:
    data = np.asarray(values, dtype=float)
    if data.size <= 1:
        center = float(data[0]) if data.size else 0.0
        return np.asarray([center - 0.5, center + 0.5], dtype=float)
    midpoints = 0.5 * (data[1:] + data[:-1])
    first = data[0] - 0.5 * (data[1] - data[0])
    last = data[-1] + 0.5 * (data[-1] - data[-2])
    return np.concatenate([[first], midpoints, [last]])


def write_extruded_overview_plots(
    solution,
    out_dir: str | Path,
    *,
    case_title: str,
) -> list[Path]:
    out_dir = _prepare_plot_output(out_dir)

    bundle = solution.bundle
    validation = solution.validation
    x = np.asarray(bundle.x, dtype=float)
    field_scale = np.asarray(bundle.field_scale, dtype=float)
    mean_velocity = np.asarray(bundle.mean_velocity, dtype=float)
    current_proxy = np.asarray(bundle.current_scaled_pressure_proxy, dtype=float)
    charge_balance = np.maximum(np.asarray(bundle.charge_balance_residual, dtype=float), 1.0e-16)
    boundary_current = np.maximum(np.asarray(bundle.boundary_current_residual, dtype=float), 1.0e-16)
    wall_leakage = np.maximum(np.asarray(bundle.wall_current_leakage, dtype=float), 1.0e-16)
    axial_current = np.asarray(bundle.axial_current, dtype=float)

    peak_index = int(np.argmax(np.abs(field_scale))) if field_scale.size else 0
    y = np.asarray(bundle.y, dtype=float)
    z = np.asarray(bundle.z, dtype=float)
    y_edges = _centers_to_edges(y)
    z_edges = _centers_to_edges(z)
    coord_x_label = "r" if bundle.geometry_kind == "pipe_ogrid" else "y"
    coord_y_label = r"$\theta$" if bundle.geometry_kind == "pipe_ogrid" else "z"

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), constrained_layout=True)
    fig.suptitle(case_title, fontsize=16)

    axes[0, 0].plot(x, mean_velocity, color="#0f766e", label="Mean velocity")
    axes[0, 0].plot(x, current_proxy, color="#b45309", linestyle="--", label="Current proxy")
    axes[0, 0].plot(x, field_scale, color="#1d4ed8", alpha=0.7, label="Field scale")
    axes[0, 0].set_title("Station response")
    axes[0, 0].set_xlabel("x")
    axes[0, 0].legend(loc="upper left", bbox_to_anchor=(0.02, 0.98))

    axes[0, 1].semilogy(x, charge_balance, color="#7c3aed", label="Charge balance")
    axes[0, 1].semilogy(x, wall_leakage, color="#dc2626", linestyle="--", label="Wall leakage")
    axes[0, 1].semilogy(x, boundary_current, color="#0891b2", linestyle=":", label="Boundary residual")
    axes[0, 1].plot(
        x,
        np.maximum(np.abs(axial_current), 1.0e-16),
        color="#111827",
        alpha=0.6,
        label="|Axial current|",
    )
    axes[0, 1].set_title(
        "Conservation audit\n"
        f"max|div J|={validation.max_charge_balance_residual:.2e}, "
        f"net boundary={validation.net_boundary_current_residual:.2e}"
    )
    axes[0, 1].set_xlabel("x")
    axes[0, 1].legend(loc="upper left", bbox_to_anchor=(0.02, 0.98))

    u_station = np.asarray(bundle.u[peak_index], dtype=float)
    phi_station = np.asarray(bundle.phi[peak_index], dtype=float)
    u_im = axes[1, 0].pcolormesh(z_edges, y_edges, u_station, shading="auto", cmap="RdBu_r")
    plt.colorbar(u_im, ax=axes[1, 0], fraction=0.046, pad=0.04)
    axes[1, 0].set_title(f"u at peak field station (x={x[peak_index]:.2f})")
    axes[1, 0].set_xlabel(coord_y_label)
    axes[1, 0].set_ylabel(coord_x_label)

    phi_im = axes[1, 1].pcolormesh(z_edges, y_edges, phi_station, shading="auto", cmap="PuOr_r")
    plt.colorbar(phi_im, ax=axes[1, 1], fraction=0.046, pad=0.04)
    axes[1, 1].set_title("Electric potential at peak field station")
    axes[1, 1].set_xlabel(coord_y_label)
    axes[1, 1].set_ylabel(coord_x_label)

    return _save_figure_pair(fig, out_dir, "extruded_overview")

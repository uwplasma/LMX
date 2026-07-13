from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from .core import Diagnostics, MHDState, Solution
from .mesh import StructuredMesh


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


def write_vtu(
    mesh: StructuredMesh, out_dir: str | Path, name: str = "pipe_mesh"
) -> Path:
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


def write_pvd(
    entries: list[tuple[float, str]], out_dir: str | Path, name: str = "series"
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{name}.pvd"
    datasets = "\n".join(
        f'<DataSet timestep="{time:.8f}" group="" part="0" file="{filename}"/>'
        for time, filename in entries
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
    paths.append(
        write_pvd(
            [(solution.state.time, paths[0].name)], out_dir, name=solution.case_name
        )
    )
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
        time_history=np.asarray(diag.time_history),
        residual_history=np.asarray(diag.residual_history),
        potential_residual_history=np.asarray(diag.potential_residual_history),
        potential_iterations_history=np.asarray(diag.potential_iterations_history),
        u_max_history=np.asarray(diag.u_max_history),
        mean_velocity_history=np.asarray(diag.mean_velocity_history),
        current_max_history=np.asarray(diag.current_max_history),
        face_current_max_history=np.asarray(diag.face_current_max_history),
        emf_max_history=np.asarray(diag.emf_max_history),
        lorentz_max_history=np.asarray(diag.lorentz_max_history),
        applied_forcing_history=np.asarray(diag.applied_forcing_history),
        pressure_proxy_history=np.asarray(diag.pressure_proxy_history),
        current_scaled_pressure_proxy_history=np.asarray(
            diag.current_scaled_pressure_proxy_history
        ),
        raw_update_max_history=np.asarray(diag.raw_update_max_history),
        limiter_scale_history=np.asarray(diag.limiter_scale_history),
        limited_fraction_history=np.asarray(diag.limited_fraction_history),
        linear_residual_history=np.asarray(diag.linear_residual_history),
        linear_iterations_history=np.asarray(diag.linear_iterations_history),
        volumetric_flow_rate_history=np.asarray(diag.volumetric_flow_rate_history),
        mean_current_magnitude_history=np.asarray(diag.mean_current_magnitude_history),
        lorentz_power_history=np.asarray(diag.lorentz_power_history),
        div_current_max_history=np.asarray(diag.div_current_max_history),
        charge_balance_residual_history=np.asarray(
            diag.charge_balance_residual_history
        ),
        gauge_residual_history=np.asarray(diag.gauge_residual_history),
        interface_current_residual_history=np.asarray(
            diag.interface_current_residual_history
        ),
        courant_like=np.asarray(diag.courant_like),
        ohmic_power=np.asarray(diag.ohmic_power),
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
        x=np.asarray(bundle.x),
        y=np.asarray(bundle.y),
        z=np.asarray(bundle.z),
        field_scale=np.asarray(bundle.field_scale),
        u=np.asarray(bundle.u),
        v=np.asarray(bundle.v),
        w=np.asarray(bundle.w),
        p=np.asarray(bundle.p),
        phi=np.asarray(bundle.phi),
        jx=np.asarray(bundle.jx),
        jy=np.asarray(bundle.jy),
        jz=np.asarray(bundle.jz),
        lorentz_x=np.asarray(bundle.lorentz_x),
        lorentz_y=np.asarray(bundle.lorentz_y),
        lorentz_z=np.asarray(bundle.lorentz_z),
        residual=np.asarray(bundle.residual),
        volumetric_flow_rate=np.asarray(bundle.volumetric_flow_rate),
        mean_velocity=np.asarray(bundle.mean_velocity),
        axial_current=np.asarray(bundle.axial_current),
        wall_current_leakage=np.asarray(bundle.wall_current_leakage),
        current_scaled_pressure_proxy=np.asarray(bundle.current_scaled_pressure_proxy),
        axial_pressure_loss_gradient=np.asarray(
            getattr(bundle, "axial_pressure_loss_gradient", np.zeros_like(bundle.x))
        ),
        transverse_pressure_difference=np.asarray(
            getattr(bundle, "transverse_pressure_difference", np.zeros_like(bundle.x))
        ),
        charge_balance_residual=np.asarray(bundle.charge_balance_residual),
        validation_station_count=np.asarray(validation.station_count),
        validation_max_residual=np.asarray(validation.max_residual),
        validation_max_charge_balance_residual=np.asarray(
            validation.max_charge_balance_residual
        ),
        validation_mean_velocity_span=np.asarray(validation.mean_velocity_span),
        validation_volumetric_flow_rate_span=np.asarray(
            validation.volumetric_flow_rate_span
        ),
        validation_axial_current_span=np.asarray(validation.axial_current_span),
        validation_max_wall_current_leakage=np.asarray(
            validation.max_wall_current_leakage
        ),
        validation_net_boundary_current_residual=np.asarray(
            validation.net_boundary_current_residual
        ),
        validation_field_mean_velocity_correlation=np.asarray(
            validation.field_mean_velocity_correlation
        ),
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
    metadata = {
        "case": case.name,
        "geometry_kind": case.geometry.kind,
        "solver_kind": case.solver.kind,
        "station_count": int(bundle.x.shape[0]),
        "description": "LMX extruded inductionless restart bundle",
        "restart_capable": True,
    }
    np.savez_compressed(
        path,
        metadata_json=json.dumps(metadata),
        station_history_json=json.dumps(station_history),
        x=np.asarray(bundle.x),
        y=np.asarray(bundle.y),
        z=np.asarray(bundle.z),
        field_scale=np.asarray(bundle.field_scale),
        u=np.asarray(bundle.u),
        v=np.asarray(bundle.v),
        w=np.asarray(bundle.w),
        p=np.asarray(bundle.p),
        phi=np.asarray(bundle.phi),
        jx=np.asarray(bundle.jx),
        jy=np.asarray(bundle.jy),
        jz=np.asarray(bundle.jz),
        lorentz_x=np.asarray(bundle.lorentz_x),
        lorentz_y=np.asarray(bundle.lorentz_y),
        lorentz_z=np.asarray(bundle.lorentz_z),
        residual=np.asarray(bundle.residual),
        volumetric_flow_rate=np.asarray(bundle.volumetric_flow_rate),
        mean_velocity=np.asarray(bundle.mean_velocity),
        axial_current=np.asarray(bundle.axial_current),
        wall_current_leakage=np.asarray(bundle.wall_current_leakage),
        current_scaled_pressure_proxy=np.asarray(bundle.current_scaled_pressure_proxy),
        axial_pressure_loss_gradient=np.asarray(
            getattr(bundle, "axial_pressure_loss_gradient", np.zeros_like(bundle.x))
        ),
        transverse_pressure_difference=np.asarray(
            getattr(bundle, "transverse_pressure_difference", np.zeros_like(bundle.x))
        ),
        charge_balance_residual=np.asarray(bundle.charge_balance_residual),
        boundary_current_residual=np.asarray(bundle.boundary_current_residual),
        iteration_residual_history=np.asarray(
            getattr(bundle, "iteration_residual_history", np.zeros((0,)))
        ),
        iteration_component_residual_history=np.asarray(
            getattr(bundle, "iteration_component_residual_history", np.zeros((0, 6)))
        ),
        iteration_pressure_residual_history=np.asarray(
            getattr(bundle, "iteration_pressure_residual_history", np.zeros((0,)))
        ),
        iteration_electric_linear_history=np.asarray(
            getattr(bundle, "iteration_electric_linear_history", np.zeros((0, 6)))
        ),
        iteration_potential_residual_history=np.asarray(
            getattr(bundle, "iteration_potential_residual_history", np.zeros((0,)))
        ),
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


def load_restart_bundle(path: str | Path) -> RestartBundle:
    path = Path(path).resolve()
    with np.load(path, allow_pickle=False) as data:
        metadata = (
            json.loads(str(data["metadata_json"])) if "metadata_json" in data else {}
        )
        state_time = (
            float(data["state_time"])
            if "state_time" in data
            else float(metadata.get("time", 0.0))
        )
        if "state_residual" in data:
            state_residual = float(data["state_residual"])
        else:
            residual_history = _load_optional_array(data, "residual_history")
            state_residual = (
                float(residual_history[-1])
                if residual_history.size
                else float(metadata.get("residual", 0.0))
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
            time_history=jnp.asarray(_load_optional_array(data, "time_history")),
            u_max_history=jnp.asarray(_load_optional_array(data, "u_max_history")),
            mean_velocity_history=jnp.asarray(
                _load_optional_array(data, "mean_velocity_history")
            ),
            applied_forcing_history=jnp.asarray(
                _load_optional_array(data, "applied_forcing_history")
            ),
            pressure_proxy_history=jnp.asarray(
                _load_optional_array(data, "pressure_proxy_history")
            ),
            current_scaled_pressure_proxy_history=jnp.asarray(
                _load_optional_array(data, "current_scaled_pressure_proxy_history")
            ),
            raw_update_max_history=jnp.asarray(
                _load_optional_array(data, "raw_update_max_history")
            ),
            limiter_scale_history=jnp.asarray(
                _load_optional_array(data, "limiter_scale_history")
            ),
            limited_fraction_history=jnp.asarray(
                _load_optional_array(data, "limited_fraction_history")
            ),
            residual_history=jnp.asarray(
                _load_optional_array(data, "residual_history")
            ),
            courant_like=jnp.asarray(_load_optional_array(data, "courant_like")),
            ohmic_power=jnp.asarray(_load_optional_array(data, "ohmic_power")),
            current_max_history=jnp.asarray(
                _load_optional_array(data, "current_max_history")
            ),
            face_current_max_history=jnp.asarray(
                _load_optional_array(data, "face_current_max_history")
            ),
            emf_max_history=jnp.asarray(_load_optional_array(data, "emf_max_history")),
            lorentz_max_history=jnp.asarray(
                _load_optional_array(data, "lorentz_max_history")
            ),
            potential_residual_history=jnp.asarray(
                _load_optional_array(data, "potential_residual_history")
            ),
            potential_iterations_history=jnp.asarray(
                _load_optional_array(data, "potential_iterations_history")
            ),
            linear_residual_history=jnp.asarray(
                _load_optional_array(data, "linear_residual_history")
            ),
            linear_iterations_history=jnp.asarray(
                _load_optional_array(data, "linear_iterations_history")
            ),
            volumetric_flow_rate_history=jnp.asarray(
                _load_optional_array(data, "volumetric_flow_rate_history")
            ),
            mean_current_magnitude_history=jnp.asarray(
                _load_optional_array(data, "mean_current_magnitude_history")
            ),
            lorentz_power_history=jnp.asarray(
                _load_optional_array(data, "lorentz_power_history")
            ),
            div_current_max_history=jnp.asarray(
                _load_optional_array(data, "div_current_max_history")
            ),
            charge_balance_residual_history=jnp.asarray(
                _load_optional_array(data, "charge_balance_residual_history")
            ),
            gauge_residual_history=jnp.asarray(
                _load_optional_array(data, "gauge_residual_history")
            ),
            interface_current_residual_history=jnp.asarray(
                _load_optional_array(data, "interface_current_residual_history")
            ),
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
        metadata = (
            json.loads(str(data["metadata_json"])) if "metadata_json" in data else {}
        )
        station_history = (
            tuple(json.loads(str(data["station_history_json"])))
            if "station_history_json" in data
            else ()
        )
        bundle = ExtrudedFieldBundle(
            x=jnp.asarray(data["x"]),
            y=jnp.asarray(data["y"]),
            z=jnp.asarray(data["z"]),
            field_scale=jnp.asarray(data["field_scale"]),
            u=jnp.asarray(data["u"]),
            v=jnp.asarray(data["v"]),
            w=jnp.asarray(data["w"]),
            p=jnp.asarray(data["p"]),
            phi=jnp.asarray(data["phi"]),
            jx=jnp.asarray(data["jx"]),
            jy=jnp.asarray(data["jy"]),
            jz=jnp.asarray(data["jz"]),
            lorentz_x=jnp.asarray(data["lorentz_x"]),
            lorentz_y=jnp.asarray(data["lorentz_y"]),
            lorentz_z=jnp.asarray(data["lorentz_z"]),
            residual=jnp.asarray(data["residual"]),
            volumetric_flow_rate=jnp.asarray(data["volumetric_flow_rate"]),
            mean_velocity=jnp.asarray(data["mean_velocity"]),
            axial_current=jnp.asarray(data["axial_current"]),
            wall_current_leakage=jnp.asarray(data["wall_current_leakage"]),
            current_scaled_pressure_proxy=jnp.asarray(
                data["current_scaled_pressure_proxy"]
            ),
            charge_balance_residual=jnp.asarray(data["charge_balance_residual"]),
            boundary_current_residual=jnp.asarray(data["boundary_current_residual"]),
            geometry_kind=str(metadata.get("geometry_kind", "unknown")),
            solver_kind=str(metadata.get("solver_kind", "extruded_inductionless")),
            axial_pressure_loss_gradient=jnp.asarray(
                _load_optional_array(data, "axial_pressure_loss_gradient")
            ),
            transverse_pressure_difference=jnp.asarray(
                _load_optional_array(data, "transverse_pressure_difference")
            ),
            iteration_residual_history=jnp.asarray(
                _load_optional_array(data, "iteration_residual_history")
            ),
            iteration_component_residual_history=jnp.asarray(
                _load_optional_array(data, "iteration_component_residual_history")
            ).reshape((-1, 6)),
            iteration_pressure_residual_history=jnp.asarray(
                _load_optional_array(data, "iteration_pressure_residual_history")
            ),
            iteration_electric_linear_history=jnp.asarray(
                _load_optional_array(data, "iteration_electric_linear_history")
            ).reshape((-1, 6)),
            iteration_potential_residual_history=jnp.asarray(
                _load_optional_array(data, "iteration_potential_residual_history")
            ),
        )
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
            raise ValueError(
                f"Restart case {metadata_name!r} does not match current case name {case_name!r}"
            )


def validate_extruded_restart_bundle(bundle: ExtrudedRestartBundle, *, case) -> None:
    from .fringing import _cross_section_mesh

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
        raise ValueError(
            "Extruded restart station count does not match current geometry.nx"
        )
    mesh = _cross_section_mesh(case)
    expected_y, expected_z = mesh.yz_shape
    if int(bundle.bundle.y.shape[0]) != int(expected_y):
        raise ValueError(
            "Extruded restart y resolution does not match the current extruded cross-section"
        )
    if int(bundle.bundle.z.shape[0]) != int(expected_z):
        raise ValueError(
            "Extruded restart z/theta resolution does not match the current extruded cross-section"
        )


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


def _write_extruded_station_archives(
    solution, case, layout: ExtrudedOutputLayout
) -> list[Path]:
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
            current_scaled_pressure_proxy=float(
                bundle.current_scaled_pressure_proxy[index]
            ),
            axial_pressure_loss_gradient=float(
                getattr(
                    bundle, "axial_pressure_loss_gradient", jnp.zeros_like(bundle.x)
                )[index]
            ),
            transverse_pressure_difference=float(
                getattr(
                    bundle, "transverse_pressure_difference", jnp.zeros_like(bundle.x)
                )[index]
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
        "fields": [
            "u",
            "v",
            "w",
            "p",
            "phi",
            "jx",
            "jy",
            "jz",
            "lorentz_x",
            "lorentz_y",
            "lorentz_z",
        ],
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
            write_profile_csv(
                out_dir / f"{case.name}_centerline.csv", extract_centerline(solution)
            ),
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
        payload["npz"] = [
            write_solution_npz(solution, case, out_dir / f"{case.name}_results.npz")
        ]
    if write_plots and case.output.write_plots:
        from .plotting import write_case_overview_plots

        payload["plots"] = write_case_overview_plots(
            solution, out_dir, case_title=case.name
        )
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
        from .plotting import write_extruded_overview_plots

        payload["plots"] = write_extruded_overview_plots(
            solution, layout.plots_dir, case_title=case.name
        )
    return payload

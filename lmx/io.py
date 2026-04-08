from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from .core import Solution
from .mesh import StructuredMesh


def _array_text(array: jnp.ndarray) -> str:
    return " ".join(f"{float(value):.12e}" for value in jnp.ravel(array))


def _rectilinear_points(mesh: StructuredMesh) -> str:
    return (
        f'<Coordinates>\n'
        f'<DataArray type="Float64" Name="X" format="ascii">{_array_text(mesh.x_faces)}</DataArray>\n'
        f'<DataArray type="Float64" Name="Y" format="ascii">{_array_text(mesh.y_faces)}</DataArray>\n'
        f'<DataArray type="Float64" Name="Z" format="ascii">{_array_text(mesh.z_faces)}</DataArray>\n'
        f'</Coordinates>'
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
        arrays.append(f'<DataArray type="Float64" Name="{name}" format="ascii">{_array_text(cell)}</DataArray>')
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
        f'{_cell_data(solution)}\n'
        f'{_rectilinear_points(mesh)}\n'
        '</Piece>\n'
        '</RectilinearGrid>\n'
        '</VTKFile>\n'
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
        '<UnstructuredGrid>\n'
        f'<Piece NumberOfPoints="{points.shape[0]}" NumberOfCells="0">\n'
        '<Points>\n'
        f'<DataArray type="Float64" NumberOfComponents="3" format="ascii">{_array_text(points)}</DataArray>\n'
        '</Points>\n'
        '<Cells>\n'
        '<DataArray type="Int64" Name="connectivity" format="ascii"></DataArray>\n'
        '<DataArray type="Int64" Name="offsets" format="ascii"></DataArray>\n'
        '<DataArray type="UInt8" Name="types" format="ascii"></DataArray>\n'
        '</Cells>\n'
        '</Piece>\n'
        '</UnstructuredGrid>\n'
        '</VTKFile>\n'
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
        '<Collection>\n'
        f"{datasets}\n"
        '</Collection>\n'
        '</VTKFile>\n'
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
        "description": "LMX solution dump",
        "geometry_kind": case.geometry.kind,
        "notes": case.notes,
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
        courant_like=np.asarray(diag.courant_like),
        ohmic_power=np.asarray(diag.ohmic_power),
    )
    return path


def write_solution_outputs(
    solution: Solution,
    case,
    out_dir: str | Path,
    *,
    write_npz: bool = True,
    write_plots: bool = False,
) -> dict[str, list[Path]]:
    from .validation import extract_centerline, extract_midplane_profile, write_profile_csv

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, list[Path]] = {"paraview": [], "csv": [], "npz": [], "plots": []}

    if case.output.write_paraview:
        payload["paraview"] = write_paraview(solution, out_dir)
    if case.output.write_csv_profiles:
        payload["csv"] = [
            write_profile_csv(out_dir / f"{case.name}_centerline.csv", extract_centerline(solution)),
            write_profile_csv(out_dir / f"{case.name}_midplane_y.csv", extract_midplane_profile(solution, axis="y", fluid_only=True)),
            write_profile_csv(out_dir / f"{case.name}_midplane_z.csv", extract_midplane_profile(solution, axis="z", fluid_only=True)),
        ]
    if write_npz and case.output.write_npz:
        payload["npz"] = [write_solution_npz(solution, case, out_dir / f"{case.name}_results.npz")]
    if write_plots and case.output.write_plots:
        from .plotting import write_case_overview_plots

        payload["plots"] = write_case_overview_plots(solution, out_dir, case_title=case.name)
    return payload

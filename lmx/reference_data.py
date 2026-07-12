from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from .mesh import StructuredMesh, generate_rect_duct_mesh_from_faces


@dataclass(frozen=True)
class ClosedChannelAnalyticalReference:
    case_kind: str
    ha: int
    coordinate: jnp.ndarray
    midplane_z: jnp.ndarray
    midplane_y: jnp.ndarray
    pressure_drop: float | None
    path: str


@dataclass(frozen=True)
class ProcessedSliceReference:
    case_kind: str
    ha: int
    columns: dict[str, jnp.ndarray]
    path: str


@dataclass(frozen=True)
class FringingPipeProfileReference:
    profile_kind: str
    coordinate: jnp.ndarray
    velocity: jnp.ndarray
    x_offset_fraction: float
    path: str


def default_closed_channel_reference_root(reference_root: str | Path | None = None) -> Path:
    if reference_root is not None:
        return Path(reference_root)
    repo_root = Path(__file__).resolve().parents[1]
    preferred = repo_root / "external" / "reference_data" / "ClosedChannel"
    if preferred.exists():
        return preferred
    return repo_root / "external" / "FreeMHDPaperAllFigures" / "FreeMHDPaperAllFigures" / "ClosedChannel"


def default_fringing_pipe_reference_root(reference_root: str | Path | None = None) -> Path:
    if reference_root is not None:
        return Path(reference_root)
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "external" / "FreeMHDPaperAllFigures" / "FreeMHDPaperAllFigures" / "FringingBPipe"


def _match_single(patterns: list[str], reference_root: Path) -> Path:
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(sorted(reference_root.glob(pattern)))
    if not matches:
        raise FileNotFoundError(f"No reference files matched {patterns} under {reference_root}")
    return matches[0]


def analytical_reference_path(case_kind: str, ha: int, reference_root: str | Path | None = None) -> Path:
    root = default_closed_channel_reference_root(reference_root) / "AnalyticalSolutions"
    stem = case_kind.capitalize()
    patterns = [f"{stem}_Analytical_Ha{ha}_*.txt"]
    return _match_single(patterns, root)


def processed_slice_reference_path(
    case_kind: str,
    ha: int,
    x_slice: str = "1m",
    reference_root: str | Path | None = None,
) -> Path:
    root = default_closed_channel_reference_root(reference_root)
    patterns = [f"{case_kind}_*Ha{ha}_*XSlice{x_slice}_*.csv", f"{case_kind}_Ha{ha}_*XSlice{x_slice}_*.csv"]
    return _match_single(patterns, root)


def _pressure_drop_from_name(path: Path) -> float | None:
    match = re.search(r"PresDrop([0-9.]+)", path.name)
    if match is None:
        return None
    return float(match.group(1).rstrip("."))


def load_closed_channel_analytical(case_kind: str, ha: int, reference_root: str | Path | None = None) -> ClosedChannelAnalyticalReference:
    path = analytical_reference_path(case_kind, ha, reference_root)
    rows = path.read_text().strip().splitlines()
    _, *body = rows
    coordinate = []
    midplane_z = []
    midplane_y = []
    for row in body:
        radius, u1, u2 = row.split()
        coordinate.append(float(radius))
        midplane_z.append(float(u1))
        midplane_y.append(float(u2))
    return ClosedChannelAnalyticalReference(
        case_kind=case_kind,
        ha=ha,
        coordinate=jnp.asarray(coordinate),
        midplane_z=jnp.asarray(midplane_z),
        midplane_y=jnp.asarray(midplane_y),
        pressure_drop=_pressure_drop_from_name(path),
        path=str(path),
    )


def load_shercliff_analytical(ha: int, reference_root: str | Path | None = None) -> ClosedChannelAnalyticalReference:
    return load_closed_channel_analytical("shercliff", ha, reference_root)


def load_hunt_analytical(ha: int, reference_root: str | Path | None = None) -> ClosedChannelAnalyticalReference:
    return load_closed_channel_analytical("hunt", ha, reference_root)


def load_processed_slice(
    case_kind: str,
    ha: int,
    x_slice: str = "1m",
    reference_root: str | Path | None = None,
) -> ProcessedSliceReference:
    path = processed_slice_reference_path(case_kind, ha, x_slice=x_slice, reference_root=reference_root)
    with path.open() as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        columns: dict[str, list[float]] = {name: [] for name in fieldnames}
        for row in reader:
            for name in fieldnames:
                columns[name].append(float(row[name]))
    return ProcessedSliceReference(
        case_kind=case_kind,
        ha=ha,
        columns={name: jnp.asarray(values) for name, values in columns.items()},
        path=str(path),
    )


def _processed_field_column(reference: ProcessedSliceReference, field_name: str, component: int | None) -> jnp.ndarray:
    column_name = field_name if component is None else f"{field_name}:{component}"
    try:
        return reference.columns[column_name]
    except KeyError as exc:
        available = ", ".join(sorted(reference.columns))
        raise KeyError(f"Processed slice {reference.path} has no column {column_name!r}; available columns: {available}") from exc


def _fill_missing_structured_values(grid: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    if not np.isnan(grid).any():
        return grid
    filled = np.array(grid, copy=True)
    for row_index in range(filled.shape[0]):
        row = filled[row_index, :]
        valid = np.isfinite(row)
        if valid.sum() >= 2:
            filled[row_index, :] = np.interp(z, z[valid], row[valid])
        elif valid.sum() == 1:
            filled[row_index, :] = row[valid][0]
    for column_index in range(filled.shape[1]):
        column = filled[:, column_index]
        valid = np.isfinite(column)
        if valid.sum() >= 2:
            filled[:, column_index] = np.interp(y, y[valid], column[valid])
        elif valid.sum() == 1:
            filled[:, column_index] = column[valid][0]
    if np.isnan(filled).any():
        fallback = float(np.nanmean(filled)) if np.isfinite(filled).any() else 0.0
        filled = np.where(np.isfinite(filled), filled, fallback)
    return filled


def processed_slice_field_grid(
    reference: ProcessedSliceReference,
    *,
    field_name: str,
    component: int | None = None,
) -> dict[str, jnp.ndarray]:
    """Return a structured ``(y, z)`` grid from a processed FreeMHD slice.

    ParaView/OpenFOAM slice exports can contain duplicated points at block
    interfaces and may not be ordered as a tensor grid. This helper averages
    duplicate point values before assembling a structured grid for quadrature,
    plotting, and reference-derived flow-rate targets.
    """

    points_y = np.asarray(reference.columns["Points:1"], dtype=float)
    points_z = np.asarray(reference.columns["Points:2"], dtype=float)
    values = np.asarray(_processed_field_column(reference, field_name, component), dtype=float)
    y = np.unique(points_y)
    z = np.unique(points_z)
    y_index = {float(value): index for index, value in enumerate(y)}
    z_index = {float(value): index for index, value in enumerate(z)}
    accumulator = np.zeros((y.size, z.size), dtype=float)
    counts = np.zeros((y.size, z.size), dtype=float)
    for point_y, point_z, value in zip(points_y, points_z, values, strict=True):
        iy = y_index[float(point_y)]
        iz = z_index[float(point_z)]
        accumulator[iy, iz] += float(value)
        counts[iy, iz] += 1.0
    grid = np.divide(accumulator, counts, out=np.full_like(accumulator, np.nan), where=counts > 0.0)
    grid = _fill_missing_structured_values(grid, y, z)
    return {"y": jnp.asarray(y), "z": jnp.asarray(z), "value": jnp.asarray(grid)}


def processed_slice_area_mean(
    reference: ProcessedSliceReference,
    *,
    field_name: str = "U",
    component: int | None = 0,
) -> float:
    """Compute an area-weighted mean over a processed ``y-z`` slice."""

    grid = processed_slice_field_grid(reference, field_name=field_name, component=component)
    y = np.asarray(grid["y"], dtype=float)
    z = np.asarray(grid["z"], dtype=float)
    values = np.asarray(grid["value"], dtype=float)
    if y.size < 2 or z.size < 2:
        return float(np.mean(values)) if values.size else 0.0
    area = (float(y[-1]) - float(y[0])) * (float(z[-1]) - float(z[0]))
    integral_z = np.trapezoid(values, z, axis=1)
    integral = float(np.trapezoid(integral_z, y))
    return integral / area


def processed_slice_point_mesh(
    reference: ProcessedSliceReference,
    *,
    length: float = 1.0,
    nx: int = 1,
) -> StructuredMesh:
    """Build a rectangular mesh whose cross-section faces match slice points."""

    y_faces = jnp.asarray(np.unique(np.asarray(reference.columns["Points:1"], dtype=float)))
    z_faces = jnp.asarray(np.unique(np.asarray(reference.columns["Points:2"], dtype=float)))
    return generate_rect_duct_mesh_from_faces(y_faces=y_faces, z_faces=z_faces, length=length, nx=nx)


def _unique_plane_profile(profile_coord: jnp.ndarray, values: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    coord_np = np.asarray(profile_coord, dtype=float)
    values_np = np.asarray(values, dtype=float)
    unique_coord = np.unique(coord_np)
    unique_values = np.asarray([np.mean(values_np[np.isclose(coord_np, coord)]) for coord in unique_coord])
    order = np.argsort(unique_coord)
    return jnp.asarray(unique_coord[order]), jnp.asarray(unique_values[order])


def _interpolated_centerline_profile(
    profile_coord: jnp.ndarray,
    cross_coord: jnp.ndarray,
    values: jnp.ndarray,
    *,
    tolerance: float = 1.0e-12,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    profile_np = np.asarray(profile_coord, dtype=float)
    cross_np = np.asarray(cross_coord, dtype=float)
    values_np = np.asarray(values, dtype=float)
    unique_cross = np.unique(cross_np)
    if unique_cross.size == 0:
        return jnp.asarray([]), jnp.asarray([])

    exact = unique_cross[np.isclose(unique_cross, 0.0, atol=tolerance, rtol=0.0)]
    if exact.size:
        target = float(exact[np.argmin(np.abs(exact))])
        mask = np.isclose(cross_np, target, atol=tolerance, rtol=0.0)
        return _unique_plane_profile(jnp.asarray(profile_np[mask]), jnp.asarray(values_np[mask]))

    lower_candidates = unique_cross[unique_cross < 0.0]
    upper_candidates = unique_cross[unique_cross > 0.0]
    if lower_candidates.size == 0 or upper_candidates.size == 0:
        target = float(unique_cross[np.argmin(np.abs(unique_cross))])
        mask = np.isclose(cross_np, target, atol=tolerance, rtol=0.0)
        return _unique_plane_profile(jnp.asarray(profile_np[mask]), jnp.asarray(values_np[mask]))

    lower = float(lower_candidates[np.argmax(lower_candidates)])
    upper = float(upper_candidates[np.argmin(upper_candidates)])
    lower_mask = np.isclose(cross_np, lower, atol=tolerance, rtol=0.0)
    upper_mask = np.isclose(cross_np, upper, atol=tolerance, rtol=0.0)
    lower_coord, lower_values = _unique_plane_profile(jnp.asarray(profile_np[lower_mask]), jnp.asarray(values_np[lower_mask]))
    upper_coord, upper_values = _unique_plane_profile(jnp.asarray(profile_np[upper_mask]), jnp.asarray(values_np[upper_mask]))
    coord = np.union1d(np.asarray(lower_coord, dtype=float), np.asarray(upper_coord, dtype=float))
    lower_interp = np.interp(coord, np.asarray(lower_coord, dtype=float), np.asarray(lower_values, dtype=float))
    upper_interp = np.interp(coord, np.asarray(upper_coord, dtype=float), np.asarray(upper_values, dtype=float))
    weight = (0.0 - lower) / (upper - lower)
    return jnp.asarray(coord), jnp.asarray((1.0 - weight) * lower_interp + weight * upper_interp)


def extract_processed_midplane_profile(reference: ProcessedSliceReference, axis: str = "y") -> dict[str, jnp.ndarray]:
    points_y = reference.columns["Points:1"]
    points_z = reference.columns["Points:2"]
    u_x = reference.columns["U:0"]
    pot_e = reference.columns.get("potE", jnp.zeros_like(u_x))

    if axis == "y":
        coordinate, u_values = _interpolated_centerline_profile(points_y, points_z, u_x)
        _, phi_values = _interpolated_centerline_profile(points_y, points_z, pot_e)
        return {
            "y": coordinate,
            "u": u_values,
            "phi": phi_values,
        }
    if axis == "z":
        coordinate, u_values = _interpolated_centerline_profile(points_z, points_y, u_x)
        _, phi_values = _interpolated_centerline_profile(points_z, points_y, pot_e)
        return {
            "z": coordinate,
            "u": u_values,
            "phi": phi_values,
        }
    raise ValueError(f"Unsupported axis {axis}")


def extract_processed_profile(
    reference: ProcessedSliceReference,
    *,
    axis: str,
    field_name: str,
    component: int | None = None,
) -> dict[str, jnp.ndarray]:
    points_y = reference.columns["Points:1"]
    points_z = reference.columns["Points:2"]
    if component is None:
        values = reference.columns[field_name]
    else:
        values = reference.columns[f"{field_name}:{component}"]

    if axis == "y":
        coordinate, profile_values = _interpolated_centerline_profile(points_y, points_z, values)
        return {
            "coordinate": coordinate,
            "value": profile_values,
        }
    if axis == "z":
        coordinate, profile_values = _interpolated_centerline_profile(points_z, points_y, values)
        return {
            "coordinate": coordinate,
            "value": profile_values,
        }
    raise ValueError(f"Unsupported axis {axis}")


def fringing_pipe_profile_reference_path(profile_kind: str, reference_root: str | Path | None = None) -> Path:
    root = default_fringing_pipe_reference_root(reference_root)
    stem = {
        "center": "CenterLine",
        "negative": "NegXLine",
        "positive": "PosXLine",
    }.get(profile_kind)
    if stem is None:
        raise ValueError(f"Unsupported fringing pipe profile kind {profile_kind!r}")
    return _match_single([f"*_{stem}_*.csv"], root)


def load_fringing_pipe_profile(profile_kind: str, reference_root: str | Path | None = None) -> FringingPipeProfileReference:
    path = fringing_pipe_profile_reference_path(profile_kind, reference_root)
    with path.open() as handle:
        reader = csv.DictReader(handle)
        points_z: list[float] = []
        velocity: list[float] = []
        points_x: list[float] = []

        def _field(row: dict[str, str], *names: str) -> str:
            for name in names:
                if name in row and row[name] is not None:
                    return row[name]
            raise KeyError(names[0])

        for row in reader:
            points_z.append(float(_field(row, "Points:2", "Points2")))
            velocity.append(float(_field(row, "U:2", "U2")))
            points_x.append(float(_field(row, "Points:0", "Points0")))
    coordinate = jnp.asarray(points_z)
    coord_scale = jnp.maximum(jnp.max(jnp.abs(coordinate)), 1.0e-12)
    return FringingPipeProfileReference(
        profile_kind=profile_kind,
        coordinate=coordinate / coord_scale,
        velocity=jnp.asarray(velocity),
        x_offset_fraction=float(jnp.mean(jnp.asarray(points_x)) / coord_scale),
        path=str(path),
    )

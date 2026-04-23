from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp


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


def extract_processed_midplane_profile(reference: ProcessedSliceReference, axis: str = "y") -> dict[str, jnp.ndarray]:
    points_y = reference.columns["Points:1"]
    points_z = reference.columns["Points:2"]
    u_x = reference.columns["U:0"]
    pot_e = reference.columns.get("potE", jnp.zeros_like(u_x))

    if axis == "y":
        target = jnp.min(jnp.abs(points_z))
        mask = jnp.isclose(jnp.abs(points_z), target)
        order = jnp.argsort(points_y[mask])
        return {
            "y": points_y[mask][order],
            "u": u_x[mask][order],
            "phi": pot_e[mask][order],
        }
    if axis == "z":
        target = jnp.min(jnp.abs(points_y))
        mask = jnp.isclose(jnp.abs(points_y), target)
        order = jnp.argsort(points_z[mask])
        return {
            "z": points_z[mask][order],
            "u": u_x[mask][order],
            "phi": pot_e[mask][order],
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
        target = jnp.min(jnp.abs(points_z))
        mask = jnp.isclose(jnp.abs(points_z), target)
        order = jnp.argsort(points_y[mask])
        return {
            "coordinate": points_y[mask][order],
            "value": values[mask][order],
        }
    if axis == "z":
        target = jnp.min(jnp.abs(points_y))
        mask = jnp.isclose(jnp.abs(points_y), target)
        order = jnp.argsort(points_z[mask])
        return {
            "coordinate": points_z[mask][order],
            "value": values[mask][order],
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

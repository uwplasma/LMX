from __future__ import annotations

import csv
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp

from .core import Solution
from .freemhd import docker_cli_available, docker_daemon_available
from .mesh import StructuredMesh
from .operators import center_coordinates
from .reference_data import (
    ClosedChannelAnalyticalReference,
    ProcessedSliceReference,
    extract_processed_midplane_profile,
    load_closed_channel_analytical,
    load_processed_slice,
)
from .specs import CaseSpec
from .solvers import solve_transient


@dataclass(frozen=True)
class ValidationReport:
    case_name: str
    metrics: dict[str, float]
    artifacts: dict[str, str]


@dataclass(frozen=True)
class AnalyticComparison:
    coordinate: jnp.ndarray
    simulated: jnp.ndarray
    reference: jnp.ndarray
    l2_error: float
    linf_error: float


@dataclass(frozen=True)
class ProfileSymmetry:
    axis: str
    mean_abs_error: float
    max_abs_error: float


@dataclass(frozen=True)
class ClosedChannelValidation:
    case_kind: str
    ha: int
    y_profile: AnalyticComparison
    z_profile: AnalyticComparison
    reference_pressure_drop: float | None
    reference_path: str


@dataclass(frozen=True)
class ProcessedSliceValidation:
    case_kind: str
    ha: int
    x_slice: str
    y_profile: AnalyticComparison
    z_profile: AnalyticComparison
    reference_path: str


@dataclass(frozen=True)
class FreeMHDCaseInspection:
    case_dir: str
    control_dicts: tuple[str, ...]
    fv_schemes: tuple[str, ...]
    fv_solutions: tuple[str, ...]
    region_properties: tuple[str, ...]
    block_mesh_dicts: tuple[str, ...]
    boundary_field_dirs: tuple[str, ...]
    latest_time_dirs: tuple[str, ...]
    region_zero_dirs: tuple[str, ...]
    zero_field_files: tuple[str, ...]
    processor_layout_dirs: tuple[str, ...]
    parallel_time_dirs: tuple[str, ...]


@dataclass(frozen=True)
class FieldMinMaxRecord:
    time: float
    field: str
    min_value: float
    max_value: float
    min_location: tuple[float, float, float] | None = None
    max_location: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class FreeMHDLineSample:
    path: str
    distance: jnp.ndarray
    pot_e: jnp.ndarray
    u_x: jnp.ndarray
    u_y: jnp.ndarray
    u_z: jnp.ndarray


@dataclass(frozen=True)
class SamplingGeometry:
    x_position: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float


def hartmann_analytic_profile(y: jnp.ndarray, ha: float) -> jnp.ndarray:
    denom = jnp.cosh(ha) - 1.0
    denom = jnp.where(jnp.abs(denom) < 1e-12, 1.0, denom)
    return 1.0 - (jnp.cosh(ha * y) - 1.0) / denom


def extract_centerline(solution: Solution) -> dict[str, jnp.ndarray]:
    mid_z = solution.state.u.shape[1] // 2
    return {
        "y": solution.mesh.y_centers,
        "u": solution.state.u[:, mid_z],
        "phi": solution.state.phi[:, mid_z],
    }


def extract_midplane_profile(solution: Solution, axis: str = "y") -> dict[str, jnp.ndarray]:
    if axis == "y":
        return extract_centerline(solution)
    if axis == "z":
        mid_y = solution.state.u.shape[0] // 2
        return {
            "z": solution.mesh.z_centers,
            "u": solution.state.u[mid_y, :],
            "phi": solution.state.phi[mid_y, :],
        }
    raise ValueError(f"Unsupported axis {axis}")


def compare_profile_to_reference(
    coordinate: jnp.ndarray,
    simulated: jnp.ndarray,
    reference: jnp.ndarray,
) -> AnalyticComparison:
    diff = simulated - reference
    l2 = float(jnp.sqrt(jnp.mean(diff**2)))
    linf = float(jnp.max(jnp.abs(diff)))
    return AnalyticComparison(
        coordinate=coordinate,
        simulated=simulated,
        reference=reference,
        l2_error=l2,
        linf_error=linf,
    )


def compare_normalized_profiles(
    simulated_coordinate: jnp.ndarray,
    simulated: jnp.ndarray,
    reference_coordinate: jnp.ndarray,
    reference: jnp.ndarray,
) -> AnalyticComparison:
    sim_coord = simulated_coordinate / jnp.max(jnp.abs(simulated_coordinate))
    ref_coord = reference_coordinate / jnp.max(jnp.abs(reference_coordinate))
    sim_scale = jnp.max(jnp.abs(simulated))
    ref_scale = jnp.max(jnp.abs(reference))
    sim_scale = jnp.where(sim_scale > 0.0, sim_scale, 1.0)
    ref_scale = jnp.where(ref_scale > 0.0, ref_scale, 1.0)
    normalized_simulated = simulated / sim_scale
    normalized_reference = reference / ref_scale
    interpolated_reference = jnp.interp(sim_coord, ref_coord, normalized_reference)
    return compare_profile_to_reference(sim_coord, normalized_simulated, interpolated_reference)


def symmetry_metrics(profile: jnp.ndarray, axis: str) -> ProfileSymmetry:
    mirrored = jnp.flip(profile)
    diff = profile - mirrored
    return ProfileSymmetry(
        axis=axis,
        mean_abs_error=float(jnp.mean(jnp.abs(diff))),
        max_abs_error=float(jnp.max(jnp.abs(diff))),
    )


def duct_profile_metrics(solution: Solution) -> dict[str, float]:
    y_profile = extract_midplane_profile(solution, axis="y")["u"]
    z_profile = extract_midplane_profile(solution, axis="z")["u"]
    y_sym = symmetry_metrics(y_profile, axis="y")
    z_sym = symmetry_metrics(z_profile, axis="z")
    return {
        "symmetry_y_mean_abs_error": y_sym.mean_abs_error,
        "symmetry_y_max_abs_error": y_sym.max_abs_error,
        "symmetry_z_mean_abs_error": z_sym.mean_abs_error,
        "symmetry_z_max_abs_error": z_sym.max_abs_error,
        "u_max": float(jnp.max(solution.state.u)),
        "u_mean": float(jnp.mean(solution.state.u)),
    }


def validation_summary(solution: Solution, case_name: str, ha: float | None = None) -> dict[str, float | str]:
    payload: dict[str, float | str] = {
        "case": case_name,
        "time": solution.state.time,
        "residual": solution.state.residual,
    }
    payload.update(duct_profile_metrics(solution))
    if case_name.startswith("hartmann") and ha is not None:
        comparison = hartmann_validation(solution, ha)
        payload["l2_error"] = comparison.l2_error
        payload["linf_error"] = comparison.linf_error
    return payload


def hartmann_validation(solution: Solution, ha: float) -> AnalyticComparison:
    profile = extract_centerline(solution)
    coordinate = profile["y"] / jnp.max(jnp.abs(profile["y"]))
    u = profile["u"]
    scale = jnp.max(jnp.abs(u))
    scale = jnp.where(scale > 0.0, scale, 1.0)
    normalized = u / scale
    reference = hartmann_analytic_profile(coordinate, ha)
    return compare_profile_to_reference(coordinate, normalized, reference)


def closed_channel_validation(
    solution: Solution,
    case_kind: str,
    ha: int,
    reference_root: str | Path | None = None,
) -> ClosedChannelValidation:
    reference: ClosedChannelAnalyticalReference = load_closed_channel_analytical(case_kind, ha, reference_root)
    y_profile = extract_midplane_profile(solution, axis="y")
    z_profile = extract_midplane_profile(solution, axis="z")
    y_comparison = compare_normalized_profiles(
        y_profile["y"],
        y_profile["u"],
        reference.coordinate,
        reference.midplane_y,
    )
    z_comparison = compare_normalized_profiles(
        z_profile["z"],
        z_profile["u"],
        reference.coordinate,
        reference.midplane_z,
    )
    return ClosedChannelValidation(
        case_kind=case_kind,
        ha=ha,
        y_profile=y_comparison,
        z_profile=z_comparison,
        reference_pressure_drop=reference.pressure_drop,
        reference_path=reference.path,
    )


def processed_slice_validation(
    solution: Solution,
    case_kind: str,
    ha: int,
    x_slice: str = "1m",
    reference_root: str | Path | None = None,
) -> ProcessedSliceValidation:
    reference: ProcessedSliceReference = load_processed_slice(
        case_kind,
        ha,
        x_slice=x_slice,
        reference_root=reference_root,
    )
    y_profile = extract_midplane_profile(solution, axis="y")
    z_profile = extract_midplane_profile(solution, axis="z")
    reference_y = extract_processed_midplane_profile(reference, axis="y")
    reference_z = extract_processed_midplane_profile(reference, axis="z")
    y_comparison = compare_normalized_profiles(
        y_profile["y"],
        y_profile["u"],
        reference_y["y"],
        reference_y["u"],
    )
    z_comparison = compare_normalized_profiles(
        z_profile["z"],
        z_profile["u"],
        reference_z["z"],
        reference_z["u"],
    )
    return ProcessedSliceValidation(
        case_kind=case_kind,
        ha=ha,
        x_slice=x_slice,
        y_profile=y_comparison,
        z_profile=z_comparison,
        reference_path=reference.path,
    )


def write_profile_csv(path: str | Path, data: dict[str, jnp.ndarray]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(data.keys())
    rows = zip(*(jnp.asarray(data[key]).tolist() for key in keys))
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(keys)
        writer.writerows(rows)
    return path


def freemhd_case_command(case_dir: str | Path, cores: int = 4, solver: str = "epotMultiRegionFoam") -> str:
    return (
        "bash -lc 'source /opt/OpenFOAM/OpenFOAM-v2206/etc/bashrc && "
        f"cd {Path(case_dir)} && "
        f"mpirun -np {cores} {solver} -parallel'"
    )


def compare_with_freemhd(case_spec: CaseSpec, freemhd_run_dir: str | Path) -> ValidationReport:
    run_dir = Path(freemhd_run_dir)
    inspection = inspect_freemhd_case(run_dir)
    expected_region_count = float(len(case_spec.regions))
    expected_solid_count = float(sum(1 for region in case_spec.regions if region.kind == "solid"))
    minmax_files = tuple(sorted(str(path.relative_to(run_dir)) for path in run_dir.glob("postProcessing/**/fieldMinMax.dat")))
    sampled_profiles = latest_sampled_profiles(run_dir)
    y_sample_path = sampled_profiles[0].path if sampled_profiles is not None else ""
    z_sample_path = sampled_profiles[1].path if sampled_profiles is not None else ""
    metrics = {
        "run_dir_exists": float(run_dir.exists()),
        "has_system": float((run_dir / "system").exists()),
        "has_constant": float((run_dir / "constant").exists()),
        "has_zero_dir": float((run_dir / "0").exists()),
        "control_dict_count": float(len(inspection.control_dicts)),
        "region_properties_count": float(len(inspection.region_properties)),
        "latest_time_count": float(len(inspection.latest_time_dirs)),
        "region_zero_dir_count": float(len(inspection.region_zero_dirs)),
        "zero_field_file_count": float(len(inspection.zero_field_files)),
        "processor_layout_count": float(len(inspection.processor_layout_dirs)),
        "parallel_time_count": float(len(inspection.parallel_time_dirs)),
        "has_potE_zero_field": float(any(path.endswith("/potE") for path in inspection.zero_field_files)),
        "has_velocity_zero_field": float(any(path.endswith("/U") for path in inspection.zero_field_files)),
        "expected_region_count": expected_region_count,
        "expected_solid_region_count": expected_solid_count,
        "field_minmax_file_count": float(len(minmax_files)),
        "sampled_profile_pair_available": float(sampled_profiles is not None),
    }
    latest_u_record = latest_field_minmax_record(run_dir, field="mag(U)")
    lmx_solution: Solution | None = None
    if latest_u_record is not None:
        lmx_solution = solve_transient(case_spec)
        lmx_u_max = float(jnp.max(jnp.abs(lmx_solution.state.u)))
        metrics["freemhd_latest_time"] = latest_u_record.time
        metrics["freemhd_u_max_latest"] = latest_u_record.max_value
        metrics["lmx_u_max"] = lmx_u_max
        metrics["u_max_abs_diff"] = abs(lmx_u_max - latest_u_record.max_value)
    if sampled_profiles is not None:
        if lmx_solution is None:
            lmx_solution = solve_transient(case_spec)
        y_sample, z_sample = sampled_profiles
        y_profile = extract_midplane_profile(lmx_solution, axis="y")
        z_profile = extract_midplane_profile(lmx_solution, axis="z")
        y_comparison = compare_normalized_profiles(
            y_profile["y"],
            y_profile["u"],
            normalize_sample_distance(y_sample.distance),
            y_sample.u_x,
        )
        z_comparison = compare_normalized_profiles(
            z_profile["z"],
            z_profile["u"],
            normalize_sample_distance(z_sample.distance),
            z_sample.u_x,
        )
        metrics["freemhd_sample_time"] = infer_sample_time_from_path(y_sample.path)
        metrics["freemhd_sample_y_l2_error"] = y_comparison.l2_error
        metrics["freemhd_sample_y_linf_error"] = y_comparison.linf_error
        metrics["freemhd_sample_z_l2_error"] = z_comparison.l2_error
        metrics["freemhd_sample_z_linf_error"] = z_comparison.linf_error
    artifacts = {
        "freemhd_run_dir": str(run_dir),
        "expected_command": freemhd_case_command(run_dir),
        "control_dicts": json.dumps(inspection.control_dicts),
        "region_properties": json.dumps(inspection.region_properties),
        "block_mesh_dicts": json.dumps(inspection.block_mesh_dicts),
        "boundary_field_dirs": json.dumps(inspection.boundary_field_dirs),
        "latest_time_dirs": json.dumps(inspection.latest_time_dirs),
        "region_zero_dirs": json.dumps(inspection.region_zero_dirs),
        "zero_field_files": json.dumps(inspection.zero_field_files),
        "processor_layout_dirs": json.dumps(inspection.processor_layout_dirs),
        "parallel_time_dirs": json.dumps(inspection.parallel_time_dirs),
        "field_minmax_files": json.dumps(minmax_files),
        "sampled_profile_y_path": y_sample_path,
        "sampled_profile_z_path": z_sample_path,
    }
    return ValidationReport(case_name=case_spec.name, metrics=metrics, artifacts=artifacts)


def write_validation_report(report: ValidationReport, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "case_name": report.case_name,
        "metrics": report.metrics,
        "artifacts": report.artifacts,
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def write_analytic_comparison(comparison: AnalyticComparison, path: str | Path, axis_name: str = "coordinate") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        axis_name: jnp.asarray(comparison.coordinate).tolist(),
        "simulated": jnp.asarray(comparison.simulated).tolist(),
        "reference": jnp.asarray(comparison.reference).tolist(),
        "l2_error": comparison.l2_error,
        "linf_error": comparison.linf_error,
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def write_closed_channel_validation(report: ClosedChannelValidation, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "case_kind": report.case_kind,
        "ha": report.ha,
        "reference_pressure_drop": report.reference_pressure_drop,
        "reference_path": report.reference_path,
        "y_profile": {
            "coordinate": jnp.asarray(report.y_profile.coordinate).tolist(),
            "simulated": jnp.asarray(report.y_profile.simulated).tolist(),
            "reference": jnp.asarray(report.y_profile.reference).tolist(),
            "l2_error": report.y_profile.l2_error,
            "linf_error": report.y_profile.linf_error,
        },
        "z_profile": {
            "coordinate": jnp.asarray(report.z_profile.coordinate).tolist(),
            "simulated": jnp.asarray(report.z_profile.simulated).tolist(),
            "reference": jnp.asarray(report.z_profile.reference).tolist(),
            "l2_error": report.z_profile.l2_error,
            "linf_error": report.z_profile.linf_error,
        },
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def write_processed_slice_validation(report: ProcessedSliceValidation, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "case_kind": report.case_kind,
        "ha": report.ha,
        "x_slice": report.x_slice,
        "reference_path": report.reference_path,
        "y_profile": {
            "coordinate": jnp.asarray(report.y_profile.coordinate).tolist(),
            "simulated": jnp.asarray(report.y_profile.simulated).tolist(),
            "reference": jnp.asarray(report.y_profile.reference).tolist(),
            "l2_error": report.y_profile.l2_error,
            "linf_error": report.y_profile.linf_error,
        },
        "z_profile": {
            "coordinate": jnp.asarray(report.z_profile.coordinate).tolist(),
            "simulated": jnp.asarray(report.z_profile.simulated).tolist(),
            "reference": jnp.asarray(report.z_profile.reference).tolist(),
            "l2_error": report.z_profile.l2_error,
            "linf_error": report.z_profile.linf_error,
        },
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def write_metrics_json(metrics: dict[str, float | str], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2))
    return path


def docker_available() -> bool:
    return docker_daemon_available()


def read_field_minmax(path: str | Path) -> tuple[FieldMinMaxRecord, ...]:
    records: list[FieldMinMaxRecord] = []
    for raw_line in Path(path).read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(
            r"^(?P<time>\S+)\s+"
            r"(?P<field>\S+)\s+"
            r"(?P<min>\S+)\s+"
            r"(?P<min_loc>\([^)]+\))\s+\S+\s+"
            r"(?P<max>\S+)\s+"
            r"(?P<max_loc>\([^)]+\))\s+\S+\s*$",
            line,
        )
        if match is None:
            continue
        min_location = parse_location_tuple(match.group("min_loc"))
        max_location = parse_location_tuple(match.group("max_loc"))
        records.append(
            FieldMinMaxRecord(
                time=float(match.group("time")),
                field=match.group("field"),
                min_value=float(match.group("min")),
                max_value=float(match.group("max")),
                min_location=min_location,
                max_location=max_location,
            )
        )
    return tuple(records)


def latest_field_minmax_record(run_dir: str | Path, field: str = "mag(U)") -> FieldMinMaxRecord | None:
    root = Path(run_dir)
    latest: FieldMinMaxRecord | None = None
    for path in root.glob("postProcessing/**/fieldMinMax.dat"):
        for record in read_field_minmax(path):
            if record.field != field:
                continue
            if latest is None or record.time > latest.time:
                latest = record
    return latest


def parse_location_tuple(text: str) -> tuple[float, float, float] | None:
    match = re.match(r"^\(\s*(\S+)\s+(\S+)\s+(\S+)\s*\)$", text.strip())
    if match is None:
        return None
    return (float(match.group(1)), float(match.group(2)), float(match.group(3)))


def infer_sampling_geometry(run_dir: str | Path, field: str = "mag(U)") -> SamplingGeometry:
    latest = latest_field_minmax_record(run_dir, field=field)
    if latest is None or latest.min_location is None or latest.max_location is None:
        raise ValueError(f"Unable to infer sampling geometry from {run_dir}")
    x_position = latest.max_location[0]
    y_extent = max(abs(latest.min_location[1]), abs(latest.max_location[1]))
    z_extent = max(abs(latest.min_location[2]), abs(latest.max_location[2]))
    return SamplingGeometry(
        x_position=x_position,
        y_min=-y_extent,
        y_max=y_extent,
        z_min=-z_extent,
        z_max=z_extent,
    )


def read_freemhd_xy_sample(path: str | Path) -> FreeMHDLineSample:
    distance = []
    pot_e = []
    u_x = []
    u_y = []
    u_z = []
    for raw_line in Path(path).read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        distance.append(float(parts[0]))
        pot_e.append(float(parts[1]))
        u_x.append(float(parts[2]))
        u_y.append(float(parts[3]))
        u_z.append(float(parts[4]))
    return FreeMHDLineSample(
        path=str(Path(path)),
        distance=jnp.asarray(distance, dtype=float),
        pot_e=jnp.asarray(pot_e, dtype=float),
        u_x=jnp.asarray(u_x, dtype=float),
        u_y=jnp.asarray(u_y, dtype=float),
        u_z=jnp.asarray(u_z, dtype=float),
    )


def infer_sample_time_from_path(path: str | Path) -> float:
    sample_path = Path(path)
    for parent in sample_path.parents:
        try:
            return float(parent.name)
        except ValueError:
            continue
    raise ValueError(f"Unable to infer sample time from path {sample_path}")


def normalize_sample_distance(distance: jnp.ndarray) -> jnp.ndarray:
    max_distance = jnp.max(distance)
    max_distance = jnp.where(max_distance > 0.0, max_distance, 1.0)
    return 2.0 * distance / max_distance - 1.0


def latest_sampled_profiles(run_dir: str | Path) -> tuple[FreeMHDLineSample, FreeMHDLineSample] | None:
    root = Path(run_dir)
    candidates = sorted(root.glob("postProcessing/*/liquid/*/centerlineY_potE_U.xy"))
    latest_y_path: Path | None = None
    latest_time: float | None = None
    for path in candidates:
        try:
            sample_time = infer_sample_time_from_path(path)
        except ValueError:
            continue
        if latest_time is None or sample_time > latest_time:
            latest_time = sample_time
            latest_y_path = path
    if latest_y_path is None:
        return None
    z_path = latest_y_path.with_name("centerlineZ_potE_U.xy")
    if not z_path.exists():
        return None
    return read_freemhd_xy_sample(latest_y_path), read_freemhd_xy_sample(z_path)


def inspect_freemhd_case(case_dir: str | Path) -> FreeMHDCaseInspection:
    root = Path(case_dir)
    if not root.exists():
        return FreeMHDCaseInspection(
            case_dir=str(root),
            control_dicts=(),
            fv_schemes=(),
            fv_solutions=(),
            region_properties=(),
            block_mesh_dicts=(),
            boundary_field_dirs=(),
            latest_time_dirs=(),
            region_zero_dirs=(),
            zero_field_files=(),
            processor_layout_dirs=(),
            parallel_time_dirs=(),
        )

    def _relative_matches(pattern: str) -> tuple[str, ...]:
        return tuple(sorted(str(path.relative_to(root)) for path in root.glob(pattern)))

    def _numeric_time_dirs() -> tuple[str, ...]:
        matches: list[str] = []
        for path in root.iterdir():
            if not path.is_dir():
                continue
            if path.name in {"0", "constant", "system", "processor0"}:
                continue
            try:
                float(path.name)
            except ValueError:
                continue
            matches.append(str(path.relative_to(root)))
        return tuple(sorted(matches, key=float))

    def _region_zero_dirs() -> tuple[str, ...]:
        zero_root = root / "0"
        if not zero_root.is_dir():
            return ()
        return tuple(sorted(str(path.relative_to(root)) for path in zero_root.iterdir() if path.is_dir()))

    def _zero_field_files() -> tuple[str, ...]:
        zero_root = root / "0"
        if not zero_root.is_dir():
            return ()
        matches: list[str] = []
        for region_dir in zero_root.iterdir():
            if not region_dir.is_dir():
                continue
            for field_path in region_dir.iterdir():
                if field_path.is_file():
                    matches.append(str(field_path.relative_to(root)))
        return tuple(sorted(matches))

    def _processor_layout_dirs() -> tuple[str, ...]:
        return tuple(sorted(str(path.relative_to(root)) for path in root.iterdir() if path.is_dir() and path.name.startswith("processors")))

    def _parallel_time_dirs() -> tuple[str, ...]:
        matches: list[str] = []
        for processor_root in root.iterdir():
            if not processor_root.is_dir() or not processor_root.name.startswith("processors"):
                continue
            for path in processor_root.iterdir():
                if not path.is_dir():
                    continue
                try:
                    float(path.name)
                except ValueError:
                    continue
                matches.append(str(path.relative_to(root)))
        return tuple(sorted(matches, key=lambda value: float(Path(value).name)))

    return FreeMHDCaseInspection(
        case_dir=str(root),
        control_dicts=_relative_matches("**/system/controlDict"),
        fv_schemes=_relative_matches("**/system/fvSchemes"),
        fv_solutions=_relative_matches("**/system/fvSolution"),
        region_properties=_relative_matches("**/constant/regionProperties"),
        block_mesh_dicts=_relative_matches("**/system/blockMeshDict"),
        boundary_field_dirs=_relative_matches("0")
        + _relative_matches("**/0"),
        latest_time_dirs=_numeric_time_dirs(),
        region_zero_dirs=_region_zero_dirs(),
        zero_field_files=_zero_field_files(),
        processor_layout_dirs=_processor_layout_dirs(),
        parallel_time_dirs=_parallel_time_dirs(),
    )


def run_freemhd_container(
    image: str,
    case_dir: str | Path,
    cores: int = 4,
    solver: str = "epotMultiRegionFoam",
) -> subprocess.CompletedProcess[str]:
    command = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{Path(case_dir).resolve()}:/workspace/case",
        image,
        "bash",
        "-lc",
        freemhd_case_command("/workspace/case", cores=cores, solver=solver),
    ]
    return subprocess.run(command, text=True, capture_output=True, check=False)

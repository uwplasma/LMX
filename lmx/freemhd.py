from __future__ import annotations

import os
import re
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np

from .cases import make_hartmann_case, make_hunt_case, make_shercliff_case
from .specs import BoundaryCondition, CaseSpec


def candidate_u_paths(case_dir: str | Path) -> list[Path]:
    root = Path(case_dir)
    return [
        root / "case" / "0" / "liquid" / "U",
        root / "case" / "0" / "fluid" / "U",
        root / "case" / "0" / "U",
        root / "0" / "liquid" / "U",
        root / "0" / "fluid" / "U",
        root / "0" / "U",
        root / "latestTime" / "liquid" / "U",
        root / "latestTime" / "fluid" / "U",
        root / "latestTime" / "U",
    ]


def _candidate_paths(case_dir: str | Path, *relative_paths: str) -> list[Path]:
    root = Path(case_dir)
    return [root / relative for relative in relative_paths]


def _first_existing(case_dir: str | Path, *relative_paths: str) -> Path | None:
    for path in _candidate_paths(case_dir, *relative_paths):
        if path.exists():
            return path
    return None


def _extract_first_scalar(text: str, *patterns: str) -> float | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.MULTILINE)
        if match is not None:
            return float(match.group(1))
    return None


def infer_initial_velocity_x(case_dir: str | Path) -> float | None:
    pattern = re.compile(r"internalField\s+uniform\s+\(\s*(\S+)\s+\S+\s+\S+\s*\)")
    for path in candidate_u_paths(case_dir):
        if not path.exists():
            continue
        text = path.read_text()
        match = pattern.search(text)
        if match is not None:
            return float(match.group(1))
    return None


def _extract_inlet_block(text: str) -> str | None:
    boundary_match = re.search(r"boundaryField\s*\{", text)
    if boundary_match is None:
        return None
    boundary_text = text[boundary_match.end() :]
    inlet_match = re.search(r"\binlet\b\s*\{", boundary_text)
    if inlet_match is None:
        return None
    start = inlet_match.end()
    depth = 1
    index = start
    while index < len(boundary_text) and depth > 0:
        char = boundary_text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        index += 1
    if depth != 0:
        return None
    return boundary_text[start : index - 1]


def infer_inlet_flow_rate(case_dir: str | Path) -> float | None:
    pattern = re.compile(r"volumetricFlowRate\s+(?:constant\s+)?([0-9eE+.\-]+)\s*;")
    for path in candidate_u_paths(case_dir):
        if not path.exists():
            continue
        inlet_block = _extract_inlet_block(path.read_text())
        if inlet_block is None:
            continue
        match = pattern.search(inlet_block)
        if match is not None:
            return float(match.group(1))
    return None


def summarize_observable_offenders(
    records: list[dict[str, object]],
    *,
    l2_target: float = 1.0e-2,
    min_reference_peak_fraction: float = 1.0e-3,
    top_n: int | None = None,
) -> list[dict[str, object]]:
    offenders: list[dict[str, object]] = []
    for record in records:
        observables = record.get("observables", {})
        if not isinstance(observables, dict):
            continue
        for observable_name, observable_payload in observables.items():
            if not isinstance(observable_payload, dict):
                continue
            observable_reference_peak = max(
                (
                    float(cut.get("reference_peak_abs", 1.0))
                    for axis in ("y", "z")
                    if isinstance((cut := observable_payload.get(axis)), dict)
                ),
                default=1.0,
            )
            for axis in ("y", "z"):
                cut = observable_payload.get(axis)
                if not isinstance(cut, dict):
                    continue
                l2_error = float(cut.get("l2_error", 0.0))
                linf_error = float(cut.get("linf_error", 0.0))
                peak_ratio = float(cut.get("peak_ratio", observable_payload.get("peak_ratio", 1.0)))
                reference_peak_abs = float(cut.get("reference_peak_abs", observable_reference_peak))
                reference_peak_fraction = reference_peak_abs / max(observable_reference_peak, 1.0e-20)
                low_signal = reference_peak_fraction < float(min_reference_peak_fraction)
                status = "low_signal" if low_signal else ("pass" if l2_error <= l2_target else "offender")
                offenders.append(
                    {
                        "case_kind": str(record.get("case_kind", "")),
                        "drive_mode": str(record.get("drive_mode", "")),
                        "observable": str(observable_name),
                        "axis": axis,
                        "l2_error": l2_error,
                        "linf_error": linf_error,
                        "peak_ratio": peak_ratio,
                        "reference_peak_abs": reference_peak_abs,
                        "reference_peak_fraction": reference_peak_fraction,
                        "l2_target": float(l2_target),
                        "target_ratio": l2_error / max(float(l2_target), 1.0e-20),
                        "status": status,
                    }
                )
    status_rank = {"offender": 2, "pass": 1, "low_signal": 0}
    offenders.sort(
        key=lambda item: (
            status_rank.get(str(item["status"]), 0),
            float(item["target_ratio"]),
            float(item["linf_error"]),
        ),
        reverse=True,
    )
    if top_n is not None:
        return offenders[: max(0, int(top_n))]
    return offenders


def summarize_observable_gate(
    records: list[dict[str, object]],
    *,
    l2_target: float = 1.0e-2,
    required_observables: tuple[str, ...] = ("velocity", "potential", "current", "lorentz"),
    required_axes: tuple[str, ...] = ("y", "z"),
    min_reference_peak_fraction: float = 1.0e-3,
) -> dict[str, object]:
    """Summarize whether a FreeMHD parity artifact has the required observables.

    The gate is intentionally based on physical outputs rather than image
    similarity: each case must carry the requested midplane cuts and every
    non-low-signal cut must stay below the configured normalized L2 target.
    """

    missing: list[dict[str, str]] = []
    for record in records:
        case_kind = str(record.get("case_kind", ""))
        observables = record.get("observables", {})
        if not isinstance(observables, dict):
            observables = {}
        for observable_name in required_observables:
            payload = observables.get(observable_name)
            if not isinstance(payload, dict):
                missing.append({"case_kind": case_kind, "observable": observable_name, "axis": "*"})
                continue
            for axis in required_axes:
                if not isinstance(payload.get(axis), dict):
                    missing.append({"case_kind": case_kind, "observable": observable_name, "axis": axis})

    ranked = summarize_observable_offenders(
        records,
        l2_target=l2_target,
        min_reference_peak_fraction=min_reference_peak_fraction,
    )
    offender_count = sum(1 for item in ranked if item["status"] == "offender")
    low_signal_count = sum(1 for item in ranked if item["status"] == "low_signal")
    pass_count = sum(1 for item in ranked if item["status"] == "pass")
    return {
        "case_count": len(records),
        "cases": sorted(str(record.get("case_kind", "")) for record in records),
        "l2_target": float(l2_target),
        "required_observables": list(required_observables),
        "required_axes": list(required_axes),
        "observable_pass_count": pass_count,
        "observable_offender_count": offender_count,
        "low_signal_count": low_signal_count,
        "missing_observable_count": len(missing),
        "missing_observables": missing,
        "top_observable_offenders": ranked[:8],
        "research_grade_validation_pass": offender_count == 0 and len(missing) == 0,
    }


def side_jet_profile_metrics(
    coordinate: object,
    values: object,
    *,
    center_exclusion_fraction: float = 0.02,
) -> dict[str, float]:
    """Return side-jet peak locations and amplitudes for a Hunt-style profile."""

    coord = np.asarray(coordinate, dtype=float)
    value = np.asarray(values, dtype=float)
    if coord.size == 0 or value.size == 0:
        return {
            "negative_location": 0.0,
            "positive_location": 0.0,
            "negative_value": 0.0,
            "positive_value": 0.0,
            "center_value": 0.0,
            "peak_value": 0.0,
            "peak_to_center_ratio": 0.0,
        }
    order = np.argsort(coord)
    coord = coord[order]
    value = value[order]
    half_width = max(float(np.max(np.abs(coord))), 1.0e-20)
    center_cut = float(center_exclusion_fraction) * half_width
    negative_mask = coord <= -center_cut
    positive_mask = coord >= center_cut
    if not negative_mask.any():
        negative_mask = coord <= 0.0
    if not positive_mask.any():
        positive_mask = coord >= 0.0

    negative_indices = np.flatnonzero(negative_mask)
    positive_indices = np.flatnonzero(positive_mask)
    negative_index = int(negative_indices[np.argmax(value[negative_indices])]) if negative_indices.size else int(np.argmax(value))
    positive_index = int(positive_indices[np.argmax(value[positive_indices])]) if positive_indices.size else int(np.argmax(value))
    center_value = float(np.interp(0.0, coord, value))
    peak_value = float(max(value[negative_index], value[positive_index]))
    return {
        "negative_location": float(coord[negative_index]),
        "positive_location": float(coord[positive_index]),
        "negative_value": float(value[negative_index]),
        "positive_value": float(value[positive_index]),
        "center_value": center_value,
        "peak_value": peak_value,
        "peak_to_center_ratio": peak_value / max(abs(center_value), 1.0e-20),
    }


def compare_side_jet_profiles(
    simulated_coordinate: object,
    simulated_values: object,
    reference_coordinate: object,
    reference_values: object,
) -> dict[str, object]:
    """Compare Hunt side-jet observables between a simulation and reference cut."""

    simulated = side_jet_profile_metrics(simulated_coordinate, simulated_values)
    reference = side_jet_profile_metrics(reference_coordinate, reference_values)
    location_scale = max(
        abs(float(reference["negative_location"])),
        abs(float(reference["positive_location"])),
        1.0e-20,
    )
    peak_scale = max(abs(float(reference["peak_value"])), 1.0e-20)
    return {
        "simulated": simulated,
        "reference": reference,
        "negative_location_error": abs(float(simulated["negative_location"]) - float(reference["negative_location"])),
        "positive_location_error": abs(float(simulated["positive_location"]) - float(reference["positive_location"])),
        "normalized_location_error": max(
            abs(float(simulated["negative_location"]) - float(reference["negative_location"])),
            abs(float(simulated["positive_location"]) - float(reference["positive_location"])),
        )
        / location_scale,
        "peak_value_relative_error": abs(float(simulated["peak_value"]) - float(reference["peak_value"])) / peak_scale,
        "peak_to_center_ratio_error": abs(
            float(simulated["peak_to_center_ratio"]) - float(reference["peak_to_center_ratio"])
        )
        / max(abs(float(reference["peak_to_center_ratio"])), 1.0e-20),
    }


def summarize_profile_error_offenders(
    records: list[dict[str, object]],
    *,
    l2_target: float = 1.0e-2,
    top_n: int | None = None,
) -> list[dict[str, object]]:
    offenders: list[dict[str, object]] = []
    for record in records:
        for axis in ("y", "z"):
            key = f"{axis}_l2_error"
            if key not in record:
                continue
            l2_error = float(record[key])
            offenders.append(
                {
                    "case_kind": str(record.get("case_kind", "")),
                    "axis": axis,
                    "l2_error": l2_error,
                    "l2_target": float(l2_target),
                    "target_ratio": l2_error / max(float(l2_target), 1.0e-20),
                    "status": "pass" if l2_error <= l2_target else "offender",
                }
            )
    offenders.sort(key=lambda item: float(item["target_ratio"]), reverse=True)
    if top_n is not None:
        return offenders[: max(0, int(top_n))]
    return offenders


def summarize_runtime_offenders(records: list[dict[str, object]]) -> list[dict[str, object]]:
    offenders: list[dict[str, object]] = []
    for record in records:
        freemhd_seconds = float(record.get("freemhd_execution_seconds", 0.0) or 0.0)
        lmx_seconds = float(record.get("lmx_execution_seconds", 0.0) or 0.0)
        if freemhd_seconds <= 0.0 or lmx_seconds <= 0.0:
            continue
        offenders.append(
            {
                "case_kind": str(record.get("case_kind", "")),
                "freemhd_execution_seconds": freemhd_seconds,
                "lmx_execution_seconds": lmx_seconds,
                "lmx_to_freemhd_runtime_ratio": lmx_seconds / freemhd_seconds,
                "status": "pass" if lmx_seconds <= freemhd_seconds else "offender",
            }
        )
    offenders.sort(key=lambda item: float(item["lmx_to_freemhd_runtime_ratio"]), reverse=True)
    return offenders


def infer_reduced_inlet_flow_rate(
    case_dir: str | Path,
    *,
    reduced_area: float,
    initial_velocity: float | None = None,
) -> float | None:
    recovered_flow_rate = infer_inlet_flow_rate(case_dir)
    if recovered_flow_rate is None:
        return None
    speed = infer_initial_velocity_x(case_dir) if initial_velocity is None else initial_velocity
    if speed is None or abs(speed) <= 1.0e-20:
        return None
    recovered_area = recovered_flow_rate / speed
    if abs(recovered_area) <= 1.0e-20:
        return None
    return recovered_flow_rate * (reduced_area / recovered_area)


def infer_inlet_drive_mode(case_dir: str | Path) -> str | None:
    type_pattern = re.compile(r"type\s+(\S+)\s*;")
    for path in candidate_u_paths(case_dir):
        if not path.exists():
            continue
        inlet_block = _extract_inlet_block(path.read_text())
        if inlet_block is None:
            continue
        match = type_pattern.search(inlet_block)
        if match is None:
            continue
        inlet_type = match.group(1)
        if inlet_type == "flowRateInletVelocity":
            return "inlet_flow_rate"
        return "inlet_velocity"
    return None


def infer_liquid_properties(case_dir: str | Path) -> tuple[float, float, float] | None:
    path = _first_existing(
        case_dir,
        "case/constant/liquid/thermophysicalProperties.liquidMetal",
        "constant/liquid/thermophysicalProperties.liquidMetal",
        "case/constant/liquid/thermophysicalProperties",
        "constant/liquid/thermophysicalProperties",
    )
    if path is None:
        return None
    text = path.read_text()
    conductivity = _extract_first_scalar(text, r"\belcond\s+(?:\[[^\]]*\])?\s*([0-9eE+.\-]+)\s*;")
    if conductivity is None:
        conductivity = _extract_first_scalar(text, r"\bsigma\s+(?:\[[^\]]*\])?\s*([0-9eE+.\-]+)\s*;")
    density = _extract_first_scalar(text, r"\brho\s+([0-9eE+.\-]+)\s*;")
    viscosity = _extract_first_scalar(text, r"\bmu\s+([0-9eE+.\-]+)\s*;")
    if conductivity is None or density is None or viscosity is None:
        return None
    return conductivity, density, viscosity


def infer_solid_conductivities(case_dir: str | Path) -> tuple[float | None, float | None]:
    solid_path = _first_existing(
        case_dir,
        "case/constant/solidWalls/thermophysicalProperties",
        "constant/solidWalls/thermophysicalProperties",
    )
    insulator_path = _first_existing(
        case_dir,
        "case/constant/insulator/thermophysicalProperties",
        "constant/insulator/thermophysicalProperties",
    )
    solid_conductivity = None
    insulator_conductivity = None
    if solid_path is not None:
        solid_conductivity = _extract_first_scalar(
            solid_path.read_text(),
            r"\belcond\s+(?:\[[^\]]*\])?\s*([0-9eE+.\-]+)\s*;",
        )
    if insulator_path is not None:
        insulator_conductivity = _extract_first_scalar(
            insulator_path.read_text(),
            r"\belcond\s+(?:\[[^\]]*\])?\s*([0-9eE+.\-]+)\s*;",
        )
    return solid_conductivity, insulator_conductivity


def infer_uniform_b0(case_dir: str | Path) -> tuple[float, float, float] | None:
    path = _first_existing(
        case_dir,
        "case/0/liquid/B0",
        "0/liquid/B0",
        "latestTime/liquid/B0",
        "case/0/B0",
        "0/B0",
        "latestTime/B0",
    )
    if path is None:
        return None
    match = re.search(r"internalField\s+uniform\s+\(\s*(\S+)\s+(\S+)\s+(\S+)\s*\)", path.read_text())
    if match is None:
        return None
    return float(match.group(1)), float(match.group(2)), float(match.group(3))


def infer_rectangular_geometry(case_dir: str | Path) -> tuple[float, float, float | None, int | None] | None:
    path = _first_existing(case_dir, "case/system/blockMeshDict", "system/blockMeshDict")
    if path is None:
        return None
    text = path.read_text()
    half_width = _extract_first_scalar(text, r"\bLy\s+([0-9eE+.\-]+)\s*;")
    outer_half_width = _extract_first_scalar(text, r"\bLy_wall\s+([0-9eE+.\-]+)\s*;")
    wall_cells = _extract_first_scalar(text, r"\bN_wall\s+([0-9eE+.\-]+)\s*;")
    if half_width is None:
        return None
    wall_thickness = None
    if outer_half_width is not None and outer_half_width >= half_width:
        wall_thickness = outer_half_width - half_width
    return 2.0 * half_width, 2.0 * half_width, wall_thickness, None if wall_cells is None else int(round(wall_cells))


def _infer_control_dict_scalar(case_dir: str | Path, key: str) -> float | None:
    path = Path(case_dir) / "system" / "controlDict"
    if not path.exists():
        return None
    pattern = re.compile(rf"{re.escape(key)}\s+(\S+)\s*;")
    match = pattern.search(path.read_text())
    if match is None:
        return None
    return float(match.group(1))


def infer_magnetic_ramp(case_dir: str | Path) -> tuple[float, float]:
    start = _infer_control_dict_scalar(case_dir, "BtStartTime")
    duration = _infer_control_dict_scalar(case_dir, "BtDuration")
    return (0.0 if start is None else start, 0.0 if duration is None else duration)


def build_case_from_freemhd_reference(
    *,
    case_kind: str,
    ha: float,
    ny: int,
    nz: int,
    dt: float,
    t_final: float,
    max_steps: int,
    reference_run_dir: str | Path,
    forcing: float | None = None,
) -> CaseSpec:
    liquid_properties = infer_liquid_properties(reference_run_dir)
    geometry = infer_rectangular_geometry(reference_run_dir)
    b0 = infer_uniform_b0(reference_run_dir)
    solid_conductivity, insulator_conductivity = infer_solid_conductivities(reference_run_dir)
    conductivity = 1.0
    density = 1.0
    viscosity = 1.0
    if liquid_properties is not None:
        conductivity, density, viscosity = liquid_properties
    width = 2.0
    height = 2.0
    wall_thickness = 0.1
    wall_cells = 8
    if geometry is not None:
        width, height, inferred_wall_thickness, inferred_wall_cells = geometry
        if inferred_wall_thickness is not None and inferred_wall_thickness > 0.0:
            wall_thickness = inferred_wall_thickness
        if inferred_wall_cells is not None and inferred_wall_cells > 0:
            wall_cells = inferred_wall_cells
    if case_kind == "hartmann":
        case = make_hartmann_case(
            ha=ha,
            width=width,
            height=height,
            ny=ny,
            nz=nz,
            conductivity=conductivity,
            density=density,
            viscosity=viscosity,
        )
    elif case_kind == "shercliff":
        case = make_shercliff_case(
            ha=ha,
            width=width,
            height=height,
            ny=ny,
            nz=nz,
            conductivity=conductivity,
            density=density,
            viscosity=viscosity,
        )
    elif case_kind == "hunt":
        case = make_hunt_case(
            ha=ha,
            width=width,
            height=height,
            ny=ny,
            nz=nz,
            wall_cells=wall_cells,
            wall_thickness=wall_thickness,
            fluid_conductivity=conductivity,
            wall_conductivity=solid_conductivity,
            insulator_conductivity=insulator_conductivity,
            density=density,
            viscosity=viscosity,
        )
    else:
        raise ValueError(f"Unsupported FreeMHD reference case kind {case_kind!r}")
    initial_velocity = infer_initial_velocity_x(reference_run_dir) or 0.0
    ramp_start, ramp_duration = infer_magnetic_ramp(reference_run_dir)
    boundary_conditions = case.boundary_conditions
    if forcing is None:
        forcing_value = case.forcing
    else:
        forcing_value = forcing

    if forcing is None:
        drive_mode = infer_inlet_drive_mode(reference_run_dir)
        reduced_area = case.geometry.width * case.geometry.height
        reduced_inlet_flow_rate = infer_reduced_inlet_flow_rate(
            reference_run_dir,
            reduced_area=reduced_area,
            initial_velocity=initial_velocity,
        )
        if drive_mode == "inlet_flow_rate":
            forcing_value = 0.0
            flow_rate = reduced_inlet_flow_rate
            if flow_rate is None:
                flow_rate = initial_velocity * reduced_area
            inlet_bc = BoundaryCondition("inlet", "inlet_flow_rate", value=flow_rate, axis="x")
            boundary_conditions = boundary_conditions + (inlet_bc,)
        elif drive_mode == "inlet_velocity":
            forcing_value = 0.0
            inlet_bc = BoundaryCondition("inlet", "inlet_velocity", value=(initial_velocity, 0.0, 0.0), axis="x")
            boundary_conditions = boundary_conditions + (inlet_bc,)

    return replace(
        case,
        boundary_conditions=boundary_conditions,
        magnetic_field=replace(
            case.magnetic_field,
            value=case.magnetic_field.value if b0 is None else b0,
            ramp_start=ramp_start,
            ramp_duration=ramp_duration,
        ),
        initial_velocity=initial_velocity,
        forcing=forcing_value,
        time_stepper=replace(case.time_stepper, dt=dt, t_final=t_final, max_steps=max_steps),
    )


def parse_freemhd_execution_seconds(run_log_path: str | Path) -> float | None:
    path = Path(run_log_path)
    if not path.exists():
        return None
    latest: float | None = None
    pattern = re.compile(r"ExecutionTime\s*=\s*([0-9eE+.\-]+)\s*s")
    for line in path.read_text(errors="ignore").splitlines():
        match = pattern.search(line)
        if match is not None:
            latest = float(match.group(1))
    return latest


def run_freemhd_demo(
    freemhd_install_dir: str | Path,
    *,
    demo_kind: str,
    nproc: int = 2,
    extra_env: dict[str, str] | None = None,
) -> Path:
    root = Path(freemhd_install_dir)
    script_map = {
        "shercliff": root / "run_shercliff.sh",
        "hunt": root / "run_hunt.sh",
    }
    script = script_map[demo_kind]
    env = None
    if extra_env is not None:
        env = {**os.environ, **extra_env}
    subprocess.run([str(script), str(nproc)], cwd=root, env=env, check=True)
    return root / "freemhd_output" / demo_kind

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import replace
from pathlib import Path

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
    pattern = re.compile(r"volumetricFlowRate\s+(\S+)\s*;")
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
    factories = {
        "hartmann": make_hartmann_case,
        "shercliff": make_shercliff_case,
        "hunt": make_hunt_case,
    }
    case = factories[case_kind](ha=ha, ny=ny, nz=nz)
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
        magnetic_field=replace(case.magnetic_field, ramp_start=ramp_start, ramp_duration=ramp_duration),
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

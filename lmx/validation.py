from __future__ import annotations

import csv
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp

from .core import Solution
from .mesh import StructuredMesh
from .operators import center_coordinates
from .specs import CaseSpec


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


def hartmann_validation(solution: Solution, ha: float) -> AnalyticComparison:
    profile = extract_centerline(solution)
    coordinate = profile["y"] / jnp.max(jnp.abs(profile["y"]))
    u = profile["u"]
    scale = jnp.max(jnp.abs(u))
    scale = jnp.where(scale > 0.0, scale, 1.0)
    normalized = u / scale
    reference = hartmann_analytic_profile(coordinate, ha)
    return compare_profile_to_reference(coordinate, normalized, reference)


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


def freemhd_case_command(case_dir: str | Path, cores: int = 4) -> str:
    return (
        "bash -lc 'source /opt/OpenFOAM/OpenFOAM-v2206/etc/bashrc && "
        f"cd {Path(case_dir)} && "
        f"mpirun -np {cores} epotMultiRegionFoam -parallel'"
    )


def compare_with_freemhd(case_spec: CaseSpec, freemhd_run_dir: str | Path) -> ValidationReport:
    run_dir = Path(freemhd_run_dir)
    metrics = {
        "run_dir_exists": float(run_dir.exists()),
        "has_system": float((run_dir / "system").exists()),
        "has_constant": float((run_dir / "constant").exists()),
        "has_zero_dir": float((run_dir / "0").exists()),
    }
    artifacts = {
        "freemhd_run_dir": str(run_dir),
        "expected_command": freemhd_case_command(run_dir),
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


def write_metrics_json(metrics: dict[str, float], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2))
    return path


def docker_available() -> bool:
    return shutil.which("docker") is not None


def run_freemhd_container(image: str, case_dir: str | Path, cores: int = 4) -> subprocess.CompletedProcess[str]:
    command = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{Path(case_dir).resolve()}:/workspace/case",
        image,
        "bash",
        "-lc",
        freemhd_case_command("/workspace/case", cores=cores),
    ]
    return subprocess.run(command, text=True, capture_output=True, check=False)

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FreeMHDCase:
    path: str
    has_allrun: bool
    has_zero_dir: bool
    has_region_properties: bool


@dataclass(frozen=True)
class FreeMHDTarget:
    path: str
    kind: str
    reason: str


def docker_cli_available() -> bool:
    return shutil.which("docker") is not None


def docker_daemon_available() -> bool:
    if not docker_cli_available():
        return False
    completed = subprocess.run(
        ["docker", "info"],
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.returncode == 0


def discover_freemhd_cases(root: str | Path) -> list[FreeMHDCase]:
    root_path = Path(root)
    if not root_path.exists():
        return []
    cases: list[FreeMHDCase] = []
    for candidate in sorted(root_path.rglob("system")):
        case_dir = candidate.parent
        if "OpenFOAM-v2206" in case_dir.parts:
            continue
        has_constant = (case_dir / "constant").is_dir()
        if not has_constant:
            continue
        has_zero_dir = (case_dir / "0").is_dir() or (case_dir / "0.orig").is_dir()
        has_region_properties = (case_dir / "constant" / "regionProperties").exists()
        has_allrun = (case_dir / "Allrun").exists()
        cases.append(
            FreeMHDCase(
                path=str(case_dir),
                has_allrun=has_allrun,
                has_zero_dir=has_zero_dir,
                has_region_properties=has_region_properties,
            )
        )
    return cases


def recommended_freemhd_target(repo_root: str | Path) -> FreeMHDTarget | None:
    cases = discover_freemhd_cases(repo_root)
    ready_cases = [case for case in cases if case.has_zero_dir]
    if ready_cases:
        ranked = sorted(
            ready_cases,
            key=lambda case: (
                not case.has_region_properties,
                not case.has_allrun,
                len(Path(case.path).parts),
                case.path,
            ),
        )
        target = ranked[0]
        return FreeMHDTarget(
            path=target.path,
            kind="freemhd_case",
            reason="Discovered runnable non-OpenFOAM case directory with system/constant and initial fields.",
        )

    smoke = Path(repo_root) / "OpenFOAM-v2206" / "tutorials" / "electromagnetics" / "mhdFoam" / "hartmann"
    if smoke.exists():
        return FreeMHDTarget(
            path=str(smoke),
            kind="openfoam_smoke_case",
            reason="No standalone FreeMHD cases were discovered locally; bundled Hartmann tutorial is the smallest environment smoke test.",
        )
    return None


def freemhd_environment_report(
    repo_root: str | Path,
    reference_root: str | Path | None = None,
) -> dict[str, object]:
    repo_path = Path(repo_root)
    reference_path = Path(reference_root) if reference_root is not None else None
    cases = discover_freemhd_cases(repo_path)
    target = recommended_freemhd_target(repo_path)
    blockers: list[str] = []
    docker_cli = docker_cli_available()
    docker_daemon = docker_daemon_available()
    if not docker_cli:
        blockers.append("docker CLI not found on PATH")
    elif not docker_daemon:
        blockers.append("docker daemon is not reachable")
    if not cases:
        blockers.append("no local FreeMHD case directories discovered under the current repo/assets")
    return {
        "docker_cli_available": docker_cli,
        "docker_daemon_available": docker_daemon,
        "freemhd_repo_exists": repo_path.exists(),
        "reference_root_exists": reference_path.exists() if reference_path is not None else False,
        "discovered_case_count": len(cases),
        "discovered_cases": [
            {
                "path": case.path,
                "has_allrun": case.has_allrun,
                "has_zero_dir": case.has_zero_dir,
                "has_region_properties": case.has_region_properties,
            }
            for case in cases[:20]
        ],
        "recommended_target": None
        if target is None
        else {
            "path": target.path,
            "kind": target.kind,
            "reason": target.reason,
        },
        "blockers": blockers,
    }

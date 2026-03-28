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


def freemhd_environment_report(
    repo_root: str | Path,
    reference_root: str | Path | None = None,
) -> dict[str, object]:
    repo_path = Path(repo_root)
    reference_path = Path(reference_root) if reference_root is not None else None
    cases = discover_freemhd_cases(repo_path)
    return {
        "docker_cli_available": docker_cli_available(),
        "docker_daemon_available": docker_daemon_available(),
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
    }

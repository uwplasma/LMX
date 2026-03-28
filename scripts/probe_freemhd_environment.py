#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def _run_shell(command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["bash", "-lc", command], text=True, capture_output=True, check=False)


def docker_cli_available() -> bool:
    return shutil.which("docker") is not None


def docker_daemon_available() -> bool:
    if not docker_cli_available():
        return False
    result = subprocess.run(["docker", "info"], text=True, capture_output=True, check=False)
    return result.returncode == 0


def probe_freemhd_environment(repo_root: str | Path) -> dict[str, object]:
    root = Path(repo_root).resolve()
    foam_root = root / "OpenFOAM-v2206"
    solver_dir = root / "MHD_Solvers" / "solvers" / "epotMultiRegionFoam"
    bashrc = foam_root / "etc" / "bashrc"
    wmkdepend = foam_root / "platforms" / "tools" / "darwin64Clang" / "wmkdepend"

    foam_check = _run_shell(f"source {bashrc} && foamSystemCheck")
    build_probe = _run_shell(f"source {bashrc} && cd {solver_dir} && wmake")

    return {
        "repo_root": str(root),
        "foam_root": str(foam_root),
        "solver_dir": str(solver_dir),
        "bashrc_exists": bashrc.exists(),
        "wmkdepend_exists": wmkdepend.exists(),
        "docker_cli_available": docker_cli_available(),
        "docker_daemon_available": docker_daemon_available(),
        "foam_system_check_returncode": foam_check.returncode,
        "foam_system_check_stdout_tail": foam_check.stdout[-4000:],
        "foam_system_check_stderr_tail": foam_check.stderr[-4000:],
        "solver_build_probe_returncode": build_probe.returncode,
        "solver_build_probe_stdout_tail": build_probe.stdout[-4000:],
        "solver_build_probe_stderr_tail": build_probe.stderr[-4000:],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Probe the local FreeMHD/OpenFOAM environment and emit JSON diagnostics.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "external" / "FreeMHD",
        help="FreeMHD checkout root containing OpenFOAM-v2206 and MHD_Solvers.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    payload = probe_freemhd_environment(args.repo_root)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

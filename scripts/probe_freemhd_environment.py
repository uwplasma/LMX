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


def classify_build_probe(stderr: str, wmkdepend_exists: bool, darwin_header_patch_detected: bool = False) -> str:
    if not wmkdepend_exists and "wmkdepend" in stderr:
        return "missing-wmkdepend"
    if "<cstring> tried including <string.h>" in stderr or "<cwchar> tried including <wchar.h>" in stderr:
        return "macos-libcxx-header-conflict"
    if "lnInclude/string.h" in stderr or "lnInclude/time.h" in stderr or "lnInclude/wchar.h" in stderr:
        return "macos-libcxx-header-conflict"
    if "fatal error: 'fvMesh.H' file not found" in stderr:
        if darwin_header_patch_detected:
            return "post-darwin-header-patch-include-regression"
        return "missing-openfoam-lninclude"
    if not stderr.strip():
        return "ok"
    return "unknown-build-failure"


def detect_shadowed_c_headers(stderr: str) -> list[str]:
    hits: list[str] = []
    for header in ("string.h", "time.h", "wchar.h"):
        if f"lnInclude/{header}" in stderr:
            hits.append(header)
    return hits


def build_issue_recommendation(issue: str, stderr: str) -> str:
    if issue == "missing-wmkdepend":
        return (
            "Build OpenFOAM wmake tools first, for example under "
            "OpenFOAM-v2206/wmake/src, before retrying the FreeMHD solver build."
        )
    if issue == "macos-libcxx-header-conflict":
        shadowed = detect_shadowed_c_headers(stderr)
        if shadowed:
            joined = ", ".join(shadowed)
            return (
                "OpenFOAM lnInclude C-header shadowing is likely. On Darwin, test demoting "
                "src/OpenFOAM/lnInclude and src/OSspecific/POSIX/lnInclude from -I to -idirafter "
                "in wmake/rules/darwin64Clang/c++ and wmake/rules/darwin64Clang/c. "
                f"Observed shadowed headers: {joined}."
            )
        return (
            "The Darwin/libc++ header environment is inconsistent. Test the darwin64Clang wmake "
            "include ordering and system header flags before retrying the build."
        )
    if issue == "ok":
        return "No build blocker detected."
    if issue == "post-darwin-header-patch-include-regression":
        return (
            "The Darwin header-shadowing workaround moved past the libc++ collision, but OpenFOAM include "
            "resolution is now incomplete. Inspect the expanded compile line and compare LIB_HEADER_DIRS plus "
            "EXE_INC before and after the Darwin patch."
        )
    if issue == "missing-openfoam-lninclude":
        return (
            "OpenFOAM lnInclude headers are not being found. Inspect the expanded compile line and verify that "
            "the OpenFOAM library lnInclude directories are present."
        )
    return "Inspect the stderr tail and compare it to the darwin64Clang wmake rules."


def probe_freemhd_environment(repo_root: str | Path) -> dict[str, object]:
    root = Path(repo_root).resolve()
    foam_root = root / "OpenFOAM-v2206"
    solver_dir = root / "MHD_Solvers" / "solvers" / "epotMultiRegionFoam"
    bashrc = foam_root / "etc" / "bashrc"
    wmkdepend = foam_root / "platforms" / "tools" / "darwin64Clang" / "wmkdepend"
    darwin_cxx_rule = foam_root / "wmake" / "rules" / "darwin64Clang" / "c++"
    darwin_c_rule = foam_root / "wmake" / "rules" / "darwin64Clang" / "c"
    darwin_patch_detected = (
        "DARWIN_LIB_HEADER_DIRS :=" in darwin_cxx_rule.read_text() if darwin_cxx_rule.exists() else False
    ) and ("DARWIN_LIB_HEADER_DIRS :=" in darwin_c_rule.read_text() if darwin_c_rule.exists() else False)

    foam_check = _run_shell(f"source {bashrc} && foamSystemCheck")
    build_probe = _run_shell(f"source {bashrc} && cd {solver_dir} && wmake")
    build_issue = classify_build_probe(build_probe.stderr, wmkdepend.exists(), darwin_patch_detected)
    shadowed_headers = detect_shadowed_c_headers(build_probe.stderr)

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
        "solver_build_issue": build_issue,
        "solver_build_shadowed_headers": shadowed_headers,
        "solver_build_recommendation": build_issue_recommendation(build_issue, build_probe.stderr),
        "darwin_header_patch_detected": darwin_patch_detected,
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

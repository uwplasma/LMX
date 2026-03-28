from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


@dataclass(frozen=True)
class FreeMHDCase:
    path: str
    source_root: str
    has_allrun: bool
    has_zero_dir: bool
    has_region_properties: bool


@dataclass(frozen=True)
class FreeMHDTarget:
    path: str
    kind: str
    reason: str


def docker_command_result(command: list[str], timeout_seconds: int | float | None = None) -> dict[str, object]:
    if not docker_cli_available():
        return {
            "command": command,
            "status": "docker-cli-unavailable",
            "returncode": None,
            "stdout_tail": "",
            "stderr_tail": "",
        }
    try:
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "status": "timeout",
            "returncode": None,
            "stdout_tail": (exc.stdout or "")[-4000:],
            "stderr_tail": (exc.stderr or "")[-4000:],
        }
    return {
        "command": command,
        "status": "ok" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }


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


def docker_image_available(image: str) -> bool:
    if not docker_daemon_available():
        return False
    completed = subprocess.run(
        ["docker", "image", "inspect", image],
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.returncode == 0


def docker_registry_image_report(image: str, timeout_seconds: int = 20) -> dict[str, object]:
    return docker_command_result(["docker", "manifest", "inspect", image], timeout_seconds=timeout_seconds)


def docker_local_image_report(image: str) -> dict[str, object]:
    if not docker_daemon_available():
        return {
            "command": ["docker", "image", "inspect", image],
            "status": "docker-daemon-unavailable",
            "returncode": None,
            "stdout_tail": "",
            "stderr_tail": "",
        }
    return docker_command_result(["docker", "image", "inspect", image])


def docker_pull_image_report(image: str, timeout_seconds: int = 20) -> dict[str, object]:
    if not docker_daemon_available():
        return {
            "command": ["docker", "pull", image],
            "status": "docker-daemon-unavailable",
            "returncode": None,
            "stdout_tail": "",
            "stderr_tail": "",
        }
    return docker_command_result(["docker", "pull", image], timeout_seconds=timeout_seconds)


def dockerfile_base_image(dockerfile_path: str | Path) -> str | None:
    path = Path(dockerfile_path)
    if not path.exists():
        return None
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.upper().startswith("FROM "):
            parts = line.split()
            if len(parts) >= 2:
                return parts[1]
    return None


def parse_docker_hub_image(image: str) -> tuple[str, str, str] | None:
    reference = image
    if "@" in reference:
        reference = reference.split("@", 1)[0]
    if ":" in reference.rsplit("/", 1)[-1]:
        repository, tag = reference.rsplit(":", 1)
    else:
        repository, tag = reference, "latest"
    parts = repository.split("/")
    if len(parts) == 1:
        namespace = "library"
        repo = parts[0]
    elif len(parts) == 2:
        namespace, repo = parts
    else:
        return None
    return namespace, repo, tag


def docker_hub_tag_report(image: str, timeout_seconds: int = 20) -> dict[str, object]:
    parsed = parse_docker_hub_image(image)
    if parsed is None:
        return {
            "image": image,
            "status": "unsupported-image-reference",
            "url": None,
        }
    namespace, repo, tag = parsed
    url = f"https://hub.docker.com/v2/namespaces/{namespace}/repositories/{repo}/tags/{tag}"
    try:
        with urlopen(url, timeout=timeout_seconds) as response:
            payload = response.read().decode("utf-8")
    except HTTPError as exc:
        return {
            "image": image,
            "status": "failed",
            "http_status": exc.code,
            "url": url,
        }
    except URLError as exc:
        return {
            "image": image,
            "status": "network-error",
            "reason": str(exc.reason),
            "url": url,
        }
    except TimeoutError:
        return {
            "image": image,
            "status": "timeout",
            "url": url,
        }
    return {
        "image": image,
        "status": "ok",
        "url": url,
        "payload_excerpt": payload[:1000],
    }


def discover_freemhd_cases(root: str | Path | None) -> list[FreeMHDCase]:
    if root is None:
        return []
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
                source_root=str(root_path),
                has_allrun=has_allrun,
                has_zero_dir=has_zero_dir,
                has_region_properties=has_region_properties,
            )
        )
    return cases


def discover_freemhd_cases_in_roots(roots: list[str | Path | None]) -> list[FreeMHDCase]:
    discovered: dict[str, FreeMHDCase] = {}
    for root in roots:
        for case in discover_freemhd_cases(root):
            discovered[case.path] = case
    return [discovered[path] for path in sorted(discovered)]


def recommended_freemhd_target(repo_root: str | Path, extra_case_roots: list[str | Path] | None = None) -> FreeMHDTarget | None:
    search_roots: list[str | Path | None] = [repo_root, *(extra_case_roots or [])]
    cases = discover_freemhd_cases_in_roots(search_roots)
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
            reason=f"Discovered runnable non-OpenFOAM case directory with system/constant and initial fields under {target.source_root}.",
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
    extra_case_roots: list[str | Path] | None = None,
) -> dict[str, object]:
    repo_path = Path(repo_root)
    reference_path = Path(reference_root) if reference_root is not None else None
    extra_paths = [Path(root) for root in extra_case_roots or []]
    cases = discover_freemhd_cases_in_roots([repo_path, *extra_paths])
    target = recommended_freemhd_target(repo_path, extra_case_roots=extra_paths)
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
        "searched_case_roots": [str(repo_path), *[str(path) for path in extra_paths]],
        "extra_case_roots": [str(path) for path in extra_paths],
        "discovered_case_count": len(cases),
        "discovered_cases": [
            {
                "path": case.path,
                "source_root": case.source_root,
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


def freemhd_container_report(
    bundle_root: str | Path,
    image: str,
    timeout_seconds: int = 20,
    check_pull: bool = False,
    pull_timeout_seconds: int = 20,
) -> dict[str, object]:
    bundle_path = Path(bundle_root)
    dockerfile_path = bundle_path / "Dockerfile"
    base_image = dockerfile_base_image(dockerfile_path)

    blockers: list[str] = []
    docker_cli = docker_cli_available()
    docker_daemon = docker_daemon_available()
    if not docker_cli:
        blockers.append("docker CLI not found on PATH")
    elif not docker_daemon:
        blockers.append("docker daemon is not reachable")
    if not dockerfile_path.exists():
        blockers.append("docker bundle Dockerfile is missing")
    if base_image is None:
        blockers.append("docker bundle base image could not be parsed")

    local_image = docker_local_image_report(image)
    base_image_local = None if base_image is None else docker_local_image_report(base_image)
    base_image_registry = None if base_image is None else docker_hub_tag_report(base_image, timeout_seconds=timeout_seconds)
    base_image_pull = None if (base_image is None or not check_pull) else docker_pull_image_report(base_image, timeout_seconds=pull_timeout_seconds)

    if local_image["status"] not in {"ok", "docker-daemon-unavailable", "docker-cli-unavailable"}:
        blockers.append(f"requested image is not available locally: {image}")
    if base_image_registry is not None and base_image_registry["status"] == "timeout":
        blockers.append(f"base image tag lookup timed out: {base_image}")
    elif base_image_registry is not None and base_image_registry["status"] not in {"ok", "unsupported-image-reference"}:
        blockers.append(f"base image tag lookup failed: {base_image}")
    if base_image_pull is not None and base_image_pull["status"] == "timeout":
        blockers.append(f"base image pull timed out: {base_image}")
    elif base_image_pull is not None and base_image_pull["status"] not in {
        "ok",
        "docker-daemon-unavailable",
        "docker-cli-unavailable",
    }:
        blockers.append(f"base image pull failed: {base_image}")

    return {
        "bundle_root": str(bundle_path.resolve()),
        "image": image,
        "check_pull": check_pull,
        "docker_cli_available": docker_cli,
        "docker_daemon_available": docker_daemon,
        "dockerfile_exists": dockerfile_path.exists(),
        "dockerfile_path": str(dockerfile_path.resolve()),
        "base_image": base_image,
        "local_image_report": local_image,
        "base_image_local_report": base_image_local,
        "base_image_registry_report": base_image_registry,
        "base_image_pull_report": base_image_pull,
        "blockers": blockers,
    }

from pathlib import Path
from subprocess import TimeoutExpired
from urllib.error import URLError

import pytest

from lmx.freemhd import (
    control_dict_application,
    decompose_par_subdomains,
    discover_freemhd_cases,
    discover_freemhd_cases_in_roots,
    docker_hub_tag_report,
    dockerfile_base_image,
    parse_docker_hub_image,
    freemhd_environment_report,
    freemhd_container_report,
    recommended_freemhd_target,
    docker_command_result,
    docker_image_available,
    docker_local_image_report,
    docker_pull_image_report,
    docker_registry_image_report,
)


pytestmark = pytest.mark.unit


def test_discover_freemhd_cases_skips_embedded_openfoam_tree(tmp_path: Path):
    embedded = tmp_path / "OpenFOAM-v2206" / "tutorials" / "foo"
    (embedded / "system").mkdir(parents=True)
    (embedded / "constant").mkdir()
    assert discover_freemhd_cases(tmp_path) == []


def test_recommended_freemhd_target_prefers_real_case(tmp_path: Path):
    case_dir = tmp_path / "ShercliffCase"
    (case_dir / "system").mkdir(parents=True)
    (case_dir / "constant").mkdir()
    (case_dir / "0").mkdir()
    target = recommended_freemhd_target(tmp_path)
    assert target is not None
    assert target.kind == "freemhd_case"
    assert target.path == str(case_dir)


def test_discover_freemhd_cases_in_roots_keeps_extra_root_case(tmp_path: Path):
    repo_root = tmp_path / "external" / "FreeMHD"
    repo_root.mkdir(parents=True)
    extra_root = tmp_path / "recovered" / "Hunt"
    case_dir = extra_root / "hunt_Ha20"
    (case_dir / "system").mkdir(parents=True)
    (case_dir / "constant").mkdir()
    (case_dir / "0").mkdir()

    cases = discover_freemhd_cases_in_roots([repo_root, extra_root])
    assert len(cases) == 1
    assert cases[0].path == str(case_dir)
    assert cases[0].source_root == str(extra_root)


def test_recommended_freemhd_target_falls_back_to_openfoam_smoke_case(tmp_path: Path):
    smoke = tmp_path / "OpenFOAM-v2206" / "tutorials" / "electromagnetics" / "mhdFoam" / "hartmann"
    smoke.mkdir(parents=True)
    target = recommended_freemhd_target(tmp_path)
    assert target is not None
    assert target.kind == "openfoam_smoke_case"
    assert target.path == str(smoke)


def test_freemhd_environment_report_includes_recommended_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    smoke = tmp_path / "OpenFOAM-v2206" / "tutorials" / "electromagnetics" / "mhdFoam" / "hartmann"
    smoke.mkdir(parents=True)
    monkeypatch.setattr("lmx.freemhd.docker_cli_available", lambda: True)
    monkeypatch.setattr("lmx.freemhd.docker_daemon_available", lambda: False)
    report = freemhd_environment_report(tmp_path)
    assert report["docker_cli_available"] is True
    assert report["docker_daemon_available"] is False
    assert report["recommended_target"]["kind"] == "openfoam_smoke_case"
    assert "docker daemon is not reachable" in report["blockers"]


def test_freemhd_environment_report_includes_extra_case_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    repo_root = tmp_path / "external" / "FreeMHD"
    repo_root.mkdir(parents=True)
    extra_root = tmp_path / "recovered" / "Shercliff"
    case_dir = extra_root / "shercliff_Ha20"
    (case_dir / "system").mkdir(parents=True)
    (case_dir / "constant").mkdir()
    (case_dir / "0").mkdir()

    monkeypatch.setattr("lmx.freemhd.docker_cli_available", lambda: True)
    monkeypatch.setattr("lmx.freemhd.docker_daemon_available", lambda: True)

    report = freemhd_environment_report(repo_root, extra_case_roots=[extra_root])

    assert report["discovered_case_count"] == 1
    assert report["searched_case_roots"] == [str(repo_root), str(extra_root)]
    assert report["extra_case_roots"] == [str(extra_root)]
    assert report["discovered_cases"][0]["source_root"] == str(extra_root)
    assert report["recommended_target"]["kind"] == "freemhd_case"
    assert report["recommended_target"]["path"] == str(case_dir)
    assert str(extra_root) in report["recommended_target"]["reason"]
    assert report["blockers"] == []


def test_dockerfile_base_image_parses_first_from(tmp_path: Path):
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text(
        "# comment\n"
        "FROM microfluidica/openfoam:2206 AS base\n"
        "RUN echo ok\n"
    )
    assert dockerfile_base_image(dockerfile) == "microfluidica/openfoam:2206"


def test_parse_docker_hub_image_handles_explicit_and_default_tags():
    assert parse_docker_hub_image("microfluidica/openfoam:2206") == ("microfluidica", "openfoam", "2206")
    assert parse_docker_hub_image("openfoam") == ("library", "openfoam", "latest")


def test_control_dict_application_reads_application(tmp_path: Path):
    case_dir = tmp_path / "case"
    (case_dir / "system").mkdir(parents=True)
    (case_dir / "system" / "controlDict").write_text("application     epotMultiRegionInterFoam;\n")
    assert control_dict_application(case_dir) == "epotMultiRegionInterFoam"


def test_decompose_par_subdomains_reads_count(tmp_path: Path):
    case_dir = tmp_path / "case"
    (case_dir / "system").mkdir(parents=True)
    (case_dir / "system" / "decomposeParDict").write_text("numberOfSubdomains 95;\n")
    assert decompose_par_subdomains(case_dir) == 95


def test_docker_hub_tag_report_handles_404(monkeypatch: pytest.MonkeyPatch):
    from urllib.error import HTTPError

    def fake_urlopen(url: str, timeout: int = 20):
        raise HTTPError(url, 404, "Not found", hdrs=None, fp=None)

    monkeypatch.setattr("lmx.freemhd.urlopen", fake_urlopen)
    report = docker_hub_tag_report("openfoam/openfoam2206-paraview:latest")
    assert report["status"] == "failed"
    assert report["http_status"] == 404


def test_docker_hub_tag_report_handles_success(monkeypatch: pytest.MonkeyPatch):
    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"name":"2206","last_updated":"2026-03-19T18:59:13Z"}'

    monkeypatch.setattr("lmx.freemhd.urlopen", lambda url, timeout=20: _Response())
    report = docker_hub_tag_report("microfluidica/openfoam:2206")
    assert report["status"] == "ok"
    assert '"name":"2206"' in report["payload_excerpt"]


def test_docker_command_result_handles_unavailable_and_timeout(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("lmx.freemhd.docker_cli_available", lambda: False)
    unavailable = docker_command_result(["docker", "version"])
    assert unavailable["status"] == "docker-cli-unavailable"

    monkeypatch.setattr("lmx.freemhd.docker_cli_available", lambda: True)

    def fake_run(*args, **kwargs):
        raise TimeoutExpired(cmd=kwargs.get("args", args[0]), timeout=1, output="out", stderr="err")

    monkeypatch.setattr("lmx.freemhd.subprocess.run", fake_run)
    timed_out = docker_command_result(["docker", "version"], timeout_seconds=1)
    assert timed_out["status"] == "timeout"
    assert "out" in timed_out["stdout_tail"]


def test_docker_image_helpers_cover_daemon_and_command_paths(monkeypatch: pytest.MonkeyPatch):
    class _Result:
        def __init__(self, returncode: int, stdout: str = "", stderr: str = ""):
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    def fake_run(command, text=True, capture_output=True, timeout=None, check=False):
        if command[:2] == ["docker", "info"]:
            return _Result(0)
        if command[:2] == ["docker", "image"]:
            return _Result(0, stdout="image\n")
        if command[:2] == ["docker", "pull"]:
            return _Result(1, stderr="pull failed\n")
        if command[:2] == ["docker", "manifest"]:
            return _Result(0, stdout="manifest ok\n")
        return _Result(0)

    monkeypatch.setattr("lmx.freemhd.docker_cli_available", lambda: True)
    monkeypatch.setattr("lmx.freemhd.subprocess.run", fake_run)

    assert docker_image_available("microfluidica/openfoam:2206") is True
    assert docker_local_image_report("microfluidica/openfoam:2206")["status"] == "ok"
    assert docker_pull_image_report("microfluidica/openfoam:2206")["status"] == "failed"
    assert docker_registry_image_report("microfluidica/openfoam:2206")["status"] == "ok"


def test_docker_hub_tag_report_handles_unsupported_reference_and_network_error(monkeypatch: pytest.MonkeyPatch):
    report = docker_hub_tag_report("a/b/c:tag")
    assert report["status"] == "unsupported-image-reference"

    monkeypatch.setattr("lmx.freemhd.urlopen", lambda url, timeout=20: (_ for _ in ()).throw(URLError("down")))
    network = docker_hub_tag_report("microfluidica/openfoam:2206")
    assert network["status"] == "network-error"


def test_freemhd_container_report_classifies_local_missing_and_registry_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    bundle_root = tmp_path / "docker"
    bundle_root.mkdir()
    (bundle_root / "Dockerfile").write_text("FROM microfluidica/openfoam:2206\n")

    monkeypatch.setattr("lmx.freemhd.docker_cli_available", lambda: True)
    monkeypatch.setattr("lmx.freemhd.docker_daemon_available", lambda: True)

    def fake_local_report(image: str) -> dict[str, object]:
        return {
            "command": ["docker", "image", "inspect", image],
            "status": "failed",
            "returncode": 1,
            "stdout_tail": "[]\n",
            "stderr_tail": f"No such image: {image}\n",
        }

    monkeypatch.setattr("lmx.freemhd.docker_local_image_report", fake_local_report)
    monkeypatch.setattr(
        "lmx.freemhd.docker_hub_tag_report",
        lambda image, timeout_seconds=20: {
            "image": image,
            "status": "timeout",
            "url": "https://hub.docker.com/example",
        },
    )

    report = freemhd_container_report(bundle_root=bundle_root, image="lmx-freemhd", timeout_seconds=5)

    assert report["base_image"] == "microfluidica/openfoam:2206"
    assert report["local_image_report"]["status"] == "failed"
    assert report["base_image_registry_report"]["status"] == "timeout"
    assert "requested image is not available locally: lmx-freemhd" in report["blockers"]
    assert "base image tag lookup timed out: microfluidica/openfoam:2206" in report["blockers"]


def test_freemhd_container_report_can_classify_base_image_pull_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    bundle_root = tmp_path / "docker"
    bundle_root.mkdir()
    (bundle_root / "Dockerfile").write_text("FROM microfluidica/openfoam:2206\n")

    monkeypatch.setattr("lmx.freemhd.docker_cli_available", lambda: True)
    monkeypatch.setattr("lmx.freemhd.docker_daemon_available", lambda: True)
    monkeypatch.setattr(
        "lmx.freemhd.docker_local_image_report",
        lambda image: {
            "command": ["docker", "image", "inspect", image],
            "status": "failed",
            "returncode": 1,
            "stdout_tail": "[]\n",
            "stderr_tail": f"No such image: {image}\n",
        },
    )
    monkeypatch.setattr(
        "lmx.freemhd.docker_hub_tag_report",
        lambda image, timeout_seconds=20: {"image": image, "status": "ok", "url": "https://hub.docker.com/example"},
    )
    monkeypatch.setattr(
        "lmx.freemhd.docker_pull_image_report",
        lambda image, timeout_seconds=20: {
            "command": ["docker", "pull", image],
            "status": "timeout",
            "returncode": None,
            "stdout_tail": "",
            "stderr_tail": "",
        },
    )

    report = freemhd_container_report(
        bundle_root=bundle_root,
        image="lmx-freemhd",
        check_pull=True,
        pull_timeout_seconds=5,
    )

    assert report["base_image_pull_report"]["status"] == "timeout"
    assert "base image pull timed out: microfluidica/openfoam:2206" in report["blockers"]

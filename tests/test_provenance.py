from __future__ import annotations

import hashlib
from pathlib import Path

from scripts import manage_provenance as provenance


def test_repository_provenance_manifests_are_canonical_and_complete():
    assert provenance.check_manifests() == []

    environment = provenance.build_environment_manifest()
    assert environment["python"]["ci_tested"] == ["3.10", "3.13"]
    assert environment["numerical_policy"]["jax_enable_x64"] is True
    assert environment["portable_gate"]["budget_seconds"] == 600
    assert environment["portable_gate"]["branch_coverage_percent"] >= 95.0
    assert "lmx/solvers.py" in environment["repository_inventory"]["modules"]
    assert "tests/test_provenance.py" in environment["repository_inventory"]["tests"]
    assert (
        "campaigns/autodiff/autodiff_profile_design_demo.py"
        in environment["repository_inventory"]["examples"]
    )


def test_feature_manifest_references_real_tests_and_all_package_modules():
    payload = provenance._read_json(provenance.FEATURES_PATH)
    assert provenance.validate_feature_manifest(payload) == []

    missing = provenance._test_reference_error("tests/test_solver.py::test_not_present")
    malformed = provenance._test_reference_error("tests/test_solver.py")
    assert (
        missing
        == "test function does not exist: tests/test_solver.py::test_not_present"
    )
    assert malformed == "test reference must use path::function: tests/test_solver.py"


def test_benchmark_registry_has_live_checksums_and_verified_benchmark_a_status():
    payload = provenance._read_json(provenance.BENCHMARKS_PATH)
    assert provenance.validate_benchmark_manifest(payload) == []

    statuses = {item["id"]: item["status"] for item in payload["benchmarks"]}
    assert statuses["A1-hartmann"] == "verified-bounded"
    assert statuses["A2-shercliff"] == "verified-bounded"
    assert statuses["A2-hunt"] == "verified-bounded"
    assert statuses["A3-high-ha"] == "verified-bounded"
    assert statuses["B1-fringing-pipe"] == "specification-frozen"
    assert statuses["B2-fringing-square"] == "specification-frozen"


def test_external_literature_verifier_reports_missing_and_changed_files(tmp_path: Path):
    content = b"independent literature artifact"
    digest = hashlib.sha256(content).hexdigest()
    payload = {
        "sources": [
            {"filename": "present.pdf", "sha256": digest},
            {"filename": "missing.pdf", "sha256": "0" * 64},
        ]
    }

    (tmp_path / "present.pdf").write_bytes(content)
    errors = provenance.verify_external_literature(payload, tmp_path)
    assert errors == ["external literature file is missing: missing.pdf"]

    (tmp_path / "present.pdf").write_bytes(b"changed")
    errors = provenance.verify_external_literature(payload, tmp_path)
    assert "external literature checksum mismatch: present.pdf" in errors

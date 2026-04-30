from __future__ import annotations

import json
from pathlib import Path
import shutil

from lmx import (
    audit_q2dmhdfoam_lmx_turbulence_match,
    write_q2dmhdfoam_lmx_turbulence_match_audit,
)


OUTPUT_DIR = Path("artifacts/examples/q2dmhdfoam_lmx_turbulence_match_audit")
DOCS_OUTPUT_DIR = Path("docs/_static/generated")
Q2DMHDFOAM_ROOT = Path("/Users/rogerio/local/tests/lmx_external_codes/Q2DmhdFoam")
CASE_RELATIVE_PATHS = (
    Path("run/lidDriven"),
    Path("run/muck_q2d_FFT"),
    Path("run/muck_q2d"),
)
COPY_TO_DOCS = True


def run_q2dmhdfoam_lmx_turbulence_match_audit(
    *,
    out_dir: Path = OUTPUT_DIR,
    q2dmhdfoam_root: Path = Q2DMHDFOAM_ROOT,
    case_relative_paths: tuple[Path, ...] = CASE_RELATIVE_PATHS,
    docs_output_dir: Path = DOCS_OUTPUT_DIR,
    copy_to_docs: bool = COPY_TO_DOCS,
) -> dict[str, object]:
    """Audit whether available Q2DmhdFoam cases can close LMX Q2D parity."""

    out_dir.mkdir(parents=True, exist_ok=True)
    docs_output_dir.mkdir(parents=True, exist_ok=True)
    root = Path(q2dmhdfoam_root)
    case_dirs = [root / relative for relative in case_relative_paths if (root / relative).exists()]
    missing_cases = [str(relative) for relative in case_relative_paths if not (root / relative).exists()]

    if not case_dirs:
        summary = {
            "case": "q2dmhdfoam_lmx_turbulence_match_audit",
            "status": "no_q2dmhdfoam_cases_found",
            "q2dmhdfoam_root": str(root),
            "requested_cases": [str(path) for path in case_relative_paths],
            "missing_cases": missing_cases,
            "strict_blocker_closed": False,
            "notes": "Clone or run Q2DmhdFoam outside the LMX tree before using this audit.",
        }
        _write_summary(summary, out_dir, docs_output_dir if copy_to_docs else None)
        return summary

    audits = [audit_q2dmhdfoam_lmx_turbulence_match(case_dir) for case_dir in case_dirs]
    artifact_paths = write_q2dmhdfoam_lmx_turbulence_match_audit(audits, out_dir)
    copied: list[str] = []
    if copy_to_docs:
        for path in artifact_paths:
            target = docs_output_dir / path.name
            shutil.copy2(path, target)
            copied.append(target.name)

    strict_admissible_cases = [
        str(audit["case_name"]) for audit in audits if bool(audit.get("strict_admissible", False))
    ]
    summary = {
        "case": "q2dmhdfoam_lmx_turbulence_match_audit",
        "status": "strict_match_audit_written",
        "q2dmhdfoam_root": str(root),
        "audited_cases": [str(path.relative_to(root)) if path.is_relative_to(root) else str(path) for path in case_dirs],
        "missing_cases": missing_cases,
        "case_count": len(audits),
        "strict_admissible_cases": strict_admissible_cases,
        "all_strict_admissible": bool(audits and len(strict_admissible_cases) == len(audits)),
        "strict_blocker_closed": False,
        "matched_parity": False,
        "artifacts": [path.name for path in artifact_paths],
        "docs_artifacts": copied,
        "audits": audits,
        "notes": (
            "This audit prevents unmatched Q2DmhdFoam outputs from being used as "
            "the strict LMX nonlinear Q2D turbulence reference. A case can be "
            "promoted only after topology, forcing, Hartmann friction, timestep "
            "window, and observable definitions are all matched."
        ),
    }
    _write_summary(summary, out_dir, docs_output_dir if copy_to_docs else None)
    return summary


def _write_summary(summary: dict[str, object], out_dir: Path, docs_output_dir: Path | None) -> Path:
    path = out_dir / "q2dmhdfoam_lmx_turbulence_match_audit_summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if docs_output_dir is not None:
        shutil.copy2(path, docs_output_dir / path.name)
    return path


if __name__ == "__main__":
    run_q2dmhdfoam_lmx_turbulence_match_audit()

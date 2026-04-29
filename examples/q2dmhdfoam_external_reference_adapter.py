from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from lmx import (
    load_q2dmhdfoam_lid_driven_observables,
    load_q2dmhdfoam_line_profile,
    q2dmhdfoam_profile_observables,
    write_q2dmhdfoam_external_reference_panel,
    write_q2dmhdfoam_profile_observable_table,
)


OUTPUT_DIR = Path("artifacts/examples/q2dmhdfoam_external_reference")
Q2DMHDFOAM_ROOT = Path("/Users/rogerio/local/tests/lmx_external_codes/Q2DmhdFoam")
PROFILE_SAMPLE_PATTERNS = (
    "FFT2_validation/tepot/samples/lineSampled_theta_Ux_*",
    "FFT2_validation/tepot_nullGr/samples/lineSampled_theta_Ux_*",
)
VETCHA_DIGITIZED_PROFILE = Path("FFT2_validation/vetcha2009/vetcha2009_Ha50_Re1e4.csv")
LID_DRIVEN_TURBULENCE_SUMMARY = Path("run/lidDriven/IDM_output_U.txt")
OUTPUT_STEM = "q2dmhdfoam_external_reference"


def run_q2dmhdfoam_external_reference_adapter(
    *,
    out_dir: Path = OUTPUT_DIR,
    q2dmhdfoam_root: Path = Q2DMHDFOAM_ROOT,
) -> dict[str, object]:
    """Ingest local Q2DmhdFoam outputs into an LMX validation artifact."""

    out_dir.mkdir(parents=True, exist_ok=True)
    root = Path(q2dmhdfoam_root)
    if not root.exists():
        summary = {
            "case": "q2dmhdfoam_external_reference_adapter",
            "status": "external_q2dmhdfoam_root_missing",
            "q2dmhdfoam_root": str(root),
            "notes": "Clone/build/run Q2DmhdFoam outside the LMX tree before using this adapter.",
        }
        (out_dir / "q2dmhdfoam_external_reference_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
        return summary

    profiles = _collect_q2dmhdfoam_profiles(root)
    vetcha_profiles = _load_vetcha_digitized_profiles(root / VETCHA_DIGITIZED_PROFILE)
    profiles.extend(vetcha_profiles)
    if not profiles:
        raise FileNotFoundError(f"No Q2DmhdFoam or Vetcha profile samples were found under {root}")

    profile_observables = [q2dmhdfoam_profile_observables(profile) for profile in profiles]
    table_path = write_q2dmhdfoam_profile_observable_table(
        profile_observables,
        out_dir / "q2dmhdfoam_external_profile_observables.csv",
    )

    turbulence_path = root / LID_DRIVEN_TURBULENCE_SUMMARY
    turbulence_observables: dict[str, object] = {}
    turbulence_table = None
    if turbulence_path.exists():
        turbulence_observables = dict(load_q2dmhdfoam_lid_driven_observables(turbulence_path))
        turbulence_table = _write_turbulence_observables_csv(
            turbulence_observables,
            out_dir / "q2dmhdfoam_lid_driven_turbulence_observables.csv",
        )

    plots = write_q2dmhdfoam_external_reference_panel(
        profiles,
        profile_observables,
        out_dir,
        turbulence_observables=turbulence_observables,
        output_stem=OUTPUT_STEM,
    )
    summary = {
        "case": "q2dmhdfoam_external_reference_adapter",
        "status": "external_reference_artifacts_written",
        "q2dmhdfoam_root": str(root),
        "profile_count": len(profiles),
        "q2dmhdfoam_profile_count": len(profiles) - len(vetcha_profiles),
        "vetcha_digitized_profile_count": len(vetcha_profiles),
        "profile_observable_table": table_path.name,
        "turbulence_observable_table": turbulence_table.name if turbulence_table else None,
        "plots": [path.name for path in plots],
        "profile_observables": profile_observables,
        "turbulence_observables": turbulence_observables,
        "notes": (
            "This adapter wires executable Q2DmhdFoam and digitized Vetcha data "
            "into LMX validation artifacts. It is a reference-data ingestion "
            "step, not yet a matched LMX-vs-Q2DmhdFoam turbulence parity claim."
        ),
    }
    (out_dir / "q2dmhdfoam_external_reference_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _collect_q2dmhdfoam_profiles(root: Path) -> list[dict[str, object]]:
    profiles: list[dict[str, object]] = []
    for pattern in PROFILE_SAMPLE_PATTERNS:
        for path in sorted(root.glob(pattern)):
            profiles.append(load_q2dmhdfoam_line_profile(path))
    return profiles


def _load_vetcha_digitized_profiles(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    data = np.genfromtxt(path, delimiter=",", skip_header=2)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    labels = ("Vetcha Gr=5e6", "Vetcha Gr=5e7", "Vetcha Gr=1e8")
    grashofs = (5.0e6, 5.0e7, 1.0e8)
    profiles: list[dict[str, object]] = []
    for index, (label, grashof) in enumerate(zip(labels, grashofs, strict=True)):
        x = data[:, 2 * index]
        u = data[:, 2 * index + 1]
        finite = np.isfinite(x) & np.isfinite(u)
        if finite.sum() < 3:
            continue
        order = np.argsort(x[finite])
        profiles.append(
            {
                "source_path": str(path),
                "label": label,
                "position": x[finite][order],
                "raw_coordinate": x[finite][order],
                "velocity": u[finite][order],
                "sample_count": int(finite.sum()),
                "hartmann": 50.0,
                "reynolds": 1.0e4,
                "grashof": grashof,
            }
        )
    return profiles


def _write_turbulence_observables_csv(observables: dict[str, object], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("observable", "value", "source"))
        writer.writeheader()
        source = str(observables.get("source_path", ""))
        for key, value in observables.items():
            if key == "source_path":
                continue
            writer.writerow({"observable": key, "value": value, "source": source})
    return path


if __name__ == "__main__":
    run_q2dmhdfoam_external_reference_adapter()

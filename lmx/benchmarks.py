from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import time
from typing import Any
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .cases import make_hartmann_case
from .solvers import solve_steady


BENCHMARK_B_SPEC_FILES = {
    "B1-fringing-pipe": "alex-b1-pipe.toml",
    "B2-fringing-square": "alex-b2-square.toml",
}


def _repository_root(root: str | Path | None = None) -> Path:
    return Path(root) if root is not None else Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_benchmark_b_spec(spec: dict[str, Any], root: Path) -> None:
    case_id = str(spec.get("id"))
    if case_id not in BENCHMARK_B_SPEC_FILES:
        raise ValueError(f"Unsupported Benchmark B id {case_id!r}")
    if spec.get("schema_version") != 1 or spec.get("status") != "frozen":
        raise ValueError(
            "Benchmark B specification must use schema 1 and status=frozen"
        )
    if spec.get("tolerances_frozen_before_production") is not True:
        raise ValueError("Benchmark B tolerances must be frozen before production")

    expected = {
        "B1-fringing-pipe": (6600.0, 10700.0, 0.027, "pipe_ogrid"),
        "B2-fringing-square": (2900.0, 540.0, 0.07, "square_duct"),
    }[case_id]
    actual = (
        float(spec["physics"]["hartmann_number"]),
        float(spec["physics"]["interaction_parameter"]),
        float(spec["wall"]["wall_conductance_ratio"]),
        str(spec["geometry"]["kind"]),
    )
    if actual != expected:
        raise ValueError(f"Benchmark B frozen parameters differ: {actual!r}")

    sources = spec.get("sources")
    if not isinstance(sources, list) or {source.get("id") for source in sources} != {
        "smolentsev-vv",
        "alex-results-1987",
    }:
        raise ValueError(
            "Benchmark B sources must identify both review and primary ALEX evidence"
        )
    for source in sources:
        if (
            not source.get("pages")
            or not source.get("figures")
            or len(source.get("sha256", "")) != 64
        ):
            raise ValueError(
                "Every Benchmark B source needs pages, figures, and SHA-256"
            )

    field = spec["field"]
    if (
        field.get("representation") != "tabulated monotone interpolation"
        or field.get("no_extrapolation") is not True
        or "exactly divergence free" not in field.get("divergence_model", "")
        or float(field.get("divergence_acceptance", math.inf)) > 1.0e-8
    ):
        raise ValueError("Benchmark B field reconstruction contract is incomplete")

    levels = spec["mesh"].get("levels")
    if not isinstance(levels, list) or [level.get("name") for level in levels] != [
        "coarse",
        "medium",
        "fine",
    ]:
        raise ValueError("Benchmark B requires coarse, medium, and fine mesh levels")
    for key in (
        "axial_stations_min",
        "hartmann_layer_cells_min",
        "side_layer_cells_min",
    ):
        values = [int(level[key]) for level in levels]
        if values != sorted(values) or len(set(values)) != 3:
            raise ValueError(
                f"Benchmark B mesh requirement {key} must increase by level"
            )
    resolution_key = (
        "radial_cells_min"
        if case_id == "B1-fringing-pipe"
        else "cross_section_cells_min"
    )
    resolution = [int(level[resolution_key]) for level in levels]
    if resolution != sorted(resolution) or len(set(resolution)) != 3:
        raise ValueError(
            f"Benchmark B mesh requirement {resolution_key} must increase by level"
        )
    wall = spec["wall"]
    if (
        not str(wall.get("numerical_realization", "")).startswith("explicit volumetric")
        or float(wall.get("nominal_thickness_over_L", 0.0)) <= 0.0
        or float(wall.get("confirmation_thickness_over_L", math.inf))
        >= float(wall.get("nominal_thickness_over_L", 0.0))
        or float(wall.get("thickness_independence_relative_max", math.inf)) > 0.02
    ):
        raise ValueError("Benchmark B thin-wall numerical realization is incomplete")

    acceptance = spec["acceptance"]
    if (
        float(acceptance.get("weighted_rms_max", math.inf)) != 1.0
        or float(acceptance.get("weighted_linf_max", math.inf)) != 2.0
        or acceptance.get("matched_freemhd_case_required") is not True
        or acceptance.get("primary_before_secondary") is not True
    ):
        raise ValueError(
            "Benchmark B uncertainty-aware acceptance contract is incomplete"
        )
    solver = spec["solver"]
    reference_uncertainty = float(spec["reference"]["combined_uncertainty_absolute"])
    steady_uncertainty_fraction = float(
        solver.get("steady_residual_uncertainty_fraction_max", math.inf)
    )
    expected_elliptic_controls = {
        "B1-fringing-pipe": (4000, 1.0e-12, 4000, 1.0e-12),
        "B2-fringing-square": (600, 1.0e-12, 4000, 1.0e-12),
    }[case_id]
    actual_elliptic_controls = (
        int(solver.get("electric_iterations_min", 0)),
        float(solver.get("electric_tolerance_max", math.inf)),
        int(solver.get("projection_iterations_min", 0)),
        float(solver.get("projection_tolerance_max", math.inf)),
    )
    acceleration = solver.get("coupling_acceleration")
    if (
        acceleration not in {"aitken", "anderson", "none"}
        or (
            acceleration == "anderson"
            and int(solver.get("coupling_history_depth", 0)) < 1
        )
        or float(solver.get("coupling_regularization", -1.0)) < 0.0
        or not 0.0 <= float(solver.get("coupling_damping", math.inf)) <= 1.0
        or (
            acceleration == "aitken"
            and (
                float(solver.get("coupling_min_relaxation", 0.0)) <= 0.0
                or float(solver.get("coupling_max_relaxation", 0.0))
                < float(solver.get("coupling_min_relaxation", 0.0))
            )
        )
        or steady_uncertainty_fraction > 0.05
        or float(solver.get("steady_residual_max", math.inf))
        > steady_uncertainty_fraction * reference_uncertainty
        or actual_elliptic_controls != expected_elliptic_controls
        or float(solver.get("tolerance_independence_factor", math.inf)) != 0.5
        or float(
            solver.get("tolerance_independence_uncertainty_fraction_max", math.inf)
        )
        > 0.25
        or float(solver.get("iteration_independence_factor", 0.0)) != 2.0
        or float(
            solver.get("iteration_independence_uncertainty_fraction_max", math.inf)
        )
        > 0.25
        or solver.get("time_or_iteration_independence_required") is not True
    ):
        raise ValueError(
            "Benchmark B tolerance and iteration independence contract is incomplete"
        )
    rights = spec["data_rights"]
    if "extracted numerical facts" not in rights.get("redistribution", ""):
        raise ValueError("Benchmark B reference-data redistribution policy is missing")

    reference = spec["reference"]
    data_path = root / str(reference["data_path"])
    if not data_path.is_file() or _sha256(data_path) != reference.get("data_sha256"):
        raise ValueError("Benchmark B reference data are missing or fail SHA-256")


def load_benchmark_b_spec(
    case_id: str, root: str | Path | None = None
) -> dict[str, Any]:
    """Load and validate one frozen ALEX Benchmark B specification."""

    repository = _repository_root(root)
    try:
        filename = BENCHMARK_B_SPEC_FILES[case_id]
    except KeyError as exc:
        raise ValueError(f"Unsupported Benchmark B id {case_id!r}") from exc
    try:
        import tomllib
    except ImportError:  # pragma: no cover - Python 3.10
        import tomli as tomllib
    with (repository / "benchmarks" / "specs" / filename).open("rb") as handle:
        spec = tomllib.load(handle)
    _validate_benchmark_b_spec(spec, repository)
    return spec


def load_benchmark_b_reference(
    case_id: str, root: str | Path | None = None
) -> dict[str, tuple[float, ...]]:
    """Load checksummed field and pressure anchors for Benchmark B."""

    repository = _repository_root(root)
    spec = load_benchmark_b_spec(case_id, repository)
    path = repository / spec["reference"]["data_path"]
    columns = (
        "x_over_L",
        "b_over_B0",
        "b_uncertainty",
        "pressure_observable",
        "pressure_uncertainty",
    )
    values = {name: [] for name in columns}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != columns:
            raise ValueError(
                "Benchmark B reference columns do not match the frozen schema"
            )
        for row in reader:
            for name in columns:
                value = float(row[name])
                if not math.isfinite(value):
                    raise ValueError("Benchmark B reference values must be finite")
                values[name].append(value)
    if len(values["x_over_L"]) < 10 or any(
        right <= left for left, right in zip(values["x_over_L"], values["x_over_L"][1:])
    ):
        raise ValueError(
            "Benchmark B reference coordinates must be strictly increasing"
        )
    geometry = spec["geometry"]
    if values["x_over_L"][0] != float(geometry["x_over_L_min"]) or values["x_over_L"][
        -1
    ] != float(geometry["x_over_L_max"]):
        raise ValueError("Benchmark B reference data do not span the frozen domain")
    if any(
        value <= 0.0
        for value in values["b_uncertainty"] + values["pressure_uncertainty"]
    ):
        raise ValueError("Benchmark B reference uncertainties must be positive")
    if any(value < 0.0 or value > 1.05 for value in values["b_over_B0"]):
        raise ValueError(
            "Benchmark B normalized magnetic field is outside its physical range"
        )
    return {name: tuple(column) for name, column in values.items()}


def evaluate_benchmark_b_acceptance(
    case_id: str,
    mesh_campaigns: dict[str, dict[str, Any]],
    matched_freemhd: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate the frozen ALEX literature, mesh, and FreeMHD gates.

    Each mesh campaign contains its ``baseline`` run record and the matching
    ``independence`` comparison produced by the Benchmark B runner.  The
    FreeMHD record is deliberately compact: it must identify the case, assert
    an exact input match and a passing comparison, and checksum its full
    external evidence.  Large fields and solver restarts remain release assets.
    """

    spec = load_benchmark_b_spec(case_id)
    levels = tuple(level["name"] for level in spec["mesh"]["levels"])
    missing = [level for level in levels if level not in mesh_campaigns]
    if missing:
        return {
            "case_id": case_id,
            "complete": False,
            "missing_mesh_levels": missing,
            "pass": False,
        }

    reference = load_benchmark_b_reference(case_id)
    reference_x = np.asarray(reference["x_over_L"], dtype=float)
    reference_p = np.asarray(reference["pressure_observable"], dtype=float)
    reference_u = np.asarray(reference["pressure_uncertainty"], dtype=float)
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    literature: dict[str, dict[str, float]] = {}
    independence: dict[str, bool] = {}
    fingerprints = set()

    for level in levels:
        campaign = mesh_campaigns[level]
        record = campaign.get("baseline", {})
        comparison = campaign.get("independence", {})
        fingerprint = str(campaign.get("source_fingerprint", ""))
        if not fingerprint or record.get("source_fingerprint") != fingerprint:
            raise ValueError(f"Benchmark B {level} source provenance does not match")
        fingerprints.add(fingerprint)
        if record.get("case_id") != case_id or record.get("mesh_level") != level:
            raise ValueError(f"Benchmark B {level} baseline metadata do not match")
        x = np.asarray(record.get("x_over_L"), dtype=float)
        observable = np.asarray(record.get("primary_observable"), dtype=float)
        if (
            x.ndim != 1
            or observable.shape != x.shape
            or x.size < 2
            or not np.all(np.isfinite(x))
            or not np.all(np.isfinite(observable))
            or not np.all(np.diff(x) > 0.0)
            or x[0] < reference_x[0]
            or x[-1] > reference_x[-1]
        ):
            raise ValueError(f"Benchmark B {level} baseline curve is invalid")
        expected = np.interp(x, reference_x, reference_p)
        uncertainty = np.interp(x, reference_x, reference_u)
        weighted = (observable - expected) / uncertainty
        reference_integral = float(np.trapezoid(expected, x))
        literature[level] = {
            "weighted_rms": float(np.sqrt(np.mean(weighted**2))),
            "weighted_linf": float(np.max(np.abs(weighted))),
            "integrated_pressure_relative_error": float(
                abs(np.trapezoid(observable, x) - reference_integral)
                / max(abs(reference_integral), float(np.trapezoid(uncertainty, x)))
            ),
        }
        curves[level] = (x, observable)
        independence[level] = bool(
            comparison.get("case_id") == case_id
            and comparison.get("complete") is True
            and comparison.get("pass") is True
        )
    if len(fingerprints) != 1:
        raise ValueError("Benchmark B mesh campaigns use different source fingerprints")

    uncertainty_floor = float(spec["reference"]["combined_uncertainty_absolute"])

    def relative_change(coarser: str, finer: str) -> float:
        coarse_x, coarse_y = curves[coarser]
        fine_x, fine_y = curves[finer]
        keep = (fine_x >= coarse_x[0]) & (fine_x <= coarse_x[-1])
        fine_y = fine_y[keep]
        delta = np.interp(fine_x[keep], coarse_x, coarse_y) - fine_y
        return float(
            np.linalg.norm(delta)
            / max(np.linalg.norm(fine_y), uncertainty_floor * np.sqrt(fine_y.size))
        )

    coarse_medium = relative_change(levels[0], levels[1])
    medium_fine = relative_change(levels[1], levels[2])
    finest = literature[levels[-1]]
    evidence_sha = str((matched_freemhd or {}).get("source_sha256", ""))
    exact_freemhd = bool(
        matched_freemhd
        and matched_freemhd.get("case_id") == case_id
        and matched_freemhd.get("exact_case_match") is True
        and matched_freemhd.get("pass") is True
        and len(evidence_sha) == 64
        and all(character in "0123456789abcdef" for character in evidence_sha.lower())
    )
    acceptance = spec["acceptance"]
    gates = {
        "all_mesh_independence": all(independence.values()),
        "weighted_rms": finest["weighted_rms"] <= float(acceptance["weighted_rms_max"]),
        "weighted_linf": finest["weighted_linf"]
        <= float(acceptance["weighted_linf_max"]),
        "integrated_pressure": finest["integrated_pressure_relative_error"]
        <= float(acceptance["integrated_pressure_relative_error_max"]),
        "finest_mesh_change": medium_fine
        <= float(acceptance["finest_mesh_change_relative_max"]),
        "monotonic_or_asymptotic_refinement": (
            literature[levels[2]]["weighted_rms"]
            <= literature[levels[1]]["weighted_rms"]
            <= literature[levels[0]]["weighted_rms"]
        )
        or medium_fine <= coarse_medium,
        "matched_freemhd": exact_freemhd,
    }
    return {
        "case_id": case_id,
        "complete": matched_freemhd is not None,
        "missing_mesh_levels": [],
        "literature": literature,
        "independence": independence,
        "mesh_change_relative": {
            "coarse_to_medium": coarse_medium,
            "medium_to_fine": medium_fine,
        },
        "freemhd": matched_freemhd,
        "gates": gates,
        "pass": all(gates.values()),
    }


def build_benchmark_b_field_profile(
    case_id: str,
    *,
    axial_stations: int,
    root: str | Path | None = None,
):
    """Reconstruct the frozen ALEX field on cell-centred axial stations.

    The computational coordinate starts at the upstream end of the published
    domain, while the returned profile retains the literature coordinate
    ``x/L`` (whose zero is the magnet pole face).  Linear interpolation is
    shape preserving for the frozen monotone anchors and never extrapolates.
    """

    if axial_stations < 2:
        raise ValueError("axial_stations must be at least 2")
    spec = load_benchmark_b_spec(case_id, root)
    reference = load_benchmark_b_reference(case_id, root)
    x_min = float(spec["geometry"]["x_over_L_min"])
    x_max = float(spec["geometry"]["x_over_L_max"])
    dx = (x_max - x_min) / axial_stations
    x_over_l = jnp.linspace(
        x_min + 0.5 * dx,
        x_max - 0.5 * dx,
        axial_stations,
    )
    anchors_x = jnp.asarray(reference["x_over_L"], dtype=float)
    anchors_b = jnp.asarray(reference["b_over_B0"], dtype=float)
    if float(x_over_l[0]) < float(anchors_x[0]) or float(x_over_l[-1]) > float(
        anchors_x[-1]
    ):
        raise ValueError("ALEX field reconstruction cannot extrapolate")
    field_scale = jnp.interp(x_over_l, anchors_x, anchors_b)
    if bool(jnp.any(jnp.diff(field_scale) > 1.0e-12)):
        raise ValueError("ALEX field reconstruction must remain monotone")

    from ._fringing_types import FringingProfile

    return FringingProfile(x=x_over_l, field_scale=field_scale, axis="y")


def build_benchmark_b_problem(
    case_id: str,
    *,
    mesh_level: str,
    root: str | Path | None = None,
    wall_realization: str = "nominal",
    num_devices: int | None = None,
):
    """Build one immutable nondimensional ALEX B1/B2 production problem.

    A sharded mesh rounds the frozen axial minimum upward to the nearest
    multiple of ``num_devices``; cross-section resolution and physics remain
    unchanged.
    """

    from ._fringing_types import ExtrudedInductionlessProblem
    from .specs import (
        BoundaryCondition,
        CaseSpec,
        GeometrySpec,
        MagneticFieldSpec,
        RegionSpec,
        SolverConfig,
        TimeStepperConfig,
    )

    spec = load_benchmark_b_spec(case_id, root)
    levels = {str(level["name"]): level for level in spec["mesh"]["levels"]}
    if mesh_level not in levels:
        raise ValueError("mesh_level must be 'coarse', 'medium', or 'fine'")
    if wall_realization not in {"nominal", "confirmation"}:
        raise ValueError("wall_realization must be 'nominal' or 'confirmation'")
    level = levels[mesh_level]
    wall = spec["wall"]
    thickness = float(wall[f"{wall_realization}_thickness_over_L"])
    conductance = float(wall["wall_conductance_ratio"])
    wall_conductivity = conductance / thickness
    ha = float(spec["physics"]["hartmann_number"])
    interaction = float(spec["physics"]["interaction_parameter"])
    reynolds = ha**2 / interaction
    viscosity = 1.0 / reynolds
    peak_field = math.sqrt(interaction)
    if num_devices is not None and num_devices < 1:
        raise ValueError("num_devices must be positive")
    nx_min = int(level["axial_stations_min"])
    nx = (
        math.ceil(nx_min / num_devices) * num_devices
        if num_devices is not None
        else nx_min
    )
    wall_cells = int(level["side_layer_cells_min"])
    x_min = float(spec["geometry"]["x_over_L_min"])
    length = float(spec["geometry"]["x_over_L_max"]) - x_min

    if case_id == "B1-fringing-pipe":
        geometry = GeometrySpec(
            kind="pipe_ogrid",
            width=2.0,
            height=2.0,
            radius=1.0,
            length=length,
            axial_origin=x_min,
            nx=nx,
            nr=int(level["radial_cells_min"]),
            ntheta=int(level["azimuthal_cells_min"]),
            wall_thickness=(thickness,) * 4,
            wall_cells=(wall_cells,) * 4,
            target_ha=ha,
            hartmann_layer_cells=int(level["hartmann_layer_cells_min"]),
        )
    else:
        cross_cells = int(level["cross_section_cells_min"])
        geometry = GeometrySpec(
            kind="layered_duct",
            width=2.0,
            height=2.0,
            length=length,
            axial_origin=x_min,
            nx=nx,
            ny=cross_cells,
            nz=cross_cells,
            wall_thickness=(thickness,) * 4,
            wall_cells=(wall_cells,) * 4,
            target_ha=ha,
            hartmann_layer_cells=int(level["hartmann_layer_cells_min"]),
        )

    case = CaseSpec(
        name=f"alex_{case_id.lower()}_{mesh_level}_{wall_realization}",
        geometry=geometry,
        regions=(
            RegionSpec("fluid", "fluid", 1.0, 1.0, viscosity),
            RegionSpec(
                "conducting_wall", "solid", wall_conductivity, 1.0, viscosity, thickness
            ),
        ),
        magnetic_field=MagneticFieldSpec(kind="constant", value=(0.0, peak_field, 0.0)),
        boundary_conditions=(
            BoundaryCondition("walls", "no_slip"),
            BoundaryCondition(
                "uniform_conducting_wall",
                "conducting_wall",
                region="conducting_wall",
                side="left,right,bottom,top",
            ),
        ),
        time_stepper=TimeStepperConfig(
            dt=0.01,
            t_final=10.0,
            max_steps=1000,
            potential_iterations=400,
            steady_tolerance=float(spec["solver"]["steady_residual_max"]),
        ),
        solver=SolverConfig(
            kind="extruded_inductionless",
            linear_solver="auto",
            coupling_iterations=64,
            coupling_tolerance=float(spec["solver"]["steady_residual_max"]),
            coupling_acceleration=str(spec["solver"]["coupling_acceleration"]),
            coupling_history_depth=int(spec["solver"]["coupling_history_depth"]),
            coupling_regularization=float(spec["solver"]["coupling_regularization"]),
            coupling_damping=float(spec["solver"]["coupling_damping"]),
            coupling_min_relaxation=float(
                spec["solver"].get("coupling_min_relaxation", 0.05)
            ),
            coupling_max_relaxation=float(
                spec["solver"].get("coupling_max_relaxation", 100.0)
            ),
        ),
        forcing=0.0,
        initial_velocity=1.0,
        reference_pressure_gradient=-1.0,
        notes=(
            f"Frozen {case_id} nondimensionalization: Re=Ha^2/N={reynolds:.12g}; "
            "reported axial pressure loss is -dp/dx and transverse pressure is high-z minus low-z."
        ),
    )
    profile = build_benchmark_b_field_profile(case_id, axial_stations=nx, root=root)
    return ExtrudedInductionlessProblem(case=case, profile=profile)


def benchmark_b_pressure_observable(solution, case_id: str) -> jnp.ndarray:
    """Return the frozen primary pressure observable in ALEX normalization."""

    spec = load_benchmark_b_spec(case_id)
    interaction = float(spec["physics"]["interaction_parameter"])
    if case_id == "B1-fringing-pipe":
        gradient = jnp.asarray(solution.bundle.axial_pressure_loss_gradient)
        if gradient.size == 0:
            raise ValueError("B1 requires the direct axial pressure-loss gradient")
        x = jnp.asarray(getattr(solution.bundle, "x", jnp.zeros((0,))))
        plateau_start = float(spec["reference"]["downstream_plateau_x_over_L_min"])
        if x.shape == gradient.shape and bool(jnp.any(x >= plateau_start)):
            downstream = jnp.nanmean(jnp.where(x >= plateau_start, gradient, jnp.nan))
        else:
            downstream = jnp.mean(gradient[-max(3, gradient.size // 10) :])
        return gradient / interaction - downstream / interaction
    difference = jnp.asarray(solution.bundle.transverse_pressure_difference)
    if difference.size == 0:
        raise ValueError("B2 requires the direct transverse pressure difference")
    x = jnp.asarray(getattr(solution.bundle, "x", jnp.zeros((0,))))
    if x.shape == difference.shape:
        upstream = float(spec["reference"]["baseline_x_over_L_upstream_max"])
        downstream = float(spec["reference"]["baseline_x_over_L_downstream_min"])
        baseline_mask = (x <= upstream) | (x >= downstream)
        if bool(jnp.any(baseline_mask)):
            baseline = jnp.nanmean(jnp.where(baseline_mask, difference, jnp.nan))
            difference = difference - baseline
    return difference / interaction


def benchmark_solver(
    repeats: int = 3, ha: float = 20.0, ny: int = 48, nz: int = 48
) -> dict[str, float | str]:
    case = make_hartmann_case(ha=ha, ny=ny, nz=nz)
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        solve_steady(case)
        timings.append(time.perf_counter() - start)
    cold = timings[0]
    warm = min(timings[1:] or timings)
    return {
        "case": case.name,
        "ha": ha,
        "ny": float(ny),
        "nz": float(nz),
        "repeats": float(repeats),
        "cold_seconds": cold,
        "warm_seconds": warm,
        "mean_seconds": sum(timings) / len(timings),
        "backend": jax.default_backend(),
        "device_kind": jax.devices()[0].device_kind,
        "jax_version": jax.__version__,
        "python_version": platform.python_version(),
    }


def write_benchmark_report(report: dict[str, float | str], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    return path

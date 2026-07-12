#!/usr/bin/env python3
"""Run the checksummed Samper Table I high-Ha integral-flow campaign."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterable

from lmx import fully_developed_power_balance, solve_steady
from lmx.cases import make_hunt_case, make_shercliff_case
from lmx.freemhd import load_samper_table_i
from lmx.validation import (
    duct_layer_resolution_gate,
    estimate_observed_order,
    validation_summary,
)


DEFAULT_MESH_LEVELS = ((63, 63), (79, 79), (99, 99))
FLOW_ERROR_TARGET = 0.01
BALANCE_TARGET = 0.001
MESH_CHANGE_TARGET = 0.0025
STEADY_RESIDUAL_TARGET = 1.0e-9
STEADY_RELATIVE_UPDATE_TARGET = 2.0e-8
POTENTIAL_RESIDUAL_STOPPING_CEILING = 1.0e-2
POTENTIAL_CURRENT_GATE_SAFETY = 0.1


def parse_mesh_levels(value: str) -> tuple[tuple[int, int], ...]:
    """Parse ``NYxNZ,NYxNZ,...`` and enforce a three-level refinement ladder."""

    levels: list[tuple[int, int]] = []
    for token in value.split(","):
        left, separator, right = token.lower().strip().partition("x")
        if not separator:
            raise ValueError(f"Invalid mesh level {token!r}; expected NYxNZ")
        try:
            level = (int(left), int(right))
        except ValueError as exc:
            raise ValueError(
                f"Invalid mesh level {token!r}; expected integer dimensions"
            ) from exc
        if min(level) < 8:
            raise ValueError("Samper mesh dimensions must each be at least 8")
        levels.append(level)
    if len(levels) < 3:
        raise ValueError("Samper validation requires at least three mesh levels")
    spacings = [1.0 / math.sqrt(ny * nz) for ny, nz in levels]
    if any(coarse <= fine for coarse, fine in zip(spacings, spacings[1:])):
        raise ValueError("Samper mesh levels must be monotonically refined")
    return tuple(levels)


def select_rows(
    table: dict[str, Any], *, case_kinds: Iterable[str], hartmann_numbers: Iterable[int]
) -> list[dict[str, Any]]:
    requested_cases = set(case_kinds)
    requested_ha = set(hartmann_numbers)
    rows = [
        row
        for row in table["cases"]
        if row["case_kind"] in requested_cases
        and int(row["hartmann_number"]) in requested_ha
    ]
    expected = {(case, ha) for case in requested_cases for ha in requested_ha}
    observed = {(row["case_kind"], int(row["hartmann_number"])) for row in rows}
    if observed != expected:
        raise ValueError(
            f"Table I does not contain requested cases: {sorted(expected - observed)}"
        )
    return sorted(
        rows, key=lambda row: (str(row["case_kind"]), int(row["hartmann_number"]))
    )


def build_samper_case(
    row: dict[str, Any],
    *,
    ny: int,
    nz: int,
    wall_thickness: float = 0.01,
    wall_cells: int = 4,
    max_steps: int = 12,
    potential_iterations: int = 4000,
    coupling_iterations: int = 512,
    linear_solver: str = "cg",
):
    """Build Table I in units where the reported dimensionless flow equals Q."""

    ha = float(row["hartmann_number"])
    common = {
        "ha": ha,
        "width": 2.0,
        "height": 2.0,
        "ny": ny,
        "nz": nz,
        "density": 1.0,
        "viscosity": 1.0,
    }
    if row["case_kind"] == "shercliff":
        case = make_shercliff_case(conductivity=1.0, **common)
    elif row["case_kind"] == "hunt":
        case = make_hunt_case(
            fluid_conductivity=1.0,
            wall_conductance_ratio=float(row["hartmann_wall_conductance"]),
            wall_thickness=wall_thickness,
            wall_cells=wall_cells,
            insulator_thickness=wall_thickness,
            insulator_cells=wall_cells,
            insulator_conductivity_ratio=0.0,
            **common,
        )
    else:
        raise ValueError(f"Unsupported Samper case kind {row['case_kind']!r}")
    steady_tolerance = min(
        STEADY_RESIDUAL_TARGET,
        STEADY_RELATIVE_UPDATE_TARGET * float(row["analytical_flow_rate"]),
    )
    controls = replace(
        case.time_stepper,
        max_steps=max_steps,
        potential_iterations=potential_iterations,
        potential_tolerance=(
            POTENTIAL_CURRENT_GATE_SAFETY
            * BALANCE_TARGET
            * float(row["analytical_flow_rate"])
            * ha
            / 4.0
        ),
        steady_potential_tolerance=POTENTIAL_RESIDUAL_STOPPING_CEILING,
        steady_tolerance=steady_tolerance,
        current_reconstruction="face_averaged",
        velocity_update_limit=0.1,
    )
    solver = replace(
        case.solver,
        linear_solver=linear_solver,
        coupling_iterations=coupling_iterations,
        coupling_tolerance=1.0e-8,
        coupling_acceleration="anderson",
        coupling_min_relaxation=0.05,
        coupling_max_relaxation=100.0,
        coupling_history_depth=6,
        coupling_regularization=1.0e-10,
        coupling_damping=1.0,
    )
    return replace(
        case,
        name=f"samper_table_i_{row['case_kind']}_ha{int(ha)}_{ny}x{nz}",
        forcing=1.0,
        reference_pressure_gradient=-1.0,
        time_stepper=controls,
        solver=solver,
    )


def dimensionless_flow_rate(case, solution) -> float:
    """Convert LMX flow to Table I's Q-tilde definition."""

    if not solution.diagnostics.volumetric_flow_rate_history.size:
        raise ValueError("Solution does not contain a volumetric-flow history")
    fluid = next(region for region in case.regions if region.kind == "fluid")
    half_width = 0.5 * float(case.geometry.width)
    half_height = 0.5 * float(case.geometry.height)
    forcing = abs(float(solution.diagnostics.applied_forcing_history[-1]))
    if forcing <= 0.0:
        raise ValueError(
            "Table I conversion requires nonzero applied pressure gradient"
        )
    flow = float(solution.diagnostics.volumetric_flow_rate_history[-1])
    return (
        flow
        * float(fluid.density)
        * float(fluid.viscosity)
        / (half_width * half_height**3 * forcing)
    )


def _balance_summary(case, solution, flow: float) -> dict[str, Any]:
    diagnostics = validation_summary(solution, case.name, case.geometry.target_ha)
    fluid = next(region for region in case.regions if region.kind == "fluid")
    bmag = math.sqrt(sum(float(value) ** 2 for value in case.magnetic_field.value))
    mean_velocity = abs(flow) / (
        float(case.geometry.width) * float(case.geometry.height)
    )
    current_scale = max(float(fluid.conductivity) * mean_velocity * bmag, 1.0e-30)
    length_scale = 0.5 * float(case.geometry.width)
    current = {
        "div_current_max_normalized": float(diagnostics["div_current_max"])
        / (current_scale / length_scale),
        "charge_balance_normalized": float(diagnostics["charge_balance_residual"])
        / (current_scale / length_scale),
        "interface_current_residual_normalized": float(
            diagnostics["interface_current_residual"]
        )
        / current_scale,
        "acceptance_target": BALANCE_TARGET,
    }
    power = fully_developed_power_balance(case, solution)
    power["acceptance_target"] = BALANCE_TARGET
    current["pass"] = (
        max(
            current["div_current_max_normalized"],
            current["charge_balance_normalized"],
            current["interface_current_residual_normalized"],
        )
        <= BALANCE_TARGET
    )
    power["pass"] = (
        max(
            power["electrical_power_relative_error"],
            power["mechanical_power_relative_error"],
        )
        <= BALANCE_TARGET
    )
    return {"current": current, "power": power}


def summarize_refinement(levels: list[dict[str, Any]]) -> dict[str, Any]:
    """Allocate one quarter of the 1% error budget to finest-mesh change."""

    if len(levels) < 3:
        raise ValueError("Refinement summary requires at least three levels")
    errors = [float(level["analytical_relative_error"]) for level in levels]
    previous = float(levels[-2]["q_tilde"])
    finest = float(levels[-1]["q_tilde"])
    finest_change = abs(finest - previous) / max(abs(finest), 1.0e-30)
    coarse_h = 1.0 / math.sqrt(math.prod(levels[0]["mesh"]))
    fine_h = 1.0 / math.sqrt(math.prod(levels[-1]["mesh"]))
    order = estimate_observed_order(errors[0], errors[-1], coarse_h, fine_h)
    monotonic = all(coarse >= fine for coarse, fine in zip(errors, errors[1:]))
    return {
        "analytical_error_monotonic": monotonic,
        "observed_order_against_analytical": order,
        "finest_mesh_change_relative": finest_change,
        "finest_mesh_change_target": MESH_CHANGE_TARGET,
        "pass": monotonic
        and order is not None
        and order >= 0.5
        and finest_change <= MESH_CHANGE_TARGET,
    }


def _solver_convergence(
    solver: dict[str, Any], *, steady_target: float = STEADY_RESIDUAL_TARGET
) -> dict[str, Any]:
    return {
        "steady_residual_target": steady_target,
        "potential_residual_stopping_ceiling": POTENTIAL_RESIDUAL_STOPPING_CEILING,
        "potential_acceptance_gate": "normalized conservative current balance",
        "pass": float(solver["residual"]) <= steady_target,
    }


def _finest_level_pass(record: dict[str, Any]) -> bool:
    fine = record["levels"][-1]
    return bool(
        fine["analytical_relative_error"] <= FLOW_ERROR_TARGET
        and record["refinement"]["pass"]
        and fine["solver_convergence"]["pass"]
        and fine["layer_resolution"]["layer_resolution_pass"]
        and fine["balances"]["current"]["pass"]
        and fine["balances"]["power"]["pass"]
    )


def reassess_campaign(summary: dict[str, Any]) -> dict[str, Any]:
    """Re-evaluate gates from stored raw metrics without rerunning a solve."""

    for record in summary.get("records", []):
        steady_target = min(
            STEADY_RESIDUAL_TARGET,
            STEADY_RELATIVE_UPDATE_TARGET * float(record["analytical_flow_rate"]),
        )
        for level in record["levels"]:
            level["solver_convergence"] = _solver_convergence(
                level["solver"], steady_target=steady_target
            )
        record["finest_level_pass"] = _finest_level_pass(record)
    summary["potential_residual_stopping_ceiling"] = POTENTIAL_RESIDUAL_STOPPING_CEILING
    summary.pop("potential_residual_target", None)
    summary["research_grade_validation_pass"] = bool(summary.get("records")) and all(
        record["finest_level_pass"] for record in summary["records"]
    )
    return summary


def run_row(
    row: dict[str, Any],
    mesh_levels: tuple[tuple[int, int], ...],
    *,
    checkpoint: Callable[[list[dict[str, Any]]], None] | None = None,
    initial_levels: Iterable[dict[str, Any]] = (),
    **controls,
) -> dict[str, Any]:
    levels = [dict(level) for level in initial_levels]
    completed_meshes = tuple(
        tuple(int(value) for value in level["mesh"]) for level in levels
    )
    if completed_meshes != mesh_levels[: len(completed_meshes)]:
        raise ValueError(
            "Checkpoint mesh levels are not a prefix of the requested ladder"
        )
    for ny, nz in mesh_levels[len(levels) :]:
        print(
            f"running {row['case_kind']} Ha={row['hartmann_number']} mesh={ny}x{nz}",
            flush=True,
        )
        case = build_samper_case(row, ny=ny, nz=nz, **controls)
        solution = solve_steady(case)
        q_tilde = dimensionless_flow_rate(case, solution)
        analytical = float(row["analytical_flow_rate"])
        published = float(row["published_numerical_flow_rate"])
        physical_flow = float(solution.diagnostics.volumetric_flow_rate_history[-1])
        balances = _balance_summary(case, solution, physical_flow)
        layer = duct_layer_resolution_gate(case, solution.mesh)
        solver = validation_summary(solution, case.name, float(row["hartmann_number"]))
        solver_convergence = _solver_convergence(
            solver, steady_target=float(case.time_stepper.steady_tolerance)
        )
        levels.append(
            {
                "mesh": [ny, nz],
                "q_tilde": q_tilde,
                "analytical_relative_error": abs(q_tilde - analytical) / analytical,
                "published_numerical_relative_error": abs(q_tilde - published)
                / published,
                "layer_resolution": layer,
                "balances": balances,
                "solver": solver,
                "solver_convergence": solver_convergence,
            }
        )
        if checkpoint is not None:
            checkpoint(levels)
    refinement = summarize_refinement(levels)
    record = {
        "case_kind": row["case_kind"],
        "hartmann_number": int(row["hartmann_number"]),
        "hartmann_wall_conductance": float(row["hartmann_wall_conductance"]),
        "analytical_flow_rate": analytical,
        "published_numerical_flow_rate": published,
        "levels": levels,
        "refinement": refinement,
        "finest_level_pass": False,
    }
    record["finest_level_pass"] = _finest_level_pass(record)
    return record


def _write_checkpoint(output: Path, summary: dict[str, Any]) -> None:
    """Atomically replace a campaign checkpoint with canonical JSON."""

    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(output)


def _implementation_fingerprint() -> dict[str, str]:
    runner = Path(__file__).resolve()
    repository = runner.parent.parent
    solver_sources = (
        repository / "lmx" / "_solvers.py",
        repository / "lmx" / "linear.py",
    )
    solver_digest = hashlib.sha256()
    for source in solver_sources:
        solver_digest.update(source.relative_to(repository).as_posix().encode("utf-8"))
        solver_digest.update(b"\0")
        solver_digest.update(source.read_bytes())
    return {
        "runner_sha256": hashlib.sha256(runner.read_bytes()).hexdigest(),
        "solver_core_sha256": solver_digest.hexdigest(),
        "lmx_version": importlib.metadata.version("lmx"),
        "solvax_version": importlib.metadata.version("solvax"),
    }


def _resume_summary(
    output: Path,
    expected: dict[str, Any],
    requested_keys: set[tuple[str, int]],
) -> dict[str, Any]:
    if not output.is_file():
        raise ValueError(f"Cannot resume missing campaign checkpoint: {output}")
    try:
        summary = json.loads(output.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read campaign checkpoint: {output}") from exc
    contract_keys = (
        "schema_version",
        "reference",
        "normalization",
        "mesh_levels",
        "flow_error_target",
        "balance_target",
        "finest_mesh_change_target",
        "steady_residual_target",
        "steady_relative_update_target",
        "potential_residual_stopping_ceiling",
        "controls",
        "implementation",
    )
    mismatches = [key for key in contract_keys if summary.get(key) != expected.get(key)]
    if mismatches:
        raise ValueError(
            f"Campaign checkpoint contract mismatch: {', '.join(mismatches)}"
        )
    completed_keys = [
        (str(record["case_kind"]), int(record["hartmann_number"]))
        for record in summary.get("records", [])
    ]
    if len(completed_keys) != len(set(completed_keys)):
        raise ValueError("Campaign checkpoint contains duplicate completed rows")
    if not set(completed_keys).issubset(requested_keys):
        raise ValueError(
            "Campaign checkpoint contains rows outside the requested campaign"
        )
    active = summary.get("active_record")
    if active is not None:
        active_key = (str(active["case_kind"]), int(active["hartmann_number"]))
        if active_key not in requested_keys or active_key in completed_keys:
            raise ValueError("Campaign checkpoint contains an inconsistent active row")
    return summary


def run_campaign(
    *,
    output: Path,
    case_kinds: tuple[str, ...],
    hartmann_numbers: tuple[int, ...],
    mesh_levels: tuple[tuple[int, int], ...] = DEFAULT_MESH_LEVELS,
    resume: bool = False,
    **controls,
) -> dict[str, Any]:
    table = load_samper_table_i()
    rows = select_rows(table, case_kinds=case_kinds, hartmann_numbers=hartmann_numbers)
    summary = {
        "schema_version": 1,
        "reference": {
            "id": table["id"],
            "path": table["path"],
            "sha256": table["sha256"],
        },
        "normalization": table["definition"],
        "mesh_levels": [list(level) for level in mesh_levels],
        "flow_error_target": FLOW_ERROR_TARGET,
        "balance_target": BALANCE_TARGET,
        "finest_mesh_change_target": MESH_CHANGE_TARGET,
        "steady_residual_target": STEADY_RESIDUAL_TARGET,
        "steady_relative_update_target": STEADY_RELATIVE_UPDATE_TARGET,
        "potential_residual_stopping_ceiling": POTENTIAL_RESIDUAL_STOPPING_CEILING,
        "controls": controls,
        "implementation": _implementation_fingerprint(),
        "records": [],
        "research_grade_validation_pass": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    requested_keys = {
        (str(row["case_kind"]), int(row["hartmann_number"])) for row in rows
    }
    if resume:
        summary = _resume_summary(output, summary, requested_keys)
    completed_keys = {
        (str(record["case_kind"]), int(record["hartmann_number"]))
        for record in summary["records"]
    }
    for row in rows:
        row_key = (str(row["case_kind"]), int(row["hartmann_number"]))
        if row_key in completed_keys:
            print(
                f"resuming: skipping completed {row_key[0]} Ha={row_key[1]}", flush=True
            )
            continue
        active = summary.get("active_record")
        initial_levels: list[dict[str, Any]] = []
        if active is not None:
            active_key = (str(active["case_kind"]), int(active["hartmann_number"]))
            if active_key != row_key:
                raise ValueError(
                    "Campaign checkpoint active row is not the next requested row"
                )
            initial_levels = list(active.get("levels", []))
            print(
                f"resuming {row_key[0]} Ha={row_key[1]} after {len(initial_levels)} mesh levels",
                flush=True,
            )

        def checkpoint(levels: list[dict[str, Any]]) -> None:
            summary["active_record"] = {
                "case_kind": row["case_kind"],
                "hartmann_number": int(row["hartmann_number"]),
                "levels": levels,
                "complete": False,
            }
            _write_checkpoint(output, summary)

        summary["records"].append(
            run_row(
                row,
                mesh_levels,
                checkpoint=checkpoint,
                initial_levels=initial_levels,
                **controls,
            )
        )
        summary.pop("active_record", None)
        _write_checkpoint(output, summary)
    summary["research_grade_validation_pass"] = all(
        record["finest_level_pass"] for record in summary["records"]
    )
    _write_checkpoint(output, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/samper/table-i-summary.json")
    )
    parser.add_argument("--cases", default="shercliff,hunt")
    parser.add_argument("--ha", default="500,5000,10000,15000")
    parser.add_argument("--meshes", default="63x63,79x79,99x99")
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--potential-iterations", type=int, default=4000)
    parser.add_argument("--coupling-iterations", type=int, default=512)
    parser.add_argument("--linear-solver", choices=("cg", "solvax_pcg"), default="cg")
    parser.add_argument("--wall-thickness", type=float, default=0.01)
    parser.add_argument("--wall-cells", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cases = tuple(value.strip() for value in args.cases.split(",") if value.strip())
    if not cases or any(case not in {"shercliff", "hunt"} for case in cases):
        parser.error("--cases must contain shercliff and/or hunt")
    try:
        ha_values = tuple(int(value) for value in args.ha.split(","))
        meshes = parse_mesh_levels(args.meshes)
        summary = run_campaign(
            output=args.output,
            case_kinds=cases,
            hartmann_numbers=ha_values,
            mesh_levels=meshes,
            max_steps=args.max_steps,
            potential_iterations=args.potential_iterations,
            coupling_iterations=args.coupling_iterations,
            linear_solver=args.linear_solver,
            wall_thickness=args.wall_thickness,
            wall_cells=args.wall_cells,
            resume=args.resume,
        )
    except ValueError as exc:
        parser.error(str(exc))
    return 0 if summary["research_grade_validation_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

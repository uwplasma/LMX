from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import jax.numpy as jnp

from .benchmarks import benchmark_solver, write_benchmark_report
from .cases import make_hartmann_case, make_hunt_case, make_shercliff_case
from .config import LoggingSpec, RunConfig, load_run_config
from .io import load_restart_bundle, validate_restart_bundle, write_paraview, write_restart_npz, write_solution_outputs
from .runtime_logging import RestartLogInfo, StreamingSolverLogger, default_log_path
from .solvers import _build_mesh, solve_steady, solve_transient
from .validation import (
    closed_channel_validation,
    combined_profile_error,
    extract_centerline,
    extract_midplane_profile,
    hartmann_acceptance,
    hartmann_validation,
    processed_slice_validation,
    validation_summary,
    write_acceptance_report,
    write_analytic_comparison,
    write_closed_channel_validation,
    write_metrics_json,
    write_processed_slice_validation,
    write_profile_csv,
)


class _EmptyDiagnostics:
    potential_residual_history = jnp.asarray([])
    potential_iterations_history = jnp.asarray([])
    linear_residual_history = jnp.asarray([])
    linear_iterations_history = jnp.asarray([])
    volumetric_flow_rate_history = jnp.asarray([])
    mean_current_magnitude_history = jnp.asarray([])
    lorentz_power_history = jnp.asarray([])
    div_current_max_history = jnp.asarray([])
    charge_balance_residual_history = jnp.asarray([])
    gauge_residual_history = jnp.asarray([])
    interface_current_residual_history = jnp.asarray([])


def _portable_path(path: str | Path, *, relative_to: str | Path | None = None) -> str:
    candidate = Path(path)
    base = Path(relative_to) if relative_to is not None else Path.cwd()
    try:
        return str(candidate.relative_to(base))
    except ValueError:
        try:
            return str(candidate.resolve().relative_to(base.resolve()))
        except ValueError:
            return candidate.name if candidate.name else str(candidate)


def _build_case(args: argparse.Namespace):
    if args.case == "hartmann":
        return make_hartmann_case(ha=args.ha, output_dir=args.output)
    if args.case == "shercliff":
        return make_shercliff_case(ha=args.ha, output_dir=args.output)
    if args.case == "hunt":
        return make_hunt_case(ha=args.ha, output_dir=args.output)
    raise ValueError(args.case)


def _solve_case_with_optional_logger(
    case,
    *,
    solve_mode: str,
    logger=None,
    initial_state=None,
    initial_diagnostics=None,
    append_diagnostics: bool = False,
    restart_info: RestartLogInfo | None = None,
):
    if solve_mode == "transient":
        try:
            return solve_transient(
                case,
                logger=logger,
                initial_state=initial_state,
                initial_diagnostics=initial_diagnostics,
                append_diagnostics=append_diagnostics,
                restart_info=restart_info,
            )
        except TypeError:
            return solve_transient(case)
    try:
        return solve_steady(
            case,
            logger=logger,
            initial_state=initial_state,
            initial_diagnostics=initial_diagnostics,
            append_diagnostics=append_diagnostics,
            restart_info=restart_info,
        )
    except TypeError:
        return solve_steady(case)


def _runtime_summary(solution, case, out_dir: Path, outputs: dict[str, list[Path]], *, restart_info: dict[str, object] | None = None) -> dict[str, object]:
    diag = getattr(solution, "diagnostics", _EmptyDiagnostics())
    geometry = getattr(getattr(case, "geometry", None), "kind", "unknown")
    u_field = getattr(solution.state, "u", jnp.asarray([0.0]))
    def _latest(name: str) -> float | None:
        history = getattr(diag, name, jnp.asarray([]))
        return float(history[-1]) if getattr(history, "size", 0) else None
    summary = {
        "case": case.name,
        "geometry": geometry,
        "solver_kind": getattr(getattr(case, "solver", None), "kind", "fully_developed_inductionless"),
        "solver_mode": getattr(getattr(case, "solver", None), "mode", "steady"),
        "time": float(solution.state.time),
        "residual": float(solution.state.residual),
        "u_max": float(jnp.max(jnp.abs(u_field))),
        "potential_residual": _latest("potential_residual_history"),
        "potential_iterations_used": _latest("potential_iterations_history"),
        "linear_residual": _latest("linear_residual_history"),
        "linear_iterations_used": _latest("linear_iterations_history"),
        "volumetric_flow_rate": _latest("volumetric_flow_rate_history"),
        "mean_current_magnitude": _latest("mean_current_magnitude_history"),
        "lorentz_power": _latest("lorentz_power_history"),
        "div_current_max": _latest("div_current_max_history"),
        "charge_balance_residual": _latest("charge_balance_residual_history"),
        "gauge_residual": _latest("gauge_residual_history"),
        "interface_current_residual": _latest("interface_current_residual_history"),
        "output": _portable_path(out_dir),
        "generated_files": {
            key: [_portable_path(path) for path in paths]
            for key, paths in outputs.items()
            if paths
        },
    }
    if restart_info is not None:
        summary["restart"] = restart_info
    return summary


def _write_run_summary(summary: dict[str, object], case, out_dir: Path) -> Path | None:
    if not getattr(case.output, "write_json_summary", True):
        return None
    path = out_dir / f"{case.name}_summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    return path


def _run_config(config: RunConfig) -> dict[str, object]:
    case = config.case
    output_dir = getattr(case, "output_dir", None)
    if output_dir is None and getattr(case, "output", None) is not None:
        output_dir = getattr(case.output, "directory", None)
    out_dir = Path(output_dir) if output_dir else Path.cwd() / "out" / case.name
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = StreamingSolverLogger(config.logging) if config.logging.enabled else None
    log_handle = None
    log_path: Path | None = None
    if logger is not None:
        log_path = default_log_path(out_dir, case.name)
        log_handle = open(log_path, "w", encoding="utf-8")
        logger.add_stream(log_handle)
    initial_state = None
    initial_diagnostics = None
    restart_log_info = RestartLogInfo(enabled=False)
    restart_summary: dict[str, object] | None = None
    if config.restart.enabled:
        if config.restart.path is None:
            raise ValueError("Restart is enabled but no restart.path was provided")
        restart_bundle = load_restart_bundle(config.restart.path)
        validate_restart_bundle(restart_bundle, mesh=_build_mesh(case), geometry_kind=case.geometry.kind, case_name=case.name)
        initial_state = restart_bundle.state
        initial_diagnostics = restart_bundle.diagnostics
        restart_log_info = RestartLogInfo(
            enabled=True,
            path=str(restart_bundle.path),
            start_time=float(restart_bundle.state.time),
            reset_histories=config.restart.reset_histories,
        )
        restart_summary = {
            "enabled": True,
            "input": str(restart_bundle.path),
            "start_time": float(restart_bundle.state.time),
            "reset_histories": bool(config.restart.reset_histories),
        }
    try:
        solution = _solve_case_with_optional_logger(
            case,
            solve_mode=config.solve_mode,
            logger=logger,
            initial_state=initial_state,
            initial_diagnostics=initial_diagnostics,
            append_diagnostics=bool(config.restart.enabled and not config.restart.reset_histories),
            restart_info=restart_log_info,
        )
    finally:
        if log_handle is not None:
            log_handle.close()
    outputs = write_solution_outputs(
        solution,
        case,
        out_dir,
        write_npz=getattr(case.output, "write_npz", True),
        write_plots=getattr(case.output, "write_plots", False),
    )
    if config.restart.write_restart:
        restart_filename = config.restart.restart_filename or f"{case.name}_restart.npz"
        restart_path = write_restart_npz(solution, case, out_dir / restart_filename)
        outputs.setdefault("restart", []).append(restart_path)
        if restart_summary is None:
            restart_summary = {"enabled": False}
        restart_summary["output"] = _portable_path(restart_path)
    if log_path is not None:
        outputs.setdefault("log", []).append(log_path)
    if config.input_path is not None and getattr(case.output, "copy_input_file", True):
        copied_input = out_dir / config.input_path.name
        shutil.copy2(config.input_path, copied_input)
        outputs.setdefault("input", []).append(copied_input)
    summary = _runtime_summary(solution, case, out_dir, outputs, restart_info=restart_summary)
    summary_path = _write_run_summary(summary, case, out_dir)
    if summary_path is not None:
        outputs.setdefault("json", []).append(summary_path)
        summary["generated_files"]["json"] = [_portable_path(summary_path)]
    print(json.dumps(summary, indent=2))
    return summary


def run_from_toml(path: str | Path) -> dict[str, object]:
    config = load_run_config(path)
    return _run_config(config)


def main(argv: list[str] | None = None) -> int:
    argv = list(argv) if argv is not None else sys.argv[1:]
    if argv and Path(argv[0]).suffix == ".toml":
        run_from_toml(argv[0])
        return 0

    parser = argparse.ArgumentParser(prog="lmx")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("case", choices=["hartmann", "shercliff", "hunt"])
    run_parser.add_argument("--ha", type=float, default=20.0)
    run_parser.add_argument("--output", type=str, default="./out")
    run_parser.add_argument("--mode", choices=["steady", "transient"], default="steady")
    run_parser.add_argument("--plots", action="store_true")
    run_parser.add_argument("--quiet", action="store_true")
    run_parser.add_argument("--verbose", action="store_true")
    run_parser.add_argument("--verbosity", choices=["quiet", "normal", "detailed", "debug"], default=None)

    bench_parser = subparsers.add_parser("benchmark")
    bench_parser.add_argument("--repeats", type=int, default=3)
    bench_parser.add_argument("--ha", type=float, default=20.0)
    bench_parser.add_argument("--ny", type=int, default=48)
    bench_parser.add_argument("--nz", type=int, default=48)
    bench_parser.add_argument("--output", type=str, default="")

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("case", choices=["hartmann", "shercliff", "hunt"])
    validate_parser.add_argument("--ha", type=float, default=20.0)
    validate_parser.add_argument("--output", type=str, default="./out")
    validate_parser.add_argument("--reference-root", type=str, default="")
    validate_parser.add_argument("--x-slice", type=str, default="1m")
    validate_parser.add_argument("--hartmann-l2-threshold", type=float, default=0.05)
    validate_parser.add_argument("--hartmann-linf-threshold", type=float, default=0.1)

    args = parser.parse_args(argv)

    if args.command == "benchmark":
        payload = benchmark_solver(repeats=args.repeats, ha=args.ha, ny=args.ny, nz=args.nz)
        if args.output:
            write_benchmark_report(payload, args.output)
        print(json.dumps(payload, indent=2))
        return 0

    if args.command == "validate":
        case = _build_case(args)
        solution = _solve_case_with_optional_logger(case, solve_mode="steady", logger=None)
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        write_paraview(solution, out_dir)
        write_profile_csv(out_dir / f"{case.name}_centerline.csv", extract_centerline(solution))
        z_profile = extract_midplane_profile(solution, axis="z")
        write_profile_csv(out_dir / f"{case.name}_midplane_z.csv", z_profile)
        payload = validation_summary(solution, case.name, ha=args.ha)
        if args.case == "hartmann":
            comparison = hartmann_validation(solution, args.ha)
            write_analytic_comparison(comparison, out_dir / f"{case.name}_analytic.json", axis_name="y")
            acceptance = hartmann_acceptance(
                solution,
                args.ha,
                l2_threshold=args.hartmann_l2_threshold,
                linf_threshold=args.hartmann_linf_threshold,
            )
            write_acceptance_report(acceptance, out_dir / f"{case.name}_acceptance.json")
            payload["accepted"] = float(acceptance.passed)
            payload["acceptance_l2_threshold"] = acceptance.l2_threshold
            payload["acceptance_linf_threshold"] = acceptance.linf_threshold
        elif args.reference_root:
            comparison = closed_channel_validation(solution, args.case, int(args.ha), reference_root=args.reference_root)
            write_closed_channel_validation(comparison, out_dir / f"{case.name}_analytic.json")
            payload["y_l2_error"] = comparison.y_profile.l2_error
            payload["y_linf_error"] = comparison.y_profile.linf_error
            payload["z_l2_error"] = comparison.z_profile.l2_error
            payload["z_linf_error"] = comparison.z_profile.linf_error
            payload["combined_l2_error"] = combined_profile_error(
                comparison.y_profile.l2_error,
                comparison.z_profile.l2_error,
            )
            payload["combined_linf_error"] = combined_profile_error(
                comparison.y_profile.linf_error,
                comparison.z_profile.linf_error,
            )
            try:
                slice_report = processed_slice_validation(
                    solution,
                    args.case,
                    int(args.ha),
                    x_slice=args.x_slice,
                    reference_root=args.reference_root,
                )
            except FileNotFoundError:
                slice_report = None
            if slice_report is not None:
                write_processed_slice_validation(slice_report, out_dir / f"{case.name}_slice.json")
                payload["slice_y_l2_error"] = slice_report.y_profile.l2_error
                payload["slice_y_linf_error"] = slice_report.y_profile.linf_error
                payload["slice_z_l2_error"] = slice_report.z_profile.l2_error
                payload["slice_z_linf_error"] = slice_report.z_profile.linf_error
                payload["slice_combined_l2_error"] = combined_profile_error(
                    slice_report.y_profile.l2_error,
                    slice_report.z_profile.l2_error,
                )
                payload["slice_combined_linf_error"] = combined_profile_error(
                    slice_report.y_profile.linf_error,
                    slice_report.z_profile.linf_error,
                )
        write_metrics_json(payload, out_dir / f"{case.name}_metrics.json")
        print(json.dumps(payload, indent=2))
        return 0

    case = _build_case(args)
    case = case.__class__(
        **{
            **case.__dict__,
            "output": case.output.__class__(
                **{
                    **case.output.__dict__,
                    "directory": args.output,
                    "write_npz": True,
                    "write_json_summary": True,
                    "write_plots": args.plots,
                }
            ),
        }
    )
    config = RunConfig(case=case, solve_mode=args.mode)
    logging = config.logging
    if args.quiet:
        logging = LoggingSpec.from_user_controls(
            enabled=False,
            verbosity="quiet",
            banner=logging.banner,
            print_regions=logging.print_regions,
            print_boundaries=logging.print_boundaries,
            print_footer=logging.print_footer,
            flush=logging.flush,
            step_stride=logging.step_stride,
        )
    elif args.verbose or args.verbosity is not None:
        logging = LoggingSpec.from_user_controls(
            enabled=True,
            verbose=True if args.verbose else None,
            verbosity=args.verbosity or ("debug" if args.verbose else logging.verbosity),
            banner=logging.banner,
            print_regions=logging.print_regions,
            print_boundaries=logging.print_boundaries,
            print_footer=logging.print_footer,
            flush=logging.flush,
            step_stride=logging.step_stride,
        )
    if logging is not config.logging:
        config = RunConfig(case=case, solve_mode=args.mode, logging=logging)
    _run_config(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

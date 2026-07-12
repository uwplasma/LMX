#!/usr/bin/env python3
"""Compare native and released-SOLVAX PCG on the same five-point system."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import platform
import statistics
import time
from typing import Any

import jax
import jax.numpy as jnp
import jaxlib
import solvax

from lmx import linear
from lmx.cases import make_hartmann_case
from lmx.solvers import fully_developed_power_balance, solve_steady


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _lmx_version() -> str:
    """Read the package version from metadata or a source-only checkout."""

    try:
        return version("lmx")
    except PackageNotFoundError:
        for line in (ROOT / "pyproject.toml").read_text(encoding="utf-8").splitlines():
            if line.startswith("version = "):
                return line.split("=", 1)[1].strip().strip('"')
        raise RuntimeError("LMX version is missing from pyproject.toml")


def _memory_dict(compiled) -> dict[str, int | None]:
    analysis = compiled.memory_analysis()
    fields = (
        "alias_size_in_bytes",
        "argument_size_in_bytes",
        "generated_code_size_in_bytes",
        "output_size_in_bytes",
        "temp_size_in_bytes",
    )
    return {
        field: None if analysis is None else int(getattr(analysis, field, 0))
        for field in fields
    }


def _compile_and_measure(function, rhs, repeats: int):
    started = time.perf_counter()
    compiled = function.lower(rhs).compile()
    compile_seconds = time.perf_counter() - started
    samples = []
    result = None
    for _ in range(repeats):
        started = time.perf_counter()
        result = compiled(rhs)
        jax.block_until_ready(result)
        samples.append(time.perf_counter() - started)
    assert result is not None
    return result, {
        "compile_seconds": compile_seconds,
        "memory": _memory_dict(compiled),
        "warm_median_seconds": statistics.median(samples),
        "warm_samples_seconds": samples,
    }


def _normalized_backend() -> tuple[str, str]:
    """Return the acceptance backend and JAX's raw backend name."""

    raw = jax.default_backend().lower()
    normalized = "gpu" if raw in {"cuda", "gpu", "rocm"} else raw
    return normalized, raw


def _relative_difference(left: float, right: float) -> float:
    return abs(left - right) / max(abs(left), abs(right), 1.0e-30)


def _run_hartmann_comparison() -> dict[str, Any]:
    """Compare native and SOLVAX backends in a converged physical solve."""

    base = make_hartmann_case(ha=5.0, ny=16, nz=16)
    base = replace(
        base,
        time_stepper=replace(
            base.time_stepper,
            max_steps=120,
            t_final=0.12,
            steady_tolerance=1.0e-9,
            potential_iterations=400,
            potential_tolerance=1.0e-10,
            steady_potential_tolerance=1.0e-10,
            current_reconstruction="face_averaged",
            post_update_potential_refresh=True,
        ),
        solver=replace(
            base.solver,
            coupling_iterations=16,
            coupling_tolerance=1.0e-9,
        ),
    )
    solutions = {
        backend: solve_steady(
            replace(base, solver=replace(base.solver, linear_solver=backend))
        )
        for backend in ("cg", "solvax_pcg")
    }
    native = solutions["cg"]
    released = solutions["solvax_pcg"]
    jax.block_until_ready((native.state.u, released.state.u))
    powers = {
        backend: fully_developed_power_balance(base, solution)
        for backend, solution in solutions.items()
    }
    field_difference = float(
        jnp.linalg.norm(native.state.u - released.state.u)
        / jnp.maximum(jnp.linalg.norm(native.state.u), jnp.finfo(native.state.u.dtype).tiny)
    )
    flow = {
        backend: float(solution.diagnostics.volumetric_flow_rate_history[-1])
        for backend, solution in solutions.items()
    }
    balance_keys = (
        "electrical_power_relative_error",
        "network_electrical_relative_error",
        "lorentz_transfer_relative_error",
        "mechanical_power_relative_error",
    )
    power_terms = (
        "pressure_power",
        "lorentz_work",
        "viscous_dissipation",
        "joule_dissipation",
        "emf_power",
    )
    max_power_difference = max(
        _relative_difference(powers["cg"][key], powers["solvax_pcg"][key])
        for key in power_terms
    )
    acceptance = {
        "field_equivalent": field_difference <= 1.0e-10,
        "flow_equivalent": _relative_difference(flow["cg"], flow["solvax_pcg"])
        <= 1.0e-10,
        "native_balance_pass": max(abs(powers["cg"][key]) for key in balance_keys)
        <= 1.0e-3,
        "power_equivalent": max_power_difference <= 1.0e-10,
        "solvax_balance_pass": max(
            abs(powers["solvax_pcg"][key]) for key in balance_keys
        )
        <= 1.0e-3,
    }
    acceptance["pass"] = all(acceptance.values())
    return {
        "acceptance": acceptance,
        "comparison": {
            "field_relative_difference": field_difference,
            "flow_relative_difference": _relative_difference(
                flow["cg"], flow["solvax_pcg"]
            ),
            "max_power_relative_difference": max_power_difference,
        },
        "problem": {"case": "hartmann", "ha": 5.0, "grid": [16, 16]},
        "native": {
            "final_linear_residual": float(
                native.diagnostics.linear_residual_history[-1]
            ),
            "final_residual": float(native.state.residual),
            "flow_rate": flow["cg"],
            "power_balance": powers["cg"],
            "steps": len(native.diagnostics.residual_history),
        },
        "solvax": {
            "final_linear_residual": float(
                released.diagnostics.linear_residual_history[-1]
            ),
            "final_residual": float(released.state.residual),
            "flow_rate": flow["solvax_pcg"],
            "power_balance": powers["solvax_pcg"],
            "steps": len(released.diagnostics.residual_history),
        },
    }


def run_backend_comparison(
    *,
    grid: int = 64,
    repeats: int = 7,
    max_steps: int = 128,
    expected_backend: str | None = None,
) -> dict[str, Any]:
    if grid < 3:
        raise ValueError("grid must be at least three")
    if repeats < 1 or max_steps < 1:
        raise ValueError("repeats and max_steps must be positive")
    dtype = jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32
    diagonal = jnp.full((grid, grid), 6.0, dtype=dtype)
    west = jnp.ones((grid, grid), dtype=dtype)
    east = jnp.ones((grid, grid), dtype=dtype)
    south = jnp.ones((grid, grid), dtype=dtype)
    north = jnp.ones((grid, grid), dtype=dtype)
    y = jnp.linspace(0.0, 1.0, grid, dtype=dtype)
    z = jnp.linspace(0.0, 1.0, grid, dtype=dtype)
    yy, zz = jnp.meshgrid(y, z, indexing="ij")
    exact = 0.5 + jnp.sin(jnp.pi * yy) * jnp.sin(2.0 * jnp.pi * zz)
    rhs = linear.apply_five_point_operator(diagonal, west, east, south, north, exact)
    tolerance = float(max(1.0e-11, 100.0 * float(jnp.finfo(dtype).eps)))

    native = jax.jit(
        lambda value: linear.solve_five_point_cg_state.__wrapped__(
            diagonal,
            west,
            east,
            south,
            north,
            value,
            iterations=max_steps,
            tolerance=tolerance,
            preconditioner="jacobi",
        )
    )
    released = jax.jit(
        lambda value: linear.solve_five_point_solvax_pcg_state.__wrapped__(
            diagonal,
            west,
            east,
            south,
            north,
            value,
            iterations=max_steps,
            tolerance=tolerance,
            preconditioner="jacobi",
        )
    )
    native_result, native_metrics = _compile_and_measure(native, rhs, repeats)
    solvax_result, solvax_metrics = _compile_and_measure(released, rhs, repeats)
    native_field, native_residual, native_iterations = native_result
    solvax_field, solvax_residual, solvax_iterations = solvax_result

    def objective(scale):
        field, _, _ = linear.solve_five_point_solvax_pcg_state(
            diagonal,
            west,
            east,
            south,
            north,
            scale * rhs,
            iterations=max_steps,
            tolerance=tolerance,
            preconditioner="jacobi",
        )
        return jnp.sum(field**2)

    gradient = float(jax.grad(objective)(1.0))
    exact_gradient = float(2.0 * jnp.sum(exact**2))
    gradient_relative_error = abs(gradient - exact_gradient) / abs(exact_gradient)

    # Audit the transpose system independently of the custom VJP. For this
    # Hermitian five-point operator, A^T = A and the adjoint right-hand side of
    # sum(x**2) is 2*x. This records the residual and the resulting gradient,
    # while jax.grad above verifies the actual implicit-VJP path.
    transpose_rhs = 2.0 * solvax_field
    transpose_field, transpose_residual, transpose_iterations = released(
        transpose_rhs
    )
    jax.block_until_ready(transpose_field)
    transpose_gradient = float(jnp.vdot(transpose_field, rhs).real)
    transpose_gradient_relative_error = abs(
        transpose_gradient - exact_gradient
    ) / abs(exact_gradient)
    field_difference = float(
        jnp.linalg.norm(native_field - solvax_field)
        / jnp.maximum(jnp.linalg.norm(native_field), jnp.finfo(dtype).tiny)
    )
    native_error = float(jnp.linalg.norm(native_field - exact) / jnp.linalg.norm(exact))
    solvax_error = float(jnp.linalg.norm(solvax_field - exact) / jnp.linalg.norm(exact))
    native_temp = native_metrics["memory"]["temp_size_in_bytes"] or 0
    solvax_temp = solvax_metrics["memory"]["temp_size_in_bytes"] or 0
    warm_ratio = (
        solvax_metrics["warm_median_seconds"] / native_metrics["warm_median_seconds"]
    )
    memory_ratio = solvax_temp / native_temp if native_temp else 1.0
    accuracy_target = float(max(1.0e-10, 1000.0 * float(jnp.finfo(dtype).eps)))
    backend, raw_backend = _normalized_backend()
    requested_backend = backend if expected_backend is None else expected_backend.lower()
    if requested_backend not in {"cpu", "gpu"}:
        raise ValueError("expected_backend must be 'cpu', 'gpu', or None")
    acceptance = {
        "backend_matches_request": backend == requested_backend,
        "forward_equivalent": field_difference <= accuracy_target,
        "gradient_verified": gradient_relative_error <= accuracy_target,
        "memory_regression_within_10_percent": memory_ratio <= 1.10,
        "native_residual_pass": float(native_residual) <= tolerance,
        "solvax_residual_pass": float(solvax_residual) <= tolerance,
        "transpose_gradient_verified": transpose_gradient_relative_error
        <= accuracy_target,
        "transpose_residual_pass": float(transpose_residual) <= tolerance,
        "warm_time_regression_within_10_percent": warm_ratio <= 1.10,
    }
    hartmann = _run_hartmann_comparison()
    acceptance["end_to_end_hartmann_pass"] = hartmann["acceptance"]["pass"]
    acceptance["backend_promotion_pass"] = all(acceptance.values())
    acceptance[f"{requested_backend}_promotion_pass"] = acceptance[
        "backend_promotion_pass"
    ]
    return {
        "schema_version": 1,
        "benchmark": "LMX native versus released SOLVAX PCG",
        "acceptance": acceptance,
        "comparison": {
            "field_relative_difference": field_difference,
            "gradient": gradient,
            "gradient_exact": exact_gradient,
            "gradient_relative_error": gradient_relative_error,
            "memory_ratio": memory_ratio,
            "transpose_gradient": transpose_gradient,
            "transpose_gradient_relative_error": transpose_gradient_relative_error,
            "warm_time_ratio": warm_ratio,
        },
        "environment": {
            "backend": backend,
            "device": str(jax.devices()[0]),
            "dtype": jnp.dtype(dtype).name,
            "jax_version": jax.__version__,
            "jaxlib_version": jaxlib.__version__,
            "jax_raw_backend": raw_backend,
            "platform": platform.platform(),
            "python_version": platform.python_version(),
        },
        "implementation": {
            "benchmark_sha256": _sha256(Path(__file__)),
            "linear_sha256": _sha256(ROOT / "lmx" / "linear.py"),
            "lmx_version": _lmx_version(),
            "solvax_version": solvax.__version__,
        },
        "end_to_end_hartmann": hartmann,
        "problem": {
            "grid": [grid, grid],
            "max_steps": max_steps,
            "repeats": repeats,
            "tolerance": tolerance,
        },
        "native": {
            **native_metrics,
            "iterations": int(native_iterations),
            "relative_error": native_error,
            "residual": float(native_residual),
        },
        "solvax": {
            **solvax_metrics,
            "iterations": int(solvax_iterations),
            "relative_error": solvax_error,
            "residual": float(solvax_residual),
        },
        "transpose_audit": {
            "gradient": transpose_gradient,
            "gradient_exact": exact_gradient,
            "gradient_relative_error": transpose_gradient_relative_error,
            "iterations": int(transpose_iterations),
            "residual": float(transpose_residual),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--expected-backend", choices=("cpu", "gpu"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/solvax-pcg-equivalence-cpu.json"),
    )
    args = parser.parse_args()
    result = run_backend_comparison(
        grid=args.grid,
        repeats=args.repeats,
        max_steps=args.max_steps,
        expected_backend=args.expected_backend,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"{args.output} backend={result['environment']['backend']} "
        f"backend_promotion_pass={result['acceptance']['backend_promotion_pass']}"
    )
    return 0 if result["acceptance"]["backend_promotion_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from lmx.autodiff import build_hartmann_autodiff_problem, hartmann_mean_velocity, solve_differentiable_hartmann
from lmx.plotting import write_autodiff_plots


def run_sensitivity_scan(
    *,
    forcing: float,
    hartmann_values: jnp.ndarray,
) -> list[dict[str, float]]:
    problem = build_hartmann_autodiff_problem(ny=48, nz=48, macro_iterations=8, potential_iterations=80, velocity_iterations=120)
    mean_velocity_fn = lambda ha: hartmann_mean_velocity(problem, forcing=forcing, hartmann_number=ha)
    mean_velocity = jax.vmap(mean_velocity_fn)(hartmann_values)
    sensitivity = jax.vmap(jax.grad(mean_velocity_fn))(hartmann_values)
    return [
        {
            "hartmann_number": float(ha),
            "mean_velocity": float(mean),
            "d_mean_velocity_d_ha": float(dmean),
        }
        for ha, mean, dmean in zip(np.asarray(hartmann_values), np.asarray(mean_velocity), np.asarray(sensitivity), strict=True)
    ]


def run_inverse_design(
    *,
    reference_forcing: float,
    target_hartmann_number: float,
    initial_guess: float,
    learning_rate: float,
    steps: int,
) -> tuple[list[dict[str, float]], dict[str, object]]:
    problem = build_hartmann_autodiff_problem(ny=48, nz=48, macro_iterations=8, potential_iterations=80, velocity_iterations=120)
    target_u, _ = solve_differentiable_hartmann(
        problem,
        forcing=reference_forcing,
        hartmann_number=target_hartmann_number,
    )
    target_profile = target_u[:, target_u.shape[1] // 2]

    def objective(forcing_value):
        recovered_u, _ = solve_differentiable_hartmann(
            problem,
            forcing=forcing_value,
            hartmann_number=target_hartmann_number,
        )
        recovered_profile = recovered_u[:, recovered_u.shape[1] // 2]
        return jnp.mean((recovered_profile - target_profile) ** 2)

    history: list[dict[str, float]] = []
    parameter = jnp.asarray(initial_guess, dtype=jnp.float32)
    for step in range(steps):
        loss, gradient = jax.value_and_grad(objective)(parameter)
        history.append(
            {
                "iteration": float(step),
                "forcing": float(parameter),
                "loss": float(loss),
                "gradient": float(gradient),
            }
        )
        parameter = jnp.clip(parameter - learning_rate * gradient, 0.05, 5.0)

    recovered_u, recovered_phi = solve_differentiable_hartmann(
        problem,
        forcing=parameter,
        hartmann_number=target_hartmann_number,
    )
    return history, {
        "target_hartmann_number": float(target_hartmann_number),
        "target_forcing": float(reference_forcing),
        "recovered_forcing": float(parameter),
        "target_profile": np.asarray(target_profile).tolist(),
        "recovered_profile": np.asarray(recovered_u[:, recovered_u.shape[1] // 2]).tolist(),
        "recovered_phi_max": float(jnp.max(jnp.abs(recovered_phi))),
    }


def run_autodiff_design_demo(
    *,
    out_dir: Path,
    forcing: float = 1.0,
    target_hartmann_number: float = 14.0,
    initial_guess: float = 0.2,
    learning_rate: float = 2000.0,
    steps: int = 24,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    hartmann_values = jnp.linspace(2.0, 30.0, 24)
    sensitivity_scan = run_sensitivity_scan(forcing=forcing, hartmann_values=hartmann_values)
    optimization_history, recovered = run_inverse_design(
        reference_forcing=forcing,
        target_hartmann_number=target_hartmann_number,
        initial_guess=initial_guess,
        learning_rate=learning_rate,
        steps=steps,
    )
    plots = write_autodiff_plots(
        sensitivity_scan,
        optimization_history,
        out_dir,
        case_title="LMX autodiff sensitivity and inverse design",
        target_parameter=forcing,
        parameter_key="forcing",
        parameter_label="Recovered forcing",
        target_label="Target forcing",
    )
    summary = {
        "forcing": forcing,
        "target_hartmann_number": target_hartmann_number,
        "initial_guess": initial_guess,
        "learning_rate": learning_rate,
        "steps": steps,
        "sensitivity_scan": sensitivity_scan,
        "optimization_history": optimization_history,
        "recovered": recovered,
        "plots": [path.name for path in plots],
    }
    (out_dir / "autodiff_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the LMX autodiff sensitivity and inverse-design demo.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/examples/autodiff_design"))
    parser.add_argument("--forcing", type=float, default=1.0)
    parser.add_argument("--target-ha", type=float, default=14.0)
    parser.add_argument("--initial-guess", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=2000.0)
    parser.add_argument("--steps", type=int, default=24)
    args = parser.parse_args(argv)

    run_autodiff_design_demo(
        out_dir=args.output,
        forcing=args.forcing,
        target_hartmann_number=args.target_ha,
        initial_guess=args.initial_guess,
        learning_rate=args.learning_rate,
        steps=args.steps,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

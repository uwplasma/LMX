"""Differentiate a Hartmann flow and recover its forcing by inverse design.

Edit the inputs below, then run ``python examples/autodiff_design_demo.py``.
The script writes a compact design trace and plot beneath ``artifacts/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from lmx.cases import (
    build_hartmann_autodiff_problem,
    hartmann_mean_velocity,
    solve_differentiable_hartmann,
)

# Inputs: mesh, solver effort, parameter scan, design controls, and outputs.
OUTPUT_DIR = Path("artifacts/examples/autodiff_design")
NY = NZ = 24  # Increase both mesh counts to resolve thinner Hartmann layers.
WIDTH = HEIGHT = 2.0
CONDUCTIVITY = 1.0
DENSITY = 1.0
VISCOSITY = 0.01
MACRO_ITERATIONS = 6
POTENTIAL_ITERATIONS = 60
VELOCITY_ITERATIONS = 80
RELAXATION = 0.9
FORCING = 1.0
HARTMANN_MIN, HARTMANN_MAX = 2.0, 30.0
SCAN_POINTS = 16
TARGET_HARTMANN_NUMBER = 14.0
INITIAL_FORCING = 0.2
LEARNING_RATE = 20_000.0
DESIGN_STEPS = 16
WRITE_PDF = True  # Disable to retain only the smaller PNG and JSON outputs.


# Construct the public differentiable problem and the response to scan.
problem = build_hartmann_autodiff_problem(
    ny=NY,
    nz=NZ,
    width=WIDTH,
    height=HEIGHT,
    conductivity=CONDUCTIVITY,
    density=DENSITY,
    viscosity=VISCOSITY,
    macro_iterations=MACRO_ITERATIONS,
    potential_iterations=POTENTIAL_ITERATIONS,
    velocity_iterations=VELOCITY_ITERATIONS,
    relaxation=RELAXATION,
)


def mean_velocity_objective(hartmann_number: jax.Array) -> jax.Array:
    """Return the mean velocity whose Hartmann sensitivity is requested."""

    return hartmann_mean_velocity(problem, forcing=FORCING, hartmann_number=hartmann_number)


hartmann_numbers = jnp.linspace(HARTMANN_MIN, HARTMANN_MAX, SCAN_POINTS)
mean_velocity, sensitivity = jax.jit(jax.vmap(jax.value_and_grad(mean_velocity_objective)))(hartmann_numbers)
scan_arrays = map(np.asarray, (hartmann_numbers, mean_velocity, sensitivity))
sensitivity_scan = [
    {
        "hartmann_number": float(ha),
        "mean_velocity": float(mean),
        "d_mean_velocity_d_ha": float(gradient),
    }
    for ha, mean, gradient in zip(*scan_arrays, strict=True)
]

# Build target data, then state the least-squares design objective explicitly.
target_u, _ = solve_differentiable_hartmann(
    problem,
    forcing=FORCING,
    hartmann_number=TARGET_HARTMANN_NUMBER,
)
target_profile = target_u[:, target_u.shape[1] // 2]


def profile_objective(forcing: jax.Array) -> jax.Array:
    """Return the centerline-profile misfit for a candidate forcing."""

    candidate_u, _ = solve_differentiable_hartmann(
        problem,
        forcing=forcing,
        hartmann_number=TARGET_HARTMANN_NUMBER,
    )
    candidate_profile = candidate_u[:, candidate_u.shape[1] // 2]
    return jnp.mean((candidate_profile - target_profile) ** 2)


# Run bounded gradient descent; replace this loop with any JAX optimizer if desired.
value_and_gradient = jax.jit(jax.value_and_grad(profile_objective))
optimization_history: list[dict[str, float]] = []
recovered_forcing = jnp.asarray(INITIAL_FORCING, dtype=jnp.float32)
for iteration in range(DESIGN_STEPS):
    loss, gradient = value_and_gradient(recovered_forcing)
    optimization_history.append(
        {
            "iteration": float(iteration),
            "forcing": float(recovered_forcing),
            "loss": float(loss),
            "gradient": float(gradient),
        }
    )
    recovered_forcing = jnp.clip(recovered_forcing - LEARNING_RATE * gradient, 0.05, 5.0)
recovered_loss = float(profile_objective(recovered_forcing))

# Plot the sensitivity and design trace without hiding example logic in the package.
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), constrained_layout=True)
axes[0].plot(hartmann_numbers, mean_velocity, label=r"$\bar{u}$")
axes[0].plot(hartmann_numbers, sensitivity, "--", label=r"$d\bar{u}/dHa$")
axes[0].set(xlabel="Hartmann number", ylabel="Response", title="Sensitivity scan")
axes[0].legend()
iterations = [row["iteration"] for row in optimization_history]
losses = [row["loss"] for row in optimization_history]
forcing_trace = [row["forcing"] for row in optimization_history]
axes[1].semilogy(iterations, losses, label="Profile misfit")
forcing_axis = axes[1].twinx()
forcing_axis.plot(iterations, forcing_trace, "--", color="tab:purple", label="Forcing")
forcing_axis.axhline(FORCING, color="black", linestyle=":", label="Target")
axes[1].set(xlabel="Gradient step", ylabel="Profile misfit", title="Inverse design")
forcing_axis.set_ylabel("Forcing")
handles = axes[1].lines + forcing_axis.lines
axes[1].legend(handles, [line.get_label() for line in handles])
plot_paths = [OUTPUT_DIR / "autodiff_summary.png"]
figure.savefig(plot_paths[0], dpi=180)
if WRITE_PDF:
    plot_paths.append(OUTPUT_DIR / "autodiff_summary.pdf")
    figure.savefig(plot_paths[-1])
plt.close(figure)

# Save machine-readable evidence alongside the plot.
summary = {
    "forcing": FORCING,
    "target_hartmann_number": TARGET_HARTMANN_NUMBER,
    "sensitivity_scan": sensitivity_scan,
    "optimization_history": optimization_history,
    "recovered": {"forcing": float(recovered_forcing), "loss": recovered_loss},
    "plots": [path.name for path in plot_paths],
}
summary_path = OUTPUT_DIR / "autodiff_summary.json"
summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))

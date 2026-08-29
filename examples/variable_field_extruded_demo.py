"""Optimize layered-duct field, wall, and geometry controls with exact LMX gradients.

Run ``python examples/variable_field_extruded_demo.py``. The deliberately
small mesh is a portable workflow demonstration, not production validation.
Artifacts are written beneath the ignored ``artifacts/`` directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from lmx.fringing import (
    build_layered_duct_extruded_problem,
    evolve_extruded_fields,
    extruded_engineering_objectives,
)

jax.config.update("jax_enable_x64", True)

# Inputs: bounded field, wall, fixed-topology geometry, solver, and optimizer controls.
OUTPUT_DIR = Path("artifacts/examples/variable_field_extruded")
NX_STATIONS, NY, NZ, WALL_CELLS = 7, 6, 6, 1
FIELD_HALF_RANGE = 0.10
WALL_HALF_RANGE = 0.50
GEOMETRY_HALF_RANGE = jnp.asarray([0.10, 0.05, 0.05])
EVOLUTION_STEPS = 8
OPTIMIZATION_STEPS = 40
LEARNING_RATE = 0.05
FINITE_DIFFERENCE_STEP = 2.0e-3

problem = build_layered_duct_extruded_problem(
    ha_peak=6.0,
    nx_stations=NX_STATIONS,
    ny=NY,
    nz=NZ,
    wall_cells=WALL_CELLS,
    length=3.0,
    entry_center=0.75,
    exit_center=2.25,
    transition_width=0.25,
)


def decode_controls(parameters: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Map unconstrained values to bounded field, wall, and geometry controls."""
    field_shape = jnp.tanh(parameters[:NX_STATIONS])
    field_scale = 1.0 + FIELD_HALF_RANGE * (field_shape - jnp.mean(field_shape))
    wall_scale = 1.0 + WALL_HALF_RANGE * jnp.tanh(parameters[NX_STATIONS])
    geometry_scale = 1.0 + GEOMETRY_HALF_RANGE * jnp.tanh(parameters[-3:])
    return field_scale, jnp.asarray([1.0, wall_scale]), geometry_scale


def metrics_for(parameters: jax.Array) -> dict[str, jax.Array]:
    """Return traced engineering metrics for one continuous design vector."""
    field_scale, material_scale, geometry_scale = decode_controls(parameters)
    fields = evolve_extruded_fields(
        problem,
        magnetic_field_scale=field_scale,
        material_conductivity_scale=material_scale,
        geometry_scale=geometry_scale,
        steps=EVOLUTION_STEPS,
    )
    return extruded_engineering_objectives(problem, fields, geometry_scale=geometry_scale, smoothing=1.0e-8)


initial_parameters = jnp.zeros(NX_STATIONS + 4)
baseline = metrics_for(initial_parameters)
power_scale = jnp.maximum(jnp.abs(baseline["pumping_power"]), 1.0e-12)
nonuniformity_scale = jnp.maximum(baseline["flow_nonuniformity"], 1.0e-12)
current_scale = jnp.maximum(baseline["wall_current_density_rms"], 1.0e-12)
flow_scale = jnp.maximum(jnp.abs(baseline["flow_rate"]), 1.0e-12)


def loss(parameters: jax.Array) -> jax.Array:
    """Balance pumping, outlet uniformity, wall current, and preserved flow."""
    metrics = metrics_for(parameters)
    smooth_power = jnp.sqrt(metrics["pumping_power"] ** 2 + (1.0e-6 * power_scale) ** 2)
    flow_error = (metrics["flow_rate"] - baseline["flow_rate"]) / flow_scale
    return (
        0.35 * smooth_power / power_scale
        + 0.30 * metrics["flow_nonuniformity"] / nonuniformity_scale
        + 0.25 * metrics["wall_current_density_rms"] / current_scale
        + 100.0 * flow_error**2
    )


def scalar_metrics(values: dict[str, jax.Array]) -> dict[str, float]:
    """Convert a traced metric mapping to JSON-ready scalars."""
    return {name: float(value) for name, value in values.items()}


# Run one compiled value/gradient evaluation per bounded Adam update.
value_and_gradient = jax.jit(jax.value_and_grad(loss))
compiled_loss = jax.jit(loss)
parameters = initial_parameters
first_moment = jnp.zeros_like(parameters)
second_moment = jnp.zeros_like(parameters)
loss_history = [float(compiled_loss(parameters))]
for step in range(1, OPTIMIZATION_STEPS + 1):
    _, gradient = value_and_gradient(parameters)
    first_moment = 0.9 * first_moment + 0.1 * gradient
    second_moment = 0.999 * second_moment + 0.001 * gradient**2
    corrected_moment = first_moment / (1.0 - 0.9**step)
    corrected_variance = second_moment / (1.0 - 0.999**step)
    parameters -= LEARNING_RATE * corrected_moment / (jnp.sqrt(corrected_variance) + 1.0e-8)
    loss_history.append(float(compiled_loss(parameters)))

final_loss, final_gradient = value_and_gradient(parameters)
_, checked_gradient = value_and_gradient(initial_parameters)
directions = jnp.eye(parameters.size)
finite_difference = jnp.asarray(
    [
        (
            compiled_loss(initial_parameters + FINITE_DIFFERENCE_STEP * direction)
            - compiled_loss(initial_parameters - FINITE_DIFFERENCE_STEP * direction)
        )
        / (2.0 * FINITE_DIFFERENCE_STEP)
        for direction in directions
    ]
)
gradient_error = jnp.linalg.norm(checked_gradient - finite_difference) / jnp.maximum(
    jnp.linalg.norm(finite_difference), 1.0e-14
)
field_scale, material_scale, geometry_scale = decode_controls(parameters)
optimized = metrics_for(parameters)
baseline_json, optimized_json = scalar_metrics(baseline), scalar_metrics(optimized)
improvements = {
    "pumping_power_magnitude": 1.0 - abs(optimized_json["pumping_power"] / baseline_json["pumping_power"]),
    "wall_current_density_rms": 1.0
    - optimized_json["wall_current_density_rms"] / baseline_json["wall_current_density_rms"],
    "flow_rate_relative_change": optimized_json["flow_rate"] / baseline_json["flow_rate"] - 1.0,
}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
figure, axes = plt.subplots(1, 3, figsize=(12.0, 3.2), constrained_layout=True)
x = np.asarray(problem.profile.x)
axes[0].plot(x, np.asarray(problem.profile.field_scale), "--", label="baseline")
axes[0].plot(
    x * float(geometry_scale[0]),
    np.asarray(problem.profile.field_scale * field_scale),
    label="optimized",
)
axes[0].set(xlabel="axial coordinate", ylabel=r"$B/B_0$", title="Field and axial geometry")
axes[0].legend(frameon=False)
axes[1].semilogy(range(len(loss_history)), loss_history)
axes[1].set(xlabel="design step", ylabel="normalized loss", title="Gradient optimization")
metric_names = ("pumping_power", "flow_nonuniformity", "wall_current_density_rms")
metric_ratios = [abs(optimized_json[name] / baseline_json[name]) for name in metric_names]
axes[2].bar(("pumping", "nonuniformity", "wall current"), metric_ratios)
axes[2].axhline(1.0, color="black", linewidth=0.8)
axes[2].tick_params(axis="x", rotation=20)
axes[2].set(ylabel="optimized / baseline", title="Engineering response")
plot_path = OUTPUT_DIR / "blanket_design_optimization.png"
figure.savefig(plot_path, dpi=160)
plt.close(figure)

summary = {
    "status": "portable differentiable design example; not external validation",
    "shape": [NX_STATIONS, NY, NZ],
    "controls": {
        "field_scale": np.asarray(field_scale).tolist(),
        "field_mean": float(jnp.mean(field_scale)),
        "wall_conductivity_scale": float(material_scale[1]),
        "geometry_scale": np.asarray(geometry_scale).tolist(),
    },
    "optimization": {
        "steps": OPTIMIZATION_STEPS,
        "initial_loss": loss_history[0],
        "final_loss": float(final_loss),
        "loss_history": loss_history,
    },
    "gradient_check": {
        "location": "baseline",
        "step": FINITE_DIFFERENCE_STEP,
        "relative_l2_error": float(gradient_error),
        "final_gradient_l2_norm": float(jnp.linalg.norm(final_gradient)),
    },
    "metrics": {"baseline": baseline_json, "optimized": optimized_json},
    "improvements": improvements,
    "plot": plot_path.name,
}
summary_path = OUTPUT_DIR / "variable_field_extruded_summary.json"
summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))

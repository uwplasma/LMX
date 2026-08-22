"""Compare explicit intact-AlN and bare-metal liquid-lithium duct walls.

Edit the inputs below, then run ``python examples/li_aln_wall_stack_example.py``.
The example exposes the material stack, mesh, solver, diagnostics, and plot
instead of hiding them behind a study-specific package wrapper. Results assess
electrical MHD performance only, not Li/AlN material compatibility.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lmx import (
    WallLayer,
    dynamic_to_kinematic_viscosity,
    effective_pinhole_conductance_ratio,
    generate_multilayer_duct_mesh,
    hartmann_number,
    interaction_parameter,
    magnetic_reynolds_number,
    normal_stack_leakage_ratio,
    reynolds_number,
    tangential_stack_conductance_ratio,
)
from lmx.cases import solve_steady
from lmx.specs import (
    BoundaryCondition,
    CaseSpec,
    GeometrySpec,
    MagneticFieldSpec,
    OutputSpec,
    RegionSpec,
    SolverConfig,
    TimeStepperConfig,
)
from lmx.validation import validation_summary

# Inputs: edit material, geometry, wall, numerics, and output choices here.
OUTPUT_DIR = Path("artifacts/examples/li_aln_wall_stack")
WALL_MODELS = ("intact_aln", "bare_metal")
LENGTH_SCALE_M = 0.05
MEAN_VELOCITY_M_S = 1.0e-2
MAGNETIC_FIELD_T = 5.0e-2
LITHIUM_DENSITY_KG_M3 = 500.0
LITHIUM_DYNAMIC_VISCOSITY_PA_S = 4.0e-4
LITHIUM_CONDUCTIVITY_S_M = 3.2e6
ALN_CONDUCTIVITY_S_M = 1.0e-8
ALN_THICKNESS_M = 2.0e-4
ALN_CELLS = 1
METAL_NAME = "316L"
METAL_CONDUCTIVITY_S_M = 1.35e6
METAL_THICKNESS_M = 1.0e-3
METAL_CELLS = 2
PINHOLE_FRACTIONS = (0.0, 1.0e-6, 1.0e-4, 1.0e-2, 1.0)
FLUID_CELLS_Y = 4
FLUID_CELLS_Z = 4
TIME_STEP_S = 1.0e-3
FINAL_TIME_S = 4.0e-3
MAX_STEPS = 4
POTENTIAL_ITERATIONS = 32


# Set up dimensional and reduced electrical properties.
width = height = 2.0 * LENGTH_SCALE_M
kinematic_viscosity = dynamic_to_kinematic_viscosity(LITHIUM_DYNAMIC_VISCOSITY_PA_S, LITHIUM_DENSITY_KG_M3)
model_layers = {
    "intact_aln": (
        WallLayer("aln", ALN_CONDUCTIVITY_S_M, ALN_THICKNESS_M, ALN_CELLS),
        WallLayer(METAL_NAME, METAL_CONDUCTIVITY_S_M, METAL_THICKNESS_M, METAL_CELLS),
    ),
    "bare_metal": (
        WallLayer(
            METAL_NAME,
            METAL_CONDUCTIVITY_S_M,
            ALN_THICKNESS_M + METAL_THICKNESS_M,
            ALN_CELLS + METAL_CELLS,
        ),
    ),
}
nondimensional = {
    "hartmann_number": hartmann_number(
        magnetic_field=MAGNETIC_FIELD_T,
        length_scale=LENGTH_SCALE_M,
        conductivity=LITHIUM_CONDUCTIVITY_S_M,
        density=LITHIUM_DENSITY_KG_M3,
        kinematic_viscosity=kinematic_viscosity,
    ),
    "reynolds_number": reynolds_number(
        velocity=MEAN_VELOCITY_M_S,
        length_scale=LENGTH_SCALE_M,
        kinematic_viscosity=kinematic_viscosity,
    ),
    "interaction_parameter": interaction_parameter(
        magnetic_field=MAGNETIC_FIELD_T,
        length_scale=LENGTH_SCALE_M,
        conductivity=LITHIUM_CONDUCTIVITY_S_M,
        density=LITHIUM_DENSITY_KG_M3,
        velocity=MEAN_VELOCITY_M_S,
    ),
    "magnetic_reynolds_number": magnetic_reynolds_number(
        velocity=MEAN_VELOCITY_M_S,
        length_scale=LENGTH_SCALE_M,
        conductivity=LITHIUM_CONDUCTIVITY_S_M,
    ),
}
intact_layers = model_layers["intact_aln"]
metal_layers = model_layers["bare_metal"]
intact_conductance = tangential_stack_conductance_ratio(
    intact_layers,
    fluid_conductivity=LITHIUM_CONDUCTIVITY_S_M,
    length_scale=LENGTH_SCALE_M,
)
metal_conductance = tangential_stack_conductance_ratio(
    metal_layers,
    fluid_conductivity=LITHIUM_CONDUCTIVITY_S_M,
    length_scale=LENGTH_SCALE_M,
)
pinhole_conductance = [
    effective_pinhole_conductance_ratio(
        intact_conductance_ratio=intact_conductance,
        metal_conductance_ratio=metal_conductance,
        pinhole_fraction=fraction,
    )
    for fraction in PINHOLE_FRACTIONS
]

# Run the same prescribed-flow solve for each explicit wall stack.
results: dict[str, dict[str, object]] = {}
meshes = {}
for model in WALL_MODELS:
    layers = model_layers[model]
    stacks = {side: layers for side in ("left", "right", "bottom", "top")}
    mesh = generate_multilayer_duct_mesh(
        width=width,
        height=height,
        length=LENGTH_SCALE_M,
        nx=1,
        ny=FLUID_CELLS_Y,
        nz=FLUID_CELLS_Z,
        wall_layers=stacks,
        fluid_conductivity=LITHIUM_CONDUCTIVITY_S_M,
    )
    case = CaseSpec(
        name=f"li_aln_{model}",
        geometry=GeometrySpec(
            kind="rect_duct",
            width=width,
            height=height,
            length=LENGTH_SCALE_M,
            ny=FLUID_CELLS_Y,
            nz=FLUID_CELLS_Z,
        ),
        regions=(
            RegionSpec(
                "fluid",
                "fluid",
                LITHIUM_CONDUCTIVITY_S_M,
                LITHIUM_DENSITY_KG_M3,
                kinematic_viscosity,
            ),
        ),
        magnetic_field=MagneticFieldSpec(kind="constant", value=(0.0, MAGNETIC_FIELD_T, 0.0)),
        boundary_conditions=(
            BoundaryCondition("walls", "no_slip"),
            BoundaryCondition(
                "flow_rate",
                "inlet_flow_rate",
                value=MEAN_VELOCITY_M_S * width * height,
                axis="x",
            ),
        ),
        time_stepper=TimeStepperConfig(
            dt=TIME_STEP_S,
            t_final=FINAL_TIME_S,
            max_steps=MAX_STEPS,
            potential_iterations=POTENTIAL_ITERATIONS,
            potential_tolerance=1.0e-7,
            relaxation=0.35,
            velocity_update_limit=2.0e-2,
        ),
        solver=SolverConfig(coupling_iterations=4, coupling_tolerance=1.0e-7),
        output=OutputSpec(
            write_paraview=False,
            write_csv_profiles=False,
            write_npz=False,
            write_json_summary=False,
        ),
        initial_velocity=MEAN_VELOCITY_M_S,
        reference_pressure_gradient=-1.0,
        reference_phi_cell=(mesh.ny // 2, mesh.nz // 2),
    )
    solution = solve_steady(case, mesh=mesh)
    diagnostics = validation_summary(solution, case.name)
    face_current = float(solution.diagnostics.face_current_max_history[-1])
    spacing = min(float(np.min(mesh.dy)), float(np.min(mesh.dz)))
    diagnostics["charge_balance_relative"] = (
        abs(float(diagnostics["charge_balance_residual"])) * spacing / max(abs(face_current), 1.0e-30)
    )
    diagnostics["interface_current_relative"] = abs(float(diagnostics["interface_current_residual"])) / max(
        abs(face_current), 1.0e-30
    )
    results[model] = {
        "layers": [layer.__dict__ for layer in layers],
        "tangential_conductance_ratio": tangential_stack_conductance_ratio(
            layers,
            fluid_conductivity=LITHIUM_CONDUCTIVITY_S_M,
            length_scale=LENGTH_SCALE_M,
        ),
        "normal_leakage_ratio": normal_stack_leakage_ratio(
            layers,
            fluid_conductivity=LITHIUM_CONDUCTIVITY_S_M,
            length_scale=LENGTH_SCALE_M,
        ),
        "mesh_shape": list(mesh.yz_shape),
        "validation": diagnostics,
    }
    meshes[model] = mesh

# Save a compact, reproducible summary and a three-panel visual comparison.
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
summary = {
    "scope": "MHD electrical performance only",
    "nondimensional": nondimensional,
    "inductionless_assumption_pass": nondimensional["magnetic_reynolds_number"] < 1.0e-2,
    "pinhole_sweep": [
        {"fraction": fraction, "effective_conductance_ratio": conductance}
        for fraction, conductance in zip(PINHOLE_FRACTIONS, pinhole_conductance)
    ],
    "models": results,
}
summary_path = OUTPUT_DIR / "li_aln_wall_stack_summary.json"
summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

figure, axes = plt.subplots(1, 3, figsize=(11.0, 3.4), constrained_layout=True)
intact_mesh = meshes["intact_aln"]
conductivity = np.asarray(intact_mesh.sigma, dtype=float)
image = axes[0].imshow(np.log10(np.maximum(conductivity, 1.0e-30)).T, origin="lower")
axes[0].set(title="Explicit Li | AlN | 316L mesh", xlabel="y cell", ylabel="z cell")
figure.colorbar(image, ax=axes[0], label=r"$\log_{10}(\sigma\,[S/m])$")
axes[1].loglog(PINHOLE_FRACTIONS[1:], pinhole_conductance[1:], "o-", color="#0f766e")
axes[1].set(title="Pinhole current path", xlabel="pinhole fraction", ylabel="effective c")
currents = [float(results[name]["validation"]["mean_current_magnitude"]) for name in WALL_MODELS]
x = np.arange(len(WALL_MODELS))
axes[2].bar(x, currents, color=("#2563eb", "#f59e0b"))
axes[2].set(title="Solved current response", ylabel="mean |J|", xticks=x, xticklabels=WALL_MODELS)
axes[2].tick_params(axis="x", rotation=15)
figure_path = OUTPUT_DIR / "li_aln_wall_stack.png"
figure.savefig(figure_path, dpi=150)
plt.close(figure)
print(json.dumps({"summary": str(summary_path), "figure": str(figure_path)}, indent=2))

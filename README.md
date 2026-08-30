# LMX

[![Python](https://img.shields.io/badge/python-3.10--3.13-3776ab.svg)](https://www.python.org/)
[![CI](https://img.shields.io/github/actions/workflow/status/uwplasma/LMX/ci.yml?branch=main&label=ci)](https://github.com/uwplasma/LMX/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/readthedocs/lmx/latest?label=docs)](https://lmx.readthedocs.io/)
[![License](https://img.shields.io/github/license/uwplasma/LMX)](LICENSE)

LMX is a compact JAX code for inductionless liquid-metal magnetohydrodynamics
in ducts. Use it for analytical-reference Hartmann, Shercliff, and Hunt flow;
three-dimensional straight ducts and pipes in spatially varying magnetic
fields; periodic Q2D vortex dynamics; and differentiable field, wall, and
fixed-topology geometry studies. LMX owns the MHD equations, boundary
conditions, coupling, diagnostics, and validation. [SOLVAX](https://github.com/uwplasma/SOLVAX)
owns reusable linear, fixed-point, preconditioning, and implicit-derivative
algorithms.

![Full-profile analytical validation for Hartmann, Shercliff, and Hunt ducts](docs/_static/analytic_velocity_profiles.webp)

## Choose where to start

| Your problem | Start with | Why |
|---|---|---|
| Learn the API or check an installation | `lmx examples/hartmann_case.toml` | Small stable solve with an analytical profile |
| Fully developed rectangular or layered duct | `python examples/hartmann_example.py` or `hunt_example.py` | Velocity, potential, current, Lorentz force, conservation, and convergence |
| Three-dimensional field entry/exit or magnetic obstacle | `python examples/fringing_benchmark_demo.py` | Editable axial field, 3-D duct fields, face-flux projection, and station diagnostics |
| Conducting or insulating blanket wall stack | `python examples/li_aln_wall_stack_example.py` | Explicit material layers and interface-current checks |
| Gradient-based field, wall, or geometry design | `python examples/variable_field_extruded_demo.py` | Production 3-D fields, checked gradients, and bounded optimization |
| Depth-averaged strong-field dynamics | `python examples/q2d_turbulence_demo.py` | Q2D vorticity, energy/enstrophy histories, poster, and optional movie |

The CLI, Hartmann, and Hunt workflows are stable portable entry points. The
wall, 3-D, design, and Q2D examples are research-stage workflows: they exercise
real supported equations, but their small default meshes are demonstrations,
not publication validation.

## Install

LMX is currently installed from source; there is no LMX release on PyPI yet.

```console
git clone https://github.com/uwplasma/LMX.git
cd LMX
python -m pip install -e ".[visualization]"
lmx examples/hartmann_case.toml
```

The last command is a complete smoke test. It should finish with a terminal
status and conservation diagnostics. JAX selects the CPU by default. Install
the accelerator-specific JAX wheel by following the
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html).

## Build your first study

The script below constructs an insulating Hartmann duct, solves it, rejects an
unfinished solve, compares the full centerline with the analytical solution,
and writes fields and profiles. Save it as `first_study.py` and run
`python first_study.py`.

```python
from pathlib import Path

import lmx
from lmx.io import write_solution_outputs
from lmx.validation import hartmann_validation

ha = 10.0
output_dir = Path("artifacts/first-study")

# 1. Define the physics, geometry, materials, mesh, and output location.
case = lmx.make_hartmann_case(
    ha=ha,
    width=2.0,
    height=2.0,
    ny=16,
    nz=16,
    conductivity=1.0,
    density=1.0,
    viscosity=1.0,
    output_dir=str(output_dir),
)

# 2. Solve through the common API and fail closed on an incomplete run.
solution = lmx.solve(case)
if not solution.converged:
    raise RuntimeError(
        f"solve ended with {solution.status!r} after {solution.steps} steps"
    )

# 3. Validate the observable that matters for this reference problem.
comparison = hartmann_validation(solution, ha)
charge = float(solution.diagnostics.charge_balance_residual_history[-1])

# 4. Save ParaView fields, CSV profiles, and a restart-capable NPZ file.
generated = write_solution_outputs(solution, case, output_dir)

print(f"status={solution.status}, residual={solution.residual:.3e}")
print(f"profile L2 error={comparison.l2_error:.3e}, charge={charge:.3e}")
print({kind: [str(path) for path in paths] for kind, paths in generated.items()})
```

To turn this into a research case, change one layer at a time and keep the
validation step beside the solve:

| Change | Code to use | Evidence to add |
|---|---|---|
| Field strength or mesh | `make_hartmann_case(ha=..., ny=..., nz=...)` | observable versus at least three mesh levels |
| Forcing or solver controls | `dataclasses.replace(case, forcing=..., time_stepper=...)` | tolerance and iteration independence |
| Conducting wall layers | `make_hunt_case(...)` or `generate_multilayer_duct_mesh(...)` | interface-current and wall-resolution checks |
| Your post-processing | `solution.fields` and `solution.diagnostics` | units, normalization, and an acceptance threshold |

The fully worked [`examples/hartmann_example.py`](examples/hartmann_example.py)
adds editable numerical controls, JSON reporting, analytical curves, and plot
generation without introducing another API.

## Run the examples

Every maintained example is a single editable file and writes generated output
under the ignored `artifacts/examples/` directory.

| Command | Typical output | Intended use |
|---|---|---|
| `lmx examples/hartmann_case.toml` | terminal solution and diagnostics | fastest CLI start |
| `python examples/hartmann_example.py` | analytical error and conservation record | first Python solve |
| `python examples/hunt_example.py` | layered conducting-wall result | fully developed wall physics |
| `python examples/li_aln_wall_stack_example.py` | wall comparison, plots, diagnostics | edit material layers |
| `python examples/fringing_benchmark_demo.py` | 3-D field/flow plots and conservation JSON | edit a spatially varying field problem |
| `python examples/variable_field_extruded_demo.py` | gradient check, 41-step optimization curve, controls, metrics | adapt a differentiable design problem |
| `python examples/q2d_turbulence_demo.py` | 41 retained frames, decay curve, poster, optional MP4 | evolve or differentiate Q2D flow |

Use the linked tutorial in [`examples/catalog.toml`](examples/catalog.toml) for
the equations, parameters, expected outputs, and evidence status of each file.

![Differentiable field, wall, and geometry design](docs/_static/blanket_design_optimization.webp)

The design figure's middle curve contains all 41 optimization iterates. The
left panel shows the seven actual axial design stations; they are control
points, not a claimed mesh-convergence curve.

## Advanced: 3-D fringing and extruded fields

Use the extruded API after the fully developed workflow is familiar. It adds
axial velocity components, pressure projection, spatially varying imposed
fields, explicit wall regions, restart state, and stationwise conservation
checks. The following script constructs the problem, replaces its field and
forcing, solves it, and checks its three principal conservation diagnostics:

```python
from dataclasses import replace

import lmx
from lmx.fringing import (
    build_layered_duct_extruded_problem,
    smooth_fringing_profile,
)

problem = build_layered_duct_extruded_problem(
    ha_peak=6.0,
    width=2.0,
    height=1.4,
    length=3.0,
    nx_stations=7,
    ny=6,
    nz=6,
    wall_cells=1,
    wall_thickness=0.08,
    entry_center=0.75,
    exit_center=2.25,
    transition_width=0.25,
)
profile = smooth_fringing_profile(
    length=3.0,
    nx=7,
    entry_center=0.75,
    exit_center=2.25,
    transition_width=0.25,
    axis="z",
)
problem = replace(
    problem,
    case=replace(problem.case, forcing=0.8),
    profile=profile,
)
result = lmx.solve(problem)

checks = {
    "status": result.status,
    "residual": result.residual,
    "mass": result.validation.max_divergence_residual,
    "charge": result.validation.max_charge_balance_residual,
    "boundary_current": result.validation.net_boundary_current_residual,
}
print(checks)
```

This mesh is intentionally small enough for a portable demonstration. Its
bounded solve may report `step_limit`; do not treat that state as an accepted
steady result. Increase the mesh and work limits together, perform mesh and
tolerance studies, and require `result.converged` for production observables.

Use `build_square_duct_extruded_problem` for an insulating rectangular duct,
`build_layered_duct_extruded_problem` for explicit wall regions,
`build_pipe_ogrid_extruded_problem` for a straight conducting pipe, and
`build_extruded_problem_from_case` when you already have a complete `CaseSpec`.
Analytic and tabulated divergence-free imposed fields share the same solve
interface. Restart, NPZ, CSV, VTK/ParaView, and plotting helpers are documented
in the [output guide](https://lmx.readthedocs.io/en/latest/how_to/restart_and_output.html).

### Advanced: differentiate a design response

Generic rectangular/layered ducts and straight pipes expose the same finite
production recurrence through `evolve_extruded_fields`. Continuous forcing,
imposed-field coefficients, fluid/wall conductivity, and fixed-topology
geometry scales can reach field outputs and engineering objectives. SOLVAX
uses implicit electric derivatives and checkpointed finite recurrences so the
reverse pass does not retain every solver iteration.

```python
import jax
import jax.numpy as jnp
from lmx.fringing import evolve_extruded_fields, extruded_engineering_objectives


def pumping_power(field_scale):
    fields = evolve_extruded_fields(
        problem,
        magnetic_field_scale=field_scale,
        steps=8,
    )
    return extruded_engineering_objectives(problem, fields)["pumping_power"]


value, gradient = jax.jit(jax.value_and_grad(pumping_power))(jnp.ones(41))
```

Pass `num_devices` to shard a generic 3-D field evolution over an evenly
divisible axial mesh. Treat mesh, topology, discrete boundary kinds, iteration
counts, and sharding layout as static controls; independently check selected
gradients before using them in an optimizer.

### Advanced: evolve Q2D flow

```python
import lmx

case = lmx.make_q2d_case(
    shape=(64, 64),
    viscosity=2.0e-3,
    hartmann_friction=4.0e-2,
    history_stride=4,
)
result = lmx.solve(case)

print(result.status, result.diagnostics.energy_budget_residual)
```

`evolve_q2d` exposes continuous state, forcing, length, viscosity, Hartmann
friction, and timestep inputs through a checkpointed finite evolution.

![Q2D vorticity and 41-frame kinetic-energy decay](docs/_static/q2d_vortex_decay.webp)

## Capability and evidence status

| Capability | Public interface | Current evidence/status |
|---|---|---|
| Hartmann, Shercliff, Hunt ducts | `make_*_case`, `solve` | analytical profiles, conservation, power balance, and mesh refinement; stable |
| Differentiable steady ducts | `solve_fully_developed_fields` | production parity, finite differences, and implicit Krylov adjoints; supported envelope |
| Generic 3-D ducts and straight pipes | `solve_extruded_inductionless` | manufactured operators, mass/current closure, restart, field gradients, and sharded correctness; research stage |
| Differentiable generic 3-D fields | `evolve_extruded_fields`, `extruded_engineering_objectives` | JVP/VJP, finite differences, batched evaluation, and bounded reverse memory; research stage |
| Periodic Q2D | `make_q2d_case`, `solve`, `evolve_q2d` | decay identities, energy closure, refinement, and bounded reverse memory; research stage |
| ALEX B1/B2 | benchmark builders and validation scripts | frozen contracts and reduced/internal gates; B2 FreeMHD smoke passes, production acceptance remains open |
| Accelerator and multi-device execution | JAX backend plus `num_devices` | portable/sharded correctness exists; real A4000 speed and strong-scaling claims remain open |

LMX does not currently claim free-surface MHD, full magnetic induction, thermal
coupling, general 3-D turbulence, curved-pipe physics, or production-accepted
ALEX B1/B2 design gradients. FreeMHD is an independently executed validation
comparator, not an LMX dependency.

## Documentation

- [Install and first run](https://lmx.readthedocs.io/en/latest/getting_started/install.html)
- [Fully developed ducts](https://lmx.readthedocs.io/en/latest/tutorials/fully_developed.html)
- [Three-dimensional fringing fields](https://lmx.readthedocs.io/en/latest/tutorials/fringing.html)
- [Walls and imposed fields](https://lmx.readthedocs.io/en/latest/tutorials/walls_and_fields.html)
- [Differentiable design](https://lmx.readthedocs.io/en/latest/tutorials/differentiation.html)
- [Q2D vortex dynamics](https://lmx.readthedocs.io/en/latest/tutorials/q2d.html)
- [Equations and numerical methods](https://lmx.readthedocs.io/en/latest/physics/equations.html)
- [Validation matrix](https://lmx.readthedocs.io/en/latest/validation/index.html)
- [Python API](https://lmx.readthedocs.io/en/latest/reference/api.html)

## Development and citation

```console
python -m pip install -e ".[dev,docs]"
python scripts/run_full_test_suite.py --changed-from HEAD
python -m sphinx -W -b html docs docs/_build/html
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for evidence boundaries and the complete
candidate gate. Cite the exact release or commit that produced a result;
metadata is in [CITATION.cff](CITATION.cff).

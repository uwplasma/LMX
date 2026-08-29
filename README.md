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

The first three fully developed examples are stable portable entry points. The
wall, 3-D, design, and Q2D examples are research-stage workflows: they exercise
real supported equations, but their small default meshes are demonstrations,
not publication validation.

## Install and run

LMX is currently installed from source; there is no LMX release on PyPI yet.

```console
git clone https://github.com/uwplasma/LMX.git
cd LMX
python -m pip install -e ".[visualization]"
lmx examples/hartmann_case.toml
```

Or use the same API from Python:

```python
import lmx

case = lmx.make_hartmann_case(ha=20.0, ny=48, nz=48)
result = lmx.solve(case)

print(result.status, result.steps, result.residual)
print(result.diagnostics.charge_balance_residual_history[-1])
```

JAX selects the CPU by default. Install the accelerator-specific JAX wheel by
following the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html).

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

## Apply LMX to your problem

An LMX study has four explicit layers:

1. Choose the closest retained model: fully developed 2-D, generic extruded
   3-D duct/pipe, or periodic Q2D.
2. Define geometry, material regions, imposed field, forcing, and numerical
   controls with immutable case objects or a convenience builder.
3. Solve and inspect termination together with mass, charge, interface-current,
   and momentum/energy diagnostics.
4. Establish mesh, tolerance, and external-code independence for the observable
   you intend to publish. A portable example result is not that evidence.

For example, start a layered 3-D duct with a smooth transverse fringe, then
edit only the quantities that define your experiment:

```python
from dataclasses import replace

import lmx
from lmx.fringing import (
    build_layered_duct_extruded_problem,
    smooth_fringing_profile,
)

problem = build_layered_duct_extruded_problem(
    ha_peak=30.0,
    width=2.0,
    height=1.4,
    length=8.0,
    nx_stations=41,
    ny=40,
    nz=32,
    wall_cells=3,
    wall_thickness=0.08,
)
profile = smooth_fringing_profile(
    length=8.0,
    nx=41,
    entry_center=2.0,
    exit_center=6.0,
    transition_width=0.3,
    axis="z",
)
problem = replace(
    problem,
    case=replace(problem.case, forcing=0.8),
    profile=profile,
)
result = lmx.solve(problem)

print(result.status)
print(result.validation.max_divergence_residual)
print(result.validation.max_charge_balance_residual)
print(result.validation.net_boundary_current_residual)
```

Use `build_square_duct_extruded_problem` for an insulating rectangular duct,
`build_layered_duct_extruded_problem` for explicit wall regions,
`build_pipe_ogrid_extruded_problem` for a straight conducting pipe, and
`build_extruded_problem_from_case` when you already have a complete `CaseSpec`.
Analytic and tabulated divergence-free imposed fields share the same solve
interface. Restart, NPZ, CSV, VTK/ParaView, and plotting helpers are documented
in the [output guide](https://lmx.readthedocs.io/en/latest/how_to/restart_and_output.html).

### Differentiate a design response

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

### Evolve Q2D flow

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

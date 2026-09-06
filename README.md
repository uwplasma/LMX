# LMX

[![Python](https://img.shields.io/badge/python-3.10--3.13-3776ab.svg)](https://www.python.org/)
[![CI](https://img.shields.io/github/actions/workflow/status/uwplasma/LMX/ci.yml?branch=main&label=ci)](https://github.com/uwplasma/LMX/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/readthedocs/lmx/latest?label=docs)](https://lmx.readthedocs.io/)
[![License](https://img.shields.io/github/license/uwplasma/LMX)](LICENSE)

LMX is a compact JAX code for inductionless liquid-metal magnetohydrodynamics
in ducts: analytical-reference Hartmann, Shercliff and Hunt flow; 3-D straight
ducts and pipes in spatially varying magnetic fields; periodic Q2D vortex
dynamics; and differentiable field, wall and fixed-topology geometry studies.
LMX owns the equations, boundary conditions, coupling, diagnostics and
validation; [SOLVAX](https://github.com/uwplasma/SOLVAX) (≥0.19) owns the
reusable linear, fixed-point, preconditioning and implicit-derivative
algorithms.

![Full-profile analytical validation for Hartmann, Shercliff, and Hunt ducts](docs/_static/analytic_velocity_profiles.webp)

## Install

```console
git clone https://github.com/uwplasma/LMX.git
cd LMX
python -m pip install -e ".[visualization]"
lmx examples/hartmann_case.toml
```

The last command is a complete smoke test and should end with a terminal
status and conservation diagnostics. JAX runs on the CPU by default; install
the accelerator wheel from the
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html)
for GPU execution.

## First study

Save this as `first_study.py` and run it. It builds an insulating Hartmann
duct, solves it, rejects an unfinished solve, compares the full centerline
with the analytical profile and writes fields and profiles.

```python
from pathlib import Path

import lmx
from lmx.io import write_solution_outputs
from lmx.validation import hartmann_validation

ha = 10.0
output_dir = Path("artifacts/first-study")

case = lmx.make_hartmann_case(
    ha=ha, width=2.0, height=2.0, ny=16, nz=16,
    conductivity=1.0, density=1.0, viscosity=1.0, output_dir=str(output_dir),
)
solution = lmx.solve(case)
if not solution.converged:
    raise RuntimeError(f"solve ended with {solution.status!r} after {solution.steps} steps")

comparison = hartmann_validation(solution, ha)
charge = float(solution.diagnostics.charge_balance_residual_history[-1])
generated = write_solution_outputs(solution, case, output_dir)

print(f"status={solution.status}, residual={solution.residual:.3e}")
print(f"profile L2 error={comparison.l2_error:.3e}, charge={charge:.3e}")
print({kind: [str(path) for path in paths] for kind, paths in generated.items()})
```

Change one layer at a time and keep the validation step beside the solve:
`make_hartmann_case(ha=..., ny=..., nz=...)` for field and mesh,
`dataclasses.replace(case, forcing=..., time_stepper=...)` for drive and
solver controls, `make_hunt_case(...)` or `generate_multilayer_duct_mesh(...)`
for conducting walls. Report an observable against at least three mesh
levels before calling it a result.

## Examples

Each example is one editable file that writes under the ignored
`artifacts/examples/` directory. Parameters, expected outputs and evidence
status are listed in [`examples/catalog.toml`](examples/catalog.toml).

| Command | What it shows | Status |
|---|---|---|
| `lmx examples/hartmann_case.toml` | CLI solve, terminal diagnostics | stable |
| `python examples/hartmann_example.py` | analytical error, conservation, convergence | stable |
| `python examples/hunt_example.py` | layered conducting-wall duct | stable |
| `python examples/li_aln_wall_stack_example.py` | explicit material layers, interface currents | research |
| `python examples/fringing_benchmark_demo.py` | 3-D fringing field, station diagnostics, plots | research |
| `python examples/variable_field_extruded_demo.py` | checked gradients, bounded 40-step design | research |
| `python examples/q2d_turbulence_demo.py` | Q2D vorticity frames, energy decay, poster, MP4 | research |

Research-stage examples run real supported equations on small demonstration
meshes; they are not publication validation.

![Differentiable field, wall, and geometry design](docs/_static/blanket_design_optimization.webp)

The design figure's loss is a pressure-tap drop and excludes prescribed
body-drive work; see the
[work conventions](docs/physics/equations.md#pressure-taps-and-mechanical-work).

## 3-D fringing fields

Extruded problems add axial velocity, pressure projection, spatially varying
imposed fields, explicit wall regions, restart state and stationwise
conservation checks:

```python
from dataclasses import replace

import lmx
from lmx.fringing import build_layered_duct_extruded_problem, smooth_fringing_profile

problem = build_layered_duct_extruded_problem(
    ha_peak=6.0, width=2.0, height=1.4, length=3.0, nx_stations=7, ny=6, nz=6,
    wall_cells=1, wall_thickness=0.08,
    entry_center=0.75, exit_center=2.25, transition_width=0.25,
)
profile = smooth_fringing_profile(
    length=3.0, nx=7, entry_center=0.75, exit_center=2.25, transition_width=0.25, axis="z",
)
problem = replace(problem, case=replace(problem.case, forcing=0.8), profile=profile)
result = lmx.solve(problem)
print(result.status, result.residual, result.validation.max_charge_balance_residual)
```

A bounded solve may report `step_limit`; require `result.converged` and a
mesh/tolerance study before using production observables. Builders:
`build_square_duct_extruded_problem`, `build_layered_duct_extruded_problem`,
`build_pipe_ogrid_extruded_problem`, `build_extruded_problem_from_case`.
Restart, NPZ, CSV and VTK output are in the
[output guide](https://lmx.readthedocs.io/en/latest/how_to/restart_and_output.html).

### Differentiate a design response

```python
import jax
import jax.numpy as jnp
from lmx.fringing import evolve_extruded_fields


def mean_squared_velocity(field_scale):
    fields = evolve_extruded_fields(problem, magnetic_field_scale=field_scale, steps=8)
    return jnp.mean(fields[0] ** 2)


field_scale = jnp.ones_like(problem.profile.field_scale)
value, gradient = jax.jit(jax.value_and_grad(mean_squared_velocity))(field_scale)
```

This is a finite-step model response, not a converged steady objective. The
generic 3-D recurrence omits convective momentum transport, and its sampled
base field is fixed, so these are not coil or moving-channel sensitivities.
Pass `num_devices` to shard an evenly divisible axial mesh; check selected
gradients independently before optimizing.

## Q2D flow

```python
import lmx

case = lmx.make_q2d_case(shape=(64, 64), viscosity=2.0e-3, hartmann_friction=4.0e-2, history_stride=4)
result = lmx.solve(case)
print(result.status, result.diagnostics.energy_budget_residual)
```

A trajectory is accepted only when its Courant and energy-budget checks pass
(`energy_budget_tolerance`, default `1e-3`); `converged` means accepted finite
evolution, not steady flow. `evolve_q2d` exposes the checkpointed evolution to
JAX transformations. See the [Q2D tutorial](docs/tutorials/q2d.md).

![Q2D vorticity and kinetic-energy decay](docs/_static/q2d_vortex_decay.webp)

## Capability and evidence

| Capability | Interface | Evidence |
|---|---|---|
| Hartmann, Shercliff, Hunt ducts | `make_*_case`, `solve` | analytical profiles, conservation, power balance, refinement; stable |
| Differentiable steady ducts | `solve_fully_developed_fields` | finite differences and implicit Krylov adjoints; supported |
| Generic 3-D ducts and pipes | `solve_extruded_inductionless` | manufactured operators, mass/current closure, restart, sharded parity; research |
| Differentiable 3-D fields | `evolve_extruded_fields`, `extruded_engineering_objectives` | JVP/VJP, finite differences, bounded reverse memory; research |
| Periodic Q2D | `make_q2d_case`, `solve`, `evolve_q2d` | decay identities, energy closure, refinement, CPU/GPU parity; research |
| ALEX B1/B2 | benchmark builders, validation scripts | frozen contracts; B2 FreeMHD smoke passes; production acceptance open |
| GPU and multi-device | JAX backend, `num_devices` | correctness only; speed and strong scaling not yet established |

LMX does not claim free-surface MHD, magnetic induction, thermal coupling,
general 3-D turbulence, curved-pipe physics or production-accepted ALEX
gradients. FreeMHD is an independently executed comparator, not a dependency.

## Documentation

[Install](https://lmx.readthedocs.io/en/latest/getting_started/install.html) ·
[Fully developed ducts](https://lmx.readthedocs.io/en/latest/tutorials/fully_developed.html) ·
[Fringing fields](https://lmx.readthedocs.io/en/latest/tutorials/fringing.html) ·
[Walls and fields](https://lmx.readthedocs.io/en/latest/tutorials/walls_and_fields.html) ·
[Differentiation](https://lmx.readthedocs.io/en/latest/tutorials/differentiation.html) ·
[Q2D](https://lmx.readthedocs.io/en/latest/tutorials/q2d.html) ·
[Equations](https://lmx.readthedocs.io/en/latest/physics/equations.html) ·
[Validation](https://lmx.readthedocs.io/en/latest/validation/index.html) ·
[API](https://lmx.readthedocs.io/en/latest/reference/api.html)

## Development and citation

```console
python -m pip install -e ".[dev,docs]"
python scripts/run_full_test_suite.py --changed-from HEAD
python -m sphinx -W -b html docs docs/_build/html
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for evidence rules and
[plan.md](plan.md) for the roadmap. Cite the exact release or commit;
metadata is in [CITATION.cff](CITATION.cff).

# LMX

[![Python](https://img.shields.io/badge/python-3.10--3.13-3776ab.svg)](https://www.python.org/)
[![CI](https://img.shields.io/github/actions/workflow/status/uwplasma/LMX/ci.yml?branch=main&label=ci)](https://github.com/uwplasma/LMX/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/readthedocs/lmx/latest?label=docs)](https://lmx.readthedocs.io/)
[![License](https://img.shields.io/github/license/uwplasma/LMX)](LICENSE)

LMX is a compact JAX toolkit for inductionless liquid-metal MHD. Solve and
differentiate fully developed duct flow, study conducting walls, explore
straight-channel fringing fields, and evolve periodic quasi-two-dimensional
(Q2D) vortices. LMX owns the physics; [SOLVAX](https://github.com/uwplasma/SOLVAX)
supplies reusable linear solvers and implicit derivatives.

The strongest validated starting point is Hartmann, Shercliff and Hunt flow.
Advanced 3-D models are research-stage, isothermal and restricted; LMX does
not yet model heat transfer, arbitrary curved blankets or general 3-D turbulence.

![Analytical and computed Hartmann, Shercliff and Hunt velocity profiles](docs/_static/analytic_velocity_profiles.webp)

## Install and run

```console
git clone https://github.com/uwplasma/LMX.git
cd LMX
python -m pip install -e ".[visualization]"
lmx examples/hartmann_case.toml
```

Use Python 3.10–3.13. The last command runs a complete duct example.
For GPUs, follow the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html).
Multi-device correctness does not imply speedup; benchmark your problem size.

## Your first study: build → solve → check → save

Save this as `first_study.py` and run `python first_study.py`:

```python
from pathlib import Path
import lmx
from lmx.io import write_solution_outputs
from lmx.validation import hartmann_validation

ha = 10.0
output = Path("artifacts/first-study")
case = lmx.make_hartmann_case(
    ha=ha, width=2.0, height=2.0, ny=16, nz=16,
    conductivity=1.0, density=1.0, viscosity=1.0,
    output_dir=str(output),
)
solution = lmx.solve(case)
if not solution.converged:
    raise RuntimeError(f"{solution.status}: residual={solution.residual:.3e}")

comparison = hartmann_validation(solution, ha)
files = write_solution_outputs(solution, case, output)
print("Profile L2 error:", comparison.l2_error)
print("Charge residual:", solution.diagnostics.charge_balance_residual_history[-1])
print(files)  # NPZ/restart, CSV profiles and VTK fields.
```

For your research, change the mesh, field and material inputs, or start from
`make_shercliff_case` / `make_hunt_case`. Use
`dataclasses.replace(case, forcing=...)` to change the drive.
Check observable convergence on at least three meshes and tighter solver
tolerances; a completed solve alone is not validation.

### Differentiate a steady duct response

This field core uses implicit derivatives of the coupled steady equations,
without retaining the linear-solver iteration history:

```python
import jax
import jax.numpy as jnp
import lmx

case = lmx.make_shercliff_case(ha=10.0, ny=16, nz=16)

def mean_velocity(forcing, field_scale):
    velocity, *_ = lmx.solve_fully_developed_fields(
        case, forcing=forcing, magnetic_field_scale=field_scale,
    )
    return jnp.mean(velocity)

value, derivatives = jax.jit(
    jax.value_and_grad(mean_velocity, argnums=(0, 1))
)(1.0, 1.0)
print(value, derivatives)  # Sensitivities to drive and imposed-field scale.
```

These inputs do not provide arbitrary geometry or live coil derivatives.
See the [differentiation tutorial](docs/tutorials/differentiation.md) for
gradient checks, finite-trajectory design and supported parameter shapes.

## Choose an example

Each script has editable settings and writes to ignored `artifacts/examples/`.
The [catalog](examples/catalog.toml) links equations, parameters and evidence.

| Run | What you can adapt |
|---|---|
| `python examples/hartmann_example.py` | First validated duct study: profiles, error and conservation |
| `python examples/hunt_example.py` | Fully developed flow with conducting walls |
| `python examples/li_aln_wall_stack_example.py` | Research: explicit conducting/insulating material layers |
| `python examples/fringing_benchmark_demo.py` | Advanced: spatially varying field, flow/current plots and diagnostics |
| `python examples/variable_field_extruded_demo.py` | Advanced: finite-step design with gradient checks and optimization history |
| `python examples/q2d_turbulence_demo.py` | Advanced: Q2D vortex **decay**, energy history and optional movie |

![Finite-step field, wall and geometry design with optimization history](docs/_static/blanket_design_optimization.webp)

This design example minimizes a pressure-tap-based objective, not certified
total pump work or thermal blanket performance. Seven axial controls are
discrete design stations; the optimization curve contains all 41 iterates.
See [work conventions](docs/physics/equations.md#pressure-taps-and-mechanical-work).

## Advanced: 3-D fringing fields

```python
from dataclasses import replace
import lmx
from lmx.fringing import build_layered_duct_extruded_problem

problem = build_layered_duct_extruded_problem(
    ha_peak=6.0, width=2.0, height=1.4, length=3.0,
    nx_stations=7, ny=6, nz=6, wall_cells=1, wall_thickness=0.08,
    entry_center=0.75, exit_center=2.25, transition_width=0.25,
)
problem = replace(problem, case=replace(problem.case, forcing=0.8))
result = lmx.solve(problem)
print(result.status, result.residual)
print("Charge residual:", result.validation.max_charge_balance_residual)
```

This small bounded demonstration may report `step_limit`; it is not an
accepted steady result. Generic duct/pipe evolution omits momentum advection
and uses a restricted projection/flow adjustment. ALEX B1/B2 production
acceptance remains open. Use the [numerical contracts](docs/physics/numerics.md),
not mass/current closure alone, to judge suitability.

Other builders cover insulating rectangles and straight pipes.
[Fringing](docs/tutorials/fringing.md), [field data](docs/tutorials/walls_and_fields.md)
and [output/restart](docs/how_to/restart_and_output.md) show how to customize them.
`evolve_extruded_fields` provides checked finite-step derivatives and optional
axial sharding; geometry scaling holds sampled fields fixed, so it is not
end-to-end coil or moving-channel optimization.

## Advanced: Q2D dynamics

```python
import lmx

case = lmx.make_q2d_case(
    shape=(64, 64), viscosity=0.01, hartmann_friction=0.1,
    dt=0.01, steps=160, history_stride=4,
)
result = lmx.solve(case)
print(result.status, result.diagnostics.energy_budget_residual)
```

Here `converged` means an accepted finite trajectory, not a steady state.
Courant and energy-budget checks must pass. This periodic depth-averaged
example shows vortex decay, not turbulent pipe flow or heat transport.

![Q2D vorticity and kinetic-energy decay](docs/_static/q2d_vortex_decay.webp)

## Documentation and contributing

[First run](docs/getting_started/first_run.md) ·
[Equations](docs/physics/equations.md) ·
[API](docs/reference/api.md) ·
[Validation](docs/validation/index.md) ·
[Q2D](docs/tutorials/q2d.md) ·
[Online documentation](https://lmx.readthedocs.io/)

```console
python -m pip install -e ".[dev,docs]"
python scripts/run_full_test_suite.py --changed-from HEAD
python -m sphinx -W -b html docs docs/_build/html
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for full qualification and
[plan.md](plan.md) for development priorities. Cite the version/commit used;
metadata is in [CITATION.cff](CITATION.cff).

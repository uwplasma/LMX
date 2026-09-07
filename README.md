# LMX

**Differentiable inductionless liquid-metal MHD in JAX.**

[![CI](https://img.shields.io/github/actions/workflow/status/uwplasma/LMX/ci.yml?branch=main&label=ci)](https://github.com/uwplasma/LMX/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/readthedocs/lmx/latest?label=docs)](https://lmx.readthedocs.io/)
[![Python](https://img.shields.io/badge/python-3.10--3.13-3776ab.svg)](https://www.python.org/)
[![License](https://img.shields.io/github/license/uwplasma/LMX)](LICENSE)

LMX solves the flow of liquid metals in strong magnetic fields, the physics of
fusion blanket channels: Hartmann, Shercliff and Hunt ducts with conducting
walls, three-dimensional ducts and pipes in spatially varying fields, and
quasi-two-dimensional vortex dynamics. Every solve runs on CPU or GPU and is
differentiable, so pressure drop, wall currents and flow profiles can be
optimised with exact gradients. Reusable solvers and implicit derivatives come
from [SOLVAX](https://github.com/uwplasma/SOLVAX).

<p align="center">
<img src="docs/_static/q2d_turbulence_256.webp" width="320" alt="Decaying quasi-two-dimensional MHD turbulence, 256 by 256, vorticity">
<br><em>Quasi-2D MHD turbulence with Hartmann-layer friction (256², 3,000 steps, 20 s on a laptop CPU).</em>
</p>

## Install

```console
pip install "git+https://github.com/uwplasma/LMX.git#egg=lmx[visualization]"
lmx examples/hartmann_case.toml
```

JAX runs on the CPU by default; install the GPU wheel from the
[JAX guide](https://docs.jax.dev/en/latest/installation.html) and LMX uses it.

## Quickstart: solve, validate, differentiate

```python
import jax, jax.numpy as jnp, lmx
from lmx.validation import hartmann_validation

case = lmx.make_hartmann_case(ha=20.0, ny=32, nz=32)
solution = lmx.solve(case)
print(solution.status, hartmann_validation(solution, 20.0).l2_error)

def mean_velocity(forcing, field_scale):
    u, *_ = lmx.solve_fully_developed_fields(case, forcing=forcing, magnetic_field_scale=field_scale)
    return jnp.mean(u)

print(jax.grad(mean_velocity, argnums=(0, 1))(1.0, 1.0))
```

The gradient comes from an implicit adjoint of the steady equations, not from
taping the linear solver.

## Gallery

| | |
|:---:|:---:|
| ![Hunt duct side-layer jets from Ha 20 to 1000](docs/_static/hunt_side_layers.webp) | ![Hartmann, Shercliff and Hunt profiles against analytical solutions](docs/_static/analytic_velocity_profiles.webp) |
| Hunt duct, Ha 20 → 1000: side-layer jets and their Ha^-1/2 thickness · `lmx run hunt --ha 1000` | Analytical validation at Ha 20 · `python examples/hartmann_example.py` |
| ![Q2D turbulence snapshots and energy spectrum](docs/_static/q2d_turbulence_poster.webp) | ![Field, wall and geometry design with gradient descent](docs/_static/blanket_design_optimization.webp) |
| Q2D vortex merging and a k⁻³ enstrophy range · `python examples/q2d_turbulence_demo.py` | Differentiable field, wall and geometry design · `python examples/variable_field_extruded_demo.py` |

## Examples

| Command | Physics |
|---|---|
| `lmx examples/hartmann_case.toml` | Hartmann duct from a TOML file, terminal diagnostics |
| `python examples/hartmann_example.py` | analytical error, conservation, mesh convergence |
| `python examples/hunt_example.py` | conducting Hartmann walls, side-layer jets |
| `python examples/li_aln_wall_stack_example.py` | explicit wall material layers and interface currents |
| `python examples/fringing_benchmark_demo.py` | 3-D duct entering a magnetic field |
| `python examples/variable_field_extruded_demo.py` | gradient-based field, wall and geometry design |
| `python examples/q2d_turbulence_demo.py` | Q2D vorticity evolution, energy decay, movie |

Each example is one editable file that writes to `artifacts/examples/`;
parameters and evidence status are in [`examples/catalog.toml`](examples/catalog.toml).
`python scripts/make_showcase_figures.py` regenerates the gallery.

## What is validated, what is research

- **Validated:** Hartmann, Shercliff and Hunt ducts against analytical profiles
  with conservation and mesh-refinement checks; implicit adjoints against
  finite differences; Q2D decay identities and energy budgets.
- **Research stage:** the 3-D duct and pipe models (no convective transport
  yet, short bounded transients), the ALEX B1/B2 fringing-field benchmarks
  (FreeMHD smoke comparison passes; production acceptance open), and
  multi-device execution (correct, not yet faster). The
  [validation matrix](https://lmx.readthedocs.io/en/latest/validation/index.html)
  and [plan](plan.md) state each gate.

## Documentation

[Install](https://lmx.readthedocs.io/en/latest/getting_started/install.html) ·
[Tutorials](https://lmx.readthedocs.io/en/latest/tutorials/fully_developed.html) ·
[Equations](https://lmx.readthedocs.io/en/latest/physics/equations.html) ·
[Validation](https://lmx.readthedocs.io/en/latest/validation/index.html) ·
[API](https://lmx.readthedocs.io/en/latest/reference/api.html) ·
[Roadmap](plan.md)

## Cite and contribute

Cite the commit or release you used; metadata is in [CITATION.cff](CITATION.cff).
Development: `pip install -e ".[dev,docs]"`, then
`python scripts/run_full_test_suite.py --changed-from HEAD`. See
[CONTRIBUTING.md](CONTRIBUTING.md).

# LMX

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![JAX](https://img.shields.io/badge/JAX-CPU%20%7C%20GPU-orange.svg)](https://docs.jax.dev/)
[![Docs](https://img.shields.io/badge/docs-read%20online-blue.svg)](https://lmx.readthedocs.io/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**LMX is a JAX-native code for inductionless liquid-metal MHD on CPUs and
GPUs.** It combines verified duct solvers, conducting walls, nonuniform magnetic
fields, automatic differentiation, and research-stage 3D flow workflows.

> Hartmann, Shercliff, and Hunt ducts are verified. Extruded fringe fields,
> magnetic obstacles, Q2D turbulence, and blanket models remain research-stage.

[Documentation](https://lmx.readthedocs.io/) ·
[Quickstart](docs/getting_started.md) ·
[Examples](docs/case_cookbook.md) ·
[Validation](docs/benchmark_matrix.md) ·
[Roadmap](plan.md)

![Analytical and LMX duct-flow profiles](docs/_static/analytic_velocity_profiles.webp)

## Install and run

```bash
git clone https://github.com/uwplasma/LMX.git
cd LMX && python -m pip install -e .
lmx examples/hartmann_case.toml
```

Or from Python:

```python
from lmx import make_hartmann_case, solve_steady

solution = solve_steady(make_hartmann_case(ha=20, ny=48, nz=48))
print(solution.diagnostics.volumetric_flow_rate_history[-1])
```

LMX provides TOML and Python inputs, restarts, diagnostics, plotting, and
checksummed research artifacts. See the [case cookbook](docs/case_cookbook.md)
for ducts, walls, fringe fields, custom fields, and mapped pipes.

## Features

✅ native and documented · ◐ partial/research-stage · ❌ not provided natively

| Capability | LMX | [FreeMHD](https://github.com/PlasmaControl/FreeMHD) | [NekRS](https://nekrs.readthedocs.io/en/latest/) |
|---|:---:|:---:|:---:|
| Inductionless liquid-metal MHD | ✅ | ✅ | ❌¹ |
| Full induction / finite magnetic Reynolds number | ❌ | ✅² | ❌¹ |
| High-Hartmann-number duct flows | ✅ | ✅ | ❌¹ |
| 3D imposed and fringing magnetic fields | ◐ | ✅ | ❌¹ |
| Insulating and conducting wall-current closure | ✅ | ✅ | ❌¹ |
| Free-surface / two-phase liquid-metal MHD | ❌ | ✅ | ❌¹ |
| Fluid–solid heat transfer | ❌ | ◐ | ✅ CFD |
| Turbulence models | ◐ Q2D | ◐ non-MHD | ✅ CFD |
| General curved / complex 3D meshes | ◐ | ✅ | ✅ |
| Parallel CPU execution | ✅ | ✅ | ✅ |
| Native GPU and multi-GPU execution | ✅ | ❌ | ✅ |
| End-to-end automatic differentiation | ✅ | ❌ | ❌ |
| Published liquid-metal experimental validation | ◐ | ✅ | ❌¹ |

¹ Stock NekRS is a scalable spectral-element thermal-fluids solver with custom
source hooks, but no native electromagnetic equations or current closure.
² The 2026 [FreeMHD2 extension](https://arxiv.org/abs/2606.18745) adds verified
finite-Rm vector-potential induction. Sources: the peer-reviewed
[FreeMHD validation paper](https://doi.org/10.1063/5.0230242), the
[NekRS capability guide](https://nekrs.readthedocs.io/en/latest/), and its
[multi-GPU paper](https://arxiv.org/abs/2104.05829). The table compares stock,
documented capability—not what could be implemented through extensions.

## Verified duct MHD

Hartmann, Shercliff, and Hunt profiles are tested against analytical solutions,
conservation identities, and mesh ladders. All eight frozen high-Hartmann rows
pass; audited closed-channel FreeMHD observables pass the 1% finite-grid gate.

<p align="center">
  <img src="docs/_static/freemhd_closed_channel_observable_parity.webp" alt="LMX and FreeMHD closed-channel observable parity" width="58%">
  <a href="docs/_static/readme_hunt_startup_2d.mp4"><img src="docs/_static/readme_hunt_startup_2d_poster.webp" alt="Hunt-flow startup movie" width="38%"></a>
</p>

[Evidence, definitions, and acceptance criteria →](docs/validation_report.md)

## Real geometries and nonuniform fields

Rectangular ducts, layered conducting walls, mapped pipe O-grids, analytic
volume fields, and tabulated fields share one inductionless solver surface.

<p align="center">
  <img src="https://github.com/uwplasma/LMX/releases/download/lmx-research-assets-v1/readme-geometries.webp" alt="LMX rectangular, layered, and mapped-pipe geometries" width="48%">
  <img src="https://github.com/uwplasma/LMX/releases/download/lmx-research-assets-v1/readme-variable-field.webp" alt="Nonuniform-field duct response and charge conservation" width="48%">
</p>

![B2 Maxwell-consistent fringe field and ALEX pressure diagnostics](https://github.com/uwplasma/LMX/releases/download/lmx-research-assets-v1/readme-alex-b2-field-pressure.webp)

The B2 diagnostics pass conservation, restart, and two-GPU parity checks;
experimental and exact matched-FreeMHD acceptance remain open.
[Fringing-field status →](docs/fringing.md)

## Differentiate the solver

Selected steady and extruded objectives support JAX gradients and inverse
design. Promoted gradients are checked against finite differences or independent
transpose solves—not merely whether `jax.grad` runs.

![LMX sensitivity and inverse design](https://github.com/uwplasma/LMX/releases/download/lmx-research-assets-v1/readme-autodiff.webp)

[Differentiable workflows →](docs/autodiff.md)

## Explore research-stage flows

LMX also exposes bounded magnetic-obstacle and reduced blanket workflows. These
visuals demonstrate implemented diagnostics and geometry, not validation claims.

<p align="center">
  <img src="https://github.com/uwplasma/LMX/releases/download/lmx-research-assets-v1/readme-magnetic-obstacle.webp" alt="Magnetic-obstacle response" width="48%">
  <a href="https://github.com/uwplasma/LMX/releases/download/lmx-research-assets-v1/readme-blanket-flow.mp4"><img src="https://github.com/uwplasma/LMX/releases/download/lmx-research-assets-v1/readme-blanket-flow-poster.webp" alt="Reduced WHAM blanket-flow movie" width="48%"></a>
</p>

[Geometry and field workflows →](docs/geometry.md)

## Scale on CPUs and GPUs

The B2 numerical checkpoint scales from 36.96 s on one RTX A4000 to 22.23 s on
two: **1.66× speedup at 83.1% efficiency**, with equivalent observables. A
fine-grid fast-diagonalization update separately reduces matched electric-solve
time **1.87×**. Broader multi-device scaling remains open.

![LMX GPU strong scaling](docs/_static/strong_scaling.webp)

[Measurement protocol and full results →](docs/performance.md)

## Quality, documentation, and citation

The portable gate currently passes **770 tests**, **95.35% branch coverage**,
and a **160.3 s** wall time on six Apple-Silicon workers. Physics and external
campaigns add analytical, conservation, convergence, FreeMHD, and experimental
evidence outside that fast gate.

[Testing](docs/testing.md) · [Theory](docs/theory.md) ·
[Numerics](docs/numerics.md) · [Contributing](CONTRIBUTING.md) ·
[Research assets](https://github.com/uwplasma/LMX/releases/tag/lmx-research-assets-v1)

LMX is MIT licensed. Cite the project using [CITATION.cff](CITATION.cff).

# LMX

[![Python](https://img.shields.io/badge/python-3.10--3.13-blue.svg)](https://github.com/uwplasma/LMX/blob/main/pyproject.toml)
[![JAX](https://img.shields.io/badge/JAX-CPU%20%7C%20GPU-orange.svg)](https://docs.jax.dev/)
[![Docs](https://img.shields.io/badge/docs-read%20online-blue.svg)](https://lmx.readthedocs.io/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/uwplasma/LMX/blob/main/LICENSE)

**LMX is JAX-native inductionless liquid-metal MHD for CPUs and GPUs.**
Verified: Hartmann, Shercliff, and Hunt ducts, conducting walls, and selected
differentiable workflows. Research-stage: imposed 3D/fringe fields, magnetic
obstacles, Q2D turbulence, blanket models, and extruded flows.

[Documentation](https://lmx.readthedocs.io/) ·
[Quickstart](https://lmx.readthedocs.io/en/latest/getting_started.html) ·
[Examples](https://lmx.readthedocs.io/en/latest/case_cookbook.html) ·
[Validation](https://lmx.readthedocs.io/en/latest/benchmark_matrix.html) ·
[Roadmap](https://github.com/uwplasma/LMX/blob/main/plan.md)

![Duct-flow profiles](https://raw.githubusercontent.com/uwplasma/LMX/main/docs/_static/analytic_velocity_profiles.webp)

## Install and run

```bash
git clone https://github.com/uwplasma/LMX.git
cd LMX && python -m pip install -e '.[visualization]'
lmx examples/hartmann_case.toml
```

```python
from lmx import make_hartmann_case, solve_steady

solution = solve_steady(make_hartmann_case(ha=20, ny=48, nz=48))
print(solution.diagnostics.volumetric_flow_rate_history[-1])
```

TOML/Python cases support restarts, diagnostics, plots, and checksummed artifacts.
For the lean core, use `pip install -e .`; see the
[case cookbook](https://lmx.readthedocs.io/en/latest/case_cookbook.html).

## Capabilities

✅ native/documented · ◐ partial/research · — no named native workflow

| Capability | LMX | [FreeMHD](https://github.com/PlasmaControl/FreeMHD) | [NekRS](https://nekrs.readthedocs.io/en/latest/) |
|---|:---:|:---:|:---:|
| Inductionless electric-potential LM MHD | ✅ | ✅ | — |
| Full-induction / finite-Rm MHD | — | —¹ | ◐ research |
| Verified high-Hartmann duct benchmarks | ✅ | ✅ | — |
| 3D imposed/fringing-field LM workflow | ◐ | ✅ | — |
| Conducting/insulating wall-current closure | ✅ | ✅ | — |
| Fully 3D transient MHD | ◐ extruded | ✅ | ◐ research |
| Free-surface / two-phase MHD | — | ✅ | — |
| Fluid–solid conjugate heat transfer | — | ✅ | ✅ |
| Turbulence models | ◐ Q2D | ◐ OpenFOAM | ✅ LES/RANS |
| Curved / complex 3D meshes | ◐ mapped | ✅ | ✅ |
| Parallel CPU solver | ◐ | ✅ | ✅ |
| Single-GPU execution | ✅ | — | ✅ |
| Multi-GPU execution | ◐ B2 | — | ✅ |
| Selected reverse-mode AD workflows | ✅ | — | — |
| Published liquid-metal experiment comparison | ◐ | ✅ | — |

¹ [FreeMHD2](https://arxiv.org/abs/2606.18745) is a separate finite-Rm extension.
Sources: [FreeMHD paper](https://doi.org/10.1063/5.0230242),
[solver](https://github.com/PlasmaControl/FreeMHD/blob/main/MHD_Solvers/solvers/epotMultiRegionInterFoam/epotMultiRegionInterFoam.C),
[NekRS](https://nekrs.readthedocs.io/en/latest/), [MHD extension](https://doi.org/10.2172/2453867),
[GPU scaling](https://doi.org/10.1016/j.parco.2022.102982).

## Verified duct MHD

Eight frozen high-Hartmann rows pass; audited closed-channel FreeMHD
observables pass the 1% finite-grid gate.

![Samper Benchmark A validation](https://raw.githubusercontent.com/uwplasma/LMX/main/docs/_static/samper_benchmark_a.webp)

![Analytical duct and FreeMHD parity](https://raw.githubusercontent.com/uwplasma/LMX/main/docs/_static/freemhd_closed_channel_observable_parity.webp)

<p align="center">
  <img src="https://raw.githubusercontent.com/uwplasma/LMX/main/docs/_static/readme-hunt-startup.webp" alt="Hunt/Shercliff startup movie" width="82%">
</p>

[Validation evidence →](https://lmx.readthedocs.io/en/latest/validation_report.html)

## Real geometries and nonuniform fields

<p align="center">
  <img src="https://raw.githubusercontent.com/uwplasma/LMX/main/docs/_static/readme-geometries.webp" alt="LMX geometries" width="48%">
  <img src="https://raw.githubusercontent.com/uwplasma/LMX/main/docs/_static/readme-variable-field.webp" alt="Field response and conservation" width="48%">
</p>

![B2 fringe-field diagnostics](https://raw.githubusercontent.com/uwplasma/LMX/main/docs/_static/readme-alex-b2-field-pressure.webp)

B2 passes exact restart and schema-6 topology gates (1/2/4 CPU; 1/2 GPU).
Shared-norm acceleration fails; production parity/scaling promotion remain open.
[Fringing status →](https://lmx.readthedocs.io/en/latest/fringing.html)

## Follow curved pipes

![Bent-pipe and Dean-vortex gates](https://raw.githubusercontent.com/uwplasma/LMX/main/docs/_static/readme-curved-pipes.webp)

Mapped pipes have a low-De inductionless baseline; Dean vortices await their
literature gate. [Geometry status →](https://lmx.readthedocs.io/en/latest/geometry.html)

## Model conducting multilayer walls

![Li/AlN wall convergence](https://github.com/uwplasma/LMX/releases/download/lmx-research-assets-v1/readme-li-aln-multilayer-convergence.webp)

At `Ha = 220`, research-stage Li/AlN pressure/current change below 10% per mesh
step; experiment/blanket validation remain open. [Wall models →](https://lmx.readthedocs.io/en/latest/wall_models.html)

## Differentiate selected workflows

![Sensitivity and inverse design](https://raw.githubusercontent.com/uwplasma/LMX/main/docs/_static/readme-autodiff.webp)

Promoted objectives pass finite-difference/independent-transpose checks.
[Differentiable workflows →](https://lmx.readthedocs.io/en/latest/autodiff.html)

## Explore research flows

<p align="center">
  <img src="https://raw.githubusercontent.com/uwplasma/LMX/main/docs/_static/readme-magnetic-obstacle.webp" alt="Magnetic-obstacle response" width="48%">
  <img src="https://raw.githubusercontent.com/uwplasma/LMX/main/docs/_static/readme-blanket-flow.webp" alt="Blanket-flow movie" width="48%">
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/uwplasma/LMX/main/docs/_static/readme-q2d-turbulence.webp" alt="Q2D turbulence movie" width="58%">
</p>

Implemented workflows; quantitative turbulent Q2D-MHDfoam parity and blanket
validation remain open.
[Geometry and fields →](https://lmx.readthedocs.io/en/latest/geometry.html) ·
[External benchmarks →](https://lmx.readthedocs.io/en/latest/external_benchmarks.html)

## Scale on CPUs and GPUs

![Multi-minute strong scaling](https://raw.githubusercontent.com/uwplasma/LMX/main/docs/_static/strong_scaling.webp)

Fixed `256 × 67 × 67` results: CPU (32 updates, three warm/rung) takes
246.691/172.410/147.465 s, reaching **1.431×/1.673×** on 2/4 JAX devices; this is
Docker allocation, not physical-core or steady-state evidence. GPU (96 updates)
takes 159–259 s, reaching 1.626× on two A4000s; foreign contexts make it
shared-host calibration.
[Protocol and results →](https://lmx.readthedocs.io/en/latest/performance.html)

## Quality and citation

Portable gate: **861 tests**, **95.42% line/branch coverage**, **124.1 s** on six
Apple-Silicon workers.
[Testing](https://lmx.readthedocs.io/en/latest/testing.html) ·
[Theory](https://lmx.readthedocs.io/en/latest/theory.html) ·
[Numerics](https://lmx.readthedocs.io/en/latest/numerics.html) ·
[Contributing](https://github.com/uwplasma/LMX/blob/main/CONTRIBUTING.md) ·
[Research assets](https://github.com/uwplasma/LMX/releases/tag/lmx-research-assets-v1)

MIT licensed; cite [CITATION.cff](https://github.com/uwplasma/LMX/blob/main/CITATION.cff).

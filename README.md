# LMX

[![Python](https://img.shields.io/badge/python-3.10--3.13-blue.svg)](https://github.com/uwplasma/LMX/blob/main/pyproject.toml)
[![JAX](https://img.shields.io/badge/JAX-CPU%20%7C%20GPU-orange.svg)](https://docs.jax.dev/)
[![Docs](https://img.shields.io/badge/docs-read%20online-blue.svg)](https://lmx.readthedocs.io/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/uwplasma/LMX/blob/main/LICENSE)

**LMX is a JAX-native code for inductionless liquid-metal MHD on CPUs and
GPUs.** It combines verified duct solvers, conducting walls, imposed 3D fields,
selected differentiable workflows, and research-stage extruded flows.

> Hartmann, Shercliff, and Hunt ducts are verified. Fringe fields, magnetic
> obstacles, Q2D turbulence, and blanket models remain research-stage.

[Documentation](https://lmx.readthedocs.io/) ·
[Quickstart](https://lmx.readthedocs.io/en/latest/getting_started.html) ·
[Examples](https://lmx.readthedocs.io/en/latest/case_cookbook.html) ·
[Validation](https://lmx.readthedocs.io/en/latest/benchmark_matrix.html) ·
[Roadmap](https://github.com/uwplasma/LMX/blob/main/plan.md)

![Analytical and LMX duct-flow profiles](https://raw.githubusercontent.com/uwplasma/LMX/main/docs/_static/analytic_velocity_profiles.webp)

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

TOML and Python inputs support restarts, diagnostics, plotting, and checksummed
artifacts. See the [case cookbook](https://lmx.readthedocs.io/en/latest/case_cookbook.html).
Use `pip install -e .` for the lean numerical core without plotting packages.

## Capabilities

✅ native and documented · ◐ partial/research-stage · — not exposed as the named native workflow

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

¹ [FreeMHD2](https://arxiv.org/abs/2606.18745) is a separate finite-Rm
extension. Sources: [FreeMHD paper](https://doi.org/10.1063/5.0230242) and
[solver](https://github.com/PlasmaControl/FreeMHD/blob/main/MHD_Solvers/solvers/epotMultiRegionInterFoam/epotMultiRegionInterFoam.C),
[NekRS native capabilities](https://nekrs.readthedocs.io/en/latest/),
[research MHD extension](https://doi.org/10.2172/2453867), and
[GPU scaling](https://doi.org/10.1016/j.parco.2022.102982).

## Verified duct MHD

Eight frozen high-Hartmann rows pass, and audited closed-channel FreeMHD
observables pass the 1% finite-grid gate.

![Accepted eight-row Samper Benchmark A validation](https://raw.githubusercontent.com/uwplasma/LMX/main/docs/_static/samper_benchmark_a.webp)

![LMX analytical duct ladder and accepted FreeMHD closed-channel observables](https://raw.githubusercontent.com/uwplasma/LMX/main/docs/_static/freemhd_closed_channel_observable_parity.webp)

<p align="center">
  <img src="https://raw.githubusercontent.com/uwplasma/LMX/main/docs/_static/readme-hunt-startup.webp" alt="Seven-second Hunt and Shercliff startup comparison" width="82%">
</p>

[Validation evidence →](https://lmx.readthedocs.io/en/latest/validation_report.html)

## Real geometries and nonuniform fields

<p align="center">
  <img src="https://raw.githubusercontent.com/uwplasma/LMX/main/docs/_static/readme-geometries.webp" alt="Rectangular, layered, and mapped-pipe geometries" width="48%">
  <img src="https://raw.githubusercontent.com/uwplasma/LMX/main/docs/_static/readme-variable-field.webp" alt="Nonuniform-field response and charge conservation" width="48%">
</p>

![B2 fringe field, pressure, and acceleration diagnostics](https://raw.githubusercontent.com/uwplasma/LMX/main/docs/_static/readme-alex-b2-field-pressure.webp)

B2 passes exact restart plus 1/2/4-CPU and 1/2-GPU schema-6 topology gates.
Shared-norm acceleration is rejected; production parity and scaling promotion
remain open.
[Fringing status →](https://lmx.readthedocs.io/en/latest/fringing.html)

## Follow curved pipes

![Bent-pipe baseline and Dean-vortex literature gate](https://raw.githubusercontent.com/uwplasma/LMX/main/docs/_static/readme-curved-pipes.webp)

Mapped pipes support a low-De inductionless baseline; Dean-vortex physics is
staged behind a literature gate. [Geometry status →](https://lmx.readthedocs.io/en/latest/geometry.html)

## Model conducting multilayer walls

![Li/AlN multilayer pressure and current mesh convergence](https://github.com/uwplasma/LMX/releases/download/lmx-research-assets-v1/readme-li-aln-multilayer-convergence.webp)

Research-stage Li/AlN wall results retain pressure and current mesh-step changes
below 10% at `Ha = 220`; experimental and blanket-level validation remain open.
[Wall models →](https://lmx.readthedocs.io/en/latest/wall_models.html)

## Differentiate selected workflows

![LMX sensitivity and inverse design](https://raw.githubusercontent.com/uwplasma/LMX/main/docs/_static/readme-autodiff.webp)

Promoted objectives pass finite-difference or independent-transpose checks.
[Differentiable workflows →](https://lmx.readthedocs.io/en/latest/autodiff.html)

## Explore research flows

<p align="center">
  <img src="https://raw.githubusercontent.com/uwplasma/LMX/main/docs/_static/readme-magnetic-obstacle.webp" alt="Magnetic-obstacle response" width="48%">
  <img src="https://raw.githubusercontent.com/uwplasma/LMX/main/docs/_static/readme-blanket-flow.webp" alt="Seven-second reduced blanket-flow movie" width="48%">
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/uwplasma/LMX/main/docs/_static/readme-q2d-turbulence.webp" alt="Seven-second nonlinear Q2D vorticity loop" width="58%">
</p>

These visuals demonstrate implemented workflows. Quantitative turbulent
Q2D-MHDfoam parity and blanket validation remain open.
[Geometry and fields →](https://lmx.readthedocs.io/en/latest/geometry.html) ·
[External benchmarks →](https://lmx.readthedocs.io/en/latest/external_benchmarks.html)

## Scale on CPUs and GPUs

![Monitored multi-minute CPU scaling and shared-host GPU calibration](https://raw.githubusercontent.com/uwplasma/LMX/main/docs/_static/strong_scaling.webp)

On a fixed `256 × 67 × 67` grid, three warm 32-update CPU trajectories per rung
take 244.181/172.682/146.768 s and reach **1.414×/1.664×** on two/four JAX
devices. This is Docker CPU-allocation scaling, not physical-core or
steady-state evidence. The 96-update GPU trajectories take 159–259 s and reach
1.626× on two A4000s, but foreign contexts keep that result a shared-host
calibration.
[Protocol and results →](https://lmx.readthedocs.io/en/latest/performance.html)

## Quality and citation

The portable gate records **863 passing tests**, **95.41% combined line/branch
coverage**, and **122.8 s** on six Apple-Silicon workers.
[Testing](https://lmx.readthedocs.io/en/latest/testing.html) ·
[Theory](https://lmx.readthedocs.io/en/latest/theory.html) ·
[Numerics](https://lmx.readthedocs.io/en/latest/numerics.html) ·
[Contributing](https://github.com/uwplasma/LMX/blob/main/CONTRIBUTING.md) ·
[Research assets](https://github.com/uwplasma/LMX/releases/tag/lmx-research-assets-v1)

LMX is MIT licensed. Cite it with
[CITATION.cff](https://github.com/uwplasma/LMX/blob/main/CITATION.cff).

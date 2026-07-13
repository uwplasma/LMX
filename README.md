# LMX

LMX is a JAX-based research code for inductionless liquid-metal
magnetohydrodynamics. Its near-term purpose is narrow and testable: reproduce
trusted duct-flow and fringing-field benchmarks with a differentiable solver
that runs on CPUs and GPUs.

> **Current status:** Hartmann, Shercliff, and Hunt duct workflows are usable;
> the 3D extruded/fringing solver is research-stage. LMX does not yet claim full
> feature-complete FreeMHD parity, production turbulent MHD, heat transfer, free surfaces, or
> full induction. See [Validation status](#validation-status).

[Documentation](https://lmx.readthedocs.io/) ·
[Getting started](docs/getting_started.md) ·
[Theory](docs/theory.md) ·
[Benchmark matrix](docs/benchmark_matrix.md) ·
[Development plan](plan.md)

![Analytical and LMX duct-flow profiles](https://github.com/uwplasma/LMX/releases/download/lmx-research-assets-v1/analytic_velocity_profiles.png)

## Why LMX

- JAX-native array kernels with `jit`, automatic differentiation, and CPU/GPU
  execution.
- Electric-potential inductionless MHD with insulating and conducting walls.
- Straight rectangular and layered ducts plus an experimental extruded 3D
  fringing-field path.
- Reproducible TOML, Python, CLI, restart, diagnostics, and artifact workflows.
- Verification against analytical Hartmann, Shercliff, and Hunt solutions and
  comparison tooling for independent FreeMHD outputs.

LMX favors a small verified physics surface over a large collection of demos.
Capabilities are promoted to the stable surface only after conservation,
convergence, reference-comparison, and differentiability gates pass.

## Install

LMX requires Python 3.10 or newer.

```bash
git clone https://github.com/uwplasma/LMX.git
cd LMX
python -m pip install -e .
```

For development and documentation:

```bash
python -m pip install -e '.[dev,docs]'
```

For the exact dependency set used by CI, install
[uv](https://docs.astral.sh/uv/) and synchronize the committed lockfile:

```bash
uv sync --locked --extra dev --extra docs
```

JAX accelerator installation depends on the platform. Install the appropriate
JAX wheel first when using CUDA or another accelerator backend.

## First run

Run a bundled Hartmann case:

```bash
lmx examples/hartmann_case.toml
```

Or use the Python API:

```python
from lmx.cases import make_hartmann_case
from lmx.solvers import solve_steady

case = make_hartmann_case(ha=20.0, ny=48, nz=48)
solution = solve_steady(case)

print("flow rate:", solution.diagnostics.volumetric_flow_rate_history[-1])
print("final residual:", solution.diagnostics.residual_history[-1])
```

Useful next examples:

```bash
lmx cases/ducts/shercliff_case.toml
lmx cases/ducts/hunt_case.toml
lmx cases/fringing/fringing_rect_case.toml
```

The [case cookbook](docs/case_cookbook.md) explains inputs, outputs, restarts,
custom fields, wall models, and post-processing.

## Physics model

The stable solver surface uses the low-magnetic-Reynolds-number,
electric-potential formulation

```text
div(u) = 0
rho (du/dt + u.grad(u)) = -grad(p) + rho nu laplacian(u) + J x B
J = sigma (-grad(phi) + u x B)
div(J) = 0
```

The imposed magnetic field is not evolved. This is appropriate when the
magnetic Reynolds number is small and excludes full-induction effects. The
[theory](docs/theory.md) and [numerics](docs/numerics.md) pages document
nondimensional groups, boundary conditions, discretization, and limitations.

## Validation status

The benchmark ladder follows the fusion-MHD verification and validation
framework proposed by Smolentsev and collaborators.

| Capability | Evidence today | Status |
|---|---|---|
| Hartmann/Shercliff/Hunt fully developed ducts | analytical profiles, convergence and conservation tests | verified within documented bounded cases |
| Samper Table I high-Ha ducts | strict mesh ladders with flow, layer, solver, current and power gates | all eight Shercliff/Hunt rows pass |
| FreeMHD closed-channel comparison | audited Docker inputs, four-level ladder, analytical and power audits | finest raw 1% gates pass; continuum reference floor is quantified separately |
| 3D laminar fringing field | internal conservation and bounded response tests | research-stage; experimental pressure-drop validation is open |
| Q2D MHD | reduced Hartmann-friction examples | model checks only; turbulent experimental parity is open |
| 3D turbulence / magnetic obstacle | bounded qualitative examples | not validated |
| MHD heat transfer and buoyancy | roadmap only | not implemented |
| Free-surface or full-induction MHD | out of current scope | not implemented |

The July 2026 audit proved the original Docker demos were physical Ha=1000,
not the advertised Ha=20 comparison, so their 6.7-10.4% errors are retained
only as workflow history. Audited Ha=20 cases now run end to end. At 85 x 63,
Shercliff velocity/Lorentz/pressure errors are 0.56%/0.40%/0.27% and Hunt
errors are 0.58%/0.86%/0.26%; conservative current and power gates also pass.
Richardson extrapolation of Hunt Lorentz force is 1.12% against a processed
FreeMHD profile with a quantified analytical error floor, so finite-grid and
continuum evidence remain separate claims. All eight strict high-Ha Table I
rows pass under one solver fingerprint. Finest analytical-flow errors for
Shercliff at Ha=500/5000/10000/15000 are 0.241%/0.369%/0.418%/0.300%; the Hunt
errors are 0.154%/0.325%/0.427%/0.507%. Every row also passes the frozen
refinement, layer, steady-solver, current, and power gates. The combined
machine-readable acceptance record is
`benchmarks/results/benchmark-a-acceptance.json`. Exact definitions, checksums,
compact results, and acceptance criteria live in the
[benchmark matrix](docs/benchmark_matrix.md) and [development plan](plan.md).

## Differentiability

LMX includes differentiable Hartmann and selected extruded response workflows.
Research-grade differentiation requires more than `jax.grad` completing:
gradients must agree with finite differences or an independent adjoint, and
the linear solves must use implicit differentiation with controlled primal and
transpose residuals. Released
[SOLVAX 0.7.0](https://github.com/uwplasma/SOLVAX/releases/tag/v0.7.0) PCG is the
default `auto` backend. Current CPU and RTX A4000 forward, implicit-gradient,
independent-transpose, resource, and end-to-end Hartmann gates pass. The
four-level Ha=20 FreeMHD and all-eight-row high-Ha acceptance record remains the
historical 0.5.1 promotion baseline until the 0.7.0 physics refresh completes.
Select `linear_solver = "cg"` explicitly for the retained native comparison
path.

## Performance and parallelism

LMX can run on JAX CPU and GPU backends. The production ALEX B2 duct solve
accepts `num_devices=N` for named axial sharding on visible JAX devices. Other
extruded paths remain single-device until their operator-specific equivalence
gates pass. LMX does
**not** yet make a general
strong-scaling claim. Performance reports must separate compilation from warm
runtime, use the real solver rather than a presentation-only kernel, include
one-device baselines, report problem size and memory, and validate identical
physics on every device count. Scaling claims require actual shard-placement
and solution-equivalence checks, not merely multiple visible devices.
The current exact-parity `102 x 77 x 77` A4000 checkpoint is `36.96 s` warm on
one GPU and `22.23 s` on two (speedup `1.66`, efficiency `0.831`). This passes
the two-device target; the release-level four-device gate remains open because
the current host has only two GPUs.
For heavy independent validation variants, use the campaign runner's
`--gpu-devices 0,1` mode: it assigns one process per GPU, disables default JAX
memory preallocation, shares a persistent compilation cache, and preserves
restart provenance. Long ALEX runs report every outer iteration and write an
atomic partial restart every eight iterations by default, so `--resume` can
recover a source-matched interrupted run. Normal single-device JAX execution
remains fastest on the Mac CPU because its kernels already use the host cores.

See [Performance and scaling](docs/performance.md) for the current commands and
the acceptance protocol.

## Tests

Fast focused development checks:

```bash
python -m pytest -m unit
```

Complete package test and branch-coverage gate (hard ten-minute budget):

```bash
uv run --locked --extra dev python scripts/run_full_test_suite.py
```

Coverage is necessary but not sufficient. Numerical correctness is enforced
separately with manufactured solutions, conservation identities, mesh/time
convergence, analytical profiles, and independent reference data. Heavy
FreeMHD and scaling campaigns remain explicit workflows because they require
external software or hardware.
The latest complete local gate passes 902 tests with 8 expected external-data
skips and 95.10% branch coverage in 113.3 seconds on six Mac workers.

## Repository policy

- Source, compact reference observables, test fixtures, and scripts belong in
  Git.
- Regenerable movies, field dumps, full meshes, and large benchmark archives
  belong in versioned GitHub/Zenodo releases with checksums and provenance.
- The first 65-file generated-media bundle is published as
  [`lmx-research-assets-v1`](https://github.com/uwplasma/LMX/releases/tag/lmx-research-assets-v1)
  and indexed by `provenance/release-assets.json`.
- Generated documentation media retained in Git will be reduced to a small
  curated set; every retained figure must have a reproducible command.
- New public API is added only when it represents a stable user concept.

## Contributing and citing

The project is under active research development. Before opening a change,
read the [contribution and authorship policy](CONTRIBUTING.md),
[developer guide](docs/developer_guide.md), [testing contract](docs/testing.md),
and [plan](plan.md). Support and vulnerability reports follow
[SUPPORT.md](SUPPORT.md) and [SECURITY.md](SECURITY.md). Benchmark changes must
record equations, parameters, reference provenance, observables, tolerances,
and convergence evidence.

LMX is released under the [MIT License](LICENSE). Citation metadata is in
[CITATION.cff](CITATION.cff); an archival DOI remains a research-release gate.

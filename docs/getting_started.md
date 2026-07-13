# Getting started

LMX supports a single-file TOML workflow and a Python API. Start with a bundled
case, inspect its compact outputs, and increase resolution only after the
workflow is understood.

## Install

```bash
git clone https://github.com/uwplasma/LMX.git
cd LMX
python -m pip install -e .
```

For tests and documentation:

```bash
python -m pip install -e '.[dev,docs]'
```

CI uses the committed lockfile:

```bash
uv sync --locked --extra dev --extra docs
```

LMX supports Python 3.10 and newer. Install the JAX wheel appropriate for your
accelerator before installing LMX when using CUDA.

## First run

```bash
lmx examples/hartmann_case.toml
lmx examples/cases/ducts/shercliff_case.toml
lmx examples/cases/ducts/hunt_case.toml
```

`python -m lmx CASE.toml` is equivalent. Each case writes to its configured
directory under ignored `artifacts/` or `out/` storage.

Hartmann and Shercliff cases use canonical insulating/conducting duct limits.
The Hunt case has conducting side walls and insulating Hartmann walls. These
fully developed cases are the stable first-run surface.

## Python API

```python
from lmx.cases import make_hartmann_case
from lmx.solvers import solve_steady

case = make_hartmann_case(ha=20.0, ny=48, nz=48)
solution = solve_steady(case)

print(solution.diagnostics.volumetric_flow_rate_history[-1])
print(solution.diagnostics.residual_history[-1])
```

Case dataclasses can be changed with `dataclasses.replace`; public constructors
avoid requiring users to know private solver internals.

## CPU or GPU

Choose the backend before Python imports JAX:

```bash
JAX_PLATFORMS=cpu lmx examples/hartmann_case.toml
JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0 lmx examples/cases/ducts/hunt_case.toml
```

Normal CPU kernels already use XLA's host thread pool. Artificially creating
many host devices is intended for sharding tests, not routine speedups.

## Custom fields and fringing flow

Use the bounded custom-field example:

```bash
python examples/variable_field_extruded_demo.py --help
```

Then inspect the reusable TOML cases:

```bash
lmx examples/cases/fringing/fringing_rect_case.toml
lmx examples/cases/fringing/fringing_tabulated_case.toml
```

Fringing and mapped-pipe workflows are research-stage. Their outputs should be
checked for current closure, divergence, solver convergence, and mesh/time
sensitivity before physical interpretation.

## Outputs

Depending on `[output]`, a run can write live logs, a JSON summary, restartable
NPZ state, CSV profiles, VTK files, and compressed plots. Keep generated results
outside Git. See [Case cookbook](case_cookbook.md) for restarts and validation,
and [Input reference](input_reference.md) for every TOML field.

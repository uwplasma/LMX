# Getting Started

LMX is designed to be usable from either a single TOML input file or a Python
driver script. The intended workflow is:

1. define a geometry, material regions, and magnetic field
2. choose a solver family and time controls
3. run from the CLI or Python
4. inspect logs, JSON summaries, `.npz` restart bundles, VTK output, and plots

## Installation

Minimal install:

```bash
git clone https://github.com/uwplasma/LMX
cd LMX
python -m pip install -e .
```

Development install:

```bash
git clone https://github.com/uwplasma/LMX
cd LMX
python -m pip install -e '.[dev,plotting,docs,extras]'
```

LMX supports Python `3.10+`. On Python `3.10`, TOML parsing falls back
automatically to `tomli`.

## Fastest first run

Run one of the shipped cases:

```bash
lmx examples/hartmann_case.toml
lmx examples/shercliff_case.toml
lmx examples/hunt_case.toml
```

The equivalent module entrypoint also works:

```bash
python -m lmx examples/hartmann_case.toml
```

## Run from Python

```python
from lmx.cases import make_hartmann_case
from lmx.config import LoggingSpec
from lmx.runtime_logging import StreamingSolverLogger
from lmx.solvers import solve_steady

case = make_hartmann_case(ha=20.0, ny=48, nz=48)
logger = StreamingSolverLogger(LoggingSpec.from_user_controls(verbose=True, verbosity="debug"))
solution = solve_steady(case, logger=logger)
```

## Control the backend

LMX inherits the active JAX backend from the shell:

```bash
JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 lmx examples/hartmann_case.toml
XLA_FLAGS=--xla_force_host_platform_device_count=8 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 lmx examples/hartmann_case.toml
JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0 lmx examples/hunt_case.toml
```

## Use custom geometry and magnetic fields from Python

The shipped `examples/variable_field_geometry_demo.py` shows how to:

- build a `CaseSpec` directly
- define a custom `GeometrySpec`
- attach an analytic magnetic field callback through `MagneticFieldSpec(kind="analytic", fn=...)`
- preview the geometry before solving
- run the solver and write publication-style plots

Run it with:

```bash
python examples/variable_field_geometry_demo.py --output artifacts/examples/variable_field_geometry
```

## Output products

A standard run can produce:

- live solver logs
- a JSON summary
- restartable `.npz` state bundles
- VTK files for ParaView
- CSV profiles
- publication-style `PNG` / `PDF` plots

The precise output selection is controlled by `[output]` in the TOML input or
the corresponding `OutputSpec` in Python.

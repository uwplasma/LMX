# LMX

[![PyPI](https://img.shields.io/pypi/v/lmx.svg)](https://pypi.org/project/lmx/)
[![Python](https://img.shields.io/pypi/pyversions/lmx.svg)](https://pypi.org/project/lmx/)
[![CI](https://img.shields.io/github/actions/workflow/status/uwplasma/LMX/ci.yml?branch=main&label=ci)](https://github.com/uwplasma/LMX/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/readthedocs/lmx/latest?label=docs)](https://lmx.readthedocs.io/)
[![License](https://img.shields.io/github/license/uwplasma/LMX)](LICENSE)

LMX solves inductionless liquid-metal magnetohydrodynamics in ducts with JAX.
It covers fully developed Hartmann, Shercliff, and Hunt flows and three-dimensional
extruded ducts and pipes in spatially varying magnetic fields. LMX owns the MHD
models, boundary conditions, coupling, diagnostics, and validation;
[SOLVAX](https://github.com/uwplasma/SOLVAX) supplies reusable linear and
fixed-point algorithms.

![Analytical duct profiles](docs/_static/analytic_velocity_profiles.webp)

## Install

```console
python -m pip install lmx
lmx --help
```

LMX supports Python 3.10–3.13. JAX selects the CPU by default; install the
appropriate accelerator wheel using the
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html).
Plots are optional: `python -m pip install "lmx[visualization]"`.

## Solve a duct

```python
import lmx

case = lmx.make_hartmann_case(ha=20.0, ny=48, nz=48)
result = lmx.solve_steady(case)

print(result.converged, result.status)
print(result.state.residual)
```

The case object contains the geometry, materials, imposed field, boundary
conditions, time stepping, solver controls, and output policy. The result
contains the final fields, convergence status, iteration counts, and physical
diagnostics. The same workflow is available from TOML:

```console
lmx examples/hartmann_case.toml --plots
```

## Solve a three-dimensional fringe

```python
from lmx.fringing import (
    build_square_duct_extruded_problem,
    solve_extruded_inductionless,
)

problem = build_square_duct_extruded_problem(
    ha_peak=20.0,
    width=2.0,
    height=2.0,
    length=6.0,
    nx_stations=21,
    ny=24,
    nz=24,
)
result = solve_extruded_inductionless(problem)

print(result.converged, result.validation.max_charge_balance_residual)
```

The 3-D formulation solves electric-potential/current closure, Lorentz force,
momentum transport, and face-flux pressure projection on rectangular,
layered-duct, straight-pipe, and mapped bent-pipe meshes. Analytic and tabulated
vector fields use the same problem interface.

![Three-dimensional fringing-field result](docs/_static/fringing_solver_family.webp)

## Capabilities and evidence

| Capability | Interface | Evidence |
|---|---|---|
| Hartmann, Shercliff, and Hunt flow | `lmx.make_*_case`, `solve_steady` | analytical profiles, conservation, power balance, mesh convergence |
| Conducting and insulating wall layers | `WallLayer`, layered mesh builders | interface-current and layer-resolution gates |
| 3-D rectangular fringing fields | `lmx.fringing` | manufactured operators, projection, restart, Benchmark B2 |
| 3-D pipe fringing fields | `lmx.fringing` | mapped operators, current closure, fixed-flow and Benchmark B1 gates |
| FreeMHD comparison | `lmx.freemhd`, validation scripts | pinned case contracts, native-output observers, executable Docker workflow |
| Differentiable Hartmann objectives | `lmx.autodiff` | finite-difference, JVP, and VJP checks |
| Constant, analytic, and tabulated fields | `lmx.field_models` | divergence and interpolation tests |

Validation status and tolerances are stated in the
[validation guide](https://lmx.readthedocs.io/en/latest/validation/index.html).
Internal diagnostics are not presented as external validation.

![LMX and FreeMHD observable comparison](docs/_static/freemhd_closed_channel_observable_parity.webp)

## Documentation

- [Install and run](https://lmx.readthedocs.io/en/latest/getting_started/install.html)
- [First 2-D solve](https://lmx.readthedocs.io/en/latest/getting_started/first_run.html)
- [3-D fringing tutorial](https://lmx.readthedocs.io/en/latest/tutorials/fringing.html)
- [Equations and assumptions](https://lmx.readthedocs.io/en/latest/physics/equations.html)
- [Numerical methods and SOLVAX boundary](https://lmx.readthedocs.io/en/latest/physics/numerics.html)
- [Python API](https://lmx.readthedocs.io/en/latest/reference/api.html)
- [CLI and TOML schema](https://lmx.readthedocs.io/en/latest/reference/cli.html)

Runnable scripts in [`examples/`](examples/) are deliberately small and write
their artifacts under an ignored `artifacts/` directory.

## Development

```console
git clone https://github.com/uwplasma/LMX.git
cd LMX
python -m pip install -e ".[dev,docs]"
python scripts/run_full_test_suite.py
python -m sphinx -W -b html docs docs/_build/html
```

Tests require at least 95% combined line/branch coverage. A release also gates
package size, lazy import behavior, distribution contents, analytical physics,
3-D conservation, and the pinned external-validation contract.

## Cite

Use the version or commit that produced the result. Citation metadata is in
[`CITATION.cff`](CITATION.cff).

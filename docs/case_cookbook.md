# Case Cookbook

## Hartmann

### CLI

```bash
python -m lmx.cli run hartmann --ha 20 --output ./out/hartmann
lmx examples/hartmann_case.toml
python -m lmx examples/hartmann_case.toml
```

### Python

```python
from lmx.cases import make_hartmann_case
from lmx.solvers import solve_steady

case = make_hartmann_case(ha=20.0, ny=48, nz=48)
solution = solve_steady(case)
```

The public Python driver surface is intended to be modified directly. Typical
research edits are:

- increase `ny` and `nz` for refinement studies
- replace the magnetic field with an analytic callable
- change the logging verbosity through `LoggingSpec`
- change the output tree without touching solver internals

## Shercliff

### CLI

```bash
python -m lmx.cli run shercliff --ha 20 --output ./out/shercliff
lmx examples/shercliff_case.toml
```

## Hunt

### CLI

```bash
python -m lmx.cli run hunt --ha 20 --output ./out/hunt
lmx examples/hunt_case.toml
```

### Python

```python
from dataclasses import replace

from lmx.cases import make_hunt_case
from lmx.solvers import solve_steady

case = make_hunt_case(ha=20.0, ny=40, nz=32, wall_cells=2)
case = replace(case, forcing=0.8)
solution = solve_steady(case)
```

This is the intended entry point for:

- changing wall conductance ratios
- changing side-wall or Hartmann-wall resolution
- staging custom benchmark studies without dropping into private solver code

## Variable fields and custom geometries

```bash
python examples/variable_field_geometry_demo.py --output ./artifacts/examples/variable_field_geometry
```

That example is the recommended starting point when a user wants to:

- define an analytic spatially varying magnetic field in Python
- alter duct width, height, or wall layout from a benchmark constructor
- preview the geometry and the material map before running
- emit a short steady solve and standard overview plots in the same directory

## Validation and convergence

```bash
python -m lmx.cli validate hartmann --ha 20 --output ./out/validation/hartmann
python scripts/run_validation_suite.py --output ./artifacts/validation
python scripts/run_convergence_suite.py --output ./artifacts/convergence --cases hartmann,shercliff,hunt --ha 20 --resolutions 16,32,48
python scripts/run_time_convergence_suite.py --output ./artifacts/time_convergence --cases hartmann,shercliff,hunt --ha 20 --resolution 32 --dts 0.002,0.001,0.0005
```

## Plotting and movies

```bash
python examples/theory_meeting_demo.py --output ./artifacts/examples/theory_meeting_demo
python examples/plot_npz_results.py --npz ./artifacts/examples/theory_meeting_demo/shercliff/shercliff_ha20_results.npz --output ./artifacts/examples/theory_meeting_demo/shercliff/replot
python examples/geometry_preview_demo.py --output ./artifacts/examples/geometry_preview
python examples/geometry_preview_demo.py --with-post-run --post-case hartmann --output ./artifacts/examples/geometry_preview_full
```

## Geometry and mesh preview

```bash
python examples/geometry_preview_demo.py --output ./artifacts/examples/geometry_preview
python examples/geometry_preview_demo.py --with-post-run --post-case hunt --output ./artifacts/examples/geometry_preview_hunt
```

That example previews:

- a uniform rectangular Hartmann duct
- a layered Hunt duct with explicit wall regions
- a mapped pipe O-grid

The default path is preview-only so it stays fast enough for routine use.
Enable `--with-post-run` when you want the same output tree to include a short
Hartmann or Hunt solve and matching overview plots.

## Parallel backend selection

```bash
JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 lmx examples/hartmann_case.toml
JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0 lmx examples/hunt_case.toml
python examples/strong_scaling_demo.py --remote-host office --output ./artifacts/examples/strong_scaling
python examples/strong_scaling_demo.py --benchmark-kind extruded_solve --profile --output ./artifacts/examples/extruded_solve_scaling
```

Routine CLI runs inherit the active JAX backend from the shell, while
`examples/strong_scaling_demo.py` is the main path for explicit
CPU and GPU scaling studies. Use `--benchmark-kind extruded_solve` for the
actual rectangular `extruded_inductionless` projection path, and the default
`extruded3d` mode for the explicitly sharded dense-operator scaling panel.

## Restart

```bash
lmx examples/hartmann_case.toml
lmx examples/hartmann_restart_case.toml
```

The second run resumes from the first run’s `.npz` state and extends the
simulation while appending diagnostics.

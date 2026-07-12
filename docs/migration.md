# API migration

## Root namespace introduced for the slim-core release

LMX now limits `lmx.__all__` to 30 deliberate convenience exports covering
case construction, fully developed solves, core meshes, nondimensional groups,
wall conductance, and the bundled straight-duct references. This keeps
`from lmx import *` small and makes the supported top-level surface legible.

No existing attribute has been removed in this release. Former root exports
remain lazily available for one deprecation cycle and emit
`DeprecationWarning` with their destination module. New code should import
advanced capabilities directly from the owning submodule:

```python
# Before (supported temporarily)
from lmx import solve_extruded_inductionless, write_case_overview_plots

# After
from lmx.fringing import solve_extruded_inductionless
from lmx.plotting import write_case_overview_plots
```

The same rule applies to research-stage families:

| Capability | Import from |
|---|---|
| Differentiation and inverse design | `lmx.autodiff` |
| Extruded/fringing solvers | `lmx.fringing` |
| Q2D models | `lmx.q2d` |
| Plotting | `lmx.plotting` |
| External-code adapters | `lmx.external_validation` or `lmx.freemhd` |
| Blanket and wall-stack studies | `lmx.blanket_flow`, `lmx.wall_study` |

The compatibility facade will be removed only after one documented release.
Applications can identify migrations during testing with:

```bash
python -W error::DeprecationWarning your_driver.py
```

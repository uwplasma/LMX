# API migration

## Slim package root

LMX now limits `lmx.__all__` to 30 deliberate convenience exports covering
case construction, fully developed solves, core meshes, nondimensional groups,
wall conductance, and the bundled straight-duct references. This keeps
`from lmx import *` small and makes the supported top-level surface legible.

The one-release compatibility period for former root aliases has ended.
Advanced capabilities now import directly from their owning submodule:

```python
# Before
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
| External-code observables and audits | `lmx.external_validation` or `lmx.freemhd` |
| Blanket studies | `lmx.blanket_flow` |
| Wall-stack physics | `lmx.wall_models`, `lmx.mesh` |

The obsolete, undocumented `build_case_from_freemhd_reference` policy adapter
and its adapter-only inference wrappers were removed after their repository
callers disappeared. Use canonical LMX builders plus independently materialized
and audited external inputs; raw FreeMHD dictionary parsers remain available in
`lmx.freemhd`.

The obsolete station-wise approximations `clone_case_with_field`,
`run_fringing_station_sweep`, and `run_extruded_inductionless_slice` were also
removed. Use `solve_extruded_inductionless`, which preserves axial coupling and
supports every documented extruded geometry. Its fallback-only `solver=`
injection argument was removed with those approximations.

The case-specific `lmx.wall_study` report wrappers were retired after their
repository callers disappeared. The editable
`examples/li_aln_wall_stack_example.py` now shows the full material, mesh,
solver, validation, and plotting workflow using the reusable wall and mesh
APIs directly.

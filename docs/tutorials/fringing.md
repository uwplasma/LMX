# Three-dimensional fringing fields

An extruded problem combines a `CaseSpec` with an axial imposed-field profile.
The convenience builder supplies a rectangular duct with a smooth entry and
exit fringe:

```python
import lmx
from lmx.fringing import build_square_duct_extruded_problem

problem = build_square_duct_extruded_problem(
    ha_peak=20,
    width=2,
    height=2,
    length=6,
    nx_stations=21,
    ny=24,
    nz=24,
    entry_center=1.5,
    exit_center=4.5,
    transition_width=0.35,
)
result = lmx.solve(problem)
```

The returned fields contain velocity, pressure, potential, current, Lorentz
force, face fluxes, station coordinates, and the imposed field. Always inspect
termination and conservation together:

```python
gate = result.validation
print(result.converged, result.status)
print(gate.max_charge_balance_residual)
print(gate.max_divergence_residual)
print(gate.net_boundary_current_residual)
```

Use `build_layered_duct_extruded_problem` for explicit wall regions and
`build_pipe_ogrid_extruded_problem` for a straight conducting pipe. Bent pipes
use the same mapped pipe operators through `build_bent_pipe_extruded_problem`.
`build_extruded_problem_from_case` is the general entry point when the complete
case is already available.

The axial, cross-section, and wall meshes are refined independently. A reported
fringing result should demonstrate stable primary observables under all three
refinements and under tighter linear and coupling tolerances. The
[FreeMHD guide](../validation/freemhd.md) describes the external comparison.

Run `python examples/fringing_benchmark_demo.py` for a bounded rectangular
diagnostic or `python examples/variable_field_extruded_demo.py` for a custom
divergence-free vector field.

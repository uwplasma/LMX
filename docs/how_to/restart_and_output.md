# Restart and write output

Two-dimensional and extruded solves use separate typed restart bundles because
their state and face-flux contracts differ.

```python
from lmx.io import write_restart_npz, load_restart_bundle, validate_restart_bundle

write_restart_npz(result, case, "restart.npz")
restart = load_restart_bundle("restart.npz")
validate_restart_bundle(restart, case=case)
```

For a three-dimensional result, use `write_extruded_restart_npz`,
`load_extruded_restart_bundle`, and `validate_extruded_restart_bundle`. The
validation rejects mismatched geometry, mesh shape, material arrays, or field
metadata before a restart enters the solver.

`write_solution_outputs` and `write_extruded_solution_outputs` honor the
case's `OutputSpec`. Prefer NPZ plus JSON for repeatable studies. Enable VTK
only when a downstream visualization tool needs it, and keep generated output
outside the repository.

Full iteration and station histories cost memory. LMX stores compact diagnostic
histories by default; use output strides and checkpoint intervals deliberately
for long 3-D runs.

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

Full iteration histories cost memory. `OutputSpec(history_stride=0)` keeps the
terminal sample, which is the default. Set a positive stride to retain the
first sample, every requested interval, and the terminal sample; use `1` only
when every iteration is needed. Positive-stride restart segments preserve
retained samples and add samples from the resumed segment; stride `0` keeps
only the latest terminal. Restart state and compact diagnostics are independent,
while benchmark builders retain every iteration required by their published
evidence contracts. `write_stride` controls station-field output and
`checkpoint_interval` controls in-progress 3-D restart callbacks.

# Fringing-Field Research Slice

LMX `1.0` does not yet ship the full `extruded_inductionless` solver family.
What it does ship now is the first explicit solver-family entry point that the
next paper phase will build on:

- a smooth axial fringing-field profile generator in `lmx/fringing.py`
- a stationwise sweep driver that reuses the fully developed solver as a cheap
  research scaffold
- an `ExtrudedInductionlessProblem -> ExtrudedInductionlessSolution` workflow
  that wraps the stacked slice as an actual public solver entry point
- a stacked axial field-bundle builder that assembles `u(x, y, z)`,
  `phi(x, y, z)`, current, Lorentz, and charge-balance histories from those
  station solves
- an explicit validation summary for that extruded slice
- a publication-style example in `examples/fringing_benchmark_demo.py`

This is explicit by design. The current scaffold is not a replacement for a
true 3D pressure-velocity solve. It is the bridge that lets users stage field
profiles, benchmark manifests, and axial response figures while the first
`extruded_inductionless` solver slice is being built.

## Run the scaffold

```bash
python examples/fringing_benchmark_demo.py \
  --output artifacts/examples/fringing_benchmark
```

The example writes:

- `fringing_benchmark_summary.json`
- `fringing_benchmark.png`
- `fringing_benchmark.pdf`
- an `extruded_bundle` section in the JSON summary with axial field-bundle shape
  and charge-balance histories
- a `validation` section with residual and field/response consistency metrics

## What the example shows

- a smooth entrance/exit fringing profile along the duct axis
- the stationwise cross-sectional mean velocity response
- the stationwise current-scaled pressure surrogate
- contour views of the stacked velocity bundle in `x-y` and `x-z`
- the stationwise charge-balance residual along the fringing region

These are the quantities we need immediately for benchmark design and for
planning the first `extruded_inductionless` acceptance set.

## Source map

- `lmx/fringing.py`
  - fringing-profile construction, stationwise sweep utilities, the extruded
    slice solver entry point, validation helpers, and stacked axial field bundles
- `examples/fringing_benchmark_demo.py`
  - user-facing fringing benchmark slice with publication-style plots
- `docs/benchmark_matrix.md`
  - benchmark targets that this scaffold is preparing

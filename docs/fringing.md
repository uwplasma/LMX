# Fringing-Field Research Slice

LMX `1.0` does not yet ship the full production `extruded_inductionless`
solver family. What it does ship now is the first explicit 3D solver-family
entry point that the next paper phase can build on:

- a smooth axial fringing-field profile generator in `lmx/fringing.py`
- a stationwise sweep driver that reuses the fully developed solver as a cheap
  research scaffold
- an `ExtrudedInductionlessProblem -> ExtrudedInductionlessSolution` workflow
  that now runs a true low-Re rectangular-duct `u, v, w, p, phi` projection
  slice and still retains the stacked-station path as a fallback scaffold for
  layered ducts
- a stacked axial field-bundle builder that exposes `u(x, y, z)`,
  `v(x, y, z)`, `w(x, y, z)`, `p(x, y, z)`, `phi(x, y, z)`, current,
  Lorentz, and charge-balance histories
- an explicit validation summary for that extruded slice
- a publication-style example in `examples/fringing_benchmark_demo.py`

This is explicit by design. The current rectangular-duct slice is a real 3D
pressure-velocity-potential iteration, but it is still a research slice rather
than the final production family. It is the bridge that lets users stage field
profiles, benchmark manifests, and axial response figures while the broader
`extruded_inductionless` solver family is hardened.

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
- a first true 3D pressure field `p(x, y, z)` inside the research slice

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

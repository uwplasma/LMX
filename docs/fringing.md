# Fringing-Field Scaffold

LMX `1.0` does not yet ship the full `extruded_inductionless` solver family.
What it does ship now is the executable scaffold that the next paper phase will
use to get there:

- a smooth axial fringing-field profile generator in `lmx/fringing.py`
- a stationwise sweep driver that reuses the fully developed solver as a cheap
  research scaffold
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

## What the example shows

- a smooth entrance/exit fringing profile along the duct axis
- the stationwise cross-sectional mean velocity response
- the stationwise current-scaled pressure surrogate

These are the quantities we need immediately for benchmark design and for
planning the first `extruded_inductionless` acceptance set.

## Source map

- `lmx/fringing.py`
  - fringing-profile construction and stationwise sweep utilities
- `examples/fringing_benchmark_demo.py`
  - user-facing fringing benchmark scaffold
- `docs/benchmark_matrix.md`
  - benchmark targets that this scaffold is preparing

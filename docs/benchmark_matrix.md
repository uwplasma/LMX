# Benchmark Matrix

LMX is being reset around a research-grade inductionless benchmark ladder.

## Mandatory for the first paper

### Benchmark A: fully developed laminar duct flow

- A1: insulating duct / Shercliff-type
- A2: conducting Hartmann walls / Hunt-type

Current solver target:

- `solver.kind = "fully_developed_inductionless"`
- geometries:
  - `rect_duct`
  - `layered_duct`

Primary validation observables:

- velocity profiles
- electric-potential profiles
- current-density profiles
- flow-rate / mean-velocity observables
- Lorentz-force observables on matched cuts

### Benchmark B: laminar fringing-field flow

- B1: conducting pipe in a fringing magnetic field
- B2: conducting square duct in a fringing magnetic field

Current implementation status:

- mesh scaffolding exists for mapped pipes
- solver support is not complete yet
- these cases will land under `solver.kind = "extruded_inductionless"`

## Staged but deferred beyond the first paper

- Benchmark C: Q2D turbulent duct flow
- Benchmark D: turbulent duct flow / magnetic obstacle
- Benchmark E: natural convection / heat transfer
- sudden expansion
- HCLL / blanket mock-up / Madarame-type coupled-duct effects

These are in scope for benchmark manifests, data loaders, and skipped tests,
but not for the first inductionless release.

# Benchmark Matrix

This page defines the benchmark ladder for the first LMX paper and `1.0`
release.

## Mandatory now: Benchmark A

### A1. Hartmann / insulating-duct style validation

- solver family: `fully_developed_inductionless`
- geometry: `rect_duct`
- observables:
  - velocity profiles
  - potential profiles
  - flow-rate and pressure-gradient surrogates

### A2. Shercliff and Hunt conducting/insulating wall validation

- solver family: `fully_developed_inductionless`
- geometries:
  - `rect_duct`
  - `layered_duct`
- observables:
  - matched `y` and `z` profiles
  - current-density profiles
  - Lorentz-force profiles
  - integral flow-rate and conservation diagnostics

## Mandatory next: Benchmark B

These are the first nontrivial 3D inductionless targets from the benchmark
ladder summarized by [Samper et al.](https://www.scipedia.com/wd/images/b/b8/Draft_Samper_360028846_6045_art042.pdf).

### B1. Conducting pipe in a fringing magnetic field

- planned solver family: `extruded_inductionless`
- required geometry support: mapped pipe O-grid
- required observables:
  - pressure drop
  - velocity distortion
  - electric-potential redistribution

### B2. Conducting square duct in a fringing magnetic field

- planned solver family: `extruded_inductionless`
- required observables:
  - cross-sectional velocity structure
  - current-density redistribution
  - Lorentz-force localization

## Staged but deferred

- Benchmark C: Q2D turbulent duct flow
- Benchmark D: turbulent duct flow / magnetic obstacle
- Benchmark E: natural convection / heat transfer
- sudden expansion
- blanket mock-up / coupled-duct effects

These remain part of the research roadmap, but not the `1.0` solver promise.

## Additional benchmark targets for the next publication cycle

The broader validation ladder used in recent inductionless liquid-metal MHD
solver papers suggests the following next additions after the current duct and
fringing set:

- closed pipe in a fringing magnetic field
  - observables:
    wall potential, pressure redistribution, and distorted axial velocity
- free-surface dam-break or sloshing benchmark
  - observables:
    front position, free-surface shape, and magnetic damping of the transient
- open-channel fringing-field benchmark
  - observables:
    free-surface deformation, recirculation, and current closure near the field
    ramp
- current-driven slotted-channel benchmark
  - observables:
    wall-current closure, jet structure, and electric-potential redistribution

Those cases extend the current validation ladder in the same direction as the
existing duct and fringing workflows: from fully developed 2D ducts to 3D
fringing response, then to free-surface and current-driven configurations.

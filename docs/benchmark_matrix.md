# Benchmark Matrix

This page defines the benchmark ladder for the first LMX paper and `1.0`
release.

## Mandatory now: Benchmark A

### A1. Hartmann / insulating-duct style validation

- solver family: `fully_developed_inductionless`
- geometry: `rect_duct`
- literature target:
  - Samper et al. Table I insulating square-duct cases at `Ha = 500, 5000,
    10000, 15000`
  - compare the dimensionless flow-rate integral `Q̃`, not only local cuts
- observables:
  - velocity profiles
  - potential profiles
  - flow-rate and pressure-gradient surrogates
  - dimensionless flow-rate integral under mesh refinement

### A2. Shercliff and Hunt conducting/insulating wall validation

- solver family: `fully_developed_inductionless`
- geometries:
  - `rect_duct`
  - `layered_duct`
- literature target:
  - Samper et al. Table I conducting-wall square-duct cases at `Ha = 500,
    5000, 10000, 15000` with Hartmann-wall conductance ratio `cw = 0.01`
- observables:
  - matched `y` and `z` profiles
  - current-density profiles
  - Lorentz-force profiles
  - integral flow-rate and conservation diagnostics
  - side-layer / jet structure in the conducting-wall case

## Mandatory next: Benchmark B

These are the first nontrivial 3D inductionless targets from the benchmark
ladder summarized by [Samper et al.](https://www.scipedia.com/wd/images/b/b8/Draft_Samper_360028846_6045_art042.pdf).

### B1. Conducting pipe in a fringing magnetic field

- solver family: `extruded_inductionless`
- required geometry support: mapped pipe O-grid
- literature target:
  - Samper et al. Table II pipe case: `Ha ≈ 6600`, `N ≈ 10700`, `cw ≈ 0.027`
- required observables:
  - dimensionless pressure drop between the documented upstream/downstream taps
  - axial velocity distortion through the magnetic-field ramp
  - electric-potential redistribution on the wall and across the pipe section
  - mesh convergence of pressure drop and current-closure metrics

### B2. Conducting square duct in a fringing magnetic field

- solver family: `extruded_inductionless`
- literature target:
  - Samper et al. Table II square-duct case: `Ha ≈ 2900`, `N ≈ 540`, `cw ≈ 0.07`
- required observables:
  - dimensionless pressure drop between the documented upstream/downstream taps
  - cross-sectional velocity distortion through the fringing region
  - current-density redistribution and Lorentz-force localization
  - mesh convergence of pressure drop, current closure, and throughput recovery

## Validation gates for Benchmarks A and B

The current codebase should be judged against a fixed set of physics and
quality gates rather than only against visual agreement:

- profile agreement
  - normalized velocity/potential/profile errors on matched cuts
- integral agreement
  - flow rate, pressure-span surrogate, axial-current span, and Lorentz-power
    trends under mesh refinement
- literature observables
  - Benchmark A: dimensionless flow-rate integral `Q̃` against the analytical
    values tabulated by Samper et al.
  - Benchmark B: dimensionless pressure drop between the documented taps and
    matched velocity/potential cuts at the reference axial stations
- conservation
  - `div J`
  - charge-balance residual
  - interface-current residual
  - wall-current leakage
  - net boundary-current residual
- fringing-response physics
  - throughput constancy outside the field ramp
  - negative field/mean-velocity correlation through the ramp
  - pressure growth in the magnetized zone and recovery downstream
- quality gates
  - restart continuation equivalence
  - stable CLI/TOML and Python-driver workflows
  - machine-readable JSON/CSV outputs
  - strict docs build
  - fast routine test lane under five minutes

## Combined validation exercise

The current executable path for the full Benchmark A/B validation sweep is:

```bash
python scripts/run_full_validation_exercise.py \
  --output artifacts/validation/full_validation_exercise \
  --ha-values 10,20 \
  --resolution 12 \
  --fringing-resolutions 8,12 \
  --skip-paraview \
  --reference-root ./references/ClosedChannel \
  --write-plot
```

That workflow produces:

- Benchmark A case directories with field/profile artifacts
- Benchmark B fringing summaries and optional fringing-resolution plot
- one combined JSON summary
- one combined CSV table
- one combined Markdown gate report

When run from the source tree, `--reference-root` can be omitted and the
workflow will use the bundled closed-channel reference dataset under
`external/FreeMHDPaperAllFigures/FreeMHDPaperAllFigures/ClosedChannel`.

For the fringing-only quantitative summary, use:

```bash
python scripts/run_benchmark_b_quantitative.py \
  --output artifacts/validation/benchmark_b_quantitative \
  --ha-peak 20 \
  --duct-ny 20 \
  --duct-nz 20 \
  --pipe-nr 20 \
  --pipe-ntheta 80 \
  --nx-stations 21 \
  --max-steps 24 \
  --coupling-iterations 12 \
  --potential-iterations 80
```

That workflow writes one JSON summary, one CSV table, one Markdown table, and
one four-panel figure over charge balance, throughput span, axial-current
span, and pressure-span range.

That internal summary is the low-cost quantitative gate. The literature-anchored
Benchmark B closure still needs a second layer:

- B1 pipe:
  - dimensionless pressure-drop comparison against the documented tap spacing
  - shared-normalization profile comparisons on center and offset cuts
- B2 square duct:
  - dimensionless pressure-drop comparison through the field ramp
  - matched cross-sectional velocity and potential cuts at the same axial
    stations used in the reference data

Current dense-slice summary at `Ha = 20`, `20×20×21` for ducts and
`20×80×21` for the mapped pipe:

- `rect_duct`
  - `max_charge_balance_residual ≈ 5.48e-6`
  - `volumetric_flow_rate_span ≈ 1.08e-3`
  - `axial_current_span ≈ 5.14e-8`
  - `pressure_span_range ≈ 8.10e-1`
- `layered_duct`
  - `max_charge_balance_residual ≈ 2.76e-5`
  - `volumetric_flow_rate_span ≈ 2.18e-3`
  - `axial_current_span ≈ 6.66e-1`
  - `pressure_span_range ≈ 3.40e1`
- `pipe_ogrid`
  - `max_charge_balance_residual ≈ 7.72e-3`
  - `volumetric_flow_rate_span ≈ 1.83e-9`
  - `axial_current_span ≈ 9.09e-10`
  - `pressure_span_range ≈ 5.51e-4`
On the current tree, rectangular and layered slices are quantitatively
well-behaved on these settings. The next dense Benchmark B hardening target is
the mapped-pipe external parity lane rather than the internal rectangular
charge-balance lane.

The current mapped-pipe external comparison remains the main Benchmark B
external gap. On the latest bounded pipe-reference comparison:

- center cut: `L2 ≈ 1.63e-1`, `L∞ ≈ 9.82e-1`
- negative offset cut: `L2 ≈ 9.93e-1`, `L∞ ≈ 1.00`
- positive offset cut: `L2 ≈ 9.93e-1`, `L∞ ≈ 1.00`

That is enough to say the comparison is quantitative, but not enough to call it
parity closure.

The underlying reason is now clearer from the bundled FreeMHD reference set:
the mapped-pipe comparison files correspond to the Bühler fringing-pipe case
at `Ha = 2000`, `Re = 20000`, while the current LMX `pipe_ogrid`
`extruded_inductionless` lane is still a low-Re inductionless research slice.
That means the present pipe comparison is still useful for qualitative shape
and sign checks, but quantitative parity will require a higher-inertia pipe
solver path rather than only denser sampling or plot cleanup.

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

## External reference datasets already staged in-tree

The repository already carries several external figure/data bundles that can be
used to stage the next validation campaigns:

- `external/FreeMHDPaperAllFigures/FreeMHDPaperAllFigures/ClosedChannel`
  - closed-channel Hartmann, Shercliff, and Hunt profile references for the
    Benchmark A family
- `external/FreeMHDPaperAllFigures/FreeMHDPaperAllFigures/FringingBPipe`
  - conducting-pipe fringing profiles for the Benchmark B1 family
- `external/FreeMHDPaperAllFigures/FreeMHDPaperAllFigures/DamBreak`
  - free-surface transient references for a post-Benchmark-B liquid-metal
    validation lane
- `external/FreeMHDPaperAllFigures/FreeMHDPaperAllFigures/LMX-U`
  - open-channel liquid-metal datasets relevant to free-surface and outlet
    modeling
- `external/FreeMHDPaperAllFigures/FreeMHDPaperAllFigures/Divertorlets`
  - current-driven / complex-geometry reference data for later engineering
    validation

## Planned geometry and field extensions

Two capability lanes should be treated as explicit benchmark projects rather
than as ad hoc feature work:

- bent-pipe geometry
  - baseline: no-field Dean-flow verification
  - inductionless extension: uniform-field bent pipe
  - nonuniform-field extension: fringing-field bent pipe
  - required observables:
    pressure span, secondary-flow structure, current closure, and mesh
    convergence with curvature
- spatially varying magnetic fields
  - baseline: manufactured divergence-free field verification
  - recovery test: reproduce the current fringing benchmarks through the
    generic field-loading path
  - extension: tabulated or analytic 3D fields for ducts and pipes
  - required observables:
    pressure redistribution, Lorentz-force localization, throughput change, and
    charge/current closure under mesh refinement

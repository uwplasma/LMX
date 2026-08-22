# Validation

LMX separates numerical verification, physics validation, and external-code
comparison. A test passes only its stated claim; finite output alone is never a
validation result.

| Model | Required evidence | Current claim |
|---|---|---|
| Hartmann duct | analytical profile, charge closure, power balance, refinement | validated within documented mesh/tolerance gates |
| Shercliff and Hunt ducts | analytical closed-channel profiles, symmetry, wall/interface current | validated within documented mesh/tolerance gates |
| High-$Ha$ fully developed flow | layer resolution, Richardson trend, integral balances | bounded accepted campaign cases |
| Rectangular 3-D fringe | manufactured operators, projection, restart, refinement, FreeMHD B2 | active validation; each artifact states the gates it passes |
| Straight-pipe 3-D fringe | mapped operators, fixed flow, annular current, Benchmark B1 data | active validation; production parity requires the complete matched gate |
| Bent pipe and magnetic obstacle | mapped/conservation tests and internal observables | development applications, not externally validated benchmarks |

For 2-D cases, `validation_summary` reports convergence, current continuity,
gauge, interface, flow, and profile metrics. `hartmann_validation` and
`closed_channel_validation` compare against analytical data.

For 3-D cases, `ExtrudedInductionlessValidation` reports maximum update,
charge-balance and divergence residuals, boundary-current closure, stationwise
flow variation, pressure variation, and response correlations. Mesh and solver
independence remain separate required checks.

The benchmark specifications and reference arrays under `lmx/data/benchmarks`
are versioned package data. `benchmarks/provenance.json` records bibliographic
sources and executable test/workflow links.

## Test gates

The portable suite includes analytical, manufactured, regression, physics, and
validation markers. Combined line/branch coverage must exceed 95%. Structural
3-D changes additionally run the reduced pinned FreeMHD Docker case before
acceptance.
